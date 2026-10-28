//! KV storage in a fixed pool of blocks, addressed through per-sequence block tables.
//!
//! Each layer's K and V are one `(num_blocks * block_size, n_kv_heads, head_dim)` tensor;
//! token slot `s` is row `s`. A step writes all new tokens of the batch with one scatter per
//! layer. Attention v1 gathers each sequence's rows into a contiguous tensor and runs standard
//! attention; v2 (Metal) runs every decode token's attention in one kernel that reads K/V
//! through the block tables, and keeps v1 for prefill segments.

use candle_core::{DType, Device, Result, Tensor};

use crate::attention::{Segment, attend_segments, segment_attention};
use crate::config::LlamaConfig;
use crate::kv::KvBatch;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionImpl {
    /// v1: gather each sequence's blocks, then standard attention.
    Gather,
    /// v2: paged-attention kernel for decode tokens (Metal), v1 for prefill.
    Kernel,
}

pub struct PagedKvCache {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
    block_size: usize,
    num_blocks: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    attention: AttentionImpl,
    #[cfg(feature = "metal")]
    pipelines: std::sync::Arc<crate::paged_attention::metal_impl::Pipelines>,
}

impl PagedKvCache {
    /// Allocates the whole pool up front.
    pub fn new(
        cfg: &LlamaConfig,
        num_blocks: usize,
        block_size: usize,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let shape = (
            num_blocks * block_size,
            cfg.num_key_value_heads,
            cfg.head_dim(),
        );
        // Zeroed on the CPU and uploaded: candle's Metal allocator rounds a fresh buffer up to
        // the next power of two and zero-fills all of it, so a 2.25 GiB pool built with
        // `Tensor::zeros` on the device took 4 GiB. Uploads are allocated at their exact size.
        let alloc = || Tensor::zeros(shape, dtype, &Device::Cpu)?.to_device(dev);
        let layers = cfg.num_hidden_layers;
        Ok(Self {
            k: (0..layers).map(|_| alloc()).collect::<Result<_>>()?,
            v: (0..layers).map(|_| alloc()).collect::<Result<_>>()?,
            block_size,
            num_blocks,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim(),
            attention: AttentionImpl::Gather,
            #[cfg(feature = "metal")]
            pipelines: Default::default(),
        })
    }

    /// Selects the attention implementation. `Kernel` falls back to `Gather` where the kernel
    /// can't run (not on Metal, or an unsupported head layout); returns what is in effect.
    pub fn set_attention(&mut self, want: AttentionImpl) -> AttentionImpl {
        let kernel_ok = cfg!(feature = "metal")
            && self.device().is_metal()
            && crate::paged_attention::supported(self.n_heads, self.n_kv_heads, self.head_dim);
        self.attention = match want {
            AttentionImpl::Kernel if kernel_ok => AttentionImpl::Kernel,
            _ => AttentionImpl::Gather,
        };
        self.attention
    }

    pub fn attention(&self) -> AttentionImpl {
        self.attention
    }

    fn kernel_op(
        &self,
        layer: usize,
        max_context: usize,
    ) -> crate::paged_attention::PagedAttention {
        crate::paged_attention::PagedAttention {
            k_cache: self.k[layer].clone(),
            v_cache: self.v[layer].clone(),
            block_size: self.block_size,
            max_context,
            #[cfg(feature = "metal")]
            pipelines: self.pipelines.clone(),
        }
    }

    /// Blocks that fit in `bytes` for this model and dtype.
    pub fn blocks_for_bytes(
        cfg: &LlamaConfig,
        bytes: usize,
        block_size: usize,
        dtype: DType,
    ) -> usize {
        bytes / (cfg.kv_bytes_per_token(dtype.size_in_bytes()) * block_size)
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn device(&self) -> &Device {
        self.k[0].device()
    }

    /// Copies whole blocks `(src, dst)` in every layer (copy-on-write for shared blocks).
    pub fn copy_blocks(&self, copies: &[(u32, u32)]) -> Result<()> {
        let bs = self.block_size;
        for (src, dst) in copies {
            for t in self.k.iter().chain(&self.v) {
                // `copy` rather than `contiguous`: a dim-0 narrow is already contiguous and would
                // alias the destination tensor.
                let block = t.narrow(0, *src as usize * bs, bs)?.copy()?;
                t.slice_set(&block, 0, *dst as usize * bs)?;
            }
        }
        Ok(())
    }

    /// Keys or values of `layer` for the given slots, as `(1, n_kv_heads, slots, head_dim)`.
    fn gather(&self, cache: &Tensor, slots: &Tensor) -> Result<Tensor> {
        cache
            .index_select(slots, 0)?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()
    }
}

/// One sequence's part of a paged step.
pub struct PagedSeq<'a> {
    pub block_table: &'a [u32],
    /// Positions already in the cache.
    pub start_pos: usize,
    /// New tokens this step.
    pub len: usize,
}

/// A step over the paged cache: slot indices are computed once and reused by every layer.
pub struct PagedBatch<'a> {
    cache: &'a PagedKvCache,
    start_pos: Vec<usize>,
    /// Destination row of every element of the new `(total_tokens, n_kv, head_dim)` K/V.
    write_index: Tensor,
    /// Per sequence: slots of every cached position, oldest first (`None` for sequences the
    /// decode kernel handles).
    read_slots: Vec<Option<Tensor>>,
    /// Inputs for the v2 kernel when the step has decode tokens and the kernel is enabled.
    decode: Option<DecodePlan>,
}

/// Decode tokens (one new token per sequence) handled by the paged-attention kernel.
struct DecodePlan {
    /// Batch index of each decoding sequence.
    seqs: Vec<usize>,
    /// `(D, max_blocks)` block tables, padded.
    tables: Tensor,
    /// `(D,)` context length including the new token.
    ctx_lens: Tensor,
    /// The longest of `ctx_lens`.
    max_context: usize,
    /// Token rows of the decode queries; `None` when every token is a decode.
    token_rows: Option<Tensor>,
    /// Maps segment-order token rows to `[decodes..., prefills...]` rows; `None` when all decode.
    reorder: Option<Tensor>,
}

impl DecodePlan {
    fn new(cache: &PagedKvCache, seqs: &[PagedSeq]) -> Result<Option<Self>> {
        let dec: Vec<usize> = (0..seqs.len()).filter(|&i| seqs[i].len == 1).collect();
        if cache.attention != AttentionImpl::Kernel || dec.is_empty() {
            return Ok(None);
        }
        let dev = cache.device();
        let max_blocks = dec
            .iter()
            .map(|&i| seqs[i].block_table.len())
            .max()
            .unwrap_or(1)
            .max(1);
        let mut tables = vec![0u32; dec.len() * max_blocks];
        for (row, &i) in dec.iter().enumerate() {
            tables[row * max_blocks..][..seqs[i].block_table.len()]
                .copy_from_slice(seqs[i].block_table);
        }
        let ctx: Vec<u32> = dec
            .iter()
            .map(|&i| (seqs[i].start_pos + 1) as u32)
            .collect();
        let total: usize = seqs.iter().map(|s| s.len).sum();
        let (token_rows, reorder) = if dec.len() == seqs.len() {
            (None, None)
        } else {
            let offsets: Vec<usize> = seqs
                .iter()
                .scan(0, |o, s| {
                    let at = *o;
                    *o += s.len;
                    Some(at)
                })
                .collect();
            let rows: Vec<u32> = dec.iter().map(|&i| offsets[i] as u32).collect();
            // Output row order is [decodes..., then each prefill segment in batch order].
            let mut reorder = vec![0u32; total];
            for (r, &i) in dec.iter().enumerate() {
                reorder[offsets[i]] = r as u32;
            }
            let mut next = dec.len() as u32;
            for (i, s) in seqs.iter().enumerate().filter(|(_, s)| s.len != 1) {
                for t in 0..s.len {
                    reorder[offsets[i] + t] = next;
                    next += 1;
                }
            }
            (
                Some(Tensor::new(rows, dev)?),
                Some(Tensor::new(reorder, dev)?),
            )
        };
        Ok(Some(Self {
            tables: Tensor::from_vec(tables, (dec.len(), max_blocks), dev)?,
            max_context: ctx.iter().copied().max().unwrap_or(1) as usize,
            ctx_lens: Tensor::new(ctx, dev)?,
            seqs: dec,
            token_rows,
            reorder,
        }))
    }
}

impl<'a> PagedBatch<'a> {
    pub fn new(cache: &'a PagedKvCache, seqs: &[PagedSeq]) -> Result<Self> {
        let bs = cache.block_size;
        let slot = |table: &[u32], pos: usize| -> Result<u32> {
            let block = table.get(pos / bs).ok_or_else(|| {
                candle_core::Error::Msg(format!("position {pos} has no block reserved"))
            })?;
            Ok(block * bs as u32 + (pos % bs) as u32)
        };
        let row = cache.n_kv_heads * cache.head_dim;
        let mut write = Vec::new();
        let mut read_slots = Vec::with_capacity(seqs.len());
        for s in seqs {
            for pos in s.start_pos..s.start_pos + s.len {
                let dst = slot(s.block_table, pos)?;
                write.extend(std::iter::repeat_n(dst, row));
            }
            // The decode kernel reads block tables directly; only gathered sequences need
            // their slot list uploaded.
            if s.len == 1 && cache.attention == AttentionImpl::Kernel {
                read_slots.push(None);
                continue;
            }
            let reads = (0..s.start_pos + s.len)
                .map(|p| slot(s.block_table, p))
                .collect::<Result<Vec<_>>>()?;
            read_slots.push(Some(Tensor::new(reads, cache.device())?));
        }
        let total: usize = seqs.iter().map(|s| s.len).sum();
        let write_index = Tensor::from_vec(
            write,
            (total, cache.n_kv_heads, cache.head_dim),
            cache.device(),
        )?;
        Ok(Self {
            cache,
            start_pos: seqs.iter().map(|s| s.start_pos).collect(),
            write_index,
            read_slots,
            decode: DecodePlan::new(cache, seqs)?,
        })
    }
}

impl PagedBatch<'_> {
    fn slots(&self, i: usize) -> Result<&Tensor> {
        self.read_slots[i]
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg(format!("sequence {i} has no gather slots")))
    }
}

impl KvBatch for PagedBatch<'_> {
    fn num_seqs(&self) -> usize {
        self.start_pos.len()
    }

    fn start_pos(&self, i: usize) -> usize {
        self.start_pos[i]
    }

    fn attend(
        &mut self,
        layer: usize,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        segments: &[Segment],
    ) -> Result<Tensor> {
        let (kc, vc) = (&self.cache.k[layer], &self.cache.v[layer]);
        // Attention runs in the cache dtype (f16 KV under f32 activations for quantized
        // models); casts are no-ops when the dtypes already match.
        let act_dtype = q.dtype();
        let (q, k, v) = (
            &q.to_dtype(kc.dtype())?,
            &k.to_dtype(kc.dtype())?,
            &v.to_dtype(kc.dtype())?,
        );
        // (1, n_kv, T, d) -> (T, n_kv, d), one scatter for the whole batch.
        let rows = |t: &Tensor| t.squeeze(0)?.transpose(0, 1)?.contiguous();
        kc.scatter_set(&self.write_index, &rows(k)?, 0)?;
        vc.scatter_set(&self.write_index, &rows(v)?, 0)?;
        let Some(plan) = &self.decode else {
            return attend_segments(q, segments, |i, _| {
                Ok((
                    self.cache.gather(kc, self.slots(i)?)?,
                    self.cache.gather(vc, self.slots(i)?)?,
                ))
            })?
            .to_dtype(act_dtype);
        };
        // (1, n_heads, T, d) -> (T, n_heads, d) query rows.
        let q_rows = q.squeeze(0)?.transpose(0, 1)?;
        let q_dec = match &plan.token_rows {
            None => q_rows.contiguous()?,
            Some(rows) => q_rows.contiguous()?.index_select(rows, 0)?,
        };
        let out_dec = q_dec.apply_op3_no_bwd(
            &plan.tables,
            &plan.ctx_lens,
            &self.cache.kernel_op(layer, plan.max_context),
        )?;
        let out_rows = match &plan.reorder {
            None => out_dec,
            Some(reorder) => {
                let mut parts = vec![out_dec];
                for (i, seg) in segments
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !plan.seqs.contains(i))
                {
                    let k_all = self.cache.gather(kc, self.slots(i)?)?;
                    let v_all = self.cache.gather(vc, self.slots(i)?)?;
                    let qs = q.narrow(2, seg.offset, seg.len)?;
                    let o = segment_attention(&qs, &k_all, &v_all, seg)?;
                    parts.push(o.squeeze(0)?.transpose(0, 1)?);
                }
                Tensor::cat(&parts, 0)?.index_select(reorder, 0)?
            }
        };
        out_rows.transpose(0, 1)?.unsqueeze(0)?.to_dtype(act_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::ContiguousKvCache;
    use crate::llama::Llama;
    use candle_nn::{VarBuilder, VarMap};

    fn tiny() -> Result<Llama> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        Llama::new(LlamaConfig::tiny(), vb, 64)
    }

    fn max_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a - b)?.abs()?.max_all()?.to_scalar::<f32>()
    }

    /// Paged steps (scattered block ids, a prompt prefilled in two chunks, mixed with
    /// decodes) give the same logits as each sequence alone with a contiguous cache.
    #[test]
    fn paged_matches_contiguous() -> Result<()> {
        let model = tiny()?;
        let cfg = model.config().clone();
        let cache = PagedKvCache::new(&cfg, 16, 4, DType::F32, &Device::Cpu)?;
        // Deliberately non-contiguous, interleaved physical blocks.
        let tables: [&[u32]; 2] = [&[9, 2, 14, 5], &[0, 11, 7, 3]];
        let a: Vec<u32> = vec![1, 5, 9, 33, 2, 70, 41, 8, 96, 3, 4];
        let b: Vec<u32> = vec![7, 7, 60, 12, 13];

        // Step 1: all of a's prompt (6 tokens) and the first 3 of b's.
        // Step 2: b's remaining 2 prompt tokens plus 5 single-token decode steps for a.
        let steps: &[[(usize, usize); 2]] = &[
            [(0, 6), (0, 3)],
            [(6, 7), (3, 5)],
            [(7, 8), (5, 5)],
            [(8, 11), (5, 5)],
        ];
        let mut ref_a = ContiguousKvCache::new(&cfg, 64, DType::F32, &Device::Cpu)?;
        let mut ref_b = ContiguousKvCache::new(&cfg, 64, DType::F32, &Device::Cpu)?;
        for step in steps {
            let mut toks: Vec<&[u32]> = Vec::new();
            let mut seqs = Vec::new();
            let mut want = Vec::new();
            for (i, &(s, e)) in step.iter().enumerate() {
                if s == e {
                    continue;
                }
                let (src, refc) = if i == 0 {
                    (&a, &mut ref_a)
                } else {
                    (&b, &mut ref_b)
                };
                toks.push(&src[s..e]);
                seqs.push(PagedSeq {
                    block_table: tables[i],
                    start_pos: s,
                    len: e - s,
                });
                want.push(model.forward(&src[s..e], refc)?);
            }
            let got = model.forward_batch(&toks, &mut PagedBatch::new(&cache, &seqs)?)?;
            for (i, w) in want.iter().enumerate() {
                let d = max_diff(w, &got.get(i)?)?;
                assert!(d < 1e-4, "seq {i}: logit diff {d}");
            }
        }
        Ok(())
    }

    #[test]
    fn copy_blocks_duplicates_contents() -> Result<()> {
        let cfg = LlamaConfig::tiny();
        let cache = PagedKvCache::new(&cfg, 4, 2, DType::F32, &Device::Cpu)?;
        let rows = Tensor::randn(
            0f32,
            1.0,
            (2, cfg.num_key_value_heads, cfg.head_dim()),
            &Device::Cpu,
        )?;
        cache.k[0].slice_set(&rows, 0, 2)?;
        cache.copy_blocks(&[(1, 3)])?;
        let copied = cache.k[0].narrow(0, 6, 2)?;
        assert_eq!(max_diff(&copied, &rows)?, 0.0);
        Ok(())
    }

    #[test]
    fn missing_block_is_an_error() -> Result<()> {
        let cache = PagedKvCache::new(&LlamaConfig::tiny(), 4, 2, DType::F32, &Device::Cpu)?;
        assert!(
            PagedBatch::new(
                &cache,
                &[PagedSeq {
                    block_table: &[0],
                    start_pos: 1,
                    len: 2
                }]
            )
            .is_err()
        );
        Ok(())
    }
}

#[cfg(all(test, feature = "metal"))]
mod metal_tests {
    use super::*;
    use crate::llama::Llama;
    use candle_nn::{VarBuilder, VarMap};

    /// Full model on Metal with head_dim 64 (kernel-compatible): v2 logits match v1 for steps
    /// mixing prefill and decode, and for all-decode steps.
    #[test]
    fn kernel_matches_gather_end_to_end() -> Result<()> {
        let dev = Device::new_metal(0)?;
        let cfg = LlamaConfig {
            hidden_size: 256,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            head_dim: Some(64),
            ..LlamaConfig::tiny()
        };
        let varmap = VarMap::new();
        let model = Llama::new(
            cfg.clone(),
            VarBuilder::from_varmap(&varmap, DType::F32, &dev),
            128,
        )?;
        let mut v1 = PagedKvCache::new(&cfg, 32, 4, DType::F32, &dev)?;
        let mut v2 = PagedKvCache::new(&cfg, 32, 4, DType::F32, &dev)?;
        assert_eq!(
            v1.set_attention(AttentionImpl::Gather),
            AttentionImpl::Gather
        );
        assert_eq!(
            v2.set_attention(AttentionImpl::Kernel),
            AttentionImpl::Kernel
        );
        let tables: [&[u32]; 3] = [&[31, 2, 17, 5, 9], &[0, 11, 7, 3, 20], &[1, 30, 8, 22, 4]];
        let prompts: [Vec<u32>; 3] = [(1..14).collect(), (20..23).collect(), (40..58).collect()];
        // (start, end) per sequence per step; equal start/end = not scheduled.
        let steps: &[[(usize, usize); 3]] = &[
            [(0, 13), (0, 3), (0, 0)],
            [(13, 14), (3, 4), (0, 18)],
            [(14, 15), (4, 5), (18, 19)],
            [(15, 16), (5, 6), (19, 20)],
        ];
        let mut toks: Vec<Vec<u32>> = prompts.to_vec();
        for t in &mut toks {
            t.extend(70..80);
        }
        for step in steps {
            let mut tokens: Vec<&[u32]> = Vec::new();
            let mut seqs = Vec::new();
            for (i, &(s, e)) in step.iter().enumerate() {
                if s < e {
                    tokens.push(&toks[i][s..e]);
                    seqs.push(PagedSeq {
                        block_table: tables[i],
                        start_pos: s,
                        len: e - s,
                    });
                }
            }
            let a = model.forward_batch(&tokens, &mut PagedBatch::new(&v1, &seqs)?)?;
            let b = model.forward_batch(&tokens, &mut PagedBatch::new(&v2, &seqs)?)?;
            let d = (a - b)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(d < 1e-4, "step {step:?}: logit diff {d}");
        }
        Ok(())
    }
}
