use std::path::Path;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::attention::{Attention, Segment, causal_mask};
use crate::config::LlamaConfig;
use crate::kv::{KvBatch, KvStore, PerSeqKv};
use crate::layers::{Linear, Mlp, RmsNorm};
use crate::rope::Rope;

#[derive(Debug, Clone)]
struct Block {
    input_norm: RmsNorm,
    attn: Attention,
    post_attn_norm: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn load(vb: VarBuilder, cfg: &LlamaConfig) -> Result<Self> {
        let (h, eps) = (cfg.hidden_size, cfg.rms_norm_eps);
        Ok(Self {
            input_norm: RmsNorm::load(vb.pp("input_layernorm"), h, eps)?,
            attn: Attention::load(vb.pp("self_attn"), cfg)?,
            post_attn_norm: RmsNorm::load(vb.pp("post_attention_layernorm"), h, eps)?,
            mlp: Mlp::load(vb.pp("mlp"), h, cfg.intermediate_size)?,
        })
    }
}

/// Loads each tensor on the CPU, casts it there, then uploads only the final dtype.
///
/// candle's default mmap backend uploads the checkpoint dtype (bf16) and casts on the device;
/// on Metal the freed bf16 buffers stay in the allocator's cache, so loading peaked about 1.3
/// GB above the steady footprint.
struct CastOnCpu(candle_core::safetensors::MmapedSafetensors);

impl candle_nn::var_builder::SimpleBackend for CastOnCpu {
    fn get(
        &self,
        shape: candle_core::Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let t = self.get_unchecked(name, dtype, dev)?;
        if t.shape() != &shape {
            candle_core::bail!(
                "{name}: expected shape {shape:?}, checkpoint has {:?}",
                t.shape()
            );
        }
        Ok(t)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        self.0
            .load(name, &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.0.get(name).is_ok()
    }
}

/// Llama-architecture decoder (RMSNorm, RoPE, grouped-query attention, SwiGLU).
#[derive(Debug, Clone)]
pub struct Llama {
    cfg: LlamaConfig,
    embed: Tensor,
    blocks: Vec<Block>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: Rope,
    dtype: DType,
    device: Device,
}

impl Llama {
    /// Loads HF safetensors from `dir`, casting weights to `dtype`. RoPE tables cover
    /// `max_seq_len` positions.
    pub fn load(
        dir: &Path,
        dtype: DType,
        device: &Device,
        max_seq_len: usize,
    ) -> anyhow::Result<Self> {
        let cfg = LlamaConfig::from_file(dir.join("config.json"))?;
        let mut files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        anyhow::ensure!(
            !files.is_empty(),
            "no .safetensors files in {}",
            dir.display()
        );
        // SAFETY: the files are opened read-only and not modified while mapped.
        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&files)? };
        let vb = VarBuilder::from_backend(Box::new(CastOnCpu(tensors)), dtype, device.clone());
        Ok(Self::new(cfg, vb, max_seq_len)?)
    }

    pub fn new(cfg: LlamaConfig, vb: VarBuilder, max_seq_len: usize) -> Result<Self> {
        let (dtype, device) = (vb.dtype(), vb.device().clone());
        let m = vb.pp("model");
        let embed = m
            .pp("embed_tokens")
            .get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(m.pp(format!("layers.{i}")), &cfg))
            .collect::<Result<_>>()?;
        let norm = RmsNorm::load(m.pp("norm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::Dense(embed.clone())
        } else {
            Linear::load(vb.pp("lm_head"), cfg.hidden_size, cfg.vocab_size)?
        };
        let max_pos = max_seq_len.min(cfg.max_position_embeddings);
        let rope = Rope::new(&cfg, max_pos, dtype, &device)?;
        Ok(Self {
            cfg,
            embed,
            blocks,
            norm,
            lm_head,
            rope,
            dtype,
            device,
        })
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.cfg
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn max_seq_len(&self) -> usize {
        self.rope.max_positions()
    }

    /// Runs every sequence's new tokens through the model in one pass and returns the final
    /// hidden states `(1, total_tokens, hidden)` with each sequence's segment.
    fn hidden(&self, tokens: &[&[u32]], kv: &mut dyn KvBatch) -> Result<(Tensor, Vec<Segment>)> {
        if tokens.len() != kv.num_seqs() {
            candle_core::bail!(
                "{} token lists for {} kv sequences",
                tokens.len(),
                kv.num_seqs()
            );
        }
        let mut segments = Vec::with_capacity(tokens.len());
        let mut positions = Vec::new();
        let mut offset = 0;
        for (i, toks) in tokens.iter().enumerate() {
            let (start_pos, len) = (kv.start_pos(i), toks.len());
            if len == 0 {
                candle_core::bail!("empty sequence in batch");
            }
            if start_pos + len > self.max_seq_len() {
                candle_core::bail!(
                    "sequence of {} positions exceeds max_seq_len {}",
                    start_pos + len,
                    self.max_seq_len()
                );
            }
            let mask = match len {
                1 => None,
                _ => Some(causal_mask(len, start_pos, &self.device)?),
            };
            segments.push(Segment {
                offset,
                len,
                start_pos,
                mask,
            });
            positions.extend(start_pos as u32..(start_pos + len) as u32);
            offset += len;
        }
        let rope = self.rope.batch(&Tensor::new(positions, &self.device)?)?;
        let ids: Vec<u32> = tokens.iter().flat_map(|t| t.iter().copied()).collect();
        let ids = Tensor::new(ids, &self.device)?;
        let mut x = self.embed.index_select(&ids, 0)?.unsqueeze(0)?;
        for (i, block) in self.blocks.iter().enumerate() {
            let h = block.input_norm.forward(&x)?;
            let h = block.attn.forward(&h, i, &segments, &rope, kv)?;
            x = (x + h)?;
            let h = block.mlp.forward(&block.post_attn_norm.forward(&x)?)?;
            x = (x + h)?;
        }
        Ok((self.norm.forward(&x)?, segments))
    }

    /// Batched step: logits at each sequence's last new position, shape `(n_seqs, vocab)`, in
    /// f32. Sequences may mix prefill (many tokens) and decode (one token).
    pub fn forward_batch(&self, tokens: &[&[u32]], kv: &mut dyn KvBatch) -> Result<Tensor> {
        let (h, segments) = self.hidden(tokens, kv)?;
        let last: Vec<u32> = segments
            .iter()
            .map(|s| (s.offset + s.len - 1) as u32)
            .collect();
        let last = Tensor::new(last, &self.device)?;
        let h = h.squeeze(0)?.index_select(&last, 0)?;
        self.lm_head.forward(&h)?.to_dtype(DType::F32)
    }

    /// Logits for the last position only, shape `(vocab,)`, in f32.
    pub fn forward(&self, tokens: &[u32], kv: &mut dyn KvStore) -> Result<Tensor> {
        self.forward_batch(&[tokens], &mut PerSeqKv(vec![kv]))?
            .squeeze(0)
    }

    /// Logits for every position, shape `(seq, vocab)`, in f32.
    pub fn forward_all_logits(&self, tokens: &[u32], kv: &mut dyn KvStore) -> Result<Tensor> {
        let (h, _) = self.hidden(&[tokens], &mut PerSeqKv(vec![kv]))?;
        self.lm_head.forward(&h)?.to_dtype(DType::F32)?.squeeze(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::ContiguousKvCache;
    use candle_nn::VarMap;

    fn per_seq(kvs: &mut [ContiguousKvCache]) -> PerSeqKv<'_> {
        PerSeqKv(kvs.iter_mut().map(|k| k as &mut dyn KvStore).collect())
    }

    fn tiny_model() -> Result<Llama> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        Llama::new(LlamaConfig::tiny(), vb, 64)
    }

    /// Prefill followed by token-by-token decode through the cache must reproduce the logits
    /// of one full-sequence forward pass.
    #[test]
    fn incremental_decode_matches_full_forward() -> Result<()> {
        let model = tiny_model()?;
        let cfg = model.config().clone();
        let tokens: Vec<u32> = vec![1, 5, 9, 33, 2, 70, 41, 8, 96, 3];
        let new_cache = || ContiguousKvCache::new(&cfg, 64, DType::F32, &Device::Cpu);

        let full = model.forward_all_logits(&tokens, &mut new_cache()?)?;

        let mut kv = new_cache()?;
        let prefill = 4;
        let mut rows = vec![model.forward_all_logits(&tokens[..prefill], &mut kv)?];
        for &t in &tokens[prefill..] {
            rows.push(model.forward(&[t], &mut kv)?.unsqueeze(0)?);
        }
        let incremental = Tensor::cat(&rows, 0)?;
        let full_tail = full.narrow(0, prefill - 1, tokens.len() - prefill + 1)?;
        let inc_tail = incremental.narrow(0, prefill - 1, tokens.len() - prefill + 1)?;
        let diff = (full_tail - inc_tail)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-4, "max logit diff {diff}");
        assert_eq!(kv.seq_len(), tokens.len());
        Ok(())
    }

    /// A sequence's logits must not depend on what else is in the batch.
    #[test]
    fn batched_forward_matches_individual() -> Result<()> {
        let model = tiny_model()?;
        let cfg = model.config().clone();
        let cache = || ContiguousKvCache::new(&cfg, 64, DType::F32, &Device::Cpu);
        let prompts: [&[u32]; 3] = [&[1, 5, 9], &[7], &[2, 40, 41, 42, 43, 44]];

        let mut alone = Vec::new();
        let mut alone_kv = Vec::new();
        for p in prompts {
            let mut kv = cache()?;
            alone.push(model.forward(p, &mut kv)?);
            alone_kv.push(kv);
        }

        let mut kvs = [cache()?, cache()?, cache()?];
        let batched = model.forward_batch(&prompts, &mut per_seq(&mut kvs))?;
        let close = |a: &Tensor, b: &Tensor| -> Result<f32> {
            (a - b)?.abs()?.max_all()?.to_scalar::<f32>()
        };
        for (i, a) in alone.iter().enumerate() {
            assert!(close(a, &batched.get(i)?)? < 1e-4, "prefill seq {i}");
        }

        // Then one mixed step: two decodes alongside a fresh prefill for sequence 2.
        let next: [&[u32]; 3] = [&[11], &[12], &[3, 4]];
        kvs[2].reset();
        let batched = model.forward_batch(&next, &mut per_seq(&mut kvs))?;
        let mut fresh = cache()?;
        let want = [
            model.forward(next[0], &mut alone_kv[0])?,
            model.forward(next[1], &mut alone_kv[1])?,
            model.forward(next[2], &mut fresh)?,
        ];
        for (i, w) in want.iter().enumerate() {
            assert!(close(w, &batched.get(i)?)? < 1e-4, "mixed step seq {i}");
        }
        Ok(())
    }

    #[test]
    fn rejects_sequences_past_max_len() -> Result<()> {
        let model = tiny_model()?;
        let mut kv = ContiguousKvCache::new(model.config(), 128, DType::F32, &Device::Cpu)?;
        assert!(model.forward(&vec![1; 65], &mut kv).is_err());
        Ok(())
    }
}
