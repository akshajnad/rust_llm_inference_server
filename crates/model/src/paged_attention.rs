//! Attention v2: decode attention that reads K/V through block tables in one kernel per layer
//! for the whole batch, instead of gathering each sequence's cache and running attention per
//! sequence (v1). Long contexts with few sequences are split into partitions that run in
//! parallel and are merged by a second kernel (see `split_plan`). Metal only; `cpu_fwd` is a
//! straightforward reference used by tests.

use candle_core::{CpuStorage, CustomOp3, DType, Layout, Result, Shape, Tensor};

/// Most query heads per KV head (the widest kernel variant) and the largest head_dim.
pub const MAX_REP: usize = 8;
pub const MAX_HEAD_DIM: usize = 128;

/// Whether the kernel supports this head layout.
pub fn supported(n_heads: usize, n_kv: usize, head_dim: usize) -> bool {
    n_heads.is_multiple_of(n_kv)
        && n_heads / n_kv <= MAX_REP
        && head_dim.is_multiple_of(32)
        && head_dim <= MAX_HEAD_DIM
}

/// Threadgroups a decode step should have before splitting contexts stops paying off: about
/// a dozen per GPU core on a 20-core M4 Pro.
const TARGET_THREADGROUPS: usize = 256;
/// Fewest positions per partition: below this the merge costs more than the parallelism wins.
const MIN_PARTITION: usize = 256;

/// How many partitions to split each context into and how many positions each covers, for
/// `pairs` (kv head, sequence) pairs and a longest context of `max_context`. One partition
/// means the single-pass kernel.
pub fn split_plan(pairs: usize, max_context: usize) -> (usize, usize) {
    let splits = TARGET_THREADGROUPS
        .div_ceil(pairs.max(1))
        .min(max_context.div_ceil(MIN_PARTITION))
        .max(1);
    (splits, max_context.div_ceil(splits).max(1))
}

/// `q (D, n_heads, d)`, `block_tables (D, max_blocks) u32`, `ctx_lens (D) u32` ->
/// `(D, n_heads, d)`. The caches are `(slots, n_kv, d)` and already hold this step's K/V.
pub struct PagedAttention {
    pub k_cache: Tensor,
    pub v_cache: Tensor,
    pub block_size: usize,
    /// The largest entry of `ctx_lens`, known on the host so the split needs no readback.
    pub max_context: usize,
    #[cfg(feature = "metal")]
    pub pipelines: std::sync::Arc<metal_impl::Pipelines>,
}

impl PagedAttention {
    fn dims(&self, q: &Layout) -> Result<(usize, usize, usize, usize)> {
        let (d, n_heads, head_dim) = q.shape().dims3()?;
        let (_, n_kv, _) = self.k_cache.dims3()?;
        if !supported(n_heads, n_kv, head_dim) {
            candle_core::bail!("paged attention: unsupported heads {n_heads}/{n_kv} x {head_dim}");
        }
        Ok((d, n_heads, n_kv, head_dim))
    }
}

impl CustomOp3 for PagedAttention {
    fn name(&self) -> &'static str {
        "paged-attention-decode"
    }

    fn cpu_fwd(
        &self,
        q: &CpuStorage,
        lq: &Layout,
        tables: &CpuStorage,
        lt: &Layout,
        lens: &CpuStorage,
        ll: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (nd, n_heads, n_kv, hd) = self.dims(lq)?;
        let max_blocks = lt.shape().dims2()?.1;
        let to_f32 = |t: &Tensor| t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
        let (kc, vc) = (to_f32(&self.k_cache)?, to_f32(&self.v_cache)?);
        let qv: Vec<f32> = match q {
            CpuStorage::F32(v) => v[lq.start_offset()..].to_vec(),
            CpuStorage::F16(v) => v[lq.start_offset()..].iter().map(|x| x.to_f32()).collect(),
            _ => candle_core::bail!("paged attention: unsupported dtype"),
        };
        let (CpuStorage::U32(tables), CpuStorage::U32(lens)) = (tables, lens) else {
            candle_core::bail!("paged attention: block tables and lengths must be u32");
        };
        let (tables, lens) = (&tables[lt.start_offset()..], &lens[ll.start_offset()..]);
        let n_rep = n_heads / n_kv;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0f32; nd * n_heads * hd];
        for s in 0..nd {
            let ctx = lens[s] as usize;
            for h in 0..n_heads {
                let kvh = h / n_rep;
                let qh = &qv[(s * n_heads + h) * hd..][..hd];
                let row = |cache: &[f32], t: usize| {
                    let slot = tables[s * max_blocks + t / self.block_size] as usize
                        * self.block_size
                        + t % self.block_size;
                    cache[(slot * n_kv + kvh) * hd..][..hd].to_vec()
                };
                let scores: Vec<f32> = (0..ctx)
                    .map(|t| row(&kc, t).iter().zip(qh).map(|(a, b)| a * b).sum::<f32>() * scale)
                    .collect();
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let w: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f32 = w.iter().sum();
                let o = &mut out[(s * n_heads + h) * hd..][..hd];
                for (t, wt) in w.iter().enumerate() {
                    for (od, vd) in o.iter_mut().zip(row(&vc, t)) {
                        *od += wt / total * vd;
                    }
                }
            }
        }
        let shape = Shape::from((nd, n_heads, hd));
        Ok(match q {
            CpuStorage::F16(_) => (
                CpuStorage::F16(out.into_iter().map(half::f16::from_f32).collect()),
                shape,
            ),
            _ => (CpuStorage::F32(out), shape),
        })
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        q: &candle_core::MetalStorage,
        lq: &Layout,
        tables: &candle_core::MetalStorage,
        lt: &Layout,
        lens: &candle_core::MetalStorage,
        ll: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal_impl::forward(self, q, lq, tables, lt, lens, ll)
    }
}

#[cfg(feature = "metal")]
pub mod metal_impl {
    use std::sync::OnceLock;

    use candle_core::backend::BackendStorage;
    use candle_core::{DType, Layout, MetalStorage, Result, Shape, Storage};
    use candle_metal_kernels::metal::ComputePipeline;
    use candle_metal_kernels::utils::EncoderProvider;
    use objc2_metal::MTLSize;

    use super::PagedAttention;

    const SOURCE: &str = include_str!("paged_attention.metal");
    const THREADS: usize = 8 * 32;

    /// Query heads per kv head of the kernel variants (`_r4`, `_r8`); a layout uses the
    /// narrowest that fits.
    const REPS: [usize; 2] = [4, 8];

    /// Kernels for one dtype; `decode` and `split` per entry of `REPS`.
    struct Kernels {
        decode: [ComputePipeline; 2],
        split: [ComputePipeline; 2],
        reduce: ComputePipeline,
    }

    /// Compiled lazily on first use; MSL is compiled at runtime, no offline toolchain needed.
    #[derive(Default)]
    pub struct Pipelines(OnceLock<(Kernels, Kernels)>);

    impl Pipelines {
        fn get(&self, device: &candle_core::MetalDevice, dtype: DType) -> Result<&Kernels> {
            if self.0.get().is_none() {
                let dev = device.metal_device();
                let lib = dev
                    .new_library_with_source(SOURCE, None)
                    .map_err(candle_core::Error::wrap)?;
                let make = |name: &str| -> Result<ComputePipeline> {
                    let f = lib
                        .get_function(name, None)
                        .map_err(candle_core::Error::wrap)?;
                    dev.new_compute_pipeline_state_with_function(&f)
                        .map_err(candle_core::Error::wrap)
                };
                let kernels = |ty: &str| -> Result<Kernels> {
                    let [r4, r8] = REPS;
                    Ok(Kernels {
                        decode: [
                            make(&format!("paged_attention_decode_{ty}_r{r4}"))?,
                            make(&format!("paged_attention_decode_{ty}_r{r8}"))?,
                        ],
                        split: [
                            make(&format!("paged_attention_decode_split_{ty}_r{r4}"))?,
                            make(&format!("paged_attention_decode_split_{ty}_r{r8}"))?,
                        ],
                        reduce: make(&format!("paged_attention_reduce_{ty}"))?,
                    })
                };
                let _ = self.0.set((kernels("f32")?, kernels("f16")?));
            }
            let (f32p, f16p) = self.0.get().expect("initialized above");
            match dtype {
                DType::F32 => Ok(f32p),
                DType::F16 => Ok(f16p),
                d => candle_core::bail!("paged attention kernel: unsupported dtype {d:?}"),
            }
        }
    }

    pub(super) fn forward(
        op: &PagedAttention,
        q: &MetalStorage,
        lq: &Layout,
        tables: &MetalStorage,
        lt: &Layout,
        lens: &MetalStorage,
        ll: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (nd, n_heads, n_kv, hd) = op.dims(lq)?;
        let max_blocks = lt.shape().dims2()?.1;
        if !(lq.is_contiguous() && lt.is_contiguous() && ll.is_contiguous()) {
            candle_core::bail!("paged attention: inputs must be contiguous");
        }
        let device = q.device();
        let dtype = q.dtype();
        let kernels = op.pipelines.get(device, dtype)?;
        let width = REPS
            .iter()
            .position(|&r| n_heads / n_kv <= r)
            .expect("supported() bounds the heads per kv head");
        let (splits, part) = super::split_plan(n_kv * nd, op.max_context);
        let (ks, kl) = op.k_cache.storage_and_layout();
        let (vs, vl) = op.v_cache.storage_and_layout();
        let (Storage::Metal(ks), Storage::Metal(vs)) = (&*ks, &*vs) else {
            candle_core::bail!("paged attention: caches must be on Metal");
        };
        let elem = dtype.size_in_bytes();
        let out = device
            .new_buffer_builder()
            .with_size_for(nd * n_heads * hd, dtype)
            .with_label("paged_attention")
            .build()?;

        let u = |x: usize| x as u32;
        let size = |width, height, depth| MTLSize {
            width,
            height,
            depth,
        };
        // Partition states, f32: (m, l) pairs and unnormalized outputs.
        let partials = (splits > 1)
            .then(|| -> Result<_> {
                let rows = nd * n_heads * splits;
                Ok((
                    device
                        .new_buffer_builder()
                        .with_size_for(rows * 2, DType::F32)
                        .with_label("paged_attention_ml")
                        .build()?,
                    device
                        .new_buffer_builder()
                        .with_size_for(rows * hd, DType::F32)
                        .with_label("paged_attention_acc")
                        .build()?,
                ))
            })
            .transpose()?;
        {
            let guard = device.command_encoder()?;
            let provider = &guard;
            let holder = provider.encoder();
            let enc = holder.as_ref();
            enc.set_compute_pipeline_state(match partials {
                None => &kernels.decode[width],
                Some(_) => &kernels.split[width],
            });
            enc.set_input_buffer(0, Some(q.buffer()), lq.start_offset() * elem);
            enc.set_input_buffer(1, Some(ks.buffer()), kl.start_offset() * elem);
            enc.set_input_buffer(2, Some(vs.buffer()), vl.start_offset() * elem);
            enc.set_input_buffer(3, Some(tables.buffer()), lt.start_offset() * 4);
            enc.set_input_buffer(4, Some(lens.buffer()), ll.start_offset() * 4);
            enc.set_bytes(6, &u(n_heads));
            enc.set_bytes(7, &u(n_kv));
            enc.set_bytes(8, &u(hd));
            enc.set_bytes(9, &u(op.block_size));
            enc.set_bytes(10, &u(max_blocks));
            enc.set_bytes(11, &(1.0f32 / (hd as f32).sqrt()));
            match &partials {
                None => enc.set_output_buffer(5, Some(&out), 0),
                Some((ml, acc)) => {
                    enc.set_output_buffer(5, Some(ml), 0);
                    enc.set_output_buffer(12, Some(acc), 0);
                    enc.set_bytes(13, &u(part));
                    enc.set_bytes(14, &u(splits));
                }
            }
            enc.dispatch_thread_groups(size(n_kv, nd, splits), size(THREADS, 1, 1));
        }
        if let Some((ml, acc)) = &partials {
            let guard = device.command_encoder()?;
            let provider = &guard;
            let holder = provider.encoder();
            let enc = holder.as_ref();
            enc.set_compute_pipeline_state(&kernels.reduce);
            enc.set_input_buffer(0, Some(ml), 0);
            enc.set_input_buffer(1, Some(acc), 0);
            enc.set_output_buffer(2, Some(&out), 0);
            enc.set_bytes(3, &u(n_heads));
            enc.set_bytes(4, &u(hd));
            enc.set_bytes(5, &u(splits));
            enc.dispatch_thread_groups(size(n_heads, nd, 1), size(hd, 1, 1));
        }
        Ok((
            MetalStorage::new(out, device.clone(), nd * n_heads * hd, dtype),
            Shape::from((nd, n_heads, hd)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::gqa_attention;
    use candle_core::{Device, IndexOp};

    struct Case {
        k: Tensor,
        v: Tensor,
        q: Tensor,
        tables: Vec<Vec<u32>>,
        lens: Vec<usize>,
        block_size: usize,
    }

    /// Random caches and scattered block tables for `lens.len()` sequences.
    fn case(
        dev: &Device,
        dtype: DType,
        n_heads: usize,
        n_kv: usize,
        hd: usize,
        lens: &[usize],
    ) -> Result<Case> {
        let bs = 16;
        let per = lens.iter().map(|l| l.div_ceil(bs)).collect::<Vec<_>>();
        let blocks: usize = per.iter().sum();
        let slots = blocks * bs;
        let rnd =
            |shape: (usize, usize, usize)| Tensor::randn(0f32, 1.0, shape, dev)?.to_dtype(dtype);
        // Hand out blocks in reverse so tables are scattered.
        let mut next = blocks as u32;
        let tables = per
            .iter()
            .map(|&n| {
                (0..n)
                    .map(|_| {
                        next -= 1;
                        next
                    })
                    .collect()
            })
            .collect();
        Ok(Case {
            k: rnd((slots, n_kv, hd))?,
            v: rnd((slots, n_kv, hd))?,
            q: rnd((lens.len(), n_heads, hd))?,
            tables,
            lens: lens.to_vec(),
            block_size: bs,
        })
    }

    fn run_op(c: &Case, dev: &Device) -> Result<Tensor> {
        let max_blocks = c.tables.iter().map(Vec::len).max().unwrap();
        let flat: Vec<u32> = c
            .tables
            .iter()
            .flat_map(|t| {
                let mut t = t.clone();
                t.resize(max_blocks, 0);
                t
            })
            .collect();
        let tables = Tensor::from_vec(flat, (c.lens.len(), max_blocks), dev)?;
        let lens = Tensor::new(c.lens.iter().map(|&l| l as u32).collect::<Vec<_>>(), dev)?;
        let op = PagedAttention {
            k_cache: c.k.clone(),
            v_cache: c.v.clone(),
            block_size: c.block_size,
            max_context: c.lens.iter().copied().max().unwrap_or(1),
            #[cfg(feature = "metal")]
            pipelines: Default::default(),
        };
        c.q.apply_op3_no_bwd(&tables, &lens, &op)
    }

    /// v1 attention over gathered rows, for one sequence.
    fn gather_reference(c: &Case, s: usize) -> Result<Tensor> {
        let slots: Vec<u32> = (0..c.lens[s])
            .map(|t| {
                c.tables[s][t / c.block_size] * c.block_size as u32 + (t % c.block_size) as u32
            })
            .collect();
        let idx = Tensor::new(slots, c.k.device())?;
        let g = |t: &Tensor| {
            t.index_select(&idx, 0)?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()
        };
        let q = c.q.i(s)?.unsqueeze(0)?.unsqueeze(2)?.contiguous()?;
        gqa_attention(&q, &g(&c.k)?, &g(&c.v)?, None)?
            .squeeze(0)?
            .squeeze(1)
    }

    fn max_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()
    }

    #[test]
    fn cpu_reference_matches_gather_attention() -> Result<()> {
        let c = case(&Device::Cpu, DType::F32, 8, 2, 64, &[1, 7, 16, 17, 40])?;
        let out = run_op(&c, &Device::Cpu)?;
        for s in 0..c.lens.len() {
            let d = max_diff(&out.i(s)?, &gather_reference(&c, s)?)?;
            assert!(d < 1e-5, "seq {s}: {d}");
        }
        Ok(())
    }

    #[test]
    fn long_contexts_with_few_sequences_are_split() {
        // Llama 3.2 1B: 8 kv heads. One 8k sequence: 32 partitions of 256 positions.
        assert_eq!(split_plan(8, 8192), (32, 256));
        // Eight of them already fill the GPU better: 4 partitions.
        assert_eq!(split_plan(64, 8192), (4, 2048));
        // Short contexts and big batches stay single-pass.
        assert_eq!(split_plan(8, 200), (1, 200));
        assert_eq!(split_plan(512, 8192).0, 1);
    }

    #[test]
    fn rejects_unsupported_layouts() {
        assert!(supported(32, 8, 64));
        assert!(supported(24, 8, 128));
        assert!(
            supported(12, 2, 128),
            "Qwen2.5 1.5B: 6 query heads per kv head"
        );
        assert!(supported(32, 4, 64), "8 query heads per kv head");
        assert!(!supported(32, 2, 64), "16 query heads per kv head");
        assert!(!supported(8, 2, 48), "head_dim not a multiple of 32");
    }

    /// The MSL kernel against the CPU reference: many sequences, contexts shorter and longer
    /// than one simdgroup sweep, both Llama 3.2 head layouts, Qwen2.5 1.5B's (6 query heads
    /// per kv head) and the 8-per-kv-head limit, f32 and f16.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_kernel_matches_cpu_reference() -> Result<()> {
        let metal = Device::new_metal(0)?;
        let lens: Vec<usize> = (0..37).map(|i| 1 + (i * 53) % 700).collect();
        for (n_heads, n_kv, hd) in [
            (32, 8, 64),
            (24, 8, 128),
            (8, 8, 64),
            (12, 2, 128),
            (32, 4, 64),
        ] {
            for (dtype, tol) in [(DType::F32, 1e-4), (DType::F16, 2e-2)] {
                let c = case(&Device::Cpu, dtype, n_heads, n_kv, hd, &lens)?;
                let want = run_op(&c, &Device::Cpu)?;
                let on_metal = Case {
                    k: c.k.to_device(&metal)?,
                    v: c.v.to_device(&metal)?,
                    q: c.q.to_device(&metal)?,
                    tables: c.tables.clone(),
                    lens: c.lens.clone(),
                    block_size: c.block_size,
                };
                let got = run_op(&on_metal, &metal)?.to_device(&Device::Cpu)?;
                let d = max_diff(&got, &want)?;
                assert!(d < tol, "{n_heads}/{n_kv}x{hd} {dtype:?}: max diff {d}");
            }
        }
        Ok(())
    }

    /// The split-context kernels against the CPU reference: few sequences with long contexts
    /// (so the context is partitioned), next to ones shorter than a partition (so some
    /// partitions are empty), for the Llama 3.2 and Qwen2.5 head layouts and both dtypes.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_split_kernel_matches_cpu_reference() -> Result<()> {
        let metal = Device::new_metal(0)?;
        for lens in [vec![8192], vec![1, 5, 300, 4000, 9000]] {
            let max = *lens.iter().max().unwrap();
            assert!(
                split_plan(8 * lens.len(), max).0 > 1,
                "{lens:?} should split"
            );
            for (n_heads, n_kv, hd) in [(32, 8, 64), (24, 8, 128), (12, 2, 128)] {
                for (dtype, tol) in [(DType::F32, 1e-4), (DType::F16, 2e-2)] {
                    let c = case(&Device::Cpu, dtype, n_heads, n_kv, hd, &lens)?;
                    let want = run_op(&c, &Device::Cpu)?;
                    let on_metal = Case {
                        k: c.k.to_device(&metal)?,
                        v: c.v.to_device(&metal)?,
                        q: c.q.to_device(&metal)?,
                        tables: c.tables.clone(),
                        lens: c.lens.clone(),
                        block_size: c.block_size,
                    };
                    let got = run_op(&on_metal, &metal)?.to_device(&Device::Cpu)?;
                    let d = max_diff(&got, &want)?;
                    assert!(
                        d < tol,
                        "{lens:?} {n_heads}/{n_kv}x{hd} {dtype:?}: max diff {d}"
                    );
                }
            }
        }
        Ok(())
    }
}
