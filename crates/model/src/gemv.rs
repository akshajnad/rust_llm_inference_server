//! Small-batch decode for Q8_0 weights on Metal: one kernel reads each weight block once for
//! all of up to `MAX_ROWS` input rows. candle 0.11's batched quantized matmul costs ~4x a
//! single-row product at these sizes, and per-row products read the weights once per row.

/// Most input rows the kernel handles (`MAX_R` in `gemv.metal`).
pub const MAX_ROWS: usize = 8;

/// `RLIS_DISABLE_GEMV=1` turns the kernel off, for A/B measurements.
pub fn disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("RLIS_DISABLE_GEMV").is_ok_and(|v| !v.is_empty() && v != "0"))
}

#[cfg(feature = "metal")]
pub use metal_impl::{Q8Weight, gemv};

#[cfg(feature = "metal")]
mod metal_impl {
    use std::sync::OnceLock;

    use candle_core::backend::BackendStorage;
    use candle_core::{CpuStorage, CustomOp1, DType, Layout, MetalStorage, Result, Shape, Tensor};
    use candle_metal_kernels::metal::{Buffer, ComputePipeline};
    use candle_metal_kernels::utils::EncoderProvider;
    use objc2_metal::MTLSize;

    /// A Q8_0 matrix `(n, k)` as raw blocks in a Metal buffer (shared with its `QTensor`).
    #[derive(Debug, Clone)]
    pub struct Q8Weight {
        pub buffer: Buffer,
        pub n: usize,
        pub k: usize,
    }

    /// `x (rows, k)` f32 with `2 <= rows <= MAX_ROWS` -> `(rows, n)` f32.
    pub fn gemv(w: &Q8Weight, x: &Tensor) -> Result<Tensor> {
        x.apply_op1_no_bwd(w)
    }

    const SIMDGROUPS: usize = 8;
    const NR: usize = 4;

    /// One kernel per row count (2..=MAX_ROWS), so loops unroll and accumulators stay in
    /// registers.
    fn pipeline(device: &candle_core::MetalDevice, rows: usize) -> Result<ComputePipeline> {
        static PIPELINES: OnceLock<Vec<ComputePipeline>> = OnceLock::new();
        if PIPELINES.get().is_none() {
            let dev = device.metal_device();
            let lib = dev
                .new_library_with_source(include_str!("gemv.metal"), None)
                .map_err(candle_core::Error::wrap)?;
            let all = (2..=super::MAX_ROWS)
                .map(|r| {
                    let f = lib
                        .get_function(&format!("q8_0_gemv_rows_{r}"), None)
                        .map_err(candle_core::Error::wrap)?;
                    dev.new_compute_pipeline_state_with_function(&f)
                        .map_err(candle_core::Error::wrap)
                })
                .collect::<Result<Vec<_>>>()?;
            let _ = PIPELINES.set(all);
        }
        Ok(PIPELINES.get().expect("initialized above")[rows - 2].clone())
    }

    impl CustomOp1 for Q8Weight {
        fn name(&self) -> &'static str {
            "q8_0-gemv-rows"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("q8_0 gemv runs on Metal only")
        }

        fn metal_fwd(&self, x: &MetalStorage, l: &Layout) -> Result<(MetalStorage, Shape)> {
            let (rows, k) = l.shape().dims2()?;
            if k != self.k
                || !(2..=super::MAX_ROWS).contains(&rows)
                || x.dtype() != DType::F32
                || !l.is_contiguous()
            {
                candle_core::bail!(
                    "q8_0 gemv: bad input {:?} {:?} for ({}, {})",
                    l.shape(),
                    x.dtype(),
                    self.n,
                    self.k
                );
            }
            let device = x.device();
            let out = device
                .new_buffer_builder()
                .with_size_for(rows * self.n, DType::F32)
                .with_label("q8_0_gemv")
                .build()?;
            let pipeline = pipeline(device, rows)?;
            let guard = device.command_encoder()?;
            let provider = &guard;
            let holder = provider.encoder();
            let enc = holder.as_ref();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_input_buffer(0, Some(&self.buffer), 0);
            enc.set_input_buffer(1, Some(x.buffer()), l.start_offset() * 4);
            enc.set_output_buffer(2, Some(&out), 0);
            enc.set_bytes(3, &(self.k as u32));
            enc.set_bytes(4, &(self.n as u32));
            let groups = self.n.div_ceil(SIMDGROUPS * NR);
            enc.dispatch_thread_groups(
                MTLSize {
                    width: groups,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 32 * SIMDGROUPS,
                    height: 1,
                    depth: 1,
                },
            );
            Ok((
                MetalStorage::new(out, device.clone(), rows * self.n, DType::F32),
                Shape::from((rows, self.n)),
            ))
        }
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
    use candle_core::{Device, Module, Tensor};

    /// Against candle's single-row matrix-vector kernel, row by row, for every supported row
    /// count and an output size that isn't a multiple of the rows per threadgroup.
    #[test]
    fn matches_candle_row_by_row() -> anyhow::Result<()> {
        let dev = Device::new_metal(0)?;
        for (n, k) in [(8192, 2048), (2050, 8192)] {
            let w = Tensor::randn(0f32, 0.05, (n, k), &dev)?;
            let (q, weight) = crate::gguf::upload_q8_0(
                QTensor::quantize(&w.to_device(&Device::Cpu)?, GgmlDType::Q8_0)?,
                &dev,
            )?;
            let reference = QMatMul::from_qtensor(q)?;
            for rows in 2..=MAX_ROWS {
                let x = Tensor::randn(0f32, 1.0, (rows, k), &dev)?;
                let got = gemv(&weight, &x)?;
                let want = reference
                    .forward(&x.reshape((rows, 1, k))?)?
                    .reshape((rows, n))?;
                let rel = ((got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?)
                    / want.abs()?.max_all()?.to_scalar::<f32>()?;
                assert!(rel < 1e-4, "({n}, {k}) rows {rows}: relative diff {rel}");
            }
        }
        Ok(())
    }
}
