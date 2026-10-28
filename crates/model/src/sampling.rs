//! Token selection on the device: greedy or plain temperature sampling for a whole batch in one
//! kernel, so only token ids come back to the host instead of `rows x vocab` logits.
//!
//! Temperature sampling uses the Gumbel-max trick with noise from a counter-based hash of
//! `(row key, token id)`. The key comes from the request seed and the token's position, so a
//! draw is reproducible and independent of what else is in the batch. `cpu_fwd` computes the
//! same hash and serves as the reference.

use candle_core::{CpuStorage, CustomOp2, DType, Layout, Result, Shape, Tensor};

pub fn lowbias32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Noise key for drawing the token at `position` of a request seeded with `seed`.
pub fn row_key(seed: u64, position: u32) -> u32 {
    lowbias32(lowbias32(lowbias32(seed as u32) ^ (seed >> 32) as u32) ^ position)
}

/// How to pick one row's token.
#[derive(Debug, Clone, Copy)]
pub enum RowSampling {
    Greedy,
    Temperature { temperature: f32, key: u32 },
}

/// Picks one token per row of `logits (rows, vocab)` f32. The kernel is Metal-only; on CUDA the
/// logits come back to the host and the CPU reference picks the same tokens.
pub fn sample_rows(logits: &Tensor, rows: &[RowSampling]) -> Result<Vec<u32>> {
    let logits = match logits.device() {
        candle_core::Device::Cuda(_) => &logits.to_device(&candle_core::Device::Cpu)?,
        _ => logits,
    };
    let params: Vec<u32> = rows
        .iter()
        .flat_map(|r| match *r {
            RowSampling::Greedy => [0, 0],
            RowSampling::Temperature { temperature, key } => [key, (1.0 / temperature).to_bits()],
        })
        .collect();
    let params = Tensor::from_vec(params, (rows.len(), 2), logits.device())?;
    let logits = logits.to_dtype(DType::F32)?.contiguous()?;
    logits
        .apply_op2_no_bwd(&params, &SampleRows)?
        .to_vec1::<u32>()
}

fn gumbel(key: u32, v: u32) -> f32 {
    let h = lowbias32(key ^ v.wrapping_mul(0x9e37_79b9));
    let u = ((h >> 8) as f32 + 0.5) * (1.0 / 16_777_216.0);
    -(-(u.ln())).ln()
}

struct SampleRows;

impl CustomOp2 for SampleRows {
    fn name(&self) -> &'static str {
        "sample-rows"
    }

    fn cpu_fwd(
        &self,
        l: &CpuStorage,
        ll: &Layout,
        p: &CpuStorage,
        lp: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (rows, vocab) = ll.shape().dims2()?;
        let (CpuStorage::F32(l), CpuStorage::U32(p)) = (l, p) else {
            candle_core::bail!("sample-rows expects f32 logits and u32 params");
        };
        let (l, p) = (&l[ll.start_offset()..], &p[lp.start_offset()..]);
        let out = (0..rows)
            .map(|r| {
                let (key, inv_t) = (p[2 * r], f32::from_bits(p[2 * r + 1]));
                let row = &l[r * vocab..(r + 1) * vocab];
                let mut best = (f32::NEG_INFINITY, u32::MAX);
                for (v, &x) in row.iter().enumerate() {
                    let v = v as u32;
                    let s = if inv_t > 0.0 {
                        x * inv_t + gumbel(key, v)
                    } else {
                        x
                    };
                    if s > best.0 || (s == best.0 && v < best.1) {
                        best = (s, v);
                    }
                }
                best.1
            })
            .collect();
        Ok((CpuStorage::U32(out), Shape::from(rows)))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        l: &candle_core::MetalStorage,
        ll: &Layout,
        p: &candle_core::MetalStorage,
        lp: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_metal_kernels::utils::EncoderProvider;
        use objc2_metal::MTLSize;

        let (rows, vocab) = ll.shape().dims2()?;
        if !(ll.is_contiguous() && lp.is_contiguous()) {
            candle_core::bail!("sample-rows: inputs must be contiguous");
        }
        let device = l.device();
        let pipeline = pipeline(device)?;
        let out = device
            .new_buffer_builder()
            .with_size_for(rows, DType::U32)
            .with_label("sample_rows")
            .build()?;
        let guard = device.command_encoder()?;
        let provider = &guard;
        let holder = provider.encoder();
        let enc = holder.as_ref();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_input_buffer(0, Some(l.buffer()), ll.start_offset() * 4);
        enc.set_input_buffer(1, Some(p.buffer()), lp.start_offset() * 4);
        enc.set_output_buffer(2, Some(&out), 0);
        enc.set_bytes(3, &(vocab as u32));
        enc.dispatch_thread_groups(
            MTLSize {
                width: rows,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            candle_core::MetalStorage::new(out, device.clone(), rows, DType::U32),
            Shape::from(rows),
        ))
    }
}

#[cfg(feature = "metal")]
fn pipeline(
    device: &candle_core::MetalDevice,
) -> Result<candle_metal_kernels::metal::ComputePipeline> {
    use std::sync::OnceLock;
    static PIPELINE: OnceLock<candle_metal_kernels::metal::ComputePipeline> = OnceLock::new();
    if let Some(p) = PIPELINE.get() {
        return Ok(p.clone());
    }
    let dev = device.metal_device();
    let lib = dev
        .new_library_with_source(include_str!("sampling.metal"), None)
        .map_err(candle_core::Error::wrap)?;
    let f = lib
        .get_function("sample_rows", None)
        .map_err(candle_core::Error::wrap)?;
    let p = dev
        .new_compute_pipeline_state_with_function(&f)
        .map_err(candle_core::Error::wrap)?;
    Ok(PIPELINE.get_or_init(|| p).clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn greedy_rows_are_argmax_with_low_index_ties() -> Result<()> {
        let l = Tensor::new(
            &[[0.1f32, 3.0, -1.0, 3.0], [5.0, 1.0, 5.0, 0.0]],
            &Device::Cpu,
        )?;
        assert_eq!(
            sample_rows(&l, &[RowSampling::Greedy, RowSampling::Greedy])?,
            [1, 0]
        );
        Ok(())
    }

    /// Gumbel-max over many keys reproduces softmax(logits / T).
    #[test]
    fn temperature_rows_follow_softmax() -> Result<()> {
        let logits = [0.0f32, 1.0, 2.0, -1.0];
        let t = 0.7f32;
        let n = 20_000;
        let l = Tensor::new(&logits, &Device::Cpu)?
            .unsqueeze(0)?
            .repeat((n, 1))?;
        let rows: Vec<RowSampling> = (0..n as u32)
            .map(|i| RowSampling::Temperature {
                temperature: t,
                key: row_key(42, i),
            })
            .collect();
        let picks = sample_rows(&l, &rows)?;
        let z: f64 = logits.iter().map(|&x| (x as f64 / t as f64).exp()).sum();
        for (v, &x) in logits.iter().enumerate() {
            let want = (x as f64 / t as f64).exp() / z;
            let got = picks.iter().filter(|&&p| p == v as u32).count() as f64 / n as f64;
            assert!((got - want).abs() < 0.015, "token {v}: {got} vs {want}");
        }
        Ok(())
    }

    #[test]
    fn same_key_same_token_regardless_of_neighbours() -> Result<()> {
        let dev = Device::Cpu;
        let l = Tensor::randn(0f32, 2.0, (3, 500), &dev)?;
        let rows = [
            RowSampling::Temperature {
                temperature: 1.0,
                key: row_key(7, 3),
            },
            RowSampling::Greedy,
            RowSampling::Temperature {
                temperature: 0.5,
                key: row_key(9, 0),
            },
        ];
        let together = sample_rows(&l, &rows)?;
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(
                sample_rows(&l.narrow(0, i, 1)?, std::slice::from_ref(r))?[0],
                together[i]
            );
        }
        Ok(())
    }

    /// The Metal kernel picks the same tokens as the CPU reference.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_matches_cpu() -> Result<()> {
        let metal = Device::new_metal(0)?;
        let rows = 64;
        let l = Tensor::randn(0f32, 3.0, (rows, 128_256), &Device::Cpu)?;
        let spec: Vec<RowSampling> = (0..rows as u32)
            .map(|i| match i % 3 {
                0 => RowSampling::Greedy,
                _ => RowSampling::Temperature {
                    temperature: 0.3 + i as f32 / 50.0,
                    key: row_key(i as u64, i),
                },
            })
            .collect();
        let cpu = sample_rows(&l, &spec)?;
        let gpu = sample_rows(&l.to_device(&metal)?, &spec)?;
        assert_eq!(cpu, gpu);
        Ok(())
    }
}
