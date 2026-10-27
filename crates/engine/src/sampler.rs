use anyhow::Result;
use candle_core::{DType, Tensor};

/// Picks the next token from last-position logits.
///
/// M1 is greedy only; temperature, top-k/top-p, repetition penalty and a seeded RNG per
/// request arrive with the server in M2.
#[derive(Debug, Clone, Default)]
pub struct Sampler;

impl Sampler {
    pub fn greedy() -> Self {
        Self
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let v = logits
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(argmax(&v))
    }
}

/// Index of the largest value; ties go to the lowest index, as with `torch.argmax`.
pub fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_prefers_lowest_index_on_ties() {
        assert_eq!(argmax(&[0.1, 3.0, -1.0, 3.0]), 1);
        assert_eq!(argmax(&[f32::NEG_INFINITY, -5.0]), 1);
    }

    #[test]
    fn greedy_sampler_reads_tensor() -> Result<()> {
        let logits = Tensor::new(&[0.5f32, 2.0, 1.0], &candle_core::Device::Cpu)?;
        assert_eq!(Sampler::greedy().sample(&logits)?, 1);
        Ok(())
    }
}
