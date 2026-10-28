//! A fixed, reproducible prompt set: slices of a text corpus with log-normal input lengths and
//! log-normal target output lengths (benchmark requests use `ignore_eos`, so every engine
//! generates exactly the target).

use std::path::Path;

use anyhow::Result;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prompt {
    pub prompt: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct LengthDist {
    pub median: f64,
    pub sigma: f64,
    pub min: usize,
    pub max: usize,
}

impl LengthDist {
    fn sample(&self, rng: &mut ChaCha8Rng) -> usize {
        let z = standard_normal(rng);
        ((self.median.ln() + self.sigma * z).exp().round() as usize).clamp(self.min, self.max)
    }
}

fn unit(rng: &mut ChaCha8Rng) -> f64 {
    ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// Box-Muller.
fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    let (u1, u2) = (unit(rng), unit(rng));
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

pub fn generate(
    corpus: &Path,
    tokenizer_dir: &Path,
    n: usize,
    seed: u64,
    input: LengthDist,
    output: LengthDist,
) -> Result<Vec<Prompt>> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer_dir.join("tokenizer.json"))
        .map_err(anyhow::Error::msg)?;
    let text = std::fs::read_to_string(corpus)?;
    let ids = tok
        .encode(text.as_str(), false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    anyhow::ensure!(ids.len() > input.max * 2, "corpus too short");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let len = input.sample(&mut rng);
            let start = (rng.next_u64() % (ids.len() - len) as u64) as usize;
            let prompt = tok
                .decode(&ids[start..start + len], true)
                .map_err(anyhow::Error::msg)?;
            Ok(Prompt {
                prompt,
                input_tokens: len,
                output_tokens: output.sample(&mut rng),
            })
        })
        .collect()
}

pub fn load(path: &Path) -> Result<Vec<Prompt>> {
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths_are_clamped_and_centered() {
        let d = LengthDist {
            median: 128.0,
            sigma: 0.6,
            min: 16,
            max: 512,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut v: Vec<usize> = (0..4001).map(|_| d.sample(&mut rng)).collect();
        assert!(v.iter().all(|&x| (16..=512).contains(&x)));
        v.sort();
        let median = v[2000] as f64;
        assert!((110.0..150.0).contains(&median), "median {median}");
    }
}
