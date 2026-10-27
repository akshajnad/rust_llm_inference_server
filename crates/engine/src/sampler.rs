use std::hash::{BuildHasher, Hasher};

use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{Rng, SeedableRng};

/// Per-request sampling settings, already validated by the API layer.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// 0 means greedy.
    pub temperature: f32,
    /// 1.0 disables nucleus filtering.
    pub top_p: f32,
    /// 0 disables top-k filtering.
    pub top_k: usize,
    /// 1.0 disables it; applied HF-style to every token seen in prompt or output.
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
    pub max_tokens: usize,
    pub stop: Vec<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            seed: None,
            max_tokens: 16,
            stop: Vec::new(),
        }
    }
}

impl SamplingParams {
    pub fn greedy(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            ..Self::default()
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature < 1e-5
    }
}

/// Picks the next token from last-position logits.
///
/// Order: repetition penalty, then greedy if temperature is 0, else temperature, top-k,
/// softmax, top-p, and a draw from a ChaCha8 RNG seeded per request. ChaCha8 is used rather
/// than `StdRng` because its output is fixed by spec, so a seed reproduces across versions.
#[derive(Debug, Clone)]
pub struct Sampler {
    params: SamplingParams,
    rng: ChaCha8Rng,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        let seed = params.seed.unwrap_or_else(random_u64);
        Self {
            params,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    pub fn greedy() -> Self {
        Self::new(SamplingParams::greedy(usize::MAX))
    }

    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// `history` holds the token slices the repetition penalty looks at (prompt and output).
    pub fn sample(&mut self, logits: &mut [f32], history: &[&[u32]]) -> u32 {
        let p = &self.params;
        if p.repetition_penalty != 1.0 {
            apply_repetition_penalty(logits, history, p.repetition_penalty);
        }
        if p.is_greedy() {
            return argmax(logits);
        }
        let mut cand: Vec<u32> = (0..logits.len() as u32).collect();
        if p.top_k > 0 && p.top_k < cand.len() {
            cand.select_nth_unstable_by(p.top_k - 1, |&a, &b| {
                logits[b as usize].total_cmp(&logits[a as usize])
            });
            cand.truncate(p.top_k);
        }
        cand.sort_unstable_by(|&a, &b| {
            logits[b as usize]
                .total_cmp(&logits[a as usize])
                .then(a.cmp(&b))
        });

        let inv_t = 1.0 / p.temperature as f64;
        let max = logits[cand[0] as usize] as f64 * inv_t;
        let mut probs: Vec<f64> = cand
            .iter()
            .map(|&i| (logits[i as usize] as f64 * inv_t - max).exp())
            .collect();
        let total: f64 = probs.iter().sum();
        if p.top_p < 1.0 {
            // Keep the smallest prefix whose mass reaches top_p (always at least one token).
            let mut acc = 0.0;
            let keep = probs
                .iter()
                .position(|&q| {
                    acc += q / total;
                    acc >= p.top_p as f64
                })
                .map_or(probs.len(), |i| i + 1);
            probs.truncate(keep);
        }
        let total: f64 = probs.iter().sum();
        let mut u = unit_f64(&mut self.rng) * total;
        for (i, q) in probs.iter().enumerate() {
            if u < *q {
                return cand[i];
            }
            u -= q;
        }
        cand[probs.len() - 1]
    }
}

/// transformers' `RepetitionPenaltyLogitsProcessor`: each distinct seen token's logit is
/// divided by the penalty if positive, multiplied if negative.
fn apply_repetition_penalty(logits: &mut [f32], history: &[&[u32]], penalty: f32) {
    let mut seen = vec![false; logits.len()];
    for &t in history.iter().flat_map(|h| h.iter()) {
        let t = t as usize;
        if t < logits.len() && !seen[t] {
            seen[t] = true;
            let l = &mut logits[t];
            *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
        }
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

/// Uniform in [0, 1) from the top 53 bits.
fn unit_f64(rng: &mut ChaCha8Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// A random seed for requests that don't set one, from std's per-process random keys.
pub fn random_u64() -> u64 {
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos()),
    );
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampler(f: impl FnOnce(&mut SamplingParams)) -> Sampler {
        let mut p = SamplingParams {
            temperature: 1.0,
            seed: Some(7),
            ..Default::default()
        };
        f(&mut p);
        Sampler::new(p)
    }

    #[test]
    fn argmax_prefers_lowest_index_on_ties() {
        assert_eq!(argmax(&[0.1, 3.0, -1.0, 3.0]), 1);
        assert_eq!(argmax(&[f32::NEG_INFINITY, -5.0]), 1);
    }

    #[test]
    fn zero_temperature_is_greedy() {
        let mut s = sampler(|p| p.temperature = 0.0);
        assert_eq!(s.sample(&mut [0.5, 2.0, 1.0], &[]), 1);
    }

    #[test]
    fn same_seed_same_tokens() {
        let logits: Vec<f32> = (0..50).map(|i| (i as f32 * 0.37).sin()).collect();
        let run = || {
            let mut s = sampler(|_| {});
            (0..20)
                .map(|_| s.sample(&mut logits.clone(), &[]))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
        let distinct: std::collections::HashSet<_> = run().into_iter().collect();
        assert!(
            distinct.len() > 1,
            "temperature 1 over flat logits should vary"
        );
    }

    #[test]
    fn top_k_one_is_greedy() {
        let mut s = sampler(|p| p.top_k = 1);
        for _ in 0..20 {
            assert_eq!(s.sample(&mut [0.0, 1.0, 0.9, 0.8], &[]), 1);
        }
    }

    #[test]
    fn top_p_cuts_the_tail() {
        // Probabilities about 0.88 / 0.12 / ~0: top_p 0.5 must always pick token 0.
        let mut s = sampler(|p| p.top_p = 0.5);
        for _ in 0..50 {
            assert_eq!(s.sample(&mut [4.0, 2.0, -10.0], &[]), 0);
        }
    }

    #[test]
    fn sampling_matches_softmax_frequencies() {
        let mut s = sampler(|_| {});
        let n = 20_000;
        let ones = (0..n)
            .filter(|_| s.sample(&mut [0.0, 1.0f32.ln() + 1.0], &[]) == 1)
            .count();
        let want = 1.0f64.exp() / (1.0 + 1.0f64.exp());
        assert!((ones as f64 / n as f64 - want).abs() < 0.02);
    }

    #[test]
    fn repetition_penalty_matches_transformers() {
        let mut logits = [2.0, -2.0, 1.0];
        apply_repetition_penalty(&mut logits, &[&[0, 1], &[0]], 2.0);
        assert_eq!(logits, [1.0, -4.0, 1.0]);
    }
}
