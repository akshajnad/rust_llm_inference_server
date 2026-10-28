use std::hash::{BuildHasher, Hasher};

use model::sampling::{RowSampling, row_key};
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
    /// Keep generating through EOS until `max_tokens` (benchmarks need exact output lengths).
    pub ignore_eos: bool,
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
            ignore_eos: false,
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

    /// Greedy or temperature-only: the device sampler can pick the token without the logits
    /// leaving the GPU. Top-k, top-p and repetition penalty take the host path.
    pub fn is_device_sampleable(&self) -> bool {
        self.repetition_penalty == 1.0
            && (self.is_greedy() || (self.top_k == 0 && self.top_p >= 1.0))
    }
}

/// Picks the next token from last-position logits.
///
/// Order: repetition penalty, then greedy if temperature is 0, else temperature, top-k,
/// softmax, top-p, and a draw from a ChaCha8 RNG seeded per request. ChaCha8 is used rather
/// than `StdRng` because its output is fixed by spec, so a seed reproduces across versions.
///
/// Every path is O(vocab) with no sort of the full vocabulary: plain temperature sampling
/// draws in one scan, top-k selects then sorts only k, and top-p grows a selected prefix until
/// it holds the nucleus. Scratch buffers are reused across calls.
#[derive(Debug, Clone)]
pub struct Sampler {
    params: SamplingParams,
    /// Resolved seed (the request's, or a random one).
    seed: u64,
    rng: ChaCha8Rng,
    idx: Vec<u32>,
    weights: Vec<f64>,
}

/// First candidate-set size tried for top-p without top-k; doubled until the nucleus fits.
const NUCLEUS_GUESS: usize = 64;

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        let seed = params.seed.unwrap_or_else(random_u64);
        Self {
            params,
            seed,
            rng: ChaCha8Rng::seed_from_u64(seed),
            idx: Vec::new(),
            weights: Vec::new(),
        }
    }

    pub fn greedy() -> Self {
        Self::new(SamplingParams::greedy(usize::MAX))
    }

    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// How the device sampler should pick the token at `position`, if it can.
    pub fn device_mode(&self, position: usize) -> Option<RowSampling> {
        let p = &self.params;
        match () {
            _ if !p.is_device_sampleable() => None,
            _ if p.is_greedy() => Some(RowSampling::Greedy),
            _ => Some(RowSampling::Temperature {
                temperature: p.temperature,
                key: row_key(self.seed, position as u32),
            }),
        }
    }

    /// `history` holds the token slices the repetition penalty looks at (prompt and output).
    pub fn sample(&mut self, logits: &mut [f32], history: &[&[u32]]) -> u32 {
        let p = self.params.clone();
        if p.repetition_penalty != 1.0 {
            apply_repetition_penalty(logits, history, p.repetition_penalty);
        }
        if p.is_greedy() {
            return argmax(logits);
        }
        let inv_t = 1.0 / p.temperature as f64;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let weight = |l: f32| ((l as f64 - max) * inv_t).exp();
        let vocab = logits.len();
        let top_k = if p.top_k == 0 || p.top_k >= vocab {
            0
        } else {
            p.top_k
        };

        if top_k == 0 && p.top_p >= 1.0 {
            // Unfiltered: draw straight from the whole distribution in vocabulary order.
            self.weights.clear();
            self.weights.extend(logits.iter().map(|&l| weight(l)));
            let total: f64 = self.weights.iter().sum();
            return draw(&self.weights, total, &mut self.rng) as u32;
        }

        // Mass the nucleus is measured against: the whole vocabulary, or the top-k set.
        let full_total = match top_k {
            0 => logits.iter().map(|&l| weight(l)).sum::<f64>(),
            _ => 0.0,
        };
        let mut k = if top_k > 0 {
            top_k
        } else {
            NUCLEUS_GUESS.min(vocab)
        };
        loop {
            self.select_top(logits, k);
            self.weights.clear();
            self.weights
                .extend(self.idx.iter().map(|&i| weight(logits[i as usize])));
            let total = if top_k > 0 {
                self.weights.iter().sum()
            } else {
                full_total
            };
            let keep = if p.top_p < 1.0 {
                nucleus(&self.weights, total, p.top_p as f64)
            } else {
                Some(k)
            };
            match keep {
                Some(n) => {
                    self.weights.truncate(n);
                    let kept: f64 = self.weights.iter().sum();
                    return self.idx[draw(&self.weights, kept, &mut self.rng)];
                }
                // The nucleus is wider than the candidates: grow (only without top-k, where
                // the threshold is against the full vocabulary).
                None if top_k == 0 && k < vocab => k = (k * 4).min(vocab),
                None => return self.idx[draw(&self.weights, total, &mut self.rng)],
            }
        }
    }

    /// Fills `idx` with the `k` highest logits, sorted descending (ties: lower id first).
    fn select_top(&mut self, logits: &[f32], k: usize) {
        let by_logit = |a: &u32, b: &u32| {
            logits[*b as usize]
                .total_cmp(&logits[*a as usize])
                .then(a.cmp(b))
        };
        self.idx.clear();
        self.idx.extend(0..logits.len() as u32);
        if k < self.idx.len() {
            self.idx.select_nth_unstable_by(k - 1, by_logit);
            self.idx.truncate(k);
        }
        self.idx.sort_unstable_by(by_logit);
    }
}

/// Smallest prefix of descending `weights` whose share of `total` reaches `top_p` (at least
/// one), or `None` if all of `weights` together fall short.
fn nucleus(weights: &[f64], total: f64, top_p: f64) -> Option<usize> {
    let mut acc = 0.0;
    for (i, w) in weights.iter().enumerate() {
        acc += w / total;
        if acc >= top_p {
            return Some(i + 1);
        }
    }
    None
}

/// Index drawn with probability proportional to `weights` (which sum to `total`).
fn draw(weights: &[f64], total: f64, rng: &mut ChaCha8Rng) -> usize {
    let mut u = unit_f64(rng) * total;
    for (i, w) in weights.iter().enumerate() {
        if u < *w {
            return i;
        }
        u -= w;
    }
    weights.iter().rposition(|w| *w > 0.0).unwrap_or(0)
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

    /// Empirical frequencies against the exact filtered softmax, for each sampling mode.
    fn check_distribution(top_k: usize, top_p: f32, logits: &[f32]) {
        let mut s = sampler(|p| {
            p.top_k = top_k;
            p.top_p = top_p;
        });
        // Exact target: softmax, top-k, renormalize, nucleus (descending, crossing included).
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        if top_k > 0 {
            order.truncate(top_k);
        }
        let w: Vec<f64> = order.iter().map(|&i| (logits[i] as f64).exp()).collect();
        let total: f64 = w.iter().sum();
        let mut keep = order.len();
        let mut acc = 0.0;
        for (j, x) in w.iter().enumerate() {
            acc += x / total;
            if acc >= top_p as f64 {
                keep = j + 1;
                break;
            }
        }
        let kept: f64 = w[..keep].iter().sum();
        let mut want = vec![0.0; logits.len()];
        for j in 0..keep {
            want[order[j]] = w[j] / kept;
        }
        let n = 40_000;
        let mut got = vec![0.0; logits.len()];
        for _ in 0..n {
            got[s.sample(&mut logits.to_vec(), &[]) as usize] += 1.0 / n as f64;
        }
        for i in 0..logits.len() {
            assert!(
                (got[i] - want[i]).abs() < 0.015,
                "k={top_k} p={top_p} token {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn every_mode_matches_the_exact_distribution() {
        let logits: Vec<f32> = (0..300).map(|i| ((i * 7919) % 300) as f32 / 60.0).collect();
        for (k, p) in [(0, 1.0), (5, 1.0), (0, 0.6), (40, 0.9), (0, 0.98)] {
            check_distribution(k, p, &logits);
        }
    }

    #[test]
    fn wide_nucleus_grows_past_the_first_guess() {
        // Near-flat over 1000 tokens: a 0.5 nucleus needs ~500 candidates, > NUCLEUS_GUESS.
        let logits: Vec<f32> = (0..1000).map(|i| i as f32 * 1e-4).collect();
        let mut s = sampler(|p| p.top_p = 0.5);
        let picks: std::collections::HashSet<u32> = (0..3000)
            .map(|_| s.sample(&mut logits.clone(), &[]))
            .collect();
        assert!(picks.len() > 300, "only {} distinct tokens", picks.len());
        assert!(
            picks.iter().all(|&t| t >= 480),
            "picked outside the top half"
        );
    }

    #[test]
    fn repetition_penalty_matches_transformers() {
        let mut logits = [2.0, -2.0, 1.0];
        apply_repetition_penalty(&mut logits, &[&[0, 1], &[0]], 2.0);
        assert_eq!(logits, [1.0, -4.0, 1.0]);
    }
}
