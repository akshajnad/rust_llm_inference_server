//! Perplexity over a text corpus, computed the way llama.cpp's `perplexity` tool does so the
//! numbers are comparable: split the tokens into `ctx`-sized windows, put BOS at the start of
//! each, and score only the second half of each window (every scored token has at least
//! `ctx / 2` tokens of context).

use std::time::Instant;

use anyhow::Result;
use candle_core::{D, Tensor};
use model::{ContiguousKvCache, Llama};

#[derive(Debug, Clone, serde::Serialize)]
pub struct Perplexity {
    pub ppl: f64,
    /// One standard error of `ppl`, from the per-token NLL variance (as llama.cpp reports).
    pub ppl_stderr: f64,
    pub mean_nll: f64,
    pub tokens_scored: usize,
    pub windows: usize,
    pub ctx: usize,
    pub seconds: f64,
}

/// Scores `tokens` (the encoded corpus) in windows of `ctx`; `max_windows` of 0 means all.
/// `progress` is called after each window with (done, total, running ppl).
pub fn perplexity(
    model: &Llama,
    tokens: &[u32],
    bos: u32,
    ctx: usize,
    max_windows: usize,
    mut progress: impl FnMut(usize, usize, f64),
) -> Result<Perplexity> {
    anyhow::ensure!(ctx >= 4, "ctx must be at least 4");
    let mut windows = tokens.len() / ctx;
    if max_windows > 0 {
        windows = windows.min(max_windows);
    }
    anyhow::ensure!(
        windows > 0,
        "corpus of {} tokens is shorter than one window of {ctx}",
        tokens.len()
    );
    let first = ctx / 2;
    let (mut sum, mut sum_sq, mut n) = (0f64, 0f64, 0usize);
    let start = Instant::now();
    for w in 0..windows {
        let mut chunk = tokens[w * ctx..(w + 1) * ctx].to_vec();
        chunk[0] = bos;
        let mut kv = ContiguousKvCache::new(model.config(), ctx, model.dtype(), model.device())?;
        // Rows first..ctx-1 predict tokens first+1..ctx; the last row has no target.
        let logits = model.forward_logits_from(&chunk, &mut kv, first)?;
        let logits = logits.narrow(0, 0, ctx - 1 - first)?;
        let targets: Vec<u32> = chunk[first + 1..].to_vec();
        let targets = Tensor::from_vec(targets, (ctx - 1 - first, 1), model.device())?;
        let logp = candle_nn::ops::log_softmax(&logits, D::Minus1)?
            .gather(&targets, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for lp in logp {
            let nll = -(lp as f64);
            sum += nll;
            sum_sq += nll * nll;
            n += 1;
        }
        progress(w + 1, windows, (sum / n as f64).exp());
    }
    let mean = sum / n as f64;
    let var = (sum_sq / n as f64 - mean * mean).max(0.0);
    let ppl = mean.exp();
    Ok(Perplexity {
        ppl,
        ppl_stderr: ppl * (var / n as f64).sqrt(),
        mean_nll: mean,
        tokens_scored: n,
        windows,
        ctx,
        seconds: start.elapsed().as_secs_f64(),
    })
}
