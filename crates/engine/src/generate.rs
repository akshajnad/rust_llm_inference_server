use std::time::{Duration, Instant};

use anyhow::Result;
use model::{ContiguousKvCache, Llama};

use crate::detokenizer::Detokenizer;
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Eos,
    MaxTokens,
}

#[derive(Debug)]
pub struct Generation {
    pub prompt_tokens: usize,
    pub tokens: Vec<u32>,
    pub finish: FinishReason,
    /// Request start to first sampled token (prefill plus one sample).
    pub ttft: Duration,
    /// Time spent producing tokens after the first.
    pub decode_time: Duration,
}

impl Generation {
    /// Decode throughput, excluding the prefill step.
    pub fn decode_tokens_per_sec(&self) -> f64 {
        let n = self.tokens.len().saturating_sub(1);
        if n == 0 || self.decode_time.is_zero() {
            return 0.0;
        }
        n as f64 / self.decode_time.as_secs_f64()
    }
}

/// Generates for one request with its own contiguous KV cache, calling `on_text` with each
/// newly completed piece of text. Stops at an EOS token (not emitted) or after `max_tokens`.
pub fn generate(
    model: &Llama,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    prompt: &[u32],
    max_tokens: usize,
    mut on_text: impl FnMut(&str),
) -> Result<Generation> {
    anyhow::ensure!(!prompt.is_empty(), "empty prompt");
    let capacity = (prompt.len() + max_tokens).min(model.max_seq_len());
    anyhow::ensure!(
        prompt.len() < capacity,
        "prompt of {} tokens leaves no room under max_seq_len {}",
        prompt.len(),
        model.max_seq_len()
    );
    let max_tokens = max_tokens.min(capacity - prompt.len());
    let mut kv = ContiguousKvCache::new(model.config(), capacity, model.dtype(), model.device())?;
    let mut detok = Detokenizer::new(prompt);

    let start = Instant::now();
    let mut next = sampler.sample(&model.forward(prompt, &mut kv)?)?;
    let ttft = start.elapsed();
    let decode_start = Instant::now();

    let mut tokens = Vec::with_capacity(max_tokens);
    let finish = loop {
        if tokenizer.is_eos(next) {
            break FinishReason::Eos;
        }
        tokens.push(next);
        on_text(&detok.push(tokenizer, next)?);
        if tokens.len() == max_tokens {
            break FinishReason::MaxTokens;
        }
        next = sampler.sample(&model.forward(&[next], &mut kv)?)?;
    };
    on_text(&detok.flush(tokenizer)?);

    Ok(Generation {
        prompt_tokens: prompt.len(),
        tokens,
        finish,
        ttft,
        decode_time: decode_start.elapsed(),
    })
}
