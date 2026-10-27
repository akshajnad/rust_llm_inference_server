//! The engine thread: owns the model and every sequence's KV cache, so the hot path needs no
//! locks. Requests arrive over a bounded channel (backpressure); each request streams its
//! events back over its own channel, and dropping that receiver cancels the request.
//!
//! M2 schedules with static batching: take whatever is queued (up to `max_batch_size`), run
//! that batch until every sequence finishes, then take the next batch. M3 replaces this with
//! iteration-level scheduling over a paged KV pool.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use model::{ContiguousKvCache, Llama, SeqInput};
use tokio::sync::mpsc;

use crate::detokenizer::Detokenizer;
use crate::sampler::{Sampler, SamplingParams};
use crate::stop::{StopMatcher, StopResult};
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub max_batch_size: usize,
    /// Requests that may wait in the queue before submissions are refused.
    pub queue_size: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            queue_size: 256,
        }
    }
}

pub struct Request {
    pub prompt: Vec<u32>,
    pub params: SamplingParams,
    pub events: mpsc::UnboundedSender<Event>,
    pub arrived: Instant,
}

impl Request {
    /// Builds a request and the receiver its events stream to.
    pub fn new(prompt: Vec<u32>, params: SamplingParams) -> (Self, mpsc::UnboundedReceiver<Event>) {
        let (events, rx) = mpsc::unbounded_channel();
        (
            Self {
                prompt,
                params,
                events,
                arrived: Instant::now(),
            },
            rx,
        )
    }
}

#[derive(Debug, Clone)]
pub enum Event {
    /// Newly completed text, at most one per generated token.
    Text(String),
    Finished(Finish),
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// EOS token or a stop string.
    Stop,
    /// Reached `max_tokens`.
    Length,
}

impl FinishReason {
    pub fn as_openai(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Finish {
    pub reason: FinishReason,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// Arrival (including queueing) to first sampled token.
    pub ttft: Duration,
    /// First token to finish.
    pub decode_time: Duration,
}

impl Finish {
    pub fn decode_tokens_per_sec(&self) -> f64 {
        let n = self.completion_tokens.saturating_sub(1);
        match self.decode_time.as_secs_f64() {
            t if n > 0 && t > 0.0 => n as f64 / t,
            _ => 0.0,
        }
    }
}

/// Live counters shared with the API layer (and Prometheus in M4).
#[derive(Debug, Default)]
pub struct EngineStats {
    pub running: AtomicUsize,
    pub alive: AtomicBool,
}

#[derive(Debug, thiserror::Error)]
pub enum SubmitError {
    #[error("request queue is full")]
    QueueFull,
    #[error("engine has stopped")]
    Stopped,
}

/// Cheap to clone; the engine thread exits once every handle is dropped.
#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<Request>,
    stats: Arc<EngineStats>,
    max_seq_len: usize,
}

impl EngineHandle {
    /// A handle plus the receiving end of its queue, for driving an engine loop (or for tests
    /// that need a queue nobody drains).
    pub fn channel(cfg: &EngineConfig, max_seq_len: usize) -> (Self, mpsc::Receiver<Request>) {
        let (tx, rx) = mpsc::channel(cfg.queue_size.max(1));
        let stats = Arc::new(EngineStats::default());
        stats.alive.store(true, Ordering::Relaxed);
        (
            Self {
                tx,
                stats,
                max_seq_len,
            },
            rx,
        )
    }

    pub fn submit(&self, req: Request) -> Result<(), SubmitError> {
        self.tx.try_send(req).map_err(|e| match e {
            mpsc::error::TrySendError::Full(_) => SubmitError::QueueFull,
            mpsc::error::TrySendError::Closed(_) => SubmitError::Stopped,
        })
    }

    pub fn queue_depth(&self) -> usize {
        self.tx.max_capacity() - self.tx.capacity()
    }

    pub fn running(&self) -> usize {
        self.stats.running.load(Ordering::Relaxed)
    }

    pub fn is_alive(&self) -> bool {
        self.stats.alive.load(Ordering::Relaxed) && !self.tx.is_closed()
    }

    /// Prompt plus completion tokens one sequence may use.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

/// Starts the engine thread, which owns `model` from here on.
pub fn spawn(
    model: Llama,
    tokenizer: Arc<Tokenizer>,
    cfg: EngineConfig,
) -> std::io::Result<EngineHandle> {
    let (handle, rx) = EngineHandle::channel(&cfg, model.max_seq_len());
    let stats = handle.stats.clone();
    std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || {
            run(&model, &tokenizer, &cfg, rx, &stats);
            stats.alive.store(false, Ordering::Relaxed);
        })?;
    Ok(handle)
}

struct Sequence {
    prompt: Vec<u32>,
    output: Vec<u32>,
    kv: ContiguousKvCache,
    sampler: Sampler,
    detok: Detokenizer,
    stop: StopMatcher,
    max_tokens: usize,
    events: mpsc::UnboundedSender<Event>,
    arrived: Instant,
    first_token_at: Option<Instant>,
    finished: bool,
}

impl Sequence {
    fn new(model: &Llama, req: Request) -> Option<Self> {
        let total = req.prompt.len() + req.params.max_tokens;
        let fail = |msg: String| {
            let _ = req.events.send(Event::Error(msg));
            None
        };
        if req.prompt.is_empty() || req.params.max_tokens == 0 {
            return fail("empty prompt or max_tokens of 0".into());
        }
        if total > model.max_seq_len() {
            return fail(format!(
                "{total} tokens exceeds max_seq_len {}",
                model.max_seq_len()
            ));
        }
        let kv = match ContiguousKvCache::new(model.config(), total, model.dtype(), model.device())
        {
            Ok(kv) => kv,
            Err(e) => return fail(format!("allocating kv cache: {e}")),
        };
        Some(Self {
            detok: Detokenizer::new(&req.prompt),
            stop: StopMatcher::new(req.params.stop.clone()),
            max_tokens: req.params.max_tokens,
            sampler: Sampler::new(req.params),
            output: Vec::new(),
            prompt: req.prompt,
            kv,
            events: req.events,
            arrived: req.arrived,
            first_token_at: None,
            finished: false,
        })
    }

    fn send_text(&self, text: String) {
        if !text.is_empty() {
            let _ = self.events.send(Event::Text(text));
        }
    }

    /// Records a sampled token, streams its text, and finishes the sequence when done.
    fn accept(&mut self, token: u32, tokenizer: &Tokenizer) -> anyhow::Result<()> {
        let now = Instant::now();
        self.first_token_at.get_or_insert(now);
        if tokenizer.is_eos(token) {
            return self.finish(FinishReason::Stop, tokenizer);
        }
        self.output.push(token);
        let text = self.detok.push(tokenizer, token)?;
        match self.stop.push(&text) {
            StopResult::Continue(t) => self.send_text(t),
            StopResult::Stopped(t) => {
                self.send_text(t);
                return self.send_finish(FinishReason::Stop);
            }
        }
        if self.output.len() >= self.max_tokens {
            return self.finish(FinishReason::Length, tokenizer);
        }
        Ok(())
    }

    /// Flushes held-back text (which may still complete a stop string), then finishes.
    fn finish(&mut self, reason: FinishReason, tokenizer: &Tokenizer) -> anyhow::Result<()> {
        let tail = self.detok.flush(tokenizer)?;
        match self.stop.push(&tail) {
            StopResult::Continue(t) => {
                self.send_text(t);
                let rest = self.stop.flush();
                self.send_text(rest);
                self.send_finish(reason)
            }
            StopResult::Stopped(t) => {
                self.send_text(t);
                self.send_finish(FinishReason::Stop)
            }
        }
    }

    fn send_finish(&mut self, reason: FinishReason) -> anyhow::Result<()> {
        self.finished = true;
        let first = self.first_token_at.unwrap_or_else(Instant::now);
        let _ = self.events.send(Event::Finished(Finish {
            reason,
            prompt_tokens: self.prompt.len(),
            completion_tokens: self.output.len(),
            ttft: first - self.arrived,
            decode_time: first.elapsed(),
        }));
        Ok(())
    }

    fn cancelled(&self) -> bool {
        self.events.is_closed()
    }
}

fn run(
    model: &Llama,
    tokenizer: &Tokenizer,
    cfg: &EngineConfig,
    mut rx: mpsc::Receiver<Request>,
    stats: &EngineStats,
) {
    tracing::info!(max_batch_size = cfg.max_batch_size, "engine started");
    while let Some(first) = rx.blocking_recv() {
        let mut batch: Vec<Sequence> = Sequence::new(model, first).into_iter().collect();
        while batch.len() < cfg.max_batch_size {
            match rx.try_recv() {
                Ok(req) => batch.extend(Sequence::new(model, req)),
                Err(_) => break,
            }
        }
        tracing::debug!(batch = batch.len(), "batch started");
        while !batch.is_empty() {
            batch.retain(|s| !s.cancelled());
            stats.running.store(batch.len(), Ordering::Relaxed);
            if batch.is_empty() {
                break;
            }
            if let Err(e) = step(model, tokenizer, &mut batch) {
                tracing::error!("engine step failed: {e:#}");
                for s in &batch {
                    let _ = s.events.send(Event::Error(format!("{e:#}")));
                }
                batch.clear();
            }
            batch.retain(|s| !s.finished);
        }
        stats.running.store(0, Ordering::Relaxed);
    }
    tracing::info!("engine stopped");
}

/// One forward pass over the batch (prefill for new sequences, one decode token for the
/// rest), then one sampled token per sequence.
fn step(model: &Llama, tokenizer: &Tokenizer, batch: &mut [Sequence]) -> anyhow::Result<()> {
    let mut inputs: Vec<SeqInput> = batch
        .iter_mut()
        .map(|s| {
            let Sequence {
                prompt, output, kv, ..
            } = s;
            let tokens = match output.last() {
                Some(last) => std::slice::from_ref(last),
                None => prompt.as_slice(),
            };
            SeqInput { tokens, kv }
        })
        .collect();
    let logits = model.forward_batch(&mut inputs)?.to_vec2::<f32>()?;
    drop(inputs);
    for (seq, mut row) in batch.iter_mut().zip(logits) {
        let token = seq.sampler.sample(&mut row, &[&seq.prompt, &seq.output]);
        seq.accept(token, tokenizer)?;
    }
    Ok(())
}
