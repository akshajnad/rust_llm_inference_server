//! The engine thread: owns the model and every sequence's KV cache, so the hot path needs no
//! locks. Requests arrive over a bounded channel (backpressure); each request streams its
//! events back over its own channel, and dropping that receiver cancels the request.
//!
//! Each iteration: take new requests off the queue, drop cancelled ones, ask the scheduler
//! what to run, run one forward pass over the shared paged KV pool, and sample one token for
//! every sequence whose tokens are all computed.
//!
//! A request can ask for several completions of one prompt (`Request::with_forks`): the prompt
//! is prefilled once, then the sequence forks into children that share its KV blocks and
//! sample their first tokens from the same logits, each with its own sampling parameters.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use candle_core::Tensor;
use kvcache::SeqId;
use model::sampling::{RowSampling, sample_rows};
use model::{AttentionImpl, Llama, PagedBatch, PagedKvCache, PagedSeq};
use tokio::sync::mpsc;

use crate::detokenizer::Detokenizer;
use crate::metrics as m;
use crate::sampler::{Sampler, SamplingParams};
use crate::scheduler::{Policy, Schedule, Scheduler, SchedulerConfig};
use crate::stop::{StopMatcher, StopResult};
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Sequences in one step.
    pub max_num_seqs: usize,
    /// Tokens in one step; longer prompts are prefilled in chunks.
    pub max_batched_tokens: usize,
    /// Requests that may wait in the queue before submissions are refused.
    pub queue_size: usize,
    /// Size of the preallocated KV pool. Never derived from free memory: on unified memory the
    /// GPU shares RAM with everything else.
    pub kv_cache_bytes: usize,
    pub block_size: usize,
    pub policy: Policy,
    /// Decode attention; `Kernel` falls back to `Gather` off Metal.
    pub attention: AttentionImpl,
    /// Reuse cached KV blocks for shared prompt prefixes (and cheap recompute after preemption).
    pub prefix_caching: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 64,
            max_batched_tokens: 2048,
            queue_size: 256,
            kv_cache_bytes: 1 << 30,
            block_size: 16,
            policy: Policy::Continuous,
            attention: AttentionImpl::Kernel,
            prefix_caching: true,
        }
    }
}

pub struct Request {
    pub prompt: Vec<u32>,
    pub params: SamplingParams,
    pub events: mpsc::UnboundedSender<Event>,
    pub arrived: Instant,
    /// More completions of the same prompt, forked off once it is prefilled.
    pub forks: Vec<Fork>,
}

/// One extra completion of a forked request, with its own parameters (seed) and stream.
pub struct Fork {
    pub params: SamplingParams,
    pub events: mpsc::UnboundedSender<Event>,
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
                forks: Vec::new(),
            },
            rx,
        )
    }

    /// One completion per entry of `params`, sharing one prefill of `prompt`. Receivers are in
    /// the order of `params`.
    pub fn with_forks(
        prompt: Vec<u32>,
        params: Vec<SamplingParams>,
    ) -> (Self, Vec<mpsc::UnboundedReceiver<Event>>) {
        let mut params = params.into_iter();
        let (mut req, rx) = Self::new(prompt, params.next().expect("at least one completion"));
        let mut rxs = vec![rx];
        for params in params {
            let (events, rx) = mpsc::unbounded_channel();
            req.forks.push(Fork { params, events });
            rxs.push(rx);
        }
        (req, rxs)
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
    /// Admitted-but-not-running: new and preempted sequences inside the engine.
    pub waiting: AtomicUsize,
    pub preemptions: AtomicU64,
    /// KV block utilization in parts per million.
    pub kv_usage_ppm: AtomicU64,
    /// KV blocks referenced by some sequence, of `kv_blocks_total`, each `kv_block_size` tokens.
    pub kv_blocks_used: AtomicUsize,
    pub kv_blocks_total: AtomicUsize,
    pub kv_block_size: AtomicUsize,
    /// Prompt tokens admitted, and how many came from the prefix cache.
    pub admitted_tokens: AtomicU64,
    pub cached_tokens: AtomicU64,
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

    pub fn stats(&self) -> &EngineStats {
        &self.stats
    }

    /// Fraction of KV blocks in use, 0.0 to 1.0.
    pub fn kv_utilization(&self) -> f64 {
        self.stats.kv_usage_ppm.load(Ordering::Relaxed) as f64 / 1e6
    }

    pub fn is_alive(&self) -> bool {
        self.stats.alive.load(Ordering::Relaxed) && !self.tx.is_closed()
    }

    /// Prompt plus completion tokens one sequence may use.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

/// Starts the engine thread, which owns `model` and the KV pool from here on.
pub fn spawn(
    model: Llama,
    tokenizer: Arc<Tokenizer>,
    cfg: EngineConfig,
) -> anyhow::Result<EngineHandle> {
    let num_blocks = PagedKvCache::blocks_for_bytes(
        model.config(),
        cfg.kv_cache_bytes,
        cfg.block_size,
        model.kv_dtype(),
    );
    anyhow::ensure!(
        num_blocks >= 2,
        "kv cache of {} bytes holds fewer than 2 blocks",
        cfg.kv_cache_bytes
    );
    let sched_cfg = SchedulerConfig {
        max_num_seqs: cfg.max_num_seqs,
        max_batched_tokens: cfg.max_batched_tokens.max(cfg.max_num_seqs),
        num_blocks,
        block_size: cfg.block_size,
        watermark_blocks: (num_blocks / 100).max(1),
        policy: cfg.policy,
        prefix_caching: cfg.prefix_caching,
    };
    let mut cache = PagedKvCache::new(
        model.config(),
        num_blocks,
        cfg.block_size,
        model.kv_dtype(),
        model.device(),
    )?;
    let attention = cache.set_attention(cfg.attention);
    let max_seq_len = model.max_seq_len().min(sched_cfg.max_seq_tokens());
    tracing::info!(
        num_blocks,
        block_size = cfg.block_size,
        tokens = num_blocks * cfg.block_size,
        mib = cfg.kv_cache_bytes >> 20,
        max_seq_len,
        ?attention,
        "kv cache allocated"
    );
    let (handle, rx) = EngineHandle::channel(&cfg, max_seq_len);
    let stats = handle.stats.clone();
    stats.kv_blocks_total.store(num_blocks, Ordering::Relaxed);
    stats.kv_block_size.store(cfg.block_size, Ordering::Relaxed);
    std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || {
            let mut engine = Engine {
                model,
                tokenizer,
                cache,
                sched: Scheduler::new(sched_cfg),
                seqs: HashMap::new(),
                next_id: 0,
                max_seq_len,
                stats: stats.clone(),
            };
            engine.run(rx);
            stats.alive.store(false, Ordering::Relaxed);
        })?;
    Ok(handle)
}

struct Sequence {
    /// Prompt followed by everything generated so far.
    tokens: Vec<u32>,
    prompt_len: usize,
    sampler: Sampler,
    detok: Detokenizer,
    stop: StopMatcher,
    max_tokens: usize,
    ignore_eos: bool,
    events: mpsc::UnboundedSender<Event>,
    arrived: Instant,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
    finished: bool,
    /// Children not yet forked (until the prompt is computed).
    forks: Vec<Fork>,
}

impl Sequence {
    fn new(req: Request, max_seq_len: usize, max_num_seqs: usize) -> Option<Self> {
        let max_tokens = std::iter::once(&req.params)
            .chain(req.forks.iter().map(|f| &f.params))
            .map(|p| p.max_tokens);
        let error = if req.prompt.is_empty() || max_tokens.clone().any(|m| m == 0) {
            Some("empty prompt or max_tokens of 0".to_string())
        } else if let Some(total) = max_tokens
            .map(|m| req.prompt.len() + m)
            .find(|&t| t > max_seq_len)
        {
            Some(format!("{total} tokens exceeds max_seq_len {max_seq_len}"))
        } else if req.forks.len() >= max_num_seqs {
            Some(format!(
                "{} completions exceed max_num_seqs {max_num_seqs}",
                req.forks.len() + 1
            ))
        } else {
            None
        };
        if let Some(e) = error {
            for events in std::iter::once(&req.events).chain(req.forks.iter().map(|f| &f.events)) {
                let _ = events.send(Event::Error(e.clone()));
            }
            return None;
        }
        Some(Self {
            detok: Detokenizer::new(&req.prompt),
            stop: StopMatcher::new(req.params.stop.clone()),
            max_tokens: req.params.max_tokens,
            ignore_eos: req.params.ignore_eos,
            sampler: Sampler::new(req.params),
            prompt_len: req.prompt.len(),
            tokens: req.prompt,
            events: req.events,
            arrived: req.arrived,
            first_token_at: None,
            last_token_at: None,
            finished: false,
            forks: req.forks,
        })
    }

    /// A child of this sequence (whose prompt is computed, nothing generated yet).
    fn fork(&self, f: Fork) -> Self {
        Self {
            detok: Detokenizer::new(&self.tokens),
            stop: StopMatcher::new(f.params.stop.clone()),
            max_tokens: f.params.max_tokens,
            ignore_eos: f.params.ignore_eos,
            sampler: Sampler::new(f.params),
            prompt_len: self.prompt_len,
            tokens: self.tokens.clone(),
            events: f.events,
            arrived: self.arrived,
            first_token_at: None,
            last_token_at: None,
            finished: false,
            forks: Vec::new(),
        }
    }

    fn output_len(&self) -> usize {
        self.tokens.len() - self.prompt_len
    }

    fn send_text(&self, text: String) {
        if !text.is_empty() {
            let _ = self.events.send(Event::Text(text));
        }
    }

    /// Records a sampled token, streams its text, and finishes the sequence when done.
    fn accept(&mut self, token: u32, tokenizer: &Tokenizer) -> anyhow::Result<()> {
        let now = Instant::now();
        match self.last_token_at {
            None => metrics::histogram!(m::TTFT).record((now - self.arrived).as_secs_f64()),
            Some(prev) => metrics::histogram!(m::ITL).record((now - prev).as_secs_f64()),
        }
        self.first_token_at.get_or_insert(now);
        self.last_token_at = Some(now);
        if tokenizer.is_eos(token) && !self.ignore_eos {
            return self.finish(FinishReason::Stop, tokenizer);
        }
        self.tokens.push(token);
        metrics::counter!(m::GENERATED_TOKENS).increment(1);
        let text = self.detok.push(tokenizer, token)?;
        match self.stop.push(&text) {
            StopResult::Continue(t) => self.send_text(t),
            StopResult::Stopped(t) => {
                self.send_text(t);
                return self.send_finish(FinishReason::Stop);
            }
        }
        if self.output_len() >= self.max_tokens {
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
        let finish = Finish {
            reason,
            prompt_tokens: self.prompt_len,
            completion_tokens: self.output_len(),
            ttft: first - self.arrived,
            decode_time: first.elapsed(),
        };
        metrics::counter!(m::REQUESTS_FINISHED, "reason" => reason.as_openai()).increment(1);
        if finish.completion_tokens > 1 {
            metrics::histogram!(m::REQUEST_DECODE_TPS).record(finish.decode_tokens_per_sec());
        }
        let _ = self.events.send(Event::Finished(finish));
        Ok(())
    }

    /// Its client and every child still waiting to fork have gone away.
    fn cancelled(&self) -> bool {
        self.events.is_closed() && self.forks.iter().all(|f| f.events.is_closed())
    }

    fn send_error(&self, msg: &str) {
        for events in std::iter::once(&self.events).chain(self.forks.iter().map(|f| &f.events)) {
            let _ = events.send(Event::Error(msg.to_string()));
        }
    }
}

struct Engine {
    model: Llama,
    tokenizer: Arc<Tokenizer>,
    cache: PagedKvCache,
    sched: Scheduler,
    seqs: HashMap<SeqId, Sequence>,
    next_id: SeqId,
    max_seq_len: usize,
    stats: Arc<EngineStats>,
}

impl Engine {
    fn run(&mut self, mut rx: mpsc::Receiver<Request>) {
        tracing::info!(policy = ?self.sched.config().policy, "engine started");
        // Each iteration drains its own autorelease pool: Metal hands back autoreleased
        // encoders and strings, and a long-lived plain thread otherwise keeps them all until it
        // exits (~25 KB per request, found by the soak test).
        while model::autorelease_pool(|| self.iterate(&mut rx)) {}
        tracing::info!("engine stopped");
    }

    /// One engine iteration; returns false once every handle is gone and no work is left.
    fn iterate(&mut self, rx: &mut mpsc::Receiver<Request>) -> bool {
        if !self.sched.has_work() {
            self.publish_stats();
            match rx.blocking_recv() {
                Some(req) => self.add(req),
                None => return false,
            }
        }
        while let Ok(req) = rx.try_recv() {
            self.add(req);
        }
        self.drop_cancelled();
        if !self.sched.has_work() {
            return true;
        }
        let step = self.sched.schedule();
        if step.seqs.is_empty() {
            // Admission guarantees any queued sequence fits an idle pool, so this means a
            // scheduler bug; fail loudly rather than spin.
            tracing::error!("scheduler made no progress");
            self.fail_all("scheduler made no progress");
            return true;
        }
        if let Err(e) = self.step(&step) {
            // Only this step's sequences may have half-written caches; the rest are intact.
            tracing::error!("engine step failed: {e:#}");
            let ids: Vec<SeqId> = step.seqs.iter().map(|q| q.id).collect();
            self.fail(&ids, &format!("{e:#}"));
        }
        self.publish_stats();
        true
    }

    fn add(&mut self, req: Request) {
        let max_num_seqs = self.sched.config().max_num_seqs;
        if let Some(seq) = Sequence::new(req, self.max_seq_len, max_num_seqs) {
            let id = self.next_id;
            self.next_id += 1;
            self.sched
                .add_forked(id, seq.tokens.clone(), seq.forks.len());
            self.seqs.insert(id, seq);
        }
    }

    /// Forks `id`'s pending children now that its prompt is computed; returns their ids.
    /// Children whose clients already left are dropped next iteration like any other.
    fn fork_children(&mut self, id: SeqId) -> Vec<SeqId> {
        let parent = self.seqs.get_mut(&id).expect("scheduled sequence exists");
        let forks = std::mem::take(&mut parent.forks);
        let mut children = Vec::new();
        for f in forks {
            let child = self.seqs[&id].fork(f);
            let child_id = self.next_id;
            self.next_id += 1;
            self.sched.fork(id, child_id);
            self.seqs.insert(child_id, child);
            children.push(child_id);
        }
        children
    }

    /// A dropped event receiver means the client went away: free its blocks now.
    fn drop_cancelled(&mut self) {
        let gone: Vec<SeqId> = self
            .seqs
            .iter()
            .filter(|(_, s)| s.cancelled())
            .map(|(&id, _)| id)
            .collect();
        for id in gone {
            self.sched.remove(id);
            self.seqs.remove(&id);
        }
    }

    fn fail_all(&mut self, msg: &str) {
        let ids: Vec<SeqId> = self.seqs.keys().copied().collect();
        self.fail(&ids, msg);
    }

    fn fail(&mut self, ids: &[SeqId], msg: &str) {
        for id in ids {
            if let Some(seq) = self.seqs.remove(id) {
                seq.send_error(msg);
            }
            self.sched.remove(*id);
        }
    }

    fn publish_stats(&self) {
        metrics::gauge!(m::RUNNING).set(self.sched.num_running() as f64);
        metrics::gauge!(m::WAITING).set(self.sched.num_waiting() as f64);
        metrics::gauge!(m::KV_UTILIZATION).set(self.sched.kv_utilization());
        metrics::counter!(m::PREEMPTIONS).absolute(self.sched.num_preemptions());
        let (admitted, cached) = self.sched.prefix_cache_stats();
        metrics::counter!(m::ADMITTED_TOKENS).absolute(admitted);
        metrics::counter!(m::PREFIX_CACHE_HIT_TOKENS).absolute(cached);
        self.stats
            .admitted_tokens
            .store(admitted, Ordering::Relaxed);
        self.stats.cached_tokens.store(cached, Ordering::Relaxed);
        let s = &self.stats;
        s.running.store(self.sched.num_running(), Ordering::Relaxed);
        s.waiting.store(self.sched.num_waiting(), Ordering::Relaxed);
        s.preemptions
            .store(self.sched.num_preemptions(), Ordering::Relaxed);
        s.kv_usage_ppm.store(
            (self.sched.kv_utilization() * 1e6) as u64,
            Ordering::Relaxed,
        );
        let total = self.sched.config().num_blocks;
        s.kv_blocks_used
            .store(total - self.sched.num_free_blocks(), Ordering::Relaxed);
    }

    /// One forward pass over everything scheduled, then one sampled token for each sequence
    /// that is fully computed. Partially prefilled sequences just advance.
    fn step(&mut self, step: &Schedule) -> anyhow::Result<()> {
        if !step.copies.is_empty() {
            let copies: Vec<(u32, u32)> = step.copies.iter().map(|c| (c.src, c.dst)).collect();
            self.cache.copy_blocks(&copies)?;
        }
        let started = Instant::now();
        metrics::histogram!(m::STEP_TOKENS).record(step.num_tokens() as f64);
        let tokens: Vec<&[u32]> = step
            .seqs
            .iter()
            .map(|q| &self.seqs[&q.id].tokens[q.start..q.end])
            .collect();
        let paged: Vec<PagedSeq> = step
            .seqs
            .iter()
            .map(|q| PagedSeq {
                block_table: self.sched.block_table(q.id),
                start_pos: q.start,
                len: q.end - q.start,
            })
            .collect();
        let mut batch = PagedBatch::new(&self.cache, &paged)?;
        let gpu_started = Instant::now();
        let logits = self.model.forward_batch(&tokens, &mut batch)?;
        drop((tokens, paged, batch));

        // Rows that produce a token this step (not mid-prefill). Greedy and temperature-only
        // rows are picked on the device in one kernel; only rows with top-k, top-p or a
        // repetition penalty bring their logits back to the host. A sequence that just finished
        // its prompt forks here, and each child samples from a copy of the parent's row.
        let mut sampled = Vec::new();
        let mut device_rows = vec![RowSampling::Greedy; step.seqs.len()];
        let mut source_rows: Vec<u32> = (0..step.seqs.len() as u32).collect();
        for (row, q) in step.seqs.iter().enumerate() {
            self.sched.advance(q.id, q.end);
            if !self.sched.is_caught_up(q.id) {
                continue;
            }
            let children = match self.sched.pending_forks(q.id) {
                0 => Vec::new(),
                _ => self.fork_children(q.id),
            };
            for (i, id) in std::iter::once(q.id).chain(children).enumerate() {
                let sample_row = match i {
                    0 => row,
                    _ => {
                        source_rows.push(row as u32);
                        device_rows.push(RowSampling::Greedy);
                        source_rows.len() - 1
                    }
                };
                let seq = &self.seqs[&id];
                let mode = seq.sampler.device_mode(seq.tokens.len());
                if let Some(m) = mode {
                    device_rows[sample_row] = m;
                }
                sampled.push((sample_row, id, mode.is_some()));
            }
        }
        let logits = match source_rows.len() > step.seqs.len() {
            true => logits.index_select(&Tensor::new(source_rows, logits.device())?, 0)?,
            false => logits,
        };
        // Reading results back waits for the GPU, so `forward_us` covers the whole pass.
        let device_tokens = match sampled.iter().any(|s| s.2) {
            true => sample_rows(&logits, &device_rows)?,
            false => Vec::new(),
        };
        let host_rows: Vec<u32> = sampled
            .iter()
            .filter(|s| !s.2)
            .map(|s| s.0 as u32)
            .collect();
        let mut host_logits = match host_rows.is_empty() {
            true => Vec::new(),
            false => {
                let idx = Tensor::new(host_rows, logits.device())?;
                logits.index_select(&idx, 0)?.to_vec2::<f32>()?
            }
        }
        .into_iter();
        let sample_started = Instant::now();

        for (row, id, on_device) in sampled {
            let seq = self.seqs.get_mut(&id).expect("scheduled sequence exists");
            let token = match on_device {
                true => device_tokens[row],
                false => {
                    let mut logits = host_logits.next().expect("one row per host sampler");
                    seq.sampler.sample(&mut logits, &[&seq.tokens])
                }
            };
            seq.accept(token, &self.tokenizer)?;
            if seq.finished {
                self.sched.remove(id);
                self.seqs.remove(&id);
            } else {
                self.sched.append_token(id, token);
            }
        }
        tracing::debug!(
            seqs = step.seqs.len(),
            tokens = step.num_tokens(),
            prep_us = (gpu_started - started).as_micros() as u64,
            forward_us = (sample_started - gpu_started).as_micros() as u64,
            sample_us = sample_started.elapsed().as_micros() as u64,
            "step"
        );
        Ok(())
    }
}

/// Default KV pool size in bytes for `gb` gigabytes.
pub fn gib(gb: f64) -> usize {
    (gb * (1u64 << 30) as f64) as usize
}
