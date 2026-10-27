//! Iteration-level scheduling over a paged KV pool.
//!
//! Every step the scheduler decides which sequences run and how many of their tokens to
//! compute. A sequence is just `num_tokens` (prompt plus generated so far) and `num_computed`
//! (tokens whose K/V are cached); each step computes the next chunk of the gap. That one rule
//! covers prefill (gap = prompt), chunked prefill (gap larger than the token budget), decode
//! (gap = 1) and recompute after preemption (cache dropped, gap = everything).
//!
//! Policy:
//! - Running sequences go first, oldest first, within `max_batched_tokens`.
//! - When a running sequence needs a block and none is free, the newest running sequence is
//!   preempted: its blocks are freed and it will recompute from scratch later.
//! - Admission is first-come, first-served: preempted sequences (oldest first) before new
//!   ones, and only while the whole known sequence fits with `watermark_blocks` to spare for
//!   decode growth. The head of the line is never skipped, so nothing starves.
//! - `Policy::Static` admits only when nothing is running (the static-batching baseline).

use std::collections::VecDeque;

use kvcache::{BlockCopy, BlockId, BlockManager, OutOfBlocks, SeqId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    Continuous,
    Static,
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_num_seqs: usize,
    pub max_batched_tokens: usize,
    pub num_blocks: usize,
    pub block_size: usize,
    /// Free blocks kept in reserve when admitting, so running sequences can grow.
    pub watermark_blocks: usize,
    pub policy: Policy,
}

impl SchedulerConfig {
    /// Longest sequence (prompt plus output) that can ever be admitted.
    pub fn max_seq_tokens(&self) -> usize {
        self.num_blocks.saturating_sub(self.watermark_blocks) * self.block_size
    }
}

#[derive(Debug, Clone)]
struct SeqState {
    id: SeqId,
    arrival: u64,
    num_tokens: usize,
    num_computed: usize,
}

/// Compute tokens `start..end` of sequence `id` this step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledSeq {
    pub id: SeqId,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Default)]
pub struct Schedule {
    pub seqs: Vec<ScheduledSeq>,
    /// Block copies (copy-on-write) to perform before the forward pass.
    pub copies: Vec<BlockCopy>,
    /// Sequences preempted this step; their caches are gone.
    pub preempted: Vec<SeqId>,
}

impl Schedule {
    pub fn num_tokens(&self) -> usize {
        self.seqs.iter().map(|s| s.end - s.start).sum()
    }
}

pub struct Scheduler {
    cfg: SchedulerConfig,
    blocks: BlockManager,
    waiting: VecDeque<SeqState>,
    /// Sorted by arrival.
    preempted: VecDeque<SeqState>,
    /// Sorted by arrival; the last entry is the preemption victim.
    running: Vec<SeqState>,
    next_arrival: u64,
    num_preemptions: u64,
}

impl Scheduler {
    pub fn new(cfg: SchedulerConfig) -> Self {
        assert!(
            cfg.max_batched_tokens >= cfg.max_num_seqs,
            "max_batched_tokens must cover one decode token per sequence"
        );
        Self {
            blocks: BlockManager::new(cfg.num_blocks, cfg.block_size),
            cfg,
            waiting: VecDeque::new(),
            preempted: VecDeque::new(),
            running: Vec::new(),
            next_arrival: 0,
            num_preemptions: 0,
        }
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.cfg
    }

    /// Queues a new sequence with `num_tokens` prompt tokens.
    pub fn add(&mut self, id: SeqId, num_tokens: usize) {
        assert!(num_tokens > 0, "empty sequence");
        self.waiting.push_back(SeqState {
            id,
            arrival: self.next_arrival,
            num_tokens,
            num_computed: 0,
        });
        self.next_arrival += 1;
    }

    pub fn schedule(&mut self) -> Schedule {
        let mut out = Schedule::default();
        let mut budget = self.cfg.max_batched_tokens;
        self.schedule_running(&mut out, &mut budget);
        let may_admit = match self.cfg.policy {
            Policy::Continuous => out.preempted.is_empty(),
            Policy::Static => self.running.is_empty(),
        };
        if may_admit {
            self.admit(&mut out, &mut budget);
        }
        out
    }

    fn schedule_running(&mut self, out: &mut Schedule, budget: &mut usize) {
        let mut i = 0;
        while i < self.running.len() && *budget > 0 {
            let s = &self.running[i];
            let want = (s.num_tokens - s.num_computed).min(*budget);
            if want == 0 {
                i += 1;
                continue;
            }
            match self.blocks.reserve(s.id, s.num_computed + want) {
                Ok(copies) => {
                    out.copies.extend(copies);
                    out.seqs.push(ScheduledSeq {
                        id: s.id,
                        start: s.num_computed,
                        end: s.num_computed + want,
                    });
                    *budget -= want;
                    i += 1;
                }
                Err(OutOfBlocks) => {
                    // Free the newest sequence; if that is this one, it waits its turn.
                    let victim = self.running.pop().expect("running is non-empty");
                    self.preempt(victim, out);
                }
            }
        }
    }

    fn preempt(&mut self, mut s: SeqState, out: &mut Schedule) {
        self.blocks.free(s.id);
        s.num_computed = 0;
        out.preempted.push(s.id);
        self.num_preemptions += 1;
        let at = self.preempted.partition_point(|p| p.arrival < s.arrival);
        self.preempted.insert(at, s);
    }

    fn admit(&mut self, out: &mut Schedule, budget: &mut usize) {
        let ignore_budget = self.cfg.policy == Policy::Static;
        while self.running.len() < self.cfg.max_num_seqs && (ignore_budget || *budget > 0) {
            let queue = if self.preempted.is_empty() {
                &mut self.waiting
            } else {
                &mut self.preempted
            };
            let Some(head) = queue.front() else { break };
            let need = self.blocks.blocks_for(head.num_tokens);
            if need + self.cfg.watermark_blocks > self.blocks.num_free_blocks() {
                break;
            }
            let s = queue.pop_front().expect("checked front");
            // Reserve the whole known sequence now, so later admissions can't overcommit.
            out.copies.extend(
                self.blocks
                    .reserve(s.id, s.num_tokens)
                    .expect("checked free blocks"),
            );
            let want = (s.num_tokens - s.num_computed).min(*budget);
            if want > 0 {
                out.seqs.push(ScheduledSeq {
                    id: s.id,
                    start: s.num_computed,
                    end: s.num_computed + want,
                });
                *budget -= want;
            }
            let at = self.running.partition_point(|r| r.arrival < s.arrival);
            self.running.insert(at, s);
        }
    }

    /// The executor computed tokens up to `end` for `id`.
    pub fn advance(&mut self, id: SeqId, end: usize) {
        if let Some(s) = self.running.iter_mut().find(|s| s.id == id) {
            debug_assert!(end <= s.num_tokens);
            s.num_computed = end;
        }
    }

    /// A token was sampled for `id`; its K/V get computed next step.
    pub fn append_token(&mut self, id: SeqId) {
        if let Some(s) = self.running.iter_mut().find(|s| s.id == id) {
            s.num_tokens += 1;
        }
    }

    /// Whether `id` has every token computed, i.e. its last step produced the next token.
    pub fn is_caught_up(&self, id: SeqId) -> bool {
        self.running
            .iter()
            .any(|s| s.id == id && s.num_computed == s.num_tokens)
    }

    /// Removes `id` wherever it is (finished or cancelled) and frees its blocks.
    pub fn remove(&mut self, id: SeqId) {
        self.running.retain(|s| s.id != id);
        self.waiting.retain(|s| s.id != id);
        self.preempted.retain(|s| s.id != id);
        self.blocks.free(id);
    }

    pub fn block_table(&self, id: SeqId) -> &[BlockId] {
        self.blocks.table(id).map_or(&[], |t| t.blocks.as_slice())
    }

    pub fn has_work(&self) -> bool {
        !(self.running.is_empty() && self.waiting.is_empty() && self.preempted.is_empty())
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting.len() + self.preempted.len()
    }

    pub fn num_preemptions(&self) -> u64 {
        self.num_preemptions
    }

    pub fn kv_utilization(&self) -> f64 {
        self.blocks.utilization()
    }

    pub fn num_free_blocks(&self) -> usize {
        self.blocks.num_free_blocks()
    }

    #[cfg(test)]
    fn check(&self) {
        self.blocks.check_invariants().unwrap();
        for s in &self.running {
            let held = self.blocks.table(s.id).map_or(0, |t| t.num_tokens);
            assert!(
                held >= s.num_computed,
                "seq {} computed {} but holds {held} slots",
                s.id,
                s.num_computed
            );
        }
        for s in self.waiting.iter().chain(&self.preempted) {
            assert!(
                !self.blocks.contains(s.id),
                "queued seq {} holds blocks",
                s.id
            );
        }
        assert!(self.running.windows(2).all(|w| w[0].arrival < w[1].arrival));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn cfg(num_blocks: usize, policy: Policy) -> SchedulerConfig {
        SchedulerConfig {
            max_num_seqs: 8,
            max_batched_tokens: 64,
            num_blocks,
            block_size: 4,
            watermark_blocks: 1,
            policy,
        }
    }

    /// A fake executor: runs whatever is scheduled, samples one token whenever a sequence is
    /// caught up, and finishes it after `outputs[id]` tokens. Returns finish order and the
    /// largest batch seen.
    fn simulate(
        sched: &mut Scheduler,
        prompts: &[usize],
        outputs: &[usize],
    ) -> (Vec<SeqId>, usize) {
        let mut produced: HashMap<SeqId, usize> = HashMap::new();
        for (i, &p) in prompts.iter().enumerate() {
            sched.add(i as SeqId, p);
        }
        let mut finished = Vec::new();
        let mut max_batch = 0;
        for _ in 0..100_000 {
            if !sched.has_work() {
                return (finished, max_batch);
            }
            let step = sched.schedule();
            assert!(!step.seqs.is_empty(), "no progress with work pending");
            assert!(step.seqs.len() <= sched.cfg.max_num_seqs);
            assert!(step.num_tokens() <= sched.cfg.max_batched_tokens);
            max_batch = max_batch.max(step.seqs.len());
            for s in &step.seqs {
                assert!(
                    sched.block_table(s.id).len() * 4 >= s.end,
                    "scheduled without slots"
                );
                sched.advance(s.id, s.end);
                if sched.is_caught_up(s.id) {
                    let n = produced.entry(s.id).or_default();
                    *n += 1;
                    if *n == outputs[s.id as usize] {
                        sched.remove(s.id);
                        finished.push(s.id);
                    } else {
                        sched.append_token(s.id);
                    }
                }
            }
            sched.check();
        }
        panic!("simulation did not finish");
    }

    #[test]
    fn everything_finishes_and_blocks_return() {
        let mut s = Scheduler::new(cfg(64, Policy::Continuous));
        let prompts: Vec<usize> = (0..20).map(|i| 1 + (i * 7) % 30).collect();
        let outputs: Vec<usize> = (0..20).map(|i| 1 + (i * 5) % 25).collect();
        let (finished, max_batch) = simulate(&mut s, &prompts, &outputs);
        assert_eq!(finished.len(), 20);
        assert_eq!(max_batch, 8, "should fill up to max_num_seqs");
        assert_eq!(s.num_free_blocks(), 64);
    }

    #[test]
    fn memory_pressure_preempts_and_still_finishes() {
        // 12 blocks of 4 = 48 slots for 8 sequences that each grow to 4 + 20 tokens.
        let mut s = Scheduler::new(cfg(12, Policy::Continuous));
        let (finished, _) = simulate(&mut s, &[4; 8], &[20; 8]);
        assert_eq!(finished.len(), 8);
        assert!(
            s.num_preemptions() > 0,
            "expected preemption under memory pressure"
        );
        assert_eq!(s.num_free_blocks(), 12);
    }

    #[test]
    fn newest_running_sequence_is_the_victim() {
        let mut s = Scheduler::new(SchedulerConfig {
            watermark_blocks: 0,
            ..cfg(3, Policy::Continuous)
        });
        s.add(0, 4);
        s.add(1, 4);
        s.add(2, 4);
        let step = s.schedule();
        assert_eq!(step.seqs.len(), 3, "all three prompts fit exactly");
        for q in &step.seqs {
            s.advance(q.id, q.end);
            s.append_token(q.id);
        }
        // Each needs a 2nd block for its 5th token; none are free, so 2 then 1 are evicted.
        let step = s.schedule();
        assert_eq!(step.preempted, [2, 1]);
        assert_eq!(
            step.seqs,
            [ScheduledSeq {
                id: 0,
                start: 4,
                end: 5
            }]
        );
    }

    #[test]
    fn preempted_sequences_are_readmitted_before_new_ones() {
        let mut s = Scheduler::new(SchedulerConfig {
            watermark_blocks: 0,
            ..cfg(3, Policy::Continuous)
        });
        for id in 0..3 {
            s.add(id, 4);
        }
        for q in s.schedule().seqs {
            s.advance(q.id, q.end);
            s.append_token(q.id);
        }
        assert_eq!(s.schedule().preempted, [2, 1]);
        s.add(9, 1);
        s.remove(0);
        let step = s.schedule();
        let ids: Vec<_> = step.seqs.iter().map(|q| q.id).collect();
        assert_eq!(
            ids,
            [1],
            "oldest preempted first, and the new one waits behind seq 2"
        );
        // Recompute covers the generated token too.
        assert_eq!(
            step.seqs[0],
            ScheduledSeq {
                id: 1,
                start: 0,
                end: 5
            }
        );
    }

    #[test]
    fn long_prompts_are_chunked_by_the_token_budget() {
        let mut s = Scheduler::new(SchedulerConfig {
            max_batched_tokens: 16,
            ..cfg(64, Policy::Continuous)
        });
        s.add(0, 40);
        let chunks: Vec<_> = (0..3)
            .map(|_| {
                let q = s.schedule().seqs[0];
                s.advance(q.id, q.end);
                (q.start, q.end)
            })
            .collect();
        assert_eq!(chunks, [(0, 16), (16, 32), (32, 40)]);
        assert!(s.is_caught_up(0));
    }

    #[test]
    fn decodes_are_not_starved_by_a_new_long_prompt() {
        let mut s = Scheduler::new(SchedulerConfig {
            max_batched_tokens: 16,
            ..cfg(64, Policy::Continuous)
        });
        s.add(0, 4);
        let q = s.schedule().seqs[0];
        s.advance(q.id, q.end);
        s.append_token(0);
        s.add(1, 100);
        let step = s.schedule();
        assert_eq!(
            step.seqs[0],
            ScheduledSeq {
                id: 0,
                start: 4,
                end: 5
            },
            "decode first"
        );
        assert_eq!(
            step.seqs[1],
            ScheduledSeq {
                id: 1,
                start: 0,
                end: 15
            },
            "prefill gets the rest"
        );
    }

    #[test]
    fn static_policy_waits_for_the_batch_to_drain() {
        let mut s = Scheduler::new(cfg(64, Policy::Static));
        s.add(0, 4);
        let q = s.schedule().seqs[0];
        s.advance(q.id, q.end);
        s.append_token(0);
        s.add(1, 4);
        let ids: Vec<_> = s.schedule().seqs.iter().map(|q| q.id).collect();
        assert_eq!(ids, [0], "no admission while a batch runs");
        s.remove(0);
        let ids: Vec<_> = s.schedule().seqs.iter().map(|q| q.id).collect();
        assert_eq!(ids, [1]);
    }

    #[test]
    fn static_policy_also_finishes_everything() {
        let mut s = Scheduler::new(cfg(64, Policy::Static));
        let (finished, _) = simulate(
            &mut s,
            &[3, 9, 1, 17, 5, 2, 8, 30, 4, 6],
            &[5, 1, 9, 3, 7, 2, 4, 6, 8, 10],
        );
        assert_eq!(finished.len(), 10);
    }

    #[test]
    fn random_workloads_finish_without_starvation() {
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut rand = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n) as usize
        };
        for trial in 0..200 {
            let n = 1 + rand(30);
            let blocks = 8 + rand(40);
            let policy = if trial % 4 == 0 {
                Policy::Static
            } else {
                Policy::Continuous
            };
            let mut s = Scheduler::new(SchedulerConfig {
                max_batched_tokens: 8 + rand(64),
                ..cfg(blocks, policy)
            });
            let cap = s.cfg.max_seq_tokens();
            let mut prompts = Vec::new();
            let mut outputs = Vec::new();
            for _ in 0..n {
                let p = 1 + rand(cap as u64 / 2);
                prompts.push(p);
                outputs.push(1 + rand((cap - p) as u64));
            }
            let (finished, _) = simulate(&mut s, &prompts, &outputs);
            assert_eq!(finished.len(), n, "trial {trial}");
            assert_eq!(s.num_free_blocks(), blocks, "trial {trial} leaked blocks");
        }
    }
}
