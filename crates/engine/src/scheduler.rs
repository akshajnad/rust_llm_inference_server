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
//! - With `prefix_caching`, every block is registered once all its tokens are computed, and an
//!   admitted sequence starts from the longest cached chain of its leading blocks. That also
//!   makes recompute after preemption cheap: the victim's blocks stay cached until reclaimed.
//! - A sequence added with `forks` children is admitted only if it has seats and blocks for
//!   them too. Once its prompt is computed, `fork` gives each child the parent's blocks: the
//!   prompt is prefilled once, and only the partially filled last block is copied (on write).

use std::collections::VecDeque;

use kvcache::{
    BlockCopy, BlockId, BlockManager, OutOfBlocks, SeqId, block_hashes, next_block_hash,
};

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
    pub prefix_caching: bool,
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
    /// Prompt plus generated tokens; `tokens.len()` is the sequence length.
    tokens: Vec<u32>,
    num_computed: usize,
    /// Leading blocks registered in the prefix cache, and the hash chain through them.
    registered: usize,
    chain: u64,
    /// Children still to be forked off once the prompt is computed.
    forks: usize,
}

impl SeqState {
    fn num_tokens(&self) -> usize {
        self.tokens.len()
    }
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
    admitted_tokens: u64,
    cached_tokens: u64,
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
            admitted_tokens: 0,
            cached_tokens: 0,
        }
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.cfg
    }

    /// Queues a new sequence with its prompt tokens.
    pub fn add(&mut self, id: SeqId, prompt: Vec<u32>) {
        self.add_forked(id, prompt, 0);
    }

    /// Queues a sequence that will `fork` into `forks` more once its prompt is computed.
    pub fn add_forked(&mut self, id: SeqId, prompt: Vec<u32>, forks: usize) {
        assert!(!prompt.is_empty(), "empty sequence");
        assert!(
            forks < self.cfg.max_num_seqs,
            "{} sequences can never run at once",
            forks + 1
        );
        self.waiting.push_back(SeqState {
            id,
            arrival: self.next_arrival,
            tokens: prompt,
            num_computed: 0,
            registered: 0,
            chain: 0,
            forks,
        });
        self.next_arrival += 1;
    }

    /// Children `id` has yet to fork.
    pub fn pending_forks(&self, id: SeqId) -> usize {
        self.running
            .iter()
            .find(|s| s.id == id)
            .map_or(0, |s| s.forks)
    }

    /// Starts `child` as a copy of `parent`, sharing its KV blocks. The parent must be running
    /// with every token computed and a fork pending; the child runs from the next step.
    pub fn fork(&mut self, parent: SeqId, child: SeqId) {
        let p = self
            .running
            .iter_mut()
            .find(|s| s.id == parent)
            .expect("fork of a sequence that isn't running");
        assert!(
            p.forks > 0 && p.num_computed == p.num_tokens(),
            "fork of seq {parent} before its prompt is computed or past its fork count"
        );
        p.forks -= 1;
        let s = SeqState {
            id: child,
            arrival: self.next_arrival,
            forks: 0,
            ..p.clone()
        };
        self.next_arrival += 1;
        self.blocks.fork(parent, child);
        // Newest arrival, so it stays sorted and is the first preemption victim.
        self.running.push(s);
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
            let want = (s.num_tokens() - s.num_computed).min(*budget);
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
        // Registered blocks stay cached (evictable) and are likely adopted again on readmission.
        self.blocks.free(s.id);
        s.num_computed = 0;
        s.registered = 0;
        s.chain = 0;
        out.preempted.push(s.id);
        self.num_preemptions += 1;
        let at = self.preempted.partition_point(|p| p.arrival < s.arrival);
        self.preempted.insert(at, s);
    }

    /// Running sequences plus the children they have yet to fork.
    fn seats(&self) -> usize {
        self.running.iter().map(|s| 1 + s.forks).sum()
    }

    fn admit(&mut self, out: &mut Schedule, budget: &mut usize) {
        let ignore_budget = self.cfg.policy == Policy::Static;
        while self.seats() < self.cfg.max_num_seqs && (ignore_budget || *budget > 0) {
            let seats = self.seats();
            let queue = if self.preempted.is_empty() {
                &mut self.waiting
            } else {
                &mut self.preempted
            };
            let Some(head) = queue.front() else { break };
            // Its children will need seats, and one block each to copy the shared last block.
            if seats + 1 + head.forks > self.cfg.max_num_seqs {
                break;
            }
            // Cached leading blocks, never the one holding the last token: its logits are
            // needed, so at least one token is always computed.
            let hashes = match self.cfg.prefix_caching {
                true => block_hashes(&head.tokens[..head.num_tokens() - 1], self.cfg.block_size),
                false => Vec::new(),
            };
            let (matched, matched_evictable) = self.blocks.prefix_match(&hashes);
            let need = self.blocks.blocks_for(head.num_tokens()) - matched + head.forks;
            let available = self.blocks.num_free_blocks() - matched_evictable;
            if need + self.cfg.watermark_blocks > available {
                break;
            }
            let mut s = queue.pop_front().expect("checked front");
            if matched > 0 {
                self.blocks.adopt_prefix(s.id, &hashes[..matched]);
                s.num_computed = matched * self.cfg.block_size;
                s.registered = matched;
                s.chain = hashes[matched - 1];
            }
            self.admitted_tokens += s.num_tokens() as u64;
            self.cached_tokens += s.num_computed as u64;
            // Reserve the whole known sequence now, so later admissions can't overcommit.
            out.copies.extend(
                self.blocks
                    .reserve(s.id, s.num_tokens())
                    .expect("checked free blocks"),
            );
            let want = (s.num_tokens() - s.num_computed).min(*budget);
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

    /// The executor computed tokens up to `end` for `id`; newly completed blocks enter the
    /// prefix cache.
    pub fn advance(&mut self, id: SeqId, end: usize) {
        let bs = self.cfg.block_size;
        let Some(s) = self.running.iter_mut().find(|s| s.id == id) else {
            return;
        };
        debug_assert!(end <= s.num_tokens());
        s.num_computed = end;
        if !self.cfg.prefix_caching {
            return;
        }
        while (s.registered + 1) * bs <= end {
            let i = s.registered;
            s.chain = next_block_hash(s.chain, &s.tokens[i * bs..(i + 1) * bs]);
            self.blocks.register_block(id, i, s.chain);
            s.registered += 1;
        }
    }

    /// `token` was sampled for `id`; its K/V get computed next step.
    pub fn append_token(&mut self, id: SeqId, token: u32) {
        if let Some(s) = self.running.iter_mut().find(|s| s.id == id) {
            s.tokens.push(token);
        }
    }

    /// Admitted tokens so far, and how many of them came from the prefix cache.
    pub fn prefix_cache_stats(&self) -> (u64, u64) {
        (self.admitted_tokens, self.cached_tokens)
    }

    /// Whether `id` has every token computed, i.e. its last step produced the next token.
    pub fn is_caught_up(&self, id: SeqId) -> bool {
        self.running
            .iter()
            .any(|s| s.id == id && s.num_computed == s.num_tokens())
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

    /// `n` prompt tokens unique to sequence `id`.
    fn toks(id: SeqId, n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| id as u32 * 100_000 + i).collect()
    }

    fn cfg(num_blocks: usize, policy: Policy) -> SchedulerConfig {
        SchedulerConfig {
            max_num_seqs: 8,
            max_batched_tokens: 64,
            num_blocks,
            block_size: 4,
            watermark_blocks: 1,
            policy,
            prefix_caching: false,
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
        let prompts: Vec<Vec<u32>> = prompts
            .iter()
            .enumerate()
            .map(|(i, &p)| toks(i as SeqId, p))
            .collect();
        simulate_prompts(sched, prompts, outputs)
    }

    fn simulate_prompts(
        sched: &mut Scheduler,
        prompts: Vec<Vec<u32>>,
        outputs: &[usize],
    ) -> (Vec<SeqId>, usize) {
        let mut produced: HashMap<SeqId, usize> = HashMap::new();
        for (i, p) in prompts.into_iter().enumerate() {
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
                        sched.append_token(s.id, 7);
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
        s.add(0, toks(0, 4));
        s.add(1, toks(1, 4));
        s.add(2, toks(2, 4));
        let step = s.schedule();
        assert_eq!(step.seqs.len(), 3, "all three prompts fit exactly");
        for q in &step.seqs {
            s.advance(q.id, q.end);
            s.append_token(q.id, 7);
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
            s.add(id, toks(id, 4));
        }
        for q in s.schedule().seqs {
            s.advance(q.id, q.end);
            s.append_token(q.id, 7);
        }
        assert_eq!(s.schedule().preempted, [2, 1]);
        s.add(9, toks(9, 1));
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
        s.add(0, toks(0, 40));
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
        s.add(0, toks(0, 4));
        let q = s.schedule().seqs[0];
        s.advance(q.id, q.end);
        s.append_token(0, 7);
        s.add(1, toks(1, 100));
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
        s.add(0, toks(0, 4));
        let q = s.schedule().seqs[0];
        s.advance(q.id, q.end);
        s.append_token(0, 7);
        s.add(1, toks(1, 4));
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

    #[test]
    fn shared_prefix_is_adopted_not_recomputed() {
        let mut s = Scheduler::new(SchedulerConfig {
            prefix_caching: true,
            ..cfg(64, Policy::Continuous)
        });
        let prompt: Vec<u32> = (0..40).collect();
        s.add(0, prompt.clone());
        let q = s.schedule().seqs[0];
        assert_eq!((q.start, q.end), (0, 40));
        s.advance(0, 40);
        s.remove(0);
        // Same 40-token prompt: blocks 0..9 (36 tokens) come from the cache; the block with
        // the last token is computed so its logits exist.
        s.add(1, prompt.clone());
        let q = s.schedule().seqs[0];
        assert_eq!((q.start, q.end), (36, 40));
        // A different suffix after a shared 20-token prefix reuses the 5 full shared blocks.
        let mut other = prompt[..20].to_vec();
        other.extend(900..915);
        s.add(2, other);
        let q = s.schedule().seqs.into_iter().find(|q| q.id == 2).unwrap();
        assert_eq!((q.start, q.end), (20, 35));
        assert_eq!(s.prefix_cache_stats(), (40 + 40 + 35, 36 + 20));
        s.check();
    }

    #[test]
    fn cached_blocks_make_recompute_after_preemption_cheap() {
        let mut s = Scheduler::new(SchedulerConfig {
            prefix_caching: true,
            watermark_blocks: 0,
            ..cfg(3, Policy::Continuous)
        });
        for id in 0..3 {
            s.add(id, toks(id, 4));
        }
        for q in s.schedule().seqs {
            s.advance(q.id, q.end);
            s.append_token(q.id, 7);
        }
        assert_eq!(s.schedule().preempted, [2, 1]);
        s.remove(0);
        // Seq 1's first block (its full 4-token prompt) is still cached: only the sampled
        // token needs recomputing.
        let q = s.schedule().seqs[0];
        assert_eq!((q.id, q.start, q.end), (1, 4, 5));
    }

    #[test]
    fn random_workloads_with_prefix_caching_finish() {
        for blocks in [12usize, 24, 48] {
            let mut s = Scheduler::new(SchedulerConfig {
                prefix_caching: true,
                ..cfg(blocks, Policy::Continuous)
            });
            // Half the prompts share a 16-token prefix, under memory pressure at 12 blocks.
            let shared: Vec<u32> = (0..16).collect();
            let prompts: Vec<Vec<u32>> = (0..8u32)
                .map(|i| match i % 2 {
                    0 => shared
                        .iter()
                        .copied()
                        .chain(1000 * i..1000 * i + 1 + i)
                        .collect(),
                    _ => toks(i as SeqId, 9 + i as usize),
                })
                .collect();
            let (finished, _) = simulate_prompts(&mut s, prompts, &[9, 4, 12, 3, 8, 6, 2, 11]);
            assert_eq!(finished.len(), 8);
            // Sequences admitted in the same step can't share blocks nobody has computed yet;
            // only ones admitted after the shared prefix was computed hit the cache.
            let (_, cached) = s.prefix_cache_stats();
            assert!(
                cached >= 16,
                "a later shared-prefix prompt should hit the cache, got {cached}"
            );
            assert_eq!(
                s.num_free_blocks(),
                blocks,
                "every block is free or reclaimable"
            );
        }
    }

    #[test]
    fn forks_share_the_prompt_and_copy_only_its_last_block() {
        let mut s = Scheduler::new(cfg(64, Policy::Continuous));
        s.add_forked(0, toks(0, 10), 3);
        let q = s.schedule().seqs[0];
        assert_eq!((q.start, q.end), (0, 10));
        s.advance(0, 10);
        assert_eq!(s.pending_forks(0), 3);
        for child in 1..4 {
            s.fork(0, child);
        }
        assert_eq!(s.pending_forks(0), 0);
        assert_eq!(s.num_free_blocks(), 64 - 3, "children allocate nothing");
        for id in 0..4 {
            assert_eq!(s.block_table(id), s.block_table(0));
            s.append_token(id, 50 + id as u32);
        }
        // Token 10 lands in the shared, half-full third block: the first three writers copy
        // it, and the last one finds it unshared.
        let step = s.schedule();
        assert_eq!(step.seqs.len(), 4);
        assert_eq!(step.copies.len(), 3);
        assert_eq!(
            s.num_free_blocks(),
            64 - 6,
            "vs 12 blocks for 4 separate prompts"
        );
        s.check();
        for id in 0..4 {
            s.remove(id);
        }
        assert_eq!(s.num_free_blocks(), 64);
    }

    #[test]
    fn admission_holds_seats_for_pending_forks() {
        let mut s = Scheduler::new(SchedulerConfig {
            max_num_seqs: 4,
            ..cfg(64, Policy::Continuous)
        });
        s.add_forked(0, toks(0, 4), 2);
        s.add(1, toks(1, 4));
        s.add(2, toks(2, 4));
        let ids: Vec<_> = s.schedule().seqs.iter().map(|q| q.id).collect();
        assert_eq!(ids, [0, 1], "seq 2 would leave no seat for a child");
    }

    /// One sequence of a fork workload: a prompt (optionally sharing a prefix with others),
    /// children to fork, and per-member output lengths and cancellation steps (index 0 is the
    /// parent).
    #[derive(Debug, Clone)]
    struct ForkCase {
        prompt: usize,
        shared: bool,
        outputs: Vec<usize>,
        cancel_at: Vec<Option<usize>>,
    }

    fn fork_case() -> impl proptest::strategy::Strategy<Value = ForkCase> {
        use proptest::prelude::*;
        (1..30usize, any::<bool>(), 0..4usize).prop_flat_map(|(prompt, shared, forks)| {
            (
                proptest::collection::vec(1..15usize, forks + 1),
                proptest::collection::vec(proptest::option::weighted(0.2, 0..40usize), forks + 1),
            )
                .prop_map(move |(outputs, cancel_at)| ForkCase {
                    prompt,
                    shared,
                    outputs,
                    cancel_at,
                })
        })
    }

    /// Runs `cases` against a fake KV pool that stores token ids. Each step: perform the
    /// copies, check every block about to be written has refcount 1, write the tokens, then
    /// check every running sequence's computed slots hold exactly its tokens. A parent forks
    /// its children when its prompt is done; members finish after their output count or are
    /// removed at their cancellation step.
    fn run_forks(sched: &mut Scheduler, cases: &[ForkCase]) {
        let bs = sched.cfg.block_size;
        let shared: Vec<u32> = (900_000..900_016).collect();
        // Member `m` of case `i`: id i * 10 + m, so ids stay unique and decode to (case, member).
        let member = |id: SeqId| ((id / 10) as usize, (id % 10) as usize);
        for (i, c) in cases.iter().enumerate() {
            let mut prompt = toks(i as SeqId, c.prompt);
            if c.shared {
                prompt.splice(0..0, shared.iter().copied());
            }
            sched.add_forked(i as SeqId * 10, prompt, c.outputs.len() - 1);
        }
        let mut pool: Vec<Option<u32>> = vec![None; sched.cfg.num_blocks * bs];
        let mut produced: HashMap<SeqId, usize> = HashMap::new();
        for step_no in 0..100_000 {
            // Cancellations, of members that exist (a cancelled parent takes its unforked
            // children with it).
            let ids: Vec<SeqId> = (0..cases.len())
                .flat_map(|i| (0..cases[i].outputs.len()).map(move |m| (i * 10 + m) as SeqId))
                .collect();
            for id in ids {
                let (i, m) = member(id);
                if cases[i].cancel_at[m] == Some(step_no) {
                    sched.remove(id);
                    sched.check();
                }
            }
            if !sched.has_work() {
                assert_eq!(
                    sched.num_free_blocks(),
                    sched.cfg.num_blocks,
                    "every block is free or reclaimable"
                );
                return;
            }
            let step = sched.schedule();
            assert!(!step.seqs.is_empty(), "no progress with work pending");
            assert!(step.seqs.len() <= sched.cfg.max_num_seqs);
            for c in &step.copies {
                let (src, dst) = (c.src as usize * bs, c.dst as usize * bs);
                pool.copy_within(src..src + bs, dst);
            }
            for q in &step.seqs {
                let s = sched.running.iter().find(|s| s.id == q.id).unwrap();
                let table = sched.block_table(q.id);
                for pos in q.start..q.end {
                    let b = table[pos / bs];
                    assert_eq!(
                        sched.blocks.refcount(b),
                        1,
                        "seq {} writes position {pos} into shared block {b}",
                        q.id
                    );
                    pool[b as usize * bs + pos % bs] = Some(s.tokens[pos]);
                }
            }
            for q in &step.seqs {
                sched.advance(q.id, q.end);
                if !sched.is_caught_up(q.id) {
                    continue;
                }
                let mut members = vec![q.id];
                while sched.pending_forks(q.id) > 0 {
                    let child = q.id
                        + (cases[member(q.id).0].outputs.len() - sched.pending_forks(q.id))
                            as SeqId;
                    sched.fork(q.id, child);
                    members.push(child);
                }
                for id in members {
                    let (i, m) = member(id);
                    let n = produced.entry(id).or_default();
                    *n += 1;
                    if *n == cases[i].outputs[m] {
                        sched.remove(id);
                    } else {
                        // Members sample different tokens, so a write into a shared block
                        // would show up in a sibling's contents.
                        sched.append_token(id, id as u32 * 1000 + *n as u32);
                    }
                }
            }
            sched.check();
            for s in &sched.running {
                let table = sched.block_table(s.id);
                for pos in 0..s.num_computed {
                    assert_eq!(
                        pool[table[pos / bs] as usize * bs + pos % bs],
                        Some(s.tokens[pos]),
                        "seq {} sees the wrong token at {pos}",
                        s.id
                    );
                }
            }
        }
        panic!("simulation did not finish");
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(500))]

        /// Forked workloads under memory pressure, with and without prefix caching, random
        /// chunking and random cancellation of parents and children: no block leaks, refcounts
        /// match holders after every step and removal, copy-on-write never writes a shared
        /// block, and every sequence always reads back exactly its own tokens.
        #[test]
        fn forks_never_leak_or_corrupt(
            cases in proptest::collection::vec(fork_case(), 1..8),
            blocks in 24..64usize,
            max_batched_tokens in 8..64usize,
            prefix_caching: bool,
        ) {
            let mut s = Scheduler::new(SchedulerConfig {
                max_batched_tokens,
                prefix_caching,
                ..cfg(blocks, Policy::Continuous)
            });
            run_forks(&mut s, &cases);
        }
    }
}
