//! Paged KV cache bookkeeping: which physical blocks hold which sequence's tokens.
//!
//! The pool is a fixed number of blocks, each holding `block_size` tokens of K and V for every
//! layer. The tensors live in the model crate; this crate only decides which block ids a
//! sequence owns, so it has no GPU dependency and is cheap to test exhaustively.
//!
//! Blocks are reference counted so sequences can share a prefix (`fork`). A shared block is
//! immutable: writing into a partially filled shared block first copies it (copy-on-write),
//! and the copy is returned for the executor to perform before the next forward pass.
//!
//! Automatic prefix caching: a full block can be registered under the hash of its tokens and
//! of every token before it (`block_hashes`). Registered blocks outlive their sequences: at
//! refcount zero they become *evictable* instead of free, and a later sequence with the same
//! prefix adopts them instead of recomputing. Evictable blocks are reclaimed least recently
//! used first, only once the free list is empty.

use std::collections::{BTreeMap, HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};

pub type BlockId = u32;
pub type SeqId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("out of KV cache blocks")]
pub struct OutOfBlocks;

/// Copy the contents of block `src` into block `dst` (all layers) before running the step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockCopy {
    pub src: BlockId,
    pub dst: BlockId,
}

/// Hashes of each full block of `tokens`, each covering every token up to the block's end.
pub fn block_hashes(tokens: &[u32], block_size: usize) -> Vec<u64> {
    let mut prev = 0u64;
    tokens
        .chunks_exact(block_size)
        .map(|block| {
            prev = next_block_hash(prev, block);
            prev
        })
        .collect()
}

/// Chains one more block onto the hash of everything before it (0 for the first block).
pub fn next_block_hash(prev: u64, block: &[u32]) -> u64 {
    // DefaultHasher::new() uses fixed keys, so hashes are stable within a process.
    let mut h = DefaultHasher::new();
    prev.hash(&mut h);
    block.hash(&mut h);
    h.finish()
}

/// Free list, reference counts and the prefix cache over a fixed pool.
#[derive(Debug, Clone)]
pub struct BlockAllocator {
    refcounts: Vec<u32>,
    /// LIFO so recently freed (likely still cache-warm) blocks are reused first.
    free: Vec<BlockId>,
    /// Content hash of registered blocks.
    hash_of: Vec<Option<u64>>,
    /// Registered blocks by hash.
    cached: HashMap<u64, BlockId>,
    /// Unreferenced registered blocks by last use, oldest first.
    evictable: BTreeMap<u64, BlockId>,
    evict_tick: Vec<Option<u64>>,
    tick: u64,
}

impl BlockAllocator {
    pub fn new(num_blocks: usize) -> Self {
        Self {
            refcounts: vec![0; num_blocks],
            free: (0..num_blocks as BlockId).rev().collect(),
            hash_of: vec![None; num_blocks],
            cached: HashMap::new(),
            evictable: BTreeMap::new(),
            evict_tick: vec![None; num_blocks],
            tick: 0,
        }
    }

    /// Unreferenced registered blocks (reclaimable).
    pub fn num_evictable(&self) -> usize {
        self.evictable.len()
    }

    pub fn lookup(&self, hash: u64) -> Option<BlockId> {
        self.cached.get(&hash).copied()
    }

    /// Records that `b` holds the tokens hashed by `hash`. If another block already holds
    /// them, `b` stays unregistered (and is freed normally).
    pub fn register(&mut self, b: BlockId, hash: u64) {
        if self.hash_of[b as usize].is_none() && !self.cached.contains_key(&hash) {
            self.hash_of[b as usize] = Some(hash);
            self.cached.insert(hash, b);
        }
    }

    /// Takes a reference to a registered block, reviving it if it was evictable.
    pub fn acquire(&mut self, b: BlockId) {
        if let Some(t) = self.evict_tick[b as usize].take() {
            self.evictable.remove(&t);
        }
        self.refcounts[b as usize] += 1;
    }

    pub fn num_blocks(&self) -> usize {
        self.refcounts.len()
    }

    /// Blocks available to allocate: free plus evictable.
    pub fn num_free(&self) -> usize {
        self.free.len() + self.evictable.len()
    }

    pub fn refcount(&self, b: BlockId) -> u32 {
        self.refcounts[b as usize]
    }

    pub fn allocate(&mut self) -> Option<BlockId> {
        let b = match self.free.pop() {
            Some(b) => b,
            None => {
                // Reclaim the least recently used cached block.
                let (_, b) = self.evictable.pop_first()?;
                self.evict_tick[b as usize] = None;
                let h = self.hash_of[b as usize]
                    .take()
                    .expect("evictable blocks are registered");
                self.cached.remove(&h);
                b
            }
        };
        debug_assert_eq!(
            self.refcounts[b as usize], 0,
            "free block {b} had references"
        );
        self.refcounts[b as usize] = 1;
        Some(b)
    }

    pub fn incref(&mut self, b: BlockId) {
        debug_assert!(self.refcounts[b as usize] > 0, "incref on free block {b}");
        self.refcounts[b as usize] += 1;
    }

    /// Drops one reference. At zero a registered block becomes evictable (still cached), any
    /// other block returns to the free list.
    pub fn decref(&mut self, b: BlockId) {
        let rc = &mut self.refcounts[b as usize];
        assert!(*rc > 0, "double free of block {b}");
        *rc -= 1;
        if *rc == 0 {
            if self.hash_of[b as usize].is_some() {
                self.tick += 1;
                self.evictable.insert(self.tick, b);
                self.evict_tick[b as usize] = Some(self.tick);
            } else {
                self.free.push(b);
            }
        }
    }
}

/// A sequence's logical-to-physical mapping: token `i` lives in `blocks[i / block_size]` at
/// offset `i % block_size`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlockTable {
    pub blocks: Vec<BlockId>,
    /// Tokens that have slots reserved.
    pub num_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct BlockManager {
    block_size: usize,
    alloc: BlockAllocator,
    tables: HashMap<SeqId, BlockTable>,
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be positive");
        Self {
            block_size,
            alloc: BlockAllocator::new(num_blocks),
            tables: HashMap::new(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.alloc.num_blocks()
    }

    pub fn num_free_blocks(&self) -> usize {
        self.alloc.num_free()
    }

    /// Fraction of blocks referenced by a sequence, 0.0 to 1.0 (cached-but-unused blocks are
    /// reclaimable, so they don't count).
    pub fn utilization(&self) -> f64 {
        match self.num_blocks() {
            0 => 0.0,
            n => (n - self.num_free_blocks()) as f64 / n as f64,
        }
    }

    /// Blocks held only by the prefix cache.
    pub fn num_cached_blocks(&self) -> usize {
        self.alloc.num_evictable()
    }

    /// How many leading `hashes` are cached, and how many of those are currently evictable
    /// (adopting them takes them out of the available count).
    pub fn prefix_match(&self, hashes: &[u64]) -> (usize, usize) {
        let mut evictable = 0;
        let n = hashes
            .iter()
            .map_while(|&h| self.alloc.lookup(h))
            .inspect(|&b| evictable += usize::from(self.alloc.refcount(b) == 0))
            .count();
        (n, evictable)
    }

    /// Starts `seq` with the cached blocks for the leading `hashes`; returns how many blocks
    /// (each full) it adopted.
    pub fn adopt_prefix(&mut self, seq: SeqId, hashes: &[u64]) -> usize {
        assert!(
            !self.tables.contains_key(&seq),
            "adopt into existing sequence {seq}"
        );
        let blocks: Vec<BlockId> = hashes.iter().map_while(|&h| self.alloc.lookup(h)).collect();
        if blocks.is_empty() {
            return 0;
        }
        for &b in &blocks {
            self.alloc.acquire(b);
        }
        let n = blocks.len();
        self.tables.insert(
            seq,
            BlockTable {
                blocks,
                num_tokens: n * self.block_size,
            },
        );
        self.debug_check();
        n
    }

    /// Registers `seq`'s block `index` (now full and computed) in the prefix cache.
    pub fn register_block(&mut self, seq: SeqId, index: usize, hash: u64) {
        if let Some(&b) = self.tables.get(&seq).and_then(|t| t.blocks.get(index)) {
            self.alloc.register(b, hash);
        }
    }

    pub fn blocks_for(&self, num_tokens: usize) -> usize {
        num_tokens.div_ceil(self.block_size)
    }

    /// Tables holding block `b` (0 for free and evictable blocks).
    pub fn refcount(&self, b: BlockId) -> u32 {
        self.alloc.refcount(b)
    }

    pub fn table(&self, seq: SeqId) -> Option<&BlockTable> {
        self.tables.get(&seq)
    }

    pub fn contains(&self, seq: SeqId) -> bool {
        self.tables.contains_key(&seq)
    }

    /// Blocks `reserve(seq, total)` would take from the free list, including a copy-on-write.
    pub fn blocks_needed(&self, seq: SeqId, total_tokens: usize) -> usize {
        let empty = BlockTable::default();
        let t = self.tables.get(&seq).unwrap_or(&empty);
        if total_tokens <= t.num_tokens {
            return 0;
        }
        let grow = self.blocks_for(total_tokens).saturating_sub(t.blocks.len());
        grow + usize::from(self.needs_cow(t))
    }

    /// The next write lands in a partially filled last block that another table shares.
    fn needs_cow(&self, t: &BlockTable) -> bool {
        !t.num_tokens.is_multiple_of(self.block_size)
            && t.blocks.last().is_some_and(|&b| self.alloc.refcount(b) > 1)
    }

    /// Ensures `seq` has slots for `total_tokens` tokens, creating its table if needed.
    /// All-or-nothing: on `OutOfBlocks` nothing changes.
    pub fn reserve(
        &mut self,
        seq: SeqId,
        total_tokens: usize,
    ) -> Result<Vec<BlockCopy>, OutOfBlocks> {
        if self.blocks_needed(seq, total_tokens) > self.alloc.num_free() {
            return Err(OutOfBlocks);
        }
        let mut copies = Vec::new();
        let cow = self
            .tables
            .get(&seq)
            .is_some_and(|t| total_tokens > t.num_tokens && self.needs_cow(t));
        let target = self.blocks_for(total_tokens);
        let t = self.tables.entry(seq).or_default();
        if total_tokens <= t.num_tokens {
            return Ok(copies);
        }
        if cow {
            let last = t.blocks.last_mut().expect("cow needs a block");
            let fresh = self.alloc.allocate().expect("checked free count");
            copies.push(BlockCopy {
                src: *last,
                dst: fresh,
            });
            self.alloc.decref(*last);
            *last = fresh;
        }
        while t.blocks.len() < target {
            t.blocks
                .push(self.alloc.allocate().expect("checked free count"));
        }
        t.num_tokens = total_tokens;
        self.debug_check();
        Ok(copies)
    }

    /// Gives `child` the same blocks as `parent` (shared, reference counted).
    pub fn fork(&mut self, parent: SeqId, child: SeqId) {
        let table = self
            .tables
            .get(&parent)
            .expect("fork of unknown sequence")
            .clone();
        assert!(
            !self.tables.contains_key(&child),
            "fork into existing sequence {child}"
        );
        for &b in &table.blocks {
            self.alloc.incref(b);
        }
        self.tables.insert(child, table);
        self.debug_check();
    }

    /// Releases every block `seq` holds. Unknown sequences are ignored.
    pub fn free(&mut self, seq: SeqId) {
        if let Some(t) = self.tables.remove(&seq) {
            for b in t.blocks {
                self.alloc.decref(b);
            }
        }
        self.debug_check();
    }

    fn debug_check(&self) {
        if cfg!(debug_assertions) {
            self.check_invariants().unwrap();
        }
    }

    /// No block is both free and referenced; each refcount equals the number of tables
    /// holding the block; every table has exactly enough blocks for its tokens.
    pub fn check_invariants(&self) -> Result<(), String> {
        let n = self.num_blocks();
        let mut holders = vec![0u32; n];
        for (seq, t) in &self.tables {
            if t.blocks.len() != self.blocks_for(t.num_tokens) {
                return Err(format!(
                    "seq {seq}: {} blocks for {} tokens",
                    t.blocks.len(),
                    t.num_tokens
                ));
            }
            for &b in &t.blocks {
                holders[b as usize] += 1;
            }
        }
        let mut in_free = vec![false; n];
        for &b in &self.alloc.free {
            if std::mem::replace(&mut in_free[b as usize], true) {
                return Err(format!("block {b} is on the free list twice"));
            }
            if self.alloc.hash_of[b as usize].is_some() {
                return Err(format!("free block {b} is still registered"));
            }
        }
        for (&t, &b) in &self.alloc.evictable {
            if std::mem::replace(&mut in_free[b as usize], true) {
                return Err(format!(
                    "block {b} is both free and evictable, or evictable twice"
                ));
            }
            if self.alloc.evict_tick[b as usize] != Some(t)
                || self.alloc.hash_of[b as usize].is_none()
            {
                return Err(format!("evictable block {b} has stale bookkeeping"));
            }
        }
        for (&h, &b) in &self.alloc.cached {
            if self.alloc.hash_of[b as usize] != Some(h) {
                return Err(format!(
                    "cache maps {h:x} to block {b}, which holds {:?}",
                    self.alloc.hash_of[b as usize]
                ));
            }
        }
        if self.alloc.hash_of.iter().flatten().count() != self.alloc.cached.len() {
            return Err("registered blocks and cache entries disagree".into());
        }
        for b in 0..n {
            let rc = self.alloc.refcounts[b];
            if rc != holders[b] {
                return Err(format!(
                    "block {b}: refcount {rc} but {} tables hold it",
                    holders[b]
                ));
            }
            if (rc == 0) != in_free[b] {
                return Err(format!(
                    "block {b}: refcount {rc}, free or evictable: {}",
                    in_free[b]
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn reserve_grows_by_whole_blocks() {
        let mut m = BlockManager::new(8, 4);
        m.reserve(1, 5).unwrap();
        assert_eq!(m.table(1).unwrap().blocks.len(), 2);
        m.reserve(1, 8).unwrap();
        assert_eq!(m.table(1).unwrap().blocks.len(), 2);
        m.reserve(1, 9).unwrap();
        assert_eq!(m.table(1).unwrap().blocks.len(), 3);
        assert_eq!(m.num_free_blocks(), 5);
        m.free(1);
        assert_eq!(m.num_free_blocks(), 8);
    }

    #[test]
    fn out_of_blocks_changes_nothing() {
        let mut m = BlockManager::new(2, 4);
        m.reserve(1, 4).unwrap();
        assert_eq!(m.reserve(1, 13), Err(OutOfBlocks));
        assert_eq!(m.table(1).unwrap().num_tokens, 4);
        assert_eq!(m.num_free_blocks(), 1);
    }

    #[test]
    fn fork_shares_and_writes_copy_on_write() {
        let mut m = BlockManager::new(8, 4);
        m.reserve(1, 6).unwrap();
        m.fork(1, 2);
        assert_eq!(m.num_free_blocks(), 6, "fork allocates nothing");
        let shared_last = m.table(1).unwrap().blocks[1];
        // Seq 2 writes token 6 into the shared, half-full last block: copy it first.
        let copies = m.reserve(2, 7).unwrap();
        let mine = m.table(2).unwrap().blocks[1];
        assert_eq!(
            copies,
            [BlockCopy {
                src: shared_last,
                dst: mine
            }]
        );
        assert_ne!(mine, shared_last);
        assert_eq!(
            m.table(2).unwrap().blocks[0],
            m.table(1).unwrap().blocks[0],
            "full block stays shared"
        );
        // Seq 1 now owns its last block alone: no copy needed.
        assert!(m.reserve(1, 7).unwrap().is_empty());
    }

    #[test]
    fn full_shared_blocks_need_no_copy() {
        let mut m = BlockManager::new(8, 4);
        m.reserve(1, 8).unwrap();
        m.fork(1, 2);
        assert!(m.reserve(2, 9).unwrap().is_empty());
        assert_eq!(m.blocks_needed(1, 9), 1);
    }

    #[test]
    fn freed_registered_blocks_stay_cached_until_needed() {
        let mut m = BlockManager::new(4, 2);
        let tokens = [1, 2, 3, 4, 5];
        let hashes = block_hashes(&tokens, 2);
        assert_eq!(hashes.len(), 2);
        m.reserve(1, 5).unwrap();
        m.register_block(1, 0, hashes[0]);
        m.register_block(1, 1, hashes[1]);
        m.free(1);
        assert_eq!((m.num_free_blocks(), m.num_cached_blocks()), (4, 2));
        // Same prefix: both blocks come back without allocating.
        assert_eq!(m.prefix_match(&hashes), (2, 2));
        assert_eq!(m.adopt_prefix(2, &hashes), 2);
        assert_eq!((m.num_free_blocks(), m.num_cached_blocks()), (2, 0));
        m.free(2);
        // A different sequence that needs every block evicts the cache, oldest first.
        m.reserve(3, 8).unwrap();
        assert_eq!(m.num_cached_blocks(), 0);
        assert_eq!(m.prefix_match(&hashes), (0, 0));
    }

    #[test]
    fn prefix_hashes_cover_everything_before() {
        let a = block_hashes(&[1, 2, 3, 4], 2);
        let b = block_hashes(&[9, 9, 3, 4], 2);
        assert_ne!(a[1], b[1], "same block, different prefix");
        assert_eq!(block_hashes(&[1, 2, 3], 2), a[..1]);
    }

    #[derive(Debug, Clone)]
    enum Op {
        Reserve {
            seq: SeqId,
            extra: usize,
        },
        Fork {
            parent: SeqId,
            child: SeqId,
        },
        Free {
            seq: SeqId,
        },
        /// Register every full block of `seq` under hashes derived from `salt`.
        Register {
            seq: SeqId,
            salt: u64,
        },
        /// Start `seq` from cached blocks of `salt`'s hash chain.
        Adopt {
            seq: SeqId,
            salt: u64,
            blocks: usize,
        },
    }

    fn op() -> impl Strategy<Value = Op> {
        prop_oneof![
            4 => (0..6u64, 1..20usize).prop_map(|(seq, extra)| Op::Reserve { seq, extra }),
            2 => (0..6u64, 0..6u64).prop_map(|(parent, child)| Op::Fork { parent, child }),
            2 => (0..6u64).prop_map(|seq| Op::Free { seq }),
            2 => (0..6u64, 0..3u64).prop_map(|(seq, salt)| Op::Register { seq, salt }),
            2 => (0..6u64, 0..3u64, 1..4usize).prop_map(|(seq, salt, blocks)| Op::Adopt { seq, salt, blocks }),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        /// Random allocate/fork/append/free sequences never leak, double-free or break
        /// refcounts, failed reservations change nothing, and freeing everything returns
        /// the pool to full.
        #[test]
        fn random_ops_keep_invariants(ops in proptest::collection::vec(op(), 1..60), bs in 1..6usize) {
            let mut m = BlockManager::new(12, bs);
            for op in ops {
                match op {
                    Op::Reserve { seq, extra } => {
                        let have = m.table(seq).map_or(0, |t| t.num_tokens);
                        let before = m.clone();
                        let needed = m.blocks_needed(seq, have + extra);
                        match m.reserve(seq, have + extra) {
                            Ok(copies) => {
                                prop_assert_eq!(before.num_free_blocks() - m.num_free_blocks(), needed);
                                prop_assert!(copies.len() <= 1);
                                let t = m.table(seq).unwrap();
                                prop_assert_eq!(t.num_tokens, have + extra);
                                // Every block the new tokens land in is this table's alone, so
                                // writing them can't change what another table sees.
                                for &b in &t.blocks[have / bs..] {
                                    prop_assert_eq!(m.refcount(b), 1);
                                }
                                for c in &copies {
                                    prop_assert_eq!(before.refcount(c.src) > 1, true);
                                    prop_assert_eq!(m.refcount(c.src), before.refcount(c.src) - 1);
                                }
                            }
                            Err(OutOfBlocks) => {
                                prop_assert!(needed > before.num_free_blocks());
                                prop_assert_eq!(m.table(seq), before.table(seq));
                                prop_assert_eq!(m.num_free_blocks(), before.num_free_blocks());
                            }
                        }
                    }
                    Op::Fork { parent, child } => {
                        if m.contains(parent) && !m.contains(child) {
                            m.fork(parent, child);
                        }
                    }
                    Op::Free { seq } => m.free(seq),
                    Op::Register { seq, salt } => {
                        if let Some(t) = m.table(seq).cloned() {
                            for i in 0..t.num_tokens / bs {
                                m.register_block(seq, i, salt * 100 + i as u64);
                            }
                        }
                    }
                    Op::Adopt { seq, salt, blocks } => {
                        if !m.contains(seq) {
                            let hashes: Vec<u64> = (0..blocks as u64).map(|i| salt * 100 + i).collect();
                            let (want, _) = m.prefix_match(&hashes);
                            prop_assert_eq!(m.adopt_prefix(seq, &hashes), want);
                        }
                    }
                }
                prop_assert_eq!(m.check_invariants(), Ok(()));
            }
            for seq in 0..6 {
                m.free(seq);
            }
            prop_assert_eq!(m.num_free_blocks(), 12);
        }
    }
}
