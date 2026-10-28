# Design

rlis is an LLM serving engine in Rust for Llama-architecture models on an Apple Silicon GPU
(Metal), with continuous batching, a paged KV cache and an OpenAI-compatible API. It was built
for Llama 3.2 1B; Qwen2 (Qwen2.5-Coder 1.5B and 3B, for `rlis fix`) loads from GGUF as Llama
plus Q/K/V biases. This document
covers how it works, why it is built this way, and what was measured along the way. Numbers are
from an M4 Pro (20-core GPU, 24 GB) running Llama 3.2 1B Instruct, except the
[CUDA run](#cuda-run) on a Tesla T4; the benchmark files under `results/` carry full provenance.

## Architecture

```
 client ──HTTP/SSE──▶ axum handlers (tokio) ──bounded queue──▶ engine thread (owns GPU + KV state)
   ▲                    validate, tokenize,     full → 503        loop:
   │                    chat template                               take new requests, drop cancelled
   └──── per-request token channel ◀────────────────────────────── schedule → forward → sample → stream
```

| Crate | Role |
|---|---|
| `server` | axum API (`/v1/completions`, `/v1/chat/completions`, `/health`, `/metrics`), CLI (`rlis serve / generate / perplexity`) |
| `engine` | engine thread, scheduler, sampling, stop strings, detokenizer, chat template, perplexity |
| `kvcache` | block allocator, block tables, refcounts / copy-on-write, prefix cache (pure logic, no GPU) |
| `model` | Llama forward pass on candle; safetensors and GGUF loading; paged KV storage; Metal kernels |
| `bench` | load generator, charts, soak test |

**Concurrency.** tokio handles all network I/O. Everything stateful — the scheduler, every
sequence, the KV pool and the GPU — belongs to one OS thread, so the hot path takes no locks.
Requests arrive over a bounded channel (a full queue answers 503 with `Retry-After`) and each
request streams back over its own unbounded channel. A dropped receiver is how cancellation
works: the engine checks `is_closed()` every step and frees the sequence's blocks at once, so a
client disconnect costs nothing but a pointer check.

## Scheduler

Each step the scheduler decides which sequences run and how many tokens each computes. A
sequence is just `tokens` (prompt plus output so far) and `num_computed` (tokens whose K/V are
cached); every step computes the next chunk of the gap. One rule covers prefill (gap = prompt),
chunked prefill (gap larger than the token budget), decode (gap = 1) and recompute after
preemption (cache dropped, gap = everything).

- **Running first.** Running sequences are scheduled oldest first within `max_batched_tokens`
  (default 2048) and `max_num_seqs` (64). Prefill and decode share a step.
- **Admission.** First come, first served: preempted sequences (oldest first) before new ones.
  A sequence is admitted only if its whole known length fits with a 1% watermark of blocks to
  spare for decode growth, and it reserves those blocks immediately so later admissions can't
  overcommit. The head of the line is never skipped, so nothing starves.
- **Preemption.** When a running sequence needs a block and none is free, the newest running
  sequence is preempted: its blocks are freed and it recomputes later. No admissions happen in
  a step that preempted, so a victim isn't immediately readmitted.
- **Static mode** (`--scheduler static`) admits only when nothing is running: the baseline for
  the continuous-batching ablation.

**Recompute, not swap.** vLLM can swap a victim's KV blocks to host memory. On Apple Silicon the
GPU and CPU share one memory pool, so "swapping to host" moves bytes without freeing any.
Recompute is the only preemption that actually releases memory here, and with prefix caching
(below) most of a victim's blocks are still cached when it returns, so recompute is cheap.

Scheduler tests drive it with a fake executor: admission, chunking, decode-before-prefill
priority, victim choice, readmission order, and 200 random workloads that must all finish with
every block returned.

## Paged KV cache

- **Pool.** Allocated once at startup from `--kv-cache-gb` (default 1 GiB), never from "free
  memory": on unified memory the GPU shares RAM with everything else. Each layer's K and V is
  one `(num_blocks × block_size, n_kv_heads, head_dim)` f16 tensor, so token slot `s` is row
  `s`. Llama 3.2 1B uses 32 KB per token, so the default pool holds 32,768 tokens.
- **Blocks.** 16 tokens each (see the block-size ablation). Each sequence has a block table
  mapping logical blocks to physical ones. A step writes every new token of the batch with one
  `scatter_set` per layer, using slot indices computed once per step.
- **Sharing.** Blocks are reference counted. A write into a partially filled shared block copies
  it first (copy-on-write); the copy is returned to the executor, which performs it before the
  forward pass.
- **Prefix caching.** A full block is registered under a chained hash of its tokens and every
  token before it. Registered blocks outlive their sequences: at refcount zero they become
  *evictable* rather than free, and are reclaimed least recently used only after the free list
  is empty. An admitted sequence adopts the longest cached chain of its leading blocks, capped
  so the block holding the last prompt token is always computed (its logits are needed).
  Sequences admitted in the same step can't share blocks nobody has computed yet.
- **Invariants**, checked after every mutation in debug builds: no block is both free and in
  use; each refcount equals the number of tables holding the block; tables have exactly enough
  blocks for their tokens; the hash map, the per-block hashes and the evictable set agree. A
  property test runs 2,000 random sequences of reserve / fork / free / register / adopt.

## Attention

- **v1 (gather).** Gather each sequence's blocks into contiguous K/V, then standard attention.
  Simple and obviously correct; it stays as the fallback and the reference.
- **v2 decode kernel** (`paged_attention.metal`). One kernel per layer for every decoding
  sequence, reading K/V straight through the block tables. A threadgroup serves all query
  heads of a GQA group, so each K/V row is loaded once. Its 8 simdgroups take blocks of 32
  positions: each lane scores one position against every head, the online softmax is updated
  once per block, and lanes then accumulate their slice of V. Long contexts with few sequences
  are split into partitions that run as separate threadgroups and are merged by a second
  kernel (flash-decoding). Without the split, one sequence was only 8 threadgroups on a
  20-core GPU, and an 8k-context decode step took 34 ms instead of 10. It is compiled for up
  to 4 and up to 8 query heads per KV head: Qwen2.5 1.5B has 6 and 3B has 8, and sizing
  Llama's per-lane state for 8 cost it ~10% of attention time. The wide variant merges its
  simdgroups' outputs 4 heads per round, since staging all 8 would take 32 KB of threadgroup
  memory. It matches a CPU reference for both Llama 3.2 head layouts, Qwen2.5 1.5B's and 8
  heads per KV head, on 37 sequences up to 700 tokens and on split contexts up to 9,000.
- **Fused prefill.** Multi-token segments use MLX's fused attention (`candle_nn::ops::sdpa`)
  with a bottom-right-aligned causal mask, which is exactly chunked-prefill semantics. The
  first version materialized f32 score matrices and spent 313 ms of a 1024-token prefill in
  attention; fused, it is 25 ms, and prefill throughput went from ~1,030 to ~1,700 tok/s.

## Model and quantization

The forward pass is written against candle tensor ops: RMSNorm, RoPE with Llama 3 frequency
scaling, grouped-query attention, SwiGLU. Weights load from HF safetensors (f32/f16) or GGUF.

GGUF needs three adjustments, each covered by a test: config from `llama.*` metadata; Llama 3
RoPE scaling stored as `rope_freqs.weight` divisors (checked to reproduce the HF frequencies);
and Q/K rows permuted for interleaved RoPE (`rope_i`). `qwen2` files differ: `qwen2.*` keys,
Q/K/V biases, and Q/K left unpermuted, so rotation stays rotate-half. Qwen2's biases also make
q·k overflow f16 (|k| reaches ~360 in the first layer), which turned every logit into NaN with
the f16 KV pool. The pool stays f16, but for Qwen2 the gathered attention (prefill, v1) runs in
f32, as llama.cpp does for this architecture; the decode kernel scores in f32 anyway. Parity
for Qwen2.5-Coder 1.5B is against transformers' own Qwen2 code holding the GGUF's weights in
f16 with f32 matmuls (f32 weights would be ~6 GB): logits within 0.016 and greedy output
token-for-token on 20 prompts × 128 tokens, alone and as a paged batch of 32 (`make
parity-qwen`). Matrices stay block-quantized, and the
token embedding stays quantized too — looked up row by row on the GPU and shared with the tied
LM head — so no 525 MB dequantized table ever exists.

**Small-batch decode.** candle's batched quantized matmul costs about the same for 2 rows as for
32, ~4× its single-row matrix-vector kernel, so a decode step with 2 sequences took 2.8× as long
as with 1. Two fixes: a Q8_0 GEMV kernel (`gemv.metal`) that reads each weight block once for
up to 8 rows, used for 2–5 rows; and, for other quantization types, candle's matrix-vector
kernel once per row up to 4 rows. Thresholds come from interleaved measurements of all three
paths. Step time at batch 2 went from 22.7 ms to 10.2 ms, at batch 4 from 36.2 ms to 16.1 ms.

| Format | WikiText-2 PPL | Top-1 agreement with f32 reference | Decode, 1 stream |
|---|---|---|---|
| f16 | 14.0009 ± 0.104 | exact (parity test) | 83 tok/s |
| Q8_0 (default) | 14.0035 ± 0.104 | 99.38% | 145 tok/s |
| Q4_K_M | 14.4563 ± 0.108 | 94.45% | 196 tok/s |

## Sampling

Greedy and temperature-only rows are sampled on the GPU in one kernel: Gumbel-max (argmax of
logit / T plus Gumbel noise is an exact draw from softmax(logit / T)), with noise from a
counter-based hash of (request seed, position, token id). A draw is reproducible and
independent of what else is in the batch, and only token ids cross back to the host. The
uniform draw takes 23 bits of the hash: with 24, (2^24 − 0.5) / 2^24 rounds to exactly 1 in
f32, the noise is infinite, and that token wins whatever its logit. That hit one vocabulary
entry in 2^24 per step, ~1% of steps over Qwen's 152k vocabulary (0.8% over Llama's), and
showed up as stray tokens mid-sentence (`median耵r`) once `rlis fix` sampled code at T = 0.7.
Rows with top-k, top-p or a repetition penalty bring their logits to the CPU sampler, which is
O(vocab) without sorting the vocabulary (an earlier version sorted all 128k entries per token
and spent half of each step doing it).

## Memory

The default run (Q8_0, 1 GiB pool) peaks at 3.1–3.3 GB under the heaviest load I could build:
64 concurrent long requests overfilling the pool with constant preemption. Two fixes got it
there from 5.0 GB:

- **Load-time cast.** candle's safetensors loader uploads bf16 and casts on the GPU; the freed
  bf16 buffers sat in the allocator's cache. Casting on the CPU before upload removed 1.3 GB.
- **Readback churn.** Temperature sampling read every sampled row's 128k logits back through a
  fresh 33 MB staging buffer per step, and the Metal driver kept freed staging memory charged to
  the process (~0.6 GB of "owned, unmapped" footprint). GPU sampling removed the readback.

Retaining candle's buffer pool across steps was tried first and made no difference — the
"unmapped" memory wasn't pool churn — which is how the readback was found.

The hour-long soak then found two slow leaks that short benchmarks can't see (the footprint
grew ~350 MB over an hour at constant load):

- **Autoreleased Metal objects.** Metal returns command encoders and label strings
  *autoreleased*. A pool is only drained when it ends, and the engine is a plain long-lived
  `std::thread` with no pool, so every encoder from every step stayed alive: `heap` showed
  +293,000 live objects in 3 minutes (encoder contexts, string containers), ~25 KB per request.
  Each engine iteration now runs inside its own autorelease pool: +655 objects in the same 3
  minutes.
- **Unscraped histograms.** The Prometheus exporter keeps raw histogram samples (one per
  generated token for inter-token latency) until a scrape or an explicit upkeep; installed with
  `install_recorder`, nothing runs the upkeep, so a server nobody scrapes grows forever. The
  server now runs upkeep every 5 s.

## Correctness

- **Reference parity** against transformers in f32 on CPU, 20 prompts × 128 tokens: teacher-forced
  logits within 5e-4 (measured 4.4e-5) and greedy output token-for-token, both single-sequence
  and as a paged batch of 32 with scattered block tables. In f16 a flip is allowed only where
  the reference's top-2 margin is under 0.02 (two such near-ties in 2,560 steps).
- **Chat template** rendered by minijinja matches `apply_chat_template` text and token ids.
- **Batching invariance**, preemption-with-recompute and prefix caching are each checked to leave
  greedy output unchanged.
- **Fuzzing** (`cargo-fuzz`): the request parser and the stop-string matcher. The fuzzer found
  that the matcher's output depended on how text was split into tokens; it now stops at the
  stop string that completes first.
- **Soak** (`make soak`): an hour at concurrency 32 with 25% of clients hanging up mid-stream.
  Result: 13,018 requests, 3,131 hang-ups, 0 errors, KV utilization back to 0.0 when idle, and
  10-minute footprint averages of 2,598 / 2,589 / 2,607 / 2,640 / 2,645 / 2,626 MB (+2.9% early
  to late). The first run grew +9.2% (to 2,954 MB) and found the two leaks under *Memory*; it
  is kept as `results/mac/soak-before-leak-fixes.json`.

## Rejected alternatives

| Option | Why not |
|---|---|
| Swap-based preemption | Unified memory: nothing is freed by "swapping to host". |
| KV pool sized from free memory | Unpredictable on a shared-memory laptop; a flag is explicit. |
| Per-request contiguous KV (M2) | Reserves max length per request; 2 GB for 32 long chats. |
| Static batching | Measured: p99 TTFT 8.4 s vs 65 ms at 32 concurrent requests. |
| Writing prefill matmul kernels | Out of scope by design; the remaining prefill gap to llama.cpp is candle's quantized matmul at large M (see the profile). |
| Locks around engine state | One owner thread is simpler and has no hot-path contention. |
| f32 activations end to end for f16 models | Doubles activation memory; f16 is within parity tolerance. |

## Ablations

Same workload as the benchmark (concurrency 4 / 16 / 32), one change at a time; full tables in
`results/mac/tables.md`.

- **Continuous vs static batching.** Static batching loses 26–35% of the throughput and its p99
  time to first token is 4–11× worse (13.9 s vs 3.5 s at 32). Its inter-token latency is
  *lower* — a static batch never has prefills cutting into its decode steps — which is the
  trade continuous batching makes: newcomers start immediately at the cost of occasional
  slower decode steps.
- **Paged-attention kernel (v2) vs gather (v1).** 2.9× the throughput at 32 concurrent requests
  (540 vs 186 tok/s); v1's per-sequence gathers and small attention launches dominate the step.
- **Small-batch GEMV.** +36% at 4 concurrent requests (214 vs 157 tok/s), no effect at 16+
  where batches are past the GEMV's 5-row range, as expected.
- **Block size 8 / 16 / 32.** Within 1% on throughput. The kernel reads through block tables,
  so the block size only matters for table size and for internal fragmentation (on average
  half a block per sequence); 16 is the middle of a flat curve.
- **Prefix caching** with a 790-character system prompt on every request: +52% throughput at 32
  (634 vs 417 tok/s) and p99 time to first token from 5.5 s to 0.53 s, because most of every
  prompt is never prefilled again.

## Benchmark summary

On the same model, KV budget and batch size, llama.cpp's server is faster up to 16 concurrent
requests (388 vs 214 tok/s at 4) and rlis is faster at 32–64 (559 vs 467 tok/s at 64, with
38% lower median inter-token latency and 3.2 vs 4.5 GB peak memory). The low-concurrency gap
is kernels, not scheduling: llama.cpp's quantized matmuls are better tuned for prefill and for
2–16 row decode than candle's plus my GEMV. At high concurrency the paged kernel, the
scheduler and the fixed pool matter more.

## Where the time goes

**GPU-bound.** Under the hottest configuration (64 concurrent requests), a Time Profiler trace
shows the engine thread using ~28 ms of CPU per second — under 3% of a core — and blocked on
the GPU the rest of the time. Of that CPU time, about half is Metal residency-set commits
inside candle's allocator, 12% building the step's slot indices, and ~8% detokenization; the
HTTP side costs ~9 ms/s, mostly socket writes. A Metal System Trace of a prefill run shows the
GPU ~99% busy. So no CPU-side optimization would move throughput; the time is in GPU kernels.

Per-stage GPU time of a 1024-token Q8_0 prefill (`RLIS_PROFILE=1`, synced spans, so read as
shares): MLP gate+up 38%, MLP down 20%, QKV and O projections 14%, transposes and RoPE copies
10%, fused attention 5%, the rest norms and adds. A Metal System Trace shows the GPU ~99% busy
during the pass, so it is GPU-bound, not limited by CPU encoding. In-model, candle's quantized
matmul runs ~1.6× slower than the same matmul in a hot-cache microbenchmark, which points at
weight re-reads at prefill sizes. That is the main reason llama.cpp prefills faster (2,063 vs
~1,700 tok/s on a 1,026-token prompt) and leads at 4–16 concurrent requests.

## CUDA run

The spec's last milestone: the same commit on a free cloud NVIDIA GPU, parity first, then the
benchmark. `make bench-cuda` pushes `scripts/cuda/kaggle.py` (pinned to `HEAD`) to Kaggle,
which clones the repo and runs `scripts/cuda/run.sh`: Rust toolchain, models, `--features
cuda` build, `make parity`, llama.cpp with CUDA, then `scripts/bench/sweep.sh`, the same sweep
as on the Mac. Parity gates the sweep, so a failed comparison records nothing. Both servers are
pinned to one of the session's two T4s, since llama.cpp would otherwise split across both.

**What runs differently.** Every Metal kernel has a portable fallback, and the CUDA build uses
them: attention v1 (gather each sequence's blocks, then standard attention) for decode, plain
matmul attention for prefill, candle's quantized matmuls without the small-batch GEMV, and the
CPU reference sampler on logits copied back to the host (~1–2 ms per step at 32 rows). On the
Mac the gather path alone cost 2.9× the throughput at 32 requests (see Ablations), so the
paged-attention kernel is the likely bulk of the gap below. The T4 run wasn't profiled, though,
so that split is an inference, not a measurement.

**Result.** Parity passed on the T4: f32 logits within 5e-4 and greedy output token-for-token
for 20 prompts × 128 tokens, alone and as a paged batch of 32; f16 within its near-tie rule.
Single-stream decode is 118 vs llama.cpp's 132 tok/s (89%, against a 60% target), a bit above
the Mac's 83%. Beyond one stream the picture inverts from the Mac's: llama.cpp leads at 4–32
(296 / 439 / 426 vs 239 / 305 / 355 tok/s), because on CUDA it keeps its tuned kernels and rlis
loses its own. What carries over is the scheduler and the fixed pool: rlis's p99 time to first
token is lower at 16 and 32 (0.82 vs 1.07 s, 2.0 vs 4.4 s), and its memory stays at 3.5–3.7 GB
while llama.cpp grows to 5.1 GB. On Linux the bench tool measures RSS plus GPU memory in use,
since RSS alone doesn't count device allocations.

**Getting a clean run took seven Kaggle sessions**, and the failures are the useful part:

- candle 0.11.0's CUDA kernels didn't compile: its `compatibility.cuh` defines `__hmax_nan` and
  `__hmin_nan` for GPUs below sm_80, and CUDA 12.8's headers now define them for every
  architecture. `run.sh` adds a toolkit-version check to that guard in the session's copy of
  the crate (same NaN-propagating semantics; `Cargo.lock` untouched).
- Nothing could link `-lcuda`: containers mount the driver's `libcuda.so.1` but not the
  unversioned name, and Kaggle's toolkit has no stub. `run.sh` points a symlink at the driver's
  library for both the Rust link and llama.cpp's CMake (`CUDA_cuda_driver_LIBRARY`).
- llama.cpp's build with unlimited `-j` ran the 4-vCPU, 31 GB session out of memory; the
  OOM-killed build then stalled for over two hours on the weekly GPU quota. Builds now get one
  job per CPU and a time limit, and every session a hard stop.
- **A real bug**: the engine's sampler had no CUDA path (`no cuda implementation for
  sample-rows`), so every step failed. The parity tests drive the model directly and passed
  three times without reaching it; the sweep's warmup was the first engine-level test on CUDA.
  On CUDA, `sample_rows` now copies the logits back and uses the CPU reference, which picks the
  same tokens. Parity gates the model, not the server, so the warmup is the real end-to-end
  check, and it runs before anything is recorded.

The spec assumed about 30 GPU-hours a week; this account had 6, which is why the cloud sweep
stops at 32 concurrent requests and builds llama.cpp on the CPUs while parity runs on the GPU.
