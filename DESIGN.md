# Design

rlis is an LLM serving engine in Rust for one Llama-architecture model on an Apple Silicon GPU
(Metal), with continuous batching, a paged KV cache and an OpenAI-compatible API. This document
covers how it works, why it is built this way, and what was measured along the way. Numbers are
from an M4 Pro (20-core GPU, 24 GB) running Llama 3.2 1B Instruct; the benchmark files under
`results/` carry full provenance.

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
  sequence, reading K/V straight through the block tables. One threadgroup per (KV head,
  sequence) serves all query heads of the GQA group, so each K/V row is loaded once; 8
  simdgroups split the context with an online softmax and merge at the end. It matches a CPU
  reference on 37 sequences with contexts up to 700 tokens for both Llama 3.2 head layouts.
- **Fused prefill.** Multi-token segments use MLX's fused attention (`candle_nn::ops::sdpa`)
  with a bottom-right-aligned causal mask, which is exactly chunked-prefill semantics. The
  first version materialized f32 score matrices and spent 313 ms of a 1024-token prefill in
  attention; fused, it is 25 ms, and prefill throughput went from ~1,030 to ~1,700 tok/s.

## Model and quantization

The forward pass is written against candle tensor ops: RMSNorm, RoPE with Llama 3 frequency
scaling, grouped-query attention, SwiGLU. Weights load from HF safetensors (f32/f16) or GGUF.

GGUF needs three adjustments, each covered by a test: config from `llama.*` metadata; Llama 3
RoPE scaling stored as `rope_freqs.weight` divisors (checked to reproduce the HF frequencies);
and Q/K rows permuted for interleaved RoPE (`rope_i`). Matrices stay block-quantized, and the
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
independent of what else is in the batch, and only token ids cross back to the host. Rows with
top-k, top-p or a repetition penalty bring their logits to the CPU sampler, which is O(vocab)
without sorting the vocabulary (an earlier version sorted all 128k entries per token and spent
half of each step doing it).

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
- **Soak**: an hour at high concurrency with random client hang-ups; every KV block must return
  and memory must stay flat (`rlis-bench soak`).

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
