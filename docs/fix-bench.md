# What do N attempts cost? (`rlis fix` feasibility)

`rlis fix --n 8` only makes sense if 8 attempts at a patch cost far less than 8 separate
requests. The plan is for all N attempts to share one prefill of the prompt and decode together
in one batch. This page measures that on the Mac: if 8 attempts cost about 2–3× one attempt,
the idea holds; if it's close to 8×, it doesn't.

**Answer: it holds, at the top of the 2–3× band.** On rlis, 8 attempts forked from one prefill
cost **2.72×** one attempt with an 8k-token prompt and **3.15×** with a 2k prompt. Without
sharing the same work costs 6.28× and 4.75×, and llama.cpp's server with N separate requests
costs 5.11× and 3.21×. llama.cpp's own `n=N` path does better (1.47× and 1.67×) and is twice
as fast at N=8 (13.8 s vs 27.4 s at 8k). For a single attempt rlis now matches llama.cpp (10.1
s vs 9.4 s at 8k), so the rest of the gap is the cost of decoding 8 sequences at once (see
[Where the time goes](#where-the-time-goes)).

These numbers come after fixing rlis's long-context decode, which first measured 3.5× slower
than llama.cpp at 8k (see [Long-context decode fix](#long-context-decode-fix)).

## Setup

- **Hardware and model:** M4 Pro (20-core GPU, 24 GB), Llama 3.2 1B Instruct Q8_0, f16 KV
  cache.
- **Workload:** each run makes N completions of one prompt, of 2k or 8k tokens, with 512
  tokens each (`ignore_eos`), temperature 0.7, and one seed per completion. Every run starts
  with a unique tag and corpus slice (WikiText-2), so no run is helped by an earlier run's
  cache. Each cell is the median of 3 repetitions, and the repetitions interleave the N values.
- **Same budget for both servers:** 73,728 KV tokens (2.25 GiB), enough for 8 unshared 8k
  prompts plus their outputs, so rlis never preempts. Both servers get 2,048 tokens per step.
  llama.cpp runs 8 slots over a unified KV buffer, with flash attention and prompt cache on.
  Its host-RAM prompt store is capped at 1 GiB (`-cram 1024`); the default of 8 GiB would
  break the memory budget.
- **Wall time** runs from the first request sent to the last completion finished.
  **Cost ratio** is wall(N) / wall(1) for the same series and prompt length.
- **KV blocks in use** is the peak of rlis's `/health` `kv_blocks_used`, sampled every 25 ms.
  llama.cpp doesn't report KV cells in use.

| Series | What it is |
|---|---|
| (a) rlis: N requests, no sharing | N concurrent `n=1` requests, prefix caching off |
| (b) rlis: n=N, forked | one request with `n=N`: the prompt is prefilled once, then N sequences fork off it |
| (a′) rlis: N requests, prefix caching | (a) with prefix caching on (the server default) |
| (c) llama.cpp: N requests, prompt cache | N concurrent `n=1` requests |
| (c′) llama.cpp: n=N | one request with `n=N` (llama.cpp shares the prompt across its children) |

Run it with `make bench-fork`, then `make bench-charts`. Raw results are in
`results/mac/fork-*.json`.

## Results

Cost ratio at N=8 (median wall time in parentheses):

| Series | 2k prompt | 8k prompt |
|---|---|---|
| (a) rlis, N requests, no sharing | 4.75× (23.6 s) | 6.28× (66.2 s) |
| **(b) rlis, n=N, forked** | **3.15× (15.4 s)** | **2.72× (27.4 s)** |
| (a′) rlis, N requests, prefix caching | 3.07× (15.0 s) | 2.67× (26.5 s) |
| (c) llama.cpp, N requests, prompt cache | 3.21× (13.3 s) | 5.11× (51.6 s) |
| (c′) llama.cpp, n=N | 1.67× (7.9 s) | 1.47× (13.8 s) |

![Cost ratio, 8k prompt](fix-cost-8k.svg)
![Wall time, 8k prompt](fix-wall-8k.svg)
![Cost ratio, 2k prompt](fix-cost-2k.svg)
![Wall time, 2k prompt](fix-wall-2k.svg)

Full table (medians of 3 runs; footprint and KV blocks are the max over the 3):

| configuration | prompt | N | wall (s) | cost ratio | mean TTFT (s) | output tok/s | peak footprint | peak KV blocks (tokens) |
|---|---|---|---|---|---|---|---|---|
| rlis: N requests, no sharing (a) | 2k | 1 | 4.96 | 1.00× | 1.00 | 103 | 4.45 GB | 160 (2560) |
| rlis: N requests, no sharing (a) | 2k | 2 | 8.22 | 1.66× | 2.36 | 125 | 4.45 GB | 321 (5136) |
| rlis: N requests, no sharing (a) | 2k | 4 | 15.06 | 3.04× | 3.78 | 136 | 4.48 GB | 640 (10240) |
| rlis: N requests, no sharing (a) | 2k | 8 | 23.56 | 4.75× | 6.00 | 174 | 4.48 GB | 1281 (20496) |
| rlis: N requests, no sharing (a) | 8k | 1 | 10.54 | 1.00× | 4.92 | 49 | 4.52 GB | 544 (8704) |
| rlis: N requests, no sharing (a) | 8k | 2 | 19.42 | 1.84× | 8.40 | 53 | 4.52 GB | 1088 (17408) |
| rlis: N requests, no sharing (a) | 8k | 4 | 36.52 | 3.46× | 13.95 | 56 | 4.53 GB | 2176 (34816) |
| rlis: N requests, no sharing (a) | 8k | 8 | 66.21 | 6.28× | 24.61 | 62 | 4.59 GB | 4348 (69568) |
| rlis: n=N, forked from one prefill (b) | 2k | 1 | 4.90 | 1.00× | 0.96 | 104 | 4.49 GB | 160 (2560) |
| rlis: n=N, forked from one prefill (b) | 2k | 2 | 7.05 | 1.44× | 1.07 | 145 | 4.49 GB | 192 (3072) |
| rlis: n=N, forked from one prefill (b) | 2k | 4 | 11.45 | 2.34× | 1.07 | 179 | 4.49 GB | 256 (4096) |
| rlis: n=N, forked from one prefill (b) | 2k | 8 | 15.41 | 3.15× | 1.00 | 266 | 4.49 GB | 384 (6144) |
| rlis: n=N, forked from one prefill (b) | 8k | 1 | 10.07 | 1.00× | 4.60 | 51 | 4.52 GB | 544 (8704) |
| rlis: n=N, forked from one prefill (b) | 8k | 2 | 13.35 | 1.33× | 4.96 | 77 | 4.52 GB | 576 (9216) |
| rlis: n=N, forked from one prefill (b) | 8k | 4 | 19.31 | 1.92× | 4.81 | 106 | 4.52 GB | 643 (10288) |
| rlis: n=N, forked from one prefill (b) | 8k | 8 | 27.40 | 2.72× | 4.64 | 149 | 4.52 GB | 768 (12288) |
| rlis: N requests, prefix caching (a') | 2k | 1 | 4.88 | 1.00× | 0.90 | 105 | 4.46 GB | 160 (2560) |
| rlis: N requests, prefix caching (a') | 2k | 2 | 7.03 | 1.44× | 1.06 | 146 | 4.47 GB | 320 (5120) |
| rlis: N requests, prefix caching (a') | 2k | 4 | 11.21 | 2.30× | 1.09 | 183 | 4.45 GB | 259 (4144) |
| rlis: N requests, prefix caching (a') | 2k | 8 | 14.96 | 3.07× | 1.15 | 274 | 4.45 GB | 391 (6256) |
| rlis: N requests, prefix caching (a') | 8k | 1 | 9.92 | 1.00× | 4.32 | 52 | 4.52 GB | 544 (8704) |
| rlis: N requests, prefix caching (a') | 8k | 2 | 12.94 | 1.30× | 4.62 | 79 | 4.52 GB | 577 (9232) |
| rlis: N requests, prefix caching (a') | 8k | 4 | 20.15 | 2.03× | 5.72 | 102 | 4.53 GB | 770 (12320) |
| rlis: N requests, prefix caching (a') | 8k | 8 | 26.52 | 2.67× | 4.46 | 154 | 4.53 GB | 775 (12400) |
| llama.cpp: N requests, prompt cache (c) | 2k | 1 | 4.16 | 1.00× | 0.77 | 123 | 4.07 GB | n/a |
| llama.cpp: N requests, prompt cache (c) | 2k | 2 | 5.95 | 1.43× | 1.61 | 172 | 4.07 GB | n/a |
| llama.cpp: N requests, prompt cache (c) | 2k | 4 | 8.66 | 2.08× | 2.54 | 236 | 4.07 GB | n/a |
| llama.cpp: N requests, prompt cache (c) | 2k | 8 | 13.34 | 3.21× | 4.18 | 307 | 4.07 GB | n/a |
| llama.cpp: N requests, prompt cache (c) | 8k | 1 | 10.09 | 1.00× | 4.61 | 51 | 4.64 GB | n/a |
| llama.cpp: N requests, prompt cache (c) | 8k | 2 | 16.97 | 1.68× | 7.03 | 60 | 4.92 GB | n/a |
| llama.cpp: N requests, prompt cache (c) | 8k | 4 | 28.39 | 2.81× | 12.31 | 72 | 5.53 GB | n/a |
| llama.cpp: N requests, prompt cache (c) | 8k | 8 | 51.55 | 5.11× | 21.52 | 79 | 6.39 GB | n/a |
| llama.cpp: n=N (c') | 2k | 1 | 4.74 | 1.00× | 0.80 | 108 | 3.38 GB | n/a |
| llama.cpp: n=N (c') | 2k | 2 | 5.63 | 1.19× | 0.81 | 182 | 3.47 GB | n/a |
| llama.cpp: n=N (c') | 2k | 4 | 6.60 | 1.39× | 0.81 | 310 | 3.55 GB | n/a |
| llama.cpp: n=N (c') | 2k | 8 | 7.92 | 1.67× | 0.82 | 517 | 3.64 GB | n/a |
| llama.cpp: n=N (c') | 8k | 1 | 9.39 | 1.00× | 4.42 | 55 | 4.89 GB | n/a |
| llama.cpp: n=N (c') | 8k | 2 | 10.61 | 1.13× | 4.47 | 97 | 4.50 GB | n/a |
| llama.cpp: n=N (c') | 8k | 4 | 11.51 | 1.23× | 4.45 | 178 | 4.58 GB | n/a |
| llama.cpp: n=N (c') | 8k | 8 | 13.78 | 1.47× | 4.74 | 297 | 4.89 GB | n/a |

## What the numbers say

- **Sharing the prefill turns about 6.3× into about 2.7×** at 8k. Forked (b) prefills an 8k
  prompt once, so time to first token stays near 4.6 s for every N. Without sharing (a) it grows
  to 25 s, because 8 prompts queue for the prefill budget. KV use drops by the same factor: 768
  blocks instead of 4,348 at N=8. Only the last, partially filled prompt block is copied for
  each child.
- **Decoding is what's left.** At N=8 the 512-token decode is over 80% of (b)'s wall time,
  and a batch-8 decode step costs about 4× a batch-1 step (next section).
- **rlis's prefix cache gets the same result here, by accident.** (a′) matches (b) within
  noise because an 8k prompt fills several 2,048-token steps: the other 7 identical requests
  wait in the queue and adopt the first one's blocks once they are computed. That depends on
  timing. In a 2k (a′) run from before the decode fix
  (`fork-rlis-independent-apc-before-decode-fix.json`), the prompt was 2,047 tokens, one under
  the step budget. A second request was admitted in the same step as the first, before
  anything was cached, and prefilled the whole prompt again (518 blocks instead of 384, mean
  time to first token 2.1 s instead of 0.9 s). A larger `--max-batched-tokens` would make that
  the common case. Fork shares every time, takes one request instead of N, and reserves seats
  for its children at admission.
- **llama.cpp with N separate requests doesn't share at all** (time to first token grows with
  N). Its `n=N` path does share, and it is the fastest series here.

## Where the time goes

Per-token decode times derived from the table, as (wall − mean time to first token) / 511:

| | 1 sequence, 2k context | 1 sequence, 8k context | 8 sequences, 8k context |
|---|---|---|---|
| rlis (b) | 7.7 ms | 10.7 ms | 44.5 ms per step |
| llama.cpp (c′) | 7.7 ms | 9.7 ms | 17.7 ms per step |

A single sequence now costs the same on both. At 8 sequences rlis's step is 2.5× llama.cpp's.
Timing the attention op on its own (`cargo run --release -p model --features metal --example
attn_bench`) splits rlis's 8k batch-8 step of about 40 ms into roughly half and half:

- **Attention: about 22 ms** for 16 layers, reading K/V at about 97 GB/s out of the M4 Pro's
  roughly 273 GB/s.
- **Everything else: about 18–20 ms.** That is the matmuls at 8 rows, which costs the same at
  a 400-token context (19.7 ms per step). DESIGN.md covers this: candle's batched quantized
  matmul costs about the same for 2 rows as for 32, and the custom GEMV kernel is only used
  for 2–5 rows.

So reaching llama.cpp's N=8 speed needs both a faster batch-8 attention and a faster 8-row
matmul. Neither is about context length any more.

## Long-context decode fix

Before the fix, rlis decoded one sequence at 8k context in 35.9 ms per token, against 9.7 ms
for llama.cpp. Each 1k tokens of context added about 3.6 ms, against llama.cpp's 0.3. The
paged decode kernel ran one threadgroup per (KV head, sequence), which is 8 threadgroups for
a single sequence on a 20-core GPU, each sweeping the whole context. Every position also paid
for four cross-lane reductions and eight exponentials in a serial chain. Two changes fixed it
(`crates/model/src/paged_attention.metal`):

- **Split contexts (flash-decoding).** When the batch has too few (KV head, sequence) pairs to
  fill the GPU, each context is cut into partitions of at least 256 positions. Each partition
  runs as its own threadgroup and writes its max score, softmax denominator and unnormalized
  output; a second kernel merges them. Big batches and short contexts keep the single pass.
- **One position per lane.** Within a block of 32 positions, each lane scores its own position
  against all 4 query heads of the GQA group. The query lives in threadgroup memory and K rows
  are read 16 bytes at a time. The online softmax is updated once per block instead of once
  per position, and then each lane accumulates its slice of V.

Decode step times (`decode_profile`), old and new kernels run back to back:

| context | batch 1 | batch 2 | batch 8 |
|---|---|---|---|
| 400 | 8.0 → 7.0 ms | 10.3 → 9.4 ms | 20.7 → 19.7 ms |
| 2,048 | 13.7 → 7.5 ms | 15.8 → 10.7 ms | 31.6 → 23.9 ms |
| 8,192 | 34.3 → 9.9 ms | 36.3 → 14.8 ms | 72.4 → 40.0 ms |

Before the fix, the fork series looked better on cost ratio (1.83× at 8k) only because a
single sequence was so slow. The end-to-end before/after data is in
`results/mac/fork-rlis-*-before-decode-fix.json`. Two other things I tried made no difference
and were dropped: unrolling the V loads, and shrinking threadgroup memory from 16 KB to 4 KB
for occupancy.

**Correctness.**

- The kernels match a CPU reference for both Llama 3.2 head layouts in f32 and f16, including
  split contexts up to 9,000 tokens, partitions left empty by short sequences, and the
  original 37-sequence case. With the cross-partition weighting deliberately broken, the split
  test fails.
- `make parity`'s f16 gate passes, including the paged batch of 32.
- Greedy output with the new kernel is identical to the gather path (v1) for 128 tokens on
  real 1.8k- and 7.2k-token prompts, including an `n=2` fork.
- The f32 parity gate wasn't run: it loads about 5 GB of f32 weights, over this work's memory
  budget.

## Caveats

- **Memory budget:** every rlis run stayed at or below 4.59 GB of footprint. llama.cpp's (c)
  series did not: at 8k its footprint reached 5.53 GB at N=4 and 6.39 GB at N=8, despite the
  1 GiB cap on its prompt store. Future llama.cpp runs should use `-cram 0`.
- **Footprint mostly reflects preallocation:** both servers allocate their whole KV budget up
  front, so footprint barely moves with N. KV blocks in use is the usage measure.
- **A memory bug found along the way:** candle's Metal allocator rounds each new buffer up to
  a power of two and zero-fills all of it. A 2.25 GiB pool built with `Tensor::zeros` on the
  device took 4 GiB (5.5 GB idle footprint). The pool is now zeroed on the CPU and uploaded,
  which allocates the exact size (3.7 GB idle).
- **Power modes:**
  - Low Power Mode was on for part of the first sweep and roughly halved GPU speed for both
    engines. The affected series were rerun with it off; the throttled data is kept as
    `results/mac/fork-*-lowpower.json`.
  - The machine switched from Automatic to High Power during the post-fix rlis rerun. It
    doesn't change these numbers: the old kernel measures the same in High Power (34.3 ms per
    step at 8k) as it did in Automatic (34.4 ms), and the rerun's per-token times match the
    kernel benchmark.
  - llama.cpp's numbers were measured in Automatic.
