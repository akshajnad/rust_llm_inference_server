# rlis — a Rust LLM inference server for Apple Silicon

An LLM serving engine in Rust with continuous batching, a paged KV cache, custom Metal kernels
and an OpenAI-compatible streaming API. It runs Llama 3.2 1B on a MacBook's GPU inside a fixed
~3 GB memory budget, matches Hugging Face transformers token-for-token, and out-serves
llama.cpp's server at high concurrency.

![Streaming output, recorded with real timing](docs/streaming.svg)

## Quick start

```sh
make serve          # downloads the model (1.3 GB, into ./models), builds, serves on :8000
```

```sh
curl -N http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Why is the sky blue?"}],"stream":true}'
```

Needs Rust (`rustup`) and an Apple Silicon Mac. Any OpenAI client works against
`http://127.0.0.1:8000/v1`. `make teardown` removes everything the project downloaded or built.

## Benchmarks

Same model (Llama 3.2 1B Instruct, GGUF Q8_0), same 32,768-token f16 KV budget, same
2,048-token batch, prompt caching off on both, on an M4 Pro (20-core GPU, 24 GB, macOS 26.6).
256 fixed prompts cut from WikiText-2 with log-normal lengths (mean 295 input / 159 output
tokens), each generating exactly its target length. llama.cpp is build `b11936` with Metal,
flash attention and a unified KV cache.

![Throughput vs concurrency](docs/throughput.svg)
![p99 time per output token vs throughput](docs/latency.svg)
![Peak memory footprint vs concurrency](docs/memory.svg)

| Concurrent requests | rlis tok/s | llama.cpp tok/s | rlis ITL p50 | llama.cpp ITL p50 | rlis peak memory | llama.cpp peak memory |
|---|---|---|---|---|---|---|
| 1 | 120 | **144** | 7.8 ms | **6.4 ms** | 2.71 GB | 1.64 GB |
| 4 | 214 | **388** | 16.0 ms | **7.8 ms** | 2.84 GB | 1.68 GB |
| 16 | 361 | **470** | 24.5 ms | **17.8 ms** | 3.08 GB | 2.19 GB |
| 32 | **524** | 516 | **30.2 ms** | 36.1 ms | 3.10 GB | 3.02 GB |
| 64 | **559** | 467 | **52.4 ms** | 83.9 ms | 3.16 GB | 4.45 GB |

llama.cpp is faster up to 16 concurrent requests: its hand-tuned quantized kernels prefill
~20% faster and decode small batches faster than candle's (the profile is in
[DESIGN.md](DESIGN.md#where-the-time-goes)). rlis pulls ahead at 32–64, where scheduling
dominates: +20% throughput and 38% lower median inter-token latency at 64. Its KV pool is
reserved up front, so memory stays flat at ~3.1 GB while llama.cpp's grows to 4.5 GB. Full
percentiles and provenance: [`results/mac/tables.md`](results/mac/tables.md).

Reproduce: `make llama-cpp bench-mac bench-charts`.

### Ablations

One change at a time against the default configuration (concurrency 4 / 16 / 32):

| Change | Throughput (tok/s) | p99 time to first token |
|---|---|---|
| **default** (continuous batching, 16-token blocks, paged-attention kernel, GEMV) | 214 / 373 / 540 | 0.49 / 1.6 / 3.5 s |
| static batching | 158 / 244 / 374 | 5.3 / 10.2 / 13.9 s |
| attention v1 (gather blocks, then attention) | 132 / 156 / 186 | 0.45 / 1.5 / 3.4 s |
| no small-batch GEMV | 157 / 363 / 533 | 0.49 / 1.6 / 3.5 s |
| block size 8 / 32 | within 1% of default | within noise |
| shared system prompt, prefix caching off | 192 / 305 / 417 | 0.74 / 2.7 / 5.5 s |
| shared system prompt, prefix caching **on** | 204 / 372 / **634** | 0.53 / 0.50 / **0.53 s** |

![Continuous vs static batching](docs/ablation-batching.svg)

### Quantization

| Format | WikiText-2 perplexity | Top-1 agreement with f32 reference | Decode, 1 stream |
|---|---|---|---|
| f16 | 14.0009 ± 0.104 | exact | 83 tok/s |
| **Q8_0** (default) | 14.0035 ± 0.104 | 99.38% | 145 tok/s |
| Q4_K_M | 14.4563 ± 0.108 | 94.45% | 196 tok/s |

## What's inside

- **Continuous batching** with chunked prefill, FCFS admission, and preemption by recompute
  (on unified memory, swapping to "host" frees nothing).
- **Paged KV cache**: fixed pool from `--kv-cache-gb`, 16-token blocks, refcounts with
  copy-on-write, and automatic prefix caching with LRU eviction.
- **Metal kernels**: paged decode attention that reads block tables directly, a small-batch
  Q8_0 GEMV, and GPU sampling (Gumbel-max with per-request counter-based noise).
- **Correctness first**: token-for-token parity with Hugging Face transformers (single and in a
  paged batch of 32), property tests for the allocator, a scheduler simulator, fuzzing of the
  request parser and stop strings, and a soak test.
- **Observability**: Prometheus `/metrics` (TTFT, ITL, throughput, queue, batch, preemptions,
  KV utilization, prefix-cache hits, memory footprint) and opt-in per-stage GPU timing
  (`RLIS_PROFILE=1`).

How it works and why: [DESIGN.md](DESIGN.md). The hardest bug:
[The 0.6 GB that wasn't in the pool](docs/blog-the-memory-that-wasnt-in-the-pool.md).

## Memory and disk

| | Measured |
|---|---|
| Default run (Q8_0, 1 GiB KV pool), benchmark load | 2.7–3.2 GB peak footprint |
| Default run, 64 long requests overfilling the pool | 3.26 GB peak footprint |
| Model files (`make models` / `make models-all`) | 1.3 GB / 4.6 GB |
| Rust build (`./target`) | ~3–4 GB |
| llama.cpp build (`./third_party`) | < 1 GB |

`make teardown` stops project processes and deletes `models/`, `target/`, `third_party/` and
any virtualenv, leaving only code and results.

## Development

```sh
make test               # CPU tests (what CI runs)
make test-metal         # GPU kernel tests
make models-all parity  # parity against transformers (reference fixtures are committed)
make lint               # fmt + clippy, with and without Metal
make soak               # one hour at concurrency 32 with random client hang-ups
```
