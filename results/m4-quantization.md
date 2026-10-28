# M4: quantization quality, memory and throughput

Llama 3.2 1B Instruct in three weight formats on the same engine.

**Setup.** Apple M4 Pro (20-core GPU, Metal 4), 24 GB unified memory, macOS 26.6.2; rustc
1.99.1, candle 0.11.0. Code: commit `955ec80` plus the uncommitted M4 working tree (2026-10-27).
Weights: `unsloth/Llama-3.2-1B-Instruct` (bf16 safetensors, run as f16) and
`bartowski/Llama-3.2-1B-Instruct-GGUF` (Q8_0, Q4_K_M). Server defaults: 1 GiB KV pool (16-token
blocks, f16 KV), attention v2, continuous batching, max 64 sequences and 2048 tokens per step.

## Results

| Format | File | WikiText-2 PPL (ctx 512) | ΔPPL vs f16 | Top-1 agreement with f32 HF | ITL p50, 1 stream | tok/s at 32 / 64 concurrent | Footprint after load | Peak footprint (sweep) |
|---|---|---|---|---|---|---|---|---|
| f16 | 2.47 GB | 14.0009 ± 0.104 | – | 100% (exact greedy in f32 parity) | 12.1 ms (83 tok/s) | 627 / 1117 | 2.48 GB* | 3.68 GB |
| **Q8_0** (default) | 1.32 GB | 14.0035 ± 0.104 | +0.02% | 99.38% | 6.9 ms (145 tok/s) | 771 / 1193 | 2.46 GB | **2.55 GB** |
| Q4_K_M | 0.81 GB | 14.4563 ± 0.108 | +3.25% | 94.45% | 5.1 ms (196 tok/s) | 726 / 1119 | 1.97 GB | 2.06 GB |

\* The f16 KV pool's pages are committed lazily, so after-load excludes most of it. The peak
includes it.

Q8_0 costs essentially no quality (+0.02% perplexity, inside the error bar), uses half the weight
memory and decodes 1.75× faster single-stream, because decode is bandwidth-bound. Q4_K_M is
faster still single-stream but loses 3.25% perplexity. Under heavy batching both quantized formats
are compute-bound in candle's quantized matmul, so their advantage over f16 shrinks.

## Methods

- **Perplexity**: `rlis perplexity --model <m>` over `wiki.test.raw` (`make wikitext`, the copy
  llama.cpp uses), with llama.cpp's method: 512-token windows, BOS first, only the second half of
  each window scored. 564 windows and 143,820 scored tokens per format; ± is one standard error.
- **Top-1 agreement**: `gguf_tracks_reference` in `crates/model/tests/parity.rs`. Teacher-forced
  over the 20 parity prompts × 128 steps against the f32 transformers reference, so one
  disagreement doesn't cascade.
- **Throughput and latency**: streaming load sweep, 64/128-token outputs, arrivals staggered over
  1 s. ITL p50 at concurrency 1 is the single-stream decode speed. The c=1 tok/s figure is
  dominated by the stagger and isn't reported. (The M5 bench crate replaces this script.)
- **Footprint**: macOS `phys_footprint` (`footprint` tool, and `/metrics`'
  `rlis_process_memory_footprint_*`), which counts Metal allocations that RSS misses.

## Memory under a full KV pool (fixed in M5)

Stress load: 64 concurrent requests of ~300 prompt + 400 sampled tokens (about 38k tokens
against a 32k-token pool, 14–17 preemptions). At M4, Q8_0 peaked at 3.58 GB, 2% over the
3.5 GB target. The excess was ~650 MB of "Owned physical footprint (unmapped) (graphics)":
temperature sampling read every sampled row's 128k logits back through a fresh 33 MB staging
buffer per step, and the Metal driver kept the freed staging memory charged to the process.
(Retaining candle's buffer pool was tried first and changed nothing.)

M5 samples greedy and temperature-only rows on the GPU (Gumbel-max with per-request,
batch-independent noise), so only token ids come back. Same load, same seeds: peak **3.26 GB**,
and ~100 MB unmapped at idle. The story is written up in
`docs/blog-the-memory-that-wasnt-in-the-pool.md`.
