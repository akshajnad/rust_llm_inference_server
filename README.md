# rlis — a Rust LLM inference server for Apple Silicon

An LLM serving engine in Rust with continuous batching, a paged KV cache, custom Metal kernels
and an OpenAI-compatible streaming API. It runs Llama 3.2 1B on a MacBook's GPU inside a fixed
~3 GB memory budget, matches Hugging Face transformers token-for-token, and out-serves
llama.cpp's server at high concurrency. The same code also builds for NVIDIA GPUs: on a free
Kaggle Tesla T4 it passes the same parity tests and reaches 89% of llama.cpp's single-stream
decode speed.

![Streaming output, recorded with real timing](docs/streaming.svg)

## Quick start

```sh
make chat           # downloads the model (1.3 GB, into ./models), builds, chats in the terminal
```

Needs Rust (`rustup`) and an Apple Silicon Mac. `make teardown` removes everything the project
downloaded or built.

### Terminal chat

`make chat` opens a chat in the terminal. `make install` puts it on your PATH as `rlis` (a link
in `~/.cargo/bin` to this build), so from any directory `rlis` opens the chat and `rlis serve`,
`rlis generate` and `rlis perplexity` run the server binary. `make uninstall` removes the link.
Replies stream in with Markdown styling (headings, bold, code, lists) under a live status
line (prefill, then decoding with tokens and tok/s so far), and each one ends with its token
count, decode speed and time to first token. Each exchange is drawn as a card (your message,
then the reply, with its stats in the bottom edge), and the prompt card's top edge shows the
model, temperature and conversation length. Typing `/` opens a command menu. If no
server answers at `--url` (default `http://127.0.0.1:8000`), it starts `rlis serve` in the
background, warms it up so the first reply is fast, and stops it on exit; an already running
server is used as is. The conversation is kept across turns, and the oldest exchanges are
dropped when it outgrows the 2,048-token context.

| Command | |
|---|---|
| `/clear` | start a new conversation |
| `/retry` | regenerate the last reply |
| `/system <prompt>` | show or set the system prompt (`/system none` removes it) |
| `/temp <x>` | show or set the temperature (default 0.6, top-p 0.9) |
| `/model <name>` | list the models (`/model`), or switch to one on disk: `3b`, `14b` (Qwen2.5-Coder), `k2` (K2 Horizon) or `llama-1b` |
| `/help`, `/exit` | |

| Key | |
|---|---|
| enter | send |
| `\` then enter | new line (shift+enter and option+enter also work in terminals that send them) |
| esc | interrupt a reply (generation stops on the server too), or clear the input |
| ↑ ↓ | previous messages, or move through the `/` menu |
| tab | complete a command |
| ctrl-c twice, or ctrl-d on an empty line | quit |

Options: `--system`, `--temperature`, `--top-p`, `--max-tokens`, `--url`, `--no-start` (fail
instead of starting a server), and `rlis serve` flags after `--` for a server it starts, e.g.
`rlis -- --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --kv-cache-gb 2`. A started
server logs to `target/release/rlis-chat-server.log`.

### `rlis code` (coding agent)

```sh
make models-k2 # K2 Horizon 3.7B, Q4_K_M (3.3 GB); or models-qwen-3b (1.9 GB)
cd path/to/project && rlis code
```

The terminal chat as a coding agent for the current directory: the model reads, creates and
edits files and runs shell commands to do what you ask, and keeps going (reading, editing,
running) until it replies with a summary. It asks in plain text, in formats a 3B model writes
reliably: `READ path` lines (`READ .` lists the directory), `RUN command` lines (or a lone
`RUN` and a fenced script), and a file's path followed by a fenced `SEARCH`/`REPLACE` block (an
edit) or by the whole file (a write; makefile recipes get the tabs make needs). Generation
stops as soon as a request is complete, so the model never makes up a file's contents or a
command's output, and the requests show as cards instead of raw text:

- reads run at once, from the project or (by absolute or `~/` path) anywhere else;
- changes show as a diff that waits for `y` (apply), `n` (decline; the model hears so and goes
  on), `a` (apply this and every later change) or esc (decline and stop). A change outside the
  directory always asks, even after `a` or with `--yes`;
- commands run at once in the project directory, their output streaming into the card (esc
  kills one; `--timeout`, default 120 s). Commands that delete, move or overwrite files, change
  git history, install packages or run with `sudo` ask first, with the same keys.

It may only change files it has read in the conversation (or new ones). Small models slip in
ways the agent corrects: a code block with no path line still creates the new file the line
before it (`### HTML (index.html)`) or its first comment (`# snake.py`) names; and a reply that
shows code without saving it, refuses an ordinary task, or claims to have done something it
didn't gets one reminder of what it can do. Each tool result restates your request, which keeps
a small model on it instead of making up edits nobody asked for. The system prompt lists the
project's files (what git doesn't ignore) and the first of `RLIS.md`, `AGENTS.md` or
`CLAUDE.md` it finds. With no server running it starts one with a 16,384-token context and
K2 Horizon if it is downloaded, else Qwen2.5-Coder 3B; `--model 3b|k2|14b|llama-1b` picks one
(`make models-qwen-14b` downloads the 14B).

K2 thinks before it answers and was trained on native tool calls, so `crates/chat/src/native.rs`
declares the agent's actions as tools, hides the thinking (the status line says `working`) and
turns each call back into the text above; the agent itself doesn't change.

`/model` lists the models and which are on disk; `/model 14b` switches to one mid-session.
It stops the running server, then starts one with the new model, so only one model is in
memory at a time. This takes a few seconds, and the conversation carries over. If the new
model fails to start, the old one comes back. Only a server `rlis` started itself can be
switched; one already running when it started is used as it is. The plain chat has `/model`
and `--model` too. On ten small tasks (`crates/chat/src/probe.rs`: create
a script, a game, a web page, a Rust and a Go program, add to a file, run the tests, fix a
failing test, delete a file, answer a question without changing anything), two runs each on
an M4 Pro (24 GB):

| `/model` | Tasks done | Decode | Peak footprint |
|---|---|---|---|
| `3b` | 15/20 | 61 tok/s | 3.9 GB |
| `k2` (default when downloaded) | 19/20 | 50 tok/s | 6.9 GB (`--kv-cache-gb 2.3`) |
| `14b` | 20/20 | 21 tok/s | 12.9 GB (`--kv-cache-gb 3`) |

The 3B's misses vary from run to run (a page or a program shown but never saved, an edit
left out); K2's miss was one snake game, which ended in an error after 200 s of writing. It writes more than the
others (a task takes it longer despite the faster decode) and needs the probe's replies
uncapped (`RLIS_PROBE_MAX_TOKENS=8000`). The 14B did everything, for about twice K2's memory
and time per token.

`--yes` applies changes and runs commands without asking; the chat's other options and
commands work too (`/clear` starts over). `cargo test --release -p chat probe -- --ignored`
with `RLIS_PROBE_URL` set runs the tasks behind that table against any server.

### `rlis fix` (prototype)

```sh
make models-qwen-3b # Qwen2.5-Coder 3B Instruct, Q4_K_M (1.9 GB)
rlis fix --repo path/to/repo --test "python -m unittest -q" --task "median() is wrong for even lists"
```

Eight attempts at a patch from one request: the task, the failing test's output and the files
it names go to the server once with `n: 8`, so the prompt is prefilled once and the attempts
decode together. Each diff is applied in its own git worktree (under the repository's `.git`,
removed afterwards; by content when `git apply` refuses it) and the test runs there, four at a
time; the smallest passing diff wins.
The working tree isn't touched unless `--apply` is given. With no server running it starts one
with the Qwen 3B model and stops it on exit. Other options: `--n`, `--temperature`, `--seed`,
`--protect <file>` (reject patches that touch it, e.g. the test), `--json <file>`. Results on
ten small buggy Python tasks: [docs/fix-bench.md](docs/fix-bench.md#rlis-fix-on-10-small-tasks).

### Server and API

```sh
make serve          # serves on http://127.0.0.1:8000
```

```sh
curl http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Why is the sky blue?"}]}'
```

With `"stream": true` (and `curl -N`) the reply arrives as server-sent events, one `data:` line
per token. Any OpenAI client works against `http://127.0.0.1:8000/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
stream = client.chat.completions.create(
    model="llama-3.2-1b-instruct",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Endpoints: `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, `GET /health`,
`GET /metrics`. `target/release/rlis serve --help` lists the server's flags (model, KV pool size,
context length, host and port, scheduler).

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

### CUDA (free Kaggle T4)

The same engine code built with `--features cuda`, run in one free Kaggle session on a Tesla T4
(driver 580.178, CUDA 12.8, 4 vCPUs) at commit `d763c10`. The reference parity tests passed on
the T4 first (f32 token-for-token, including the paged batch of 32; f16 within its near-tie
rule), then the same sweep ran against llama.cpp `b11936` built with CUDA, both on one GPU, with
the same flags as on the Mac. Concurrency stops at 32: Kaggle's free GPU time is 6 hours a week.

![Throughput vs concurrency on a Tesla T4](docs/cuda-throughput.svg)

| Concurrent requests | rlis tok/s | llama.cpp tok/s | rlis ITL p50 | llama.cpp ITL p50 | rlis p99 TTFT | llama.cpp p99 TTFT | rlis peak memory | llama.cpp peak memory |
|---|---|---|---|---|---|---|---|---|
| 1 | 118 | **132** | 8.2 ms | **7.2 ms** | **106 ms** | 266 ms | **3.54 GB** | 3.76 GB |
| 4 | 239 | **296** | 15.0 ms | **10.7 ms** | 273 ms | **214 ms** | **3.54 GB** | 3.80 GB |
| 16 | 305 | **439** | 39.4 ms | **20.8 ms** | **0.82 s** | 1.07 s | **3.69 GB** | 4.30 GB |
| 32 | 355 | **426** | 73.3 ms | **43.7 ms** | **2.0 s** | 4.4 s | **3.70 GB** | 5.06 GB |

Single-stream decode reaches 89% of llama.cpp's (the spec's CUDA target is 60%), but unlike on
the Mac llama.cpp stays ahead at every concurrency. rlis on CUDA runs without its four Metal
kernels (paged decode attention, small-batch GEMV, GPU sampling, fused prefill attention) and
falls back to gather-then-attend, candle's generic matmuls and host-side sampling; on the Mac
the gather path alone cost 2.9× at 32 requests. A CUDA port of the paged-attention kernel is the
spec's first stretch goal. Memory on Linux is process RSS plus GPU memory in use: rlis's pool is
reserved up front (3.5–3.7 GB), llama.cpp's grows to 5.1 GB. Details, and what it took to get
a clean run: [DESIGN.md](DESIGN.md#cuda-run). Results and provenance:
[`results/cuda/`](results/cuda/tables.md).

Reproduce: `make bench-cuda` (needs a Kaggle account with phone verification; it pushes the
session pinned to `HEAD`, waits, and downloads the results).

### Targets

The spec's starting targets against what was measured on the Mac (and the T4 for the CUDA row):

| Target | Result | |
|---|---|---|
| Single-stream decode ≥ 70% of llama.cpp | 83% (120 vs 144 tok/s) | met |
| Throughput at 32 concurrent requests ≥ 4× single-stream | 4.4× (524 vs 120 tok/s) | met |
| p99 time to first token at moderate load within 2× of unloaded | 2.7× at 4 concurrent, 9.4× at 16 (llama.cpp: 2.1×, 6.8×) | missed |
| KV utilization under load > 90% with no out-of-memory errors | pool reaches 100%; 18 preemptions, 0 errors (64-request stress test) | met |
| Peak footprint, default run ≤ 3.5 GB | 3.26 GB in the worst case | met |
| Memory flat over a 1-hour soak | 10-minute averages within 2,589–2,645 MB (+2.9% early to late) | met |
| Single-stream decode on CUDA ≥ 60% of llama.cpp | 89% (118 vs 132 tok/s, Tesla T4) | met |

The time-to-first-token miss comes from the workload: every benchmark client sends a ~300-token
prompt the moment its last request finishes, so a new prompt queues behind other prompts'
prefill chunks. Shared prefixes remove most of it (with prefix caching, p99 stays at ~0.5 s
from 4 to 32 concurrent requests, see Ablations); unique prompts need faster prefill.

## What's inside

- **Continuous batching** with chunked prefill, FCFS admission, and preemption by recompute
  (on unified memory, swapping to "host" frees nothing).
- **Paged KV cache**: fixed pool from `--kv-cache-gb`, 16-token blocks, refcounts with
  copy-on-write, and automatic prefix caching with LRU eviction.
- **Metal kernels**: paged decode attention that reads block tables directly, a small-batch
  Q8_0 GEMV, and GPU sampling (Gumbel-max with per-request counter-based noise).
- **Correctness first**: token-for-token parity with Hugging Face transformers (single and in a
  paged batch of 32), property tests for the allocator, a scheduler simulator, fuzzing of the
  request parser and stop strings, and a one-hour soak (13,018 requests with random client
  hang-ups, 0 errors, every KV block returned, memory flat within 3%).
- **CUDA build** (`--features cuda`): portable fallbacks for every Metal kernel; parity and
  the benchmark sweep verified on a Tesla T4 (`make bench-cuda`).
- **Terminal client** (`rlis-chat`): a dependency-light chat UI over the same API, with a
  redrawn-in-place prompt box, streaming Markdown rendering and interrupt-by-disconnect.
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
| Upper-bound run (3B Q4_K_M, 2 GiB pool) | not measured yet; the spec's budget estimate is ~5 GB |
| Model files (`make models` / `make models-all` / `make models-qwen` / `make models-qwen-3b` / `make models-k2` / `make models-qwen-14b`) | 1.3 GB / 4.6 GB / 1.7 GB / 1.9 GB / 3.3 GB / 9.0 GB |
| Rust build (`./target`, release + debug, with and without Metal) | 13 GB |
| llama.cpp build (`./third_party`) | 0.36 GB |

`make teardown` stops project processes and deletes `models/`, `target/`, `third_party/` and
any virtualenv, leaving only code and results. Checked on a fresh clone with the models
downloaded, a release build and a running server: afterwards no project process was left and
the folder went from 2.3 GB to 3.7 MB. Outside the folder only the Rust toolchain remains
(`rustup self uninstall` removes it).

## Development

```sh
make test               # CPU tests (what CI runs)
make test-metal         # GPU kernel tests
make models-all parity  # parity against transformers (reference fixtures are committed)
make lint               # fmt + clippy, with and without Metal
make soak               # one hour at concurrency 32 with random client hang-ups
make bench-cuda         # parity + sweep on a free Kaggle T4 (results/cuda/)
```
