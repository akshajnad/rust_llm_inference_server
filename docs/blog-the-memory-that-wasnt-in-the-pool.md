# The 0.6 GB that wasn't in the pool

My Rust LLM server has a memory budget: about 3.5 GB for a 1B model on a 24 GB MacBook, where
the GPU shares RAM with everything else. Under ordinary load it peaked at 2.55 GB. Under the
nastiest load I could build — 64 concurrent long requests that overfill the KV cache and force
constant preemption — it peaked at 3.58 GB. Two percent over. This is how I found the missing
memory, and why my first fix was for the wrong problem.

## The obvious suspect

Weights (1.3 GB) plus the KV pool (1.07 GB) are 2.4 GB, so about 1.2 GB was transient.
`footprint`, macOS's view of a process's memory, broke it down: 2.3 GB of GPU allocations, a
few hundred MB of malloc, and **~650 MB of "Owned physical footprint (unmapped) (graphics)"** —
GPU memory the process no longer maps but the driver still charges to it.

candle, the tensor library underneath, keeps a pool of Metal buffers and drops every unused one
at each GPU sync. My engine syncs once per step. So: each step frees its activation buffers, the
next step allocates fresh ones, and the driver holds onto the freed pages. Churn. The fix seemed
obvious — keep the pool, reuse buffers across steps.

## The experiment that said no

Patching candle meant vendoring it, so before designing a real patch I ran the cheap version:
an environment variable that makes `drop_unused_buffers` a no-op, then the same stress load.

```
default pool:   peak 3.58 GB   unmapped (idle) 113 MB
pool retained:  peak 3.68 GB   unmapped (idle) 590 MB
```

Nothing was ever freed, and the unmapped memory was still there. Worse, actually. Whatever was
churning, it wasn't candle's pool. I deleted the vendored copy.

## What else allocates a buffer every step?

Host↔GPU transfers. Reading a tensor back to the CPU goes through a freshly allocated staging
buffer that is released afterwards. Greedy decoding reads back one token id per sequence. But
the stress load used temperature sampling, and my sampler ran on the CPU — so every step read
back every sampled row's logits: 64 rows × 128,256 vocabulary × 4 bytes = **33 MB of staging
buffer per step**, allocated and freed dozens of times a second.

Same load, greedy instead of temperature: peak 3.21 GB, unmapped 97 MB. That was it.

## The fix: don't move the logits

Shrinking the copy (f16, top-k on the GPU) would have helped. Not making it was better.
Temperature sampling is exactly the Gumbel-max trick: argmax over `logit / T + g`, with `g`
Gumbel noise, is an exact draw from `softmax(logit / T)`. An argmax is a reduction the GPU is
good at, so one kernel picks a token for every row — greedy rows just skip the noise — and only
the token ids come back.

The noise needed care. A stateful RNG would make a request's output depend on which other
requests happen to share its batch. Instead the noise is a counter-based hash of
`(request seed, position, token id)`: every draw is a pure function of the request itself.
Seeded requests reproduce exactly, alone or in a batch of 64, and after preemption and
recompute. A CPU reference implementation of the same hash lets the tests check that the Metal
kernel picks the same token on 64 × 128k random logits, and that the empirical distribution
matches the softmax.

Under the same stress load the peak went from 3.58 GB to **3.26 GB**, and the "unmapped" figure
is back to ~100 MB at idle. Rows that use top-k, top-p or a repetition penalty still sample on
the CPU — a later fix made that path O(vocab) instead of sorting all 128k entries per token,
which on its own doubled throughput for those requests.

## What I'd keep

- **Run the cheap experiment before the expensive fix.** Vendoring a dependency to patch its
  allocator would have cost an afternoon and fixed nothing.
- **"Freed" isn't free on unified memory.** RSS doesn't even see GPU allocations; `footprint`
  does, and its categories pointed at the driver rather than my code.
- **The best transfer is the one you don't make.** Moving sampling to where the logits already
  were removed the memory, the latency and the CPU work in one change.
