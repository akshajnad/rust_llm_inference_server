## rlis vs llama.cpp

| configuration | concurrency | tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | ITL p99 (ms) | peak memory |
|---|---|---|---|---|---|---|
| rlis (Q8_0) | 1 | 120 | 77 / 175 | 7.8 / 8.2 | 9 | 2.71 GB |
| rlis (Q8_0) | 4 | 214 | 105 / 481 | 16.9 / 19.7 | 76 | 2.84 GB |
| rlis (Q8_0) | 16 | 361 | 266 / 1654 | 35.8 / 96.2 | 298 | 3.08 GB |
| rlis (Q8_0) | 32 | 524 | 265 / 3632 | 54.2 / 81.6 | 341 | 3.10 GB |
| rlis (Q8_0) | 64 | 559 | 381 / 8868 | 104.5 / 215.7 | 589 | 3.16 GB |
| llama.cpp (Q8_0) | 1 | 144 | 73 / 190 | 6.4 / 6.7 | 7 | 1.64 GB |
| llama.cpp (Q8_0) | 4 | 388 | 115 / 391 | 8.7 / 11.1 | 63 | 1.68 GB |
| llama.cpp (Q8_0) | 16 | 470 | 209 / 1297 | 27.0 / 44.5 | 256 | 2.19 GB |
| llama.cpp (Q8_0) | 32 | 516 | 200 / 2945 | 57.2 / 73.1 | 318 | 3.02 GB |
| llama.cpp (Q8_0) | 64 | 467 | 332 / 7254 | 121.5 / 159.9 | 699 | 4.45 GB |

## Ablations

| configuration | concurrency | tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | ITL p99 (ms) | peak memory |
|---|---|---|---|---|---|---|
| baseline (continuous, block 16, attn v2) | 4 | 214 | 105 / 491 | 16.9 / 19.7 | 76 | 2.82 GB |
| baseline (continuous, block 16, attn v2) | 16 | 373 | 251 / 1580 | 34.2 / 56.8 | 284 | 3.06 GB |
| baseline (continuous, block 16, attn v2) | 32 | 540 | 249 / 3493 | 51.6 / 112.5 | 319 | 3.08 GB |
| static batching | 4 | 158 | 2746 / 5280 | 12.2 / 13.9 | 14 | 2.82 GB |
| static batching | 16 | 244 | 5546 / 10206 | 24.3 / 60.8 | 24 | 3.08 GB |
| static batching | 32 | 374 | 7612 / 13908 | 35.7 / 113.0 | 887 | 3.11 GB |
| block size 8 | 4 | 212 | 107 / 495 | 17.2 / 19.9 | 76 | 2.83 GB |
| block size 8 | 16 | 369 | 259 / 1581 | 34.8 / 57.0 | 284 | 3.06 GB |
| block size 8 | 32 | 537 | 264 / 3677 | 52.0 / 78.8 | 347 | 3.09 GB |
| block size 32 | 4 | 214 | 104 / 492 | 16.9 / 19.7 | 76 | 2.82 GB |
| block size 32 | 16 | 373 | 308 / 1612 | 34.9 / 74.6 | 283 | 3.06 GB |
| block size 32 | 32 | 538 | 261 / 3642 | 52.1 / 75.1 | 361 | 3.09 GB |
| attention v1 (gather) | 4 | 132 | 115 / 453 | 28.6 / 30.3 | 78 | 2.82 GB |
| attention v1 (gather) | 16 | 156 | 328 / 1531 | 86.6 / 112.7 | 290 | 3.08 GB |
| attention v1 (gather) | 32 | 186 | 458 / 3374 | 162.5 / 183.1 | 429 | 3.22 GB |
| no small-batch GEMV | 4 | 157 | 109 / 491 | 23.7 / 26.6 | 76 | 2.83 GB |
| no small-batch GEMV | 16 | 363 | 240 / 1574 | 34.3 / 49.6 | 270 | 3.07 GB |
| no small-batch GEMV | 32 | 533 | 259 / 3506 | 51.7 / 112.4 | 327 | 3.09 GB |

## Prefix caching (shared ~170-token system prompt)

| configuration | concurrency | tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | ITL p99 (ms) | peak memory |
|---|---|---|---|---|---|---|
| shared system prompt, prefix caching off | 4 | 192 | 173 / 736 | 18.4 / 22.7 | 142 | 3.03 GB |
| shared system prompt, prefix caching off | 16 | 305 | 326 / 2655 | 44.4 / 136.7 | 372 | 3.06 GB |
| shared system prompt, prefix caching off | 32 | 417 | 410 / 5536 | 67.3 / 129.6 | 506 | 3.09 GB |
| shared system prompt, prefix caching on | 4 | 204 | 120 / 526 | 17.7 / 20.8 | 90 | 2.83 GB |
| shared system prompt, prefix caching on | 16 | 372 | 165 / 497 | 35.9 / 51.7 | 274 | 2.84 GB |
| shared system prompt, prefix caching on | 32 | 634 | 172 / 530 | 43.8 / 62.7 | 290 | 3.08 GB |
