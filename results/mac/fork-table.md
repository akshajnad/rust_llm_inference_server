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
