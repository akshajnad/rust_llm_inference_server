| configuration | prompt | N | wall (s) | cost ratio | mean TTFT (s) | output tok/s | peak footprint | peak KV blocks (tokens) |
|---|---|---|---|---|---|---|---|---|
| rlis: N requests, no sharing (a) | 2k | 1 | 8.39 | 1.00× | 0.90 | 61 | 4.45 GB | 160 (2560) |
| rlis: N requests, no sharing (a) | 2k | 2 | 10.33 | 1.23× | 1.32 | 99 | 4.47 GB | 320 (5120) |
| rlis: N requests, no sharing (a) | 2k | 4 | 15.82 | 1.89× | 3.53 | 129 | 4.48 GB | 640 (10240) |
| rlis: N requests, no sharing (a) | 2k | 8 | 26.10 | 3.11× | 5.70 | 157 | 4.48 GB | 1281 (20496) |
| rlis: N requests, no sharing (a) | 8k | 1 | 22.83 | 1.00× | 4.45 | 22 | 4.53 GB | 545 (8720) |
| rlis: N requests, no sharing (a) | 8k | 2 | 28.49 | 1.25× | 6.99 | 36 | 4.53 GB | 1089 (17424) |
| rlis: N requests, no sharing (a) | 8k | 4 | 40.53 | 1.78× | 11.40 | 51 | 4.54 GB | 2176 (34816) |
| rlis: N requests, no sharing (a) | 8k | 8 | 79.11 | 3.47× | 34.49 | 52 | 4.54 GB | 4348 (69568) |
| rlis: n=N, forked from one prefill (b) | 2k | 1 | 8.91 | 1.00× | 1.13 | 57 | 4.50 GB | 160 (2560) |
| rlis: n=N, forked from one prefill (b) | 2k | 2 | 10.24 | 1.15× | 1.06 | 100 | 4.50 GB | 192 (3072) |
| rlis: n=N, forked from one prefill (b) | 2k | 4 | 13.91 | 1.56× | 1.07 | 147 | 4.50 GB | 256 (4096) |
| rlis: n=N, forked from one prefill (b) | 2k | 8 | 20.65 | 2.32× | 1.11 | 198 | 4.50 GB | 384 (6144) |
| rlis: n=N, forked from one prefill (b) | 8k | 1 | 23.36 | 1.00× | 5.01 | 22 | 4.53 GB | 544 (8704) |
| rlis: n=N, forked from one prefill (b) | 8k | 2 | 23.40 | 1.00× | 3.87 | 44 | 4.53 GB | 577 (9232) |
| rlis: n=N, forked from one prefill (b) | 8k | 4 | 27.67 | 1.18× | 4.30 | 74 | 4.53 GB | 643 (10288) |
| rlis: n=N, forked from one prefill (b) | 8k | 8 | 42.69 | 1.83× | 4.56 | 96 | 4.53 GB | 775 (12400) |
| rlis: N requests, prefix caching (a') | 2k | 1 | 8.28 | 1.00× | 0.86 | 62 | 4.45 GB | 160 (2560) |
| rlis: N requests, prefix caching (a') | 2k | 2 | 9.49 | 1.15× | 0.89 | 108 | 4.45 GB | 193 (3088) |
| rlis: N requests, prefix caching (a') | 2k | 4 | 12.39 | 1.50× | 0.90 | 165 | 4.45 GB | 259 (4144) |
| rlis: N requests, prefix caching (a') | 2k | 8 | 17.98 | 2.17× | 1.01 | 228 | 4.48 GB | 518 (8288) |
| rlis: N requests, prefix caching (a') | 8k | 1 | 22.65 | 1.00× | 4.51 | 23 | 4.52 GB | 544 (8704) |
| rlis: N requests, prefix caching (a') | 8k | 2 | 23.81 | 1.05× | 4.49 | 43 | 4.52 GB | 577 (9232) |
| rlis: N requests, prefix caching (a') | 8k | 4 | 27.18 | 1.20× | 4.77 | 75 | 4.52 GB | 643 (10288) |
| rlis: N requests, prefix caching (a') | 8k | 8 | 43.20 | 1.91× | 5.29 | 95 | 4.52 GB | 902 (14432) |
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
