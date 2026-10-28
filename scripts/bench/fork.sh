#!/usr/bin/env bash
# Cost of N completions of one prompt (rlis-bench fork): rlis with N independent requests
# (prefix caching off and on), rlis with one n=N request (forked from one prefill), and
# llama.cpp's server both ways. Writes $OUT/fork-*.json (default results/mac).
# Usage: scripts/bench/fork.sh [extra rlis-bench fork args, e.g. --reps 1 --n 1,8]
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL=models/Llama-3.2-1B-Instruct-Q8_0.gguf
PORT=${PORT:-18200}
OUT=${OUT:-results/mac}
mkdir -p "$OUT"
BENCH=target/release/rlis-bench
# Same KV budget for both servers: 73,728 tokens (2.25 GiB of f16 K/V), enough for 8
# unshared 8k-token prompts plus 512 generated tokens each, so nothing is ever preempted.
KV_TOKENS=73728

wait_ready() {
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 0.5
  done
  echo "server did not become ready" >&2
  return 1
}

PID=
trap '[ -n "$PID" ] && kill $PID 2>/dev/null || true' EXIT
stop() { kill -TERM "$PID"; wait "$PID" 2>/dev/null || true; PID=; }

run() { # name label engine mode
  "$BENCH" fork --url "http://127.0.0.1:$PORT" --pid "$PID" --label "$2" --engine "$3" \
    --mode "$4" --out "$OUT/$1.json" "${@:5}"
}

rlis() { # log-name extra-serve-args...
  target/release/rlis serve --port "$PORT" --model "$MODEL" --kv-cache-gb 2.25 \
    --max-seq-len 9216 --max-num-seqs 64 --max-batched-tokens 2048 "${@:2}" >"$OUT/$1.log" 2>&1 &
  PID=$!
  wait_ready
}

rlis fork-rlis-off --no-prefix-caching
run fork-rlis-independent "rlis: N requests, no sharing (a)" rlis independent "$@"
run fork-rlis-n "rlis: n=N, forked from one prefill (b)" rlis n "$@"
stop

rlis fork-rlis-on
run fork-rlis-independent-apc "rlis: N requests, prefix caching (a')" rlis independent "$@"
stop

# llama.cpp: 8 slots over one unified KV buffer, flash attention, logical batch 2048, prompt
# cache on (its default) with the host-RAM prompt store capped at 1 GiB to stay in budget.
third_party/llama.cpp/build/bin/llama-server --port "$PORT" -m "$MODEL" -ngl 99 \
  -c "$KV_TOKENS" -np 8 -kvu -fa on -b 2048 -ub 512 --cache-prompt -cram 1024 \
  >"$OUT/fork-llamacpp.log" 2>&1 &
PID=$!
wait_ready
run fork-llamacpp-independent "llama.cpp: N requests, prompt cache (c)" llama.cpp independent "$@"
run fork-llamacpp-n "llama.cpp: n=N (c')" llama.cpp n "$@"
stop
trap - EXIT
