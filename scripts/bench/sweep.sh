#!/usr/bin/env bash
# Concurrency sweep of rlis and llama.cpp's server on the same model, quantization, KV budget
# and batch size. Writes $OUT/<name>.json (default results/mac). Usage: scripts/bench/sweep.sh [suffix]
set -euo pipefail
cd "$(dirname "$0")/../.."
SUFFIX=${1:-}
MODEL=models/Llama-3.2-1B-Instruct-Q8_0.gguf
PORT=${PORT:-18200}
OUT=${OUT:-results/mac}
mkdir -p "$OUT"
BENCH=target/release/rlis-bench
LEVELS=${LEVELS:-1,4,16,32,64}

wait_ready() {
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 0.5
  done
  echo "server did not become ready" >&2
  return 1
}

run_one() { # name label engine pid
  "$BENCH" run --url "http://127.0.0.1:$PORT" --pid "$4" --label "$2" --engine "$3" \
    --quantization Q8_0 --concurrency "$LEVELS" --out "$OUT/$1$SUFFIX.json"
}

# rlis: 1 GiB KV pool (32768 tokens), up to 64 sequences and 2048 tokens per step; prefix
# caching off like llama.cpp's prompt cache below (the prompt set shares no prefixes anyway).
target/release/rlis serve --port "$PORT" --model "$MODEL" --kv-cache-gb 1 \
  --max-num-seqs 64 --max-batched-tokens 2048 --no-prefix-caching >"$OUT/rlis$SUFFIX.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
wait_ready
run_one rlis-q8_0 "rlis (Q8_0)" rlis "$PID"
kill -TERM "$PID"; wait "$PID" 2>/dev/null || true

# llama.cpp: same model, a unified 32768-token f16 KV cache shared by 64 slots, flash
# attention, logical batch 2048, prompt cache off (rlis has no prefix caching here either).
third_party/llama.cpp/build/bin/llama-server --port "$PORT" -m "$MODEL" -ngl 99 \
  -c 32768 -np 64 -kvu -fa on -b 2048 -ub 512 --no-cache-prompt >"$OUT/llamacpp$SUFFIX.log" 2>&1 &
PID=$!
wait_ready
run_one llamacpp-q8_0 "llama.cpp (Q8_0)" llama.cpp "$PID"
kill -TERM "$PID"; wait "$PID" 2>/dev/null || true
trap - EXIT
