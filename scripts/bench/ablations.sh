#!/usr/bin/env bash
# One-at-a-time ablations of rlis on the Q8_0 model, each against the same baseline:
# continuous batching, 16-token blocks, attention v2, Q8_0 GEMV on, prefix caching off.
# Writes results/mac/ablation-<name>.json. Usage: scripts/bench/ablations.sh [name ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL=models/Llama-3.2-1B-Instruct-Q8_0.gguf
PORT=${PORT:-18300}
OUT=results/mac
LEVELS=${LEVELS:-4,16,32}
mkdir -p "$OUT"

wait_ready() {
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 0.5
  done
  echo "server did not become ready" >&2
  return 1
}

# name | label | extra server flags | extra env | extra bench flags
CONFIGS=(
  "baseline|baseline (continuous, block 16, attn v2)|||"
  "static|static batching|--scheduler static||"
  "block8|block size 8|--block-size 8||"
  "block32|block size 32|--block-size 32||"
  "attn-v1|attention v1 (gather)|--attention v1||"
  "no-gemv|no small-batch GEMV||RLIS_DISABLE_GEMV=1|"
  "shared-prefix-off|shared system prompt, prefix caching off|||--shared-prefix bench/system_prompt.txt"
  "shared-prefix-on|shared system prompt, prefix caching on|--prefix-caching||--shared-prefix bench/system_prompt.txt"
)

want=("$@")
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name label flags envs bench_flags <<<"$cfg"
  if [ ${#want[@]} -gt 0 ] && [[ ! " ${want[*]} " =~ " $name " ]]; then continue; fi
  caching="--no-prefix-caching"
  if [[ "$flags" == *--prefix-caching* ]]; then caching=""; flags=${flags//--prefix-caching/}; fi
  # shellcheck disable=SC2086
  env $envs target/release/rlis serve --port "$PORT" --model "$MODEL" $flags $caching \
    >"$OUT/ablation-$name.log" 2>&1 &
  PID=$!
  trap 'kill $PID 2>/dev/null || true' EXIT
  wait_ready
  echo "== $name"
  # shellcheck disable=SC2086
  target/release/rlis-bench run --url "http://127.0.0.1:$PORT" --pid "$PID" --label "$label" \
    --concurrency "$LEVELS" $bench_flags --out "$OUT/ablation-$name.json"
  kill -TERM "$PID"; wait "$PID" 2>/dev/null || true
  trap - EXIT
done
