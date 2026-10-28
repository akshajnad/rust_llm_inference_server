#!/usr/bin/env bash
# Soak: the default server (prefix caching on) at concurrency 32 with random client hang-ups.
# Passes when every KV block returns once idle and memory stays flat. Usage: soak.sh [seconds]
set -euo pipefail
cd "$(dirname "$0")/../.."
SECS=${1:-3600}
PORT=${PORT:-18400}
target/release/rlis serve --port "$PORT" >results/mac/soak.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
for _ in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 0.5; done
target/release/rlis-bench soak --url "http://127.0.0.1:$PORT" --pid "$PID" --duration-s "$SECS" \
  --concurrency 32 --hang-up-prob 0.25 --out results/mac/soak.json
