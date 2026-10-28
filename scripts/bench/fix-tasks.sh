#!/usr/bin/env bash
# `rlis fix` on the buggy Python tasks in bench/fix-tasks: each task is copied to
# target/fix-tasks/<name>, made a git repository there (so no repository is nested in rlis),
# and fixed with one n=$N request to a Qwen2.5-Coder server. Writes $OUT/<name>.json per
# task, $OUT/table.md (pass@1, pass@N, times) and the server's peak footprint.
# Usage: scripts/bench/fix-tasks.sh [extra rlis fix args, e.g. --seed 1]
# MODEL, TOKENIZER and NAME pick another model (default Qwen2.5-Coder 1.5B Q8_0).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
PORT=${PORT:-18300}
OUT=${OUT:-results/mac/fix}
N=${N:-8}
MODEL=${MODEL:-models/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf}
TOKENIZER=${TOKENIZER:-models/qwen2.5-coder-1.5b-instruct}
NAME=${NAME:-qwen2.5-coder-1.5b-instruct}
WORK=target/fix-tasks
mkdir -p "$OUT" "$WORK"

PID=
SAMPLER=
cleanup() {
  [ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null || true
  [ -n "$PID" ] && kill "$PID" 2>/dev/null || true
  rm -rf .venv .uv-cache "$WORK"
}
trap cleanup EXIT

# The tasks' tests need only the standard library: a throwaway interpreter, deleted on exit.
uv venv --clear -q .venv --python 3.12
export PATH="$ROOT/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1

pmset -g | grep -i powermode >"$OUT/powermode.txt" || true
target/release/rlis serve --port "$PORT" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --served-model-name "$NAME" --max-seq-len 8192 >"$OUT/server.log" 2>&1 &
PID=$!
for _ in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 0.5
done
# The first request compiles GPU pipelines; keep that out of the first task's time.
curl -sf -H 'Content-Type: application/json' -d '{"prompt": "Hello", "max_tokens": 1}' \
  "http://127.0.0.1:$PORT/v1/completions" >/dev/null

# Peak physical footprint of the server, sampled every 0.5 s.
(
  peak=0
  while kill -0 "$PID" 2>/dev/null; do
    b=$(footprint -f bytes "$PID" 2>/dev/null | sed -n 's/.*Footprint: \([0-9]*\) B.*/\1/p' | head -1)
    if [ -n "$b" ] && [ "$b" -gt "$peak" ]; then
      peak=$b
      echo "$peak" >"$OUT/server-peak-footprint-bytes"
    fi
    sleep 0.5
  done
) &
SAMPLER=$!

field() { python -c 'import json, sys; v = json.load(open(sys.argv[1]))[sys.argv[2]]; print("\n".join(v) if isinstance(v, list) else v)' "$1" "$2"; }

for dir in bench/fix-tasks/*/; do
  name=$(basename "$dir")
  repo="$WORK/$name"
  rm -rf "$repo"
  cp -R "$dir" "$repo"
  rm "$repo/task.json"
  git -C "$repo" init -q
  git -C "$repo" add -A
  git -C "$repo" -c user.name=rlis -c user.email=rlis@localhost commit -qm "buggy"
  protect=()
  while IFS= read -r p; do protect+=(--protect "$p"); done < <(field "$dir/task.json" protect)
  echo "== $name"
  target/release/rlis-chat fix --repo "$repo" --task "$(field "$dir/task.json" task)" \
    --test "$(field "$dir/task.json" test)" "${protect[@]}" --n "$N" \
    --url "http://127.0.0.1:$PORT" --no-start --json "$OUT/$name.json" "$@" | tee "$OUT/$name.log"
done

python -I scripts/bench/fix_table.py "$OUT" | tee "$OUT/table.md"
