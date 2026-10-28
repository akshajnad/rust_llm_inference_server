#!/usr/bin/env bash
# Runs scripts/cuda/run.sh on a free Kaggle T4, pinned to HEAD (which must be on GitHub), waits
# for it, and downloads the results into results/cuda/. Needs the Kaggle CLI ($KAGGLE) and
# ~/.kaggle/kaggle.json from a phone-verified account (GPU and internet access).
set -euo pipefail
cd "$(dirname "$0")/../.."
KAGGLE=${KAGGLE:-kaggle}
OUT=results/cuda
COMMIT=$(git rev-parse HEAD)
[ -n "$(git branch -r --contains "$COMMIT")" ] || {
  echo "push $COMMIT to GitHub first: the kernel clones it" >&2
  exit 1
}
USER=$(python3 -c 'import json, os; print(json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"])')
KERNEL=$USER/rlis-cuda

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
sed "s/@COMMIT@/$COMMIT/" scripts/cuda/kaggle.py >"$STAGE/rlis_cuda.py"
cat >"$STAGE/kernel-metadata.json" <<EOF
{"id": "$KERNEL", "title": "rlis-cuda", "code_file": "rlis_cuda.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
EOF
"$KAGGLE" kernels push -p "$STAGE" --accelerator NvidiaTeslaT4
echo "running $KERNEL at $COMMIT; https://www.kaggle.com/code/$KERNEL"

# Kaggle caps a session at 12 hours; poll every minute.
while :; do
  sleep 60
  status=$("$KAGGLE" kernels status "$KERNEL" 2>&1 || true)
  echo "$(date +%H:%M:%S) $status"
  case $status in
    *RUNNING* | *QUEUED*) ;;
    *) break ;;
  esac
done

"$KAGGLE" kernels output "$KERNEL" -p "$STAGE/out" -o >/dev/null
mkdir -p "$OUT"
cp -R "$STAGE/out/cuda/." "$OUT/" 2>/dev/null || true
# The kernel log is JSON lines of {stream, data}; keep it as plain text.
python3 -c 'import json, sys; print("".join(e["data"] for e in json.load(open(sys.argv[1]))), end="")' \
  "$STAGE/out/rlis-cuda.log" >"$OUT/kaggle.log"
echo "$status"
ls -l "$OUT"
case $status in *COMPLETE*) ;; *) exit 1 ;; esac
