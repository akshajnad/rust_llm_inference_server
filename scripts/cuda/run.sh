#!/usr/bin/env bash
# The whole CUDA session (M6), run from a clone of the pinned commit on a cloud NVIDIA GPU:
# toolchain, build, models, reference parity, then the same rlis vs llama.cpp sweep as on the
# Mac. Parity gates the sweep: if it fails, no benchmark is recorded. Writes $OUT (default
# results/cuda): parity log, sweep JSON and logs, and session.json with timings and versions.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OUT=${OUT:-results/cuda}
mkdir -p "$OUT"
# Both engines on one GPU (Kaggle's T4 sessions have two; llama.cpp would split across them).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CC=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=compute_cap --format=csv,noheader | tr -d .)
export CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP:-$CC}
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
# Containers mount the driver's libcuda.so.1 but not the unversioned libcuda.so that -lcuda
# looks for (and Kaggle's toolkit has no stub), so give the linker a symlink to the real one.
LIBCUDA=$(ldconfig -p | awk '/libcuda\.so\.1 /{print $NF; exit}')
[ -n "$LIBCUDA" ] || LIBCUDA=$(find / -xdev -name 'libcuda.so.1' 2>/dev/null | head -1)
CUDA_LINK=$(mktemp -d)
ln -s "$LIBCUDA" "$CUDA_LINK/libcuda.so"
echo "linking against $LIBCUDA"
export LIBRARY_PATH=$CUDA_LINK${LIBRARY_PATH:+:$LIBRARY_PATH}

TIMINGS=$OUT/timings.tsv
: >"$TIMINGS"
# Step functions chain with && because set -e is off inside a command that is tested.
step() { # name command...
  local name=$1 start=$SECONDS rc=0
  shift
  echo "== $(date -u +%H:%M:%S) $name"
  "$@" || rc=$?
  printf '%s\t%s\n' "$name" $((SECONDS - start)) >>"$TIMINGS"
  return $rc
}

session() { # parity status
  python3 - "$OUT" "$1" <<'EOF'
import json, subprocess, sys
out, parity = sys.argv[1], sys.argv[2]
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    return r.stdout.strip()
smi = sh("nvidia-smi -i ${CUDA_VISIBLE_DEVICES} --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader").split(", ")
timings = {}
for line in open(f"{out}/timings.tsv"):
    k, v = line.rstrip("\n").split("\t")
    timings[k] = int(v)
json.dump({
    "gpu": smi[0], "driver": smi[1], "gpu_memory": smi[2], "compute_capability": smi[3],
    "cuda_runtime": sh("nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"),
    "nvcc": sh("nvcc --version | tail -1"),
    "commit": sh("git rev-parse HEAD"),
    "rustc": sh("rustc --version"),
    "llama_cpp": sh("git -C third_party/llama.cpp describe --tags 2>/dev/null"),
    "os": sh("uname -srm"), "cpu": sh("nproc") + " vCPU " + sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"),
    "parity": parity, "step_seconds": timings,
}, open(f"{out}/session.json", "w"), indent=2)
EOF
}

install_rust() {
  { command -v rustup >/dev/null ||
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none; } &&
    rustup toolchain install # the version pinned in rust-toolchain.toml
}
# candle-kernels 0.11.0 defines __hmax_nan/__hmin_nan for GPUs below sm_80, but CUDA 12.8's
# headers already define them for every architecture, so nvcc rejects the redefinition. Skip
# candle's copies on those toolkits (same NaN-propagating semantics). Edits this session's
# registry copy only, so Cargo.lock and the checkout stay untouched.
patch_candle() {
  local h
  cargo fetch &&
    h=$(echo "${CARGO_HOME:-$HOME/.cargo}"/registry/src/*/candle-kernels-0.11.0/src/compatibility.cuh) &&
    sed -i 's/^#if __CUDA_ARCH__ < 800$/#if __CUDA_ARCH__ < 800 \&\& (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 \&\& __CUDACC_VER_MINOR__ < 8))/' "$h" &&
    grep -q '__CUDACC_VER_MINOR__ < 8' "$h"
}
build_rlis() {
  patch_candle &&
    cargo build --release --features cuda -p server && cargo build --release -p bench
}
parity() {
  make parity FEATURES=cuda 2>&1 | tee "$OUT/parity.log"
}
build_llama_cpp() {
  { command -v uv >/dev/null || pip install -q uv; } &&
    make llama-cpp LLAMA_CPP_CMAKE_FLAGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CUDA_COMPUTE_CAP"
}

step rust install_rust
step models make models-all
step build-rlis build_rlis
if ! step parity parity; then
  session failed
  echo "parity failed: recording nothing" >&2
  exit 1
fi
step build-llama-cpp build_llama_cpp
step sweep scripts/bench/sweep.sh
session passed
