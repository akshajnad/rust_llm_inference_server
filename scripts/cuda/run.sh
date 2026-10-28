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
# llama.cpp's CMake needs the same library for its CUDA::cuda_driver target.
# Look where the loader will find it at runtime first, not at copies other tools bundle.
LIBCUDA=$(ldconfig -p | awk '/libcuda\.so\.1 /{print $NF; exit}')
for d in ${LD_LIBRARY_PATH//:/ } /usr/local/nvidia/lib64 /usr/lib64-nvidia /usr/lib/x86_64-linux-gnu; do
  [ -n "$LIBCUDA" ] || { [ -e "$d/libcuda.so.1" ] && LIBCUDA=$d/libcuda.so.1; }
done
[ -n "$LIBCUDA" ] || LIBCUDA=$(find / -name 'libcuda.so.1' -not -path '/proc/*' 2>/dev/null | head -1)
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
# Kaggle sets UV_* variables that send `uv pip install` to the system Python instead of the
# Makefile's throwaway venv; clear them (a subshell, so only for this step). One compile job per
# CPU: unlimited jobs ran the session out of memory, and the OOM-killed build stalled for hours,
# so the step is also capped.
build_llama_cpp() (
  for v in $(compgen -e | grep '^UV_'); do unset "$v"; done
  { command -v uv >/dev/null || pip install -q uv; } &&
    timeout 75m make llama-cpp LLAMA_CPP_JOBS="$(nproc)" \
      LLAMA_CPP_CMAKE_FLAGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CUDA_COMPUTE_CAP \
      -DCUDA_cuda_driver_LIBRARY=$CUDA_LINK/libcuda.so"
)

step rust install_rust
step models make models-all
step build-rlis build_rlis
# llama.cpp compiles on the CPUs while parity runs on the GPU (free GPU time is rationed); only
# the sweep needs both, and nothing is recorded unless parity passes.
build_llama_cpp >"$OUT/llama-cpp-build.log" 2>&1 &
LLAMA_CPP_BUILD=$!
if ! step parity parity; then
  session failed
  echo "parity failed: recording nothing" >&2
  exit 1
fi
step build-llama-cpp wait "$LLAMA_CPP_BUILD" || {
  tail -40 "$OUT/llama-cpp-build.log"
  session passed-but-llama-cpp-build-failed
  exit 1
}
# The spec's sweep is 1/4/16/32, with 64 only if it fits; free GPU time is about 6 hours a week,
# so the cloud run stops at 32.
step sweep env LEVELS="${LEVELS:-1,4,16,32}" scripts/bench/sweep.sh
session passed
