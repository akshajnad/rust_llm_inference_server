#!/usr/bin/env bash
# Charts (docs/*.svg) and tables (results/{mac,cuda}/tables.md) from the result files.
set -euo pipefail
cd "$(dirname "$0")/../.."
B=target/release/rlis-bench
R=results/mac
CMP=("$R/rlis-q8_0.json" "$R/llamacpp-q8_0.json")

$B chart --inputs "${CMP[@]}" --x concurrency --y throughput_tok_s \
  --title "Output throughput vs concurrency (Llama 3.2 1B Q8_0, M4 Pro)" --out docs/throughput.svg
$B chart --inputs "${CMP[@]}" --x throughput_tok_s --y tpot_p99 \
  --title "p99 time per output token vs throughput" --out docs/latency.svg
$B chart --inputs "${CMP[@]}" --x concurrency --y footprint_peak_gb \
  --title "Peak memory footprint vs concurrency" --out docs/memory.svg
$B chart --inputs "$R/ablation-baseline.json" "$R/ablation-static.json" --x concurrency --y ttft_p99 \
  --title "Continuous vs static batching: p99 time to first token" --out docs/ablation-batching.svg

{
  echo "## rlis vs llama.cpp"; echo
  $B table --inputs "${CMP[@]}"
  echo; echo "## Ablations"; echo
  $B table --inputs "$R"/ablation-{baseline,static,block8,block32,attn-v1,no-gemv}.json
  echo; echo "## Prefix caching (shared ~170-token system prompt)"; echo
  $B table --inputs "$R"/ablation-shared-prefix-{off,on}.json
} > "$R/tables.md"
echo "wrote docs/*.svg and $R/tables.md"

# Cost of N completions of one prompt (make bench-fork), when its results are present.
F=("$R"/fork-rlis-independent.json "$R"/fork-rlis-n.json "$R"/fork-rlis-independent-apc.json
   "$R"/fork-llamacpp-independent.json "$R"/fork-llamacpp-n.json)
if [ -f "${F[0]}" ]; then
  for p in 2048 8192; do
    k=$((p / 1024))
    $B fork-chart --inputs "${F[@]}" --prompt-tokens $p --out docs/fix-cost-${k}k.svg \
      --title "Cost of N completions vs 1, ${k}k-token prompt, 512 tokens each (M4 Pro)"
    $B fork-chart --inputs "${F[@]}" --prompt-tokens $p --metric wall --out docs/fix-wall-${k}k.svg \
      --title "Wall time for N completions, ${k}k-token prompt, 512 tokens each (M4 Pro)"
  done
  $B fork-table --inputs "${F[@]}" > "$R/fork-table.md"
  echo "wrote docs/fix-cost-*.svg and $R/fork-table.md"
fi

# The CUDA run (make bench-cuda), when its results are present.
C=results/cuda
CCMP=("$C/rlis-q8_0.json" "$C/llamacpp-q8_0.json")
if [ -f "${CCMP[0]}" ] && [ -f "${CCMP[1]}" ]; then
  $B chart --inputs "${CCMP[@]}" --x concurrency --y throughput_tok_s \
    --title "Output throughput vs concurrency (Llama 3.2 1B Q8_0, Tesla T4)" --out docs/cuda-throughput.svg
  $B chart --inputs "${CCMP[@]}" --x throughput_tok_s --y tpot_p99 \
    --title "p99 time per output token vs throughput (Tesla T4)" --out docs/cuda-latency.svg
  { echo "## rlis vs llama.cpp (CUDA)"; echo; $B table --inputs "${CCMP[@]}"; } > "$C/tables.md"
  echo "wrote docs/cuda-*.svg and $C/tables.md"
fi
