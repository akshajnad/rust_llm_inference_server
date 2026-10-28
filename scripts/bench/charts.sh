#!/usr/bin/env bash
# Charts (docs/*.svg) and tables (results/mac/tables.md) from the result files.
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
