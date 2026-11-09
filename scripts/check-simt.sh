#!/usr/bin/env bash
# Compile-check the cuda-oxide SIMT kernels without a GPU:
#   1. build crates/s2s-simt through the cuda-oxide rustc backend,
#   2. dump the generated PTX (`cargo oxide inspect`),
#   3. assemble it with ptxas for every target architecture.
#
# Needs: Rust 1.99.1 (rust-src, rustc-dev, llvm-tools), cargo-oxide, and a CUDA
# tree with ptxas (see scripts/fetch-cuda-wheels.sh) in CUDA_TOOLKIT_PATH.
set -euo pipefail

: "${CUDA_TOOLKIT_PATH:?set CUDA_TOOLKIT_PATH (see scripts/fetch-cuda-wheels.sh)}"
ARCHS=(sm_80 sm_89 sm_90 sm_100)
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

cd "$HERE/crates/s2s-simt"
cargo oxide inspect > "$OUT/inspect.txt"

# The PTX follows the "PTX (<name>.ptx)" banner and its separator line.
awk 'f>=2{print} /^PTX \(/{f=1;next} f==1&&/^=+$/{f=2}' "$OUT/inspect.txt" > "$OUT/s2s_simt.ptx"

for kernel in apply_token_bitmask embedding_sum_fwd embedding_sum_bwd apply_delays; do
  grep -q "\.entry ${kernel}(" "$OUT/s2s_simt.ptx" || { echo "missing kernel: $kernel"; exit 1; }
done

for arch in "${ARCHS[@]}"; do
  "$CUDA_TOOLKIT_PATH/bin/ptxas" -arch="$arch" -o "$OUT/s2s_simt_${arch}.cubin" "$OUT/s2s_simt.ptx"
  echo "ptxas $arch: ok ($(stat -c %s "$OUT/s2s_simt_${arch}.cubin") bytes)"
done
