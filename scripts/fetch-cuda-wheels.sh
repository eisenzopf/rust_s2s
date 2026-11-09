#!/usr/bin/env bash
# Assemble a minimal CUDA 13.5 toolkit tree from NVIDIA's official PyPI wheels.
#
# This is enough to BUILD and COMPILE-CHECK both CUDA Rust kernel models
# without a GPU or a full CUDA install:
#   - cuTile (s2s-kernels --features cuda): headers to build, `tileiras` to
#     assemble Tile IR to cubins;
#   - cuda-oxide (crates/s2s-simt): `ptxas` to assemble PTX, libdevice/libNVVM/
#     nvJitLink for `cargo oxide doctor`.
# Running kernels still needs a real GPU and NVIDIA driver.
#
# Usage: scripts/fetch-cuda-wheels.sh [DEST]   (default: .cuda-wheels)
# Then:  export CUDA_TOOLKIT_PATH="$DEST/nvidia/cu13"
#        export PATH="$CUDA_TOOLKIT_PATH/bin:$PATH" LD_LIBRARY_PATH="$CUDA_TOOLKIT_PATH/lib"
set -euo pipefail

DEST="${1:-.cuda-wheels}"
CUDA_VER="13.5.*"
mkdir -p "$DEST/wheels"

python3 -m pip download --no-deps --quiet --dest "$DEST/wheels" \
  "nvidia-cuda-runtime==${CUDA_VER}" \
  "nvidia-cuda-crt==${CUDA_VER}" \
  "nvidia-cuda-nvcc==${CUDA_VER}" \
  "nvidia-cuda-tileiras==${CUDA_VER}" \
  "nvidia-nvvm==${CUDA_VER}" \
  "nvidia-nvjitlink==${CUDA_VER}" \
  "nvidia-curand==10.*"

for whl in "$DEST"/wheels/*.whl; do
  python3 -I -m zipfile -e "$whl" "$DEST/"
done

ROOT="$DEST/nvidia/cu13"
# Zip extraction drops the executable bit; tileiras invokes ptxas, so both
# must be executable.
chmod +x "$ROOT"/bin/* 2>/dev/null || true
# tileiras looks for libNVVM at the toolkit-standard nvvm/lib64 location.
mkdir -p "$ROOT/nvvm/lib64"
ln -sf "$(cd "$ROOT/lib" && pwd)/libnvvm.so.4" "$ROOT/nvvm/lib64/libnvvm.so"
ln -sf libnvvm.so.4 "$ROOT/lib/libnvvm.so"
ln -sf libnvJitLink.so.13 "$ROOT/lib/libnvJitLink.so"

test -f "$ROOT/include/cuda.h"
test -f "$ROOT/include/crt/host_config.h"
"$ROOT/bin/tileiras" --version | tail -n 2
"$ROOT/bin/ptxas" --version | tail -n 1
echo "CUDA_TOOLKIT_PATH=$(cd "$ROOT" && pwd)"
