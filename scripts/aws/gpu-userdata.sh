#!/bin/bash
# EC2 user-data for scripts/aws/run_gpu_tests.py. Runs once at first boot on
# a GPU instance, as root. The launcher substitutes the @@...@@ placeholders.
#
# It runs every GPU test, uploads the log to a presigned S3 URL, then powers
# off. The instance is launched with shutdown behavior "terminate", so
# powering off deletes it. A hard deadline powers off even if something hangs.
set -uo pipefail

REPO_URL="@@REPO_URL@@"
COMMIT="@@COMMIT@@"
RESULT_PUT_URL="@@RESULT_PUT_URL@@"
DEADLINE_MIN="@@DEADLINE_MIN@@"
CUDA_RUST_REV="@@CUDA_RUST_REV@@"

# Safety net first: whatever happens below, the machine goes away.
shutdown -h "+${DEADLINE_MIN}" "rust_s2s GPU test deadline reached"

LOG=/var/log/s2s-gpu-tests.log
exec > >(tee -a "$LOG" /dev/console) 2>&1

upload_and_exit() {
  echo "=== S2S_GPU_TESTS_DONE status=$1 ==="
  curl -sS --retry 5 -X PUT -H "Content-Type: text/plain" --upload-file "$LOG" "$RESULT_PUT_URL" || true
  sleep 5
  shutdown -h now
  exit 0
}

section() { echo; echo "=== $* ($(date -u +%H:%M:%S)) ==="; }
STATUS=0
run() { # run a step, record failure, keep going
  section "$1"; shift
  if "$@"; then echo "--- OK"; else echo "--- FAILED (exit $?)"; STATUS=1; fi
}

section "GPU and driver"
nvidia-smi || upload_and_exit "no-gpu"
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
echo "driver major version: $DRIVER_MAJOR (CUDA 13 needs >= 580)"

section "System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q && apt-get install -y -q git curl build-essential clang libclang-dev python3-pip python3-venv pkg-config \
  || upload_and_exit "apt-failed"

section "Rust toolchains"
export HOME=/root CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable || upload_and_exit "rustup-failed"
export PATH="/root/.cargo/bin:$PATH"
rustup toolchain install 1.99.1 --profile minimal -c rust-src -c rustc-dev -c llvm-tools || upload_and_exit "rust-1.99.1-failed"

section "Checkout $COMMIT"
git clone -q "$REPO_URL" /opt/rust_s2s && cd /opt/rust_s2s && git checkout -q "$COMMIT" || upload_and_exit "clone-failed"
git log -1 --oneline

section "CUDA toolkit from NVIDIA PyPI wheels"
python3 -m venv /opt/pyenv && /opt/pyenv/bin/pip install -q --upgrade pip
PATH="/opt/pyenv/bin:$PATH" S2S_CUDA_RUNTIME_LIBS=1 scripts/fetch-cuda-wheels.sh /opt/cuda-wheels || upload_and_exit "cuda-wheels-failed"
export CUDA_TOOLKIT_PATH=/opt/cuda-wheels/nvidia/cu13
export PATH="$CUDA_TOOLKIT_PATH/bin:$PATH"
# Real driver libcuda comes from the system; the wheel libs supply nvvm etc.
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_PATH/lib"

run "CPU tests" cargo test --workspace
run "cuTile compile to cubin" env S2S_REQUIRE_TILEIRAS=1 cargo test -p s2s-kernels --features cuda --test cuda_compile
run "cuTile GPU parity (vs CPU reference)" cargo test -p s2s-kernels --features cuda --test cuda_parity -- --ignored --nocapture
run "s2s-cuda GPU (cuBLASLt, NCCL, cuTile interop)" cargo test -p s2s-cuda --features cuda --test gpu -- --ignored --nocapture

section "cargo-oxide"
cargo +1.99.1 install --locked --git https://github.com/NVIDIA/cuda-rust --rev "$CUDA_RUST_REV" cargo-oxide \
  || { echo "--- FAILED"; STATUS=1; }
run "cuda-oxide doctor" bash -c "cd crates/s2s-simt && cargo oxide doctor"
run "cuda-oxide compile to cubin" scripts/check-simt.sh
run "cuda-oxide GPU self-test (vs CPU reference)" bash -c "cd crates/s2s-simt && cargo oxide run"

if [ "$STATUS" -eq 0 ]; then upload_and_exit "all-passed"; else upload_and_exit "some-failed"; fi
