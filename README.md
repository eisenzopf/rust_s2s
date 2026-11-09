# rust_s2s

A full-duplex speech-to-speech model (STT + LLM + TTS + tool calling fused into one network, in the style of Kyutai's Moshi / Delayed Streams Modeling), trained and served on CUDA from Rust using NVIDIA's CUDA Rust libraries: **cuTile Rust** and **cuda-oxide**.

- 📐 [Implementation plan](docs/IMPLEMENTATION_PLAN.md): design, kernels, training recipe, streaming runtime, milestones
- 🧭 [ADR 0001: GPU toolchain](docs/decisions/0001-gpu-toolchain.md): how cuTile and cuda-oxide are used, and what has been verified

**Status:** milestone M0. CPU reference kernels and the first GPU kernels in both CUDA Rust models are written and compile to GPU machine code. **No kernel has run on a GPU yet.**

## Layout

| Path | What | Toolchain |
|---|---|---|
| `crates/s2s-core` | dtypes, shapes, host tensors, deterministic RNG | stable (1.89+) |
| `crates/s2s-kernels` | CPU reference kernels, forward + backward, with finite-difference gradient checks; dense GPU kernels in **cuTile Rust** behind feature `cuda` | stable (1.89+) |
| `crates/s2s-simt` | sparse/irregular GPU kernels in **cuda-oxide** (bitmask, embedding gather/scatter-add, delays) plus a GPU self-test | Rust 1.99.1 + `cargo oxide` (standalone workspace) |
| `scripts/fetch-cuda-wheels.sh` | assembles a CUDA 13.5 toolkit tree (headers, `tileiras`, `ptxas`, libNVVM) from NVIDIA's PyPI wheels; no full CUDA install needed to compile | — |
| `scripts/check-simt.sh` | cuda-oxide kernels → PTX → cubins for sm_80/89/90/100 | — |

## Build and test

CPU only (no CUDA):

```bash
cargo test --workspace
```

cuTile kernels, compiled to cubins without a GPU:

```bash
scripts/fetch-cuda-wheels.sh .cuda-wheels
export CUDA_TOOLKIT_PATH=$PWD/.cuda-wheels/nvidia/cu13
export PATH=$CUDA_TOOLKIT_PATH/bin:$PATH LD_LIBRARY_PATH=$CUDA_TOOLKIT_PATH/lib
S2S_REQUIRE_TILEIRAS=1 cargo test -p s2s-kernels --features cuda
```

cuda-oxide kernels, compiled to cubins without a GPU (same environment as above):

```bash
rustup toolchain install 1.99.1 -c rust-src -c rustc-dev -c llvm-tools
cargo +1.99.1 install --locked --git https://github.com/NVIDIA/cuda-rust \
  --rev 367bc0f8e671070e00eebc7e5498f3e8eebefc72 cargo-oxide
scripts/check-simt.sh
```

On a machine with an sm_80+ GPU and a CUDA 13 driver, run the GPU parity checks against the CPU references:

```bash
cargo test -p s2s-kernels --features cuda --test cuda_parity -- --ignored   # cuTile
(cd crates/s2s-simt && cargo oxide run)                                     # cuda-oxide
```

### GPU tests on AWS

`scripts/aws/run_gpu_tests.py` runs all of the above on a short-lived EC2 GPU instance (default `g6.xlarge`, NVIDIA L4) that terminates itself when done, with a hard 2-hour deadline. It needs AWS credentials in the environment. `scripts/aws/iam-policy.json` is a minimal IAM policy for it: it may launch only Ampere-or-newer GPU instances and write only to its own log bucket.

```bash
python3 scripts/aws/run_gpu_tests.py --dry-run   # show the plan, no AWS calls
python3 scripts/aws/run_gpu_tests.py             # launch, wait, print the log, terminate
```

