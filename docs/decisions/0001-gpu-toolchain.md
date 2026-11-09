# ADR 0001 — GPU kernel toolchain: NVIDIA CUDA Rust (cuTile + cuda-oxide)

| | |
|---|---|
| Status | Accepted (M0) |
| Date | 2026-11-09 |
| Supersedes | The provisional toolchain split in `IMPLEMENTATION_PLAN.md` §7.1 |

## Context

The plan calls for GPU kernels written in Rust with NVIDIA's newly announced CUDA Rust libraries. NVIDIA announced these as two NVlabs projects:

| Announced as | Where it lives now |
|---|---|
| `github.com/NVlabs/cutile-rs` (Tile model) | **Archived.** Its README says it moved to `github.com/NVIDIA/cuda-rust` (`cutile-rs/`). |
| `github.com/NVlabs/cuda-oxide` (SIMT model) | Resolves to the same repository: `NVlabs/cuda-oxide` and `NVIDIA/cuda-rust` report the same `HEAD` (`367bc0f`). |

So both libraries now live in one repository, **`github.com/NVIDIA/cuda-rust`** (Apache-2.0). It contains the two kernel models plus a shared host runtime (`cuda-bindings`, `cuda-core`, `cuda-async`). Both models are labelled early-stage or alpha, and NVIDIA warns to expect API breakage.

Everything below was checked by reading the source at `367bc0f` and by **building and running the toolchains in a GPU-less container**.

## Findings

### cuTile Rust (Tile model)

- **Programming model:** kernels are Rust functions inside `#[cutile::module]` / `#[cutile::entry]`. Mutable outputs are split into disjoint partitions, and the partition sets the launch grid.
- **Compilation:** kernels are JIT-compiled at first launch: Rust DSL → Tile IR (pure Rust) → bytecode → `tileiras` → cubin.
- **Toolchain:** stable Rust 1.89 or newer. CUDA 13.2+ (13.3 recommended). **sm_80+ only** (no V100 or T4).
- **Driver-free compilation:** `KernelCompiler` lowers a kernel to Tile IR bytecode **without a driver**. With `tileiras`, the bytecode assembles to a real cubin, still without a GPU.
- **Interop:** `Tensor::from_foreign` wraps device memory owned by another library (for example `cudarc`) with no copy.
- **Ready-made kernels:** NVIDIA's unpublished `cutile-kernels` crate has forward-only f16 inference kernels: flash attention (causal, GQA prefill, split-K decode), RMSNorm, fused residual-add + RMSNorm, RoPE, KV cache and embeddings. **No backward kernels.**
- **Performance:** NVIDIA's paper (arXiv 2606.15991) reports f16 GEMM at 98% of cuBLAS on B200. cuTile is used by candle, mistral.rs and Hugging Face's Grout.

### cuda-oxide (SIMT model)

- **What it is:** a custom rustc codegen backend: Rust MIR → Pliron IR → LLVM IR → PTX, compiled ahead of time.
- **Device API:** `#[kernel]` functions inside `#[cuda_module]`, with `thread::index_1d()`, `DisjointSlice`, scoped atomics, shared memory, warp ops and TMA.
- **Launches:** `#[launch_contract]` gives safe, checked launches.
- **Toolchain:**
  - pinned **Rust 1.99.1** with `rust-src`, `rustc-dev` and `llvm-tools`;
  - the `cargo oxide` driver;
  - `llc` 21+, which the toolchain's `llvm-tools` provides (LLVM 23);
  - clang/libclang for bindgen. clang 18 works.
- **Packaging:** not on crates.io; projects depend on it by git.

> ⚠️ **The crates.io crate called `cuda-oxide` (0.4.0) is an unrelated GPL-3.0 project by a different author. Never depend on it.**

- **Runtime constraint:** NVIDIA's template warns that host crates **must all come from the same repository**. A crates.io `cuda-core` next to a git `cuda-core` gives two runtimes, which breaks Tile/SIMT interop.

### What NVIDIA recommends

NVIDIA's README recommends Tile for "math over densely packed numerical data" and SIMT for "sparse, pointer chasing, mapping traditional data structures". Kernels from both models can share streams and buffers.

### Gaps

- **Libraries:** neither runtime wraps cuBLAS/cuBLASLt or NCCL. `cudarc` 0.19 does.
- **Float atomics:** cuda-oxide currently lowers `DeviceAtomicF32::fetch_add` to a **compare-and-swap loop**, even with `--arch sm_90`, instead of the native `red.global.add.f32`. Results are correct but slower under heavy contention.

### Verified in a GPU-less container (2026-11-09)

| Check | Result |
|---|---|
| CUDA toolkit tree from NVIDIA PyPI wheels (`scripts/fetch-cuda-wheels.sh`) | ✅ headers, `tileiras` 13.5, `ptxas`, libNVVM, nvJitLink, libdevice |
| cuTile kernels (`s2s-kernels --features cuda`) → Tile IR → **cubin** via `tileiras`, for sm_80/89/90/100 | ✅ `tests/cuda_compile.rs` |
| cuda-oxide: `cargo oxide doctor` / build NVIDIA's `vecadd` example to PTX | ✅ |
| cuda-oxide kernels (`crates/s2s-simt`) → PTX → **cubin** via `ptxas`, for sm_80/89/90/100 | ✅ `scripts/check-simt.sh` |
| Any kernel **executed on a GPU** | ❌ not yet — no GPU available |

## Decision

1. **Use both NVIDIA CUDA Rust models, split by kernel shape, following NVIDIA's guidance:**
   - **cuTile Rust (Tile)** for dense math: norms, RoPE, SwiGLU, attention (training and decode), fused linear + cross-entropy, softmax, AdamW, and the Mimi convolutions. These live in `crates/s2s-kernels` behind feature `cuda`.
   - **cuda-oxide (SIMT)** for sparse or irregular work: token-bitmask masking (constrained tool calls), multi-stream embedding gather and **atomic scatter-add**, delay patterns, KV-cache ring writes, sampling (top-k/top-p), and RVQ nearest-codeword search. These live in `crates/s2s-simt`.
2. **One pinned git revision for every CUDA Rust crate** (`NVIDIA/cuda-rust@367bc0f`), in the root workspace and in `s2s-simt`, so both models share one `cuda-core` runtime. No crates.io copies of CUDA Rust crates. Upgrades are deliberate, one PR each.
3. **`crates/s2s-simt` is a standalone workspace** with its own `rust-toolchain.toml` (1.99.1), built by `cargo oxide`. That keeps cuda-oxide's pinned compiler away from the rest of the project, which builds on stable Rust 1.89+.
4. **`cudarc` provides cuBLASLt (dense GEMMs) and NCCL (collectives).** Buffers move to cuTile through `Tensor::from_foreign`, with no copies.
5. **Reuse NVIDIA's `cutile-kernels` forward kernels** for inference once they've been checked against our references. We write the bf16 variants and **all backward kernels** ourselves.
6. **Minimums:** Tile IR 13.3, so the CUDA 13.3+ toolkit (we use 13.5), sm_80+ GPUs and a CUDA 13 driver (R580+).
7. **CI has four tiers.** None needs a GPU except the last.
   - `cpu`: fmt, clippy, tests, docs.
   - `cuda-compile`: CUDA wheels, then cuTile kernels → cubins.
   - `simt-compile`: Rust 1.99.1 + `cargo-oxide`, then SIMT kernels → PTX → cubins.
   - Later, a self-hosted GPU job that runs `tests/cuda_parity.rs` and `cargo oxide run` (`s2s-simt-selftest`).

## Consequences

- **Hardware:** development GPUs must be Ampere or newer.
- **Toolchains:** two Rust toolchains (stable, plus 1.99.1 for `s2s-simt`), each confined to its own workspace.
- **Shared revision:** both models move together on one pinned revision. A breaking upstream change touches both crates in one upgrade PR.
- **Scatter-add performance:** heavy-contention scatter-add (embedding backward) may need an inline-PTX native atomic until cuda-oxide lowers `fetch_add` natively.

## Open items

- Run `tests/cuda_parity.rs` and `s2s-simt-selftest` on a real GPU.
- Raise the float-atomic lowering with NVIDIA, or use inline PTX `red.global.add.f32`.
- Add bf16 kernels, then backward kernels, in plan §7.2 priority order.
- Decide between vendoring `cutile-kernels` and a git dependency.
- Run the GitHub Actions workflow on GitHub. It was written and its steps were run locally, but the workflow file itself has not run yet.
