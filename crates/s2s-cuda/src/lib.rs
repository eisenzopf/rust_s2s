//! GPU library wrappers for rust_s2s, built on [`cudarc`](https://docs.rs/cudarc)
//! 0.19 (ADR 0001, decision 4).
//!
//! Everything that touches CUDA is behind feature **`cuda`**; without it this
//! crate is only the CPU-side [`layout`] planner and builds with no CUDA
//! toolkit at all.
//!
//! With `cuda`:
//! - `gemm`: cuBLASLt GEMMs in the row-major layouts of
//!   [`s2s_kernels::reference::gemm`] (both transposes; f32 or bf16 inputs
//!   with f32 accumulation; f32 or bf16 output; `alpha`/`beta`; fused bias
//!   epilogue).
//! - `nccl`: `NcclComm`, an [`s2s_train::comm::Communicator`] over NCCL, so the
//!   DDP / ZeRO-1 code in [`s2s_train::dist`] runs unchanged across GPUs.
//! - `interop`: zero-copy hand-off of cudarc device buffers to cuTile tensors
//!   through `cutile::tensor::Tensor::from_foreign`, NVIDIA's documented
//!   pattern.
//!
//! cudarc is built with `dynamic-loading`: libcuda, libcublasLt and libnccl
//! are opened on first use, so building needs no CUDA libraries and no GPU.
//! (The `cuda` feature still needs CUDA headers, via `CUDA_TOOLKIT_PATH`,
//! because cuTile generates its driver bindings at build time.)
//!
//! **Status:** compile-checked only. The GPU tests (`tests/gpu.rs`) are
//! `#[ignore]`d until they run on real hardware:
//! `cargo test -p s2s-cuda --features cuda -- --ignored`.
//!
//! All `unsafe` code (cuBLASLt FFI calls and the cuTile `DeviceAllocation`
//! impl) lives in one private module, `raw`, with a `SAFETY` comment on each
//! block; everything exported is safe.

pub mod layout;

#[cfg(feature = "cuda")]
pub mod gemm;
#[cfg(feature = "cuda")]
pub mod interop;
#[cfg(feature = "cuda")]
pub mod nccl;
#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
mod raw;

#[cfg(feature = "cuda")]
pub use cudarc;
#[cfg(feature = "cuda")]
pub use cutile;
#[cfg(feature = "cuda")]
pub use half;
