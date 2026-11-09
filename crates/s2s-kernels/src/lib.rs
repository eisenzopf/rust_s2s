//! Kernels for rust_s2s.
//!
//! [`reference`](mod@reference) holds plain-Rust CPU implementations of every operation the
//! model needs, forward and backward. They are written for clarity, not speed:
//! they are the oracles that the GPU kernels (cuTile Rust / cuda-oxide /
//! cuBLASLt) are tested against, and they let the trainer and model logic be
//! developed and tested without a GPU.
//!
//! Conventions used throughout `reference`:
//!
//! - All buffers are contiguous and row-major. Dimensions are passed
//!   explicitly, mirroring the signatures GPU kernels will have.
//! - Functions are generic over [`s2s_core::Float`], so gradient checks can run
//!   in `f64`.
//! - Backward functions **accumulate** into gradient buffers (`+=`), so callers
//!   zero them once per step and shared parameters just work.
//! - Shape errors are programming errors and panic via `assert!`, like an
//!   out-of-bounds index would.

#![forbid(unsafe_code)]

pub mod reference;

#[cfg(feature = "cuda")]
pub mod cuda;
