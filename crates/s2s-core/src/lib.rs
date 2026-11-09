//! Core types shared by every rust_s2s crate.
//!
//! This crate is deliberately small and dependency-light. It holds:
//!
//! - [`DType`]: the closed set of element types the project uses.
//! - [`Shape`]: a row-major shape with helpers for element counts and strides.
//! - [`HostTensor`]: a simple owned, contiguous, row-major host tensor.
//! - [`Float`]: the float trait the CPU reference kernels are generic over
//!   (so gradient checks can run in `f64` while normal tests run in `f32`).
//! - [`SplitMix64`]: a tiny deterministic RNG for tests and weight init.
//! - [`Error`] / [`Result`].

#![forbid(unsafe_code)]

mod dtype;
mod error;
mod float;
mod rng;
mod shape;
mod tensor;

pub use dtype::DType;
pub use error::{Error, Result};
pub use float::{cast, Float};
pub use rng::SplitMix64;
pub use shape::Shape;
pub use tensor::HostTensor;
