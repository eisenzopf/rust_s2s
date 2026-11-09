//! Neural-network layers with explicit forward and backward passes.
//!
//! Design (plan §6.1): the model is a fixed composition of a handful of layer
//! types, so instead of a general autograd graph every layer implements
//! `forward(...) -> (output, Ctx)` and `backward(ctx, d_output) -> d_input`.
//! `Ctx` holds whatever the backward pass needs. This keeps memory use
//! predictable, makes activation checkpointing a per-block choice
//! ([`block::Checkpoint`]), and costs far less code than a tape-based autograd.
//!
//! Layers are generic over [`s2s_core::Float`] so whole-model gradient checks
//! can run in `f64`. They currently call the CPU reference kernels in
//! `s2s_kernels::reference`; the GPU path will swap those for cuTile /
//! cuda-oxide / cuBLASLt kernels behind the same layer API.
//!
//! Conventions:
//! - Activations are row-major `[rows, features]`, where `rows = batch * seq`.
//! - Backward passes **accumulate** into parameter gradients; call
//!   [`Module::zero_grad`] once per optimizer step.
//! - Parameters carry hierarchical names (`blocks.0.attn.wq`) used for
//!   checkpoints and optimizer parameter groups.

#![forbid(unsafe_code)]

pub mod attention;
pub mod block;
pub mod embedding;
pub mod linear;
pub mod lm;
pub mod mlp;
pub mod norm;
mod param;
mod util;

pub use param::{Module, Param};
