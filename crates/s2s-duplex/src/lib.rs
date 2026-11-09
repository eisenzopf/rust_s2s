//! Our full-duplex multistream model (plan §3).
//!
//! Every position of the **Temporal Transformer** sees one row of the
//! per-frame token grid (all streams of [`s2s_core::streams::StreamLayout`],
//! embedded and summed) or one injected text token. Its output `z` conditions
//! the small **Depth Transformer**, which predicts the next row's
//! `K = num_generated()` tokens autoregressively: user transcript (`Tu`),
//! assistant text slots (`Tm`), assistant audio codebooks (`Am`).
//!
//! - [`DuplexModel::forward_loss`] / [`DuplexModel::backward`] train on
//!   batches from `s2s-data` (teacher forcing inside the depth transformer,
//!   per-target loss weights, injection positions).
//! - [`stream`] runs it frame by frame: [`DuplexModel::feed_row`],
//!   [`DuplexModel::feed_injection`] and [`DuplexModel::sample_next`], with a
//!   per-sub-step hook for grammar masks (tool calls) and forced tokens
//!   (STT/TTS modes).
//! - [`train`] is the training driver behind the `s2s-duplex-train` binary:
//!   shards in, AdamW with a schedule, checkpoints with exact resume.
//!
//! Positions follow `s2s-data`'s assembly exactly: the token at position `p`
//! predicts the generated streams of position `p + 1`; injected text is never
//! predicted, but the last injected position predicts the following frame.
//!
//! This crate is distinct from `s2s-model` (Kyutai-compatible inference):
//! this is the extended architecture we train (user-text stream, several
//! assistant text slots per frame, injection segments).

#![forbid(unsafe_code)]

mod config;
mod model;
pub mod stream;
pub mod train;

pub use config::DuplexConfig;
pub use model::{DuplexBatch, DuplexCtx, DuplexModel, LossReport};
