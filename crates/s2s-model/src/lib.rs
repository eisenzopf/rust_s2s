//! Model definitions and streaming inference for rust_s2s.
//!
//! - [`kyutai`]: a CPU port of Kyutai's multistream language model (the
//!   Temporal Transformer + Depth Transformer "depformer" of Moshi and of the
//!   Delayed-Streams-Modeling STT/TTS models). It loads Kyutai's published
//!   checkpoints by their tensor names and reproduces their streaming
//!   inference frame by frame. This is the stepping stone to plan milestone
//!   M4 (run existing Kyutai weights through our code) before our own
//!   extended model (plan §3) is built.
//!
//! Inference only for now: no backward passes in this crate yet.

#![forbid(unsafe_code)]

pub mod kyutai;
