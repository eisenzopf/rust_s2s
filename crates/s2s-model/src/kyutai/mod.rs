//! CPU port of Kyutai's multistream language model.
//!
//! Source of truth: Kyutai's Rust implementation, `rust/moshi-core` at
//! commit `e6a55d27` of <https://github.com/kyutai-labs/moshi> (= crates.io
//! `moshi 0.6.4`), MIT/Apache-2.0. Module map:
//!
//! | Here | Kyutai | What |
//! |---|---|---|
//! | [`config`] | `lm.rs`, `transformer.rs`, `conditioner.rs`, `lm_generate_multistream.rs` | Configs (serde-compatible) and presets |
//! | [`weights`] | candle `VarBuilder` | Lazy safetensors reader, name/shape-checked loading |
//! | [`LmModel`] | `lm.rs` (`LmModel`, `DepFormer`) | Temporal transformer step, text head, depformer sampling |
//! | `layers` (private) | `transformer.rs`, `kv_cache.rs` | Attention with rotating KV cache, MLP, norms, positions |
//! | [`conditioner`] | `conditioner.rs` | LUT / continuous conditioners |
//! | [`MultistreamState`] | `lm_generate_multistream.rs` | Moshi full-duplex driver |
//! | [`AsrState`] | `asr.rs` | DSM speech-to-text driver (takes Mimi codes) |
//!
//! # Model
//!
//! Each 80 ms frame carries one text token and `audio_codebooks` Mimi codes.
//! [`LmModel::step`] sums the embeddings of the frame's input tokens (text +
//! every audio codebook, plus an optional condition vector), runs one step
//! of the temporal transformer (pre-norm, fused-QKV attention with
//! interleaved RoPE over a sliding `context` window kept in a KV cache,
//! gated MLP), applies `out_norm` and the text head. For Moshi-style models
//! [`LmModel::depformer_sample`] then generates the frame's audio codebooks
//! one by one with the small depth transformer: slice `s` projects the
//! temporal output (`linear_in`), adds the embedding of the previous slice's
//! token (the text token for slice 0), runs its own transformer layers over
//! the slices so far, and samples from its head (`linear_out`). STT models
//! have no depformer: they read audio and write the (delayed) text stream.
//!
//! # Numerics
//!
//! Everything runs in `f32` on the CPU reference kernels. Kyutai-specific
//! details reproduced exactly: RoPE angles in `f32` (`rope_tables_f32`),
//! RMSNorm `eps = 1e-8`, LayerNorm `eps = 1e-5`, SiLU as `x / (1 + e^-x)`,
//! erf-GELU through candle's `erf`, gated hidden size `11 d / 4` when
//! `dim_feedforward = 4 d`, the rotating KV cache's slot order, and
//! candle's greedy tie-breaking. Remaining differences are float summation
//! order in matrix products (candle's `gemm` vs naive loops), ~1e-6
//! relative; see `tests/kyutai_parity.rs` for the measured agreement.
//!
//! # Not ported (yet)
//!
//! Cross-attention (vision / TTS models: rejected at load), classifier-free
//! guidance, repetition penalty and pad multiplier in the multistream driver,
//! batching, quantized (GGUF) weights, and Kyutai's PyTorch checkpoint
//! naming (`depformer.layers.*` with `weights_per_step`, `depformer_in.*`,
//! `linears.*`): the candle checkpoints (`kyutai/*-candle*`) load directly.

mod accel;
mod asr;
pub mod conditioner;
pub mod config;
pub mod error;
mod layers;
mod lm;
mod multistream;
pub mod sampling;
pub mod weights;

pub use accel::LmAccelerator;
pub use asr::{AsrMsg, AsrState};
pub use conditioner::{Condition, ConditionProvider};
pub use config::{AsrConfig, LmConfig, MultistreamConfig};
pub use error::{Error, Result};
pub use lm::{DepformerOutput, LmModel, NameCheck, StepOutput};
pub use multistream::{ForcedAudioTokens, FrameOutput, MultistreamState, UNGENERATED};
pub use sampling::Sampler;
pub use weights::{SafeTensors, StoredDtype, TensorInfo, TensorMap, WeightSource};
