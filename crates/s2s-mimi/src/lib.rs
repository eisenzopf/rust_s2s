//! Mimi, Kyutai's streaming neural audio codec, on CPU in plain Rust
//! (plan §3.1, milestone M3).
//!
//! 24 kHz mono audio <-> 12.5 Hz frames of `n_q` codes (codebook 0 semantic,
//! the rest acoustic; Moshi and Kyutai's STT/TTS use 8; the checkpoint holds
//! more). The pipeline mirrors Kyutai's Rust implementation
//! (`moshi` crate 0.6.4, `rust/moshi-core/src/mimi.rs` at commit `e6a55d27`):
//!
//! ```text
//! encode: pcm -> SEANet encoder (causal convs, ELU, ratios 4·5·6·8) -> 25 Hz latent
//!             -> encoder transformer -> learnt downsample (stride 2) -> split RVQ -> codes
//! decode: codes -> split RVQ decode -> learnt depthwise upsample (stride 2)
//!             -> decoder transformer -> SEANet decoder (transposed convs) -> pcm
//! ```
//!
//! Every op is a CPU reference kernel from [`s2s_kernels::reference`]
//! (`conv1d`, `conv_transpose1d`, `elu`, `gelu`, `layer_norm`, `rope`,
//! `attention`, `rvq_ip`, `rvq`).
//!
//! # Weights
//!
//! [`Mimi::load`] reads Kyutai-format safetensors, with their exact tensor
//! names (`encoder.model.0.conv.conv.weight`,
//! `quantizer.rvq_first.vq.layers.0._codebook.embedding_sum`, ...). Conv
//! weights may be stored folded (`weight`) or weight-normalized
//! (`weight_g` + `weight_v`); `F32`, `BF16`, `F16` and `F64` tensors are
//! accepted. Use [`MimiConfig::v0_1`] with the published checkpoint,
//! `tokenizer-e351c8d8-checkpoint125.safetensors`. Download it from Hugging
//! Face: it ships with Kyutai's Moshi releases, e.g.
//! `https://huggingface.co/kyutai/moshiko-pytorch-bf16` (the default repo of
//! Kyutai's PyTorch loader), and Kyutai's other model repos (candle/MLX
//! Moshi, STT, TTS) carry the same codec, sometimes under another file name.
//! The `kyutai/mimi` repo itself hosts the same model converted to Hugging
//! Face `transformers` naming (`model.safetensors`, different tensor
//! names), which this loader does not read.
//!
//! The `#[ignore]`d test `real_checkpoint_round_trip` runs on it:
//! `S2S_MIMI_WEIGHTS=/path/to/tokenizer-e351c8d8-checkpoint125.safetensors
//! cargo test -p s2s-mimi -- --ignored --nocapture`.
//!
//! # Streaming
//!
//! [`EncoderState`] / [`DecoderState`] hold one stream's causal state (conv
//! history, transposed-conv overlap, transformer KV cache), so one [`Mimi`]
//! serves many sessions. Pushing 1920 samples yields one code frame;
//! pushing one code frame yields 1920 samples. Streaming output equals the
//! offline output **bit for bit** (every kernel reduces each output element
//! in an order that does not depend on how the sequence is chunked).
//!
//! # Differences from Kyutai's implementation
//!
//! - Float summation order differs from candle's GEMM/conv kernels, so
//!   activations agree to ~1e-6 relative rather than bitwise, and a code can
//!   flip where two codewords are almost equidistant (see the parity tests).
//! - Kyutai's *streaming* transformer keeps a rotating KV cache of `context`
//!   entries, so once a stream is longer than `context` steps (10 s) each
//!   query sees one or two keys fewer than in their offline mask
//!   (`q - k <= context`). Here streaming keeps exactly the offline window,
//!   so streaming equals offline; this matches Kyutai's offline output
//!   everywhere and their streaming output for the first `context` steps.
//! - Offline encode of a length that is not a multiple of 1920 uses Kyutai's
//!   causal extra padding; streaming only emits whole frames.

#![forbid(unsafe_code)]

mod config;
mod conv;
mod error;
mod model;
mod quantizer;
mod seanet;
mod transformer;
mod types;
mod weights;

pub use config::MimiConfig;
pub use error::{MimiError, Result};
pub use model::{
    DecodeStages, DecoderState, EncodeStages, EncoderState, LoadReport, Mimi, StreamingDecoder,
    StreamingEncoder,
};
pub use types::{Codes, Signal};
pub use weights::{SafetensorsCheckpoint, TensorMap, TensorSource};
