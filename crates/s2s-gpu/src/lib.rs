//! CUDA inference runtime for rust_s2s (feature `cuda`).
//!
//! `kyutai::GpuLm` runs Kyutai's multistream LM (Moshi, DSM STT) on the
//! GPU as an `s2s_model::kyutai::LmAccelerator`: build the model with
//! `kyutai::load_lm` and use it like any `s2s_model::kyutai::LmModel`
//! (multistream driver, ASR driver, `KyutaiEngine`).
//!
//! Per decode step, everything is queued on one CUDA stream with no host
//! synchronization until the logits are read back:
//!
//! - GEMMs (every linear layer, batch 1) go to cuBLASLt through
//!   `s2s_cuda::gemm`, with bf16 weights (or f32 for exact parity checks);
//! - everything else (RMSNorm, RoPE, KV-cache append, split-K
//!   flash-decoding attention, gated MLP, residuals, bf16 casts) is a
//!   cuda-oxide SIMT kernel from `crates/s2s-simt` (`infer` module), loaded
//!   from PTX prebuilt by `scripts/build-simt-ptx.sh`.
//!
//! Embedding lookups and sampling stay on the CPU in `LmModel`, so the
//! token-level semantics are exactly the CPU port's. See ADR 0002.

#[cfg(feature = "cuda")]
pub mod kyutai;
#[cfg(feature = "cuda")]
mod simt;
