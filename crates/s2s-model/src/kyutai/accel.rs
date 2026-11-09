//! Offloading the multistream LM's heavy compute to another device.
//!
//! [`LmModel`](super::LmModel) keeps everything that is cheap or needs exact
//! host semantics on the CPU: input embeddings and conditions, depformer
//! token embeddings, extra heads, sampling, and the multistream / ASR
//! drivers. An [`LmAccelerator`] runs the rest: the temporal transformer with
//! its KV cache, `out_norm` and the text head, and each depformer slice's
//! input projection, transformer and output head. `crates/s2s-gpu` implements
//! it on CUDA.
//!
//! Build an accelerated model with
//! [`LmModel::load_accelerated`](super::LmModel::load_accelerated); the
//! accelerator loads its own weights (usually straight from the same
//! checkpoint), and the CPU side loads only the tensors it uses.

use super::error::Result;

/// The compute-heavy half of a Kyutai multistream LM, batch 1, one frame per
/// [`step`](LmAccelerator::step). Semantics match the CPU path of
/// [`LmModel`](super::LmModel) up to floating-point rounding.
pub trait LmAccelerator: Send + std::fmt::Debug {
    /// One temporal-transformer step on the input embedding `x` (`[d_model]`,
    /// conditions already added): positional encoding at
    /// [`current_seq_len`](LmAccelerator::current_seq_len), every layer
    /// (appending to the KV cache), `out_norm`, and the text head. Returns
    /// `(hidden, text_logits)`.
    fn step(&mut self, x: &[f32]) -> Result<(Vec<f32>, Vec<f32>)>;

    /// Clear the temporal transformer's KV cache.
    fn reset(&mut self);

    /// Frames processed since the last reset.
    fn current_seq_len(&self) -> usize;

    /// Start a depformer pass for one frame on `hidden` (`[d_model]`):
    /// clears the depformer's KV cache.
    fn depformer_begin(&mut self, hidden: &[f32]) -> Result<()>;

    /// Depformer slice `s` (called for `s = 0, 1, ...` after
    /// [`depformer_begin`](LmAccelerator::depformer_begin)): input
    /// `linear_in_s(hidden) + emb` (`emb`: `[d_dep]`, the embedding of the
    /// previous token, or `None`), slice `s`'s transformer, and its output
    /// head. Returns the logits (`[audio_vocab_size - 1]`).
    fn depformer_slice(&mut self, s: usize, emb: Option<&[f32]>) -> Result<Vec<f32>>;
}
