//! Mimi hyper-parameters.

use serde::{Deserialize, Serialize};

use crate::{MimiError, Result};

/// Mimi configuration.
///
/// Field names follow Kyutai's `moshi::mimi::Config` (with the SEANet and
/// transformer sub-configs flattened). Options that every published Mimi
/// uses, and that this implementation supports only in that setting, are not
/// configurable: causal convolutions with zero padding (SEANet) and
/// replicate padding (downsample), weight-normalized SEANet convs (stored
/// folded or as `weight_g`/`weight_v`), ELU activations, no LSTM, learnt
/// stride-`downsample_stride` down/up-sampling, a split RVQ (1 semantic +
/// `n_q - 1` acoustic levels) with input/output projections, and pre-norm
/// transformers with LayerNorm (eps 1e-5), interleaved RoPE, bias-free
/// attention and MLP, exact GELU, and layer scale.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MimiConfig {
    /// Audio channels (1 for every published Mimi).
    pub channels: usize,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Code frame rate in Hz.
    pub frame_rate: f64,
    /// SEANet latent dimension, also the transformers' input/output size.
    pub dimension: usize,
    /// SEANet base channel count.
    pub n_filters: usize,
    /// Residual blocks per SEANet stage.
    pub n_residual_layers: usize,
    /// SEANet up-sampling ratios, decoder order (the encoder uses them reversed).
    pub ratios: Vec<usize>,
    /// Kernel size of the first SEANet conv.
    pub kernel_size: usize,
    /// Kernel size of the residual blocks' first conv.
    pub residual_kernel_size: usize,
    /// Kernel size of the last SEANet conv.
    pub last_kernel_size: usize,
    /// Residual block `j` uses dilation `dilation_base^j`.
    pub dilation_base: usize,
    /// Residual blocks' hidden size is `channels / compress`.
    pub compress: usize,
    /// Identity skip connections (else a 1x1 conv shortcut).
    pub true_skip: bool,
    /// ELU `alpha`.
    pub elu_alpha: f64,
    /// Transformer width.
    pub transformer_d_model: usize,
    /// Attention heads.
    pub transformer_num_heads: usize,
    /// Transformer layers (each of the encoder and decoder transformers).
    pub transformer_num_layers: usize,
    /// MLP hidden size.
    pub transformer_dim_feedforward: usize,
    /// Attention span in transformer steps: a query at step `q` attends to
    /// keys at steps `k` with `0 <= q - k <= context` (Kyutai's Rust
    /// semantics; their PyTorch code uses `q - k < context`, i.e. this with
    /// `context - 1`).
    pub transformer_context: usize,
    /// RoPE base period.
    pub transformer_max_period: f64,
    /// Whether the layers have `layer_scale_{1,2}` parameters.
    pub transformer_layer_scale: bool,
    /// Codebooks used (1 semantic + `n_q - 1` acoustic). A checkpoint may hold more.
    pub quantizer_n_q: usize,
    /// Entries per codebook.
    pub quantizer_bins: usize,
    /// Codebook dimension.
    pub quantizer_dim: usize,
}

impl MimiConfig {
    /// Kyutai's published Mimi (`moshi::mimi::Config::v0_1`), with the 8
    /// codebooks Moshi and Kyutai's STT/TTS models use. The checkpoint holds more
    /// (Kyutai's `v0_1(None)` uses 16); use [`Self::with_n_q`] for another bitrate.
    pub fn v0_1() -> Self {
        Self {
            channels: 1,
            sample_rate: 24_000.0,
            frame_rate: 12.5,
            dimension: 512,
            n_filters: 64,
            n_residual_layers: 1,
            ratios: vec![8, 6, 5, 4],
            kernel_size: 7,
            residual_kernel_size: 3,
            last_kernel_size: 3,
            dilation_base: 2,
            compress: 2,
            true_skip: true,
            elu_alpha: 1.0,
            transformer_d_model: 512,
            transformer_num_heads: 8,
            transformer_num_layers: 8,
            transformer_dim_feedforward: 2048,
            transformer_context: 250,
            transformer_max_period: 10_000.0,
            transformer_layer_scale: true,
            quantizer_n_q: 8,
            quantizer_bins: 2048,
            quantizer_dim: 256,
        }
    }

    /// The same config with `n_q` codebooks.
    pub fn with_n_q(mut self, n_q: usize) -> Self {
        self.quantizer_n_q = n_q;
        self
    }

    /// Parse a JSON config (the field names above).
    pub fn from_json(s: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(s).map_err(|e| MimiError::Config(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// SEANet hop: input samples per encoder (transformer) step.
    pub fn hop_length(&self) -> usize {
        self.ratios.iter().product()
    }

    /// Transformer step rate in Hz (25 for Mimi).
    pub fn encoder_frame_rate(&self) -> f64 {
        self.sample_rate / self.hop_length() as f64
    }

    /// Transformer steps per code frame (2 for Mimi).
    pub fn downsample_stride(&self) -> usize {
        // Same truncation as Kyutai's `(encoder_frame_rate / frame_rate) as usize`.
        (self.encoder_frame_rate() / self.frame_rate) as usize
    }

    /// Samples per code frame (1920 for Mimi).
    pub fn frame_size(&self) -> usize {
        self.hop_length() * self.downsample_stride()
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.transformer_d_model / self.transformer_num_heads
    }

    /// Check internal consistency.
    pub fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(MimiError::Config(m));
        if self.channels == 0 || self.ratios.is_empty() || self.ratios.contains(&0) {
            return bad("channels and ratios must be positive".into());
        }
        if self.n_filters == 0 || self.dimension == 0 || self.compress == 0 {
            return bad("n_filters, dimension and compress must be positive".into());
        }
        if !self.n_filters.is_multiple_of(self.compress) {
            return bad("compress must divide n_filters".into());
        }
        if self.downsample_stride() == 0 {
            return bad("frame_rate must not exceed the encoder frame rate".into());
        }
        let (d, h) = (self.transformer_d_model, self.transformer_num_heads);
        if h == 0 || d % h != 0 || (d / h) % 2 != 0 {
            return bad(format!(
                "d_model {d} must split into {h} heads of even size"
            ));
        }
        if self.quantizer_n_q == 0 || self.quantizer_bins == 0 || self.quantizer_dim == 0 {
            return bad("quantizer n_q, bins and dim must be positive".into());
        }
        Ok(())
    }
}

impl Default for MimiConfig {
    fn default() -> Self {
        Self::v0_1()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v0_1_frame_geometry() {
        let c = MimiConfig::v0_1();
        c.validate().unwrap();
        assert_eq!(c.hop_length(), 960);
        assert_eq!(c.encoder_frame_rate(), 25.0);
        assert_eq!(c.downsample_stride(), 2);
        assert_eq!(c.frame_size(), s2s_core::streams::FRAME_SAMPLES);
        assert_eq!(c.sample_rate as u32, s2s_core::streams::SAMPLE_RATE);
        assert_eq!(c.quantizer_bins as u32, s2s_core::streams::CODEBOOK_SIZE);
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.clone().with_n_q(32).quantizer_n_q, 32);
    }

    #[test]
    fn json_round_trip() {
        let c = MimiConfig::v0_1();
        let s = serde_json::to_string(&c).unwrap();
        assert_eq!(MimiConfig::from_json(&s).unwrap(), c);
        assert!(MimiConfig::from_json("{}").is_err());
    }
}
