use s2s_core::streams::{StreamLayout, AUDIO_VOCAB};
use s2s_kernels::reference::attention::AttnMask;
use s2s_kernels::reference::rope::{RopeConfig, RopeLayout};
use s2s_nn::attention::AttentionConfig;
use s2s_nn::block::BlockConfig;

/// Model hyper-parameters (plan §3.6).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DuplexConfig {
    /// Stream layout (text slots, codebooks, delays).
    pub layout: StreamLayout,
    /// Text vocabulary size (tokenizer vocab incl. special tokens).
    pub text_vocab: usize,
    /// Audio vocabulary size (codebook entries + `AUDIO_PAD`).
    pub audio_vocab: usize,
    /// Id of the text `<pad>` token.
    pub text_pad: u32,
    /// Temporal Transformer width.
    pub d_model: usize,
    /// Temporal Transformer depth.
    pub n_layers: usize,
    /// Temporal query heads.
    pub q_heads: usize,
    /// Temporal key/value heads.
    pub kv_heads: usize,
    /// Temporal MLP hidden width.
    pub mlp_hidden: usize,
    /// Sliding attention window over positions (`None` = full causal).
    pub window: Option<usize>,
    /// Depth Transformer width.
    pub depth_d: usize,
    /// Depth Transformer depth.
    pub depth_layers: usize,
    /// Depth heads (query and key/value).
    pub depth_heads: usize,
    /// Depth MLP hidden width.
    pub depth_mlp_hidden: usize,
    /// RoPE base.
    pub rope_theta: f64,
    /// RMSNorm epsilon.
    pub norm_eps: f64,
}

impl DuplexConfig {
    /// A tiny config for tests and CPU runs.
    pub fn tiny(layout: StreamLayout, text_vocab: usize, text_pad: u32) -> Self {
        Self {
            layout,
            text_vocab,
            audio_vocab: AUDIO_VOCAB as usize,
            text_pad,
            d_model: 32,
            n_layers: 2,
            q_heads: 4,
            kv_heads: 2,
            mlp_hidden: 64,
            window: None,
            depth_d: 24,
            depth_layers: 1,
            depth_heads: 2,
            depth_mlp_hidden: 48,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
        }
    }

    /// Vocabulary of stream `i` (text or audio).
    pub fn stream_vocab(&self, i: usize) -> usize {
        if self.layout.kind(i).is_text() {
            self.text_vocab
        } else {
            self.audio_vocab
        }
    }

    /// Per-stream pad tokens (the row fed for "nothing here").
    pub fn pad_row(&self) -> Vec<u32> {
        self.layout.pad_tokens(self.text_pad)
    }

    pub(crate) fn temporal_block(&self) -> BlockConfig {
        BlockConfig {
            attn: AttentionConfig {
                d_model: self.d_model,
                q_heads: self.q_heads,
                kv_heads: self.kv_heads,
                head_dim: self.d_model / self.q_heads,
                rope: RopeConfig {
                    theta: self.rope_theta,
                    layout: RopeLayout::Half,
                },
                mask: AttnMask {
                    causal: true,
                    window: self.window,
                    pinned_prefix: 0,
                },
            },
            mlp_hidden: self.mlp_hidden,
            norm_eps: self.norm_eps,
        }
    }

    pub(crate) fn depth_block(&self) -> BlockConfig {
        BlockConfig {
            attn: AttentionConfig {
                d_model: self.depth_d,
                q_heads: self.depth_heads,
                kv_heads: self.depth_heads,
                head_dim: self.depth_d / self.depth_heads,
                rope: RopeConfig {
                    theta: self.rope_theta,
                    layout: RopeLayout::Half,
                },
                mask: AttnMask {
                    causal: true,
                    ..Default::default()
                },
            },
            mlp_hidden: self.depth_mlp_hidden,
            norm_eps: self.norm_eps,
        }
    }
}
