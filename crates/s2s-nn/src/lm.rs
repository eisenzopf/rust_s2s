//! A decoder-only transformer language model.
//!
//! Used for trainer bring-up (plan M5: "XS config trained on a text corpus")
//! and as the shape of the Temporal Transformer backbone.

use s2s_core::{Float, SplitMix64};
use s2s_kernels::reference::attention::AttnMask;
use s2s_kernels::reference::loss;
use s2s_kernels::reference::rope::{RopeConfig, RopeLayout};

use crate::attention::AttentionConfig;
use crate::block::{Block, BlockConfig, BlockCtx, Checkpoint};
use crate::embedding::{EmbeddingCtx, MultiEmbedding};
use crate::linear::{Linear, LinearCtx};
use crate::norm::{RmsNorm, RmsNormCtx};
use crate::{Module, Param};

/// Model hyper-parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LmConfig {
    /// Vocabulary size.
    pub vocab: usize,
    /// Model width.
    pub d_model: usize,
    /// Number of blocks.
    pub n_layers: usize,
    /// Query heads.
    pub q_heads: usize,
    /// Key/value heads.
    pub kv_heads: usize,
    /// MLP hidden width.
    pub mlp_hidden: usize,
    /// RoPE base.
    pub rope_theta: f64,
    /// RMSNorm epsilon.
    pub norm_eps: f64,
}

impl LmConfig {
    /// A tiny config for tests and CPU demos.
    pub fn tiny(vocab: usize) -> Self {
        Self {
            vocab,
            d_model: 32,
            n_layers: 2,
            q_heads: 4,
            kv_heads: 2,
            mlp_hidden: 64,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
        }
    }

    fn block(&self) -> BlockConfig {
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
                    ..Default::default()
                },
            },
            mlp_hidden: self.mlp_hidden,
            norm_eps: self.norm_eps,
        }
    }
}

/// Decoder-only transformer LM: embedding → blocks → final norm → head.
#[derive(Clone, Debug)]
pub struct TransformerLm<T> {
    /// Token embedding.
    pub embed: MultiEmbedding<T>,
    /// Transformer blocks.
    pub blocks: Vec<Block<T>>,
    /// Final norm.
    pub norm: RmsNorm<T>,
    /// Output head `[vocab, d_model]` (untied).
    pub head: Linear<T>,
    cfg: LmConfig,
}

/// Saved activations for [`TransformerLm::backward`].
#[derive(Debug)]
pub struct LmCtx<T> {
    embed: EmbeddingCtx,
    blocks: Vec<BlockCtx<T>>,
    norm: RmsNormCtx<T>,
    head: LinearCtx<T>,
    logits: Vec<T>,
    lse: Vec<T>,
    total_weight: T,
}

impl<T: Float> TransformerLm<T> {
    /// New model with seeded random init.
    pub fn new(cfg: LmConfig, rng: &mut SplitMix64) -> Self {
        assert!(
            cfg.d_model.is_multiple_of(cfg.q_heads),
            "d_model must divide by q_heads"
        );
        let bcfg = cfg.block();
        Self {
            embed: MultiEmbedding::new("embed", &[cfg.vocab], cfg.d_model, 1.0, rng),
            blocks: (0..cfg.n_layers)
                .map(|i| Block::new(&format!("blocks.{i}"), bcfg, cfg.n_layers, rng))
                .collect(),
            norm: RmsNorm::new("norm", cfg.d_model, cfg.norm_eps),
            head: Linear::new("head", cfg.d_model, cfg.vocab, false, None, rng),
            cfg,
        }
    }

    /// The configuration.
    pub fn config(&self) -> &LmConfig {
        &self.cfg
    }

    /// Logits `[batch * seq, vocab]` for `tokens: [batch, seq]` (no loss).
    pub fn logits(&self, tokens: &[u32], batch: usize, seq: usize) -> Vec<T> {
        let rows = batch * seq;
        let positions = positions(batch, seq);
        let mut h = self.embed.forward(tokens, rows).0;
        for b in &self.blocks {
            h = b.forward(&h, batch, seq, &positions, Checkpoint::Off).0;
        }
        let h = self.norm.forward(&h).0;
        self.head.forward(&h, rows).0
    }

    /// Forward with mean cross-entropy loss against `targets: [batch, seq]`
    /// (use `loss::IGNORE_INDEX` to skip positions). Returns the loss and the
    /// context for [`backward`](Self::backward).
    pub fn forward_loss(
        &self,
        tokens: &[u32],
        targets: &[i64],
        batch: usize,
        seq: usize,
        ckpt: Checkpoint,
    ) -> (T, LmCtx<T>) {
        let rows = batch * seq;
        assert_eq!(tokens.len(), rows);
        assert_eq!(targets.len(), rows);
        let positions = positions(batch, seq);
        let (mut h, embed) = self.embed.forward(tokens, rows);
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            let (y, c) = b.forward(&h, batch, seq, &positions, ckpt);
            h = y;
            blocks.push(c);
        }
        let (hn, norm) = self.norm.forward(&h);
        let (logits, head) = self.head.forward(&hn, rows);
        let mut lse = vec![T::zero(); rows];
        let (sum, total) =
            loss::cross_entropy_fwd(&logits, targets, None, &mut lse, rows, self.cfg.vocab);
        let mean = if total > T::zero() {
            sum / total
        } else {
            T::zero()
        };
        let ctx = LmCtx {
            embed,
            blocks,
            norm,
            head,
            logits,
            lse,
            total_weight: total,
        };
        (mean, ctx)
    }

    /// Backward from the mean loss (`dloss = 1`). Accumulates gradients.
    pub fn backward(&mut self, ctx: LmCtx<T>, targets: &[i64]) {
        let rows = ctx.lse.len();
        let mut dlogits = vec![T::zero(); ctx.logits.len()];
        if ctx.total_weight > T::zero() {
            let scale = T::one() / ctx.total_weight;
            loss::cross_entropy_bwd(
                &ctx.logits,
                targets,
                None,
                &ctx.lse,
                scale,
                &mut dlogits,
                rows,
                self.cfg.vocab,
            );
        }
        let dh = self.head.backward(ctx.head, &dlogits);
        let mut dh = self.norm.backward(ctx.norm, &dh);
        for (b, c) in self.blocks.iter_mut().zip(ctx.blocks).rev() {
            dh = b.backward(c, &dh);
        }
        self.embed.backward(ctx.embed, &dh);
    }
}

/// Positions `0..seq` for every sequence in the batch.
fn positions(batch: usize, seq: usize) -> Vec<i64> {
    (0..batch).flat_map(|_| 0..seq as i64).collect()
}

impl<T: Float> Module<T> for TransformerLm<T> {
    fn visit_params(&self, f: &mut dyn FnMut(&Param<T>)) {
        self.embed.visit_params(f);
        for b in &self.blocks {
            b.visit_params(f);
        }
        self.norm.visit_params(f);
        self.head.visit_params(f);
    }
    fn visit_params_mut(&mut self, f: &mut dyn FnMut(&mut Param<T>)) {
        self.embed.visit_params_mut(f);
        for b in &mut self.blocks {
            b.visit_params_mut(f);
        }
        self.norm.visit_params_mut(f);
        self.head.visit_params_mut(f);
    }
}
