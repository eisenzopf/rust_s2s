//! Pre-norm transformer block: `x += attn(norm(x)); x += mlp(norm(x))`.

use s2s_core::{Float, SplitMix64};

use crate::attention::{Attention, AttentionConfig, AttentionCtx, KvCache};
use crate::mlp::{MlpCtx, SwiGluMlp};
use crate::norm::{RmsNorm, RmsNormCtx};
use crate::util::add_assign;
use crate::{Module, Param};

/// Whether a block keeps its activations for the backward pass or recomputes
/// them (activation checkpointing: less memory, one extra forward).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Checkpoint {
    /// Keep all activations.
    #[default]
    Off,
    /// Keep only the block input; recompute the rest during backward.
    On,
}

/// Block hyper-parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockConfig {
    /// Attention settings (includes `d_model`).
    pub attn: AttentionConfig,
    /// MLP hidden width.
    pub mlp_hidden: usize,
    /// RMSNorm epsilon.
    pub norm_eps: f64,
}

/// One transformer block.
#[derive(Clone, Debug)]
pub struct Block<T> {
    /// Pre-attention norm.
    pub norm1: RmsNorm<T>,
    /// Self-attention.
    pub attn: Attention<T>,
    /// Pre-MLP norm.
    pub norm2: RmsNorm<T>,
    /// Feed-forward.
    pub mlp: SwiGluMlp<T>,
}

/// Activations saved by [`Block::forward`].
#[derive(Clone, Debug)]
pub enum BlockCtx<T> {
    /// Everything needed for backward.
    Full(Box<BlockSaved<T>>),
    /// Only the inputs; backward recomputes the forward pass first.
    Recompute {
        /// Block input.
        x: Vec<T>,
        /// Batch size.
        batch: usize,
        /// Sequence length.
        seq: usize,
        /// Token positions.
        positions: Vec<i64>,
    },
}

/// Full set of saved activations.
#[derive(Clone, Debug)]
pub struct BlockSaved<T> {
    norm1: RmsNormCtx<T>,
    attn: AttentionCtx<T>,
    norm2: RmsNormCtx<T>,
    mlp: MlpCtx<T>,
}

impl<T: Float> Block<T> {
    /// New block. Output projections are initialized with
    /// `std = 1 / sqrt(2 * n_layers * d_model)` so the residual stream stays
    /// well scaled at depth.
    pub fn new(name: &str, cfg: BlockConfig, n_layers: usize, rng: &mut SplitMix64) -> Self {
        let d = cfg.attn.d_model;
        let out_std = Some(1.0 / ((2 * n_layers.max(1) * d) as f64).sqrt());
        Self {
            norm1: RmsNorm::new(&format!("{name}.norm1"), d, cfg.norm_eps),
            attn: Attention::new(&format!("{name}.attn"), cfg.attn, out_std, rng),
            norm2: RmsNorm::new(&format!("{name}.norm2"), d, cfg.norm_eps),
            mlp: SwiGluMlp::new(&format!("{name}.mlp"), d, cfg.mlp_hidden, out_std, rng),
        }
    }

    fn forward_full(
        &self,
        x: &[T],
        batch: usize,
        seq: usize,
        positions: &[i64],
    ) -> (Vec<T>, BlockSaved<T>) {
        let rows = batch * seq;
        let (h1, norm1) = self.norm1.forward(x);
        let (a, attn) = self.attn.forward(&h1, batch, seq, positions);
        let mut x2 = x.to_vec();
        add_assign(&mut x2, &a);
        let (h2, norm2) = self.norm2.forward(&x2);
        let (m, mlp) = self.mlp.forward(&h2, rows);
        let mut y = x2;
        add_assign(&mut y, &m);
        (
            y,
            BlockSaved {
                norm1,
                attn,
                norm2,
                mlp,
            },
        )
    }

    /// Forward over `x: [batch * seq, d_model]`.
    pub fn forward(
        &self,
        x: &[T],
        batch: usize,
        seq: usize,
        positions: &[i64],
        ckpt: Checkpoint,
    ) -> (Vec<T>, BlockCtx<T>) {
        let (y, saved) = self.forward_full(x, batch, seq, positions);
        let ctx = match ckpt {
            Checkpoint::Off => BlockCtx::Full(Box::new(saved)),
            Checkpoint::On => BlockCtx::Recompute {
                x: x.to_vec(),
                batch,
                seq,
                positions: positions.to_vec(),
            },
        };
        (y, ctx)
    }

    /// Streaming step for one new token (batch 1), using and extending
    /// `cache`. `x` is `[d_model]`.
    pub fn step(&self, x: &[T], position: i64, cache: &mut KvCache<T>) -> Vec<T> {
        let h1 = self.norm1.forward(x).0;
        let a = self.attn.step(&h1, position, cache);
        let mut x2 = x.to_vec();
        add_assign(&mut x2, &a);
        let h2 = self.norm2.forward(&x2).0;
        let m = self.mlp.forward(&h2, 1).0;
        add_assign(&mut x2, &m);
        x2
    }

    /// Backward: accumulates parameter gradients, returns `dx`.
    pub fn backward(&mut self, ctx: BlockCtx<T>, dy: &[T]) -> Vec<T> {
        let saved = match ctx {
            BlockCtx::Full(s) => *s,
            BlockCtx::Recompute {
                x,
                batch,
                seq,
                positions,
            } => self.forward_full(&x, batch, seq, &positions).1,
        };
        // y = x2 + mlp(norm2(x2))  =>  dx2 = dy + norm2'(mlp'(dy))
        let dm = self.mlp.backward(saved.mlp, dy);
        let mut dx2 = self.norm2.backward(saved.norm2, &dm);
        add_assign(&mut dx2, dy);
        // x2 = x + attn(norm1(x))  =>  dx = dx2 + norm1'(attn'(dx2))
        let da = self.attn.backward(saved.attn, &dx2);
        let mut dx = self.norm1.backward(saved.norm1, &da);
        add_assign(&mut dx, &dx2);
        dx
    }
}

impl<T: Float> Module<T> for Block<T> {
    fn visit_params(&self, f: &mut dyn FnMut(&Param<T>)) {
        self.norm1.visit_params(f);
        self.attn.visit_params(f);
        self.norm2.visit_params(f);
        self.mlp.visit_params(f);
    }
    fn visit_params_mut(&mut self, f: &mut dyn FnMut(&mut Param<T>)) {
        self.norm1.visit_params_mut(f);
        self.attn.visit_params_mut(f);
        self.norm2.visit_params_mut(f);
        self.mlp.visit_params_mut(f);
    }
}
