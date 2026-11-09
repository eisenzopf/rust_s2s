use s2s_core::{cast, Float, SplitMix64};
use s2s_kernels::reference::embedding::NO_TOKEN;
use s2s_kernels::reference::loss;
use s2s_nn::block::{Block, BlockCtx, Checkpoint};
use s2s_nn::embedding::{EmbeddingCtx, MultiEmbedding};
use s2s_nn::linear::{Linear, LinearCtx};
use s2s_nn::norm::{RmsNorm, RmsNormCtx};
use s2s_nn::{Module, Param};

use crate::DuplexConfig;

/// One training batch in model terms (see [`DuplexBatch::from_data`]).
#[derive(Clone, Debug, PartialEq)]
pub struct DuplexBatch<T> {
    /// `B`.
    pub batch: usize,
    /// `T` positions per window.
    pub window: usize,
    /// `[B, T, S]` input tokens.
    pub tokens: Vec<u32>,
    /// `[B, T, K]` targets (`IGNORE_INDEX` = no loss).
    pub targets: Vec<i64>,
    /// `[B, T, K]` loss weights.
    pub weights: Vec<T>,
    /// `[B, T]` 1 where the position is an injected text token.
    pub is_injection: Vec<u8>,
    /// `[B, T]` RoPE positions.
    pub positions: Vec<i64>,
}

impl<T: Float> DuplexBatch<T> {
    /// Convert a batch from the `s2s-data` loader.
    ///
    /// Use the loader's crop mode (one conversation per window): attention
    /// does not yet mask across packed segments.
    pub fn from_data(b: &s2s_data::Batch) -> Self {
        Self {
            batch: b.batch_size,
            window: b.window,
            tokens: b.tokens.clone(),
            targets: b.targets.clone(),
            weights: b.weights.iter().map(|&w| cast(w as f64)).collect(),
            is_injection: b.is_injection.clone(),
            positions: b.positions.iter().map(|&p| p as i64).collect(),
        }
    }
}

/// Per-generated-stream loss bookkeeping.
#[derive(Clone, Debug, PartialEq)]
pub struct LossReport {
    /// Weighted mean loss over all generated streams.
    pub loss: f64,
    /// `(weighted loss sum, total weight)` per generated stream `k`.
    pub per_stream: Vec<(f64, f64)>,
}

impl LossReport {
    /// Mean loss of generated stream `k` (0 if it had no weight).
    pub fn stream_mean(&self, k: usize) -> f64 {
        let (s, w) = self.per_stream[k];
        if w > 0.0 {
            s / w
        } else {
            0.0
        }
    }
}

/// The multistream model.
#[derive(Clone, Debug)]
pub struct DuplexModel<T> {
    /// One embedding table per stream, summed per position.
    pub embed: MultiEmbedding<T>,
    /// Added at injection positions (`e_inject`, plan §3.5).
    pub inject: Param<T>,
    /// Temporal Transformer blocks.
    pub blocks: Vec<Block<T>>,
    /// Final temporal norm, producing `z`.
    pub norm: RmsNorm<T>,
    /// Per-sub-step input projections `z -> depth_d`.
    pub depth_in: Vec<Linear<T>>,
    /// Per-sub-step embeddings of the previous sub-step's token (sub-steps 1..K).
    pub depth_emb: Vec<MultiEmbedding<T>>,
    /// Depth Transformer blocks (shared across sub-steps).
    pub depth_blocks: Vec<Block<T>>,
    /// Final depth norm.
    pub depth_norm: RmsNorm<T>,
    /// Per-sub-step output heads `depth_d -> vocab_k`.
    pub heads: Vec<Linear<T>>,
    pub(crate) cfg: DuplexConfig,
}

struct HeadCtx<T> {
    head: LinearCtx<T>,
    logits: Vec<T>,
    lse: Vec<T>,
    targets: Vec<i64>,
    weights: Vec<T>,
}

/// Saved activations from [`DuplexModel::forward_loss`].
pub struct DuplexCtx<T> {
    rows: usize,
    is_injection: Vec<u8>,
    embed: EmbeddingCtx,
    blocks: Vec<BlockCtx<T>>,
    norm: RmsNormCtx<T>,
    depth_in: Vec<LinearCtx<T>>,
    depth_emb: Vec<EmbeddingCtx>,
    depth_blocks: Vec<BlockCtx<T>>,
    depth_norm: RmsNormCtx<T>,
    heads: Vec<HeadCtx<T>>,
    total_weight: T,
}

impl<T> std::fmt::Debug for DuplexCtx<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuplexCtx")
            .field("rows", &self.rows)
            .finish_non_exhaustive()
    }
}

impl<T: Float> DuplexCtx<T> {
    /// Logits of generated stream `k`, `[B * T, vocab_k]`.
    pub fn logits(&self, k: usize) -> &[T] {
        &self.heads[k].logits
    }
}

impl<T: Float> DuplexModel<T> {
    /// New model with seeded random init.
    pub fn new(cfg: DuplexConfig, rng: &mut SplitMix64) -> Self {
        let l = cfg.layout;
        let (s, k) = (l.num_streams(), l.num_generated());
        assert!(cfg.d_model.is_multiple_of(cfg.q_heads));
        assert!(cfg.depth_d.is_multiple_of(cfg.depth_heads));
        assert!(
            cfg.audio_vocab > s2s_core::streams::AUDIO_PAD as usize,
            "audio_vocab must include AUDIO_PAD ({}), which pads audio streams",
            s2s_core::streams::AUDIO_PAD
        );
        assert!(
            (cfg.text_pad as usize) < cfg.text_vocab,
            "text_pad must be in the text vocab"
        );
        let vocabs: Vec<usize> = (0..s).map(|i| cfg.stream_vocab(i)).collect();
        // Sum of S embeddings: scale each so the sum has roughly unit variance.
        let emb_std = 1.0 / (s as f64).sqrt();
        let tb = cfg.temporal_block();
        let db = cfg.depth_block();
        Self {
            embed: MultiEmbedding::new("embed", &vocabs, cfg.d_model, emb_std, rng),
            inject: Param::normal("embed.inject", &[cfg.d_model], emb_std, rng),
            blocks: (0..cfg.n_layers)
                .map(|i| Block::new(&format!("blocks.{i}"), tb, cfg.n_layers, rng))
                .collect(),
            norm: RmsNorm::new("norm", cfg.d_model, cfg.norm_eps),
            depth_in: (0..k)
                .map(|i| {
                    Linear::new(
                        &format!("depth.in{i}"),
                        cfg.d_model,
                        cfg.depth_d,
                        false,
                        None,
                        rng,
                    )
                })
                .collect(),
            depth_emb: (1..k)
                .map(|i| {
                    MultiEmbedding::new(
                        &format!("embed.depth{i}"),
                        &[vocabs[i - 1]],
                        cfg.depth_d,
                        1.0,
                        rng,
                    )
                })
                .collect(),
            depth_blocks: (0..cfg.depth_layers)
                .map(|i| Block::new(&format!("depth.blocks.{i}"), db, cfg.depth_layers, rng))
                .collect(),
            depth_norm: RmsNorm::new("depth.norm", cfg.depth_d, cfg.norm_eps),
            heads: (0..k)
                .map(|i| {
                    Linear::new(
                        &format!("depth.head{i}"),
                        cfg.depth_d,
                        vocabs[i],
                        false,
                        None,
                        rng,
                    )
                })
                .collect(),
            cfg,
        }
    }

    /// The configuration.
    pub fn config(&self) -> &DuplexConfig {
        &self.cfg
    }

    /// Training forward pass with the weighted loss over all generated
    /// streams. Returns `(mean loss, per-stream report, context)`.
    pub fn forward_loss(
        &self,
        b: &DuplexBatch<T>,
        ckpt: Checkpoint,
    ) -> (T, LossReport, DuplexCtx<T>) {
        let l = self.cfg.layout;
        let (s, k_gen) = (l.num_streams(), l.num_generated());
        let (d, dd) = (self.cfg.d_model, self.cfg.depth_d);
        let rows = b.batch * b.window;
        assert_eq!(b.tokens.len(), rows * s, "tokens must be [B, T, S]");
        assert_eq!(b.targets.len(), rows * k_gen, "targets must be [B, T, K]");
        assert_eq!(b.weights.len(), rows * k_gen);
        assert_eq!(b.is_injection.len(), rows);
        assert_eq!(b.positions.len(), rows);

        // Temporal Transformer.
        let (mut h, embed) = self.embed.forward(&b.tokens, rows);
        for r in (0..rows).filter(|&r| b.is_injection[r] != 0) {
            for j in 0..d {
                h[r * d + j] = h[r * d + j] + self.inject.value[j];
            }
        }
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            let (y, c) = blk.forward(&h, b.batch, b.window, &b.positions, ckpt);
            h = y;
            blocks.push(c);
        }
        let (z, norm) = self.norm.forward(&h);

        // Depth input: per sub-step projection of z plus (teacher-forced)
        // embedding of the previous sub-step's target token.
        let mut x = vec![T::zero(); rows * k_gen * dd];
        let mut depth_in = Vec::with_capacity(k_gen);
        let mut depth_emb = Vec::with_capacity(k_gen.saturating_sub(1));
        for k in 0..k_gen {
            let (yk, c) = self.depth_in[k].forward(&z, rows);
            add_into_substep(&mut x, &yk, rows, k_gen, dd, k);
            depth_in.push(c);
            if k >= 1 {
                let prev = substep_tokens(&b.targets, rows, k_gen, k - 1);
                let (ek, c) = self.depth_emb[k - 1].forward(&prev, rows);
                add_into_substep(&mut x, &ek, rows, k_gen, dd, k);
                depth_emb.push(c);
            }
        }
        let dpos: Vec<i64> = (0..rows).flat_map(|_| 0..k_gen as i64).collect();
        let mut depth_blocks = Vec::with_capacity(self.depth_blocks.len());
        for blk in &self.depth_blocks {
            let (y, c) = blk.forward(&x, rows, k_gen, &dpos, ckpt);
            x = y;
            depth_blocks.push(c);
        }
        let (hn, depth_norm) = self.depth_norm.forward(&x);

        // Heads and weighted cross-entropy per sub-step.
        let mut heads = Vec::with_capacity(k_gen);
        let (mut sum, mut total) = (T::zero(), T::zero());
        let mut per_stream = Vec::with_capacity(k_gen);
        for k in 0..k_gen {
            let hk = gather_substep(&hn, rows, k_gen, dd, k);
            let (logits, head) = self.heads[k].forward(&hk, rows);
            let v = self.heads[k].d_out();
            let targets: Vec<i64> = (0..rows).map(|r| b.targets[r * k_gen + k]).collect();
            let weights: Vec<T> = (0..rows).map(|r| b.weights[r * k_gen + k]).collect();
            let mut lse = vec![T::zero(); rows];
            let (sk, tk) =
                loss::cross_entropy_fwd(&logits, &targets, Some(&weights), &mut lse, rows, v);
            sum = sum + sk;
            total = total + tk;
            per_stream.push((
                sk.to_f64().unwrap_or(f64::NAN),
                tk.to_f64().unwrap_or(f64::NAN),
            ));
            heads.push(HeadCtx {
                head,
                logits,
                lse,
                targets,
                weights,
            });
        }
        let mean = if total > T::zero() {
            sum / total
        } else {
            T::zero()
        };
        let report = LossReport {
            loss: mean.to_f64().unwrap_or(f64::NAN),
            per_stream,
        };
        let ctx = DuplexCtx {
            rows,
            is_injection: b.is_injection.clone(),
            embed,
            blocks,
            norm,
            depth_in,
            depth_emb,
            depth_blocks,
            depth_norm,
            heads,
            total_weight: total,
        };
        (mean, report, ctx)
    }

    /// Backward from the mean loss (`dloss = 1`). Accumulates gradients.
    pub fn backward(&mut self, ctx: DuplexCtx<T>) {
        let l = self.cfg.layout;
        let k_gen = l.num_generated();
        let (d, dd) = (self.cfg.d_model, self.cfg.depth_d);
        let rows = ctx.rows;
        if ctx.total_weight <= T::zero() {
            return;
        }
        let scale = T::one() / ctx.total_weight;

        let mut dhn = vec![T::zero(); rows * k_gen * dd];
        for (k, hc) in ctx.heads.into_iter().enumerate() {
            let v = self.heads[k].d_out();
            let mut dlogits = vec![T::zero(); hc.logits.len()];
            loss::cross_entropy_bwd(
                &hc.logits,
                &hc.targets,
                Some(&hc.weights),
                &hc.lse,
                scale,
                &mut dlogits,
                rows,
                v,
            );
            let dhk = self.heads[k].backward(hc.head, &dlogits);
            add_into_substep(&mut dhn, &dhk, rows, k_gen, dd, k);
        }
        let mut dx = self.depth_norm.backward(ctx.depth_norm, &dhn);
        for (blk, c) in self.depth_blocks.iter_mut().zip(ctx.depth_blocks).rev() {
            dx = blk.backward(c, &dx);
        }

        let mut dz = vec![T::zero(); rows * d];
        let mut emb_ctx = ctx.depth_emb.into_iter();
        for (k, c) in ctx.depth_in.into_iter().enumerate() {
            let dxk = gather_substep(&dx, rows, k_gen, dd, k);
            let g = self.depth_in[k].backward(c, &dxk);
            for (a, b) in dz.iter_mut().zip(&g) {
                *a = *a + *b;
            }
            if k >= 1 {
                let ec = emb_ctx
                    .next()
                    .expect("one embedding context per sub-step >= 1");
                self.depth_emb[k - 1].backward(ec, &dxk);
            }
        }

        let mut dh = self.norm.backward(ctx.norm, &dz);
        for (blk, c) in self.blocks.iter_mut().zip(ctx.blocks).rev() {
            dh = blk.backward(c, &dh);
        }
        for r in (0..rows).filter(|&r| ctx.is_injection[r] != 0) {
            for j in 0..d {
                self.inject.grad[j] = self.inject.grad[j] + dh[r * d + j];
            }
        }
        self.embed.backward(ctx.embed, &dh);
    }
}

/// Tokens of sub-step `k` from `[rows, K]` targets, with ignored targets
/// mapped to `NO_TOKEN` (they contribute no embedding).
pub(crate) fn substep_tokens(targets: &[i64], rows: usize, k_gen: usize, k: usize) -> Vec<u32> {
    (0..rows)
        .map(|r| {
            let t = targets[r * k_gen + k];
            if t < 0 {
                NO_TOKEN
            } else {
                t as u32
            }
        })
        .collect()
}

/// `x[(r, k), :] += y[r, :]` for `x: [rows, K, dd]`.
fn add_into_substep<T: Float>(
    x: &mut [T],
    y: &[T],
    rows: usize,
    k_gen: usize,
    dd: usize,
    k: usize,
) {
    for r in 0..rows {
        let dst = (r * k_gen + k) * dd;
        for j in 0..dd {
            x[dst + j] = x[dst + j] + y[r * dd + j];
        }
    }
}

/// `x[(r, k), :]` for all `r`, as `[rows, dd]`.
fn gather_substep<T: Float>(x: &[T], rows: usize, k_gen: usize, dd: usize, k: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(rows * dd);
    for r in 0..rows {
        let src = (r * k_gen + k) * dd;
        out.extend_from_slice(&x[src..src + dd]);
    }
    out
}

impl<T: Float> Module<T> for DuplexModel<T> {
    fn visit_params(&self, f: &mut dyn FnMut(&Param<T>)) {
        self.embed.visit_params(f);
        f(&self.inject);
        self.blocks.iter().for_each(|b| b.visit_params(f));
        self.norm.visit_params(f);
        self.depth_in.iter().for_each(|m| m.visit_params(f));
        self.depth_emb.iter().for_each(|m| m.visit_params(f));
        self.depth_blocks.iter().for_each(|b| b.visit_params(f));
        self.depth_norm.visit_params(f);
        self.heads.iter().for_each(|m| m.visit_params(f));
    }
    fn visit_params_mut(&mut self, f: &mut dyn FnMut(&mut Param<T>)) {
        self.embed.visit_params_mut(f);
        f(&mut self.inject);
        self.blocks.iter_mut().for_each(|b| b.visit_params_mut(f));
        self.norm.visit_params_mut(f);
        self.depth_in.iter_mut().for_each(|m| m.visit_params_mut(f));
        self.depth_emb
            .iter_mut()
            .for_each(|m| m.visit_params_mut(f));
        self.depth_blocks
            .iter_mut()
            .for_each(|b| b.visit_params_mut(f));
        self.depth_norm.visit_params_mut(f);
        self.heads.iter_mut().for_each(|m| m.visit_params_mut(f));
    }
}
