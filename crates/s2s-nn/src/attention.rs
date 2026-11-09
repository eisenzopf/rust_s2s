//! Multi-head self-attention with GQA and RoPE.

use s2s_core::{Float, SplitMix64};
use s2s_kernels::reference::attention::{self as attn, AttnMask, AttnParams};
use s2s_kernels::reference::rope::{self, RopeConfig};

use crate::linear::{Linear, LinearCtx};
use crate::util::{add_assign, bhtd_to_bthd, bthd_to_bhtd};
use crate::{Module, Param};

/// Attention hyper-parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionConfig {
    /// Model width.
    pub d_model: usize,
    /// Query heads.
    pub q_heads: usize,
    /// Key/value heads (divides `q_heads`).
    pub kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Rotary embedding settings.
    pub rope: RopeConfig,
    /// Mask (causal, sliding window, pinned prefix).
    pub mask: AttnMask,
}

/// Self-attention layer.
#[derive(Clone, Debug)]
pub struct Attention<T> {
    /// Query projection `[q_heads * head_dim, d_model]`.
    pub wq: Linear<T>,
    /// Key projection `[kv_heads * head_dim, d_model]`.
    pub wk: Linear<T>,
    /// Value projection `[kv_heads * head_dim, d_model]`.
    pub wv: Linear<T>,
    /// Output projection `[d_model, q_heads * head_dim]`.
    pub wo: Linear<T>,
    cfg: AttentionConfig,
}

/// Per-layer key/value cache for streaming inference (batch 1).
///
/// Keys are stored after RoPE, row-major `[len, kv_heads, head_dim]`.
/// Everything is kept; a sliding-window mask in the config limits what each
/// query sees. (Evicting rows outside the window, and a ring buffer like the
/// GPU `kv_ring_write` kernel, is a memory optimization for later.)
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KvCache<T> {
    k: Vec<T>,
    v: Vec<T>,
    len: usize,
}

impl<T> KvCache<T> {
    /// An empty cache.
    pub fn new() -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            len: 0,
        }
    }

    /// Cached positions.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether nothing is cached yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Saved values for the backward pass.
#[derive(Clone, Debug)]
pub struct AttentionCtx<T> {
    wq: LinearCtx<T>,
    wk: LinearCtx<T>,
    wv: LinearCtx<T>,
    wo: LinearCtx<T>,
    /// Rotated q/k and v in `[b, h, t, d]`, attention output and lse.
    q: Vec<T>,
    k: Vec<T>,
    v: Vec<T>,
    o: Vec<T>,
    lse: Vec<T>,
    positions: Vec<i64>,
    batch: usize,
    seq: usize,
}

impl<T: Float> Attention<T> {
    /// New layer. `out_std` sets the output-projection init.
    pub fn new(
        name: &str,
        cfg: AttentionConfig,
        out_std: Option<f64>,
        rng: &mut SplitMix64,
    ) -> Self {
        let (d, hd) = (cfg.d_model, cfg.head_dim);
        Self {
            wq: Linear::new(&format!("{name}.wq"), d, cfg.q_heads * hd, false, None, rng),
            wk: Linear::new(
                &format!("{name}.wk"),
                d,
                cfg.kv_heads * hd,
                false,
                None,
                rng,
            ),
            wv: Linear::new(
                &format!("{name}.wv"),
                d,
                cfg.kv_heads * hd,
                false,
                None,
                rng,
            ),
            wo: Linear::new(
                &format!("{name}.wo"),
                cfg.q_heads * hd,
                d,
                false,
                out_std,
                rng,
            ),
            cfg,
        }
    }

    /// The configuration.
    pub fn config(&self) -> &AttentionConfig {
        &self.cfg
    }

    fn params(&self, batch: usize, seq: usize) -> AttnParams {
        AttnParams {
            batch,
            q_heads: self.cfg.q_heads,
            kv_heads: self.cfg.kv_heads,
            t: seq,
            s: seq,
            d: self.cfg.head_dim,
            scale: 1.0 / (self.cfg.head_dim as f64).sqrt(),
            q_offset: 0,
            mask: self.cfg.mask,
        }
    }

    /// Forward over `x: [batch * seq, d_model]`. `positions` gives each
    /// token's RoPE position (`batch * seq` entries), so packed sequences and
    /// injection segments can carry their own positions.
    pub fn forward(
        &self,
        x: &[T],
        batch: usize,
        seq: usize,
        positions: &[i64],
    ) -> (Vec<T>, AttentionCtx<T>) {
        let rows = batch * seq;
        let (hq, hkv, hd) = (self.cfg.q_heads, self.cfg.kv_heads, self.cfg.head_dim);
        assert_eq!(positions.len(), rows, "one position per token");
        let (mut q, wq) = self.wq.forward(x, rows);
        let (mut k, wk) = self.wk.forward(x, rows);
        let (v, wv) = self.wv.forward(x, rows);
        rope::rope_fwd(&mut q, positions, rows, hq, hd, self.cfg.rope);
        rope::rope_fwd(&mut k, positions, rows, hkv, hd, self.cfg.rope);
        let q = bthd_to_bhtd(&q, batch, seq, hq, hd);
        let k = bthd_to_bhtd(&k, batch, seq, hkv, hd);
        let v = bthd_to_bhtd(&v, batch, seq, hkv, hd);
        let p = self.params(batch, seq);
        let mut o = vec![T::zero(); q.len()];
        let mut lse = vec![T::zero(); batch * hq * seq];
        attn::attention_fwd(&p, &q, &k, &v, &mut o, &mut lse);
        let o_flat = bhtd_to_bthd(&o, batch, hq, seq, hd);
        let (y, wo) = self.wo.forward(&o_flat, rows);
        let ctx = AttentionCtx {
            wq,
            wk,
            wv,
            wo,
            q,
            k,
            v,
            o,
            lse,
            positions: positions.to_vec(),
            batch,
            seq,
        };
        (y, ctx)
    }

    /// Streaming step for one new token (batch 1): appends its key/value to
    /// `cache` and attends over everything cached so far. `x` is `[d_model]`;
    /// returns `[d_model]`. Equivalent to the last row of [`forward`] over the
    /// whole sequence.
    ///
    /// [`forward`]: Self::forward
    pub fn step(&self, x: &[T], position: i64, cache: &mut KvCache<T>) -> Vec<T> {
        let (hq, hkv, hd) = (self.cfg.q_heads, self.cfg.kv_heads, self.cfg.head_dim);
        let (mut q, _) = self.wq.forward(x, 1);
        let (mut k, _) = self.wk.forward(x, 1);
        let (v, _) = self.wv.forward(x, 1);
        rope::rope_fwd(&mut q, &[position], 1, hq, hd, self.cfg.rope);
        rope::rope_fwd(&mut k, &[position], 1, hkv, hd, self.cfg.rope);
        cache.k.extend_from_slice(&k);
        cache.v.extend_from_slice(&v);
        cache.len += 1;
        let s = cache.len;
        let kc = bthd_to_bhtd(&cache.k, 1, s, hkv, hd);
        let vc = bthd_to_bhtd(&cache.v, 1, s, hkv, hd);
        let p = AttnParams {
            t: 1,
            s,
            q_offset: s - 1,
            ..self.params(1, 1)
        };
        let mut o = vec![T::zero(); hq * hd];
        let mut lse = vec![T::zero(); hq];
        attn::attention_fwd(&p, &q, &kc, &vc, &mut o, &mut lse);
        // o is [1, hq, 1, hd] == [hq * hd] row-major, same as [1, 1, hq, hd].
        self.wo.forward(&o, 1).0
    }

    /// Backward: accumulates parameter gradients, returns `dx`.
    pub fn backward(&mut self, ctx: AttentionCtx<T>, dy: &[T]) -> Vec<T> {
        let (b, t) = (ctx.batch, ctx.seq);
        let rows = b * t;
        let (hq, hkv, hd) = (self.cfg.q_heads, self.cfg.kv_heads, self.cfg.head_dim);
        let do_flat = self.wo.backward(ctx.wo, dy);
        let d_o = bthd_to_bhtd(&do_flat, b, t, hq, hd);
        let mut dq = vec![T::zero(); ctx.q.len()];
        let mut dk = vec![T::zero(); ctx.k.len()];
        let mut dv = vec![T::zero(); ctx.v.len()];
        let p = self.params(b, t);
        attn::attention_bwd(
            &p, &ctx.q, &ctx.k, &ctx.v, &ctx.o, &ctx.lse, &d_o, &mut dq, &mut dk, &mut dv,
        );
        let mut dq = bhtd_to_bthd(&dq, b, hq, t, hd);
        let mut dk = bhtd_to_bthd(&dk, b, hkv, t, hd);
        let dv = bhtd_to_bthd(&dv, b, hkv, t, hd);
        rope::rope_bwd(&mut dq, &ctx.positions, rows, hq, hd, self.cfg.rope);
        rope::rope_bwd(&mut dk, &ctx.positions, rows, hkv, hd, self.cfg.rope);
        let mut dx = self.wq.backward(ctx.wq, &dq);
        add_assign(&mut dx, &self.wk.backward(ctx.wk, &dk));
        add_assign(&mut dx, &self.wv.backward(ctx.wv, &dv));
        dx
    }
}

impl<T: Float> Module<T> for Attention<T> {
    fn visit_params(&self, f: &mut dyn FnMut(&Param<T>)) {
        self.wq.visit_params(f);
        self.wk.visit_params(f);
        self.wv.visit_params(f);
        self.wo.visit_params(f);
    }
    fn visit_params_mut(&mut self, f: &mut dyn FnMut(&mut Param<T>)) {
        self.wq.visit_params_mut(f);
        self.wk.visit_params_mut(f);
        self.wv.visit_params_mut(f);
        self.wo.visit_params_mut(f);
    }
}
