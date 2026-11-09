//! Inference-only transformer blocks with Kyutai's numerics.
//!
//! Why not `s2s-nn`: its layers are built for training (explicit backward,
//! separate q/k/v projections, half-split RoPE with f64 angles, SwiGLU with a
//! hidden size given directly). Kyutai's checkpoints need a fused `in_proj`,
//! interleaved RoPE with f32 angles, RMSNorm with `eps = 1e-8` *and* LayerNorm,
//! layer scale, sinusoidal positions, several gating activations with
//! Kyutai's hidden-size rule, and a rotating KV cache whose key order matches
//! candle's. These blocks call the `s2s_kernels::reference` kernels directly.
//!
//! Every block processes one frame (one row) at a time: streaming decode.

use s2s_kernels::reference::attention::{attention_decode, AttnMask};
use s2s_kernels::reference::elementwise::{self, glu_fwd};
use s2s_kernels::reference::gemm::linear_fwd;
use s2s_kernels::reference::layernorm::layernorm_fwd;
use s2s_kernels::reference::norm::rmsnorm_fwd;
use s2s_kernels::reference::rope::{rope_fwd_tables, rope_tables_f32, RopeLayout};

use super::config::{Activation, NormType, PositionalEmbedding, TransformerConfig};
use super::error::{Error, Result};
use super::weights::{join, Loader};

/// `y = x W^T (+ b)` with `W: [d_out, d_in]`.
#[derive(Clone, Debug)]
pub(crate) struct Linear {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
    pub d_in: usize,
    pub d_out: usize,
}

impl Linear {
    /// Load `prefix.weight` (and `prefix.bias` when `bias`).
    pub fn load(
        l: &mut Loader,
        prefix: &str,
        d_in: usize,
        d_out: usize,
        bias: bool,
    ) -> Result<Self> {
        let w = l.get(&join(prefix, "weight"), &[d_out, d_in])?;
        let b = if bias {
            Some(l.get(&join(prefix, "bias"), &[d_out])?)
        } else {
            None
        };
        Ok(Self { w, b, d_in, d_out })
    }

    pub fn from_parts(w: Vec<f32>, b: Option<Vec<f32>>, d_in: usize, d_out: usize) -> Self {
        Self { w, b, d_in, d_out }
    }

    /// `x: [rows, d_in]` → `[rows, d_out]`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let rows = x.len() / self.d_in;
        let mut y = vec![0.0; rows * self.d_out];
        linear_fwd(
            x,
            &self.w,
            self.b.as_deref(),
            &mut y,
            rows,
            self.d_in,
            self.d_out,
        );
        y
    }
}

/// RMSNorm (`eps = 1e-8`) or LayerNorm (`eps = 1e-5`), as in Kyutai's `Norm`.
#[derive(Clone, Debug)]
pub(crate) enum Norm {
    Rms { alpha: Vec<f32> },
    Layer { w: Vec<f32>, b: Vec<f32> },
}

const RMS_EPS: f32 = 1e-8;
const LN_EPS: f32 = 1e-5;

impl Norm {
    pub fn load(l: &mut Loader, prefix: &str, d: usize, ty: NormType) -> Result<Self> {
        Ok(match ty {
            NormType::RmsNorm => Norm::Rms {
                alpha: l.get(&join(prefix, "alpha"), &[1, 1, d])?,
            },
            NormType::LayerNorm => {
                let b = l.get(&join(prefix, "bias"), &[d])?;
                let alpha = join(prefix, "alpha");
                let w = if l.contains(&alpha) {
                    l.get(&alpha, &[1, 1, d])?
                } else {
                    l.get(&join(prefix, "weight"), &[d])?
                };
                Norm::Layer { w, b }
            }
        })
    }

    /// Normalize each row of `x: [rows, d]`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0; x.len()];
        match self {
            Norm::Rms { alpha } => {
                let d = alpha.len();
                let rows = x.len() / d;
                let mut rstd = vec![0.0; rows];
                rmsnorm_fwd(x, alpha, RMS_EPS as f64, &mut y, &mut rstd, rows, d);
            }
            Norm::Layer { w, b } => {
                let d = w.len();
                layernorm_fwd(x, Some(w), Some(b), LN_EPS as f64, &mut y, x.len() / d, d);
            }
        }
        y
    }
}

/// Map a config activation to the kernel's.
pub(crate) fn kernel_activation(a: Activation) -> Result<elementwise::Activation> {
    use elementwise::Activation as K;
    Ok(match a {
        Activation::Gelu => K::GeluErf,
        Activation::NewGelu | Activation::GeluPytorchTanh => K::GeluTanh,
        Activation::Relu => K::Relu,
        Activation::Relu2 => K::Relu2,
        Activation::Relu6 => K::Relu6,
        Activation::Silu => K::Silu,
        Activation::Sigmoid => K::Sigmoid,
        Activation::HardSigmoid => K::HardSigmoid,
        Activation::Swish => K::Swish,
        Activation::Mish => K::Mish,
        Activation::HardSwish => K::HardSwish,
        Activation::Elu(a) => K::Elu(a),
        Activation::LeakyRelu(s) => K::LeakyRelu(s),
        Activation::Swiglu => {
            return Err(Error::Unsupported(
                "swiglu is not a valid MLP gating activation".into(),
            ))
        }
    })
}

/// Kyutai's MLP: gated (`gating.linear_in → act(first half) * second half →
/// gating.linear_out`) or ungated (`linear1 → GELU(erf) → linear2`).
#[derive(Clone, Debug)]
pub(crate) enum Mlp {
    NoGating {
        linear1: Linear,
        linear2: Linear,
    },
    Gating {
        linear_in: Linear,
        linear_out: Linear,
        act: elementwise::Activation,
        hidden: usize,
    },
}

impl Mlp {
    pub fn load(l: &mut Loader, prefix: &str, cfg: &TransformerConfig) -> Result<Self> {
        let d = cfg.d_model;
        Ok(match cfg.gating {
            None => Mlp::NoGating {
                linear1: Linear::load(
                    l,
                    &join(prefix, "linear1"),
                    d,
                    cfg.dim_feedforward,
                    cfg.bias_ff,
                )?,
                linear2: Linear::load(
                    l,
                    &join(prefix, "linear2"),
                    cfg.dim_feedforward,
                    d,
                    cfg.bias_ff,
                )?,
            },
            Some(a) => {
                let hidden = cfg.gated_hidden();
                let p = join(prefix, "gating");
                Mlp::Gating {
                    linear_in: Linear::load(l, &join(&p, "linear_in"), d, 2 * hidden, cfg.bias_ff)?,
                    linear_out: Linear::load(l, &join(&p, "linear_out"), hidden, d, cfg.bias_ff)?,
                    act: kernel_activation(a)?,
                    hidden,
                }
            }
        })
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        match self {
            Mlp::NoGating { linear1, linear2 } => {
                let mut h = linear1.forward(x);
                elementwise::activation_fwd(elementwise::Activation::GeluErf, &mut h);
                linear2.forward(&h)
            }
            Mlp::Gating {
                linear_in,
                linear_out,
                act,
                hidden,
            } => {
                let y = linear_in.forward(x);
                let rows = y.len() / (2 * hidden);
                let mut h = vec![0.0; rows * hidden];
                for r in 0..rows {
                    let row = &y[r * 2 * hidden..(r + 1) * 2 * hidden];
                    glu_fwd(
                        *act,
                        &row[..*hidden],
                        &row[*hidden..],
                        &mut h[r * hidden..(r + 1) * hidden],
                    );
                }
                linear_out.forward(&h)
            }
        }
    }
}

/// Rotating KV cache holding the last `capacity` positions, with candle's
/// `RotatingKvCache` slot order (slot `i` is overwritten every `capacity`
/// steps) so attention sums over keys in the same order as Kyutai.
///
/// Layout: `[heads, capacity, head_dim]`.
#[derive(Clone, Debug)]
pub(crate) struct KvCache {
    heads: usize,
    head_dim: usize,
    capacity: usize,
    k: Vec<f32>,
    v: Vec<f32>,
    /// Filled slots (`<= capacity`).
    len: usize,
    /// Slot written next.
    offset: usize,
    /// Positions appended since the last reset (Kyutai's `current_seq_len`).
    total: usize,
}

impl KvCache {
    pub fn new(heads: usize, head_dim: usize, capacity: usize) -> Self {
        Self {
            heads,
            head_dim,
            capacity,
            k: Vec::new(),
            v: Vec::new(),
            len: 0,
            offset: 0,
            total: 0,
        }
    }

    pub fn reset(&mut self) {
        self.len = 0;
        self.offset = 0;
        self.total = 0;
    }

    /// Positions appended since the last reset.
    pub fn current_seq_len(&self) -> usize {
        self.total
    }

    /// Append one position (`k`, `v`: `[heads, head_dim]`).
    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        let (h, d, cap) = (self.heads, self.head_dim, self.capacity);
        if self.k.is_empty() {
            // Allocated lazily (the depformer's caches are tiny, the temporal
            // transformer's are `context` long).
            self.k = vec![0.0; h * cap * d];
            self.v = vec![0.0; h * cap * d];
        }
        for head in 0..h {
            let dst = (head * cap + self.offset) * d;
            self.k[dst..dst + d].copy_from_slice(&k[head * d..(head + 1) * d]);
            self.v[dst..dst + d].copy_from_slice(&v[head * d..(head + 1) * d]);
        }
        self.offset = (self.offset + 1) % cap;
        self.len = (self.len + 1).min(cap);
        self.total += 1;
    }

    /// Cached keys and values as contiguous `[heads, len, head_dim]` buffers
    /// (borrowed when the cache is full, gathered otherwise), and `len`.
    pub fn view(
        &self,
    ) -> (
        std::borrow::Cow<'_, [f32]>,
        std::borrow::Cow<'_, [f32]>,
        usize,
    ) {
        use std::borrow::Cow;
        if self.len == self.capacity {
            return (Cow::Borrowed(&self.k), Cow::Borrowed(&self.v), self.len);
        }
        let (d, cap, n) = (self.head_dim, self.capacity, self.len);
        let mut k = Vec::with_capacity(self.heads * n * d);
        let mut v = Vec::with_capacity(self.heads * n * d);
        for head in 0..self.heads {
            let s = head * cap * d;
            k.extend_from_slice(&self.k[s..s + n * d]);
            v.extend_from_slice(&self.v[s..s + n * d]);
        }
        (Cow::Owned(k), Cow::Owned(v), n)
    }
}

/// Positional encoding settings of one transformer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PosEnc {
    kind: PositionalEmbedding,
    d_model: usize,
    head_dim: usize,
    max_period: usize,
}

impl PosEnc {
    pub fn new(cfg: &TransformerConfig) -> Self {
        Self {
            kind: cfg.positional_embedding,
            d_model: cfg.d_model,
            head_dim: cfg.head_dim(),
            max_period: cfg.max_period,
        }
    }

    /// RoPE cos/sin tables for position `pos` (f32 angles, like candle).
    pub fn rope(&self, pos: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        (self.kind == PositionalEmbedding::Rope)
            .then(|| rope_tables_f32(&[pos as i64], self.head_dim, self.max_period as f32))
    }

    /// Add the sinusoidal embedding of `pos` to `x` (no-op unless `Sin`):
    /// `cat(cos(pos * f), sin(pos * f))` with `f_i = 1 / max_period^(i / (d/2 - 1))`.
    pub fn add_sin(&self, x: &mut [f32], pos: usize) {
        if self.kind != PositionalEmbedding::Sin {
            return;
        }
        let half = self.d_model / 2;
        let theta = self.max_period as f32;
        let p = pos as f32;
        for i in 0..half {
            let f = p * (1f32 / theta.powf(i as f32 / (half - 1) as f32));
            x[i] += f.cos();
            x[half + i] += f.sin();
        }
    }
}

/// Self-attention with a fused `in_proj` (`[3 d, d]`, rows ordered q, k, v and
/// heads within each) and interleaved RoPE.
#[derive(Clone, Debug)]
pub(crate) struct Attention {
    in_proj: Linear,
    out_proj: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn load(l: &mut Loader, prefix: &str, cfg: &TransformerConfig) -> Result<Self> {
        let d = cfg.d_model;
        let rows = cfg.in_proj_rows();
        let w = l.get(&join(prefix, "in_proj_weight"), &[rows, d])?;
        let b = if cfg.bias_attn {
            Some(l.get(&join(prefix, "in_proj_bias"), &[rows])?)
        } else {
            None
        };
        Ok(Self {
            in_proj: Linear::from_parts(w, b, d, rows),
            out_proj: Linear::load(l, &join(prefix, "out_proj"), d, d, cfg.bias_attn)?,
            heads: cfg.num_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// One position: `x: [d]` → `[d]`, appending this position's k/v to `cache`.
    pub fn step(
        &self,
        x: &[f32],
        rope: Option<&(Vec<f32>, Vec<f32>)>,
        cache: &mut KvCache,
    ) -> Vec<f32> {
        let (h, hd) = (self.heads, self.head_dim);
        let d = h * hd;
        let qkv = self.in_proj.forward(x);
        let mut q = qkv[..d].to_vec();
        let mut k = qkv[d..2 * d].to_vec();
        let v = &qkv[2 * d..3 * d];
        if let Some((cos, sin)) = rope {
            rope_fwd_tables(&mut q, cos, sin, 1, h, hd, RopeLayout::Interleaved);
            rope_fwd_tables(&mut k, cos, sin, 1, h, hd, RopeLayout::Interleaved);
        }
        cache.append(&k, v);
        let (kc, vc, n) = cache.view();
        let mut o = vec![0.0; d];
        // Keys are in cache-slot order, not time order, and every cached key
        // is visible (the cache only holds the last `context` positions), so
        // no mask is needed — exactly Kyutai's single-step path.
        attention_decode(1, h, h, n, hd, AttnMask::default(), &q, &kc, &vc, &mut o);
        self.out_proj.forward(&o)
    }
}

/// Pre-norm transformer layer.
#[derive(Clone, Debug)]
pub(crate) struct Layer {
    norm1: Norm,
    attn: Attention,
    norm2: Norm,
    mlp: Mlp,
    layer_scale_1: Option<Vec<f32>>,
    layer_scale_2: Option<Vec<f32>>,
}

impl Layer {
    pub fn load(l: &mut Loader, prefix: &str, cfg: &TransformerConfig) -> Result<Self> {
        let d = cfg.d_model;
        let ls = |l: &mut Loader, name: &str| -> Result<Option<Vec<f32>>> {
            match cfg.layer_scale {
                None => Ok(None),
                Some(_) => Ok(Some(l.get(&join(&join(prefix, name), "scale"), &[d])?)),
            }
        };
        Ok(Self {
            mlp: Mlp::load(l, prefix, cfg)?,
            norm1: Norm::load(l, &join(prefix, "norm1"), d, cfg.norm)?,
            norm2: Norm::load(l, &join(prefix, "norm2"), d, cfg.norm)?,
            layer_scale_1: ls(l, "layer_scale_1")?,
            layer_scale_2: ls(l, "layer_scale_2")?,
            attn: Attention::load(l, &join(prefix, "self_attn"), cfg)?,
        })
    }

    pub fn step(&self, x: &mut [f32], rope: Option<&(Vec<f32>, Vec<f32>)>, cache: &mut KvCache) {
        let mut a = self.attn.step(&self.norm1.forward(x), rope, cache);
        if let Some(s) = &self.layer_scale_1 {
            a.iter_mut().zip(s).for_each(|(v, s)| *v *= s);
        }
        x.iter_mut().zip(&a).for_each(|(x, a)| *x += a);
        let mut m = self.mlp.forward(&self.norm2.forward(x));
        if let Some(s) = &self.layer_scale_2 {
            m.iter_mut().zip(s).for_each(|(v, s)| *v *= s);
        }
        x.iter_mut().zip(&m).for_each(|(x, m)| *x += m);
    }
}

/// Load `prefix.layers.{0..num_layers}`.
pub(crate) fn load_layers(
    l: &mut Loader,
    prefix: &str,
    cfg: &TransformerConfig,
) -> Result<Vec<Layer>> {
    (0..cfg.num_layers)
        .map(|i| Layer::load(l, &join(&join(prefix, "layers"), i), cfg))
        .collect()
}

/// One KV cache per layer.
pub(crate) fn new_caches(cfg: &TransformerConfig) -> Vec<KvCache> {
    (0..cfg.num_layers)
        .map(|_| KvCache::new(cfg.num_heads, cfg.head_dim(), cfg.context))
        .collect()
}

/// Run one position through `layers` (Kyutai's `StreamingTransformer::forward`
/// with `t = 1`): positional encoding at the caches' current length, then
/// every layer.
pub(crate) fn run_layers(
    layers: &[Layer],
    caches: &mut [KvCache],
    pos_enc: &PosEnc,
    x: &[f32],
) -> Vec<f32> {
    let pos = caches.first().map_or(0, |c| c.current_seq_len());
    let rope = pos_enc.rope(pos);
    let mut x = x.to_vec();
    pos_enc.add_sin(&mut x, pos);
    for (layer, cache) in layers.iter().zip(caches.iter_mut()) {
        layer.step(&mut x, rope.as_ref(), cache);
    }
    x
}

/// The temporal transformer: layers plus their persistent KV caches.
#[derive(Clone, Debug)]
pub(crate) struct StreamingTransformer {
    layers: Vec<Layer>,
    caches: Vec<KvCache>,
    pos_enc: PosEnc,
}

impl StreamingTransformer {
    pub fn load(l: &mut Loader, prefix: &str, cfg: &TransformerConfig) -> Result<Self> {
        Ok(Self {
            layers: load_layers(l, prefix, cfg)?,
            caches: new_caches(cfg),
            pos_enc: PosEnc::new(cfg),
        })
    }

    pub fn step(&mut self, x: &[f32]) -> Vec<f32> {
        run_layers(&self.layers, &mut self.caches, &self.pos_enc, x)
    }

    pub fn reset(&mut self) {
        self.caches.iter_mut().for_each(KvCache::reset);
    }

    pub fn current_seq_len(&self) -> usize {
        self.caches.first().map_or(0, |c| c.current_seq_len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use s2s_core::SplitMix64;
    use s2s_kernels::reference::attention::{attention_fwd, AttnParams};

    /// The rotating cache + unmasked decode equals sliding-window attention
    /// over the full history.
    #[test]
    fn rotating_cache_matches_sliding_window() {
        let (h, d, ctx, steps) = (2, 4, 3, 8);
        let mut r = SplitMix64::new(5);
        let qs: Vec<f32> = r.normal_vec(steps * h * d, 1.0);
        let ks: Vec<f32> = r.normal_vec(steps * h * d, 1.0);
        let vs: Vec<f32> = r.normal_vec(steps * h * d, 1.0);
        let mut cache = KvCache::new(h, d, ctx);
        // Full history in [h, t, d] layout for the reference.
        let to_htd = |x: &[f32]| {
            let mut o = vec![0.0; x.len()];
            for t in 0..steps {
                for hh in 0..h {
                    for i in 0..d {
                        o[(hh * steps + t) * d + i] = x[(t * h + hh) * d + i];
                    }
                }
            }
            o
        };
        let (q_all, k_all, v_all) = (to_htd(&qs), to_htd(&ks), to_htd(&vs));
        let p = AttnParams {
            batch: 1,
            q_heads: h,
            kv_heads: h,
            t: steps,
            s: steps,
            d,
            scale: 1.0 / (d as f64).sqrt(),
            q_offset: 0,
            mask: AttnMask {
                causal: true,
                window: Some(ctx),
                pinned_prefix: 0,
            },
        };
        let mut o_ref = vec![0.0f32; q_all.len()];
        let mut lse = vec![0.0f32; h * steps];
        attention_fwd(&p, &q_all, &k_all, &v_all, &mut o_ref, &mut lse);
        for t in 0..steps {
            let sl = |x: &[f32]| x[t * h * d..(t + 1) * h * d].to_vec();
            cache.append(&sl(&ks), &sl(&vs));
            let (kc, vc, n) = cache.view();
            assert_eq!(n, (t + 1).min(ctx));
            let mut o = vec![0.0f32; h * d];
            attention_decode(
                1,
                h,
                h,
                n,
                d,
                AttnMask::default(),
                &sl(&qs),
                &kc,
                &vc,
                &mut o,
            );
            for hh in 0..h {
                for i in 0..d {
                    let want = o_ref[(hh * steps + t) * d + i];
                    assert!((o[hh * d + i] - want).abs() < 1e-5);
                }
            }
        }
        assert_eq!(cache.current_seq_len(), steps);
        cache.reset();
        assert_eq!(cache.current_seq_len(), 0);
        assert_eq!(cache.view().2, 0);
    }

    #[test]
    fn sin_embedding_values() {
        let cfg = super::super::config::LmConfig::v0_1().transformer;
        let mut pe = PosEnc::new(&cfg);
        pe.kind = PositionalEmbedding::Sin;
        pe.d_model = 6;
        pe.max_period = 100;
        let mut x = vec![0.0f32; 6];
        pe.add_sin(&mut x, 2);
        // f = [1, 1/10, 1/100]
        let want = [
            2f32.cos(),
            0.2f32.cos(),
            0.02f32.cos(),
            2f32.sin(),
            0.2f32.sin(),
            0.02f32.sin(),
        ];
        for (a, b) in x.iter().zip(want) {
            assert!((a - b).abs() < 1e-6);
        }
        assert!(pe.rope(3).is_none());
    }
}
