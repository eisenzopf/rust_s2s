//! The bottleneck transformers (Kyutai's `ProjectedTransformer` with
//! `conv_layout = true`): pre-norm LayerNorm layers with interleaved RoPE,
//! causal sliding-window attention, exact-GELU MLP and layer scale.

use s2s_kernels::reference::attention::{attention_fwd, AttnMask, AttnParams};
use s2s_kernels::reference::conv1d::linear_lanes_fwd;
use s2s_kernels::reference::gelu::gelu_erf_inplace;
use s2s_kernels::reference::layernorm::layernorm_fwd;
use s2s_kernels::reference::rope::{rope_fwd, RopeConfig, RopeLayout};

use crate::weights::Loader;
use crate::{MimiConfig, Result, Signal};

const LN_EPS: f64 = 1e-5;

#[derive(Clone, Debug)]
struct LayerNorm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl LayerNorm {
    fn load(l: &Loader, prefix: &str, d: usize) -> Result<Self> {
        // Kyutai accepts `alpha` [1, 1, d] (their export) or `weight` [d].
        let alpha = format!("{prefix}.alpha");
        let w = if l.has(&alpha) {
            l.get_any(&alpha, &[&[1, 1, d], &[d]])?
        } else {
            l.get(&format!("{prefix}.weight"), &[d])?
        };
        let b = l.get(&format!("{prefix}.bias"), &[d])?;
        Ok(Self { w, b })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let d = self.w.len();
        let mut y = vec![0.0; x.len()];
        layernorm_fwd(x, Some(&self.w), Some(&self.b), LN_EPS, &mut y, rows, d);
        y
    }
}

fn linear(x: &[f32], w: &[f32], rows: usize, d_in: usize, d_out: usize) -> Vec<f32> {
    let mut y = vec![0.0; rows * d_out];
    linear_lanes_fwd(x, w, None, &mut y, rows, d_in, d_out);
    y
}

#[derive(Clone, Debug)]
struct Layer {
    norm1: LayerNorm,
    norm2: LayerNorm,
    in_proj: Vec<f32>,
    out_proj: Vec<f32>,
    linear1: Vec<f32>,
    linear2: Vec<f32>,
    scale1: Option<Vec<f32>>,
    scale2: Option<Vec<f32>>,
}

/// Keys and values of the last `context` steps for one layer, time-major
/// `[n, heads, head_dim]`, oldest first.
#[derive(Clone, Debug, Default)]
struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    n: usize,
}

/// Streaming state of a [`Transformer`]: the absolute step position and the
/// per-layer KV caches.
#[derive(Clone, Debug, Default)]
pub(crate) struct TransformerState {
    pos: usize,
    caches: Vec<KvCache>,
}

/// One bottleneck transformer.
#[derive(Clone, Debug)]
pub(crate) struct Transformer {
    input_proj: Option<Vec<f32>>,
    output_proj: Option<Vec<f32>>,
    layers: Vec<Layer>,
    dim: usize,
    d_model: usize,
    heads: usize,
    head_dim: usize,
    ff: usize,
    context: usize,
    rope: RopeConfig,
}

impl Transformer {
    pub(crate) fn load(l: &Loader, prefix: &str, cfg: &MimiConfig) -> Result<Self> {
        let (dim, d, ff) = (
            cfg.dimension,
            cfg.transformer_d_model,
            cfg.transformer_dim_feedforward,
        );
        let input_proj = if dim != d {
            Some(l.get(&format!("{prefix}.input_proj.weight"), &[d, dim])?)
        } else {
            None
        };
        let output_proj = if dim != d {
            Some(l.get(&format!("{prefix}.output_projs.0.weight"), &[dim, d])?)
        } else {
            None
        };
        let mut layers = Vec::with_capacity(cfg.transformer_num_layers);
        for i in 0..cfg.transformer_num_layers {
            let p = format!("{prefix}.transformer.layers.{i}");
            let scale = |n: &str| -> Result<Option<Vec<f32>>> {
                if cfg.transformer_layer_scale {
                    Ok(Some(l.get(&format!("{p}.{n}.scale"), &[d])?))
                } else {
                    Ok(None)
                }
            };
            layers.push(Layer {
                norm1: LayerNorm::load(l, &format!("{p}.norm1"), d)?,
                norm2: LayerNorm::load(l, &format!("{p}.norm2"), d)?,
                in_proj: l.get(&format!("{p}.self_attn.in_proj_weight"), &[3 * d, d])?,
                out_proj: l.get(&format!("{p}.self_attn.out_proj.weight"), &[d, d])?,
                linear1: l.get(&format!("{p}.linear1.weight"), &[ff, d])?,
                linear2: l.get(&format!("{p}.linear2.weight"), &[d, ff])?,
                scale1: scale("layer_scale_1")?,
                scale2: scale("layer_scale_2")?,
            });
        }
        Ok(Self {
            input_proj,
            output_proj,
            layers,
            dim,
            d_model: d,
            heads: cfg.transformer_num_heads,
            head_dim: cfg.head_dim(),
            ff,
            context: cfg.transformer_context,
            rope: RopeConfig {
                theta: cfg.transformer_max_period,
                layout: RopeLayout::Interleaved,
            },
        })
    }

    pub(crate) fn state(&self) -> TransformerState {
        TransformerState {
            pos: 0,
            caches: vec![KvCache::default(); self.layers.len()],
        }
    }

    /// Offline forward over a whole sequence `[dim, t]` (positions from 0).
    pub(crate) fn forward(&self, x: &Signal) -> Signal {
        let mut st = self.state();
        self.step(&mut st, x)
    }

    /// Streaming forward of the next `x.len` steps. Bit-identical to
    /// [`Self::forward`] on the concatenated input: every query sees exactly
    /// the keys at steps `q - context ..= q`, in the same order.
    pub(crate) fn step(&self, st: &mut TransformerState, x: &Signal) -> Signal {
        assert_eq!(x.channels, self.dim, "transformer: input channels");
        let t = x.len;
        if t == 0 {
            return Signal::new(self.dim, 0, Vec::new());
        }
        // conv layout [dim, t] -> time-major [t, dim]
        let mut h = x.transposed();
        if let Some(w) = &self.input_proj {
            h = linear(&h, w, t, self.dim, self.d_model);
        }
        let positions: Vec<i64> = (st.pos..st.pos + t).map(|p| p as i64).collect();
        for (layer, cache) in self.layers.iter().zip(&mut st.caches) {
            h = self.layer_step(layer, cache, &h, t, &positions);
        }
        st.pos += t;
        if let Some(w) = &self.output_proj {
            h = linear(&h, w, t, self.d_model, self.dim);
        }
        Signal::from_time_major(self.dim, t, &h)
    }

    fn layer_step(
        &self,
        layer: &Layer,
        cache: &mut KvCache,
        x: &[f32],
        t: usize,
        positions: &[i64],
    ) -> Vec<f32> {
        let (d, nh, hd) = (self.d_model, self.heads, self.head_dim);
        // Self-attention block.
        let h = layer.norm1.forward(x, t);
        let qkv = linear(&h, &layer.in_proj, t, d, 3 * d);
        // qkv rows are [3, heads, head_dim].
        let mut q = vec![0.0; t * d];
        let mut k = vec![0.0; t * d];
        let mut v = vec![0.0; t * d];
        for i in 0..t {
            let row = &qkv[i * 3 * d..(i + 1) * 3 * d];
            q[i * d..(i + 1) * d].copy_from_slice(&row[..d]);
            k[i * d..(i + 1) * d].copy_from_slice(&row[d..2 * d]);
            v[i * d..(i + 1) * d].copy_from_slice(&row[2 * d..]);
        }
        rope_fwd(&mut q, positions, t, nh, hd, self.rope);
        rope_fwd(&mut k, positions, t, nh, hd, self.rope);

        // Keys: the cached past (at most `context` steps) then the new ones.
        let n_c = cache.n;
        let s = n_c + t;
        let k_all: Vec<f32> = cache.k.iter().chain(&k).copied().collect();
        let v_all: Vec<f32> = cache.v.iter().chain(&v).copied().collect();
        let p = AttnParams {
            batch: 1,
            q_heads: nh,
            kv_heads: nh,
            t,
            s,
            d: hd,
            scale: (hd as f64).powf(-0.5),
            q_offset: n_c,
            mask: AttnMask {
                causal: true,
                // q - k <= context
                window: Some(self.context + 1),
                pinned_prefix: 0,
            },
        };
        let q_h = to_head_major(&q, t, nh, hd);
        let k_h = to_head_major(&k_all, s, nh, hd);
        let v_h = to_head_major(&v_all, s, nh, hd);
        let mut o_h = vec![0.0; q_h.len()];
        let mut lse = vec![0.0; nh * t];
        attention_fwd(&p, &q_h, &k_h, &v_h, &mut o_h, &mut lse);
        let o = from_head_major(&o_h, t, nh, hd);
        let attn = linear(&o, &layer.out_proj, t, d, d);

        // Keep the last `context` steps for the next call.
        let keep = s.min(self.context);
        cache.k = k_all[(s - keep) * d..].to_vec();
        cache.v = v_all[(s - keep) * d..].to_vec();
        cache.n = keep;

        let mut y = residual(x, &attn, layer.scale1.as_deref(), d);

        // MLP block.
        let h = layer.norm2.forward(&y, t);
        let mut f = linear(&h, &layer.linear1, t, d, self.ff);
        gelu_erf_inplace(&mut f);
        let f = linear(&f, &layer.linear2, t, self.ff, d);
        y = residual(&y, &f, layer.scale2.as_deref(), d);
        y
    }
}

/// `x + scale * f` (per channel), or `x + f` without layer scale.
fn residual(x: &[f32], f: &[f32], scale: Option<&[f32]>, d: usize) -> Vec<f32> {
    x.iter()
        .zip(f)
        .enumerate()
        .map(|(i, (&a, &b))| match scale {
            Some(s) => a + b * s[i % d],
            None => a + b,
        })
        .collect()
}

/// `[n, heads, d]` -> `[heads, n, d]`.
fn to_head_major(x: &[f32], n: usize, heads: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for i in 0..n {
        for h in 0..heads {
            out[(h * n + i) * d..(h * n + i + 1) * d]
                .copy_from_slice(&x[(i * heads + h) * d..(i * heads + h + 1) * d]);
        }
    }
    out
}

/// `[heads, n, d]` -> `[n, heads, d]`.
fn from_head_major(x: &[f32], n: usize, heads: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for i in 0..n {
        for h in 0..heads {
            out[(i * heads + h) * d..(i * heads + h + 1) * d]
                .copy_from_slice(&x[(h * n + i) * d..(h * n + i + 1) * d]);
        }
    }
    out
}
