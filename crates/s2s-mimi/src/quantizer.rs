//! Split residual vector quantizer (Kyutai's `SplitResidualVectorQuantizer`):
//! one semantic RVQ level (`rvq_first`) and `n_q - 1` acoustic levels
//! (`rvq_rest`), both quantizing the **same** projected latent, each with its
//! own 1x1 input/output projection.

use s2s_kernels::reference::conv1d::linear_lanes_fwd;
use s2s_kernels::reference::rvq::rvq_decode;
use s2s_kernels::reference::rvq_ip::{half_sq_norms, rvq_encode_ip};

use crate::weights::Loader;
use crate::{Codes, MimiConfig, MimiError, Result, Signal};

/// `cluster_usage` floor when deriving codewords (Kyutai's `epsilon`).
const USAGE_EPS: f32 = 1e-5;

#[derive(Clone, Debug)]
struct Rvq {
    input_proj: Vec<f32>,
    output_proj: Vec<f32>,
    /// Codewords `embedding_sum / max(cluster_usage, eps)`, `[bins, qdim]` per level.
    codebooks: Vec<Vec<f32>>,
    half_norms: Vec<Vec<f32>>,
    dim: usize,
    qdim: usize,
    bins: usize,
}

impl Rvq {
    fn load(l: &Loader, prefix: &str, levels: usize, cfg: &MimiConfig) -> Result<Self> {
        let (dim, qdim, bins) = (cfg.dimension, cfg.quantizer_dim, cfg.quantizer_bins);
        let input_proj = l.get(&format!("{prefix}.input_proj.weight"), &[qdim, dim, 1])?;
        let output_proj = l.get(&format!("{prefix}.output_proj.weight"), &[dim, qdim, 1])?;
        let mut codebooks = Vec::with_capacity(levels);
        let mut half_norms = Vec::with_capacity(levels);
        for i in 0..levels {
            let p = format!("{prefix}.vq.layers.{i}._codebook");
            let usage = l.get(&format!("{p}.cluster_usage"), &[bins])?;
            let sum = l.get(&format!("{p}.embedding_sum"), &[bins, qdim])?;
            let cb: Vec<f32> = sum
                .iter()
                .enumerate()
                .map(|(j, &s)| s / usage[j / qdim].max(USAGE_EPS))
                .collect();
            half_norms.push(half_sq_norms(&cb, bins, qdim));
            codebooks.push(cb);
        }
        Ok(Self {
            input_proj,
            output_proj,
            codebooks,
            half_norms,
            dim,
            qdim,
            bins,
        })
    }

    /// `x: [frames, dim]` time-major -> codes `[frames, levels]`.
    fn encode(&self, x: &[f32], frames: usize) -> Vec<u32> {
        let mut z = vec![0.0; frames * self.qdim];
        linear_lanes_fwd(
            x,
            &self.input_proj,
            None,
            &mut z,
            frames,
            self.dim,
            self.qdim,
        );
        let levels = self.codebooks.len();
        let cbs: Vec<&[f32]> = self.codebooks.iter().map(Vec::as_slice).collect();
        let norms: Vec<&[f32]> = self.half_norms.iter().map(Vec::as_slice).collect();
        let mut codes = vec![0u32; frames * levels];
        rvq_encode_ip(
            &z,
            &cbs,
            &norms,
            &vec![self.bins; levels],
            frames,
            self.qdim,
            &mut codes,
            None,
        );
        codes
    }

    /// codes `[frames, levels]` (the first `levels <= self.levels` levels) -> `[frames, dim]`.
    fn decode(&self, codes: &[u32], frames: usize, levels: usize) -> Vec<f32> {
        let cbs: Vec<&[f32]> = self.codebooks[..levels].iter().map(Vec::as_slice).collect();
        let mut z = vec![0.0; frames * self.qdim];
        rvq_decode(
            codes,
            &cbs,
            &vec![self.bins; levels],
            levels,
            frames,
            self.qdim,
            &mut z,
        );
        let mut y = vec![0.0; frames * self.dim];
        linear_lanes_fwd(
            &z,
            &self.output_proj,
            None,
            &mut y,
            frames,
            self.qdim,
            self.dim,
        );
        y
    }
}

/// The split RVQ.
#[derive(Clone, Debug)]
pub(crate) struct SplitRvq {
    first: Rvq,
    rest: Rvq,
    n_q: usize,
    dim: usize,
    bins: usize,
}

impl SplitRvq {
    pub(crate) fn load(l: &Loader, prefix: &str, cfg: &MimiConfig) -> Result<Self> {
        let n_q = cfg.quantizer_n_q;
        Ok(Self {
            first: Rvq::load(l, &format!("{prefix}.rvq_first"), 1, cfg)?,
            rest: Rvq::load(l, &format!("{prefix}.rvq_rest"), n_q - 1, cfg)?,
            n_q,
            dim: cfg.dimension,
            bins: cfg.quantizer_bins,
        })
    }

    /// Latent `[dim, frames]` -> codes `[n_q, frames]`.
    pub(crate) fn encode(&self, x: &Signal) -> Codes {
        assert_eq!(x.channels, self.dim, "quantizer: latent channels");
        let frames = x.len;
        let xt = x.transposed();
        let c0 = self.first.encode(&xt, frames);
        let rest_levels = self.n_q - 1;
        let cr = self.rest.encode(&xt, frames);
        let mut data = vec![0u32; self.n_q * frames];
        for f in 0..frames {
            data[f] = c0[f];
            for q in 0..rest_levels {
                data[(q + 1) * frames + f] = cr[f * rest_levels + q];
            }
        }
        Codes::new(self.n_q, frames, data).expect("consistent sizes")
    }

    /// Codes `[k, frames]` with `1 <= k <= n_q` -> latent `[dim, frames]`
    /// (levels beyond `k` are treated as absent).
    pub(crate) fn decode(&self, codes: &Codes) -> Result<Signal> {
        let (k, frames) = (codes.n_q(), codes.frames());
        if k == 0 || k > self.n_q {
            return Err(MimiError::Input(format!(
                "codes have {k} codebooks, the model decodes 1..={}",
                self.n_q
            )));
        }
        if let Some(&c) = codes.data().iter().find(|&&c| c as usize >= self.bins) {
            return Err(MimiError::Input(format!(
                "code {c} out of range (bins = {})",
                self.bins
            )));
        }
        let c0: Vec<u32> = codes.data()[..frames].to_vec();
        let mut y = self.first.decode(&c0, frames, 1);
        let rest_levels = k - 1;
        if rest_levels > 0 {
            let mut cr = vec![0u32; frames * rest_levels];
            for f in 0..frames {
                for q in 0..rest_levels {
                    cr[f * rest_levels + q] = codes.get(q + 1, f);
                }
            }
            let yr = self.rest.decode(&cr, frames, rest_levels);
            for (a, b) in y.iter_mut().zip(&yr) {
                *a += b;
            }
        }
        Ok(Signal::from_time_major(self.dim, frames, &y))
    }
}
