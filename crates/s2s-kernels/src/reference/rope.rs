//! Rotary position embedding (RoPE).
//!
//! Positions are passed explicitly per token rather than assumed to be
//! `0..T`. That supports KV-cache decoding at an offset, packed sequences,
//! and the text-only injection segments of the model design (plan §3.5),
//! where positions continue contiguously across frames and injected tokens.

use s2s_core::{cast, Float};

/// How the two halves of each rotated pair are laid out in the head dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeLayout {
    /// Pair `(i, i + d/2)`: the "rotate half" layout used by Llama-family
    /// checkpoints in Hugging Face format.
    Half,
    /// Pair `(2i, 2i + 1)`: the interleaved layout of the original RoPE paper
    /// (and of some Kyutai checkpoints).
    Interleaved,
}

/// RoPE settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeConfig {
    /// Frequency base, e.g. 10_000.0.
    pub theta: f64,
    /// Pair layout.
    pub layout: RopeLayout,
}

fn pair_index(layout: RopeLayout, d: usize, i: usize) -> (usize, usize) {
    match layout {
        RopeLayout::Half => (i, i + d / 2),
        RopeLayout::Interleaved => (2 * i, 2 * i + 1),
    }
}

/// Apply RoPE (`sign = 1`) or its inverse (`sign = -1`) in place.
fn rotate<T: Float>(
    x: &mut [T],
    positions: &[i64],
    tokens: usize,
    heads: usize,
    d: usize,
    cfg: RopeConfig,
    sign: f64,
) {
    assert!(d.is_multiple_of(2), "rope: head dim must be even");
    assert_eq!(x.len(), tokens * heads * d);
    assert_eq!(positions.len(), tokens);
    let half = d / 2;
    for t in 0..tokens {
        let pos = positions[t] as f64;
        for i in 0..half {
            let inv_freq = cfg.theta.powf(-(2.0 * i as f64) / d as f64);
            let ang = pos * inv_freq;
            let (c, s): (T, T) = (cast(ang.cos()), cast(sign * ang.sin()));
            let (a, b) = pair_index(cfg.layout, d, i);
            for h in 0..heads {
                let base = (t * heads + h) * d;
                let x1 = x[base + a];
                let x2 = x[base + b];
                x[base + a] = x1 * c - x2 * s;
                x[base + b] = x1 * s + x2 * c;
            }
        }
    }
}

/// RoPE forward, in place. `x` is `[tokens, heads, d]`.
pub fn rope_fwd<T: Float>(
    x: &mut [T],
    positions: &[i64],
    tokens: usize,
    heads: usize,
    d: usize,
    cfg: RopeConfig,
) {
    rotate(x, positions, tokens, heads, d, cfg, 1.0);
}

/// RoPE backward, in place: rotating the upstream gradient by `-angle`
/// (the transpose of a rotation is its inverse). `dx` is `[tokens, heads, d]`.
pub fn rope_bwd<T: Float>(
    dx: &mut [T],
    positions: &[i64],
    tokens: usize,
    heads: usize,
    d: usize,
    cfg: RopeConfig,
) {
    rotate(dx, positions, tokens, heads, d, cfg, -1.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    const CFG: RopeConfig = RopeConfig {
        theta: 10_000.0,
        layout: RopeLayout::Half,
    };

    #[test]
    fn position_zero_is_identity() {
        let mut x = vec![1.0f64, 2.0, 3.0, 4.0];
        let orig = x.clone();
        rope_fwd(&mut x, &[0], 1, 1, 4, CFG);
        assert_eq!(x, orig);
    }

    #[test]
    fn preserves_norm_and_inverts() {
        for layout in [RopeLayout::Half, RopeLayout::Interleaved] {
            let cfg = RopeConfig { layout, ..CFG };
            let mut x: Vec<f64> = (0..2 * 3 * 8).map(|i| (i as f64 * 0.37).sin()).collect();
            let orig = x.clone();
            rope_fwd(&mut x, &[5, 1234], 2, 3, 8, cfg);
            let n0: f64 = orig.iter().map(|v| v * v).sum();
            let n1: f64 = x.iter().map(|v| v * v).sum();
            assert!((n0 - n1).abs() < 1e-9);
            rope_bwd(&mut x, &[5, 1234], 2, 3, 8, cfg);
            for (a, b) in x.iter().zip(&orig) {
                assert!((a - b).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn relative_position_property() {
        // <rope(q, m), rope(k, n)> depends only on m - n.
        let q: Vec<f64> = (0..8).map(|i| (i as f64 + 0.5).cos()).collect();
        let k: Vec<f64> = (0..8).map(|i| (i as f64 * 1.3).sin()).collect();
        let dot_at = |m: i64, n: i64| {
            let (mut a, mut b) = (q.clone(), k.clone());
            rope_fwd(&mut a, &[m], 1, 1, 8, CFG);
            rope_fwd(&mut b, &[n], 1, 1, 8, CFG);
            a.iter().zip(&b).map(|(x, y)| x * y).sum::<f64>()
        };
        assert!((dot_at(10, 3) - dot_at(107, 100)).abs() < 1e-9);
    }
}
