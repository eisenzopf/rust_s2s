//! LayerNorm forward (mean-centred, affine), used by the Mimi transformers.
//!
//! `y = (x - mean) / sqrt(var + eps) * gamma + beta` per row, with the biased
//! variance. Mean and variance are computed in two passes, accumulated in
//! `T`.

use s2s_core::{cast, Float};

/// LayerNorm over the last dimension of `x: [rows, d]`.
pub fn layer_norm_fwd<T: Float>(
    x: &[T],
    gamma: &[T],
    beta: &[T],
    eps: f64,
    y: &mut [T],
    rows: usize,
    d: usize,
) {
    assert_eq!(x.len(), rows * d, "layer_norm_fwd: x has wrong length");
    assert_eq!(y.len(), rows * d, "layer_norm_fwd: y has wrong length");
    assert!(
        gamma.len() == d && beta.len() == d,
        "layer_norm_fwd: affine params"
    );
    let n: T = cast(d as f64);
    let eps: T = cast(eps);
    for r in 0..rows {
        let xr = &x[r * d..(r + 1) * d];
        let mean = xr.iter().copied().sum::<T>() / n;
        let var = xr.iter().map(|&v| (v - mean) * (v - mean)).sum::<T>() / n;
        let inv_std = T::one() / (var + eps).sqrt();
        for ((o, &v), (&g, &b)) in y[r * d..(r + 1) * d]
            .iter_mut()
            .zip(xr)
            .zip(gamma.iter().zip(beta))
        {
            *o = (v - mean) * inv_std * g + b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_rows() {
        let x = [1.0f64, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0];
        let mut y = [0.0; 8];
        layer_norm_fwd(&x, &[1.0; 4], &[0.0; 4], 0.0, &mut y, 2, 4);
        let mean: f64 = y[..4].iter().sum::<f64>() / 4.0;
        let var: f64 = y[..4].iter().map(|v| v * v).sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-12 && (var - 1.0).abs() < 1e-12);
        // Constant row with eps = 0 would divide by zero; with eps it is 0.
        layer_norm_fwd(&x, &[1.0; 4], &[0.0; 4], 1e-5, &mut y, 2, 4);
        assert!(y[4..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn affine() {
        let x = [-1.0f64, 1.0];
        let mut y = [0.0; 2];
        layer_norm_fwd(&x, &[2.0, 3.0], &[0.5, -0.5], 0.0, &mut y, 1, 2);
        assert!((y[0] - (-2.0 + 0.5)).abs() < 1e-12);
        assert!((y[1] - (3.0 - 0.5)).abs() < 1e-12);
    }
}
