//! LayerNorm, forward only (inference).
//!
//! `y = (x - mean) / sqrt(var + eps) * w + b` per row, with the biased
//! variance `var = mean((x - mean)^2)`. `w` and `b` are optional (identity
//! scale / zero shift when absent).

use s2s_core::{cast, Float};

/// LayerNorm forward over the last dimension.
///
/// - `x`, `y`: `[rows, d]`
/// - `w`, `b`: `[d]` when given.
pub fn layernorm_fwd<T: Float>(
    x: &[T],
    w: Option<&[T]>,
    b: Option<&[T]>,
    eps: f64,
    y: &mut [T],
    rows: usize,
    d: usize,
) {
    assert_eq!(x.len(), rows * d);
    assert_eq!(y.len(), rows * d);
    if let Some(w) = w {
        assert_eq!(w.len(), d);
    }
    if let Some(b) = b {
        assert_eq!(b.len(), d);
    }
    let n: T = cast(d as f64);
    let eps: T = cast(eps);
    for r in 0..rows {
        let xr = &x[r * d..(r + 1) * d];
        let mean = xr.iter().copied().sum::<T>() / n;
        let var = xr.iter().map(|&v| (v - mean) * (v - mean)).sum::<T>() / n;
        let rs = T::one() / (var + eps).sqrt();
        for i in 0..d {
            let mut v = (xr[i] - mean) * rs;
            if let Some(w) = w {
                v = v * w[i];
            }
            if let Some(b) = b {
                v = v + b[i];
            }
            y[r * d + i] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_mean_unit_variance() {
        let x = [1.0f64, 2.0, 3.0, 4.0, -1.0, -1.0, 5.0, 5.0];
        let mut y = [0.0; 8];
        layernorm_fwd(&x, None, None, 0.0, &mut y, 2, 4);
        for r in 0..2 {
            let row = &y[r * 4..(r + 1) * 4];
            let mean: f64 = row.iter().sum::<f64>() / 4.0;
            let var: f64 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 4.0;
            assert!(mean.abs() < 1e-12);
            assert!((var - 1.0).abs() < 1e-12);
        }
        // Row 1 is (-1, -1, 5, 5): mean 2, std 3.
        assert!((y[4] + 1.0).abs() < 1e-12 && (y[6] - 1.0).abs() < 1e-12);

        let w = [2.0, 2.0, 2.0, 2.0];
        let b = [0.5, 0.5, 0.5, 0.5];
        let mut z = [0.0; 8];
        layernorm_fwd(&x, Some(&w), Some(&b), 0.0, &mut z, 2, 4);
        for i in 0..8 {
            assert!((z[i] - (2.0 * y[i] + 0.5)).abs() < 1e-12);
        }
    }
}
