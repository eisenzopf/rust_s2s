//! RMSNorm.
//!
//! `y = x * rstd * w`, with `rstd = 1 / sqrt(mean(x^2) + eps)` per row.

use s2s_core::{cast, Float};

/// RMSNorm forward over the last dimension.
///
/// - `x`, `y`: `[rows, d]`
/// - `w`: `[d]`
/// - `rstd`: `[rows]`, saved for the backward pass.
pub fn rmsnorm_fwd<T: Float>(
    x: &[T],
    w: &[T],
    eps: f64,
    y: &mut [T],
    rstd: &mut [T],
    rows: usize,
    d: usize,
) {
    assert_eq!(x.len(), rows * d);
    assert_eq!(y.len(), rows * d);
    assert_eq!(w.len(), d);
    assert_eq!(rstd.len(), rows);
    let n: T = cast(d as f64);
    let eps: T = cast(eps);
    for r in 0..rows {
        let xr = &x[r * d..(r + 1) * d];
        let ms = xr.iter().map(|&v| v * v).sum::<T>() / n;
        let rs = T::one() / (ms + eps).sqrt();
        rstd[r] = rs;
        for i in 0..d {
            y[r * d + i] = xr[i] * rs * w[i];
        }
    }
}

/// RMSNorm backward. Accumulates into `dx` and `dw`.
///
/// With `g = w * dy`:
/// `dx = rstd * (g - x * rstd^2 * mean(g * x))`, `dw = sum_rows(dy * x * rstd)`.
pub fn rmsnorm_bwd<T: Float>(
    x: &[T],
    w: &[T],
    rstd: &[T],
    dy: &[T],
    dx: &mut [T],
    dw: &mut [T],
    rows: usize,
    d: usize,
) {
    assert_eq!(x.len(), rows * d);
    assert_eq!(dy.len(), rows * d);
    assert_eq!(dx.len(), rows * d);
    assert_eq!(dw.len(), d);
    let n: T = cast(d as f64);
    for r in 0..rows {
        let o = r * d;
        let rs = rstd[r];
        let mut dot = T::zero();
        for i in 0..d {
            dot = dot + w[i] * dy[o + i] * x[o + i];
        }
        let coef = rs * rs * dot / n;
        for i in 0..d {
            let g = w[i] * dy[o + i];
            dx[o + i] = dx[o + i] + rs * (g - x[o + i] * coef);
            dw[i] = dw[i] + dy[o + i] * x[o + i] * rs;
        }
    }
}

/// Fused residual add + RMSNorm forward, as used between transformer
/// sub-blocks: `h = residual + x; y = rmsnorm(h)`. Writes `h` (the new
/// residual stream) as well as `y`.
pub fn add_rmsnorm_fwd<T: Float>(
    residual: &[T],
    x: &[T],
    w: &[T],
    eps: f64,
    h: &mut [T],
    y: &mut [T],
    rstd: &mut [T],
    rows: usize,
    d: usize,
) {
    assert_eq!(residual.len(), rows * d);
    assert_eq!(x.len(), rows * d);
    assert_eq!(h.len(), rows * d);
    for i in 0..rows * d {
        h[i] = residual[i] + x[i];
    }
    rmsnorm_fwd(h, w, eps, y, rstd, rows, d);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_rms_after_norm() {
        let x = [3.0f64, -4.0, 0.0, 0.0];
        let w = [1.0; 4];
        let mut y = [0.0; 4];
        let mut rstd = [0.0; 1];
        rmsnorm_fwd(&x, &w, 0.0, &mut y, &mut rstd, 1, 4);
        let ms: f64 = y.iter().map(|v| v * v).sum::<f64>() / 4.0;
        assert!((ms - 1.0).abs() < 1e-12);
        assert!((rstd[0] - 1.0 / 2.5).abs() < 1e-12); // mean(x^2) = 25/4
    }
}
