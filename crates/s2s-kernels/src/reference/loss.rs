//! Softmax, log-softmax and weighted cross-entropy.
//!
//! Cross-entropy supports per-row weights so the model's per-stream loss
//! weighting (text vs. semantic vs. acoustic tokens, down-weighted `<pad>`,
//! plan §3.4) is a property of the data, not of the kernel.

use s2s_core::Float;

/// Target value meaning "no loss for this row" (injection positions, padding).
pub const IGNORE_INDEX: i64 = -100;

fn row_max<T: Float>(x: &[T]) -> T {
    x.iter().fold(T::neg_infinity(), |m, &v| m.max(v))
}

fn row_lse<T: Float>(x: &[T]) -> T {
    let m = row_max(x);
    if m == T::neg_infinity() {
        return m;
    }
    m + x.iter().map(|&v| (v - m).exp()).sum::<T>().ln()
}

/// Row-wise softmax over the last dimension. `x`, `y`: `[rows, v]`.
pub fn softmax_fwd<T: Float>(x: &[T], y: &mut [T], rows: usize, v: usize) {
    assert!(x.len() == rows * v && y.len() == rows * v);
    for r in 0..rows {
        let lse = row_lse(&x[r * v..(r + 1) * v]);
        for c in 0..v {
            y[r * v + c] = (x[r * v + c] - lse).exp();
        }
    }
}

/// Softmax backward from its output: `dx += y * (dy - sum(dy * y))`.
pub fn softmax_bwd<T: Float>(y: &[T], dy: &[T], dx: &mut [T], rows: usize, v: usize) {
    for r in 0..rows {
        let o = r * v;
        let dot: T = (0..v).map(|c| dy[o + c] * y[o + c]).sum();
        for c in 0..v {
            dx[o + c] = dx[o + c] + y[o + c] * (dy[o + c] - dot);
        }
    }
}

/// Row-wise log-softmax. `x`, `y`: `[rows, v]`.
pub fn log_softmax_fwd<T: Float>(x: &[T], y: &mut [T], rows: usize, v: usize) {
    assert!(x.len() == rows * v && y.len() == rows * v);
    for r in 0..rows {
        let lse = row_lse(&x[r * v..(r + 1) * v]);
        for c in 0..v {
            y[r * v + c] = x[r * v + c] - lse;
        }
    }
}

/// Weighted cross-entropy forward.
///
/// - `logits`: `[rows, v]`, `targets`: `[rows]` (`IGNORE_INDEX` to skip)
/// - `weights`: optional `[rows]`; default 1
/// - `lse`: `[rows]`, written for the backward pass
///
/// Returns `(weighted_sum, total_weight)`. The mean loss is their ratio;
/// returning both lets data-parallel workers reduce them exactly.
pub fn cross_entropy_fwd<T: Float>(
    logits: &[T],
    targets: &[i64],
    weights: Option<&[T]>,
    lse: &mut [T],
    rows: usize,
    v: usize,
) -> (T, T) {
    assert_eq!(logits.len(), rows * v);
    assert_eq!(targets.len(), rows);
    assert_eq!(lse.len(), rows);
    if let Some(w) = weights {
        assert_eq!(w.len(), rows);
    }
    let (mut sum, mut total) = (T::zero(), T::zero());
    for r in 0..rows {
        let row = &logits[r * v..(r + 1) * v];
        lse[r] = row_lse(row);
        let t = targets[r];
        if t == IGNORE_INDEX {
            continue;
        }
        assert!(
            t >= 0 && (t as usize) < v,
            "target {t} out of range (vocab {v})"
        );
        let w = weights.map_or(T::one(), |w| w[r]);
        sum = sum + w * (lse[r] - row[t as usize]);
        total = total + w;
    }
    (sum, total)
}

/// Weighted cross-entropy backward. Accumulates into `dlogits`.
///
/// `dlogits[r] += scale * w[r] * (softmax(logits[r]) - onehot(target[r]))`.
/// For the mean loss, pass `scale = dloss / total_weight`.
pub fn cross_entropy_bwd<T: Float>(
    logits: &[T],
    targets: &[i64],
    weights: Option<&[T]>,
    lse: &[T],
    scale: T,
    dlogits: &mut [T],
    rows: usize,
    v: usize,
) {
    assert_eq!(dlogits.len(), rows * v);
    for r in 0..rows {
        let t = targets[r];
        if t == IGNORE_INDEX {
            continue;
        }
        let w = weights.map_or(T::one(), |w| w[r]) * scale;
        for c in 0..v {
            let p = (logits[r * v + c] - lse[r]).exp();
            let g = if c == t as usize { p - T::one() } else { p };
            dlogits[r * v + c] = dlogits[r * v + c] + w * g;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_rows_sum_to_one_and_stable() {
        let x = [1000.0f64, 1001.0, 1002.0, -5.0, 0.0, 5.0];
        let mut y = [0.0; 6];
        softmax_fwd(&x, &mut y, 2, 3);
        for r in 0..2 {
            let s: f64 = y[r * 3..r * 3 + 3].iter().sum();
            assert!((s - 1.0).abs() < 1e-12);
        }
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn uniform_logits_loss_is_log_v() {
        let logits = [0.0f64; 8];
        let mut lse = [0.0; 2];
        let (s, t) = cross_entropy_fwd(&logits, &[1, IGNORE_INDEX], None, &mut lse, 2, 4);
        assert_eq!(t, 1.0);
        assert!((s - 4f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn weights_scale_contribution() {
        let logits = [0.0f64; 4];
        let mut lse = [0.0; 2];
        let (s, t) = cross_entropy_fwd(&logits, &[0, 1], Some(&[0.5, 2.0]), &mut lse, 2, 2);
        assert!((t - 2.5).abs() < 1e-12);
        assert!((s - 2.5 * 2f64.ln()).abs() < 1e-12);
    }
}
