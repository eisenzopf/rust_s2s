//! AdamW and gradient-norm clipping.

use s2s_core::{cast, Float};

/// AdamW hyper-parameters (decoupled weight decay, Loshchilov & Hutter).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamWConfig {
    /// Learning rate for this step (schedules are applied by the caller).
    pub lr: f64,
    /// First-moment decay.
    pub beta1: f64,
    /// Second-moment decay.
    pub beta2: f64,
    /// Denominator epsilon.
    pub eps: f64,
    /// Decoupled weight decay coefficient.
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
        }
    }
}

/// One fused AdamW step over a parameter tensor.
///
/// `step` is 1-based (the first update uses `step = 1`). `grad_scale`
/// multiplies the gradient before use, which fuses global-norm clipping
/// (pass `clip_scale(..)`) into the update.
pub fn adamw_step<T: Float>(
    param: &mut [T],
    grad: &[T],
    m: &mut [T],
    v: &mut [T],
    step: u64,
    cfg: &AdamWConfig,
    grad_scale: f64,
) {
    let n = param.len();
    assert!(grad.len() == n && m.len() == n && v.len() == n);
    assert!(step >= 1, "AdamW step is 1-based");
    let (b1, b2) = (cfg.beta1, cfg.beta2);
    let bc1 = 1.0 - b1.powi(step as i32);
    let bc2 = 1.0 - b2.powi(step as i32);
    let (b1t, b2t, gs) = (cast::<T>(b1), cast::<T>(b2), cast::<T>(grad_scale));
    let (one_b1, one_b2) = (cast::<T>(1.0 - b1), cast::<T>(1.0 - b2));
    let (lr, eps, wd) = (
        cast::<T>(cfg.lr),
        cast::<T>(cfg.eps),
        cast::<T>(cfg.weight_decay),
    );
    let (bc1, bc2) = (cast::<T>(bc1), cast::<T>(bc2));
    for i in 0..n {
        let g = grad[i] * gs;
        m[i] = b1t * m[i] + one_b1 * g;
        v[i] = b2t * v[i] + one_b2 * g * g;
        let mhat = m[i] / bc1;
        let vhat = v[i] / bc2;
        param[i] = param[i] - lr * (mhat / (vhat.sqrt() + eps) + wd * param[i]);
    }
}

/// Global L2 norm over a set of gradient tensors, accumulated in `f64`.
pub fn global_grad_norm<T: Float>(grads: &[&[T]]) -> f64 {
    grads
        .iter()
        .flat_map(|g| g.iter())
        .map(|&x| {
            let x = x.to_f64().unwrap_or(f64::NAN);
            x * x
        })
        .sum::<f64>()
        .sqrt()
}

/// Gradient scale that clips the global norm to `max_norm` (1.0 if already within).
pub fn clip_scale(norm: f64, max_norm: f64) -> f64 {
    if norm.is_finite() && norm > max_norm {
        max_norm / (norm + 1e-6)
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_step_moves_by_lr_times_sign() {
        // With bias correction, the first AdamW step is lr * g/|g| (+ decay).
        let cfg = AdamWConfig {
            lr: 0.1,
            weight_decay: 0.0,
            eps: 0.0,
            ..Default::default()
        };
        let mut p = [1.0f64, 1.0];
        let (mut m, mut v) = ([0.0; 2], [0.0; 2]);
        adamw_step(&mut p, &[0.5, -3.0], &mut m, &mut v, 1, &cfg, 1.0);
        assert!((p[0] - 0.9).abs() < 1e-12);
        assert!((p[1] - 1.1).abs() < 1e-12);
    }

    #[test]
    fn weight_decay_is_decoupled() {
        let cfg = AdamWConfig {
            lr: 0.1,
            weight_decay: 0.5,
            ..Default::default()
        };
        let mut p = [2.0f64];
        let (mut m, mut v) = ([0.0], [0.0]);
        adamw_step(&mut p, &[0.0], &mut m, &mut v, 1, &cfg, 1.0);
        assert!((p[0] - (2.0 - 0.1 * 0.5 * 2.0)).abs() < 1e-12);
    }

    #[test]
    fn clipping() {
        let g1 = [3.0f64];
        let g2 = [4.0f64];
        let n = global_grad_norm(&[&g1, &g2]);
        assert!((n - 5.0).abs() < 1e-12);
        assert!((clip_scale(n, 1.0) * n - 1.0).abs() < 1e-6);
        assert_eq!(clip_scale(n, 10.0), 1.0);
        assert_eq!(clip_scale(f64::NAN, 1.0), 1.0);
    }
}
