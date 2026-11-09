//! SwiGLU: `h = silu(gate) * up`, with `silu(g) = g * sigmoid(g)`.

use s2s_core::Float;

fn sigmoid<T: Float>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}

/// SwiGLU forward, element-wise over equal-length `gate`, `up`, `h`.
pub fn swiglu_fwd<T: Float>(gate: &[T], up: &[T], h: &mut [T]) {
    assert_eq!(gate.len(), up.len());
    assert_eq!(gate.len(), h.len());
    for i in 0..h.len() {
        let g = gate[i];
        h[i] = g * sigmoid(g) * up[i];
    }
}

/// SwiGLU backward. Accumulates into `dgate` and `dup`.
///
/// `d silu(g)/dg = s * (1 + g * (1 - s))` with `s = sigmoid(g)`.
pub fn swiglu_bwd<T: Float>(gate: &[T], up: &[T], dh: &[T], dgate: &mut [T], dup: &mut [T]) {
    let n = dh.len();
    assert!(gate.len() == n && up.len() == n && dgate.len() == n && dup.len() == n);
    for i in 0..n {
        let g = gate[i];
        let s = sigmoid(g);
        let silu = g * s;
        let dsilu = s * (T::one() + g * (T::one() - s));
        dgate[i] = dgate[i] + dh[i] * up[i] * dsilu;
        dup[i] = dup[i] + dh[i] * silu;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        let mut h = [0.0f64; 3];
        swiglu_fwd(&[0.0, 1.0, -1.0], &[2.0, 2.0, 2.0], &mut h);
        assert_eq!(h[0], 0.0);
        let s1 = 1.0 / (1.0 + (-1.0f64).exp());
        assert!((h[1] - 2.0 * s1).abs() < 1e-12);
        assert!((h[2] - 2.0 * -(1.0 - s1)).abs() < 1e-12);
    }
}
