//! ELU activation, the nonlinearity of Mimi's SEANet encoder/decoder.
//!
//! `elu(x) = x` for `x >= +0`, `alpha * (exp(x) - 1)` otherwise. The branch
//! tests the sign bit (so `-0.0` takes the exponential branch and maps to
//! `+0.0`), matching candle's CPU op used by Kyutai's Mimi.

use s2s_core::Float;

/// ELU of one value.
#[inline]
pub fn elu<T: Float>(x: T, alpha: T) -> T {
    if x.is_sign_positive() {
        x
    } else {
        (x.exp() - T::one()) * alpha
    }
}

/// ELU forward, element-wise: `y[i] = elu(x[i])`.
pub fn elu_fwd<T: Float>(x: &[T], alpha: T, y: &mut [T]) {
    assert_eq!(x.len(), y.len(), "elu_fwd: length mismatch");
    for (d, &s) in y.iter_mut().zip(x) {
        *d = elu(s, alpha);
    }
}

/// In-place [`elu_fwd`].
pub fn elu_inplace<T: Float>(x: &mut [T], alpha: T) {
    for v in x.iter_mut() {
        *v = elu(*v, alpha);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values() {
        let x = [-2.0f64, -0.5, -0.0, 0.0, 1.5];
        let mut y = [9.0; 5];
        elu_fwd(&x, 1.0, &mut y);
        assert!((y[0] - ((-2.0f64).exp() - 1.0)).abs() < 1e-15);
        assert!((y[1] - ((-0.5f64).exp() - 1.0)).abs() < 1e-15);
        assert_eq!(y[2], 0.0);
        assert!(y[2].is_sign_positive());
        assert_eq!(y[3], 0.0);
        assert_eq!(y[4], 1.5);
        let mut z = x;
        elu_inplace(&mut z, 2.0);
        assert!((z[0] - 2.0 * ((-2.0f64).exp() - 1.0)).abs() < 1e-15);
        assert_eq!(z[4], 1.5);
    }

    #[test]
    fn continuous_at_zero_with_unit_slope() {
        let h = 1e-6f64;
        assert!((elu(-h, 1.0) + h).abs() < 1e-11);
        assert_eq!(elu(h, 1.0), h);
    }
}
