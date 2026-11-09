//! Exact (erf-based) GELU, as used by the Mimi transformer MLPs.
//!
//! `gelu(x) = x / 2 * (1 + erf(x / sqrt 2))`, evaluated in `f64` and rounded
//! once to `T`, exactly like candle's `gelu_erf` CPU op. The error function is
//! [`elementwise::erf`](super::elementwise::erf) (candle's vendored statrs
//! implementation), shared with the Kyutai LM port, so there is one `erf` in
//! the crate and results match candle bit for bit.

use s2s_core::{cast, Float};

pub use super::elementwise::erf;

/// Exact GELU of one value, computed in `f64`.
#[inline]
pub fn gelu_erf(v: f64) -> f64 {
    (erf(v / std::f64::consts::SQRT_2) + 1.0) * 0.5 * v
}

/// Exact GELU forward, element-wise: `y[i] = gelu(x[i])`.
pub fn gelu_erf_fwd<T: Float>(x: &[T], y: &mut [T]) {
    assert_eq!(x.len(), y.len(), "gelu_erf_fwd: length mismatch");
    for (d, &s) in y.iter_mut().zip(x) {
        *d = cast(gelu_erf(s.to_f64().unwrap_or(f64::NAN)));
    }
}

/// In-place [`gelu_erf_fwd`].
pub fn gelu_erf_inplace<T: Float>(x: &mut [T]) {
    for s in x.iter_mut() {
        *s = cast(gelu_erf(s.to_f64().unwrap_or(f64::NAN)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_known_values() {
        // High-precision reference values. The approximation, as vendored in
        // candle, is accurate to ~1e-10 (e.g. erf(0.5) is off by 5e-11): far
        // below f32 resolution, and what Kyutai's Mimi computes.
        for &(x, e) in &[
            (0.1, 0.112_462_916_018_284_9),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (3.0, 0.999_977_909_503_001_4),
        ] {
            assert!((erf(x) - e).abs() < 1e-10, "erf({x})");
            assert_eq!(erf(-x), -erf(x));
        }
        // Bit-exact values of candle 0.9.1's `candle::cpu::erf::erf`.
        assert_eq!(erf(0.5), 0.520_499_877_763_653_8);
        assert_eq!(erf(1.0), 0.842_700_792_942_714_9);
        assert_eq!(erf(0.0), 0.0);
        assert_eq!(erf(f64::INFINITY), 1.0);
        assert_eq!(erf(200.0), 1.0);
    }

    #[test]
    fn gelu_values() {
        let x = [-3.0f64, -1.0, 0.0, 0.5, 2.0];
        let mut y = [0.0; 5];
        gelu_erf_fwd(&x, &mut y);
        let e = [
            -0.004_049_694_094_890_31,
            -0.158_655_253_931_457_07,
            0.0,
            0.345_731_230_637_006_56,
            1.954_499_736_103_641_6,
        ];
        for (a, b) in y.iter().zip(&e) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
        let mut z = x;
        gelu_erf_inplace(&mut z);
        assert_eq!(z, y);
    }
}
