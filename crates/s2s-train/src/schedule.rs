//! Learning-rate schedules. Steps are 0-based: `lr(0)` is the first update.

use serde::{Deserialize, Serialize};

/// A learning-rate schedule.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum LrSchedule {
    /// Constant learning rate.
    Constant {
        /// The rate.
        lr: f64,
    },
    /// Linear warmup to `peak`, then cosine decay to `min` at `total` steps.
    WarmupCosine {
        /// Warmup steps.
        warmup: u64,
        /// Total steps (end of decay).
        total: u64,
        /// Peak rate.
        peak: f64,
        /// Final rate.
        min: f64,
    },
    /// Warmup-stable-decay: linear warmup, constant `peak`, then linear decay
    /// to `min` over the last `decay` steps.
    Wsd {
        /// Warmup steps.
        warmup: u64,
        /// Steps at the peak rate.
        stable: u64,
        /// Decay steps.
        decay: u64,
        /// Peak rate.
        peak: f64,
        /// Final rate.
        min: f64,
    },
}

impl LrSchedule {
    /// Learning rate for 0-based `step`.
    pub fn lr(&self, step: u64) -> f64 {
        let warm = |w: u64, peak: f64| peak * (step + 1) as f64 / w as f64;
        match *self {
            LrSchedule::Constant { lr } => lr,
            LrSchedule::WarmupCosine {
                warmup,
                total,
                peak,
                min,
            } => {
                if step < warmup {
                    warm(warmup, peak)
                } else if step >= total {
                    min
                } else {
                    let p = (step - warmup) as f64 / (total - warmup).max(1) as f64;
                    min + 0.5 * (peak - min) * (1.0 + (std::f64::consts::PI * p).cos())
                }
            }
            LrSchedule::Wsd {
                warmup,
                stable,
                decay,
                peak,
                min,
            } => {
                if step < warmup {
                    warm(warmup, peak)
                } else if step < warmup + stable {
                    peak
                } else {
                    let p = ((step - warmup - stable) as f64 / decay.max(1) as f64).min(1.0);
                    peak + (min - peak) * p
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_cosine_shape() {
        let s = LrSchedule::WarmupCosine {
            warmup: 10,
            total: 110,
            peak: 1.0,
            min: 0.1,
        };
        assert!((s.lr(0) - 0.1).abs() < 1e-12);
        assert!((s.lr(9) - 1.0).abs() < 1e-12);
        assert!((s.lr(10) - 1.0).abs() < 1e-12);
        assert!((s.lr(60) - 0.55).abs() < 1e-12); // halfway through the cosine
        assert!((s.lr(110) - 0.1).abs() < 1e-12);
        assert!((s.lr(10_000) - 0.1).abs() < 1e-12);
        for t in 10..110 {
            assert!(s.lr(t + 1) <= s.lr(t) + 1e-15, "cosine must not increase");
        }
    }

    #[test]
    fn wsd_shape() {
        let s = LrSchedule::Wsd {
            warmup: 2,
            stable: 5,
            decay: 4,
            peak: 2.0,
            min: 0.0,
        };
        assert_eq!(s.lr(0), 1.0);
        assert_eq!(s.lr(1), 2.0);
        assert_eq!(s.lr(6), 2.0);
        assert_eq!(s.lr(9), 1.0);
        assert_eq!(s.lr(11), 0.0);
        assert_eq!(s.lr(500), 0.0);
    }
}
