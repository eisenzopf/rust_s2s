//! Reference-based audio metrics: SI-SDR and log-mel spectral distance.
//!
//! These are **codec-quality proxies**: cheap signal-level comparisons of a
//! reconstruction against its reference, useful for regression tests and
//! parity checks of the Mimi codec (plan §11). They do not replace
//! perceptual metrics (ViSQOL, MOS-style ratings) and say nothing about
//! intelligibility (use WER via ASR for that).
//!
//! # SI-SDR
//!
//! Scale-invariant signal-to-distortion ratio (Le Roux et al., 2019), in dB.
//! Both signals are made zero-mean, the reference `s` is scaled to the
//! estimate `ŝ` by `α = ⟨ŝ, s⟩ / ‖s‖²`, and
//! `SI-SDR = 10 log10(‖α s‖² / ‖α s − ŝ‖²)`. Scaling the estimate does not
//! change it; a perfect (scaled) estimate gives `+∞`.
//!
//! # Log-mel distance
//!
//! Both signals go through the same log-mel front end ([`StftConfig`],
//! defaults for 24 kHz audio):
//!
//! - **STFT**: frames of `n_fft = 1024` samples (42.7 ms) every `hop = 256`
//!   samples (10.7 ms), periodic Hann window, no centering: frame `k` covers
//!   samples `[k·hop, k·hop + n_fft)`, and the signal is zero-padded at the
//!   end so that every sample falls in a frame (at least one frame).
//! - **Power spectrum** `|X[k]|²` for bins `0..=n_fft/2`.
//! - **Mel filterbank**: `n_mels = 80` triangular filters on the HTK mel
//!   scale (`mel = 2595 log10(1 + f / 700)`), centers equally spaced in mel
//!   between `f_min = 0` and `f_max = sample_rate / 2`, peak 1, no area
//!   normalization; filter weights are evaluated at the bin frequencies
//!   `k · sample_rate / n_fft`.
//! - **Log**: `log10(mel_power + eps)` with `eps = 1e-5`.
//!
//! The distance between the two log-mel spectrograms is reported as the
//! mean absolute difference ([`LogMelDistance::l1`]) and the root mean
//! square difference ([`LogMelDistance::rmse`]) over all frames and mel
//! bands, in log10 units (1.0 = a factor of 10 in power). Signals of
//! different lengths are compared over the shorter length (codecs often pad
//! to whole frames).

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use serde::Serialize;

/// Errors from audio metrics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AudioError {
    /// A signal is empty.
    Empty,
    /// The signals have different lengths (SI-SDR needs equal lengths).
    LengthMismatch {
        /// Reference length.
        reference: usize,
        /// Estimate length.
        estimate: usize,
    },
    /// The reference has no energy (after removing its mean).
    SilentReference,
    /// Invalid STFT configuration.
    Config(String),
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => f.write_str("empty signal"),
            Self::LengthMismatch {
                reference,
                estimate,
            } => write!(
                f,
                "length mismatch: reference {reference} samples, estimate {estimate}"
            ),
            Self::SilentReference => f.write_str("reference signal has no energy"),
            Self::Config(e) => write!(f, "invalid STFT configuration: {e}"),
        }
    }
}

impl std::error::Error for AudioError {}

/// SI-SDR of `estimate` against `reference`, in dB (see the
/// [module docs](self)).
pub fn si_sdr(reference: &[f32], estimate: &[f32]) -> Result<f64, AudioError> {
    if reference.is_empty() || estimate.is_empty() {
        return Err(AudioError::Empty);
    }
    if reference.len() != estimate.len() {
        return Err(AudioError::LengthMismatch {
            reference: reference.len(),
            estimate: estimate.len(),
        });
    }
    let n = reference.len() as f64;
    let mr = reference.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let me = estimate.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let (mut dot, mut energy) = (0.0f64, 0.0f64);
    for (&r, &e) in reference.iter().zip(estimate) {
        let (r, e) = (f64::from(r) - mr, f64::from(e) - me);
        dot += r * e;
        energy += r * r;
    }
    if energy == 0.0 {
        return Err(AudioError::SilentReference);
    }
    let alpha = dot / energy;
    let (mut target, mut noise) = (0.0f64, 0.0f64);
    for (&r, &e) in reference.iter().zip(estimate) {
        let t = alpha * (f64::from(r) - mr);
        let d = f64::from(e) - me - t;
        target += t * t;
        noise += d * d;
    }
    Ok(10.0 * (target / noise).log10())
}

/// Log-mel front-end parameters (see the [module docs](self)).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct StftConfig {
    /// Sample rate of the signals, in Hz.
    pub sample_rate: u32,
    /// FFT size and window length, in samples.
    pub n_fft: usize,
    /// Frame step, in samples.
    pub hop: usize,
    /// Number of mel bands.
    pub n_mels: usize,
    /// Lowest filterbank frequency, in Hz.
    pub f_min: f64,
    /// Highest filterbank frequency, in Hz (`None`: Nyquist).
    pub f_max: Option<f64>,
    /// Floor added before the logarithm.
    pub eps: f64,
}

impl Default for StftConfig {
    /// 24 kHz, `n_fft` 1024, hop 256, 80 mel bands, 0 Hz to Nyquist, eps 1e-5.
    fn default() -> Self {
        Self {
            sample_rate: 24_000,
            n_fft: 1024,
            hop: 256,
            n_mels: 80,
            f_min: 0.0,
            f_max: None,
            eps: 1e-5,
        }
    }
}

impl StftConfig {
    fn check(&self) -> Result<f64, AudioError> {
        let nyquist = f64::from(self.sample_rate) / 2.0;
        let f_max = self.f_max.unwrap_or(nyquist);
        if self.sample_rate == 0 || self.n_fft < 2 || self.hop == 0 || self.n_mels == 0 {
            return Err(AudioError::Config(format!("{self:?}")));
        }
        if !(0.0 <= self.f_min && self.f_min < f_max && f_max <= nyquist) {
            return Err(AudioError::Config(format!(
                "need 0 <= f_min < f_max <= {nyquist}"
            )));
        }
        if self.eps.is_nan() || self.eps <= 0.0 {
            return Err(AudioError::Config("eps must be positive".into()));
        }
        Ok(f_max)
    }

    /// Frequency bins per frame: `n_fft / 2 + 1`.
    pub fn bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Frames for a signal of `len` samples (at least one).
    pub fn frames(&self, len: usize) -> usize {
        1 + len.saturating_sub(self.n_fft).div_ceil(self.hop)
    }
}

/// HTK mel scale: `2595 log10(1 + hz / 700)`.
pub fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Inverse of [`hz_to_mel`].
pub fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

/// The mel filterbank: `n_mels` rows of `bins()` weights.
pub fn mel_filterbank(cfg: &StftConfig) -> Result<Vec<Vec<f64>>, AudioError> {
    let f_max = cfg.check()?;
    let (lo, hi) = (hz_to_mel(cfg.f_min), hz_to_mel(f_max));
    let edges: Vec<f64> = (0..cfg.n_mels + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (cfg.n_mels + 1) as f64))
        .collect();
    let bin_hz = f64::from(cfg.sample_rate) / cfg.n_fft as f64;
    Ok((0..cfg.n_mels)
        .map(|m| {
            let (l, c, r) = (edges[m], edges[m + 1], edges[m + 2]);
            (0..cfg.bins())
                .map(|k| {
                    let f = k as f64 * bin_hz;
                    let up = (f - l) / (c - l);
                    let down = (r - f) / (r - c);
                    up.min(down).max(0.0)
                })
                .collect()
        })
        .collect())
}

/// Periodic Hann window of length `n`.
pub fn hann(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .collect()
}

struct Stft {
    cfg: StftConfig,
    window: Vec<f64>,
    fft: Arc<dyn Fft<f64>>,
}

impl Stft {
    fn new(cfg: StftConfig) -> Self {
        Self {
            cfg,
            window: hann(cfg.n_fft),
            fft: FftPlanner::new().plan_fft_forward(cfg.n_fft),
        }
    }

    /// Power spectrogram of the first `len` samples: `frames × bins`.
    fn power(&self, signal: &[f32], len: usize) -> Vec<Vec<f64>> {
        let c = &self.cfg;
        let mut buf = vec![Complex::new(0.0, 0.0); c.n_fft];
        (0..c.frames(len))
            .map(|f| {
                let start = f * c.hop;
                for (i, b) in buf.iter_mut().enumerate() {
                    let x = if start + i < len {
                        f64::from(signal[start + i])
                    } else {
                        0.0
                    };
                    *b = Complex::new(x * self.window[i], 0.0);
                }
                self.fft.process(&mut buf);
                buf[..c.bins()].iter().map(|z| z.norm_sqr()).collect()
            })
            .collect()
    }
}

/// Power spectrogram (`frames × (n_fft/2 + 1)`) of `signal`.
pub fn power_spectrogram(signal: &[f32], cfg: &StftConfig) -> Result<Vec<Vec<f64>>, AudioError> {
    cfg.check()?;
    if signal.is_empty() {
        return Err(AudioError::Empty);
    }
    Ok(Stft::new(*cfg).power(signal, signal.len()))
}

fn log_mel(power: &[Vec<f64>], bank: &[Vec<f64>], eps: f64) -> Vec<Vec<f64>> {
    power
        .iter()
        .map(|frame| {
            bank.iter()
                .map(|w| {
                    let e: f64 = w.iter().zip(frame).map(|(a, b)| a * b).sum();
                    (e + eps).log10()
                })
                .collect()
        })
        .collect()
}

/// Log-mel spectrogram (`frames × n_mels`, log10 units) of `signal`.
pub fn log_mel_spectrogram(signal: &[f32], cfg: &StftConfig) -> Result<Vec<Vec<f64>>, AudioError> {
    let bank = mel_filterbank(cfg)?;
    Ok(log_mel(&power_spectrogram(signal, cfg)?, &bank, cfg.eps))
}

/// Distance between two log-mel spectrograms (see the [module docs](self)).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct LogMelDistance {
    /// Mean absolute difference, log10 units.
    pub l1: f64,
    /// Root mean square difference, log10 units.
    pub rmse: f64,
    /// Frames compared.
    pub frames: usize,
    /// Samples compared (the shorter signal's length).
    pub samples: usize,
}

/// Log-mel distance between `reference` and `estimate` over their common
/// length (see the [module docs](self)).
pub fn log_mel_distance(
    reference: &[f32],
    estimate: &[f32],
    cfg: &StftConfig,
) -> Result<LogMelDistance, AudioError> {
    let bank = mel_filterbank(cfg)?;
    let len = reference.len().min(estimate.len());
    if len == 0 {
        return Err(AudioError::Empty);
    }
    let stft = Stft::new(*cfg);
    let a = log_mel(&stft.power(reference, len), &bank, cfg.eps);
    let b = log_mel(&stft.power(estimate, len), &bank, cfg.eps);
    let (mut l1, mut l2, mut n) = (0.0, 0.0, 0usize);
    for (fa, fb) in a.iter().zip(&b) {
        for (x, y) in fa.iter().zip(fb) {
            let d = x - y;
            l1 += d.abs();
            l2 += d * d;
            n += 1;
        }
    }
    Ok(LogMelDistance {
        l1: l1 / n as f64,
        rmse: (l2 / n as f64).sqrt(),
        frames: a.len(),
        samples: len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn si_sdr_hand_computed() {
        // s and n are zero-mean and orthogonal, ‖s‖² = ‖n‖² = 4.
        let s = [1.0f32, -1.0, 1.0, -1.0];
        let n = [1.0f32, 1.0, -1.0, -1.0];
        let est: Vec<f32> = s.iter().zip(&n).map(|(a, b)| a + b).collect();
        // α = 1, target = s, noise = n: 10 log10(4 / 4) = 0 dB.
        assert!(close(si_sdr(&s, &est).unwrap(), 0.0, 1e-12));
        // Half the noise: 10 log10(4 / 1) = 6.0206 dB.
        let est: Vec<f32> = s.iter().zip(&n).map(|(a, b)| a + 0.5 * b).collect();
        assert!(close(si_sdr(&s, &est).unwrap(), 10.0 * 4f64.log10(), 1e-9));
        // Scale invariance and DC invariance.
        let scaled: Vec<f32> = est.iter().map(|x| 3.0 * x + 0.25).collect();
        assert!(close(
            si_sdr(&s, &scaled).unwrap(),
            10.0 * 4f64.log10(),
            1e-6
        ));
        // A perfect estimate up to scale.
        let perfect: Vec<f32> = s.iter().map(|x| -2.0 * x).collect();
        assert_eq!(si_sdr(&s, &perfect).unwrap(), f64::INFINITY);
        // Errors.
        assert_eq!(si_sdr(&[], &[]), Err(AudioError::Empty));
        assert_eq!(
            si_sdr(&s, &s[..3]),
            Err(AudioError::LengthMismatch {
                reference: 4,
                estimate: 3
            })
        );
        assert_eq!(si_sdr(&[0.5; 4], &s), Err(AudioError::SilentReference));
    }

    #[test]
    fn mel_scale_and_filterbank() {
        assert!(close(hz_to_mel(0.0), 0.0, 1e-12));
        assert!(close(hz_to_mel(700.0), 2595.0 * 2f64.log10(), 1e-9));
        assert!(close(mel_to_hz(hz_to_mel(1234.5)), 1234.5, 1e-9));

        let cfg = StftConfig {
            sample_rate: 8000,
            n_fft: 16,
            hop: 8,
            n_mels: 3,
            ..StftConfig::default()
        };
        let bank = mel_filterbank(&cfg).unwrap();
        assert_eq!(bank.len(), 3);
        assert!(bank.iter().all(|r| r.len() == 9));
        // Edges equally spaced in mel between 0 and 4000 Hz.
        let hi = hz_to_mel(4000.0);
        let edges: Vec<f64> = (0..5).map(|i| mel_to_hz(hi * i as f64 / 4.0)).collect();
        for (m, row) in bank.iter().enumerate() {
            for (k, &w) in row.iter().enumerate() {
                let f = k as f64 * 500.0;
                let expect = if f <= edges[m] || f >= edges[m + 2] {
                    0.0
                } else if f <= edges[m + 1] {
                    (f - edges[m]) / (edges[m + 1] - edges[m])
                } else {
                    (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1])
                };
                assert!(close(w, expect, 1e-12), "m {m} k {k}");
                assert!((0.0..=1.0).contains(&w));
            }
        }
        // DC and Nyquist bins lie on the outer edges.
        assert_eq!(bank[0][0], 0.0);
        assert_eq!(bank[2][8], 0.0);

        assert!(mel_filterbank(&StftConfig {
            f_max: Some(9000.0),
            ..cfg
        })
        .is_err());
        assert!(mel_filterbank(&StftConfig { hop: 0, ..cfg }).is_err());
    }

    #[test]
    fn stft_matches_a_naive_dft() {
        let cfg = StftConfig {
            sample_rate: 8000,
            n_fft: 8,
            hop: 4,
            n_mels: 2,
            ..StftConfig::default()
        };
        assert_eq!(cfg.frames(8), 1);
        assert_eq!(cfg.frames(9), 2);
        assert_eq!(cfg.frames(12), 2);
        assert_eq!(cfg.frames(13), 3);
        assert_eq!(cfg.frames(3), 1);
        let x: Vec<f32> = (0..10).map(|i| ((i * 7 % 5) as f32) - 2.0).collect();
        let p = power_spectrogram(&x, &cfg).unwrap();
        assert_eq!(p.len(), 2);
        let w = hann(8);
        assert!(close(w[0], 0.0, 1e-15) && close(w[4], 1.0, 1e-15));
        for (f, frame) in p.iter().enumerate() {
            for (k, &pk) in frame.iter().enumerate() {
                let (mut re, mut im) = (0.0, 0.0);
                for (i, wi) in w.iter().enumerate() {
                    let idx = f * 4 + i;
                    let xi = if idx < x.len() {
                        f64::from(x[idx])
                    } else {
                        0.0
                    };
                    let ang = -2.0 * std::f64::consts::PI * (k * i) as f64 / 8.0;
                    re += xi * wi * ang.cos();
                    im += xi * wi * ang.sin();
                }
                assert!(close(pk, re * re + im * im, 1e-9), "frame {f} bin {k}");
            }
        }
    }

    /// Deterministic broadband test signal.
    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0) as f32
            })
            .collect()
    }

    #[test]
    fn log_mel_distance_hand_computed() {
        let cfg = StftConfig::default();
        let x = noise(24_000, 1);
        let d = log_mel_distance(&x, &x, &cfg).unwrap();
        assert_eq!((d.l1, d.rmse), (0.0, 0.0));
        assert_eq!(d.frames, cfg.frames(24_000));
        assert_eq!(d.frames, 1 + (24_000 - 1024usize).div_ceil(256));

        // 10x amplitude = 100x power = +2 in log10 wherever eps is negligible
        // (broadband noise puts plenty of energy in every band).
        let loud: Vec<f32> = x.iter().map(|v| 10.0 * v).collect();
        let d = log_mel_distance(&x, &loud, &cfg).unwrap();
        assert!(close(d.l1, 2.0, 1e-3), "{d:?}");
        assert!(close(d.rmse, 2.0, 1e-3), "{d:?}");

        // Compared over the common length.
        let d = log_mel_distance(&x, &x[..12_000], &cfg).unwrap();
        assert_eq!(d.samples, 12_000);
        assert_eq!(d.l1, 0.0);
        assert_eq!(log_mel_distance(&x, &[], &cfg), Err(AudioError::Empty));

        // A 3 kHz tone peaks in the band whose center is nearest to 3 kHz.
        let tone: Vec<f32> = (0..4096)
            .map(|i| (2.0 * std::f64::consts::PI * 3000.0 * i as f64 / 24_000.0).sin() as f32)
            .collect();
        let lm = log_mel_spectrogram(&tone, &cfg).unwrap();
        let frame = &lm[1];
        let best = (0..cfg.n_mels)
            .max_by(|&a, &b| frame[a].total_cmp(&frame[b]))
            .unwrap();
        let hi = hz_to_mel(12_000.0);
        let centers: Vec<f64> = (1..=cfg.n_mels)
            .map(|i| mel_to_hz(hi * i as f64 / (cfg.n_mels + 1) as f64))
            .collect();
        let nearest = (0..cfg.n_mels)
            .min_by(|&a, &b| {
                (centers[a] - 3000.0)
                    .abs()
                    .total_cmp(&(centers[b] - 3000.0).abs())
            })
            .unwrap();
        assert_eq!(best, nearest);
        // Silence sits at the floor: log10(eps).
        let silent = log_mel_spectrogram(&[0.0; 2048], &cfg).unwrap();
        assert!(silent.iter().flatten().all(|&v| close(v, -5.0, 1e-12)));
    }
}
