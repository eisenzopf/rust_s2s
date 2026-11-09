//! WAV input/output and resampling for offline tokenization (plan §9.3).
//!
//! Mimi works on 24 kHz mono. [`read_wav_mono_24k`] accepts any WAV that
//! `hound` reads (8/16/24/32-bit integer or 32-bit float, any rate, any
//! channel count), downmixes to mono by averaging channels and resamples
//! with linear interpolation. Linear interpolation is enough for codec
//! tokenization of speech; use a proper resampler upstream if the source is
//! far from 24 kHz and fidelity matters.

use std::path::Path;

use crate::{MimiError, Result};

/// Target sample rate.
pub const SAMPLE_RATE: u32 = 24_000;

fn io_err(e: hound::Error) -> MimiError {
    MimiError::Input(format!("wav: {e}"))
}

/// Read a WAV file as 24 kHz mono `f32` samples in `[-1, 1]`.
pub fn read_wav_mono_24k(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).map_err(io_err)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        return Err(MimiError::Input("wav has no channels".into()));
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()
            .map_err(io_err)?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<std::result::Result<_, _>>()
                .map_err(io_err)?
        }
    };
    let mono: Vec<f32> = samples
        .chunks_exact(channels)
        .map(|c| c.iter().sum::<f32>() / channels as f32)
        .collect();
    Ok(resample_linear(&mono, spec.sample_rate, SAMPLE_RATE))
}

/// Write 24 kHz mono samples as a 16-bit PCM WAV (clipped to `[-1, 1]`).
pub fn write_wav_24k(path: impl AsRef<Path>, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).map_err(io_err)?;
    for &s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16)
            .map_err(io_err)?;
    }
    w.finalize().map_err(io_err)
}

/// Linear-interpolation resampling from `from` Hz to `to` Hz.
pub fn resample_linear(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let n_out = ((x.len() as u64 * to as u64) / from as u64) as usize;
    let step = from as f64 / to as f64;
    (0..n_out)
        .map(|i| {
            let pos = i as f64 * step;
            let j = pos.floor() as usize;
            let frac = (pos - j as f64) as f32;
            let a = x[j.min(x.len() - 1)];
            let b = x[(j + 1).min(x.len() - 1)];
            a + (b - a) * frac
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_lengths_and_values() {
        let x: Vec<f32> = (0..480).map(|i| i as f32).collect();
        let y = resample_linear(&x, 48_000, 24_000);
        assert_eq!(y.len(), 240);
        assert_eq!(y[10], 20.0);
        let z = resample_linear(&x[..100], 16_000, 24_000);
        assert_eq!(z.len(), 150);
        assert!((z[3] - 2.0).abs() < 1e-6);
        assert_eq!(resample_linear(&x, 24_000, 24_000), x);
    }

    #[test]
    fn wav_round_trip_with_downmix_and_resample() {
        let dir = std::env::temp_dir().join(format!("s2s-mimi-audio-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // 48 kHz stereo float input; both channels carry the same tone.
        let src = dir.join("in.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(&src, spec).unwrap();
        for i in 0..4800 {
            let s = (i as f32 * 0.01).sin() * 0.5;
            w.write_sample(s).unwrap();
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();
        let mono = read_wav_mono_24k(&src).unwrap();
        assert_eq!(mono.len(), 2400);
        assert!((mono[100] - (200.0f32 * 0.01).sin() * 0.5).abs() < 1e-4);

        let out = dir.join("out.wav");
        write_wav_24k(&out, &mono).unwrap();
        let back = read_wav_mono_24k(&out).unwrap();
        assert_eq!(back.len(), mono.len());
        for (a, b) in back.iter().zip(&mono) {
            assert!((a - b).abs() < 1e-4);
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
