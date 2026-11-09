//! Data containers: channel-major signals and code grids.

use crate::{MimiError, Result};

/// A multi-channel sequence, channel-major `[channels, len]` (Kyutai's conv
/// layout without the batch dimension). Used for PCM and latents.
#[derive(Clone, Debug, PartialEq)]
pub struct Signal {
    /// Number of channels.
    pub channels: usize,
    /// Number of time steps.
    pub len: usize,
    /// `channels * len` values, channel-major.
    pub data: Vec<f32>,
}

impl Signal {
    /// Wrap `data` (`[channels, len]`).
    ///
    /// # Panics
    /// If `data.len() != channels * len`.
    pub fn new(channels: usize, len: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), channels * len, "Signal: data length");
        Self {
            channels,
            len,
            data,
        }
    }

    /// Mono signal from samples.
    pub fn mono(samples: &[f32]) -> Self {
        Self::new(1, samples.len(), samples.to_vec())
    }

    /// Channel `c` as a slice.
    pub fn channel(&self, c: usize) -> &[f32] {
        &self.data[c * self.len..(c + 1) * self.len]
    }

    /// Time-major copy `[len, channels]`.
    pub fn transposed(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.data.len()];
        for c in 0..self.channels {
            for t in 0..self.len {
                out[t * self.channels + c] = self.data[c * self.len + t];
            }
        }
        out
    }

    /// From time-major `[len, channels]` data.
    pub fn from_time_major(channels: usize, len: usize, x: &[f32]) -> Self {
        assert_eq!(x.len(), channels * len, "Signal: data length");
        let mut data = vec![0.0; x.len()];
        for t in 0..len {
            for c in 0..channels {
                data[c * len + t] = x[t * channels + c];
            }
        }
        Self::new(channels, len, data)
    }

    /// Steps `from..to` of every channel.
    pub fn slice(&self, from: usize, to: usize) -> Self {
        assert!(from <= to && to <= self.len, "Signal::slice out of range");
        let mut data = Vec::with_capacity(self.channels * (to - from));
        for c in 0..self.channels {
            data.extend_from_slice(&self.data[c * self.len + from..c * self.len + to]);
        }
        Self::new(self.channels, to - from, data)
    }

    /// Concatenate along time.
    ///
    /// # Panics
    /// If channel counts differ.
    pub fn concat(parts: &[Signal]) -> Self {
        let channels = parts.first().map_or(0, |p| p.channels);
        assert!(
            parts.iter().all(|p| p.channels == channels),
            "Signal::concat: channels"
        );
        let len = parts.iter().map(|p| p.len).sum();
        let mut data = Vec::with_capacity(channels * len);
        for c in 0..channels {
            for p in parts {
                data.extend_from_slice(p.channel(c));
            }
        }
        Self::new(channels, len, data)
    }
}

/// Audio codes `[n_q, frames]`, row-major (codebook-major), each in
/// `0..bins`. Row 0 is the semantic codebook.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Codes {
    n_q: usize,
    frames: usize,
    data: Vec<u32>,
}

impl Codes {
    /// Wrap `data` (`[n_q, frames]`).
    pub fn new(n_q: usize, frames: usize, data: Vec<u32>) -> Result<Self> {
        if data.len() != n_q * frames {
            return Err(MimiError::Input(format!(
                "codes: {} values for [{n_q}, {frames}]",
                data.len()
            )));
        }
        Ok(Self { n_q, frames, data })
    }

    /// Build from per-frame code vectors (each of length `n_q`).
    pub fn from_frames(frames: &[Vec<u32>]) -> Result<Self> {
        let n_q = frames.first().map_or(0, Vec::len);
        if frames.iter().any(|f| f.len() != n_q) {
            return Err(MimiError::Input("codes: frames of different sizes".into()));
        }
        let mut data = vec![0u32; n_q * frames.len()];
        for (f, fr) in frames.iter().enumerate() {
            for (q, &c) in fr.iter().enumerate() {
                data[q * frames.len() + f] = c;
            }
        }
        Self::new(n_q, frames.len(), data)
    }

    /// Number of codebooks.
    pub fn n_q(&self) -> usize {
        self.n_q
    }

    /// Number of frames.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// The `[n_q, frames]` data.
    pub fn data(&self) -> &[u32] {
        &self.data
    }

    /// Code of codebook `q` at frame `f`.
    pub fn get(&self, q: usize, f: usize) -> u32 {
        self.data[q * self.frames + f]
    }

    /// The `n_q` codes of frame `f`.
    pub fn frame(&self, f: usize) -> Vec<u32> {
        (0..self.n_q).map(|q| self.get(q, f)).collect()
    }

    /// Frames `from..to`.
    pub fn slice(&self, from: usize, to: usize) -> Self {
        assert!(from <= to && to <= self.frames, "Codes::slice out of range");
        let frames: Vec<Vec<u32>> = (from..to).map(|f| self.frame(f)).collect();
        if frames.is_empty() {
            return Self {
                n_q: self.n_q,
                frames: 0,
                data: Vec::new(),
            };
        }
        Self::from_frames(&frames).expect("consistent")
    }

    /// The first `k` codebooks.
    pub fn truncate_codebooks(&self, k: usize) -> Self {
        let k = k.min(self.n_q);
        Self {
            n_q: k,
            frames: self.frames,
            data: self.data[..k * self.frames].to_vec(),
        }
    }

    /// Concatenate along frames.
    pub fn concat(parts: &[Codes]) -> Result<Self> {
        let n_q = parts.first().map_or(0, |p| p.n_q);
        if parts.iter().any(|p| p.n_q != n_q) {
            return Err(MimiError::Input("codes: different codebook counts".into()));
        }
        let frames: usize = parts.iter().map(|p| p.frames).sum();
        let mut data = Vec::with_capacity(n_q * frames);
        for q in 0..n_q {
            for p in parts {
                data.extend_from_slice(&p.data[q * p.frames..(q + 1) * p.frames]);
            }
        }
        Self::new(n_q, frames, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_layouts() {
        let s = Signal::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(s.transposed(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(Signal::from_time_major(2, 3, &s.transposed()), s);
        let parts = [s.slice(0, 1), s.slice(1, 3)];
        assert_eq!(Signal::concat(&parts), s);
        assert_eq!(s.channel(1), [4.0, 5.0, 6.0]);
    }

    #[test]
    fn code_layouts() {
        let c = Codes::from_frames(&[vec![1, 2], vec![3, 4], vec![5, 6]]).unwrap();
        assert_eq!((c.n_q(), c.frames()), (2, 3));
        assert_eq!(c.data(), [1, 3, 5, 2, 4, 6]);
        assert_eq!(c.frame(1), [3, 4]);
        assert_eq!(c.get(1, 2), 6);
        let parts = [c.slice(0, 2), c.slice(2, 3)];
        assert_eq!(Codes::concat(&parts).unwrap(), c);
        assert_eq!(c.truncate_codebooks(1).data(), [1, 3, 5]);
        assert!(Codes::new(2, 2, vec![0; 3]).is_err());
    }
}
