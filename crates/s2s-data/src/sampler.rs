//! Deterministic, resumable sampling of fixed-length windows.
//!
//! Everything random is a pure function of `(seed, epoch, ...)`, so the
//! sampler has no hidden RNG state: a [`Cursor`] (epoch, index into that
//! epoch's order, positions already consumed from the current sample) pins
//! down the exact next window, and restoring a saved cursor reproduces the
//! stream bit for bit.

use s2s_core::SplitMix64;
use serde::{Deserialize, Serialize};

use crate::assemble::{Example, IGNORE_INDEX};
use crate::{Error, Result};

/// Where the loader is in its data stream. Save it with a checkpoint
/// ([`to_json`](Self::to_json)) and restore it to resume exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Cursor {
    /// Sampler seed (checked on restore).
    pub seed: u64,
    /// Dataset size (checked on restore).
    pub num_samples: u64,
    /// Epoch (each epoch is one pass over a fresh permutation).
    pub epoch: u64,
    /// Index into the epoch's sample order.
    pub index: u64,
    /// Positions of the current sample already consumed ([`WindowMode::Pack`]).
    pub offset: u64,
}

impl Cursor {
    /// The beginning of epoch 0.
    pub fn start(seed: u64, num_samples: u64) -> Self {
        Self {
            seed,
            num_samples,
            epoch: 0,
            index: 0,
            offset: 0,
        }
    }

    /// Serialize as JSON.
    pub fn to_json(&self) -> String {
        // Plain integers: serialization cannot fail.
        serde_json::to_string(self).unwrap_or_default()
    }

    /// Parse from JSON.
    pub fn from_json(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    /// Advance to the next sample (wrapping into the next epoch).
    pub(crate) fn next_sample(&mut self) {
        self.offset = 0;
        self.index += 1;
        if self.index >= self.num_samples {
            self.index = 0;
            self.epoch += 1;
        }
    }
}

/// Hash several words into one (SplitMix64 finalizer chain).
pub(crate) fn mix(words: &[u64]) -> u64 {
    words.iter().fold(0x5DEE_CE66_D1CE_4E5Bu64, |h, &w| {
        SplitMix64::new(h ^ w.wrapping_mul(0x9E37_79B9_7F4A_7C15)).next_u64()
    })
}

const ORDER_TAG: u64 = 0x006F_7264_6572;
const CROP_TAG: u64 = 0x6372_6F70;

/// The sample order of one epoch: a seeded Fisher–Yates permutation of
/// `0..n`, or the identity when `shuffle` is false.
pub fn epoch_order(seed: u64, epoch: u64, n: usize, shuffle: bool) -> Vec<u32> {
    let mut v: Vec<u32> = (0..n as u32).collect();
    if shuffle {
        let mut rng = SplitMix64::new(mix(&[ORDER_TAG, seed, epoch]));
        for i in (1..n).rev() {
            let j = rng.below(i as u64 + 1) as usize;
            v.swap(i, j);
        }
    }
    v
}

/// Maps `(epoch, index)` to a sample id, caching the current epoch's order.
#[derive(Clone, Debug)]
pub struct Sampler {
    seed: u64,
    n: usize,
    shuffle: bool,
    epoch: Option<u64>,
    order: Vec<u32>,
}

impl Sampler {
    /// A sampler over `n` samples. Errors if `n` is 0 or exceeds `u32::MAX`.
    pub fn new(seed: u64, n: usize, shuffle: bool) -> Result<Self> {
        if n == 0 || n > u32::MAX as usize {
            return Err(Error::Invalid(format!(
                "sampler needs 1..=u32::MAX samples, got {n}"
            )));
        }
        Ok(Self {
            seed,
            n,
            shuffle,
            epoch: None,
            order: Vec::new(),
        })
    }

    /// Sample id at `index` of `epoch`'s order.
    ///
    /// # Panics
    /// If `index >= n`.
    pub fn sample_at(&mut self, epoch: u64, index: u64) -> usize {
        if self.epoch != Some(epoch) {
            self.order = epoch_order(self.seed, epoch, self.n, self.shuffle);
            self.epoch = Some(epoch);
        }
        self.order[index as usize] as usize
    }

    /// Crop start for an example of `positions` positions in a window of
    /// `window` positions ([`WindowMode::Crop`]); 0 if it fits.
    pub fn crop_offset(&self, epoch: u64, sample: usize, positions: usize, window: usize) -> usize {
        if positions <= window {
            return 0;
        }
        let span = (positions - window + 1) as u64;
        (mix(&[CROP_TAG, self.seed, epoch, sample as u64]) % span) as usize
    }
}

/// How examples become fixed-length windows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowMode {
    /// One window per sample: a seeded random crop if it is longer than the
    /// window, else the whole sample followed by filler.
    Crop,
    /// Concatenate samples (in epoch order) and cut consecutive windows, so
    /// no position is wasted; long samples span several windows and a window
    /// can hold the tail of one sample and the head of the next
    /// (distinguished by [`Window::segment_ids`]).
    #[default]
    Pack,
}

/// One fixed-length window of `len` positions.
///
/// Filler positions (after the last sample in [`WindowMode::Crop`]) hold the
/// pad tokens, [`IGNORE_INDEX`] targets, weight 0 and segment id 0.
#[derive(Clone, Debug, PartialEq)]
pub struct Window {
    /// Window length `T`.
    pub len: usize,
    /// Streams per position `S`.
    pub num_streams: usize,
    /// Generated streams `K`.
    pub num_generated: usize,
    /// `[T, S]` input tokens.
    pub tokens: Vec<u32>,
    /// `[T, K]` targets.
    pub targets: Vec<i64>,
    /// `[T, K]` loss weights.
    pub weights: Vec<f32>,
    /// `[T]` injection mask.
    pub is_injection: Vec<u8>,
    /// `[T]` 1-based id of the sample segment each position belongs to
    /// within this window (0 = filler). Attention must not cross segments.
    pub segment_ids: Vec<u32>,
    /// `[T]` position index within the segment's example (for RoPE).
    pub positions: Vec<u32>,
}

impl Window {
    pub(crate) fn with_capacity(len: usize, num_streams: usize, num_generated: usize) -> Self {
        Self {
            len: 0,
            num_streams,
            num_generated,
            tokens: Vec::with_capacity(len * num_streams),
            targets: Vec::with_capacity(len * num_generated),
            weights: Vec::with_capacity(len * num_generated),
            is_injection: Vec::with_capacity(len),
            segment_ids: Vec::with_capacity(len),
            positions: Vec::with_capacity(len),
        }
    }

    /// Append positions `start..start + n` of `ex` as segment `segment`.
    pub(crate) fn push_slice(&mut self, ex: &Example, start: usize, n: usize, segment: u32) {
        let (s, k) = (self.num_streams, self.num_generated);
        self.tokens
            .extend_from_slice(&ex.tokens[start * s..(start + n) * s]);
        self.targets
            .extend_from_slice(&ex.targets[start * k..(start + n) * k]);
        self.weights
            .extend_from_slice(&ex.weights[start * k..(start + n) * k]);
        self.is_injection
            .extend_from_slice(&ex.is_injection[start..start + n]);
        self.segment_ids.extend(std::iter::repeat_n(segment, n));
        self.positions.extend((start..start + n).map(|p| p as u32));
        self.len += n;
    }

    /// Append filler positions until the window has `len` positions.
    pub(crate) fn fill_to(&mut self, len: usize, pad_row: &[u32]) {
        while self.len < len {
            self.tokens.extend_from_slice(pad_row);
            self.targets
                .extend(std::iter::repeat_n(IGNORE_INDEX, self.num_generated));
            self.weights
                .extend(std::iter::repeat_n(0.0, self.num_generated));
            self.is_injection.push(0);
            self.segment_ids.push(0);
            self.positions.push(0);
            self.len += 1;
        }
    }
}
