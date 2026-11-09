//! Offline tokenization: audio files to Mimi codes and back (plan §9.3).
//!
//! The codes file is JSON:
//! `{"n_q": 8, "frame_rate": 12.5, "frames": [[c0, …, c7], …]}`, one row of
//! `n_q` codes per 80 ms frame. `frames` is exactly the per-turn
//! `codes` format of `s2s-data` synthesis scripts, so recorded or TTS audio
//! can be placed on a dialogue timeline.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{audio, Codes, Mimi, MimiError, Result};

/// A tokenized audio file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodesFile {
    /// Codebooks per frame.
    pub n_q: usize,
    /// Frames per second (12.5).
    pub frame_rate: f64,
    /// One row of `n_q` codes per frame.
    pub frames: Vec<Vec<u32>>,
}

/// Encode 24 kHz mono samples. A trailing partial frame is zero-padded, so
/// the result covers all of the audio.
pub fn encode_samples(mimi: &Mimi, pcm: &[f32]) -> Result<CodesFile> {
    let frame = mimi.config().frame_size();
    let mut padded = pcm.to_vec();
    padded.resize(pcm.len().div_ceil(frame).max(1) * frame, 0.0);
    let mut enc = mimi.streaming_encoder();
    let codes = enc.push(&padded)?;
    Ok(CodesFile {
        n_q: codes.n_q(),
        frame_rate: mimi.config().frame_rate,
        frames: (0..codes.frames()).map(|f| codes.frame(f)).collect(),
    })
}

/// Decode a codes file to 24 kHz mono samples.
pub fn decode_codes(mimi: &Mimi, file: &CodesFile) -> Result<Vec<f32>> {
    if file.frames.iter().any(|r| r.len() != file.n_q) {
        return Err(MimiError::Input("every frame must have n_q codes".into()));
    }
    let mut data = vec![0u32; file.n_q * file.frames.len()];
    for (f, row) in file.frames.iter().enumerate() {
        for (q, &c) in row.iter().enumerate() {
            data[q * file.frames.len() + f] = c;
        }
    }
    let codes = Codes::new(file.n_q, file.frames.len(), data)?;
    mimi.streaming_decoder().push(&codes)
}

/// Encode a WAV file to a codes file.
pub fn encode_wav(mimi: &Mimi, wav: impl AsRef<Path>, out: impl AsRef<Path>) -> Result<CodesFile> {
    let pcm = audio::read_wav_mono_24k(wav)?;
    let file = encode_samples(mimi, &pcm)?;
    let json = serde_json::to_string(&file).map_err(|e| MimiError::Input(e.to_string()))?;
    std::fs::write(out, json).map_err(|e| MimiError::Input(format!("write codes: {e}")))?;
    Ok(file)
}

/// Decode a codes file to a 24 kHz mono 16-bit WAV.
pub fn decode_to_wav(mimi: &Mimi, codes: impl AsRef<Path>, out: impl AsRef<Path>) -> Result<usize> {
    let text =
        std::fs::read_to_string(codes).map_err(|e| MimiError::Input(format!("read codes: {e}")))?;
    let file: CodesFile =
        serde_json::from_str(&text).map_err(|e| MimiError::Input(e.to_string()))?;
    let pcm = decode_codes(mimi, &file)?;
    audio::write_wav_24k(out, &pcm)?;
    Ok(file.frames.len())
}
