//! Audio codec interface for the engine.

use std::sync::Arc;

use s2s_core::streams::{CODEBOOK_SIZE, FRAME_SAMPLES, SAMPLE_RATE};
use s2s_mimi::{Codes, DecoderState, EncoderState, Mimi};

/// A streaming audio codec: one 80 ms frame of 24 kHz PCM in, one code per
/// codebook out, and back. Implementations keep their own streaming state
/// (one instance per session).
pub trait FrameCodec: Send {
    /// Codebooks per frame.
    fn codebooks(&self) -> usize;
    /// Entries per codebook. The stream layout assumes `CODEBOOK_SIZE`
    /// (audio tokens `0..CODEBOOK_SIZE`, plus `AUDIO_PAD`).
    fn codebook_size(&self) -> u32;
    /// Encode one frame of `FRAME_SAMPLES` samples.
    fn encode(&mut self, pcm: &[f32]) -> Vec<u32>;
    /// Decode one frame of codes into `FRAME_SAMPLES` samples.
    fn decode(&mut self, codes: &[u32]) -> Vec<f32>;
    /// Codes for a silent frame (forced on unused audio streams in STT/TTS
    /// modes).
    fn silence_codes(&self) -> Vec<u32>;
}

/// A deterministic stand-in codec for development and tests.
///
/// Encoding maps the frame's RMS level to a code (the same code on every
/// codebook); decoding plays a 440 Hz tone whose amplitude follows codebook
/// 0. It carries no speech, but it exercises every engine path with real
/// timing and frame sizes.
#[derive(Clone, Debug)]
pub struct ToneCodec {
    codebooks: usize,
    phase: f64,
}

impl ToneCodec {
    /// New codec with `codebooks` codebooks.
    pub fn new(codebooks: usize) -> Self {
        Self {
            codebooks,
            phase: 0.0,
        }
    }
}

impl FrameCodec for ToneCodec {
    fn codebooks(&self) -> usize {
        self.codebooks
    }

    fn codebook_size(&self) -> u32 {
        CODEBOOK_SIZE
    }

    fn encode(&mut self, pcm: &[f32]) -> Vec<u32> {
        assert_eq!(pcm.len(), FRAME_SAMPLES);
        let rms = (pcm.iter().map(|x| x * x).sum::<f32>() / pcm.len() as f32).sqrt();
        let code = ((rms.min(1.0) * (CODEBOOK_SIZE - 1) as f32) as u32).min(CODEBOOK_SIZE - 1);
        vec![code; self.codebooks]
    }

    fn decode(&mut self, codes: &[u32]) -> Vec<f32> {
        let amp = codes.first().map_or(0.0, |&c| {
            c.min(CODEBOOK_SIZE - 1) as f64 / CODEBOOK_SIZE as f64
        });
        let step = 2.0 * std::f64::consts::PI * 440.0 / SAMPLE_RATE as f64;
        (0..FRAME_SAMPLES)
            .map(|_| {
                let s = (amp * self.phase.sin()) as f32;
                self.phase = (self.phase + step) % (2.0 * std::f64::consts::PI);
                s
            })
            .collect()
    }

    fn silence_codes(&self) -> Vec<u32> {
        vec![0; self.codebooks]
    }
}

/// The Mimi codec (`s2s-mimi`) as a [`FrameCodec`]. One loaded model is
/// shared by every session (`Arc<Mimi>`); each codec instance holds one
/// session's streaming encoder and decoder state.
#[derive(Clone, Debug)]
pub struct MimiCodec {
    mimi: Arc<Mimi>,
    enc: EncoderState,
    dec: DecoderState,
    silence: Vec<u32>,
}

impl MimiCodec {
    /// A new session codec over a shared model.
    ///
    /// # Panics
    /// If the model is not mono or its frame size is not `FRAME_SAMPLES`.
    pub fn new(mimi: Arc<Mimi>) -> Self {
        let cfg = mimi.config();
        assert_eq!(cfg.channels, 1, "the engine streams mono audio");
        assert_eq!(
            cfg.frame_size(),
            FRAME_SAMPLES,
            "Mimi frame size must be 80 ms at 24 kHz"
        );
        // The codes Mimi assigns to a silent frame from a fresh state.
        let silence = mimi
            .encode_step(&mut mimi.encoder_state(), &[0.0; FRAME_SAMPLES])
            .map(|c| c.frame(0))
            .unwrap_or_else(|_| vec![0; cfg.quantizer_n_q]);
        Self {
            enc: mimi.encoder_state(),
            dec: mimi.decoder_state(),
            silence,
            mimi,
        }
    }
}

impl FrameCodec for MimiCodec {
    fn codebooks(&self) -> usize {
        self.mimi.config().quantizer_n_q
    }

    fn codebook_size(&self) -> u32 {
        self.mimi.config().quantizer_bins as u32
    }

    fn encode(&mut self, pcm: &[f32]) -> Vec<u32> {
        // Exactly one frame in gives exactly one code frame out; on an
        // (unexpected) error, fall back to silence rather than stall the clock.
        match self.mimi.encode_step(&mut self.enc, pcm) {
            Ok(c) if c.frames() == 1 => c.frame(0),
            _ => self.silence.clone(),
        }
    }

    fn decode(&mut self, codes: &[u32]) -> Vec<f32> {
        Codes::new(codes.len(), 1, codes.to_vec())
            .and_then(|c| self.mimi.decode_step(&mut self.dec, &c))
            .unwrap_or_else(|_| vec![0.0; FRAME_SAMPLES])
    }

    fn silence_codes(&self) -> Vec<u32> {
        self.silence.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tone_codec_round_trip_levels() {
        let mut c = ToneCodec::new(3);
        let silent = vec![0.0f32; FRAME_SAMPLES];
        assert_eq!(c.encode(&silent), c.silence_codes());
        let loud = vec![0.5f32; FRAME_SAMPLES];
        let codes = c.encode(&loud);
        assert_eq!(codes.len(), 3);
        assert!(codes[0] > 900);
        let pcm = c.decode(&codes);
        assert_eq!(pcm.len(), FRAME_SAMPLES);
        assert!(pcm.iter().any(|&x| x.abs() > 0.1));
        assert!(c.decode(&[0, 0, 0]).iter().all(|&x| x == 0.0));
    }
}
