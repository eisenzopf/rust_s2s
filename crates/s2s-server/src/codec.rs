//! Audio codecs at the network edge.
//!
//! Everything inside the server is 24 kHz mono `f32` PCM in `[-1, 1]`. This
//! module converts between that and the wire formats of [`AudioFormat`]:
//!
//! - **Opus** via libopus (the `opus` crate; libopus is built from bundled
//!   source when the system has none). Opus runs natively at 24 kHz, so no
//!   resampling is needed. The decoder accepts any legal packet duration
//!   (2.5–120 ms); the encoder emits one packet per 80 ms frame.
//! - **Raw PCM**, `s16le` or `f32le`, for testing and simple clients.

use s2s_core::streams::{FRAME_SAMPLES, SAMPLE_RATE};

use crate::protocol::{AudioChunk, AudioFormat};

/// Largest number of samples one Opus packet can decode to at 24 kHz
/// (120 ms).
pub const OPUS_MAX_PACKET_SAMPLES: usize = SAMPLE_RATE as usize * 120 / 1000;

/// Upper bound on the size of an encoded Opus packet we produce, in bytes.
pub const OPUS_MAX_PACKET_BYTES: usize = 4000;

/// Default Opus bitrate for server → client audio, in bits per second.
pub const DEFAULT_OPUS_BITRATE: i32 = 32_000;

/// A codec failure.
#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    /// libopus reported an error.
    #[error("opus: {0}")]
    Opus(#[from] opus::Error),
    /// A PCM payload was not a whole number of samples.
    #[error("PCM payload of {len} bytes is not a multiple of {sample_bytes}")]
    PartialSample {
        /// Payload length in bytes.
        len: usize,
        /// Bytes per sample for the format.
        sample_bytes: usize,
    },
    /// A frame passed to the encoder had the wrong length.
    #[error(
        "cannot encode {0} samples; Opus at 24 kHz takes 2.5, 5, 10, 20, 40, 60, 80, 100 or 120 ms"
    )]
    BadFrameLength(usize),
}

/// Opus decoder producing 24 kHz mono `f32`.
#[derive(Debug)]
pub struct OpusDecoder {
    inner: opus::Decoder,
    scratch: Vec<f32>,
}

impl OpusDecoder {
    /// A fresh decoder.
    pub fn new() -> Result<Self, CodecError> {
        Ok(Self {
            inner: opus::Decoder::new(SAMPLE_RATE, opus::Channels::Mono)?,
            scratch: vec![0.0; OPUS_MAX_PACKET_SAMPLES],
        })
    }

    /// Decodes one packet, appending its samples to `out`. Returns the
    /// number of samples appended.
    pub fn decode(&mut self, packet: &[u8], out: &mut Vec<f32>) -> Result<usize, CodecError> {
        let n = self.inner.decode_float(packet, &mut self.scratch, false)?;
        out.extend_from_slice(&self.scratch[..n]);
        Ok(n)
    }
}

/// Opus encoder taking 24 kHz mono `f32`.
#[derive(Debug)]
pub struct OpusEncoder {
    inner: opus::Encoder,
    buf: Vec<u8>,
}

impl OpusEncoder {
    /// A VoIP-tuned encoder at `bitrate` bits per second.
    pub fn new(bitrate: i32) -> Result<Self, CodecError> {
        let mut inner =
            opus::Encoder::new(SAMPLE_RATE, opus::Channels::Mono, opus::Application::Voip)?;
        inner.set_bitrate(opus::Bitrate::Bits(bitrate))?;
        Ok(Self {
            inner,
            buf: vec![0; OPUS_MAX_PACKET_BYTES],
        })
    }

    /// Encodes `pcm` into one packet. `pcm` must be a legal Opus frame
    /// duration at 24 kHz; the server always passes one 80 ms frame (1920
    /// samples).
    pub fn encode(&mut self, pcm: &[f32]) -> Result<Vec<u8>, CodecError> {
        // 2.5, 5, 10, 20, 40, 60, 80, 100, 120 ms at 24 kHz.
        const LEGAL: [usize; 9] = [60, 120, 240, 480, 960, 1440, 1920, 2400, 2880];
        if !LEGAL.contains(&pcm.len()) {
            return Err(CodecError::BadFrameLength(pcm.len()));
        }
        let n = self.inner.encode_float(pcm, &mut self.buf)?;
        Ok(self.buf[..n].to_vec())
    }
}

/// Appends little-endian `i16` samples, scaled to `[-1, 1)`, to `out`.
pub fn s16le_to_f32(bytes: &[u8], out: &mut Vec<f32>) -> Result<usize, CodecError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(CodecError::PartialSample {
            len: bytes.len(),
            sample_bytes: 2,
        });
    }
    out.extend(
        bytes
            .chunks_exact(2)
            .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0),
    );
    Ok(bytes.len() / 2)
}

/// Appends little-endian `f32` samples to `out`. Non-finite values become 0
/// and everything is clamped to `[-1, 1]`, so a bad client cannot inject
/// NaNs into the engine.
pub fn f32le_to_f32(bytes: &[u8], out: &mut Vec<f32>) -> Result<usize, CodecError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(CodecError::PartialSample {
            len: bytes.len(),
            sample_bytes: 4,
        });
    }
    out.extend(bytes.chunks_exact(4).map(|b| {
        let x = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        if x.is_finite() {
            x.clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }));
    Ok(bytes.len() / 4)
}

/// Appends `pcm` as little-endian `i16` (clamped, rounded) to `out`.
pub fn f32_to_s16le(pcm: &[f32], out: &mut Vec<u8>) {
    out.reserve(pcm.len() * 2);
    for &x in pcm {
        let s = (x.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&s.to_le_bytes());
    }
}

/// Appends `pcm` as little-endian `f32` to `out`.
pub fn f32_to_f32le(pcm: &[f32], out: &mut Vec<u8>) {
    out.reserve(pcm.len() * 4);
    for &x in pcm {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

/// Decodes client audio chunks of any format into PCM. Holds one Opus
/// decoder (created on first use) so packet-to-packet state carries over.
#[derive(Debug, Default)]
pub struct AudioDecoder {
    opus: Option<OpusDecoder>,
}

impl AudioDecoder {
    /// A decoder with no Opus state yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Decodes `chunk`, appending samples to `out`; returns how many.
    pub fn decode(&mut self, chunk: &AudioChunk, out: &mut Vec<f32>) -> Result<usize, CodecError> {
        match chunk.format {
            AudioFormat::Opus => {
                let mut dec = match self.opus.take() {
                    Some(dec) => dec,
                    None => OpusDecoder::new()?,
                };
                let result = dec.decode(&chunk.payload, out);
                self.opus = Some(dec);
                result
            }
            AudioFormat::PcmS16le => s16le_to_f32(&chunk.payload, out),
            AudioFormat::PcmF32le => f32le_to_f32(&chunk.payload, out),
        }
    }
}

/// Encodes 80 ms server frames into chunks of one fixed format.
#[derive(Debug)]
pub struct AudioEncoder {
    kind: EncoderKind,
}

#[derive(Debug)]
enum EncoderKind {
    Opus(Box<OpusEncoder>),
    PcmS16le,
    PcmF32le,
}

impl AudioEncoder {
    /// An encoder for `format`; `opus_bitrate` is used only for Opus.
    pub fn new(format: AudioFormat, opus_bitrate: i32) -> Result<Self, CodecError> {
        let kind = match format {
            AudioFormat::Opus => EncoderKind::Opus(Box::new(OpusEncoder::new(opus_bitrate)?)),
            AudioFormat::PcmS16le => EncoderKind::PcmS16le,
            AudioFormat::PcmF32le => EncoderKind::PcmF32le,
        };
        Ok(Self { kind })
    }

    /// The output format.
    pub fn format(&self) -> AudioFormat {
        match self.kind {
            EncoderKind::Opus(_) => AudioFormat::Opus,
            EncoderKind::PcmS16le => AudioFormat::PcmS16le,
            EncoderKind::PcmF32le => AudioFormat::PcmF32le,
        }
    }

    /// Encodes one frame of [`FRAME_SAMPLES`] samples into one chunk.
    pub fn encode_frame(&mut self, frame: &[f32]) -> Result<AudioChunk, CodecError> {
        if frame.len() != FRAME_SAMPLES {
            return Err(CodecError::BadFrameLength(frame.len()));
        }
        let format = self.format();
        let payload = match &mut self.kind {
            EncoderKind::Opus(enc) => enc.encode(frame)?,
            EncoderKind::PcmS16le => {
                let mut v = Vec::new();
                f32_to_s16le(frame, &mut v);
                v
            }
            EncoderKind::PcmF32le => {
                let mut v = Vec::new();
                f32_to_f32le(frame, &mut v);
                v
            }
        };
        Ok(AudioChunk { format, payload })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, amp: f32, start: usize, len: usize) -> Vec<f32> {
        (start..start + len)
            .map(|i| {
                amp * (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE as f32).sin()
            })
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn pcm_s16_round_trip() {
        let pcm = vec![0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0];
        let mut bytes = Vec::new();
        f32_to_s16le(&pcm, &mut bytes);
        assert_eq!(bytes.len(), 14);
        let mut back = Vec::new();
        assert_eq!(s16le_to_f32(&bytes, &mut back).unwrap(), 7);
        let expect = [0.0, 0.5, -0.5, 1.0, -1.0, 1.0, -1.0];
        for (a, b) in back.iter().zip(expect) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        assert!(s16le_to_f32(&[0, 0, 0], &mut back).is_err());
    }

    #[test]
    fn pcm_f32_round_trip_and_sanitizing() {
        let pcm = vec![0.25, -0.75, f32::NAN, 3.0];
        let mut bytes = Vec::new();
        f32_to_f32le(&pcm, &mut bytes);
        let mut back = Vec::new();
        f32le_to_f32(&bytes, &mut back).unwrap();
        assert_eq!(back, vec![0.25, -0.75, 0.0, 1.0]);
        assert!(f32le_to_f32(&[0; 5], &mut back).is_err());
    }

    #[test]
    fn opus_round_trip_preserves_sine() {
        let mut enc = AudioEncoder::new(AudioFormat::Opus, 48_000).unwrap();
        let mut dec = AudioDecoder::new();
        let frames = 25; // 2 s
        let input = sine(440.0, 0.5, 0, frames * FRAME_SAMPLES);
        let mut output = Vec::new();
        for f in input.chunks(FRAME_SAMPLES) {
            let chunk = enc.encode_frame(f).unwrap();
            assert_eq!(chunk.format, AudioFormat::Opus);
            assert!(!chunk.payload.is_empty() && chunk.payload.len() < OPUS_MAX_PACKET_BYTES);
            // Through the wire framing as well.
            let wire = AudioChunk::from_bytes(&chunk.to_bytes()).unwrap();
            let n = dec.decode(&wire, &mut output).unwrap();
            assert_eq!(n, FRAME_SAMPLES, "an 80 ms packet decodes to one frame");
        }
        assert_eq!(output.len(), input.len());
        // Skip the first half second (codec warm-up and lookahead), then
        // find the codec delay by cross-correlation and compare.
        let skip = SAMPLE_RATE as usize / 2;
        let a = &input[skip..skip + 9600];
        let best_lag = (0..200)
            .max_by(|&x, &y| {
                let cx: f32 = a.iter().zip(&output[skip + x..]).map(|(p, q)| p * q).sum();
                let cy: f32 = a.iter().zip(&output[skip + y..]).map(|(p, q)| p * q).sum();
                cx.total_cmp(&cy)
            })
            .unwrap();
        let b = &output[skip + best_lag..skip + best_lag + 9600];
        let err: Vec<f32> = a.iter().zip(b).map(|(p, q)| p - q).collect();
        let snr_db = 20.0 * (rms(a) / rms(&err)).log10();
        assert!((rms(b) - rms(a)).abs() < 0.1 * rms(a), "level preserved");
        assert!(snr_db > 10.0, "SNR {snr_db:.1} dB at lag {best_lag}");
    }

    #[test]
    fn opus_decodes_other_packet_durations() {
        let mut enc = OpusEncoder::new(DEFAULT_OPUS_BITRATE).unwrap();
        let mut dec = OpusDecoder::new().unwrap();
        let mut out = Vec::new();
        for len in [480, 960, 1920, 2880] {
            let packet = enc.encode(&sine(300.0, 0.3, 0, len)).unwrap();
            out.clear();
            assert_eq!(dec.decode(&packet, &mut out).unwrap(), len);
            assert_eq!(out.len(), len);
        }
        assert!(matches!(
            enc.encode(&[0.0; 1000]),
            Err(CodecError::BadFrameLength(1000))
        ));
    }

    #[test]
    fn pcm_encoder_frame_sizes() {
        let frame = sine(440.0, 0.5, 0, FRAME_SAMPLES);
        let mut s16 = AudioEncoder::new(AudioFormat::PcmS16le, 0).unwrap();
        let mut f32e = AudioEncoder::new(AudioFormat::PcmF32le, 0).unwrap();
        assert_eq!(s16.encode_frame(&frame).unwrap().payload.len(), 2 * 1920);
        let chunk = f32e.encode_frame(&frame).unwrap();
        assert_eq!(chunk.payload.len(), 4 * 1920);
        let mut back = Vec::new();
        AudioDecoder::new().decode(&chunk, &mut back).unwrap();
        assert_eq!(back, frame);
        assert!(s16.encode_frame(&frame[..100]).is_err());
    }
}
