//! The Mimi codec: loading, offline and streaming encode/decode.

use std::path::Path;

use s2s_kernels::reference::conv1d::{CausalConv1dState, PadMode};
use s2s_kernels::reference::conv_transpose1d::ConvTranspose1dState;

use crate::conv::{Conv1d, ConvSpec, ConvTr1d};
use crate::quantizer::SplitRvq;
use crate::seanet::{SeaNetDecoder, SeaNetDecoderState, SeaNetEncoder, SeaNetEncoderState};
use crate::transformer::{Transformer, TransformerState};
use crate::weights::{Loader, SafetensorsCheckpoint, TensorSource};
use crate::{Codes, MimiConfig, MimiError, Result, Signal};

/// What loading did not use.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadReport {
    /// Checkpoint tensors the model never read (e.g. `_initialized` flags,
    /// codebooks beyond `quantizer_n_q`).
    pub unused: Vec<String>,
}

/// Every intermediate of [`Mimi::encode_stages`].
#[derive(Clone, Debug, PartialEq)]
pub struct EncodeStages {
    /// SEANet encoder output `[dimension, n / hop]`.
    pub seanet: Signal,
    /// Encoder transformer output `[dimension, n / hop]`.
    pub transformer: Signal,
    /// Downsampled latent before quantization `[dimension, frames]`.
    pub latent: Signal,
    /// Codes `[n_q, frames]`.
    pub codes: Codes,
}

/// Every intermediate of [`Mimi::decode_stages`].
#[derive(Clone, Debug, PartialEq)]
pub struct DecodeStages {
    /// Dequantized latent `[dimension, frames]`.
    pub quantized: Signal,
    /// After the learnt upsampling `[dimension, frames * stride]`.
    pub upsampled: Signal,
    /// Decoder transformer output `[dimension, frames * stride]`.
    pub transformer: Signal,
    /// Decoded audio `[channels, frames * frame_size]`.
    pub pcm: Signal,
}

/// A loaded Mimi codec. Immutable: per-session streaming state lives in
/// [`EncoderState`] / [`DecoderState`], so one model can serve many streams.
#[derive(Clone, Debug)]
pub struct Mimi {
    config: MimiConfig,
    encoder: SeaNetEncoder,
    decoder: SeaNetDecoder,
    encoder_transformer: Transformer,
    decoder_transformer: Transformer,
    downsample: Conv1d,
    upsample: ConvTr1d,
    quantizer: SplitRvq,
}

/// Streaming encoder state for one stream.
#[derive(Clone, Debug)]
pub struct EncoderState {
    seanet: SeaNetEncoderState,
    transformer: TransformerState,
    downsample: CausalConv1dState<f32>,
}

/// Streaming decoder state for one stream.
#[derive(Clone, Debug)]
pub struct DecoderState {
    upsample: ConvTranspose1dState<f32>,
    transformer: TransformerState,
    seanet: SeaNetDecoderState,
}

impl Mimi {
    /// Load a Kyutai-format `.safetensors` checkpoint (e.g.
    /// `tokenizer-e351c8d8-checkpoint125.safetensors`).
    pub fn load(path: impl AsRef<Path>, config: MimiConfig) -> Result<Self> {
        let ck = SafetensorsCheckpoint::open(path)?;
        Self::from_source(&ck, config)
    }

    /// Build from any tensor source using Kyutai's parameter names.
    pub fn from_source(src: &dyn TensorSource, config: MimiConfig) -> Result<Self> {
        Ok(Self::from_source_with_report(src, config)?.0)
    }

    /// [`Self::from_source`], also reporting unused tensors.
    pub fn from_source_with_report(
        src: &dyn TensorSource,
        config: MimiConfig,
    ) -> Result<(Self, LoadReport)> {
        config.validate()?;
        let l = Loader::new(src);
        let dim = config.dimension;
        let stride = config.downsample_stride();
        let model = Self {
            encoder: SeaNetEncoder::load(&l, "encoder", &config)?,
            decoder: SeaNetDecoder::load(&l, "decoder", &config)?,
            encoder_transformer: Transformer::load(&l, "encoder_transformer", &config)?,
            decoder_transformer: Transformer::load(&l, "decoder_transformer", &config)?,
            downsample: Conv1d::load(
                &l,
                "downsample.conv.conv.conv",
                ConvSpec {
                    c_in: dim,
                    c_out: dim,
                    kernel: 2 * stride,
                    stride,
                    dilation: 1,
                    bias: false,
                    pad: PadMode::Replicate,
                },
            )?,
            // Depthwise (groups = dim), no bias.
            upsample: ConvTr1d::load(
                &l,
                "upsample.convtr.convtr.convtr",
                dim,
                dim,
                2 * stride,
                stride,
                dim,
                false,
            )?,
            quantizer: SplitRvq::load(&l, "quantizer", &config)?,
            config,
        };
        let report = LoadReport { unused: l.unused() };
        Ok((model, report))
    }

    /// The configuration.
    pub fn config(&self) -> &MimiConfig {
        &self.config
    }

    fn pcm_signal(&self, pcm: &[f32]) -> Result<Signal> {
        let c = self.config.channels;
        if !pcm.len().is_multiple_of(c) {
            return Err(MimiError::Input(format!(
                "{} samples is not a multiple of {c} channels",
                pcm.len()
            )));
        }
        Ok(Signal::new(c, pcm.len() / c, pcm.to_vec()))
    }

    /// Offline encode with every intermediate. `pcm` is channel-major
    /// `[channels, n]` (plain samples for mono). A trailing partial frame is
    /// completed with Kyutai's causal extra padding.
    pub fn encode_stages(&self, pcm: &[f32]) -> Result<EncodeStages> {
        let x = self.pcm_signal(pcm)?;
        let seanet = self.encoder.forward(&x);
        let transformer = self.encoder_transformer.forward(&seanet);
        let latent = self.downsample.forward(&transformer);
        let codes = self.quantizer.encode(&latent);
        Ok(EncodeStages {
            seanet,
            transformer,
            latent,
            codes,
        })
    }

    /// Offline encode: `pcm` (`[channels, n]`) -> codes `[n_q, ceil(n / frame_size)]`.
    pub fn encode(&self, pcm: &[f32]) -> Result<Codes> {
        Ok(self.encode_stages(pcm)?.codes)
    }

    /// Offline encode up to the latent before quantization `[dimension, frames]`.
    pub fn encode_latent(&self, pcm: &[f32]) -> Result<Signal> {
        Ok(self.encode_stages(pcm)?.latent)
    }

    /// Offline decode with every intermediate. `codes` may hold fewer
    /// codebooks than `quantizer_n_q` (lower bitrate).
    pub fn decode_stages(&self, codes: &Codes) -> Result<DecodeStages> {
        let quantized = self.quantizer.decode(codes)?;
        let upsampled = self.upsample.forward(&quantized);
        let transformer = self.decoder_transformer.forward(&upsampled);
        let pcm = self.decoder.forward(&transformer);
        Ok(DecodeStages {
            quantized,
            upsampled,
            transformer,
            pcm,
        })
    }

    /// Offline decode: codes `[k, frames]` -> pcm `[channels, frames * frame_size]`.
    pub fn decode(&self, codes: &Codes) -> Result<Vec<f32>> {
        Ok(self.decode_stages(codes)?.pcm.data)
    }

    /// Fresh streaming encoder state.
    pub fn encoder_state(&self) -> EncoderState {
        EncoderState {
            seanet: self.encoder.state(),
            transformer: self.encoder_transformer.state(),
            downsample: CausalConv1dState::new(),
        }
    }

    /// Fresh streaming decoder state.
    pub fn decoder_state(&self) -> DecoderState {
        DecoderState {
            upsample: ConvTranspose1dState::new(),
            transformer: self.decoder_transformer.state(),
            seanet: self.decoder.state(),
        }
    }

    /// Feed the next `pcm` samples (`[channels, n]`, any `n`) of a stream.
    /// Returns the code frames that became complete (exactly one for each
    /// whole `frame_size` of input). Over whole frames the result equals
    /// [`Self::encode`] on the concatenated input, bit for bit.
    pub fn encode_step(&self, st: &mut EncoderState, pcm: &[f32]) -> Result<Codes> {
        let x = self.pcm_signal(pcm)?;
        let h = self.encoder.step(&mut st.seanet, &x);
        let h = self.encoder_transformer.step(&mut st.transformer, &h);
        let latent = self.downsample.step(&mut st.downsample, &h);
        Ok(self.quantizer.encode(&latent))
    }

    /// Feed the next code frames of a stream; returns `frames * frame_size`
    /// samples per channel (`[channels, n]`). Equals [`Self::decode`] on the
    /// concatenated codes, bit for bit.
    pub fn decode_step(&self, st: &mut DecoderState, codes: &Codes) -> Result<Vec<f32>> {
        let q = self.quantizer.decode(codes)?;
        let h = self.upsample.step(&mut st.upsample, &q);
        let h = self.decoder_transformer.step(&mut st.transformer, &h);
        Ok(self.decoder.step(&mut st.seanet, &h).data)
    }

    /// A streaming encoder bound to this model.
    pub fn streaming_encoder(&self) -> StreamingEncoder<'_> {
        StreamingEncoder {
            mimi: self,
            state: self.encoder_state(),
        }
    }

    /// A streaming decoder bound to this model.
    pub fn streaming_decoder(&self) -> StreamingDecoder<'_> {
        StreamingDecoder {
            mimi: self,
            state: self.decoder_state(),
        }
    }
}

/// Per-session streaming encoder: PCM in, code frames out.
#[derive(Clone, Debug)]
pub struct StreamingEncoder<'a> {
    mimi: &'a Mimi,
    state: EncoderState,
}

impl StreamingEncoder<'_> {
    /// Push exactly one frame (`frame_size` samples per channel); returns its
    /// `n_q` codes.
    pub fn push_frame(&mut self, frame: &[f32]) -> Result<Vec<u32>> {
        let n = self.mimi.config.frame_size() * self.mimi.config.channels;
        if frame.len() != n {
            return Err(MimiError::Input(format!(
                "push_frame: expected {n} samples, got {}",
                frame.len()
            )));
        }
        let codes = self.mimi.encode_step(&mut self.state, frame)?;
        debug_assert_eq!(codes.frames(), 1);
        Ok(codes.frame(0))
    }

    /// Push any number of samples; returns the frames that became complete.
    pub fn push(&mut self, pcm: &[f32]) -> Result<Codes> {
        self.mimi.encode_step(&mut self.state, pcm)
    }

    /// Start a new stream.
    pub fn reset(&mut self) {
        self.state = self.mimi.encoder_state();
    }
}

/// Per-session streaming decoder: code frames in, PCM out.
#[derive(Clone, Debug)]
pub struct StreamingDecoder<'a> {
    mimi: &'a Mimi,
    state: DecoderState,
}

impl StreamingDecoder<'_> {
    /// Push one frame of codes (`1..=n_q` values); returns `frame_size`
    /// samples per channel.
    pub fn push_frame(&mut self, codes: &[u32]) -> Result<Vec<f32>> {
        let c = Codes::new(codes.len(), 1, codes.to_vec())?;
        self.mimi.decode_step(&mut self.state, &c)
    }

    /// Push several frames.
    pub fn push(&mut self, codes: &Codes) -> Result<Vec<f32>> {
        self.mimi.decode_step(&mut self.state, codes)
    }

    /// Start a new stream.
    pub fn reset(&mut self) {
        self.state = self.mimi.decoder_state();
    }
}
