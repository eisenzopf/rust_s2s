//! Moshi-style full-duplex generation driver, mirroring
//! `moshi::lm_generate_multistream::State` (without classifier-free guidance,
//! repetition penalty or pad multiplier, which only matter for random
//! sampling).
//!
//! Per step, with `G` generated and `I` input codebooks and acoustic delay `τ`:
//!
//! 1. The user's audio codes for this frame are stored in the token grid.
//! 2. The model input for codebook `c` is the grid value of frame `step - 1`
//!    for the semantic codebooks (`c == 0` and `c == G`), and of frame
//!    `step - τ - 1` for the acoustic ones; the audio padding token before
//!    that (i.e. inputs are delayed by one frame plus the codebook's delay).
//! 3. The temporal transformer gives text logits; the text token is sampled
//!    (or forced).
//! 4. The depformer generates `G` audio tokens conditioned on the text token;
//!    for the first `τ` steps, acoustic slices are fed the padding token
//!    ([`ForcedAudioTokens`]).
//! 5. Generated tokens are written to the grid at frame `step - delay(c)`
//!    ("undelayed" time).

use super::conditioner::Condition;
use super::config::MultistreamConfig;
use super::error::{Error, Result};
use super::lm::{DepformerOutput, LmModel};
use super::sampling::Sampler;

/// Grid value of a token that has not been generated yet.
pub const UNGENERATED: u32 = u32::MAX;

/// Teacher-forced depformer inputs during the first `acoustic_delay` steps
/// (`moshi::lm::ForcedAudioTokens`): for each stream, the semantic slice is
/// free and the acoustic slices are fed the padding token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForcedAudioTokens {
    acoustic_delay: usize,
    pre_delay_tokens: Vec<Option<u32>>,
}

impl ForcedAudioTokens {
    /// `stream_codebooks`: codebooks per audio stream, in depformer order.
    pub fn new(acoustic_delay: usize, audio_pad_token: u32, stream_codebooks: &[usize]) -> Self {
        let pre_delay_tokens = stream_codebooks
            .iter()
            .flat_map(|&n| (0..n).map(move |c| if c == 0 { None } else { Some(audio_pad_token) }))
            .collect();
        Self {
            acoustic_delay,
            pre_delay_tokens,
        }
    }

    /// Forced tokens at `step_idx` (empty once past the delay).
    pub fn forced_tokens(&self, step_idx: usize) -> &[Option<u32>] {
        if step_idx < self.acoustic_delay {
            &self.pre_delay_tokens
        } else {
            &[]
        }
    }
}

/// Everything one [`MultistreamState::step`] computed.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameOutput {
    /// Text token (sampled or forced) for this frame.
    pub text_token: u32,
    /// Text logits.
    pub text_logits: Vec<f32>,
    /// Depformer tokens and logits (`None` without depformer).
    pub depformer: Option<DepformerOutput>,
    /// Text token fed to the model.
    pub text_input: Option<u32>,
    /// Audio tokens fed to the model, per codebook.
    pub audio_input: Vec<u32>,
}

/// Frame-by-frame generation state.
#[derive(Debug)]
pub struct MultistreamState {
    model: LmModel,
    audio_tokens: Vec<Vec<u32>>,
    text_tokens: Vec<u32>,
    audio_sampler: Sampler,
    text_sampler: Sampler,
    step_idx: usize,
    forced_audio_tokens: ForcedAudioTokens,
    config: MultistreamConfig,
}

impl MultistreamState {
    /// New state for up to `max_step_idx` steps.
    ///
    /// The forced-token layout is Kyutai's hard-coded two streams of 8
    /// codebooks (`&[8, 8]` in `State::new`); use
    /// [`MultistreamState::with_forced_stream_codebooks`] to change it.
    pub fn new(
        model: LmModel,
        max_step_idx: usize,
        audio_sampler: Sampler,
        text_sampler: Sampler,
        config: MultistreamConfig,
    ) -> Self {
        let rows = max_step_idx + config.acoustic_delay;
        Self {
            model,
            audio_tokens: vec![vec![UNGENERATED; config.total_audio_codebooks()]; rows],
            text_tokens: vec![UNGENERATED; rows],
            audio_sampler,
            text_sampler,
            step_idx: 0,
            forced_audio_tokens: ForcedAudioTokens::new(
                config.acoustic_delay,
                config.audio_pad_token(),
                &[8, 8],
            ),
            config,
        }
    }

    /// Override the per-stream codebook counts used for forced tokens.
    pub fn with_forced_stream_codebooks(mut self, stream_codebooks: &[usize]) -> Self {
        self.forced_audio_tokens = ForcedAudioTokens::new(
            self.config.acoustic_delay,
            self.config.audio_pad_token(),
            stream_codebooks,
        );
        self
    }

    /// Steps taken.
    pub fn step_idx(&self) -> usize {
        self.step_idx
    }

    /// The driver settings.
    pub fn config(&self) -> &MultistreamConfig {
        &self.config
    }

    /// The model.
    pub fn model(&self) -> &LmModel {
        &self.model
    }

    /// Give the model back.
    pub fn into_model(self) -> LmModel {
        self.model
    }

    /// One frame (`State::step_`): `text_token` is the text input (the
    /// previous text token, or the start token on the first step),
    /// `input_audio_tokens` the user's codes for this frame (one per input
    /// codebook), `force_text_token` overrides sampling.
    pub fn step(
        &mut self,
        text_token: Option<u32>,
        input_audio_tokens: &[u32],
        force_text_token: Option<u32>,
        condition: Option<&Condition>,
    ) -> Result<FrameOutput> {
        let cfg = &self.config;
        let (gen, total, delay) = (
            cfg.generated_audio_codebooks,
            cfg.total_audio_codebooks(),
            cfg.acoustic_delay,
        );
        let pad = cfg.audio_pad_token();
        if self.step_idx >= self.audio_tokens.len() {
            return Err(Error::Input("max step-idx reached".into()));
        }
        if gen + input_audio_tokens.len() > total {
            return Err(Error::Input(format!(
                "{} input audio tokens, expected at most {}",
                input_audio_tokens.len(),
                cfg.input_audio_codebooks
            )));
        }
        for (c, &t) in input_audio_tokens.iter().enumerate() {
            self.audio_tokens[self.step_idx][c + gen] = t;
        }
        let mut codes = Vec::with_capacity(total);
        for codebook in 0..total {
            let t = if codebook == 0 || codebook == gen {
                if self.step_idx == 0 {
                    pad
                } else {
                    self.audio_tokens[self.step_idx - 1][codebook]
                }
            } else if self.step_idx <= delay {
                pad
            } else {
                self.audio_tokens[self.step_idx - delay - 1][codebook]
            };
            if t == UNGENERATED {
                return Err(Error::Input(format!(
                    "internal error, ungenerated {} {codebook}",
                    self.step_idx
                )));
            }
            codes.push(t);
        }
        let audio: Vec<Option<u32>> = codes.iter().map(|&c| Some(c)).collect();
        let out = self.model.step(text_token, &audio, condition)?;
        let sampled = match force_text_token {
            Some(t) => t,
            None => self.text_sampler.sample(&out.text_logits)?,
        };
        self.text_tokens[self.step_idx] = sampled;
        let dep = self.model.depformer_sample(
            &out.hidden,
            Some(sampled),
            self.forced_audio_tokens.forced_tokens(self.step_idx),
            &mut self.audio_sampler,
        )?;
        for c in 0..gen {
            let d = if c == 0 || c == gen { 0 } else { delay };
            // Overwrites earlier rows for the first steps (saturating_sub),
            // exactly like Kyutai.
            self.audio_tokens[self.step_idx.saturating_sub(d)][c] =
                dep.as_ref().map_or(pad, |o| o.tokens[c]);
        }
        self.step_idx += 1;
        let result = FrameOutput {
            text_token: sampled,
            text_logits: out.text_logits,
            depformer: dep,
            text_input: text_token,
            audio_input: codes,
        };
        if self.step_idx >= self.audio_tokens.len() {
            return Err(Error::Input("max step-idx reached".into()));
        }
        Ok(result)
    }

    /// Audio token grid (`[frame][codebook]`, generated then input), in
    /// undelayed time. Only the rows of steps taken unless `include_all`.
    pub fn audio_tokens(&self, include_all: bool) -> &[Vec<u32>] {
        if include_all {
            &self.audio_tokens
        } else {
            &self.audio_tokens[..self.step_idx.min(self.audio_tokens.len())]
        }
    }

    /// Text tokens per step.
    pub fn text_tokens(&self, include_all: bool) -> &[u32] {
        if include_all {
            &self.text_tokens
        } else {
            &self.text_tokens[..self.step_idx.min(self.text_tokens.len())]
        }
    }

    /// The most recent frame whose codebooks are all generated, or `None`
    /// during the first `acoustic_delay + 1` steps or if any code is a
    /// padding / out-of-vocabulary value (ready for the Mimi decoder).
    pub fn last_audio_tokens(&self) -> Option<Vec<u32>> {
        if self.step_idx <= self.config.acoustic_delay {
            return None;
        }
        let row = &self.audio_tokens[self.step_idx - self.config.acoustic_delay - 1];
        if row
            .iter()
            .any(|&v| v as usize >= self.config.audio_vocab_size - 1)
        {
            None
        } else {
            Some(row.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_tokens_layout() {
        let f = ForcedAudioTokens::new(2, 99, &[3, 2]);
        assert_eq!(
            f.forced_tokens(0),
            &[None, Some(99), Some(99), None, Some(99)]
        );
        assert_eq!(f.forced_tokens(1).len(), 5);
        assert!(f.forced_tokens(2).is_empty());
    }
}
