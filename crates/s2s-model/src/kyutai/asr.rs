//! Delayed-Streams-Modeling speech-to-text driver, mirroring `moshi::asr::State`
//! for one stream (batch 1), fed Mimi audio codes instead of PCM.
//!
//! Per frame:
//! 1. The audio input of codebook `c` is the previous frame's code (the audio
//!    padding token on the first step): the model sees audio one frame late.
//! 2. The text input is the previously sampled text token (the start token on
//!    the first step).
//! 3. One temporal-transformer step (plus the optional condition) gives text
//!    logits and extra heads; the text token is decoded greedily
//!    (temperature 0) or sampled.
//! 4. After `asr_delay_in_tokens` steps, tokens are grouped into words: a
//!    padding token (3) or end-of-padding token (0) closes the current word
//!    ([`AsrMsg::Word`]); an end-of-padding token also marks the end time of
//!    the last word ([`AsrMsg::EndWord`]). Times are in seconds of audio
//!    (frames / 12.5, shifted back by the delay).
//!
//! The text tokens are SentencePiece ids; detokenization is up to the caller.

use serde::{Deserialize, Serialize};

use super::conditioner::Condition;
use super::config::AsrConfig;
use super::error::{Error, Result};
use super::lm::LmModel;
use super::sampling::Sampler;

use s2s_kernels::reference::loss::softmax_fwd;

/// Messages emitted by [`AsrState::step_tokens`] (same shape and serde
/// encoding as Kyutai's `AsrMsg`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AsrMsg {
    /// Extra-head probabilities for this step: `prs[head]` holds the
    /// probability of class 0 for each batch item (one item here). Only sent
    /// when the model has extra heads.
    Step {
        /// Model steps so far (1-based).
        step_idx: usize,
        /// Per head, per batch item.
        prs: Vec<Vec<f32>>,
    },
    /// A completed word.
    Word {
        /// SentencePiece token ids.
        tokens: Vec<u32>,
        /// Start time in seconds.
        start_time: f64,
        /// Always 0 (batch 1).
        batch_idx: usize,
    },
    /// End time of the last word.
    EndWord {
        /// Stop time in seconds.
        stop_time: f64,
        /// Always 0 (batch 1).
        batch_idx: usize,
    },
}

/// Kyutai's literal text ids in `asr.rs`: padding and end-of-padding.
const TEXT_PAD: u32 = 3;
const TEXT_EOP: u32 = 0;
/// Mimi frame rate used for word times.
const FRAME_RATE: f64 = 12.5;

/// Streaming STT state.
#[derive(Debug)]
pub struct AsrState {
    lm: LmModel,
    asr_delay_in_tokens: usize,
    sampler: Sampler,
    condition: Option<Condition>,
    model_step_idx: usize,
    // ItemState
    step_idx: usize,
    text_token: u32,
    word_tokens: Vec<u32>,
    unended_word: bool,
    last_stop_time: f64,
    next_codebooks: Vec<u32>,
    last_text_logits: Vec<f32>,
    last_extra_heads: Vec<Vec<f32>>,
}

impl AsrState {
    /// New state. `temperature <= 0` decodes greedily like Kyutai
    /// (`Tensor::argmax`, first maximum on ties); otherwise tokens are sampled
    /// with a seeded RNG (see [`Sampler`]). `condition` is added to every
    /// input embedding.
    pub fn new(
        lm: LmModel,
        asr_delay_in_tokens: usize,
        temperature: f64,
        condition: Option<Condition>,
    ) -> Self {
        let sampler = if temperature <= 0.0 {
            Sampler::greedy_first()
        } else {
            Sampler::new(
                s2s_kernels::reference::sampling::SamplingParams {
                    temperature,
                    top_k: None,
                    top_p: None,
                },
                0,
            )
        };
        let pad = lm.audio_pad_token();
        let n = lm.in_audio_codebooks();
        let mut s = Self {
            lm,
            asr_delay_in_tokens,
            sampler,
            condition,
            model_step_idx: 0,
            step_idx: 0,
            text_token: 0,
            word_tokens: vec![],
            unended_word: false,
            last_stop_time: 0.0,
            next_codebooks: vec![pad; n],
            last_text_logits: vec![],
            last_extra_heads: vec![],
        };
        s.reset();
        s
    }

    /// New state from `moshi-server` ASR settings. The condition follows
    /// `BatchedAsr`: `conditioning_delay = x` gives
    /// `condition_cont("delay", -x)`, `conditioning_learnt_padding` gives the
    /// `delay` conditioner's learnt padding; a model with conditioners needs
    /// exactly one of the two.
    pub fn from_config(lm: LmModel, cfg: &AsrConfig) -> Result<Self> {
        let condition = match lm.condition_provider() {
            None => None,
            Some(cp) => match (cfg.conditioning_delay, cfg.conditioning_learnt_padding) {
                (Some(d), false) => Some(cp.condition_cont("delay", -d)?),
                (None, true) => Some(cp.learnt_padding("delay")?),
                (Some(_), true) => {
                    return Err(Error::Config(
                        "conditioning_delay/conditioning_learnt_padding cannot be both set".into(),
                    ))
                }
                (None, false) => {
                    return Err(Error::Config(
                        "conditioning_delay/conditioning_learnt_padding is required".into(),
                    ))
                }
            },
        };
        Ok(Self::new(
            lm,
            cfg.asr_delay_in_tokens,
            cfg.temperature.unwrap_or(0.0),
            condition,
        ))
    }

    /// Reset the model and the stream state.
    pub fn reset(&mut self) {
        self.lm.reset_state();
        self.model_step_idx = 0;
        self.step_idx = 0;
        self.text_token = 0;
        self.word_tokens.clear();
        self.unended_word = false;
        self.last_stop_time = 0.0;
        self.next_codebooks.fill(self.lm.audio_pad_token());
    }

    /// Model steps taken.
    pub fn model_step_idx(&self) -> usize {
        self.model_step_idx
    }

    /// The transcript delay in frames.
    pub fn asr_delay_in_tokens(&self) -> usize {
        self.asr_delay_in_tokens
    }

    /// The last decoded text token.
    pub fn text_token(&self) -> u32 {
        self.text_token
    }

    /// Text logits of the last step.
    pub fn last_text_logits(&self) -> &[f32] {
        &self.last_text_logits
    }

    /// Full softmax of each extra head at the last step.
    pub fn last_extra_heads(&self) -> &[Vec<f32>] {
        &self.last_extra_heads
    }

    /// The model.
    pub fn model(&self) -> &LmModel {
        &self.lm
    }

    /// Feed one frame of Mimi codes (one per codebook the model reads).
    pub fn step_tokens(&mut self, codes: &[u32]) -> Result<Vec<AsrMsg>> {
        let n = self.lm.in_audio_codebooks();
        if codes.len() != n {
            return Err(Error::Input(format!(
                "{} codes for {n} codebooks",
                codes.len()
            )));
        }
        let first = self.step_idx == 0;
        let pad = self.lm.audio_pad_token();
        let audio: Vec<Option<u32>> = (0..n)
            .map(|c| {
                let prev = std::mem::replace(&mut self.next_codebooks[c], codes[c]);
                Some(if first { pad } else { prev })
            })
            .collect();
        let text = if first {
            self.lm.text_start_token()
        } else {
            self.text_token
        };
        let out = self.lm.step(Some(text), &audio, self.condition.as_ref())?;
        self.model_step_idx += 1;
        let mut msgs = vec![];
        self.last_extra_heads = self
            .lm
            .extra_heads(&out.hidden)
            .into_iter()
            .map(|h| {
                let mut p = vec![0.0f32; h.len()];
                softmax_fwd(&h, &mut p, 1, h.len());
                p
            })
            .collect();
        if !self.last_extra_heads.is_empty() {
            msgs.push(AsrMsg::Step {
                step_idx: self.model_step_idx,
                prs: self.last_extra_heads.iter().map(|p| vec![p[0]]).collect(),
            });
        }
        let token = self.sampler.sample(&out.text_logits)?;
        self.last_text_logits = out.text_logits;

        self.text_token = token;
        self.step_idx += 1;
        if self.step_idx >= self.asr_delay_in_tokens {
            if token == TEXT_PAD || token == TEXT_EOP {
                if !self.word_tokens.is_empty() {
                    msgs.push(AsrMsg::Word {
                        tokens: std::mem::take(&mut self.word_tokens),
                        start_time: self.last_stop_time,
                        batch_idx: 0,
                    });
                    self.unended_word = true;
                }
            } else {
                self.word_tokens.push(token);
            }
            if token == TEXT_EOP {
                let stop_time = (self.step_idx - self.asr_delay_in_tokens) as f64 / FRAME_RATE;
                if self.unended_word {
                    self.unended_word = false;
                    msgs.push(AsrMsg::EndWord {
                        stop_time,
                        batch_idx: 0,
                    });
                }
                self.last_stop_time = stop_time;
            }
        }
        Ok(msgs)
    }
}
