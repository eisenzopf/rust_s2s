//! [`KyutaiEngine`]: Kyutai's published checkpoints behind the server's
//! `Engine` trait (plan milestone M4).
//!
//! Two kinds of checkpoint, each through the `s2s-model` driver that matches
//! Kyutai's own:
//!
//! - **Moshi** (full duplex, [`KyutaiModel::Moshi`]): serves `s2s` sessions
//!   through [`MultistreamState`]. Per frame, the user's PCM is encoded to
//!   the input codebooks, the model samples one text token (its "inner
//!   monologue", sent as assistant text) and the depformer generates the
//!   assistant's codebooks; once a frame is complete in undelayed time it is
//!   decoded to PCM.
//! - **DSM speech-to-text** (`stt-*` checkpoints, [`KyutaiModel::Stt`]):
//!   serves `stt` sessions through [`AsrState`]; every completed word is sent
//!   as user transcript text.
//!
//! These checkpoints have no system prompt, tool calling or text injection
//! (they predate our extended layout), so this adapter rejects sessions that
//! declare tools and TTS sessions, and ignores the system prompt, the voice
//! and injected text / tool results. Use [`crate::DuplexEngine`] with our own
//! model for those.
//!
//! The model's weights and its KV cache live in one [`LmModel`], so the
//! engine serves **one session at a time**: the model moves into the session
//! on `open_session` and back (with a cleared cache) on `close_session`.
//!
//! Text tokens are SentencePiece ids; they are decoded through
//! [`TokenizerInfo`] (for example `s2s_tokenizer::HfTokenizer` over a
//! `tokenizer.json` converted from Kyutai's `tokenizer_spm_*.model`).

use std::sync::Arc;

use s2s_core::streams::FRAME_SAMPLES;
use s2s_kernels::reference::sampling::SamplingParams;
use s2s_model::kyutai::{
    AsrConfig, AsrMsg, AsrState, LmModel, MultistreamConfig, MultistreamState, Sampler,
};
use s2s_server::protocol::TurnEvent;
use s2s_server::{Engine, EngineError, EngineEvent, FrameInput, FrameOutput, Mode, SessionConfig};
use s2s_tools::{TokenPiece, TokenizerInfo};

use crate::codec::FrameCodec;
use crate::engine::{TurnTracker, Utf8Buf, USER_SPEECH_RMS};

/// A loaded Kyutai checkpoint and the driver settings it runs with.
#[derive(Debug)]
pub enum KyutaiModel {
    /// A Moshi-style full-duplex model (with depformer).
    Moshi {
        /// The language model.
        lm: LmModel,
        /// Multistream driver settings (codebooks, delays, special text ids).
        driver: MultistreamConfig,
    },
    /// A Delayed-Streams-Modeling speech-to-text model.
    Stt {
        /// The language model.
        lm: LmModel,
        /// ASR driver settings (transcript delay, conditioning, temperature).
        asr: AsrConfig,
    },
}

/// Engine settings.
#[derive(Clone, Debug)]
pub struct KyutaiEngineConfig {
    /// Sampling for Moshi's text stream (temperature `<= 0` is greedy).
    pub text_sampling: SamplingParams,
    /// Sampling for Moshi's audio codebooks.
    pub audio_sampling: SamplingParams,
    /// Base seed; each session derives its own.
    pub seed: u64,
    /// Most frames per Moshi session (the driver's token grid is sized by
    /// it; the KV cache itself is a sliding window).
    pub max_frames: usize,
}

impl Default for KyutaiEngineConfig {
    /// Moshi's defaults (`moshi-server`): text 0.7 / top-k 25, audio 0.8 /
    /// top-k 250; one hour per session.
    fn default() -> Self {
        Self {
            text_sampling: SamplingParams {
                temperature: 0.7,
                top_k: Some(25),
                top_p: None,
            },
            audio_sampling: SamplingParams {
                temperature: 0.8,
                top_k: Some(250),
                top_p: None,
            },
            seed: 0,
            max_frames: 45_000,
        }
    }
}

type CodecFactory = Box<dyn Fn() -> Box<dyn FrameCodec> + Send>;

enum Kind {
    Moshi(MultistreamConfig),
    Stt(AsrConfig),
}

enum Driver {
    Moshi {
        state: Box<MultistreamState>,
        /// Text input of the next step (the start token, then the previous
        /// sampled token).
        prev_text: u32,
    },
    Stt(Box<AsrState>),
}

impl Driver {
    fn into_model(self) -> LmModel {
        let mut lm = match self {
            Driver::Moshi { state, .. } => state.into_model(),
            Driver::Stt(state) => state.into_model(),
        };
        lm.reset_state();
        lm
    }
}

struct Session {
    id: u64,
    driver: Driver,
    codec: Box<dyn FrameCodec>,
    text: Utf8Buf,
    user_turn: TurnTracker,
    asst_turn: TurnTracker,
}

/// A Kyutai Moshi or STT checkpoint as a frame-synchronous engine.
pub struct KyutaiEngine {
    kind: Kind,
    /// The model while no session holds it.
    idle: Option<LmModel>,
    tok: Arc<dyn TokenizerInfo>,
    new_codec: CodecFactory,
    cfg: KyutaiEngineConfig,
    session: Option<Session>,
}

impl std::fmt::Debug for KyutaiEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KyutaiEngine")
            .field(
                "kind",
                &match self.kind {
                    Kind::Moshi(_) => "moshi",
                    Kind::Stt(_) => "stt",
                },
            )
            .field("session", &self.session.as_ref().map(|s| s.id))
            .finish_non_exhaustive()
    }
}

impl KyutaiEngine {
    /// New engine. `tok` decodes the model's text tokens; `new_codec` makes
    /// one codec per session (Mimi for real checkpoints). Fails if the
    /// driver settings don't fit the model.
    pub fn new(
        model: KyutaiModel,
        tok: Arc<dyn TokenizerInfo>,
        new_codec: impl Fn() -> Box<dyn FrameCodec> + Send + 'static,
        cfg: KyutaiEngineConfig,
    ) -> s2s_model::kyutai::Result<Self> {
        use s2s_model::kyutai::Error;
        let (kind, lm) = match model {
            KyutaiModel::Moshi { lm, driver } => {
                if lm.generated_audio_codebooks() == 0 {
                    return Err(Error::Config(
                        "a Moshi model needs a depformer (this looks like an STT model)".into(),
                    ));
                }
                if driver.generated_audio_codebooks != lm.generated_audio_codebooks()
                    || driver.total_audio_codebooks() != lm.in_audio_codebooks()
                {
                    return Err(Error::Config(format!(
                        "driver expects {} generated + {} input codebooks; the model generates {} and reads {}",
                        driver.generated_audio_codebooks,
                        driver.input_audio_codebooks,
                        lm.generated_audio_codebooks(),
                        lm.in_audio_codebooks()
                    )));
                }
                if driver.audio_vocab_size != lm.config().audio_vocab_size {
                    return Err(Error::Config(format!(
                        "driver audio vocabulary {} differs from the model's {}",
                        driver.audio_vocab_size,
                        lm.config().audio_vocab_size
                    )));
                }
                if cfg.max_frames == 0 {
                    return Err(Error::Config("max_frames must be positive".into()));
                }
                (Kind::Moshi(driver), lm)
            }
            KyutaiModel::Stt { lm, asr } => {
                if lm.generated_audio_codebooks() != 0 {
                    return Err(Error::Config(
                        "an STT model has no depformer (this looks like a Moshi model)".into(),
                    ));
                }
                // Validates the conditioning settings against the model.
                let lm = AsrState::from_config(lm, &asr)?.into_model();
                (Kind::Stt(asr), lm)
            }
        };
        Ok(Self {
            kind,
            idle: Some(lm),
            tok,
            new_codec: Box::new(new_codec),
            cfg,
            session: None,
        })
    }

    /// The session currently served, if any.
    pub fn current_session(&self) -> Option<u64> {
        self.session.as_ref().map(|s| s.id)
    }

    fn session_seed(&self, id: u64) -> u64 {
        self.cfg.seed ^ id.wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    /// Bytes of an ordinary text token (`None` for special or out-of-range
    /// ids).
    fn token_bytes(&self, id: u32) -> Option<Vec<u8>> {
        if id as usize >= self.tok.vocab_size() {
            return None;
        }
        match self.tok.token(id) {
            TokenPiece::Bytes(b) => Some(b),
            TokenPiece::Special(_) => None,
        }
    }

    fn step_session(
        &self,
        s: &mut Session,
        input: &mut FrameInput,
        out: &mut FrameOutput,
    ) -> Result<(), EngineError> {
        let sid = s.id;
        let fail = move |e: s2s_model::kyutai::Error| {
            EngineError::StepFailed(format!("session {sid}: {e}"))
        };
        // Neither checkpoint kind can read injected text or tool results.
        input.injections.clear();
        let rms = (input.audio.iter().map(|x| x * x).sum::<f32>() / FRAME_SAMPLES as f32).sqrt();
        match s.user_turn.update(rms > USER_SPEECH_RMS) {
            Some(true) => out.events.push(EngineEvent::Turn(TurnEvent::UserStarted)),
            Some(false) => out.events.push(EngineEvent::Turn(TurnEvent::UserStopped)),
            None => {}
        }
        let codes = s.codec.encode(&input.audio[..]);
        match &mut s.driver {
            Driver::Moshi { state, prev_text } => {
                let frame = state.step(Some(*prev_text), &codes, None, None);
                let frame = frame.map_err(fail)?;
                *prev_text = frame.text_token;
                let cfg = state.config();
                let t = frame.text_token;
                let is_pad = t == cfg.text_pad_token || t == cfg.text_eop_token;
                if !is_pad && t != cfg.text_start_token {
                    if let Some(bytes) = self.token_bytes(t) {
                        let text = s.text.push(&bytes);
                        if !text.is_empty() {
                            out.events.push(EngineEvent::AssistantText(text));
                        }
                    }
                }
                match s.asst_turn.update(t != cfg.text_pad_token) {
                    Some(true) => out
                        .events
                        .push(EngineEvent::Turn(TurnEvent::AssistantStarted)),
                    Some(false) => out
                        .events
                        .push(EngineEvent::Turn(TurnEvent::AssistantStopped)),
                    None => {}
                }
                if let Some(row) = state.last_audio_tokens() {
                    let gen = cfg.generated_audio_codebooks;
                    let pcm = s.codec.decode(&row[..gen]);
                    let n = pcm.len().min(out.audio.len());
                    out.audio[..n].copy_from_slice(&pcm[..n]);
                }
            }
            Driver::Stt(state) => {
                for msg in state.step_tokens(&codes).map_err(fail)? {
                    if let AsrMsg::Word { tokens, .. } = msg {
                        let mut text = String::new();
                        for t in tokens {
                            if let Some(bytes) = self.token_bytes(t) {
                                text.push_str(&s.text.push(&bytes));
                            }
                        }
                        if !text.is_empty() {
                            out.events.push(EngineEvent::UserText(text));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl Engine for KyutaiEngine {
    fn open_session(&mut self, id: u64, config: &SessionConfig) -> Result<(), EngineError> {
        if self.session.is_some() {
            return Err(EngineError::Rejected(
                "the Kyutai engine serves one session at a time".into(),
            ));
        }
        if !config.tools.is_empty() {
            return Err(EngineError::Rejected(
                "Kyutai checkpoints have no tool calling; serve tools with our duplex model".into(),
            ));
        }
        match (&self.kind, config.mode) {
            (Kind::Moshi(_), Mode::S2s) | (Kind::Stt(_), Mode::Stt) => {}
            (Kind::Moshi(_), m) => {
                return Err(EngineError::Rejected(format!(
                    "a Moshi checkpoint serves s2s sessions, not {m:?}"
                )))
            }
            (Kind::Stt(_), m) => {
                return Err(EngineError::Rejected(format!(
                    "a speech-to-text checkpoint serves stt sessions, not {m:?}"
                )))
            }
        }
        let lm = self.idle.as_ref().expect("model is idle without a session");
        let codec = (self.new_codec)();
        let needs = match &self.kind {
            Kind::Moshi(d) => {
                if d.generated_audio_codebooks != d.input_audio_codebooks {
                    return Err(EngineError::Rejected(format!(
                        "one codec can't serve {} input and {} generated codebooks",
                        d.input_audio_codebooks, d.generated_audio_codebooks
                    )));
                }
                d.input_audio_codebooks
            }
            Kind::Stt(_) => lm.in_audio_codebooks(),
        };
        if codec.codebooks() != needs {
            return Err(EngineError::Rejected(format!(
                "codec has {} codebooks; the model needs {needs}",
                codec.codebooks()
            )));
        }
        let audio_vocab = lm.config().audio_vocab_size;
        if codec.codebook_size() as usize + 1 != audio_vocab {
            return Err(EngineError::Rejected(format!(
                "codec has {} entries per codebook; the model's audio vocabulary is {audio_vocab} (codes plus padding)",
                codec.codebook_size()
            )));
        }

        let seed = self.session_seed(id);
        let lm = self.idle.take().expect("checked above");
        let driver = match &self.kind {
            Kind::Moshi(d) => {
                let text_start = d.text_start_token;
                let state = MultistreamState::new(
                    lm,
                    self.cfg.max_frames,
                    Sampler::new(self.cfg.audio_sampling, seed ^ 1),
                    Sampler::new(self.cfg.text_sampling, seed),
                    d.clone(),
                );
                Driver::Moshi {
                    state: Box::new(state),
                    prev_text: text_start,
                }
            }
            // Validated in `new`, so this cannot fail.
            Kind::Stt(asr) => Driver::Stt(Box::new(
                AsrState::from_config(lm, asr).expect("ASR settings validated in new"),
            )),
        };
        self.session = Some(Session {
            id,
            driver,
            codec,
            text: Utf8Buf::default(),
            user_turn: TurnTracker::default(),
            asst_turn: TurnTracker::default(),
        });
        Ok(())
    }

    fn close_session(&mut self, id: u64) {
        if self.session.as_ref().is_some_and(|s| s.id == id) {
            let s = self.session.take().expect("checked");
            self.idle = Some(s.driver.into_model());
        }
    }

    fn step(
        &mut self,
        inputs: &mut [FrameInput],
        outputs: &mut [FrameOutput],
    ) -> Result<(), EngineError> {
        let Some(mut s) = self.session.take() else {
            return match inputs.first() {
                Some(i) => Err(EngineError::StepFailed(format!(
                    "unknown session {}",
                    i.session
                ))),
                None => Ok(()),
            };
        };
        let mut result = Ok(());
        for (input, out) in inputs.iter_mut().zip(outputs.iter_mut()) {
            if input.session != s.id {
                result = Err(EngineError::StepFailed(format!(
                    "unknown session {}",
                    input.session
                )));
                break;
            }
            if let Err(e) = self.step_session(&mut s, input, out) {
                result = Err(e);
                break;
            }
        }
        match result {
            Ok(()) => self.session = Some(s),
            // A failed step ends the session: take the model back so the
            // engine can serve the next one.
            Err(_) => self.idle = Some(s.driver.into_model()),
        }
        result
    }
}
