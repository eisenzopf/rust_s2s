//! [`DuplexEngine`]: the duplex model behind the server's `Engine` trait.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use s2s_core::streams::{special, StreamKind, StreamLayout, AUDIO_PAD, FRAME_SAMPLES};
use s2s_core::SplitMix64;
use s2s_data::TextTokenizer;
use s2s_duplex::stream::{DuplexState, SubstepHook};
use s2s_duplex::DuplexModel;
use s2s_kernels::reference::sampling::{apply_token_bitmask, sample, SamplingParams};
use s2s_server::protocol::{ToolState, TurnEvent};
use s2s_server::{Engine, EngineError, EngineEvent, FrameInput, FrameOutput, Mode, SessionConfig};
use s2s_tools::{
    CallId, DefaultPromptRenderer, Executor, ExecutorConfig, ExecutorEvent, InjectionConfig,
    InjectionFormatter, Marker, ParseEvent, TokenPiece, TokenizerInfo, ToolCallConstraint,
    ToolCallGrammar, ToolCallParser, ToolHandler, ToolPromptRenderer, ToolRegistry, ToolSpec,
    ToolStatus,
};
use tokio::sync::mpsc::UnboundedReceiver;

use crate::codec::FrameCodec;
use crate::tokenizer::Tokenizer;

/// Engine settings.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Sampling for text streams.
    pub text_sampling: SamplingParams,
    /// Sampling for audio streams.
    pub audio_sampling: SamplingParams,
    /// Base seed; each session derives its own.
    pub seed: u64,
    /// Tool executor timeouts.
    pub executor: ExecutorConfig,
    /// Result injection formatting (budget, cancelled-call policy).
    pub injection: InjectionConfig,
    /// Most sessions at once (`open_session` rejects beyond this).
    pub max_sessions: usize,
}

impl Default for EngineConfig {
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
            executor: ExecutorConfig::default(),
            injection: InjectionConfig::default(),
            max_sessions: 64,
        }
    }
}

type CodecFactory = Box<dyn Fn() -> Box<dyn FrameCodec> + Send>;
type HookFactory = Box<dyn Fn(u64) -> Box<dyn SubstepHook<f32> + Send> + Send>;

/// Special-token ids the engine needs.
#[derive(Clone, Copy, Debug)]
struct Ids {
    pad: u32,
    epad: u32,
    sys: u32,
    sys_end: u32,
    tool_result: u32,
    tool_result_end: u32,
}

/// Incremental UTF-8 assembly of byte-level tokens.
#[derive(Debug, Default)]
struct Utf8Buf(Vec<u8>);

impl Utf8Buf {
    /// Append bytes and return whatever is now complete UTF-8.
    fn push(&mut self, bytes: &[u8]) -> String {
        self.0.extend_from_slice(bytes);
        let valid = match std::str::from_utf8(&self.0) {
            Ok(_) => self.0.len(),
            Err(e) if e.error_len().is_some() => {
                // Invalid sequence: emit it lossily rather than stall.
                let s = String::from_utf8_lossy(&self.0).into_owned();
                self.0.clear();
                return s;
            }
            Err(e) => e.valid_up_to(),
        };
        let rest = self.0.split_off(valid);
        String::from_utf8(std::mem::replace(&mut self.0, rest)).expect("valid prefix")
    }
}

struct Session {
    mode: Mode,
    state: DuplexState<f32>,
    codec: Box<dyn FrameCodec>,
    /// Aligned user codes of recent frames (for the acoustic delay).
    user_hist: VecDeque<Vec<u32>>,
    /// Generated assistant audio rows (delayed domain) of recent frames.
    asst_hist: VecDeque<Vec<u32>>,
    parser: ToolCallParser,
    constraint: Option<ToolCallConstraint>,
    executor: Executor,
    events: UnboundedReceiver<ExecutorEvent>,
    rng: SplitMix64,
    hook: Option<Box<dyn SubstepHook<f32> + Send>>,
    user_text: Utf8Buf,
    /// TTS: words waiting to be spoken, and tokens of the word being forced.
    tts_words: VecDeque<Vec<u32>>,
    tts_forcing: VecDeque<u32>,
    /// Injections to feed after the next row.
    pending: Vec<Vec<u32>>,
    /// Turn tracking for the user (input energy) and the assistant (text).
    user_turn: TurnTracker,
    asst_turn: TurnTracker,
}

/// Frames of inactivity before a speaker counts as stopped (~0.5 s).
const TURN_HANGOVER_FRAMES: u32 = 6;
/// RMS level above which a user frame counts as speech.
const USER_SPEECH_RMS: f32 = 0.02;

/// Started/stopped detection with a hangover: active frames start a turn,
/// and `TURN_HANGOVER_FRAMES` inactive frames in a row end it.
#[derive(Debug, Default)]
struct TurnTracker {
    speaking: bool,
    quiet: u32,
}

impl TurnTracker {
    /// Update with this frame's activity; returns `Some(true)` on a start
    /// and `Some(false)` on a stop.
    fn update(&mut self, active: bool) -> Option<bool> {
        if active {
            self.quiet = 0;
            if !self.speaking {
                self.speaking = true;
                return Some(true);
            }
        } else if self.speaking {
            self.quiet += 1;
            if self.quiet >= TURN_HANGOVER_FRAMES {
                self.speaking = false;
                return Some(false);
            }
        }
        None
    }
}

/// The duplex model as a frame-synchronous engine.
pub struct DuplexEngine<K: Tokenizer> {
    model: Arc<DuplexModel<f32>>,
    tok: Arc<K>,
    ids: Ids,
    new_codec: CodecFactory,
    hook_factory: Option<HookFactory>,
    handlers: Vec<(String, Arc<dyn ToolHandler>)>,
    cfg: EngineConfig,
    sessions: HashMap<u64, Session>,
}

impl<K: Tokenizer> std::fmt::Debug for DuplexEngine<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuplexEngine")
            .field("sessions", &self.sessions.len())
            .finish_non_exhaustive()
    }
}

struct ArcHandler(Arc<dyn ToolHandler>);

impl ToolHandler for ArcHandler {
    fn call(&self, arguments: serde_json::Value) -> s2s_tools::HandlerFuture {
        self.0.call(arguments)
    }
}

impl<K: Tokenizer> DuplexEngine<K> {
    /// New engine. `new_codec` creates one codec per session; its codebook
    /// count must match the model's layout.
    pub fn new(
        model: Arc<DuplexModel<f32>>,
        tok: Arc<K>,
        new_codec: impl Fn() -> Box<dyn FrameCodec> + Send + 'static,
        cfg: EngineConfig,
    ) -> Result<Self, EngineError> {
        let get = |name: &str| {
            TextTokenizer::special_id(tok.as_ref(), name)
                .ok_or_else(|| EngineError::Rejected(format!("tokenizer lacks {name}")))
        };
        let ids = Ids {
            pad: get(special::PAD)?,
            epad: get(special::EPAD)?,
            sys: get(special::SYS)?,
            sys_end: get(special::SYS_END)?,
            tool_result: get(special::TOOL_RESULT)?,
            tool_result_end: get(special::TOOL_RESULT_END)?,
        };
        if ids.pad != model.config().text_pad {
            return Err(EngineError::Rejected(
                "tokenizer <pad> differs from the model's text_pad".into(),
            ));
        }
        Ok(Self {
            model,
            tok,
            ids,
            new_codec: Box::new(new_codec),
            hook_factory: None,
            handlers: Vec::new(),
            cfg,
            sessions: HashMap::new(),
        })
    }

    /// Register a server-side tool implementation, available to every
    /// session that declares a tool of this name.
    pub fn register_tool(&mut self, name: &str, handler: impl ToolHandler) {
        self.handlers.push((name.to_string(), Arc::new(handler)));
    }

    /// Replace the sampler with a custom per-session token chooser (it still
    /// sees grammar-masked logits for tool calls). For tests and scripted
    /// demos.
    pub fn with_hook_factory(
        mut self,
        f: impl Fn(u64) -> Box<dyn SubstepHook<f32> + Send> + Send + 'static,
    ) -> Self {
        self.hook_factory = Some(Box::new(f));
        self
    }

    fn layout(&self) -> StreamLayout {
        self.model.config().layout
    }

    /// `<open>body<close>` from the runtime's formatted text, as tokens.
    fn marked(&self, text: &str, open: &str, close: &str, open_id: u32, close_id: u32) -> Vec<u32> {
        let body = text
            .strip_prefix(open)
            .and_then(|t| t.strip_suffix(close))
            .unwrap_or(text);
        let mut toks = vec![open_id];
        toks.extend(TextTokenizer::encode(self.tok.as_ref(), body));
        toks.push(close_id);
        toks
    }

    fn step_session(&self, s: &mut Session, input: &mut FrameInput, out: &mut FrameOutput) {
        let l = self.layout();
        let (m, q) = (l.text_slots, l.codebooks);
        let formatter = InjectionFormatter::new(self.cfg.injection.clone());

        // Client control input.
        for inj in std::mem::take(&mut input.injections) {
            match inj {
                s2s_server::engine::Injection::Text(text) => {
                    if s.mode == Mode::Tts {
                        for (i, w) in text.split_whitespace().enumerate() {
                            let word = if i == 0 && s.tts_words.is_empty() {
                                w.to_string()
                            } else {
                                format!(" {w}")
                            };
                            s.tts_words
                                .push_back(TextTokenizer::encode(self.tok.as_ref(), &word));
                        }
                    } else {
                        s.pending
                            .push(TextTokenizer::encode(self.tok.as_ref(), &text));
                    }
                }
                s2s_server::engine::Injection::ToolResult { call_id, json } => {
                    if s.executor
                        .submit_client_result(&CallId(call_id.clone()), json)
                        .is_err()
                    {
                        out.events.push(EngineEvent::ToolStatus {
                            call_id,
                            state: ToolState::Failed,
                        });
                    }
                }
                s2s_server::engine::Injection::CancelTool { call_id } => {
                    s.executor.cancel(&CallId(call_id));
                }
            }
        }

        // Turn events: the user by input energy (a simple VAD; the model's own
        // turn-taking is learned, this is only for client UIs and metrics).
        if s.mode != Mode::Tts {
            let rms =
                (input.audio.iter().map(|x| x * x).sum::<f32>() / FRAME_SAMPLES as f32).sqrt();
            match s.user_turn.update(rms > USER_SPEECH_RMS) {
                Some(true) => out.events.push(EngineEvent::Turn(TurnEvent::UserStarted)),
                Some(false) => out.events.push(EngineEvent::Turn(TurnEvent::UserStopped)),
                None => {}
            }
        }

        // 1. User audio into the delayed domain.
        let silence = s.codec.silence_codes();
        let user_codes = if s.mode == Mode::Tts {
            silence.clone()
        } else {
            s.codec.encode(&input.audio[..])
        };
        s.user_hist.push_front(user_codes);
        s.user_hist.truncate(l.acoustic_delay + 1);
        let user_row: Vec<u32> = (0..q)
            .map(|c| {
                let lag = if c == 0 { 0 } else { l.acoustic_delay };
                s.user_hist.get(lag).map_or(AUDIO_PAD, |codes| codes[c])
            })
            .collect();

        // 2. Sample this row's generated tokens.
        let mut chooser = Chooser {
            mode: s.mode,
            layout: l,
            ids: self.ids,
            silence: &silence,
            constraint: s.constraint.as_mut(),
            tts_words: &mut s.tts_words,
            tts_forcing: &mut s.tts_forcing,
            inner: s.hook.as_deref_mut(),
            rng: &mut s.rng,
            text: &self.cfg.text_sampling,
            audio: &self.cfg.audio_sampling,
        };
        let generated = self.model.sample_next(&mut s.state, &mut chooser);

        // The assistant by text activity: any non-pad token on its text slots.
        let asst_active = (0..m).any(|slot| {
            let t = generated[l.index(StreamKind::AssistantText(slot))];
            t != self.ids.pad
        });
        match s.asst_turn.update(asst_active) {
            Some(true) => out
                .events
                .push(EngineEvent::Turn(TurnEvent::AssistantStarted)),
            Some(false) => out
                .events
                .push(EngineEvent::Turn(TurnEvent::AssistantStopped)),
            None => {}
        }

        // 3. Events from the text streams.
        let tu = generated[l.index(StreamKind::UserText)];
        if tu != self.ids.pad && tu != self.ids.epad {
            if let TokenPiece::Bytes(b) = TokenizerInfo::token(self.tok.as_ref(), tu) {
                let text = s.user_text.push(&b);
                if !text.is_empty() {
                    out.events.push(EngineEvent::UserText(text));
                }
            }
        }
        let mut parse_events = Vec::new();
        for slot in 0..m {
            let t = generated[l.index(StreamKind::AssistantText(slot))];
            match TokenizerInfo::token(self.tok.as_ref(), t) {
                TokenPiece::Bytes(b) => parse_events.extend(s.parser.push_bytes(&b)),
                TokenPiece::Special(name) => {
                    if let Some(mk) = Marker::ALL.iter().find(|mk| mk.as_str() == name) {
                        parse_events.extend(s.parser.push_marker(*mk));
                    }
                }
            }
        }
        for ev in parse_events {
            match ev {
                ParseEvent::Text(t) if !t.is_empty() => {
                    out.events.push(EngineEvent::AssistantText(t))
                }
                ParseEvent::ToolCall(call) => {
                    let call_id = call.call_id.0.clone();
                    match s.executor.dispatch(call) {
                        Ok(()) => out.events.push(EngineEvent::ToolStatus {
                            call_id,
                            state: ToolState::Dispatched,
                        }),
                        Err(_) => out.events.push(EngineEvent::ToolStatus {
                            call_id,
                            state: ToolState::Failed,
                        }),
                    }
                }
                ParseEvent::CancelRequested => {
                    s.executor.cancel_all();
                }
                _ => {}
            }
        }

        // 4. Assistant audio: undelay and decode frame (row - acoustic_delay).
        let am_row: Vec<u32> = (0..q)
            .map(|c| generated[l.index(StreamKind::AssistantAudio(c))])
            .collect();
        s.asst_hist.push_front(am_row.clone());
        s.asst_hist.truncate(l.acoustic_delay + 1);
        if s.mode != Mode::Stt {
            if let Some(old) = s.asst_hist.get(l.acoustic_delay) {
                let frame: Vec<u32> = (0..q)
                    .map(|c| if c == 0 { old[0] } else { am_row[c] })
                    .collect();
                if frame.iter().all(|&c| c != AUDIO_PAD) {
                    let pcm = s.codec.decode(&frame);
                    out.audio[..].copy_from_slice(&pcm[..FRAME_SAMPLES]);
                }
            }
        }

        // 5. Feed the row back, then any injections.
        let mut row = generated;
        row.extend_from_slice(&user_row);
        self.model.feed_row(&mut s.state, &row);
        while let Ok(ev) = s.events.try_recv() {
            match ev {
                ExecutorEvent::Dispatched { call_id, .. } => {
                    out.events.push(EngineEvent::ToolStatus {
                        call_id: call_id.0,
                        state: ToolState::Running,
                    })
                }
                ExecutorEvent::ClientCall(call) => out.events.push(EngineEvent::ToolCall {
                    call_id: call.call_id.0,
                    name: call.name,
                    arguments: call.arguments,
                }),
                ExecutorEvent::Finished(outcome) => {
                    let state = match outcome.status {
                        ToolStatus::Ok(_) => ToolState::Injected,
                        ToolStatus::Cancelled => ToolState::Cancelled,
                        _ => ToolState::Failed,
                    };
                    if let Some(seg) = formatter.format(&outcome) {
                        s.pending.push(self.marked(
                            &seg.text,
                            special::TOOL_RESULT,
                            special::TOOL_RESULT_END,
                            self.ids.tool_result,
                            self.ids.tool_result_end,
                        ));
                    }
                    out.events.push(EngineEvent::ToolStatus {
                        call_id: outcome.call_id.0,
                        state,
                    });
                }
            }
        }
        for toks in std::mem::take(&mut s.pending) {
            self.model.feed_injection(&mut s.state, &toks);
        }
    }
}

/// Per-sub-step token choice for one session and frame.
struct Chooser<'a> {
    mode: Mode,
    layout: StreamLayout,
    ids: Ids,
    silence: &'a [u32],
    constraint: Option<&'a mut ToolCallConstraint>,
    tts_words: &'a mut VecDeque<Vec<u32>>,
    tts_forcing: &'a mut VecDeque<u32>,
    inner: Option<&'a mut (dyn SubstepHook<f32> + Send + 'static)>,
    rng: &'a mut SplitMix64,
    text: &'a SamplingParams,
    audio: &'a SamplingParams,
}

impl Chooser<'_> {
    fn pick(&mut self, k: usize, logits: &mut [f32], is_text: bool) -> u32 {
        if let Some(h) = self.inner.as_deref_mut() {
            return h.choose(k, logits);
        }
        let p = if is_text { self.text } else { self.audio };
        sample(logits, p, self.rng.next_f64()).unwrap_or(0) as u32
    }
}

impl SubstepHook<f32> for Chooser<'_> {
    fn choose(&mut self, k: usize, logits: &mut [f32]) -> u32 {
        match self.layout.kind(k) {
            StreamKind::UserText => {
                if self.mode == Mode::Tts {
                    self.ids.pad
                } else {
                    self.pick(k, logits, true)
                }
            }
            StreamKind::AssistantText(slot) => match self.mode {
                Mode::Stt => self.ids.pad,
                Mode::Tts => {
                    // Model-paced text consumption: the model asks for the
                    // next word by sampling <epad> on slot 0.
                    if let Some(t) = self.tts_forcing.pop_front() {
                        return t;
                    }
                    if slot == 0 && !self.tts_words.is_empty() {
                        let (pad, epad) = (self.ids.pad as usize, self.ids.epad as usize);
                        let lp = logits.get(pad).copied().unwrap_or(f32::NEG_INFINITY);
                        let le = logits.get(epad).copied().unwrap_or(f32::NEG_INFINITY);
                        if le >= lp {
                            if let Some(w) = self.tts_words.pop_front() {
                                self.tts_forcing.extend(w);
                            }
                            return self.ids.epad;
                        }
                    }
                    self.ids.pad
                }
                Mode::S2s => {
                    if let Some(c) = self.constraint.as_deref_mut() {
                        if let Ok(Some(mask)) = c.mask() {
                            apply_token_bitmask(logits, &mask);
                        }
                    }
                    let t = self.pick(k, logits, true);
                    if let Some(c) = self.constraint.as_deref_mut() {
                        // A token the grammar rejects resets the constraint
                        // (the parser reports the span as invalid).
                        if c.observe(t).is_err() {
                            c.reset();
                        }
                    }
                    t
                }
            },
            StreamKind::AssistantAudio(c) => {
                if self.mode == Mode::Stt {
                    self.silence[c]
                } else {
                    self.pick(k, logits, false)
                }
            }
            StreamKind::UserAudio(_) => unreachable!("user audio is not generated"),
        }
    }
}

impl<K: Tokenizer> Engine for DuplexEngine<K> {
    fn open_session(&mut self, id: u64, config: &SessionConfig) -> Result<(), EngineError> {
        if self.sessions.len() >= self.cfg.max_sessions {
            return Err(EngineError::Rejected("engine is at capacity".into()));
        }
        let specs: Vec<ToolSpec> = config
            .tools
            .iter()
            .map(|t| serde_json::from_value(t.clone()))
            .collect::<Result<_, _>>()
            .map_err(|e| EngineError::Rejected(format!("bad tool declaration: {e}")))?;
        let registry = Arc::new(
            ToolRegistry::from_specs(specs)
                .map_err(|e| EngineError::Rejected(format!("tools: {e}")))?,
        );
        let constraint = if registry.is_empty() {
            None
        } else {
            let tok: Arc<dyn TokenizerInfo> = self.tok.clone();
            let grammar = ToolCallGrammar::new(&registry, tok, Default::default())
                .map_err(|e| EngineError::Rejected(format!("tool grammar: {e}")))?;
            Some(ToolCallConstraint::new(Arc::new(grammar)))
        };
        let (executor, events) = Executor::new(registry.clone(), self.cfg.executor.clone());
        for (name, h) in &self.handlers {
            if registry.get(name).is_some() {
                executor
                    .register_handler(name, ArcHandler(h.clone()))
                    .map_err(|e| EngineError::Rejected(format!("tool handler {name}: {e}")))?;
            }
        }
        let codec = (self.new_codec)();
        if codec.codebooks() != self.layout().codebooks {
            return Err(EngineError::Rejected(
                "codec codebooks differ from the model layout".into(),
            ));
        }
        if codec.codebook_size() != s2s_core::streams::CODEBOOK_SIZE {
            return Err(EngineError::Rejected(format!(
                "codec has {} entries per codebook; the stream layout needs {}",
                codec.codebook_size(),
                s2s_core::streams::CODEBOOK_SIZE
            )));
        }

        let mut state = self.model.new_state();
        // The "frame -1" start position, then the system segment, exactly as
        // s2s-data lays out training examples.
        self.model
            .feed_row(&mut state, &self.model.config().pad_row());
        if !config.system_prompt.is_empty() || !registry.is_empty() {
            let specs: Vec<&ToolSpec> = registry.specs().collect();
            let body = DefaultPromptRenderer.render(&config.system_prompt, &specs);
            let mut toks = vec![self.ids.sys];
            toks.extend(TextTokenizer::encode(self.tok.as_ref(), &body));
            toks.push(self.ids.sys_end);
            self.model.feed_injection(&mut state, &toks);
        }

        let session = Session {
            mode: config.mode,
            state,
            codec,
            user_hist: VecDeque::new(),
            asst_hist: VecDeque::new(),
            parser: ToolCallParser::new(registry),
            constraint,
            executor,
            events,
            rng: SplitMix64::new(self.cfg.seed ^ id.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            hook: self.hook_factory.as_ref().map(|f| f(id)),
            user_text: Utf8Buf::default(),
            tts_words: VecDeque::new(),
            tts_forcing: VecDeque::new(),
            pending: Vec::new(),
            user_turn: TurnTracker::default(),
            asst_turn: TurnTracker::default(),
        };
        self.sessions.insert(id, session);
        Ok(())
    }

    fn close_session(&mut self, id: u64) {
        if let Some(s) = self.sessions.remove(&id) {
            s.executor.cancel_all();
        }
    }

    fn step(
        &mut self,
        inputs: &mut [FrameInput],
        outputs: &mut [FrameOutput],
    ) -> Result<(), EngineError> {
        let mut sessions = std::mem::take(&mut self.sessions);
        let result = (|| {
            for (input, out) in inputs.iter_mut().zip(outputs.iter_mut()) {
                let s = sessions.get_mut(&input.session).ok_or_else(|| {
                    EngineError::StepFailed(format!("unknown session {}", input.session))
                })?;
                self.step_session(s, input, out);
            }
            Ok(())
        })();
        self.sessions = sessions;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::{TurnTracker, Utf8Buf, TURN_HANGOVER_FRAMES};

    #[test]
    fn turn_tracker_starts_and_stops_with_hangover() {
        let mut t = TurnTracker::default();
        assert_eq!(t.update(false), None);
        assert_eq!(t.update(true), Some(true));
        assert_eq!(t.update(true), None);
        for _ in 0..TURN_HANGOVER_FRAMES - 1 {
            assert_eq!(t.update(false), None);
        }
        // A brief pause shorter than the hangover does not end the turn.
        assert_eq!(t.update(true), None);
        for _ in 0..TURN_HANGOVER_FRAMES - 1 {
            assert_eq!(t.update(false), None);
        }
        assert_eq!(t.update(false), Some(false));
        assert_eq!(t.update(false), None);
    }

    #[test]
    fn utf8_assembly_waits_for_complete_characters() {
        let mut b = Utf8Buf::default();
        let bytes = "é日".as_bytes();
        assert_eq!(b.push(&bytes[..1]), "");
        assert_eq!(b.push(&bytes[1..3]), "é");
        assert_eq!(b.push(&bytes[3..]), "日");
        assert_eq!(b.push(&[0xff]), "\u{fffd}");
    }
}
