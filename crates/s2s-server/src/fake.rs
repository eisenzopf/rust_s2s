//! A deterministic stand-in for the model runtime, for development and
//! tests.
//!
//! Per mode:
//!
//! - **S2S**: echoes the user's audio back after `echo_delay_frames` frames.
//!   An energy detector on the input produces `user_started`/`user_stopped`
//!   turn events and a synthetic user transcript (`"u1 "`, `"u2 "`, … one
//!   word every `transcript_every` voiced frames); the same detector on the
//!   echoed output produces `assistant_*` turn events and an assistant
//!   transcript (`"a1 "`, `"a2 "`, …).
//! - **STT**: silent output; user turn events and transcript as above.
//! - **TTS**: client audio is ignored. Each word of `text.input` becomes
//!   `frames_per_word` frames of a sine tone, with the word sent as an
//!   assistant transcript delta when its tone starts.
//!
//! In S2S and STT modes, `text.input` acts as a typed user turn: it is
//! echoed as a user transcript delta and answered with
//! `"You said: <text>"`. A text input of the form `/tool <name> [<json>]`
//! instead makes the engine emit a `tool.call` (ids `c_1`, `c_2`, …) with
//! `<json>` as arguments (default `{}`), followed by `tool.status
//! dispatched`. A `tool.result` for a pending call yields `tool.status
//! injected` and an assistant delta quoting the result; a
//! `session.cancel_tool` for a pending call yields `tool.status cancelled`,
//! as does a result for an unknown or cancelled call.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use s2s_core::streams::{FRAME_SAMPLES, SAMPLE_RATE};
use serde_json::Value;

use crate::engine::{
    Engine, EngineError, EngineEvent, Frame, FrameInput, FrameOutput, Injection, SessionConfig,
    SessionId,
};
use crate::protocol::{Mode, ToolState, TurnEvent};

/// Tuning for [`FakeEngine`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FakeEngineConfig {
    /// Echo delay in frames (S2S).
    pub echo_delay_frames: usize,
    /// RMS level above which a frame counts as speech.
    pub vad_threshold: f32,
    /// Quiet frames before a speaker counts as stopped.
    pub hangover_frames: u32,
    /// Voiced frames per synthetic transcript word.
    pub transcript_every: u64,
    /// Tone frames per word (TTS).
    pub frames_per_word: u32,
    /// Tone frequency in Hz (TTS).
    pub tone_hz: f32,
    /// Tone amplitude (TTS).
    pub tone_amplitude: f32,
}

impl Default for FakeEngineConfig {
    fn default() -> Self {
        Self {
            echo_delay_frames: 2,
            vad_threshold: 0.02,
            hangover_frames: 3,
            transcript_every: 5,
            frames_per_word: 3,
            tone_hz: 440.0,
            tone_amplitude: 0.3,
        }
    }
}

/// Counters shared with whoever created the engine, so tests can observe
/// batching after the engine has moved into the frame clock.
#[derive(Debug, Default)]
pub struct FakeEngineStats {
    /// `step` calls so far.
    pub steps: AtomicU64,
    /// Batch size of the most recent step.
    pub last_batch: AtomicUsize,
    /// Largest batch seen.
    pub max_batch: AtomicUsize,
    /// Sessions opened.
    pub opened: AtomicU64,
    /// Sessions closed.
    pub closed: AtomicU64,
}

/// Simple energy-based speaking detector with hangover.
#[derive(Debug, Default)]
struct Vad {
    speaking: bool,
    quiet_run: u32,
}

impl Vad {
    /// Returns `Some(true)` on a start, `Some(false)` on a stop.
    fn update(&mut self, voiced: bool, hangover: u32) -> Option<bool> {
        if voiced {
            self.quiet_run = 0;
            if !self.speaking {
                self.speaking = true;
                return Some(true);
            }
        } else if self.speaking {
            self.quiet_run += 1;
            if self.quiet_run >= hangover {
                self.speaking = false;
                self.quiet_run = 0;
                return Some(false);
            }
        }
        None
    }
}

#[derive(Debug)]
struct FakeSession {
    mode: Mode,
    delay_line: VecDeque<Box<Frame>>,
    user_vad: Vad,
    assistant_vad: Vad,
    user_voiced: u64,
    assistant_voiced: u64,
    tts_words: VecDeque<String>,
    tts_frames_left: u32,
    tone_phase: f32,
    next_call: u64,
    pending_calls: BTreeSet<String>,
}

impl FakeSession {
    fn new(mode: Mode) -> Self {
        Self {
            mode,
            delay_line: VecDeque::new(),
            user_vad: Vad::default(),
            assistant_vad: Vad::default(),
            user_voiced: 0,
            assistant_voiced: 0,
            tts_words: VecDeque::new(),
            tts_frames_left: 0,
            tone_phase: 0.0,
            next_call: 1,
            pending_calls: BTreeSet::new(),
        }
    }
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

/// See the [module docs](self).
#[derive(Debug, Default)]
pub struct FakeEngine {
    config: FakeEngineConfig,
    sessions: HashMap<SessionId, FakeSession>,
    stats: Arc<FakeEngineStats>,
}

impl FakeEngine {
    /// An engine with the given tuning.
    pub fn new(config: FakeEngineConfig) -> Self {
        Self {
            config,
            sessions: HashMap::new(),
            stats: Arc::default(),
        }
    }

    /// A handle to this engine's counters.
    pub fn stats(&self) -> Arc<FakeEngineStats> {
        Arc::clone(&self.stats)
    }

    fn handle_injection(s: &mut FakeSession, inj: Injection, ev: &mut Vec<EngineEvent>) {
        match inj {
            Injection::Text(text) if s.mode == Mode::Tts => {
                s.tts_words
                    .extend(text.split_whitespace().map(str::to_owned));
            }
            Injection::Text(text) => {
                if let Some(rest) = text.strip_prefix("/tool ") {
                    let rest = rest.trim();
                    let (name, args) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
                    let arguments = if args.trim().is_empty() {
                        Value::Object(Default::default())
                    } else {
                        serde_json::from_str(args.trim())
                            .unwrap_or_else(|_| Value::String(args.trim().to_owned()))
                    };
                    let call_id = format!("c_{}", s.next_call);
                    s.next_call += 1;
                    s.pending_calls.insert(call_id.clone());
                    ev.push(EngineEvent::ToolCall {
                        call_id: call_id.clone(),
                        name: name.to_owned(),
                        arguments,
                    });
                    ev.push(EngineEvent::ToolStatus {
                        call_id,
                        state: ToolState::Dispatched,
                    });
                } else {
                    ev.push(EngineEvent::UserText(text.clone()));
                    ev.push(EngineEvent::AssistantText(format!("You said: {text}")));
                }
            }
            Injection::ToolResult { call_id, json } => {
                if s.pending_calls.remove(&call_id) {
                    ev.push(EngineEvent::ToolStatus {
                        call_id,
                        state: ToolState::Injected,
                    });
                    ev.push(EngineEvent::AssistantText(format!(
                        "The tool returned {json}."
                    )));
                } else {
                    ev.push(EngineEvent::ToolStatus {
                        call_id,
                        state: ToolState::Cancelled,
                    });
                }
            }
            Injection::CancelTool { call_id } => {
                if s.pending_calls.remove(&call_id) {
                    ev.push(EngineEvent::ToolStatus {
                        call_id,
                        state: ToolState::Cancelled,
                    });
                }
            }
        }
    }

    fn step_one(
        cfg: &FakeEngineConfig,
        s: &mut FakeSession,
        input: &mut FrameInput,
        output: &mut FrameOutput,
    ) {
        let ev = &mut output.events;
        for inj in std::mem::take(&mut input.injections) {
            Self::handle_injection(s, inj, ev);
        }

        // User side.
        if s.mode != Mode::Tts {
            let voiced = rms(&input.audio[..]) > cfg.vad_threshold;
            match s.user_vad.update(voiced, cfg.hangover_frames) {
                Some(true) => ev.push(EngineEvent::Turn(TurnEvent::UserStarted)),
                Some(false) => ev.push(EngineEvent::Turn(TurnEvent::UserStopped)),
                None => {}
            }
            if voiced {
                s.user_voiced += 1;
                if s.user_voiced.is_multiple_of(cfg.transcript_every) {
                    let n = s.user_voiced / cfg.transcript_every;
                    ev.push(EngineEvent::UserText(format!("u{n} ")));
                }
            }
        }

        // Assistant audio.
        match s.mode {
            Mode::S2s => {
                s.delay_line.push_back(input.audio.clone());
                if s.delay_line.len() > cfg.echo_delay_frames {
                    if let Some(f) = s.delay_line.pop_front() {
                        output.audio = f;
                    }
                }
            }
            Mode::Stt => {}
            Mode::Tts => {
                if s.tts_frames_left == 0 {
                    if let Some(word) = s.tts_words.pop_front() {
                        ev.push(EngineEvent::AssistantText(format!("{word} ")));
                        s.tts_frames_left = cfg.frames_per_word;
                    }
                }
                if s.tts_frames_left > 0 {
                    s.tts_frames_left -= 1;
                    let step = 2.0 * std::f32::consts::PI * cfg.tone_hz / SAMPLE_RATE as f32;
                    for x in output.audio.iter_mut() {
                        *x = cfg.tone_amplitude * s.tone_phase.sin();
                        s.tone_phase = (s.tone_phase + step) % (2.0 * std::f32::consts::PI);
                    }
                }
            }
        }

        // Assistant side.
        let voiced = rms(&output.audio[..]) > cfg.vad_threshold;
        match s.assistant_vad.update(voiced, cfg.hangover_frames) {
            Some(true) => ev.push(EngineEvent::Turn(TurnEvent::AssistantStarted)),
            Some(false) => ev.push(EngineEvent::Turn(TurnEvent::AssistantStopped)),
            None => {}
        }
        if voiced && s.mode == Mode::S2s {
            s.assistant_voiced += 1;
            if s.assistant_voiced.is_multiple_of(cfg.transcript_every) {
                let n = s.assistant_voiced / cfg.transcript_every;
                ev.push(EngineEvent::AssistantText(format!("a{n} ")));
            }
        }
        debug_assert_eq!(output.audio.len(), FRAME_SAMPLES);
    }
}

impl Engine for FakeEngine {
    fn open_session(&mut self, id: SessionId, config: &SessionConfig) -> Result<(), EngineError> {
        if self.sessions.contains_key(&id) {
            return Err(EngineError::Rejected(format!("session {id} already open")));
        }
        self.sessions.insert(id, FakeSession::new(config.mode));
        self.stats.opened.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn close_session(&mut self, id: SessionId) {
        if self.sessions.remove(&id).is_some() {
            self.stats.closed.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn step(
        &mut self,
        inputs: &mut [FrameInput],
        outputs: &mut [FrameOutput],
    ) -> Result<(), EngineError> {
        if inputs.len() != outputs.len() {
            return Err(EngineError::StepFailed(format!(
                "{} inputs but {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }
        for (input, output) in inputs.iter_mut().zip(outputs.iter_mut()) {
            let s = self.sessions.get_mut(&input.session).ok_or_else(|| {
                EngineError::StepFailed(format!("unknown session {}", input.session))
            })?;
            Self::step_one(&self.config, s, input, output);
        }
        let n = inputs.len();
        self.stats.steps.fetch_add(1, Ordering::Relaxed);
        self.stats.last_batch.store(n, Ordering::Relaxed);
        self.stats.max_batch.fetch_max(n, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::silent_frame;
    use serde_json::json;

    fn tone_frame(amp: f32) -> Box<Frame> {
        let mut f = silent_frame();
        for (i, x) in f.iter_mut().enumerate() {
            *x = amp * (i as f32 * 0.1).sin();
        }
        f
    }

    fn step1(
        e: &mut FakeEngine,
        id: SessionId,
        idx: u64,
        audio: Box<Frame>,
        inj: Vec<Injection>,
    ) -> FrameOutput {
        let mut inputs = vec![FrameInput {
            session: id,
            frame_index: idx,
            audio,
            injections: inj,
        }];
        let mut outputs = vec![FrameOutput::new()];
        e.step(&mut inputs, &mut outputs).unwrap();
        outputs.pop().unwrap()
    }

    #[test]
    fn s2s_echoes_with_delay_and_emits_events() {
        let mut e = FakeEngine::default();
        let cfg = FakeEngineConfig::default();
        e.config = cfg;
        e.open_session(1, &SessionConfig::default()).unwrap();
        let mut events = Vec::new();
        let mut outs = Vec::new();
        for i in 0..20 {
            let audio = if i < 10 {
                tone_frame(0.5)
            } else {
                silent_frame()
            };
            let out = step1(&mut e, 1, i, audio, vec![]);
            events.extend(out.events.iter().cloned().map(|ev| (i, ev)));
            outs.push(out.audio);
        }
        let d = cfg.echo_delay_frames;
        assert!(outs[..d].iter().all(|f| rms(&f[..]) == 0.0));
        assert_eq!(*outs[d], *tone_frame(0.5));
        assert_eq!(*outs[10 + d], *silent_frame());
        let has = |want: &EngineEvent| events.iter().any(|(_, e)| e == want);
        assert!(has(&EngineEvent::Turn(TurnEvent::UserStarted)));
        assert!(has(&EngineEvent::Turn(TurnEvent::UserStopped)));
        assert!(has(&EngineEvent::Turn(TurnEvent::AssistantStarted)));
        assert!(has(&EngineEvent::Turn(TurnEvent::AssistantStopped)));
        assert!(has(&EngineEvent::UserText("u1 ".into())));
        assert!(has(&EngineEvent::UserText("u2 ".into())));
        assert!(has(&EngineEvent::AssistantText("a2 ".into())));
        let started = |t| {
            events
                .iter()
                .find(|(_, e)| *e == EngineEvent::Turn(t))
                .unwrap()
                .0
        };
        assert_eq!(started(TurnEvent::UserStarted), 0);
        assert_eq!(started(TurnEvent::AssistantStarted), d as u64);
    }

    #[test]
    fn tool_round_trip() {
        let mut e = FakeEngine::default();
        e.open_session(3, &SessionConfig::default()).unwrap();
        let out = step1(
            &mut e,
            3,
            0,
            silent_frame(),
            vec![Injection::Text(
                r#"/tool get_weather {"city":"Paris"}"#.into(),
            )],
        );
        assert_eq!(
            out.events,
            vec![
                EngineEvent::ToolCall {
                    call_id: "c_1".into(),
                    name: "get_weather".into(),
                    arguments: json!({"city": "Paris"})
                },
                EngineEvent::ToolStatus {
                    call_id: "c_1".into(),
                    state: ToolState::Dispatched
                }
            ]
        );
        let out = step1(
            &mut e,
            3,
            1,
            silent_frame(),
            vec![Injection::ToolResult {
                call_id: "c_1".into(),
                json: json!({"temp": 18}),
            }],
        );
        assert_eq!(
            out.events[0],
            EngineEvent::ToolStatus {
                call_id: "c_1".into(),
                state: ToolState::Injected
            }
        );
        assert_eq!(
            out.events[1],
            EngineEvent::AssistantText(r#"The tool returned {"temp":18}."#.into())
        );
        // Cancel, then a late result.
        let out = step1(
            &mut e,
            3,
            2,
            silent_frame(),
            vec![
                Injection::Text("/tool ping".into()),
                Injection::CancelTool {
                    call_id: "c_2".into(),
                },
                Injection::ToolResult {
                    call_id: "c_2".into(),
                    json: Value::Null,
                },
            ],
        );
        let states: Vec<_> = out
            .events
            .iter()
            .filter_map(|e| match e {
                EngineEvent::ToolStatus { state, .. } => Some(*state),
                _ => None,
            })
            .collect();
        assert_eq!(
            states,
            vec![
                ToolState::Dispatched,
                ToolState::Cancelled,
                ToolState::Cancelled
            ]
        );
    }

    #[test]
    fn tts_speaks_words_as_tones() {
        let mut e = FakeEngine::default();
        let cfg = SessionConfig {
            mode: Mode::Tts,
            ..SessionConfig::default()
        };
        e.open_session(9, &cfg).unwrap();
        let mut texts = Vec::new();
        let mut voiced = 0;
        for i in 0..10 {
            let inj = if i == 0 {
                vec![Injection::Text("hello there".into())]
            } else {
                vec![]
            };
            // User audio is ignored in TTS mode.
            let out = step1(&mut e, 9, i, tone_frame(0.9), inj);
            if rms(&out.audio[..]) > 0.1 {
                voiced += 1;
            }
            for ev in out.events {
                if let EngineEvent::AssistantText(t) = ev {
                    texts.push(t);
                }
            }
        }
        assert_eq!(texts, vec!["hello ", "there "]);
        assert_eq!(voiced, 6);
    }

    #[test]
    fn batches_and_stats() {
        let mut e = FakeEngine::default();
        let stats = e.stats();
        for id in 0..3 {
            e.open_session(id, &SessionConfig::default()).unwrap();
        }
        assert!(e.open_session(1, &SessionConfig::default()).is_err());
        let mut inputs: Vec<_> = (0..3)
            .map(|id| FrameInput {
                session: id,
                frame_index: 0,
                audio: silent_frame(),
                injections: vec![],
            })
            .collect();
        let mut outputs: Vec<_> = (0..3).map(|_| FrameOutput::new()).collect();
        e.step(&mut inputs, &mut outputs).unwrap();
        assert_eq!(stats.max_batch.load(Ordering::Relaxed), 3);
        e.close_session(1);
        assert!(e.step(&mut inputs, &mut outputs).is_err(), "closed session");
        assert!(
            e.step(&mut inputs[..1], &mut outputs).is_err(),
            "length mismatch"
        );
        assert_eq!(stats.opened.load(Ordering::Relaxed), 3);
        assert_eq!(stats.closed.load(Ordering::Relaxed), 1);
    }
}
