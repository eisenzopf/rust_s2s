//! The interface between the server and the model runtime.
//!
//! The server owns the network, codecs, jitter buffers and the 80 ms frame
//! clock. Every tick it calls [`Engine::step`] **once** with one
//! [`FrameInput`] per active session (the whole batch, plan §10.3), and
//! routes each [`FrameOutput`] back to its session. Sessions join and leave
//! only between ticks, through [`Engine::open_session`] and
//! [`Engine::close_session`].
//!
//! The real implementation (`s2s-infer`: Mimi encode, Temporal + Depth
//! Transformer, sampling, Mimi decode; plan §10.1) will implement this trait.
//! [`FakeEngine`](crate::fake::FakeEngine) is a development stand-in.

use s2s_core::streams::FRAME_SAMPLES;
use serde_json::Value;

use crate::protocol::{Mode, ServerMessage, SessionStart, ToolState, TurnEvent};

/// Server-assigned session id, unique for the server's lifetime.
pub type SessionId = u64;

/// One 80 ms frame of 24 kHz mono PCM.
pub type Frame = [f32; FRAME_SAMPLES];

/// A zeroed, heap-allocated frame.
pub fn silent_frame() -> Box<Frame> {
    Box::new([0.0; FRAME_SAMPLES])
}

/// What the engine needs to set up a session (from `session.start`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionConfig {
    /// System prompt, to inject as a `<sys>` segment.
    pub system_prompt: String,
    /// Tool declarations (JSON schemas), unparsed.
    pub tools: Vec<Value>,
    /// Requested voice, if any.
    pub voice: Option<String>,
    /// Endpoint mode.
    pub mode: Mode,
}

impl From<&SessionStart> for SessionConfig {
    fn from(s: &SessionStart) -> Self {
        Self {
            system_prompt: s.system_prompt.clone(),
            tools: s.tools.clone(),
            voice: s.voice.clone(),
            mode: s.mode,
        }
    }
}

/// Non-audio input the client sent since the previous tick, in arrival
/// order. The engine schedules these as injection segments (plan §3.5).
#[derive(Clone, Debug, PartialEq)]
pub enum Injection {
    /// `text.input`: TTS text, or a typed user turn.
    Text(String),
    /// `tool.result`: inject `<tool_result>{json}</tool_result>`.
    ToolResult {
        /// The call this answers.
        call_id: String,
        /// The result.
        json: Value,
    },
    /// `session.cancel_tool`.
    CancelTool {
        /// The call to cancel.
        call_id: String,
    },
}

/// One session's input for one tick.
#[derive(Debug)]
pub struct FrameInput {
    /// Which session.
    pub session: SessionId,
    /// Frames this session has been through before this one (0 on its first
    /// tick).
    pub frame_index: u64,
    /// The user's audio for this frame, from the jitter buffer (silence when
    /// the client sent nothing, and always silence in TTS mode).
    pub audio: Box<Frame>,
    /// Control input received since the last tick. The engine may take
    /// (`std::mem::take`) these.
    pub injections: Vec<Injection>,
}

/// One session's output for one tick.
#[derive(Debug)]
pub struct FrameOutput {
    /// The assistant's audio for this frame. Starts zeroed each tick.
    pub audio: Box<Frame>,
    /// Events produced this frame, in order. Starts empty each tick.
    pub events: Vec<EngineEvent>,
}

impl FrameOutput {
    /// Silent audio, no events.
    pub fn new() -> Self {
        Self {
            audio: silent_frame(),
            events: Vec::new(),
        }
    }
}

impl Default for FrameOutput {
    fn default() -> Self {
        Self::new()
    }
}

/// A non-audio output event for one session.
#[derive(Clone, Debug, PartialEq)]
pub enum EngineEvent {
    /// New user transcript text (`Tu` stream).
    UserText(String),
    /// New assistant text (`Tm` stream, outside tool calls).
    AssistantText(String),
    /// A complete, parsed tool call.
    ToolCall {
        /// Id for matching the result.
        call_id: String,
        /// Tool name.
        name: String,
        /// Arguments.
        arguments: Value,
    },
    /// A tool call changed state.
    ToolStatus {
        /// The call.
        call_id: String,
        /// New state.
        state: ToolState,
    },
    /// A turn-taking event.
    Turn(TurnEvent),
}

impl EngineEvent {
    /// The wire message for this event; `t` is the session time in seconds.
    pub fn into_message(self, t: f64) -> ServerMessage {
        match self {
            EngineEvent::UserText(text) => ServerMessage::UserTranscriptDelta { text, t },
            EngineEvent::AssistantText(text) => ServerMessage::AssistantTranscriptDelta { text, t },
            EngineEvent::ToolCall {
                call_id,
                name,
                arguments,
            } => ServerMessage::ToolCall {
                call_id,
                name,
                arguments,
            },
            EngineEvent::ToolStatus { call_id, state } => {
                ServerMessage::ToolStatus { call_id, state }
            }
            EngineEvent::Turn(event) => ServerMessage::TurnEvent { event, t },
        }
    }
}

/// An engine failure.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EngineError {
    /// The engine will not take this session (bad config, no capacity, …).
    #[error("session rejected: {0}")]
    Rejected(String),
    /// A step failed. Every session in that batch is ended.
    #[error("engine step failed: {0}")]
    StepFailed(String),
}

/// A batched, frame-synchronous model runtime.
///
/// Calls come from one task at a time, in this pattern:
/// `open_session`/`close_session` (between ticks) and `step` (once per tick
/// with every open session). `step` runs on a blocking thread
/// (`tokio::task::spawn_blocking`), so it may take as long as its compute
/// needs, but it must finish well inside 80 ms to keep up with real time.
/// The engine moves between threads across ticks, hence `Send`.
pub trait Engine: Send + 'static {
    /// Sets up state for a new session (KV cache, `<sys>` injection, …).
    /// The session takes part in the next `step`.
    fn open_session(&mut self, id: SessionId, config: &SessionConfig) -> Result<(), EngineError>;

    /// Frees a session's state. It is not part of later steps.
    fn close_session(&mut self, id: SessionId);

    /// Advances every session by one 80 ms frame.
    ///
    /// `inputs` and `outputs` have the same length; `outputs[i]` belongs to
    /// `inputs[i].session`. Each output arrives with zeroed audio and no
    /// events. Every open session appears exactly once, in no particular
    /// order. An error ends every session in the batch.
    fn step(
        &mut self,
        inputs: &mut [FrameInput],
        outputs: &mut [FrameOutput],
    ) -> Result<(), EngineError>;
}
