//! The client/server wire protocol (plan §10.5, Appendix C).
//!
//! One WebSocket carries one session. It uses two kinds of frames:
//!
//! - **Text frames** hold JSON control messages, internally tagged by a
//!   `"type"` field: [`ClientMessage`] from the client and [`ServerMessage`]
//!   from the server.
//! - **Binary frames** hold audio: the plan's `audio.chunk {opus}` message in
//!   both directions. See [`AudioChunk`] for the framing.
//!
//! # Binary audio framing
//!
//! ```text
//! byte 0      : format tag
//!                 0x01  Opus     — exactly one Opus packet (RFC 6716), mono, 24 kHz
//!                 0x02  PCM s16  — little-endian i16 samples, mono, 24 kHz
//!                 0x03  PCM f32  — little-endian f32 samples, mono, 24 kHz
//! bytes 1..   : payload
//! ```
//!
//! Client → server chunks may be any length (an Opus packet of any legal
//! duration, or any whole number of PCM samples); the server's jitter buffer
//! reassembles them into 80 ms frames. Server → client chunks are always
//! exactly one 80 ms frame (1920 samples; for Opus, one 80 ms packet) in the
//! session's output format, one chunk per frame clock tick.
//!
//! # Session flow
//!
//! 1. The client opens the socket and sends `session.start`.
//! 2. The server admits the session (or replies `error` with
//!    [`ErrorCode::SessionLimit`] and closes) and replies `session.started`.
//! 3. Both sides stream audio chunks; the server also sends transcript, tool
//!    and turn events. The client may send `text.input`, `tool.result` and
//!    `session.cancel_tool` at any time.
//! 4. The client sends `session.end` (or just closes); the server flushes and
//!    closes the socket.
//!
//! `session.started` is the one message beyond the plan's list: it tells the
//! client the session id and the audio parameters it will receive.

use std::fmt;

use s2s_core::streams::{FRAME_SAMPLES, SAMPLE_RATE};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Which of the model's three endpoints a session uses (plan §5).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Full-duplex speech-to-speech (default).
    #[default]
    S2s,
    /// Speech-to-text only: the server sends transcripts and no audio.
    Stt,
    /// Text-to-speech only: client audio is ignored; text comes from
    /// `text.input`.
    Tts,
}

/// Audio encoding of a binary frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioFormat {
    /// One Opus packet per chunk.
    Opus,
    /// Little-endian signed 16-bit PCM.
    PcmS16le,
    /// Little-endian 32-bit float PCM.
    PcmF32le,
}

impl AudioFormat {
    /// The binary frame tag byte for this format.
    pub fn tag(self) -> u8 {
        match self {
            AudioFormat::Opus => 0x01,
            AudioFormat::PcmS16le => 0x02,
            AudioFormat::PcmF32le => 0x03,
        }
    }

    /// The format for a tag byte, if it is a known one.
    pub fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0x01 => Some(AudioFormat::Opus),
            0x02 => Some(AudioFormat::PcmS16le),
            0x03 => Some(AudioFormat::PcmF32le),
            _ => None,
        }
    }

    /// Parses the snake_case name used in JSON (`opus`, `pcm_s16le`,
    /// `pcm_f32le`).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "opus" => Some(AudioFormat::Opus),
            "pcm_s16le" => Some(AudioFormat::PcmS16le),
            "pcm_f32le" => Some(AudioFormat::PcmF32le),
            _ => None,
        }
    }

    /// The snake_case name used in JSON.
    pub fn name(self) -> &'static str {
        match self {
            AudioFormat::Opus => "opus",
            AudioFormat::PcmS16le => "pcm_s16le",
            AudioFormat::PcmF32le => "pcm_f32le",
        }
    }
}

impl fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One binary audio frame: a format tag and its payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioChunk {
    /// How the payload is encoded.
    pub format: AudioFormat,
    /// Encoded audio (one Opus packet, or raw PCM samples).
    pub payload: Vec<u8>,
}

/// Why a binary frame could not be parsed.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FrameError {
    /// The frame had no bytes at all.
    #[error("empty binary frame")]
    Empty,
    /// The first byte is not a known format tag.
    #[error("unknown audio format tag 0x{0:02x}")]
    UnknownTag(u8),
    /// The payload is not a whole number of PCM samples.
    #[error("{format} payload of {len} bytes is not a whole number of samples")]
    PartialSample {
        /// The PCM format.
        format: AudioFormat,
        /// Payload length in bytes.
        len: usize,
    },
}

impl AudioChunk {
    /// Serializes to the binary frame layout: tag byte, then payload.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + self.payload.len());
        out.push(self.format.tag());
        out.extend_from_slice(&self.payload);
        out
    }

    /// Parses a binary frame, checking that PCM payloads hold whole samples.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FrameError> {
        let (&tag, payload) = bytes.split_first().ok_or(FrameError::Empty)?;
        let format = AudioFormat::from_tag(tag).ok_or(FrameError::UnknownTag(tag))?;
        let sample_bytes = match format {
            AudioFormat::Opus => 1,
            AudioFormat::PcmS16le => 2,
            AudioFormat::PcmF32le => 4,
        };
        if payload.len() % sample_bytes != 0 {
            return Err(FrameError::PartialSample {
                format,
                len: payload.len(),
            });
        }
        Ok(Self {
            format,
            payload: payload.to_vec(),
        })
    }
}

/// Body of `session.start`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionStart {
    /// System prompt, injected as a `<sys>` segment before the first frame.
    #[serde(default)]
    pub system_prompt: String,
    /// Tool declarations (JSON schemas), passed through to the engine
    /// unchanged.
    #[serde(default)]
    pub tools: Vec<Value>,
    /// Voice name, if the engine supports several.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// Endpoint mode (default `s2s`).
    #[serde(default)]
    pub mode: Mode,
    /// Format of server → client audio chunks. Defaults to the server's
    /// configured format. (Client → server chunks are self-describing.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<AudioFormat>,
}

/// A JSON control message from the client.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    /// `session.start {system_prompt, tools[], voice, mode}`. Must be the
    /// first message on the socket.
    #[serde(rename = "session.start")]
    SessionStart(SessionStart),
    /// `text.input {text}`: TTS input, or a typed user turn.
    #[serde(rename = "text.input")]
    TextInput {
        /// The text.
        text: String,
    },
    /// `tool.result {call_id, json}`: the result of a client-side tool.
    #[serde(rename = "tool.result")]
    ToolResult {
        /// The id from the matching `tool.call`.
        call_id: String,
        /// The tool's result.
        json: Value,
    },
    /// `session.cancel_tool {call_id}`: abandon an in-flight call.
    #[serde(rename = "session.cancel_tool")]
    CancelTool {
        /// The id from the matching `tool.call`.
        call_id: String,
    },
    /// `session.end`: finish the session and close the socket.
    #[serde(rename = "session.end")]
    SessionEnd,
}

/// Lifecycle state of a tool call, reported by `tool.status`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolState {
    /// The call was parsed and handed to the executor (or the client).
    Dispatched,
    /// The executor reports it is running.
    Running,
    /// The result was injected into the model's context.
    Injected,
    /// The call was cancelled (by the model, the user or the runtime).
    Cancelled,
    /// The call failed or timed out.
    Failed,
}

/// Turn-taking events, reported by `turn.event`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnEvent {
    /// The user started speaking.
    UserStarted,
    /// The user stopped speaking.
    UserStopped,
    /// The assistant started speaking.
    AssistantStarted,
    /// The assistant stopped speaking.
    AssistantStopped,
}

/// Machine-readable error category of an `error` message.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// A message could not be parsed.
    BadMessage,
    /// A message was valid but not allowed now (e.g. audio before
    /// `session.start`, or a second `session.start`).
    ProtocolViolation,
    /// The server is at its session limit; try again later.
    SessionLimit,
    /// No `session.start` arrived in time.
    Timeout,
    /// The client sent control messages faster than the engine consumes them;
    /// the message was dropped.
    RateLimited,
    /// The client did not read server messages fast enough and was
    /// disconnected.
    SlowClient,
    /// Audio could not be decoded.
    BadAudio,
    /// The engine rejected the session or failed.
    EngineFailure,
    /// The server is shutting down.
    ShuttingDown,
    /// Anything else.
    Internal,
}

/// A JSON control message from the server.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    /// Reply to `session.start` once the session has joined the frame clock.
    #[serde(rename = "session.started")]
    SessionStarted {
        /// Server-assigned session id.
        session_id: u64,
        /// Mode in effect.
        mode: Mode,
        /// Audio sample rate (24000).
        sample_rate: u32,
        /// Samples per frame (1920 = 80 ms).
        frame_samples: usize,
        /// Format of server → client audio chunks.
        output_format: AudioFormat,
    },
    /// `transcript.user.delta {text, t}`: new user transcript text (STT).
    #[serde(rename = "transcript.user.delta")]
    UserTranscriptDelta {
        /// New text (append to what came before).
        text: String,
        /// Session time in seconds at the end of the frame that produced it.
        t: f64,
    },
    /// `transcript.assistant.delta {text, t}`: new assistant text.
    #[serde(rename = "transcript.assistant.delta")]
    AssistantTranscriptDelta {
        /// New text (append to what came before).
        text: String,
        /// Session time in seconds at the end of the frame that produced it.
        t: f64,
    },
    /// `tool.call {call_id, name, arguments}`.
    #[serde(rename = "tool.call")]
    ToolCall {
        /// Id to quote in `tool.result` / `session.cancel_tool`.
        call_id: String,
        /// Tool name.
        name: String,
        /// Arguments (valid JSON matching the tool's schema).
        arguments: Value,
    },
    /// `tool.status {call_id, state}`.
    #[serde(rename = "tool.status")]
    ToolStatus {
        /// The call.
        call_id: String,
        /// Its new state.
        state: ToolState,
    },
    /// `turn.event {event, t}`.
    #[serde(rename = "turn.event")]
    TurnEvent {
        /// What happened.
        event: TurnEvent,
        /// Session time in seconds at the end of the frame that produced it.
        t: f64,
    },
    /// `error {code, message}`.
    #[serde(rename = "error")]
    Error {
        /// Category.
        code: ErrorCode,
        /// Human-readable detail.
        message: String,
    },
}

impl ServerMessage {
    /// Shorthand for an [`ServerMessage::Error`].
    pub fn error(code: ErrorCode, message: impl Into<String>) -> Self {
        ServerMessage::Error {
            code,
            message: message.into(),
        }
    }

    /// `session.started` for a session.
    pub fn started(session_id: u64, mode: Mode, output_format: AudioFormat) -> Self {
        ServerMessage::SessionStarted {
            session_id,
            mode,
            sample_rate: SAMPLE_RATE,
            frame_samples: FRAME_SAMPLES,
            output_format,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn round_trip_client(m: ClientMessage) {
        let s = serde_json::to_string(&m).unwrap();
        let back: ClientMessage = serde_json::from_str(&s).unwrap();
        assert_eq!(back, m, "{s}");
    }

    fn round_trip_server(m: ServerMessage) {
        let s = serde_json::to_string(&m).unwrap();
        let back: ServerMessage = serde_json::from_str(&s).unwrap();
        assert_eq!(back, m, "{s}");
    }

    #[test]
    fn client_messages_round_trip() {
        round_trip_client(ClientMessage::SessionStart(SessionStart {
            system_prompt: "You are helpful.".into(),
            tools: vec![json!({"name": "get_weather", "parameters": {"type": "object"}})],
            voice: Some("alto".into()),
            mode: Mode::Tts,
            output_format: Some(AudioFormat::PcmS16le),
        }));
        round_trip_client(ClientMessage::SessionStart(SessionStart::default()));
        round_trip_client(ClientMessage::TextInput { text: "hi".into() });
        round_trip_client(ClientMessage::ToolResult {
            call_id: "c_1".into(),
            json: json!({"temp": 18}),
        });
        round_trip_client(ClientMessage::CancelTool {
            call_id: "c_1".into(),
        });
        round_trip_client(ClientMessage::SessionEnd);
    }

    #[test]
    fn server_messages_round_trip() {
        round_trip_server(ServerMessage::started(7, Mode::S2s, AudioFormat::Opus));
        round_trip_server(ServerMessage::UserTranscriptDelta {
            text: "hello".into(),
            t: 1.28,
        });
        round_trip_server(ServerMessage::AssistantTranscriptDelta {
            text: "hi".into(),
            t: 1.36,
        });
        round_trip_server(ServerMessage::ToolCall {
            call_id: "c_1".into(),
            name: "f".into(),
            arguments: json!({"a": [1, 2]}),
        });
        for state in [
            ToolState::Dispatched,
            ToolState::Running,
            ToolState::Injected,
            ToolState::Cancelled,
            ToolState::Failed,
        ] {
            round_trip_server(ServerMessage::ToolStatus {
                call_id: "c_1".into(),
                state,
            });
        }
        for event in [
            TurnEvent::UserStarted,
            TurnEvent::UserStopped,
            TurnEvent::AssistantStarted,
            TurnEvent::AssistantStopped,
        ] {
            round_trip_server(ServerMessage::TurnEvent { event, t: 0.08 });
        }
        round_trip_server(ServerMessage::error(ErrorCode::SessionLimit, "full"));
    }

    #[test]
    fn wire_names_match_the_plan() {
        let v = serde_json::to_value(ClientMessage::CancelTool {
            call_id: "c".into(),
        })
        .unwrap();
        assert_eq!(v, json!({"type": "session.cancel_tool", "call_id": "c"}));
        let v = serde_json::to_value(ClientMessage::SessionEnd).unwrap();
        assert_eq!(v, json!({"type": "session.end"}));
        let v = serde_json::to_value(ServerMessage::TurnEvent {
            event: TurnEvent::AssistantStopped,
            t: 2.0,
        })
        .unwrap();
        assert_eq!(
            v,
            json!({"type": "turn.event", "event": "assistant_stopped", "t": 2.0})
        );
        let v = serde_json::to_value(ServerMessage::error(ErrorCode::SlowClient, "x")).unwrap();
        assert_eq!(
            v,
            json!({"type": "error", "code": "slow_client", "message": "x"})
        );
    }

    #[test]
    fn session_start_defaults() {
        let m: ClientMessage = serde_json::from_str(r#"{"type":"session.start"}"#).unwrap();
        assert_eq!(m, ClientMessage::SessionStart(SessionStart::default()));
        let m: ClientMessage = serde_json::from_str(
            r#"{"type":"session.start","system_prompt":"p","tools":[{"name":"x"}],"voice":"v","mode":"stt","output_format":"pcm_f32le"}"#,
        )
        .unwrap();
        let ClientMessage::SessionStart(s) = m else {
            panic!("wrong variant")
        };
        assert_eq!(s.mode, Mode::Stt);
        assert_eq!(s.output_format, Some(AudioFormat::PcmF32le));
        assert_eq!(s.tools.len(), 1);
        assert!(serde_json::from_str::<ClientMessage>(r#"{"type":"bogus"}"#).is_err());
        assert!(
            serde_json::from_str::<ClientMessage>(r#"{"type":"session.start","mode":"x"}"#)
                .is_err()
        );
    }

    #[test]
    fn appendix_c_examples_parse() {
        let server = [
            r#"{"type":"transcript.user.delta","text":"what's the weather in Paris","t":2.48}"#,
            r#"{"type":"transcript.assistant.delta","text":"Let me check.","t":2.64}"#,
            r#"{"type":"tool.call","call_id":"c_17","name":"get_weather","arguments":{"city":"Paris","unit":"celsius"}}"#,
            r#"{"type":"tool.status","call_id":"c_17","state":"injected"}"#,
            r#"{"type":"transcript.assistant.delta","text":"It's 18 degrees and sunny in Paris.","t":4.10}"#,
        ];
        for s in server {
            let m: ServerMessage = serde_json::from_str(s).unwrap();
            let v: Value = serde_json::from_str(s).unwrap();
            assert_eq!(serde_json::to_value(&m).unwrap(), v);
        }
        let m: ClientMessage = serde_json::from_str(
            r#"{"type":"tool.result","call_id":"c_17","json":{"temp":18,"conditions":"sunny"}}"#,
        )
        .unwrap();
        assert_eq!(
            m,
            ClientMessage::ToolResult {
                call_id: "c_17".into(),
                json: json!({"temp": 18, "conditions": "sunny"})
            }
        );
    }

    #[test]
    fn audio_chunk_framing() {
        for format in [
            AudioFormat::Opus,
            AudioFormat::PcmS16le,
            AudioFormat::PcmF32le,
        ] {
            assert_eq!(AudioFormat::from_tag(format.tag()), Some(format));
            assert_eq!(AudioFormat::from_name(format.name()), Some(format));
            let json = serde_json::to_value(format).unwrap();
            assert_eq!(json, Value::String(format.name().into()));
            let chunk = AudioChunk {
                format,
                payload: vec![1, 2, 3, 4, 5, 6, 7, 8],
            };
            let bytes = chunk.to_bytes();
            assert_eq!(bytes[0], format.tag());
            assert_eq!(bytes.len(), 9);
            assert_eq!(AudioChunk::from_bytes(&bytes).unwrap(), chunk);
        }
        assert_eq!(AudioChunk::from_bytes(&[]), Err(FrameError::Empty));
        assert_eq!(
            AudioChunk::from_bytes(&[0x7f, 0]),
            Err(FrameError::UnknownTag(0x7f))
        );
        assert_eq!(
            AudioChunk::from_bytes(&[0x02, 0, 0, 0]),
            Err(FrameError::PartialSample {
                format: AudioFormat::PcmS16le,
                len: 3
            })
        );
        assert!(AudioChunk::from_bytes(&[0x03, 0, 0]).is_err());
        // A tag with no payload is a valid (empty) chunk.
        assert!(AudioChunk::from_bytes(&[0x01]).unwrap().payload.is_empty());
    }
}
