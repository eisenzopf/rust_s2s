//! Streaming media server for rust_s2s (plan §2, §10).
//!
//! The server sits between clients and the model runtime:
//!
//! ```text
//! client ──WebSocket──▶ decode (Opus/PCM) ──▶ jitter buffer ─┐
//!                                                            ▼
//!                                        frame clock (80 ms, all sessions)
//!                                                            │ Engine::step(batch)
//! client ◀─WebSocket── encode (Opus/PCM) ◀── outbound queue ◀┘
//! ```
//!
//! - [`protocol`]: JSON control messages and the binary audio framing.
//! - [`codec`]: Opus and raw PCM at the edge (24 kHz mono `f32` inside).
//! - [`jitter`]: assembles exact 1920-sample frames, filling gaps with
//!   silence.
//! - [`engine`]: the [`Engine`] trait the model runtime implements, stepped
//!   once per frame for the whole batch of sessions.
//! - [`fake`]: [`FakeEngine`], an echo/tone engine with synthetic transcript,
//!   turn and tool events, for development and tests.
//! - [`session`]: admission control, per-session bounded queues and their
//!   backpressure policy.
//! - [`clock`]: the lockstep frame clock.
//! - [`server`]: the axum WebSocket server, [`ServerConfig`] and [`start`].
//!
//! The `s2s-server` binary runs the server with the [`FakeEngine`] and
//! serves a small browser client at `/`.

pub mod clock;
pub mod codec;
pub mod engine;
pub mod fake;
pub mod jitter;
pub mod protocol;
pub mod server;
pub mod session;

pub use engine::{Engine, EngineError, EngineEvent, FrameInput, FrameOutput, SessionConfig};
pub use fake::{FakeEngine, FakeEngineConfig};
pub use protocol::{AudioChunk, AudioFormat, ClientMessage, Mode, ServerMessage};
pub use server::{start, RunningServer, ServerConfig};
