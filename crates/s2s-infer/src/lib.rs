//! Streaming inference engine (plan §10.1): runs [`s2s_duplex::DuplexModel`]
//! frame by frame behind the [`s2s_server::Engine`] trait.
//!
//! Per session and per 80 ms frame, [`DuplexEngine`]:
//! 1. encodes the user's PCM frame with the session's [`FrameCodec`] and
//!    builds the user-audio part of the row in the model's **delayed** domain
//!    (acoustic codebooks lag by `acoustic_delay`);
//! 2. samples the next row's generated tokens. Assistant text slots go
//!    through the tool-call grammar mask while a `<tool_call>` span is open;
//!    STT and TTS modes force the streams they don't use;
//! 3. turns the tokens into events: user transcript (`Tu`), assistant text
//!    and parsed tool calls (`Tm`, through `s2s_tools::ToolCallParser`);
//! 4. undelays the assistant audio codes and decodes one PCM frame;
//! 5. feeds the full row back into the model, then any tool results that
//!    arrived (formatted by `s2s_tools::InjectionFormatter`) as injection
//!    segments, so the next row is predicted from them, exactly as the
//!    training data is laid out (`s2s-data`).
//!
//! Tool calls go to a per-session `s2s_tools::Executor`: server-side tools
//! registered with [`DuplexEngine::register_tool`] run there, client-side
//! ones become `tool.call` events and complete when the client's
//! `tool.result` arrives.
//!
//! The codec is pluggable: [`ToneCodec`] is a deterministic stand-in for
//! development and tests; [`MimiCodec`] wraps the Mimi codec (`s2s-mimi`) for
//! real use.

#![forbid(unsafe_code)]

mod codec;
mod engine;
pub mod tokenizer;

pub use codec::{FrameCodec, MimiCodec, ToneCodec};
pub use engine::{DuplexEngine, EngineConfig};
pub use tokenizer::Tokenizer;
