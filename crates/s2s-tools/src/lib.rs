//! Tool-calling runtime for the rust_s2s full-duplex speech model (plan §4).
//!
//! The model writes tool calls into its assistant text stream (`Tm`) while it
//! keeps listening and talking:
//!
//! ```text
//! Let me check.<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>
//! ```
//!
//! This crate is everything between that text and the
//! `<tool_result>{…}</tool_result>` injection the model reads a moment later.
//! It is CPU-only and model-agnostic.
//!
//! # Flow
//!
//! ```text
//!   session.start {system_prompt, tools[]}
//!        │
//!        ▼
//!   ToolRegistry ──► system_segment(renderer)  ──► inject "<sys>…</sys>" once
//!        │
//!        ├──► ToolCallGrammar (llguidance) ──► ToolCallConstraint
//!        │       per Tm step inside a span:  mask ─► apply_token_bitmask ─► sample ─► observe
//!        │
//!   model text ──► ToolCallParser ──► ParseEvent::Text           ──► transcript.assistant.delta
//!                                 ├─► ParseEvent::ToolCall        ──► Executor::dispatch
//!                                 └─► ParseEvent::CancelRequested ──► Executor::cancel_all
//!                                                                         │
//!   Executor (tokio) ── server tools: async handlers, timeout, cancel ────┤
//!                    └─ client tools: ExecutorEvent::ClientCall ──► tool.call to client
//!                                     tool.result ──► submit_client_result
//!                                                                         ▼
//!   ExecutorEvent::Finished(ToolOutcome) ──► InjectionFormatter ──► "<tool_result>{…}</tool_result>"
//!                                                                   injected before the next frame
//! ```
//!
//! 1. **Declare** — [`ToolRegistry`] holds the session's [`ToolSpec`]s with
//!    compiled JSON-Schema validators. A [`ToolPromptRenderer`] (default:
//!    [`DefaultPromptRenderer`]) renders them into the `<sys>` segment.
//! 2. **Constrain** — [`ToolCallGrammar`] compiles one JSON Schema covering all
//!    tools with `llguidance`. While a span is open, [`ToolCallConstraint`]
//!    yields a per-step token bitmask (the format of
//!    `s2s_kernels::reference::sampling::apply_token_bitmask`), so sampled
//!    calls always parse and match a declared schema. Tokenizers plug in
//!    through [`TokenizerInfo`].
//! 3. **Parse** — [`ToolCallParser`] is fed the assistant text as it is
//!    emitted, passes ordinary text through, and turns spans into validated
//!    [`ToolCall`]s with session-unique [`CallId`]s (`c_1`, `c_2`, …).
//! 4. **Execute** — [`Executor`] runs calls concurrently on tokio with
//!    per-call timeouts and cancellation by id, and bridges client-side tools
//!    (`tool.call` out, `tool.result` in). Each call ends in exactly one
//!    [`ToolOutcome`].
//! 5. **Inject** — [`InjectionFormatter`] turns an outcome into
//!    `<tool_result>{"call_id": …, "status": …, …}</tool_result>`, truncated to
//!    a token/char budget (default 512 tokens), with stub forms for errors,
//!    timeouts and cancellations.
//!
//! # Example
//!
//! ```
//! use std::sync::Arc;
//! use serde_json::{json, Value};
//! use s2s_tools::*;
//! use s2s_kernels::reference::sampling::{apply_token_bitmask, sample, SamplingParams};
//!
//! // 1. Declare tools.
//! let registry = Arc::new(ToolRegistry::from_specs([ToolSpec::new(
//!     "get_weather",
//!     "Current weather for a city.",
//!     json!({
//!         "type": "object",
//!         "properties": {"city": {"type": "string", "maxLength": 20}},
//!         "required": ["city"],
//!         "additionalProperties": false
//!     }),
//! )]).unwrap());
//! let sys = registry.system_segment(&DefaultPromptRenderer, "You are a voice assistant.");
//! assert!(sys.starts_with("<sys>") && sys.contains("## get_weather"));
//!
//! // 2. Grammar-constrained decoding of one span (a real model supplies the
//! //    logits; here a fixed preference stands in for it).
//! let tokenizer = Arc::new(ByteTokenizer::with_json_merges());
//! let grammar = Arc::new(
//!     ToolCallGrammar::new(&registry, tokenizer.clone(), GrammarConfig::default()).unwrap(),
//! );
//! let ids = grammar.special_ids();
//! let mut constraint = ToolCallConstraint::new(grammar.clone());
//! constraint.observe(ids.tool_call).unwrap(); // the model sampled <tool_call>
//! let mut span = Vec::new();
//! while constraint.in_tool_call() {
//!     let mask = constraint.mask().unwrap().expect("inside a span");
//!     let mut logits: Vec<f32> = (0..grammar.vocab_size())
//!         .map(|t| if t == usize::from(b'P') { 1.0 } else { 0.0 })
//!         .collect();
//!     apply_token_bitmask(&mut logits, &mask);
//!     let t = sample(&logits, &SamplingParams::GREEDY, 0.0).unwrap() as u32;
//!     constraint.observe(t).unwrap();
//!     span.push(t);
//! }
//! let text = format!("Let me check.<tool_call>{}", tokenizer.decode(&span));
//!
//! // 3. Parse the assistant text stream (in any chunking).
//! let mut parser = ToolCallParser::new(registry.clone());
//! let mut calls = Vec::new();
//! for chunk in text.as_bytes().chunks(7) {
//!     for event in parser.push_bytes(chunk) {
//!         match event {
//!             ParseEvent::Text(t) => print!("{t}"), // transcript.assistant.delta
//!             ParseEvent::ToolCall(call) => calls.push(call),
//!             _ => {}
//!         }
//!     }
//! }
//! assert_eq!(calls.len(), 1);
//! assert_eq!(calls[0].call_id, CallId::from("c_1"));
//! assert_eq!(calls[0].name, "get_weather");
//!
//! // 4. Execute, 5. inject.
//! let rt = tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
//! rt.block_on(async {
//!     let (executor, mut events) = Executor::new(registry.clone(), ExecutorConfig::default());
//!     executor
//!         .register_handler("get_weather", |args: Value| async move {
//!             Ok(json!({"city": args["city"], "temp": 18, "conditions": "sunny"}))
//!         })
//!         .unwrap();
//!     executor.dispatch(calls[0].clone()).unwrap();
//!     let formatter = InjectionFormatter::new(InjectionConfig::default())
//!         .with_tokenizer(tokenizer.clone());
//!     while let Some(event) = events.recv().await {
//!         if let ExecutorEvent::Finished(outcome) = event {
//!             let segment = formatter.format(&outcome).unwrap();
//!             assert!(segment.text.starts_with(
//!                 r#"<tool_result>{"call_id":"c_1","name":"get_weather","status":"ok","#
//!             ));
//!             break;
//!         }
//!     }
//! });
//! ```

#![forbid(unsafe_code)]

mod executor;
mod grammar;
mod inject;
mod parser;
mod spec;
pub mod tokenizer;

pub use executor::{
    DispatchError, Executor, ExecutorConfig, ExecutorEvent, HandlerFuture, RegisterError,
    SubmitError, ToolHandler, ToolOutcome, ToolStatus,
};
pub use grammar::{
    render_tool_call, tool_call_schema, GrammarConfig, GrammarError, JsonStyle, SpanState,
    SpecialIds, ToolCallConstraint, ToolCallGrammar, ToolCallMatcher,
};
pub use inject::{
    CancelledPolicy, InjectionConfig, InjectionFormatter, InjectionSegment, ResultBudget,
};
pub use parser::{
    CallId, Marker, ParseEvent, ParserConfig, ToolCall, ToolCallError, ToolCallParser,
};
pub use spec::{
    ArgumentError, DefaultPromptRenderer, RegistryError, ToolLocation, ToolPromptRenderer,
    ToolRegistry, ToolSpec,
};
pub use tokenizer::{ByteTokenizer, TokenPiece, TokenizerError, TokenizerInfo};

#[cfg(test)]
mod tests {
    use super::*;

    fn send_sync<T: Send + Sync>() {}
    fn send<T: Send>() {}

    #[test]
    fn thread_safety() {
        send_sync::<ToolRegistry>();
        send_sync::<ToolCallGrammar>();
        send_sync::<Executor>();
        send_sync::<InjectionFormatter>();
        send_sync::<ToolCallParser>();
        send::<ToolCallMatcher>();
        send::<ToolCallConstraint>();
    }
}
