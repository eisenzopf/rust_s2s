//! Incremental parser for the assistant text stream: splits transcript text
//! from `<tool_call>` … `</tool_call>` spans and `<tool_cancel>` markers.

use std::sync::Arc;

use s2s_core::streams::special;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::spec::{ArgumentError, ToolRegistry};

/// Identifier of one tool call within a session (e.g. `c_17`). Results are
/// matched to calls by this id (plan §4.2, "parallel calls").
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CallId(pub String);

impl CallId {
    /// The id as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CallId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for CallId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for CallId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// A parsed, schema-valid tool call. Serializes to the body of the
/// `tool.call {call_id, name, arguments}` server message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Session-unique id assigned by the parser.
    pub call_id: CallId,
    /// Tool name (registered).
    pub name: String,
    /// Arguments, valid against the tool's schema.
    pub arguments: Value,
}

/// Why a closed span did not yield a [`ToolCall`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolCallError {
    /// The span is not valid JSON (or not UTF-8).
    Json(String),
    /// Valid JSON, but not `{"name": "<string>", "arguments": {…}}`.
    Shape(String),
    /// The tool name is not registered.
    UnknownTool(String),
    /// The arguments violate the tool's schema.
    InvalidArguments(Vec<String>),
    /// The span exceeded [`ParserConfig::max_span_bytes`].
    TooLong,
    /// The stream ended (or a new `<tool_call>` began) before `</tool_call>`.
    Unterminated,
}

impl std::fmt::Display for ToolCallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(e) => write!(f, "tool call is not valid JSON: {e}"),
            Self::Shape(e) => write!(f, "malformed tool call: {e}"),
            Self::UnknownTool(n) => write!(f, "unknown tool {n:?}"),
            Self::InvalidArguments(e) => write!(f, "invalid arguments: {}", e.join("; ")),
            Self::TooLong => f.write_str("tool call too long"),
            Self::Unterminated => f.write_str("unterminated tool call"),
        }
    }
}

impl std::error::Error for ToolCallError {}

/// Output of [`ToolCallParser`].
#[derive(Clone, Debug, PartialEq)]
pub enum ParseEvent {
    /// Ordinary transcript text (outside any span). `<pad>`/`<epad>` are
    /// dropped.
    Text(String),
    /// `<tool_call>` opened a span; the runtime should switch the sampler to
    /// grammar-constrained mode.
    ToolCallStarted {
        /// Id the call will carry.
        call_id: CallId,
    },
    /// `</tool_call>` closed a span holding a valid call: dispatch it.
    ToolCall(ToolCall),
    /// A span closed (or was cut off) without a valid call.
    InvalidToolCall {
        /// Id assigned when the span opened.
        call_id: CallId,
        /// Span content as written (lossy UTF-8).
        raw: String,
        /// What was wrong.
        error: ToolCallError,
    },
    /// `<tool_cancel>` inside an open span: the model abandoned the call it
    /// was writing. Nothing was dispatched.
    ToolCallAborted {
        /// Id of the abandoned span.
        call_id: CallId,
    },
    /// `<tool_cancel>` outside a span: the model asks to cancel its in-flight
    /// calls (typically [`Executor::cancel_all`](crate::Executor::cancel_all)).
    CancelRequested,
}

/// A special token of the assistant text stream, for callers that feed the
/// parser token by token and know which ids are special
/// ([`ToolCallParser::push_marker`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Marker {
    /// `<tool_call>`.
    ToolCall,
    /// `</tool_call>`.
    ToolCallEnd,
    /// `<tool_cancel>`.
    ToolCancel,
    /// `<pad>` (no text).
    Pad,
    /// `<epad>` (end of padding).
    Epad,
}

impl Marker {
    /// All markers the parser recognizes in text.
    pub const ALL: [Marker; 5] = [
        Marker::ToolCall,
        Marker::ToolCallEnd,
        Marker::ToolCancel,
        Marker::Pad,
        Marker::Epad,
    ];

    /// The token's text form.
    pub fn as_str(self) -> &'static str {
        match self {
            Marker::ToolCall => special::TOOL_CALL,
            Marker::ToolCallEnd => special::TOOL_CALL_END,
            Marker::ToolCancel => special::TOOL_CANCEL,
            Marker::Pad => special::PAD,
            Marker::Epad => special::EPAD,
        }
    }

    /// The marker spelled `name`, if any.
    pub fn from_name(name: &str) -> Option<Marker> {
        Marker::ALL.into_iter().find(|m| m.as_str() == name)
    }
}

/// [`ToolCallParser`] settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParserConfig {
    /// Prefix of generated call ids (`c_` gives `c_1`, `c_2`, …).
    pub call_id_prefix: String,
    /// Recognize markers spelled out inside text passed to
    /// [`push_str`](ToolCallParser::push_str)/[`push_bytes`](ToolCallParser::push_bytes).
    /// Token-level callers that report special tokens with
    /// [`push_marker`](ToolCallParser::push_marker) can turn this off, so a
    /// JSON string that happens to contain `</tool_call>` cannot end a span.
    pub scan_text_markers: bool,
    /// Longest span accepted, in bytes; longer spans are reported as
    /// [`ToolCallError::TooLong`] and their remainder is skipped.
    pub max_span_bytes: usize,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            call_id_prefix: "c_".into(),
            scan_text_markers: true,
            max_span_bytes: 16 * 1024,
        }
    }
}

#[derive(Debug)]
enum State {
    Text,
    Span {
        call_id: CallId,
        buf: Vec<u8>,
    },
    /// Inside an over-long span, dropping content until it ends.
    Skipping,
}

/// Incremental parser for the assistant text stream (plan §4.2).
///
/// Feed it text as the model emits it, in arbitrarily split chunks (markers
/// and multi-byte UTF-8 characters may straddle chunks). It returns
/// [`ParseEvent`]s: transcript text passes through, `<tool_call>{…}</tool_call>`
/// spans become [`ToolCall`]s after JSON parsing and schema validation, and
/// `<tool_cancel>` becomes an abort or cancel request.
///
/// Call ids are assigned when a span opens, in order: `c_1`, `c_2`, ….
#[derive(Debug)]
pub struct ToolCallParser {
    registry: Arc<ToolRegistry>,
    config: ParserConfig,
    state: State,
    /// Bytes not yet classified: a possible marker prefix at the end.
    pending: Vec<u8>,
    /// Transcript bytes not yet emitted (may end in a partial UTF-8 char).
    text: Vec<u8>,
    next_id: u64,
}

impl ToolCallParser {
    /// A parser validating calls against `registry`, with default settings.
    pub fn new(registry: Arc<ToolRegistry>) -> Self {
        Self::with_config(registry, ParserConfig::default())
    }

    /// A parser with explicit settings.
    pub fn with_config(registry: Arc<ToolRegistry>, config: ParserConfig) -> Self {
        Self {
            registry,
            config,
            state: State::Text,
            pending: Vec::new(),
            text: Vec::new(),
            next_id: 1,
        }
    }

    /// Whether a `<tool_call>` span is open.
    pub fn in_tool_call(&self) -> bool {
        !matches!(self.state, State::Text)
    }

    /// Feed text.
    pub fn push_str(&mut self, s: &str) -> Vec<ParseEvent> {
        self.push_bytes(s.as_bytes())
    }

    /// Feed raw bytes (e.g. a byte-level BPE token's bytes, which may split a
    /// UTF-8 character).
    pub fn push_bytes(&mut self, bytes: &[u8]) -> Vec<ParseEvent> {
        let mut ev = Vec::new();
        if !self.config.scan_text_markers {
            self.content(bytes, &mut ev);
            self.flush_text(false, &mut ev);
            return ev;
        }
        self.pending.extend_from_slice(bytes);
        let pending = std::mem::take(&mut self.pending);
        let mut i = 0;
        while i < pending.len() {
            let Some(off) = pending[i..].iter().position(|&b| b == b'<') else {
                self.content(&pending[i..], &mut ev);
                i = pending.len();
                break;
            };
            self.content(&pending[i..i + off], &mut ev);
            i += off;
            let rest = &pending[i..];
            if let Some(m) = Marker::ALL
                .into_iter()
                .find(|m| rest.starts_with(m.as_str().as_bytes()))
            {
                i += m.as_str().len();
                self.marker(m, &mut ev);
            } else if Marker::ALL
                .iter()
                .any(|m| m.as_str().as_bytes().starts_with(rest))
            {
                // Could still become a marker: wait for more input.
                break;
            } else {
                self.content(b"<", &mut ev);
                i += 1;
            }
        }
        self.pending = pending[i..].to_vec();
        self.flush_text(false, &mut ev);
        ev
    }

    /// Report a special token directly (token-level callers).
    pub fn push_marker(&mut self, marker: Marker) -> Vec<ParseEvent> {
        let mut ev = Vec::new();
        // Anything held back as a possible marker prefix is plain content.
        let pending = std::mem::take(&mut self.pending);
        self.content(&pending, &mut ev);
        self.marker(marker, &mut ev);
        self.flush_text(false, &mut ev);
        ev
    }

    /// End of stream: flush held-back text and report an open span as
    /// [`ToolCallError::Unterminated`]. The parser can be reused afterwards.
    pub fn finish(&mut self) -> Vec<ParseEvent> {
        let mut ev = Vec::new();
        let pending = std::mem::take(&mut self.pending);
        self.content(&pending, &mut ev);
        self.flush_text(true, &mut ev);
        match std::mem::replace(&mut self.state, State::Text) {
            State::Span { call_id, buf } => ev.push(ParseEvent::InvalidToolCall {
                call_id,
                raw: String::from_utf8_lossy(&buf).into_owned(),
                error: ToolCallError::Unterminated,
            }),
            State::Skipping | State::Text => {}
        }
        ev
    }

    fn content(&mut self, bytes: &[u8], ev: &mut Vec<ParseEvent>) {
        if bytes.is_empty() {
            return;
        }
        match &mut self.state {
            State::Text => self.text.extend_from_slice(bytes),
            State::Span { buf, .. } => {
                buf.extend_from_slice(bytes);
                if buf.len() > self.config.max_span_bytes {
                    let State::Span { call_id, buf } =
                        std::mem::replace(&mut self.state, State::Skipping)
                    else {
                        unreachable!()
                    };
                    ev.push(ParseEvent::InvalidToolCall {
                        call_id,
                        raw: String::from_utf8_lossy(&buf).into_owned(),
                        error: ToolCallError::TooLong,
                    });
                }
            }
            State::Skipping => {}
        }
    }

    fn marker(&mut self, m: Marker, ev: &mut Vec<ParseEvent>) {
        match m {
            Marker::Pad | Marker::Epad => {}
            Marker::ToolCall => {
                self.flush_text(true, ev);
                if let State::Span { call_id, buf } =
                    std::mem::replace(&mut self.state, State::Text)
                {
                    ev.push(ParseEvent::InvalidToolCall {
                        call_id,
                        raw: String::from_utf8_lossy(&buf).into_owned(),
                        error: ToolCallError::Unterminated,
                    });
                }
                let call_id = CallId(format!("{}{}", self.config.call_id_prefix, self.next_id));
                self.next_id += 1;
                ev.push(ParseEvent::ToolCallStarted {
                    call_id: call_id.clone(),
                });
                self.state = State::Span {
                    call_id,
                    buf: Vec::new(),
                };
            }
            Marker::ToolCallEnd => match std::mem::replace(&mut self.state, State::Text) {
                State::Span { call_id, buf } => ev.push(self.close(call_id, buf)),
                // A stray `</tool_call>` in text carries no transcript.
                State::Text | State::Skipping => {}
            },
            Marker::ToolCancel => match std::mem::replace(&mut self.state, State::Text) {
                State::Span { call_id, .. } => ev.push(ParseEvent::ToolCallAborted { call_id }),
                State::Skipping => {}
                State::Text => {
                    self.flush_text(true, ev);
                    ev.push(ParseEvent::CancelRequested);
                }
            },
        }
    }

    fn close(&self, call_id: CallId, buf: Vec<u8>) -> ParseEvent {
        let raw = String::from_utf8_lossy(&buf).into_owned();
        match parse_call(&self.registry, &buf) {
            Ok((name, arguments)) => ParseEvent::ToolCall(ToolCall {
                call_id,
                name,
                arguments,
            }),
            Err(error) => ParseEvent::InvalidToolCall {
                call_id,
                raw,
                error,
            },
        }
    }

    /// Emit buffered transcript text. Unless `all`, a trailing incomplete
    /// UTF-8 sequence is kept for the next chunk.
    fn flush_text(&mut self, all: bool, ev: &mut Vec<ParseEvent>) {
        if self.text.is_empty() {
            return;
        }
        let mut out = String::new();
        let mut rest: &[u8] = &self.text;
        loop {
            match std::str::from_utf8(rest) {
                Ok(s) => {
                    out.push_str(s);
                    rest = &[];
                    break;
                }
                Err(e) => {
                    let (valid, tail) = rest.split_at(e.valid_up_to());
                    out.push_str(std::str::from_utf8(valid).unwrap_or_default());
                    match e.error_len() {
                        Some(n) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            rest = &tail[n..];
                        }
                        None if all => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            rest = &[];
                            break;
                        }
                        None => {
                            rest = tail;
                            break;
                        }
                    }
                }
            }
        }
        self.text = rest.to_vec();
        if !out.is_empty() {
            match ev.last_mut() {
                Some(ParseEvent::Text(t)) => t.push_str(&out),
                _ => ev.push(ParseEvent::Text(out)),
            }
        }
    }
}

/// Parse and validate the content of a span.
fn parse_call(registry: &ToolRegistry, buf: &[u8]) -> Result<(String, Value), ToolCallError> {
    let s = std::str::from_utf8(buf).map_err(|e| ToolCallError::Json(e.to_string()))?;
    let v: Value =
        serde_json::from_str(s.trim()).map_err(|e| ToolCallError::Json(e.to_string()))?;
    let Value::Object(mut obj) = v else {
        return Err(ToolCallError::Shape("expected a JSON object".into()));
    };
    let name = match obj.remove("name") {
        Some(Value::String(n)) => n,
        Some(_) => return Err(ToolCallError::Shape("\"name\" must be a string".into())),
        None => return Err(ToolCallError::Shape("missing \"name\"".into())),
    };
    let arguments = match obj.remove("arguments") {
        Some(a @ Value::Object(_)) => a,
        // Lenient: an omitted/null argument object is an empty one.
        None | Some(Value::Null) => Value::Object(Default::default()),
        Some(_) => {
            return Err(ToolCallError::Shape(
                "\"arguments\" must be an object".into(),
            ))
        }
    };
    match registry.validate_arguments(&name, &arguments) {
        Ok(()) => Ok((name, arguments)),
        Err(ArgumentError::UnknownTool(n)) => Err(ToolCallError::UnknownTool(n)),
        Err(ArgumentError::Invalid(e)) => Err(ToolCallError::InvalidArguments(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::ToolSpec;
    use serde_json::json;

    fn registry() -> Arc<ToolRegistry> {
        Arc::new(
            ToolRegistry::from_specs([ToolSpec::new(
                "get_weather",
                "",
                json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": false
                }),
            )])
            .unwrap(),
        )
    }

    /// Concatenate adjacent text events (chunking may split them).
    fn normalize(events: Vec<ParseEvent>) -> Vec<ParseEvent> {
        let mut out: Vec<ParseEvent> = Vec::new();
        for e in events {
            match (out.last_mut(), e) {
                (Some(ParseEvent::Text(a)), ParseEvent::Text(b)) => a.push_str(&b),
                (_, e) => out.push(e),
            }
        }
        out
    }

    fn parse_all(p: &mut ToolCallParser, chunks: &[&[u8]]) -> Vec<ParseEvent> {
        let mut ev = Vec::new();
        for c in chunks {
            ev.extend(p.push_bytes(c));
        }
        ev.extend(p.finish());
        normalize(ev)
    }

    const STREAM: &str = "<epad>Let me check.<tool_call>{\"name\": \"get_weather\", \
        \"arguments\": {\"city\": \"Paris\"}}</tool_call><pad><pad> One moment… <3";

    fn expected() -> Vec<ParseEvent> {
        vec![
            ParseEvent::Text("Let me check.".into()),
            ParseEvent::ToolCallStarted {
                call_id: "c_1".into(),
            },
            ParseEvent::ToolCall(ToolCall {
                call_id: "c_1".into(),
                name: "get_weather".into(),
                arguments: json!({"city": "Paris"}),
            }),
            ParseEvent::Text(" One moment… <3".into()),
        ]
    }

    #[test]
    fn whole_stream() {
        let mut p = ToolCallParser::new(registry());
        assert_eq!(parse_all(&mut p, &[STREAM.as_bytes()]), expected());
        assert!(!p.in_tool_call());
    }

    #[test]
    fn every_split_point_and_byte_by_byte() {
        let b = STREAM.as_bytes();
        for cut in 0..=b.len() {
            let mut p = ToolCallParser::new(registry());
            assert_eq!(
                parse_all(&mut p, &[&b[..cut], &b[cut..]]),
                expected(),
                "cut {cut}"
            );
        }
        let mut p = ToolCallParser::new(registry());
        let bytes: Vec<&[u8]> = b.chunks(1).collect();
        assert_eq!(parse_all(&mut p, &bytes), expected());
    }

    #[test]
    fn text_streams_without_waiting_for_spans() {
        let mut p = ToolCallParser::new(registry());
        assert_eq!(p.push_str("Hello "), [ParseEvent::Text("Hello ".into())]);
        // "<to" might be a marker: held back.
        assert_eq!(p.push_str("world <to"), [ParseEvent::Text("world ".into())]);
        // "<tox" cannot be: released.
        assert_eq!(p.push_str("x"), [ParseEvent::Text("<tox".into())]);
        // A partial UTF-8 char is held back too.
        let e = "é".as_bytes();
        assert_eq!(p.push_bytes(&e[..1]), []);
        assert_eq!(p.push_bytes(&e[1..]), [ParseEvent::Text("é".into())]);
        assert_eq!(p.push_str("<tool_call>").len(), 1);
        assert!(p.in_tool_call());
        assert_eq!(p.push_str("{\"name\""), []);
    }

    #[test]
    fn ids_increase_and_parallel_spans() {
        let mut p = ToolCallParser::new(registry());
        let call =
            "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"X\"}}</tool_call>";
        let ev = parse_all(&mut p, &[call.as_bytes(), b"and ", call.as_bytes()]);
        let ids: Vec<_> = ev
            .iter()
            .filter_map(|e| match e {
                ParseEvent::ToolCall(c) => Some(c.call_id.as_str().to_string()),
                _ => None,
            })
            .collect();
        assert_eq!(ids, ["c_1", "c_2"]);
    }

    type ErrorCheck = fn(&ToolCallError) -> bool;

    #[test]
    fn invalid_calls_are_reported() {
        let cases: [(&str, ErrorCheck); 6] = [
            ("{\"name\": \"get_weather\", ", |e| {
                matches!(e, ToolCallError::Json(_))
            }),
            ("[1, 2]", |e| matches!(e, ToolCallError::Shape(_))),
            ("{\"arguments\": {}}", |e| {
                matches!(e, ToolCallError::Shape(_))
            }),
            ("{\"name\": \"x\", \"arguments\": 3}", |e| {
                matches!(e, ToolCallError::Shape(_))
            }),
            ("{\"name\": \"nope\", \"arguments\": {}}", |e| {
                *e == ToolCallError::UnknownTool("nope".into())
            }),
            (
                "{\"name\": \"get_weather\", \"arguments\": {\"town\": 1}}",
                |e| matches!(e, ToolCallError::InvalidArguments(v) if v.len() == 2),
            ),
        ];
        for (body, check) in cases {
            let mut p = ToolCallParser::new(registry());
            let ev = parse_all(
                &mut p,
                &[format!("<tool_call>{body}</tool_call>ok").as_bytes()],
            );
            match &ev[..] {
                [ParseEvent::ToolCallStarted { .. }, ParseEvent::InvalidToolCall { raw, error, .. }, ParseEvent::Text(t)] =>
                {
                    assert_eq!(raw, body);
                    assert!(check(error), "{body}: {error}");
                    assert_eq!(t, "ok");
                }
                other => panic!("{body}: {other:?}"),
            }
        }
    }

    #[test]
    fn missing_arguments_means_empty_object() {
        let reg = Arc::new(
            ToolRegistry::from_specs([ToolSpec::new("ping", "", json!({"type": "object"}))])
                .unwrap(),
        );
        let mut p = ToolCallParser::new(reg);
        let ev = parse_all(&mut p, &[b"<tool_call>{\"name\":\"ping\"}</tool_call>"]);
        assert!(
            matches!(&ev[1], ParseEvent::ToolCall(c) if c.arguments == json!({})),
            "{ev:?}"
        );
    }

    #[test]
    fn cancel_inside_and_outside_spans() {
        let mut p = ToolCallParser::new(registry());
        let ev = parse_all(
            &mut p,
            &[
                b"<tool_call>{\"name\":\"get_wea<tool_cancel>Sorry, ",
                b"never mind.<tool_cancel>",
            ],
        );
        assert_eq!(
            ev,
            [
                ParseEvent::ToolCallStarted {
                    call_id: "c_1".into()
                },
                ParseEvent::ToolCallAborted {
                    call_id: "c_1".into()
                },
                ParseEvent::Text("Sorry, never mind.".into()),
                ParseEvent::CancelRequested,
            ]
        );
    }

    #[test]
    fn unterminated_and_reopened_spans() {
        let mut p = ToolCallParser::new(registry());
        let ev = parse_all(&mut p, &[b"<tool_call>{\"a\"<tool_call>{\"b\""]);
        assert_eq!(
            ev,
            [
                ParseEvent::ToolCallStarted {
                    call_id: "c_1".into()
                },
                ParseEvent::InvalidToolCall {
                    call_id: "c_1".into(),
                    raw: "{\"a\"".into(),
                    error: ToolCallError::Unterminated
                },
                ParseEvent::ToolCallStarted {
                    call_id: "c_2".into()
                },
                ParseEvent::InvalidToolCall {
                    call_id: "c_2".into(),
                    raw: "{\"b\"".into(),
                    error: ToolCallError::Unterminated
                },
            ]
        );
        assert!(!p.in_tool_call());
        // Stray end markers vanish; a trailing partial marker is flushed as text.
        let ev = parse_all(&mut p, &[b"a</tool_call>b<tool_c"]);
        assert_eq!(ev, [ParseEvent::Text("ab<tool_c".into())]);
    }

    #[test]
    fn too_long_span_is_skipped() {
        let cfg = ParserConfig {
            max_span_bytes: 8,
            ..ParserConfig::default()
        };
        let mut p = ToolCallParser::with_config(registry(), cfg);
        let ev = parse_all(&mut p, &[b"<tool_call>0123456789abcdef</tool_call>after"]);
        assert!(
            matches!(
                &ev[1],
                ParseEvent::InvalidToolCall {
                    error: ToolCallError::TooLong,
                    ..
                }
            ),
            "{ev:?}"
        );
        assert_eq!(ev[2], ParseEvent::Text("after".into()));
        assert_eq!(ev.len(), 3);
    }

    #[test]
    fn token_level_markers_without_text_scanning() {
        let cfg = ParserConfig {
            scan_text_markers: false,
            call_id_prefix: "call-".into(),
            ..ParserConfig::default()
        };
        let mut p = ToolCallParser::with_config(registry(), cfg);
        let mut ev = p.push_str("Checking <tool_call> literally. ");
        ev.extend(p.push_marker(Marker::ToolCall));
        // A JSON string containing the end marker's spelling cannot end the span.
        ev.extend(
            p.push_str("{\"name\": \"get_weather\", \"arguments\": {\"city\": \"</tool_call>\"}}"),
        );
        ev.extend(p.push_marker(Marker::ToolCallEnd));
        ev.extend(p.finish());
        assert_eq!(
            normalize(ev),
            [
                ParseEvent::Text("Checking <tool_call> literally. ".into()),
                ParseEvent::ToolCallStarted {
                    call_id: "call-1".into()
                },
                ParseEvent::ToolCall(ToolCall {
                    call_id: "call-1".into(),
                    name: "get_weather".into(),
                    arguments: json!({"city": "</tool_call>"}),
                }),
            ]
        );
    }

    #[test]
    fn invalid_utf8_is_replaced() {
        let mut p = ToolCallParser::new(registry());
        let ev = parse_all(&mut p, &[b"a\xffb\xc3"]);
        assert_eq!(ev, [ParseEvent::Text("a\u{fffd}b\u{fffd}".into())]);
    }

    #[test]
    fn marker_names_round_trip() {
        for m in Marker::ALL {
            assert_eq!(Marker::from_name(m.as_str()), Some(m));
        }
        assert_eq!(Marker::from_name("<sys>"), None);
        assert_eq!(CallId::from("c_9").to_string(), "c_9");
        let call = ToolCall {
            call_id: "c_17".into(),
            name: "get_weather".into(),
            arguments: json!({"city": "Paris"}),
        };
        assert_eq!(
            serde_json::to_value(&call).unwrap(),
            json!({"call_id": "c_17", "name": "get_weather", "arguments": {"city": "Paris"}})
        );
    }
}
