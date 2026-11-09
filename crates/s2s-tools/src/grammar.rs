//! Grammar-constrained decoding of tool calls with `llguidance`.
//!
//! While a `<tool_call>` span is open, the sampler must only produce JSON of
//! the form `{"name": <tool>, "arguments": <args>}` where `<args>` matches that
//! tool's parameter schema (plan §4.2). [`ToolCallGrammar`] compiles one
//! JSON Schema covering every declared tool:
//!
//! ```json
//! {"anyOf": [
//!   {"type": "object",
//!    "properties": {"name": {"const": "get_weather"}, "arguments": <get_weather parameters>},
//!    "required": ["name", "arguments"], "additionalProperties": false},
//!   …one branch per tool…
//! ]}
//! ```
//!
//! and hands it to llguidance, whose token environment is built from a
//! [`TokenizerInfo`] with `</tool_call>` as the end-of-sequence token. So once
//! the JSON is complete, the only token the mask allows is `</tool_call>`.
//!
//! Masks use the format of
//! `s2s_kernels::reference::sampling::apply_token_bitmask`: `ceil(vocab / 32)`
//! little-endian `u32` words, bit `i % 32` of word `i / 32` set = token `i`
//! allowed.
//!
//! # Limitations
//!
//! - JSON Schema support is llguidance's. Schemas it cannot compile (or finds
//!   unsatisfiable) make [`ToolCallGrammar::new`] fail rather than produce a
//!   mask that could emit invalid calls. Remote `$ref`s are not resolved.
//! - JSON itself allows numbers such as `1e999` that overflow `f64`, which
//!   `serde_json` (and so the parser) rejects. By default the grammar
//!   therefore gives every unbounded `"type": "number"` subschema the bounds
//!   `±GrammarConfig::number_limit` (10^15), which also removes exponent
//!   notation from the grammar. This only narrows what the model may write;
//!   everything it can write still validates against the declared schema.
//!   Numbers inside schema-less values (`{}`, or `additionalProperties`
//!   defaulting to "anything") are not covered: declare
//!   `"additionalProperties": false` and type every value to get the full
//!   guarantee.

use std::sync::Arc;

use llguidance::api::TopLevelGrammar;
use llguidance::{Matcher, ParserFactory};
use s2s_core::streams::special;
use serde_json::{json, Value};

use crate::spec::ToolRegistry;
use crate::tokenizer::{tok_env, TokenizerError, TokenizerInfo};

/// JSON formatting the grammar enforces inside spans.
///
/// The separators should match how tool calls are written in training data
/// (and in the base LLM's function-calling format). The span parser accepts
/// any valid JSON regardless.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonStyle {
    /// Between items/members, e.g. `", "` or `","`.
    pub item_separator: String,
    /// Between a key and its value, e.g. `": "` or `":"`.
    pub key_separator: String,
    /// Allow arbitrary whitespace between tokens instead of exactly the
    /// separators above. Off by default: free whitespace wastes text slots
    /// and makes masked random sampling wander.
    pub whitespace_flexible: bool,
}

impl Default for JsonStyle {
    /// `{"name": "get_weather", "arguments": {"city": "Paris"}}` (plan §4.3).
    fn default() -> Self {
        Self {
            item_separator: ", ".into(),
            key_separator: ": ".into(),
            whitespace_flexible: false,
        }
    }
}

/// [`ToolCallGrammar`] settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrammarConfig {
    /// JSON formatting inside spans.
    pub style: JsonStyle,
    /// Also allow `<tool_cancel>` at every step inside a span, so the model can
    /// abandon a call mid-way when the user barges in (plan §4.2).
    pub allow_cancel: bool,
    /// Bounds `[-limit, limit]` added to every `"type": "number"` subschema
    /// that lacks a minimum/maximum, so sampled numbers stay within `f64`
    /// range (see the module docs). `None` keeps llguidance's unbounded
    /// number syntax, exponents included.
    pub number_limit: Option<u64>,
}

impl Default for GrammarConfig {
    fn default() -> Self {
        Self {
            style: JsonStyle::default(),
            allow_cancel: true,
            number_limit: Some(1_000_000_000_000_000),
        }
    }
}

/// Errors from grammar construction or use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GrammarError {
    /// No tools are registered, so no call can be valid.
    NoTools,
    /// The tokenizer lacks a required special token, or its vocabulary is unusable.
    Tokenizer(TokenizerError),
    /// llguidance rejected the combined schema or failed while matching.
    Engine(String),
    /// The sampled token is not allowed in the current state.
    TokenRejected(u32),
    /// The span is already closed or aborted.
    Finished,
    /// The output buffer for a mask is too short.
    MaskTooShort {
        /// Words needed.
        needed: usize,
        /// Words given.
        got: usize,
    },
}

impl std::fmt::Display for GrammarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoTools => f.write_str("no tools registered"),
            Self::Tokenizer(e) => write!(f, "tokenizer: {e}"),
            Self::Engine(e) => write!(f, "grammar engine: {e}"),
            Self::TokenRejected(t) => write!(f, "token {t} is not allowed here"),
            Self::Finished => f.write_str("tool-call span already finished"),
            Self::MaskTooShort { needed, got } => {
                write!(f, "mask buffer has {got} words, needs {needed}")
            }
        }
    }
}

impl std::error::Error for GrammarError {}

impl From<TokenizerError> for GrammarError {
    fn from(e: TokenizerError) -> Self {
        Self::Tokenizer(e)
    }
}

fn engine(e: impl std::fmt::Display) -> GrammarError {
    GrammarError::Engine(e.to_string())
}

/// Ids of the special tokens the constraint reacts to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpecialIds {
    /// `<tool_call>`.
    pub tool_call: u32,
    /// `</tool_call>`.
    pub tool_call_end: u32,
    /// `<tool_cancel>`, if the vocabulary has it.
    pub tool_cancel: Option<u32>,
}

/// The compiled tool-call grammar for one tool set and tokenizer. Build it
/// once per session (or per tool set) and start a [`ToolCallMatcher`] for
/// every span; starting one is a cheap clone.
pub struct ToolCallGrammar {
    /// A matcher in the initial state, cloned for each span.
    proto: Matcher,
    schema: Value,
    ids: SpecialIds,
    vocab_size: usize,
    allow_cancel: bool,
}

impl std::fmt::Debug for ToolCallGrammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolCallGrammar")
            .field("ids", &self.ids)
            .field("vocab_size", &self.vocab_size)
            .field("allow_cancel", &self.allow_cancel)
            .finish_non_exhaustive()
    }
}

/// Schema keywords whose values are data, not subschemas.
const DATA_KEYWORDS: [&str; 4] = ["const", "enum", "default", "examples"];

/// Add `minimum`/`maximum` of `±limit` to every number subschema missing them.
fn bound_numbers(schema: &mut Value, limit: u64) {
    match schema {
        Value::Object(obj) => {
            let is_number = match obj.get("type") {
                Some(Value::String(t)) => t == "number",
                Some(Value::Array(ts)) => ts.iter().any(|t| t == "number"),
                _ => false,
            };
            if is_number {
                if !obj.contains_key("minimum") && !obj.contains_key("exclusiveMinimum") {
                    obj.insert("minimum".into(), json!(-(limit as i64)));
                }
                if !obj.contains_key("maximum") && !obj.contains_key("exclusiveMaximum") {
                    obj.insert("maximum".into(), json!(limit));
                }
            }
            for (k, v) in obj.iter_mut() {
                if !DATA_KEYWORDS.contains(&k.as_str()) {
                    bound_numbers(v, limit);
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(|v| bound_numbers(v, limit)),
        _ => {}
    }
}

/// The JSON Schema a span must match: one `anyOf` branch per tool.
pub fn tool_call_schema(registry: &ToolRegistry) -> Value {
    let branches: Vec<Value> = registry
        .specs()
        .map(|t| {
            json!({
                "type": "object",
                "properties": {
                    "name": {"const": t.name},
                    "arguments": t.parameters,
                },
                "required": ["name", "arguments"],
                "additionalProperties": false,
            })
        })
        .collect();
    json!({ "anyOf": branches })
}

impl ToolCallGrammar {
    /// Compile the grammar for every tool in `registry`.
    pub fn new(
        registry: &ToolRegistry,
        tokenizer: Arc<dyn TokenizerInfo>,
        config: GrammarConfig,
    ) -> Result<Self, GrammarError> {
        if registry.is_empty() {
            return Err(GrammarError::NoTools);
        }
        let missing = |n| GrammarError::Tokenizer(TokenizerError::MissingSpecialToken(n));
        let ids = SpecialIds {
            tool_call: tokenizer
                .special_token_id(special::TOOL_CALL)
                .ok_or(missing(special::TOOL_CALL))?,
            tool_call_end: tokenizer
                .special_token_id(special::TOOL_CALL_END)
                .ok_or(missing(special::TOOL_CALL_END))?,
            tool_cancel: tokenizer.special_token_id(special::TOOL_CANCEL),
        };
        let vocab_size = tokenizer.vocab_size();
        let env = tok_env(tokenizer, special::TOOL_CALL_END)?;

        let mut schema = tool_call_schema(registry);
        if let Some(limit) = config.number_limit {
            bound_numbers(&mut schema, limit.min(i64::MAX as u64));
        }
        let mut with_options = schema.clone();
        with_options["x-guidance"] = json!({
            "item_separator": config.style.item_separator,
            "key_separator": config.style.key_separator,
            "whitespace_flexible": config.style.whitespace_flexible,
        });
        let mut factory = ParserFactory::new_simple(&env).map_err(engine)?;
        factory.quiet();
        let parser = factory
            .create_parser(TopLevelGrammar::from_json_schema(with_options))
            .map_err(engine)?;
        let proto = Matcher::new(Ok(parser));
        if let Some(e) = proto.get_error() {
            return Err(GrammarError::Engine(e));
        }
        Ok(Self {
            proto,
            schema,
            ids,
            vocab_size,
            allow_cancel: config.allow_cancel && ids.tool_cancel.is_some(),
        })
    }

    /// The schema the grammar enforces: [`tool_call_schema`] plus any
    /// number bounds from [`GrammarConfig::number_limit`].
    pub fn schema(&self) -> &Value {
        &self.schema
    }

    /// Special token ids.
    pub fn special_ids(&self) -> SpecialIds {
        self.ids
    }

    /// Tokenizer vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Words in a mask: `ceil(vocab_size / 32)`.
    pub fn mask_words(&self) -> usize {
        self.vocab_size.div_ceil(32)
    }

    /// Start matching a new span (call right after `<tool_call>` is sampled).
    pub fn start(&self) -> ToolCallMatcher {
        ToolCallMatcher {
            matcher: self.proto.deep_clone(),
            ids: self.ids,
            vocab_size: self.vocab_size,
            words: self.mask_words(),
            allow_cancel: self.allow_cancel,
            state: SpanState::Open,
        }
    }
}

/// Where a span is after consuming a token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanState {
    /// Still writing JSON.
    Open,
    /// `</tool_call>` was consumed after complete JSON.
    Closed,
    /// `<tool_cancel>` was consumed.
    Aborted,
}

/// Grammar state for one open `<tool_call>` span.
pub struct ToolCallMatcher {
    matcher: Matcher,
    ids: SpecialIds,
    vocab_size: usize,
    words: usize,
    allow_cancel: bool,
    state: SpanState,
}

impl std::fmt::Debug for ToolCallMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolCallMatcher")
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl ToolCallMatcher {
    /// Current state.
    pub fn state(&self) -> SpanState {
        self.state
    }

    /// Words in a mask.
    pub fn mask_words(&self) -> usize {
        self.words
    }

    /// Write the allowed-token bitmask for the next token into `out`
    /// (`out.len() >= mask_words()`; extra words are zeroed, so model vocab
    /// padding is never allowed).
    pub fn fill_mask(&mut self, out: &mut [u32]) -> Result<(), GrammarError> {
        if self.state != SpanState::Open {
            return Err(GrammarError::Finished);
        }
        if out.len() < self.words {
            return Err(GrammarError::MaskTooShort {
                needed: self.words,
                got: out.len(),
            });
        }
        let vob = self.matcher.compute_mask_or_eos().map_err(engine)?;
        out.fill(0);
        let src = vob.as_slice();
        let n = src.len().min(self.words);
        out[..n].copy_from_slice(&src[..n]);
        // Clear bits past the vocabulary in the last word.
        let rem = self.vocab_size % 32;
        if rem != 0 {
            out[self.words - 1] &= (1u32 << rem) - 1;
        }
        if self.allow_cancel {
            if let Some(c) = self.ids.tool_cancel {
                out[c as usize / 32] |= 1 << (c % 32);
            }
        }
        Ok(())
    }

    /// The allowed-token bitmask for the next token.
    pub fn compute_mask(&mut self) -> Result<Vec<u32>, GrammarError> {
        let mut out = vec![0; self.words];
        self.fill_mask(&mut out)?;
        Ok(out)
    }

    /// Advance by the sampled token. A token outside the last mask is
    /// rejected with [`GrammarError::TokenRejected`] and leaves the state
    /// unchanged.
    pub fn consume(&mut self, token: u32) -> Result<SpanState, GrammarError> {
        if self.state != SpanState::Open {
            return Err(GrammarError::Finished);
        }
        if Some(token) == self.ids.tool_cancel && self.allow_cancel {
            self.state = SpanState::Aborted;
        } else if token == self.ids.tool_call_end {
            if !self.matcher.is_accepting().map_err(engine)? {
                return Err(GrammarError::TokenRejected(token));
            }
            self.state = SpanState::Closed;
        } else {
            // Check first: a rejected token puts llguidance's matcher into a
            // permanent error state.
            let ok = self.matcher.validate_tokens(&[token]).map_err(engine)?;
            if ok != 1 {
                return Err(GrammarError::TokenRejected(token));
            }
            self.matcher.consume_token(token).map_err(engine)?;
        }
        Ok(self.state)
    }
}

/// Token-level constraint for the assistant text stream: unconstrained
/// outside spans, grammar-masked inside.
///
/// Per step: call [`mask`](Self::mask) before sampling (apply it with
/// `apply_token_bitmask` when it is `Some`), then [`observe`](Self::observe)
/// the sampled token.
#[derive(Debug)]
pub struct ToolCallConstraint {
    grammar: Arc<ToolCallGrammar>,
    active: Option<ToolCallMatcher>,
}

impl ToolCallConstraint {
    /// A constraint using `grammar`.
    pub fn new(grammar: Arc<ToolCallGrammar>) -> Self {
        Self {
            grammar,
            active: None,
        }
    }

    /// Whether a span is open (the next token is constrained).
    pub fn in_tool_call(&self) -> bool {
        self.active.is_some()
    }

    /// The mask for the next token, or `None` outside a span.
    pub fn mask(&mut self) -> Result<Option<Vec<u32>>, GrammarError> {
        self.active.as_mut().map(|m| m.compute_mask()).transpose()
    }

    /// Like [`mask`](Self::mask) into a caller buffer; returns whether a mask
    /// was written (`false` = unconstrained, `out` untouched).
    pub fn fill_mask(&mut self, out: &mut [u32]) -> Result<bool, GrammarError> {
        match &mut self.active {
            Some(m) => m.fill_mask(out).map(|()| true),
            None => Ok(false),
        }
    }

    /// Record the sampled token. Returns the span state when the token was
    /// inside a span (or opened one), `None` for ordinary text.
    pub fn observe(&mut self, token: u32) -> Result<Option<SpanState>, GrammarError> {
        match &mut self.active {
            None if token == self.grammar.ids.tool_call => {
                self.active = Some(self.grammar.start());
                Ok(Some(SpanState::Open))
            }
            None => Ok(None),
            Some(m) => {
                let state = m.consume(token)?;
                if state != SpanState::Open {
                    self.active = None;
                }
                Ok(Some(state))
            }
        }
    }

    /// Drop any open span (e.g. on session reset).
    pub fn reset(&mut self) {
        self.active = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{Marker, ParseEvent, ParserConfig, ToolCallParser};
    use crate::spec::ToolSpec;
    use crate::tokenizer::{ByteTokenizer, TokenPiece};
    use s2s_core::SplitMix64;
    use s2s_kernels::reference::sampling::{apply_token_bitmask, sample, SamplingParams};

    fn tools() -> Vec<ToolSpec> {
        vec![
            ToolSpec::new(
                "get_weather",
                "Current weather.",
                json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "minLength": 1, "maxLength": 12},
                        "unit": {"enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["city"],
                    "additionalProperties": false
                }),
            ),
            ToolSpec::new(
                "set_timer",
                "Start a timer.",
                json!({
                    "type": "object",
                    "properties": {
                        "seconds": {"type": "integer", "minimum": 1, "maximum": 3600},
                        "label": {"type": "string", "maxLength": 8},
                        "loud": {"type": "boolean"}
                    },
                    "required": ["seconds"],
                    "additionalProperties": false
                }),
            ),
            ToolSpec::client(
                "search",
                "Search the catalog.",
                json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "maxLength": 10},
                        "max_price": {"type": "number", "minimum": 0, "maximum": 1000},
                        "tags": {
                            "type": "array",
                            "items": {"enum": ["new", "sale", "eco"]},
                            "maxItems": 3
                        },
                        "where": {
                            "type": "object",
                            "properties": {"lat": {"type": "number"}, "near": {"type": "boolean"}},
                            "required": ["near"],
                            "additionalProperties": false
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }),
            ),
        ]
    }

    fn setup(cfg: GrammarConfig) -> (Arc<ByteTokenizer>, Arc<ToolRegistry>, Arc<ToolCallGrammar>) {
        let tok = Arc::new(ByteTokenizer::with_json_merges());
        let reg = Arc::new(ToolRegistry::from_specs(tools()).unwrap());
        let g = Arc::new(ToolCallGrammar::new(&reg, tok.clone(), cfg).unwrap());
        (tok, reg, g)
    }

    fn allowed(mask: &[u32], t: u32) -> bool {
        mask[t as usize / 32] >> (t % 32) & 1 == 1
    }

    /// Run one constrained span with masked sampling and return the tokens
    /// (excluding `<tool_call>`, including the final token).
    fn run_span(
        g: &Arc<ToolCallGrammar>,
        rng: &mut SplitMix64,
        params: &SamplingParams,
        max_steps: usize,
    ) -> Vec<u32> {
        let mut c = ToolCallConstraint::new(g.clone());
        assert_eq!(
            c.observe(g.special_ids().tool_call).unwrap(),
            Some(SpanState::Open)
        );
        let mut out = Vec::new();
        for _ in 0..max_steps {
            let mask = c.mask().unwrap().expect("inside a span");
            let mut logits: Vec<f32> = (0..g.vocab_size())
                .map(|_| (rng.next_f64() as f32) * 8.0 - 4.0)
                .collect();
            apply_token_bitmask(&mut logits, &mask);
            let t = sample(&logits, params, rng.next_f64()).expect("mask is never empty") as u32;
            assert!(allowed(&mask, t));
            out.push(t);
            match c.observe(t).unwrap() {
                Some(SpanState::Open) => {}
                Some(_) => {
                    assert!(!c.in_tool_call());
                    return out;
                }
                None => unreachable!("still inside the span"),
            }
        }
        panic!("span did not finish in {max_steps} steps");
    }

    /// Feed a sampled span to the parser the way the runtime would: special
    /// tokens as markers, ordinary tokens as bytes.
    fn parse_span(reg: &Arc<ToolRegistry>, tok: &ByteTokenizer, toks: &[u32]) -> Vec<ParseEvent> {
        let cfg = ParserConfig {
            scan_text_markers: false,
            ..ParserConfig::default()
        };
        let mut p = ToolCallParser::with_config(reg.clone(), cfg);
        let mut ev = p.push_marker(Marker::ToolCall);
        for &t in toks {
            ev.extend(match tok.token(t) {
                TokenPiece::Bytes(b) => p.push_bytes(&b),
                TokenPiece::Special(n) => p.push_marker(Marker::from_name(&n).unwrap()),
            });
        }
        ev.extend(p.finish());
        ev
    }

    fn assert_valid_call(reg: &Arc<ToolRegistry>, tok: &ByteTokenizer, toks: &[u32]) -> String {
        let text = tok.decode(toks);
        // 1. The parser accepts it (JSON + schema validation via `jsonschema`).
        let ev = parse_span(reg, tok, toks);
        let call = match &ev[..] {
            [ParseEvent::ToolCallStarted { .. }, ParseEvent::ToolCall(c)] => c.clone(),
            other => panic!("not a valid call: {text:?} -> {other:?}"),
        };
        // 2. Independently re-validate.
        assert!(reg.validate_arguments(&call.name, &call.arguments).is_ok());
        // 3. Same result when the text (with marker names) is scanned instead.
        let mut p = ToolCallParser::new(reg.clone());
        let mut ev2 = p.push_str(special::TOOL_CALL);
        ev2.extend(p.push_str(&text));
        ev2.extend(p.finish());
        assert_eq!(
            ev2.last(),
            Some(&ParseEvent::ToolCall(call.clone())),
            "{text:?}"
        );
        text
    }

    #[test]
    fn schema_has_one_branch_per_tool() {
        let reg = ToolRegistry::from_specs(tools()).unwrap();
        let s = tool_call_schema(&reg);
        let branches = s["anyOf"].as_array().unwrap();
        assert_eq!(branches.len(), 3);
        assert_eq!(
            branches[1]["properties"]["name"],
            json!({"const": "set_timer"})
        );
        assert_eq!(
            branches[2]["properties"]["arguments"],
            tools()[2].parameters
        );
    }

    #[test]
    fn construction_errors() {
        let tok: Arc<dyn TokenizerInfo> = Arc::new(ByteTokenizer::with_json_merges());
        let empty = ToolRegistry::new();
        assert_eq!(
            ToolCallGrammar::new(&empty, tok.clone(), GrammarConfig::default()).err(),
            Some(GrammarError::NoTools)
        );
        // A schema llguidance cannot satisfy (a valid JSON Schema, so the
        // registry accepts it) fails loudly instead of producing a mask that
        // could emit invalid calls.
        let reg = ToolRegistry::from_specs([ToolSpec::new(
            "f",
            "",
            json!({
                "type": "object",
                "properties": {"n": {"type": "integer", "minimum": 5, "maximum": 1}},
                "required": ["n"]
            }),
        )])
        .unwrap();
        let err = ToolCallGrammar::new(&reg, tok, GrammarConfig::default()).unwrap_err();
        assert!(matches!(err, GrammarError::Engine(_)), "{err}");
    }

    #[test]
    fn forced_prefix_and_end_token() {
        let (tok, reg, g) = setup(GrammarConfig {
            allow_cancel: false,
            ..GrammarConfig::default()
        });
        let ids = g.special_ids();
        let mut m = g.start();
        assert_eq!(m.mask_words(), g.vocab_size().div_ceil(32));
        // Every branch starts with the forced bytes `{"name": "`; llguidance
        // only allows their canonical tokenization (here the `{"` merge).
        let mask = m.compute_mask().unwrap();
        let set: Vec<u32> = (0..g.vocab_size() as u32)
            .filter(|&t| allowed(&mask, t))
            .collect();
        assert_eq!(set, tok.tokenize(b"{\"")[..1]);
        // Walk a fixed call token by token, checking every token is allowed.
        let text = "{\"name\": \"set_timer\", \"arguments\": {\"seconds\": 90}}";
        for t in tok.tokenize(text.as_bytes()) {
            let mask = m.compute_mask().unwrap();
            assert!(allowed(&mask, t), "{:?} not allowed", tok.token(t));
            assert!(!allowed(&mask, ids.tool_call_end));
            assert_eq!(m.consume(t).unwrap(), SpanState::Open);
        }
        // JSON complete: only `</tool_call>` remains.
        let mask = m.compute_mask().unwrap();
        let set: Vec<u32> = (0..g.vocab_size() as u32)
            .filter(|&t| allowed(&mask, t))
            .collect();
        assert_eq!(set, [ids.tool_call_end]);
        assert_eq!(m.consume(ids.tool_call_end).unwrap(), SpanState::Closed);
        assert_eq!(m.consume(ids.tool_call_end), Err(GrammarError::Finished));
        assert_eq!(m.compute_mask(), Err(GrammarError::Finished));
        let _ = reg;
    }

    #[test]
    fn rejected_tokens_leave_state_unchanged() {
        let (tok, _reg, g) = setup(GrammarConfig {
            allow_cancel: false,
            ..GrammarConfig::default()
        });
        let ids = g.special_ids();
        let mut m = g.start();
        let x = u32::from(b'x');
        assert_eq!(m.consume(x), Err(GrammarError::TokenRejected(x)));
        // Ending before the JSON is complete is rejected too.
        assert_eq!(
            m.consume(ids.tool_call_end),
            Err(GrammarError::TokenRejected(ids.tool_call_end))
        );
        // Cancel is just another rejected token when not allowed.
        let cancel = ids.tool_cancel.unwrap();
        assert!(!allowed(&m.compute_mask().unwrap(), cancel));
        assert_eq!(m.consume(cancel), Err(GrammarError::TokenRejected(cancel)));
        // Unknown tool names are impossible.
        for t in tok.tokenize(b"{\"name\": \"") {
            m.consume(t).unwrap();
        }
        let mask = m.compute_mask().unwrap();
        assert!(allowed(&mask, u32::from(b'g')));
        assert!(allowed(&mask, u32::from(b's')));
        assert!(!allowed(&mask, u32::from(b'x')));
        // Still usable after the rejections.
        assert_eq!(m.consume(u32::from(b's')).unwrap(), SpanState::Open);
        let mut short = vec![0u32; 1];
        assert!(matches!(
            m.fill_mask(&mut short),
            Err(GrammarError::MaskTooShort { .. })
        ));
        // Longer buffers (padded model vocab) get zeroed extra words.
        let mut long = vec![u32::MAX; m.mask_words() + 2];
        m.fill_mask(&mut long).unwrap();
        assert_eq!(long[m.mask_words()..], [0, 0]);
    }

    #[test]
    fn cancel_is_always_allowed_when_enabled() {
        let (tok, _reg, g) = setup(GrammarConfig::default());
        let cancel = g.special_ids().tool_cancel.unwrap();
        let mut c = ToolCallConstraint::new(g.clone());
        assert_eq!(c.mask().unwrap(), None);
        let mut buf = vec![7u32; g.mask_words()];
        assert!(!c.fill_mask(&mut buf).unwrap());
        assert_eq!(buf[0], 7, "untouched outside spans");
        assert_eq!(c.observe(u32::from(b'h')).unwrap(), None);
        c.observe(g.special_ids().tool_call).unwrap();
        for t in tok.tokenize(b"{\"name\": \"get_") {
            assert!(allowed(&c.mask().unwrap().unwrap(), cancel));
            c.observe(t).unwrap();
        }
        assert!(c.fill_mask(&mut buf).unwrap());
        assert!(allowed(&buf, cancel));
        assert_eq!(c.observe(cancel).unwrap(), Some(SpanState::Aborted));
        assert!(!c.in_tool_call());
        // A new span starts fresh.
        c.observe(g.special_ids().tool_call).unwrap();
        assert!(c.in_tool_call());
        c.reset();
        assert!(!c.in_tool_call());
    }

    #[test]
    fn masked_greedy_sampling_yields_valid_calls() {
        let (tok, reg, g) = setup(GrammarConfig {
            allow_cancel: false,
            ..GrammarConfig::default()
        });
        let mut rng = SplitMix64::new(1);
        for _ in 0..40 {
            let toks = run_span(&g, &mut rng, &SamplingParams::GREEDY, 400);
            assert_eq!(*toks.last().unwrap(), g.special_ids().tool_call_end);
            assert_valid_call(&reg, &tok, &toks);
        }
    }

    #[test]
    fn masked_random_sampling_yields_valid_calls() {
        let (tok, reg, g) = setup(GrammarConfig {
            allow_cancel: false,
            ..GrammarConfig::default()
        });
        let params = SamplingParams {
            temperature: 1.0,
            top_k: None,
            top_p: None,
        };
        let mut rng = SplitMix64::new(42);
        let mut names = std::collections::HashSet::new();
        for _ in 0..150 {
            let toks = run_span(&g, &mut rng, &params, 600);
            let text = assert_valid_call(&reg, &tok, &toks);
            let v: Value =
                serde_json::from_str(&text[..text.len() - "</tool_call>".len()]).unwrap();
            names.insert(v["name"].as_str().unwrap().to_string());
        }
        // Every tool is reachable.
        assert_eq!(names.len(), 3, "{names:?}");
    }

    #[test]
    fn masked_sampling_with_cancel_yields_valid_or_aborted() {
        let (tok, reg, g) = setup(GrammarConfig::default());
        let cancel = g.special_ids().tool_cancel.unwrap();
        let params = SamplingParams {
            temperature: 1.0,
            top_k: Some(20),
            top_p: Some(0.95),
        };
        let mut rng = SplitMix64::new(7);
        let (mut closed, mut aborted) = (0, 0);
        for _ in 0..60 {
            let toks = run_span(&g, &mut rng, &params, 600);
            if *toks.last().unwrap() == cancel {
                aborted += 1;
                let ev = parse_span(&reg, &tok, &toks);
                assert!(
                    matches!(ev.last(), Some(ParseEvent::ToolCallAborted { .. })),
                    "{ev:?}"
                );
            } else {
                closed += 1;
                assert_valid_call(&reg, &tok, &toks);
            }
        }
        assert_eq!(closed + aborted, 60);
    }

    #[test]
    fn number_bounds() {
        let mut s = json!({
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": ["number", "null"], "exclusiveMinimum": 0},
                "c": {"type": "number", "minimum": 1, "maximum": 2},
                "d": {"type": "integer"},
                "e": {"const": {"type": "number"}, "default": {"type": "number"}},
            }
        });
        bound_numbers(&mut s, 10);
        let p = &s["properties"];
        assert_eq!(
            p["a"],
            json!({"type": "number", "minimum": -10, "maximum": 10})
        );
        assert_eq!(
            p["b"],
            json!({"type": ["number", "null"], "exclusiveMinimum": 0, "maximum": 10})
        );
        assert_eq!(
            p["c"],
            json!({"type": "number", "minimum": 1, "maximum": 2})
        );
        assert_eq!(p["d"], json!({"type": "integer"}));
        // Data keywords are left alone.
        assert_eq!(p["e"]["const"], json!({"type": "number"}));
        assert_eq!(p["e"]["default"], json!({"type": "number"}));

        // The grammar enforces the bounded schema: no exponent after digits.
        let tok = Arc::new(ByteTokenizer::with_json_merges());
        let reg = ToolRegistry::from_specs([ToolSpec::new(
            "f",
            "",
            json!({"type": "object", "properties": {"x": {"type": "number"}},
                   "required": ["x"], "additionalProperties": false}),
        )])
        .unwrap();
        let prefix = b"{\"name\": \"f\", \"arguments\": {\"x\": 12";
        let mask_after_prefix = |cfg: GrammarConfig| {
            let g = ToolCallGrammar::new(&reg, tok.clone(), cfg).unwrap();
            let mut m = g.start();
            for t in tok.tokenize(prefix) {
                m.consume(t).unwrap();
            }
            (g.schema().clone(), m.compute_mask().unwrap())
        };
        let (schema, mask) = mask_after_prefix(GrammarConfig::default());
        assert_eq!(
            schema["anyOf"][0]["properties"]["arguments"]["properties"]["x"]["maximum"],
            json!(1_000_000_000_000_000u64)
        );
        assert!(!allowed(&mask, u32::from(b'e')));
        assert!(allowed(&mask, u32::from(b'.')));
        let (_, mask) = mask_after_prefix(GrammarConfig {
            number_limit: None,
            ..GrammarConfig::default()
        });
        assert!(allowed(&mask, u32::from(b'e')));
    }

    #[test]
    fn compact_style() {
        let (tok, _reg, g) = setup(GrammarConfig {
            style: JsonStyle {
                item_separator: ",".into(),
                key_separator: ":".into(),
                whitespace_flexible: false,
            },
            allow_cancel: false,
            ..GrammarConfig::default()
        });
        let mut m = g.start();
        for t in tok.tokenize(b"{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Oslo\"}}") {
            m.consume(t).unwrap();
        }
        assert_eq!(
            m.consume(g.special_ids().tool_call_end).unwrap(),
            SpanState::Closed
        );
        let mut m = g.start();
        m.consume(u32::from(b'{')).unwrap();
        m.consume(u32::from(b'"')).unwrap();
        for t in tok.tokenize(b"name\"") {
            m.consume(t).unwrap();
        }
        assert!(!allowed(&m.compute_mask().unwrap(), u32::from(b' ')));
    }
}
