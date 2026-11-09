//! End to end: llguidance grammar masks over the real tokenizer wrapper.

use std::collections::HashSet;
use std::sync::Arc;

use s2s_core::streams::special;
use s2s_core::SplitMix64;
use s2s_kernels::reference::sampling::{apply_token_bitmask, sample, SamplingParams};
use s2s_tokenizer::HfTokenizer;
use s2s_tools::{
    render_tool_call, GrammarConfig, JsonStyle, Marker, ParseEvent, ParserConfig, SpanState,
    TokenPiece, TokenizerInfo, ToolCall, ToolCallConstraint, ToolCallGrammar, ToolCallParser,
    ToolRegistry, ToolSpec,
};
use serde_json::{json, Value};

use crate::common::all;

/// The tool set of the s2s-tools grammar tests.
pub fn tools() -> Vec<ToolSpec> {
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

fn allowed(mask: &[u32], t: u32) -> bool {
    mask[t as usize / 32] >> (t % 32) & 1 == 1
}

fn grammar(
    tok: &Arc<HfTokenizer>,
    cfg: GrammarConfig,
) -> (Arc<ToolRegistry>, Arc<ToolCallGrammar>) {
    let reg = Arc::new(ToolRegistry::from_specs(tools()).unwrap());
    let g = Arc::new(ToolCallGrammar::new(&reg, tok.clone(), cfg).unwrap());
    (reg, g)
}

/// One constrained span with random logits; returns the tokens after
/// `<tool_call>`, including the final one.
fn run_span(
    g: &Arc<ToolCallGrammar>,
    rng: &mut SplitMix64,
    params: &SamplingParams,
    max_steps: usize,
) -> Vec<u32> {
    let mut c = ToolCallConstraint::new(g.clone());
    c.observe(g.special_ids().tool_call).unwrap();
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
            Some(_) => return out,
            None => unreachable!("inside a span"),
        }
    }
    panic!("span did not finish in {max_steps} steps");
}

/// Parse a span the way the runtime does: specials as markers, other
/// tokens as bytes.
fn parse_span(reg: &Arc<ToolRegistry>, tok: &HfTokenizer, toks: &[u32]) -> Vec<ParseEvent> {
    let cfg = ParserConfig {
        scan_text_markers: false,
        ..ParserConfig::default()
    };
    let mut p = ToolCallParser::with_config(reg.clone(), cfg);
    let mut ev = p.push_marker(Marker::ToolCall);
    for &t in toks {
        ev.extend(match TokenizerInfo::token(tok, t) {
            TokenPiece::Bytes(b) => p.push_bytes(&b),
            TokenPiece::Special(n) => p.push_marker(Marker::from_name(&n).unwrap()),
        });
    }
    ev.extend(p.finish());
    ev
}

fn valid_call(reg: &Arc<ToolRegistry>, tok: &HfTokenizer, toks: &[u32]) -> ToolCall {
    let ev = parse_span(reg, tok, toks);
    let call = match &ev[..] {
        [ParseEvent::ToolCallStarted { .. }, ParseEvent::ToolCall(c)] => c.clone(),
        other => panic!("not a valid call: {:?} -> {other:?}", tok.decode(toks)),
    };
    assert!(reg.validate_arguments(&call.name, &call.arguments).is_ok());
    // The text form (marker names scanned from text) parses the same.
    let mut p = ToolCallParser::new(reg.clone());
    let mut ev2 = p.push_str(special::TOOL_CALL);
    ev2.extend(p.push_str(&tok.decode(toks)));
    ev2.extend(p.finish());
    assert_eq!(ev2.last(), Some(&ParseEvent::ToolCall(call.clone())));
    call
}

/// Walk `text`'s encoding through a fresh matcher, checking every token is
/// allowed, then close the span.
pub fn walk(g: &ToolCallGrammar, tok: &HfTokenizer, text: &str) {
    let ids = g.special_ids();
    let mut m = g.start();
    for t in tok.encode(text) {
        let mask = m.compute_mask().unwrap();
        assert!(
            allowed(&mask, t),
            "{text:?}: token {:?} not allowed",
            tok.token_str(t)
        );
        assert_eq!(m.consume(t).unwrap(), SpanState::Open);
    }
    assert!(allowed(&m.compute_mask().unwrap(), ids.tool_call_end));
    assert_eq!(m.consume(ids.tool_call_end).unwrap(), SpanState::Closed);
}

#[test]
fn masked_random_tool_calls_parse_and_validate() {
    let params = SamplingParams {
        temperature: 1.0,
        top_k: None,
        top_p: None,
    };
    for (label, tok) in all() {
        let (reg, g) = grammar(
            &tok,
            GrammarConfig {
                allow_cancel: false,
                ..GrammarConfig::default()
            },
        );
        assert_eq!(g.vocab_size(), tok.vocab_size());
        assert_eq!(g.special_ids().tool_call, tok.special_ids().tool_call);
        let style = JsonStyle::default();
        let mut rng = SplitMix64::new(42);
        let mut names = HashSet::new();
        for _ in 0..120 {
            let toks = run_span(&g, &mut rng, &params, 600);
            assert_eq!(*toks.last().unwrap(), g.special_ids().tool_call_end);
            let call = valid_call(&reg, &tok, &toks);
            // The span is the call's JSON (long decimals may print
            // differently from `serde_json`, so compare parsed values).
            let json = tok.decode(&toks[..toks.len() - 1]);
            let schema = reg.get(&call.name).map(|t| &t.parameters);
            let rendered = render_tool_call(&call.name, &call.arguments, schema, &style);
            let reparsed: Value = serde_json::from_str(&json).unwrap();
            assert_eq!(
                reparsed,
                serde_json::from_str::<Value>(&rendered).unwrap(),
                "{label}"
            );
            assert_eq!(
                reparsed,
                json!({"name": call.name, "arguments": call.arguments})
            );
            names.insert(call.name);
        }
        assert_eq!(names.len(), 3, "{label}: {names:?}");
    }
}

#[test]
fn masked_greedy_and_cancellable_sampling() {
    for (label, tok) in all() {
        let (reg, g) = grammar(&tok, GrammarConfig::default());
        let cancel = g.special_ids().tool_cancel.unwrap();
        assert_eq!(cancel, tok.special_ids().tool_cancel);
        let mut rng = SplitMix64::new(7);
        let params = SamplingParams {
            temperature: 1.0,
            top_k: Some(20),
            top_p: Some(0.95),
        };
        let (mut closed, mut aborted) = (0, 0);
        for i in 0..60 {
            let p = if i % 2 == 0 {
                SamplingParams::GREEDY
            } else {
                params
            };
            let toks = run_span(&g, &mut rng, &p, 600);
            if *toks.last().unwrap() == cancel {
                aborted += 1;
                let ev = parse_span(&reg, &tok, &toks);
                assert!(
                    matches!(ev.last(), Some(ParseEvent::ToolCallAborted { .. })),
                    "{label}: {ev:?}"
                );
            } else {
                closed += 1;
                valid_call(&reg, &tok, &toks);
            }
        }
        assert_eq!(closed + aborted, 60);
    }
}

#[test]
fn whole_string_encodings_of_rendered_calls_are_admissible() {
    // Training data encodes a call's JSON as one string; the grammar (which
    // forces canonical tokens) must allow exactly that token sequence.
    let cases = [
        ("get_weather", json!({"unit": "celsius", "city": "Paris"})),
        ("get_weather", json!({"city": "Zürich"})),
        ("get_weather", json!({"city": "日本 €"})),
        (
            "set_timer",
            json!({"loud": true, "seconds": 90, "label": "tea"}),
        ),
        (
            "search",
            json!({"where": {"near": true, "lat": 48.5}, "tags": ["sale", "eco"], "query": "shoes", "max_price": 20}),
        ),
        ("search", json!({"query": "a\"b\\c", "tags": []})),
    ];
    for (label, tok) in all() {
        assert!(tok.is_canonical(), "{label}");
        let (reg, g) = grammar(&tok, GrammarConfig::default());
        for (name, args) in &cases {
            let schema = reg.get(name).map(|t| &t.parameters);
            let text = render_tool_call(name, args, schema, &JsonStyle::default());
            walk(&g, &tok, &text);
        }
    }
}

#[test]
fn forced_prefix_uses_the_tokenizers_own_tokens() {
    for (label, tok) in all() {
        let (_reg, g) = grammar(&tok, GrammarConfig::default());
        let mut m = g.start();
        let mask = m.compute_mask().unwrap();
        let set: Vec<u32> = (0..g.vocab_size() as u32)
            .filter(|&t| allowed(&mask, t))
            .filter(|&t| Some(t) != g.special_ids().tool_cancel)
            .collect();
        // Every call starts with `{"name": "`; with canonical tokenization
        // llguidance allows only the start of its encoding.
        let forced = tok.encode("{\"name\": \"");
        assert_eq!(set, forced[..1], "{label}");
        let v: Value = json!(tok.decode(&set));
        assert!(v.as_str().unwrap().starts_with('{'));
    }
}
