//! End to end: s2s-data's synthesizer with the real tokenizer wrapper, and
//! the synthesized assistant text back through the runtime's parser.

use std::sync::Arc;

use s2s_core::streams::StreamKind;
use s2s_data::{synthesize, validate, InjectionKind, Script, SynthConfig, TextTokenizer};
use s2s_tokenizer::HfTokenizer;
use s2s_tools::{
    CallId, GrammarConfig, Marker, ParseEvent, ParserConfig, TokenPiece, TokenizerInfo,
    ToolCallGrammar, ToolCallParser, ToolRegistry, ToolSpec,
};
use serde_json::json;

use crate::common::all;
use crate::grammar::walk;

const SCRIPT: &str = r#"{
  "id": "weather-hf",
  "system": "You are a helpful voice assistant.",
  "tools": [
    {
      "name": "get_weather",
      "description": "Current weather for a city.",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}, "unit": {"enum": ["celsius", "fahrenheit"]}},
        "required": ["city"],
        "additionalProperties": false
      }
    }
  ],
  "seed": 7,
  "turns": [
    {"speaker": "user", "text": "what's the weather in Zürich?", "start": {"at": 0.2}, "duration": 1.6},
    {
      "speaker": "assistant",
      "text": "Let me check.",
      "start": {"after": 0.3},
      "duration": 0.8,
      "tool_call": {
        "name": "get_weather",
        "arguments": {"unit": "celsius", "city": "Zürich"},
        "result": {"temp": 18, "conditions": "sunny"},
        "latency": 1.2
      }
    },
    {"speaker": "user", "text": "mm-hmm", "start": {"with_previous": 0.4}, "duration": 0.3, "backchannel": true},
    {"speaker": "assistant", "text": "It's 18 degrees and sunny in Zürich.", "start": {"after_tool_result": 0.3}, "duration": 2.0}
  ]
}"#;

fn registry(script: &Script) -> Arc<ToolRegistry> {
    let specs: Vec<ToolSpec> = script
        .tools
        .iter()
        .map(|v| serde_json::from_value(v.clone()).unwrap())
        .collect();
    Arc::new(ToolRegistry::from_specs(specs).unwrap())
}

/// Feed assistant-stream tokens to the parser like the runtime: special
/// tokens as markers, ordinary tokens as bytes.
fn parse_tokens(reg: &Arc<ToolRegistry>, tok: &HfTokenizer, ids: &[u32]) -> Vec<ParseEvent> {
    let mut p = ToolCallParser::with_config(
        reg.clone(),
        ParserConfig {
            scan_text_markers: false,
            ..ParserConfig::default()
        },
    );
    let mut ev = Vec::new();
    for &id in ids {
        ev.extend(match TokenizerInfo::token(tok, id) {
            TokenPiece::Bytes(b) => p.push_bytes(&b),
            TokenPiece::Special(n) => p.push_marker(Marker::from_name(&n).expect("stream marker")),
        });
    }
    ev.extend(p.finish());
    ev
}

fn text_of(ev: &[ParseEvent]) -> String {
    ev.iter()
        .filter_map(|e| match e {
            ParseEvent::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn synthesized_sample_round_trips_through_the_parser() {
    let script = Script::from_json(SCRIPT).unwrap();
    let reg = registry(&script);
    for (label, tok) in all() {
        let syn = synthesize(&script, &*tok, &SynthConfig::default()).unwrap();
        let s = &syn.sample;
        validate::check(s).unwrap();
        assert_eq!(s.text_vocab as usize, tok.vocab_size(), "{label}");
        assert_eq!(s.text_pad, tok.special_ids().pad);
        assert_eq!(s.meta.tokenizer, tok.name());

        // Token-level parse of the assistant stream.
        let tm = s.assistant_text();
        let ev = parse_tokens(&reg, &tok, &tm);
        let calls: Vec<_> = ev
            .iter()
            .filter_map(|e| match e {
                ParseEvent::ToolCall(c) => Some(c.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 1, "{label}: {ev:?}");
        assert_eq!(calls[0].call_id, CallId::from("c_1"));
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            calls[0].arguments,
            json!({"city": "Zürich", "unit": "celsius"})
        );
        assert_eq!(
            text_of(&ev),
            "Let me check. It's 18 degrees and sunny in Zürich.",
            "{label}"
        );

        // Text-level parse (decoded text with marker names) agrees.
        let decoded = TextTokenizer::decode(&*tok, &tm);
        assert!(decoded.starts_with("<epad>Let<epad> me"), "{decoded}");
        let mut p = ToolCallParser::new(reg.clone());
        let mut ev2 = p.push_str(&decoded);
        ev2.extend(p.finish());
        assert_eq!(text_of(&ev2), text_of(&ev), "{label}");
        assert!(
            ev2.contains(&ParseEvent::ToolCall(calls[0].clone())),
            "{label}"
        );

        // The call text in the sample is the exact rendering, and the
        // grammar admits its token sequence.
        assert_eq!(
            syn.tool_calls[0].call_json,
            r#"{"name": "get_weather", "arguments": {"city": "Zürich", "unit": "celsius"}}"#
        );
        let g = ToolCallGrammar::new(&reg, tok.clone(), GrammarConfig::default()).unwrap();
        walk(&g, &tok, &syn.tool_calls[0].call_json);
        let sp = tok.special_ids();
        let open = tm.iter().position(|&t| t == sp.tool_call).unwrap();
        let close = tm.iter().position(|&t| t == sp.tool_call_end).unwrap();
        assert_eq!(
            &tm[open + 1..close],
            &tok.encode(&syn.tool_calls[0].call_json)[..]
        );

        // Words are the per-word encodings (leading space after the first).
        let user: Vec<u32> = s.user_text();
        let words = tok.encode_words("what's the weather in Zürich?".split(' '));
        let first: Vec<u32> = words.concat();
        assert_eq!(&user[..first.len()], &first[..], "{label}");

        // Injections: the system prompt and the tool result decode to the
        // runtime's text.
        let sys = s
            .injections
            .iter()
            .find(|i| i.kind == InjectionKind::System)
            .unwrap();
        assert_eq!(sys.tokens[0], sp.sys);
        assert_eq!(*sys.tokens.last().unwrap(), sp.sys_end);
        let sys_text = TextTokenizer::decode(&*tok, &sys.tokens);
        assert!(sys_text.starts_with("<sys>You are a helpful voice assistant."));
        assert!(sys_text.contains("get_weather"));
        let res = s
            .injections
            .iter()
            .find(|i| i.kind == InjectionKind::ToolResult)
            .unwrap();
        assert_eq!(
            TextTokenizer::decode(&*tok, &res.tokens),
            r#"<tool_result>{"call_id":"c_1","name":"get_weather","status":"ok","result":{"temp":18,"conditions":"sunny"}}</tool_result>"#
        );
        assert_eq!(Some(res.frame as usize), syn.tool_calls[0].result_frame);

        // Tool-call frames are packed into all text slots.
        let tm_streams: Vec<usize> = (0..s.layout.text_slots)
            .map(|i| s.layout.index(StreamKind::AssistantText(i)))
            .collect();
        let f = syn.tool_calls[0].start_frame + 1;
        if f < syn.tool_calls[0].end_frame {
            for &st in &tm_streams {
                assert_ne!(s.token(f, st), sp.pad, "{label}");
            }
        }
    }
}
