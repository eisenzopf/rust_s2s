use s2s_core::streams::{StreamKind, StreamLayout};
use s2s_data::synth::{Role, Start, ToolCallSpec, Turn};
use s2s_data::{
    assemble, synthesize, validate, AssemblyConfig, ByteTokenizer, Error, InjectionKind, Sample,
    Script, ShardReader, SpecialIds, SynthConfig, TextTokenizer, IGNORE_INDEX,
};

use crate::common::{fixture, write_shard};

fn weather() -> Script {
    Script::from_json(&std::fs::read_to_string(fixture("weather.json")).unwrap()).unwrap()
}

fn byte(c: char) -> u32 {
    ByteTokenizer::NUM_SPECIAL + c as u32
}

/// Frames and slots of every `Tm` cell holding `id`.
fn tm_cells(s: &Sample, id: u32) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for f in 0..s.frames {
        for slot in 0..s.layout.text_slots {
            if s.token(f, s.layout.index(StreamKind::AssistantText(slot))) == id {
                out.push((f, slot));
            }
        }
    }
    out
}

#[test]
fn tool_call_decodes_back_and_result_lands_at_the_right_frame() {
    let tok = ByteTokenizer;
    let sp = SpecialIds::from_tokenizer(&tok).unwrap();
    let script = weather();
    let syn = synthesize(&script, &tok, &SynthConfig::default()).unwrap();
    let s = &syn.sample;
    validate::check(s).unwrap();
    assert_eq!(s.layout, StreamLayout::default());
    assert_eq!(s.meta.id, "weather-001");
    assert_eq!(s.meta.license, "CC-BY-4.0");
    assert_eq!(s.meta.tokenizer, "bytes");

    // The call JSON between <tool_call> and </tool_call>, in reading order.
    let text = s.assistant_text();
    let open = text.iter().position(|&t| t == sp.tool_call).unwrap();
    let close = text.iter().position(|&t| t == sp.tool_call_end).unwrap();
    let json: serde_json::Value =
        serde_json::from_str(&tok.decode(&text[open + 1..close])).unwrap();
    let spec = script.turns[1].tool_call.as_ref().unwrap();
    assert_eq!(json["name"], serde_json::json!(spec.name));
    assert_eq!(json["arguments"], spec.arguments);
    assert_eq!(
        json["arguments"],
        serde_json::json!({"city": "Paris", "unit": "celsius"})
    );

    // Packed 4 tokens per frame: every frame strictly inside the call is full.
    let call = &syn.tool_calls[0];
    let start = tm_cells(s, sp.tool_call)[0];
    let end = tm_cells(s, sp.tool_call_end)[0];
    assert_eq!((call.start_frame, call.end_frame), (start.0, end.0));
    for f in start.0 + 1..end.0 {
        for slot in 0..4 {
            let id = s.token(f, s.layout.index(StreamKind::AssistantText(slot)));
            assert_ne!(id, sp.pad, "frame {f} slot {slot} inside the call is empty");
        }
    }
    let n_call = close - open + 1;
    assert_eq!(end.0 - start.0, (start.1 + n_call - 1) / 4);

    // Result injected latency (1.2 s = 15 frames) after the frame of </tool_call>.
    let result_frame = end.0 + 1 + 15;
    assert_eq!(call.result_frame, Some(result_frame));
    let res: Vec<_> = s
        .injections
        .iter()
        .filter(|i| i.kind == InjectionKind::ToolResult)
        .collect();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].frame as usize, result_frame);
    let decoded = tok.decode(&res[0].tokens);
    let inner = decoded
        .strip_prefix("<tool_result>")
        .and_then(|d| d.strip_suffix("</tool_result>"))
        .unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(inner).unwrap(),
        spec.result
    );

    // System prompt + tool schema at frame 0, first.
    assert_eq!(s.injections[0].kind, InjectionKind::System);
    assert_eq!(s.injections[0].frame, 0);
    let sys = tok.decode(&s.injections[0].tokens);
    assert!(sys.starts_with("<sys>You are a helpful voice assistant.\n{"));
    assert!(sys.ends_with("}</sys>") && sys.contains("get_weather"));

    // The answer turn starts 0.3 s (4 frames) after the result; its first
    // word's <epad> is one frame before.
    let tm0 = s.layout.index(StreamKind::AssistantText(0));
    assert_eq!(s.token(result_frame + 3, tm0), sp.epad);
    assert_eq!(s.token(result_frame + 4, tm0), byte(' '));
}

#[test]
fn text_streams_read_back_and_lead_audio() {
    let tok = ByteTokenizer;
    let sp = SpecialIds::from_tokenizer(&tok).unwrap();
    let syn = synthesize(&weather(), &tok, &SynthConfig::default()).unwrap();
    let s = &syn.sample;
    let call = &syn.tool_calls[0].call_json;
    assert_eq!(
        tok.decode(&s.assistant_text()),
        format!(
            "<epad>Let<epad> me<epad> check.<tool_call>{call}</tool_call>\
             <epad> It's<epad> 18<epad> degrees<epad> and<epad> sunny<epad> in<epad> Paris."
        )
    );
    assert_eq!(
        tok.decode(&s.user_text()),
        "what's the weather in Paris? mm-hmm"
    );

    // User turn at 0.2 s -> frame 3 (Tu is aligned; d_u is applied at assembly).
    assert_eq!(s.token(3, 0), byte('w'));
    assert!((0..3).all(|f| s.token(f, 0) == sp.pad));
    // Assistant turn: 1.8 s + 0.3 s gap = 2.1 s -> frame 26; <epad> at 25 and
    // the word's first token on the frame its audio starts.
    let tm0 = s.layout.index(StreamKind::AssistantText(0));
    assert_eq!(s.token(25, tm0), sp.epad);
    assert_eq!(s.token(26, tm0), byte('L'));
    // Monologue words use slot 0 only; slots 1..3 hold only tool-call tokens.
    let call_frames = syn.tool_calls[0].start_frame..=syn.tool_calls[0].end_frame;
    for f in 0..s.frames {
        if !call_frames.contains(&f) {
            for slot in 1..4 {
                assert_eq!(
                    s.token(f, s.layout.index(StreamKind::AssistantText(slot))),
                    sp.pad
                );
            }
        }
    }
    // Word side table: user words on stream 0, assistant words on Tm slot 0.
    let users: Vec<_> = s
        .words
        .iter()
        .filter(|w| w.stream == 0)
        .map(|w| w.text.as_str())
        .collect();
    assert_eq!(
        users,
        ["what's", "the", "weather", "in", "Paris?", "mm-hmm"]
    );
    assert_eq!(s.words[0].start_ms, 200);
    assert_eq!(
        s.words.iter().filter(|w| w.stream == tm0 as u32).count(),
        10
    );
    // The byte tokenizer is slow (one token per frame), so some text lags.
    assert!(!syn.warnings.is_empty());
}

#[test]
fn precomputed_codes_and_silence_are_placed_exactly() {
    let script = Script {
        silence_codes: Some(vec![0, 1]),
        turns: vec![
            Turn {
                speaker: Role::User,
                text: "hi".into(),
                start: Some(Start::At(0.0)),
                codes: Some(vec![vec![1, 2], vec![3, 4], vec![5, 6]]),
                ..Turn::default()
            },
            Turn {
                speaker: Role::Assistant,
                text: "yo".into(),
                start: Some(Start::At(0.4)),
                codes: Some(vec![vec![7, 8], vec![9, 10]]),
                ..Turn::default()
            },
        ],
        ..Script::default()
    };
    let layout = StreamLayout {
        text_slots: 2,
        codebooks: 2,
        user_text_delay: 1,
        acoustic_delay: 1,
    };
    let cfg = SynthConfig {
        layout,
        tail: 0.0,
        ..SynthConfig::default()
    };
    let s = synthesize(&script, &ByteTokenizer, &cfg).unwrap().sample;
    assert_eq!(s.frames, 7);
    let col = |kind| {
        (0..s.frames)
            .map(|f| s.token(f, layout.index(kind)))
            .collect::<Vec<_>>()
    };
    assert_eq!(col(StreamKind::UserAudio(0)), [1, 3, 5, 0, 0, 0, 0]);
    assert_eq!(col(StreamKind::UserAudio(1)), [2, 4, 6, 1, 1, 1, 1]);
    assert_eq!(col(StreamKind::AssistantAudio(0)), [0, 0, 0, 0, 0, 7, 9]);
    assert_eq!(col(StreamKind::AssistantAudio(1)), [1, 1, 1, 1, 1, 8, 10]);
    let (p, e) = (0, 1); // <pad>, <epad>
    assert_eq!(
        col(StreamKind::UserText),
        [byte('h'), byte('i'), p, p, p, p, p]
    );
    assert_eq!(
        col(StreamKind::AssistantText(0)),
        [p, p, p, p, e, byte('y'), byte('o')]
    );
    assert!(s.injections.is_empty());
}

#[test]
fn interruption_cuts_audio_and_words() {
    let script = Script::from_json(
        r#"{"seed": 3, "silence_codes": [5, 5, 5, 5, 5, 5, 5, 5], "turns": [
            {"speaker": "assistant", "text": "one two three four five", "start": {"at": 0.0},
             "duration": 2.0, "cut_off_at": 0.9},
            {"speaker": "user", "text": "wait", "start": {"after": -0.3}, "duration": 0.5}
        ]}"#,
    )
    .unwrap();
    let s = synthesize(&script, &ByteTokenizer, &SynthConfig::default())
        .unwrap()
        .sample;
    let am0 = s.layout.index(StreamKind::AssistantAudio(0));
    // 0.9 s -> 11 frames of assistant audio, silence afterwards.
    assert!((11..s.frames).all(|f| s.token(f, am0) == 5));
    let words: Vec<_> = s
        .words
        .iter()
        .map(|w| (w.stream, w.text.as_str()))
        .collect();
    // Assistant ends at 11 frames = 0.88 s; the user barges in 0.3 s earlier.
    // The side table is sorted by start time.
    assert_eq!(words, [(1, "one"), (1, "two"), (0, "wait"), (1, "three")]);
    assert_eq!(s.words[2].start_ms, 580);
}

#[test]
fn cancelled_call_emits_tool_cancel_and_drops_result() {
    let tok = ByteTokenizer;
    let sp = SpecialIds::from_tokenizer(&tok).unwrap();
    let script = Script {
        turns: vec![Turn {
            speaker: Role::Assistant,
            text: "One moment.".into(),
            tool_call: Some(ToolCallSpec {
                id: Some("c_1".into()),
                name: "lookup".into(),
                arguments: serde_json::json!({"q": "x"}),
                result: serde_json::json!({"ok": true}),
                latency: 2.0,
                start: None,
                cancel_after: Some(0.5),
            }),
            ..Turn::default()
        }],
        ..Script::default()
    };
    let syn = synthesize(&script, &tok, &SynthConfig::default()).unwrap();
    let call = &syn.tool_calls[0];
    assert_eq!(call.result_frame, None);
    assert_eq!(call.cancel_frame, Some(call.end_frame + 1 + 6));
    assert!(call
        .call_json
        .starts_with("{\"id\":\"c_1\",\"name\":\"lookup\""));
    assert!(syn.sample.injections.is_empty());
    assert_eq!(
        tm_cells(&syn.sample, sp.tool_cancel),
        [(call.end_frame + 7, 0)]
    );
}

#[test]
fn inconsistent_scripts_are_errors() {
    let bad = [
        // Two user turns overlapping.
        r#"{"turns": [{"speaker": "user", "text": "a b", "start": {"at": 0.0}, "duration": 1.0},
                      {"speaker": "user", "text": "c", "start": {"at": 0.5}, "duration": 1.0}]}"#,
        // No tool result to start after.
        r#"{"turns": [{"speaker": "assistant", "text": "x", "start": {"after_tool_result": 0.1}}]}"#,
        // Users do not call tools.
        r#"{"turns": [{"speaker": "user", "text": "x", "tool_call": {"name": "t"}}]}"#,
        // Empty turn.
        r#"{"turns": [{"speaker": "user", "text": ""}]}"#,
        // Wrong number of codes per frame.
        r#"{"turns": [{"speaker": "user", "text": "x", "codes": [[1, 2]]}]}"#,
        // Out-of-order assistant turns.
        r#"{"turns": [{"speaker": "assistant", "text": "x", "start": {"at": 2.0}},
                      {"speaker": "assistant", "text": "y", "start": {"at": 1.0}}]}"#,
        // Negative start.
        r#"{"turns": [{"speaker": "user", "text": "x", "start": {"after": -1.0}}]}"#,
    ];
    for (i, json) in bad.iter().enumerate() {
        let script = Script::from_json(json).unwrap();
        let res = synthesize(&script, &ByteTokenizer, &SynthConfig::default());
        assert!(matches!(res, Err(Error::Script(_))), "case {i}: {res:?}");
    }
    assert!(Script::from_json("{\"turns\": [{\"speaker\": \"robot\"}]}").is_err());
}

#[test]
fn synthesis_is_deterministic_and_seeded() {
    let tok = ByteTokenizer;
    let cfg = SynthConfig::default();
    let a = synthesize(&weather(), &tok, &cfg).unwrap();
    let b = synthesize(&weather(), &tok, &cfg).unwrap();
    assert_eq!(a, b);
    let mut other = weather();
    other.seed = 8;
    let c = synthesize(&other, &tok, &cfg).unwrap();
    assert_ne!(a.sample.tokens, c.sample.tokens);
    assert_eq!(a.sample.assistant_text(), c.sample.assistant_text());
}

#[test]
fn synthesized_sample_through_shard_and_assembly() {
    let tok = ByteTokenizer;
    let syn = synthesize(&weather(), &tok, &SynthConfig::default()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = write_shard(&dir.path().join("w.s2s"), std::slice::from_ref(&syn.sample));
    let s = ShardReader::open(&path).unwrap().get(0).unwrap();
    assert_eq!(s, syn.sample);
    assert_eq!(
        s.meta.extra["tool_calls"][0]["end_frame"],
        syn.tool_calls[0].end_frame
    );

    let ex = assemble(&s, &AssemblyConfig::for_sample(&s)).unwrap();
    let ns = ex.num_streams;
    let k = ex.num_generated;
    let tm0 = s.layout.index(StreamKind::AssistantText(0));
    assert_eq!(
        ex.is_injection.iter().filter(|&&m| m == 1).count(),
        s.injection_tokens()
    );
    // The tool result run sits right before delayed row `result_frame`.
    let rf = syn.tool_calls[0].result_frame.unwrap() as i32;
    let row = ex.frame_index.iter().position(|&f| f == rf).unwrap();
    let res = &s.injections[1];
    let first = row - res.tokens.len();
    assert!(ex.is_injection[first..row].iter().all(|&m| m == 1));
    assert_eq!(ex.frame_index[first - 1], rf - 1);
    let injected: Vec<u32> = (first..row).map(|p| ex.tokens[p * ns + tm0]).collect();
    assert_eq!(injected, res.tokens);
    // Nothing predicts injected tokens; the last one predicts row `rf`.
    for p in first - 1..row - 1 {
        assert!(ex.targets[p * k..(p + 1) * k]
            .iter()
            .all(|&t| t == IGNORE_INDEX));
    }
    assert_eq!(
        ex.targets[(row - 1) * k + tm0],
        ex.tokens[row * ns + tm0] as i64
    );
    assert!(ex.weights[(row - 1) * k + tm0] > 0.0);
}
