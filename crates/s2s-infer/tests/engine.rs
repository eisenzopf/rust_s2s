//! Engine tests: the duplex model behind the server's `Engine` trait.

use std::sync::{Arc, Mutex};

use s2s_core::streams::{special, StreamKind, StreamLayout, FRAME_SAMPLES};
use s2s_core::SplitMix64;
use s2s_data::TextTokenizer;
use s2s_duplex::stream::SubstepHook;
use s2s_duplex::{DuplexConfig, DuplexModel};
use s2s_infer::tokenizer::ByteTokenizer;
use s2s_infer::{DuplexEngine, EngineConfig, ToneCodec};
use s2s_server::engine::{silent_frame, Injection};
use s2s_server::protocol::ToolState;
use s2s_server::{Engine, EngineEvent, FrameInput, FrameOutput, Mode, SessionConfig};
use serde_json::json;

fn model() -> Arc<DuplexModel<f32>> {
    let tok = ByteTokenizer;
    let pad = TextTokenizer::special_id(&tok, special::PAD).unwrap();
    let cfg = DuplexConfig::tiny(
        StreamLayout::default(),
        TextTokenizer::vocab_size(&tok) as usize,
        pad,
    );
    Arc::new(DuplexModel::new(cfg, &mut SplitMix64::new(7)))
}

fn engine() -> DuplexEngine<ByteTokenizer> {
    DuplexEngine::new(
        model(),
        Arc::new(ByteTokenizer),
        || Box::new(ToneCodec::new(8)),
        EngineConfig::default(),
    )
    .unwrap()
}

fn weather_tool() -> serde_json::Value {
    json!({
        "name": "weather",
        "description": "Current weather.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    })
}

/// Forces assistant text from a script (checking every forced token is
/// allowed by the logits the engine shows, i.e. by the grammar mask), pads
/// the user transcript and emits silent assistant audio.
struct Scripted {
    tm: std::collections::VecDeque<u32>,
    pad: u32,
    layout: StreamLayout,
    masked_steps: Arc<Mutex<usize>>,
}

impl SubstepHook<f32> for Scripted {
    fn choose(&mut self, k: usize, logits: &mut [f32]) -> u32 {
        match self.layout.kind(k) {
            StreamKind::UserText => self.pad,
            StreamKind::AssistantText(_) => {
                let t = self.tm.pop_front().unwrap_or(self.pad);
                assert!(
                    logits[t as usize] > f32::NEG_INFINITY,
                    "forced token {t} is masked out"
                );
                if logits.contains(&f32::NEG_INFINITY) {
                    *self.masked_steps.lock().unwrap() += 1;
                }
                t
            }
            _ => 0,
        }
    }
}

fn run_frames(
    e: &mut DuplexEngine<ByteTokenizer>,
    session: u64,
    frames: std::ops::Range<u64>,
    audio: f32,
) -> Vec<EngineEvent> {
    let mut events = Vec::new();
    for f in frames {
        let mut a = silent_frame();
        a.iter_mut().for_each(|x| *x = audio);
        let mut inputs = vec![FrameInput {
            session,
            frame_index: f,
            audio: a,
            injections: Vec::new(),
        }];
        let mut outputs = vec![FrameOutput::new()];
        e.step(&mut inputs, &mut outputs).unwrap();
        assert_eq!(outputs[0].audio.len(), FRAME_SAMPLES);
        events.append(&mut outputs[0].events);
    }
    events
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tool_call_dispatches_runs_and_injects_the_result() {
    let tok = ByteTokenizer;
    let id = move |n| TextTokenizer::special_id(&tok, n).unwrap();
    let call = s2s_tools::render_tool_call(
        "weather",
        &json!({"city": "Paris"}),
        Some(&weather_tool()["parameters"]),
        &s2s_tools::JsonStyle::default(),
    );
    let mut tm: Vec<u32> = TextTokenizer::encode(&tok, "One sec.");
    tm.push(id(special::TOOL_CALL));
    tm.extend(TextTokenizer::encode(&tok, &call));
    tm.push(id(special::TOOL_CALL_END));
    let masked = Arc::new(Mutex::new(0));
    let masked2 = masked.clone();
    let tm = Mutex::new(Some(tm));
    let mut e = engine().with_hook_factory(move |_| {
        Box::new(Scripted {
            tm: tm.lock().unwrap().take().unwrap_or_default().into(),
            pad: id(special::PAD),
            layout: StreamLayout::default(),
            masked_steps: masked2.clone(),
        })
    });
    e.register_tool("weather", |args: serde_json::Value| async move {
        assert_eq!(args["city"], "Paris");
        Ok(json!({"temp": 18}))
    });
    e.open_session(
        1,
        &SessionConfig {
            system_prompt: "Be brief.".into(),
            tools: vec![weather_tool()],
            ..Default::default()
        },
    )
    .unwrap();

    // 4 text slots per frame: the call takes ~15 frames; then let the
    // handler finish and the result come back.
    let mut events = run_frames(&mut e, 1, 0..30, 0.0);
    for f in 30..60 {
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        events.extend(run_frames(&mut e, 1, f..f + 1, 0.0));
    }

    let text: String = events
        .iter()
        .filter_map(|ev| match ev {
            EngineEvent::AssistantText(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "One sec.");
    let states: Vec<ToolState> = events
        .iter()
        .filter_map(|ev| match ev {
            EngineEvent::ToolStatus { call_id, state } if call_id == "c_1" => Some(*state),
            _ => None,
        })
        .collect();
    assert_eq!(
        states,
        [
            ToolState::Dispatched,
            ToolState::Running,
            ToolState::Injected
        ]
    );
    // The grammar constrained every token inside the span.
    assert!(*masked.lock().unwrap() >= call.len());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn client_tool_round_trip() {
    let tok = ByteTokenizer;
    let id = move |n| TextTokenizer::special_id(&tok, n).unwrap();
    let client_tool = json!({"name": "lookup", "location": "client",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}});
    let call = s2s_tools::render_tool_call(
        "lookup",
        &json!({"q": "x"}),
        Some(&client_tool["parameters"]),
        &Default::default(),
    );
    let mut tm = vec![id(special::TOOL_CALL)];
    tm.extend(TextTokenizer::encode(&tok, &call));
    tm.push(id(special::TOOL_CALL_END));
    let tm = Mutex::new(Some(tm));
    let mut e = engine().with_hook_factory(move |_| {
        Box::new(Scripted {
            tm: tm.lock().unwrap().take().unwrap_or_default().into(),
            pad: id(special::PAD),
            layout: StreamLayout::default(),
            masked_steps: Arc::new(Mutex::new(0)),
        })
    });
    e.open_session(
        9,
        &SessionConfig {
            tools: vec![client_tool],
            ..Default::default()
        },
    )
    .unwrap();
    let events = run_frames(&mut e, 9, 0..20, 0.0);
    assert!(events.iter().any(|ev| matches!(ev,
        EngineEvent::ToolCall { call_id, name, arguments }
            if call_id == "c_1" && name == "lookup" && arguments == &json!({"q": "x"}))));

    // The client answers; the next tick injects it.
    let mut inputs = vec![FrameInput {
        session: 9,
        frame_index: 20,
        audio: silent_frame(),
        injections: vec![Injection::ToolResult {
            call_id: "c_1".into(),
            json: json!({"hits": 3}),
        }],
    }];
    let mut outputs = vec![FrameOutput::new()];
    e.step(&mut inputs, &mut outputs).unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    let mut events = outputs.pop().unwrap().events;
    events.extend(run_frames(&mut e, 9, 21..23, 0.0));
    assert!(events.iter().any(|ev| matches!(ev,
        EngineEvent::ToolStatus { call_id, state: ToolState::Injected } if call_id == "c_1")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stt_mode_is_silent_and_writes_no_assistant_text() {
    let mut e = engine();
    e.open_session(
        2,
        &SessionConfig {
            mode: Mode::Stt,
            ..Default::default()
        },
    )
    .unwrap();
    for f in 0..15 {
        let mut inputs = vec![FrameInput {
            session: 2,
            frame_index: f,
            audio: silent_frame(),
            injections: Vec::new(),
        }];
        let mut outputs = vec![FrameOutput::new()];
        e.step(&mut inputs, &mut outputs).unwrap();
        assert!(
            outputs[0].audio.iter().all(|&x| x == 0.0),
            "STT mode must not speak"
        );
        assert!(!outputs[0]
            .events
            .iter()
            .any(|ev| matches!(ev, EngineEvent::AssistantText(_))));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tts_mode_speaks_the_given_text_in_order() {
    let mut e = engine();
    e.open_session(
        3,
        &SessionConfig {
            mode: Mode::Tts,
            ..Default::default()
        },
    )
    .unwrap();
    let mut inputs = vec![FrameInput {
        session: 3,
        frame_index: 0,
        audio: silent_frame(),
        injections: vec![Injection::Text("hello brave world".into())],
    }];
    let mut outputs = vec![FrameOutput::new()];
    e.step(&mut inputs, &mut outputs).unwrap();
    let mut events = outputs.pop().unwrap().events;
    events.extend(run_frames(&mut e, 3, 1..200, 0.0));
    let text: String = events
        .iter()
        .filter_map(|ev| match ev {
            EngineEvent::AssistantText(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    // The model decides when each word starts (by sampling <epad>); every
    // word must come out, in order, exactly once.
    assert_eq!(text, "hello brave world");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rejects_bad_sessions() {
    let mut e = engine();
    let bad = SessionConfig {
        tools: vec![json!({"name": "x", "parameters": "not a schema"})],
        ..Default::default()
    };
    assert!(e.open_session(1, &bad).is_err());
    let mut wrong = DuplexEngine::new(
        model(),
        Arc::new(ByteTokenizer),
        || Box::new(ToneCodec::new(3)),
        EngineConfig::default(),
    )
    .unwrap();
    assert!(
        wrong.open_session(1, &SessionConfig::default()).is_err(),
        "codec/model codebook mismatch"
    );
}

fn reduced_mimi() -> Arc<s2s_mimi::Mimi> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/mimi");
    let cfg = s2s_mimi::MimiConfig::from_json(
        &std::fs::read_to_string(root.join("reduced_config.json")).unwrap(),
    )
    .unwrap();
    Arc::new(s2s_mimi::Mimi::load(root.join("reduced.safetensors"), cfg).unwrap())
}

#[test]
fn mimi_codec_matches_mimi_streaming() {
    use s2s_infer::FrameCodec;
    let mimi = reduced_mimi();
    let mut codec = s2s_infer::MimiCodec::new(mimi.clone());
    let mut enc = mimi.streaming_encoder();
    let mut dec = mimi.streaming_decoder();
    for f in 0..6 {
        let pcm: Vec<f32> = (0..FRAME_SAMPLES)
            .map(|n| ((n + f * FRAME_SAMPLES) as f32 * 0.03).sin() * 0.3)
            .collect();
        let codes = codec.encode(&pcm);
        assert_eq!(codes, enc.push_frame(&pcm).unwrap());
        assert_eq!(codec.decode(&codes), dec.push_frame(&codes).unwrap());
    }
    assert_eq!(codec.silence_codes().len(), codec.codebooks());
}

#[test]
fn engine_rejects_a_codec_with_the_wrong_codebook_size() {
    // The reduced test Mimi has 64-entry codebooks; the model's audio streams
    // are laid out for 2048. Mismatches must fail loudly at session start,
    // not decode out-of-range codes to silence.
    let mimi = reduced_mimi();
    let tok = ByteTokenizer;
    let pad = TextTokenizer::special_id(&tok, special::PAD).unwrap();
    let layout = StreamLayout {
        codebooks: 4,
        ..StreamLayout::default()
    };
    let cfg = DuplexConfig::tiny(layout, TextTokenizer::vocab_size(&tok) as usize, pad);
    let model = Arc::new(DuplexModel::new(cfg, &mut SplitMix64::new(11)));
    let m2 = mimi.clone();
    let mut e = DuplexEngine::new(
        model,
        Arc::new(ByteTokenizer),
        move || Box::new(s2s_infer::MimiCodec::new(m2.clone())),
        EngineConfig::default(),
    )
    .unwrap();
    let err = e.open_session(4, &SessionConfig::default()).unwrap_err();
    assert!(err.to_string().contains("entries per codebook"), "{err}");
}
