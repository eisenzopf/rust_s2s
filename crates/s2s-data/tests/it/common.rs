//! Helpers shared by the integration tests.

use std::path::{Path, PathBuf};

use s2s_core::streams::{StreamLayout, AUDIO_VOCAB};
use s2s_core::SplitMix64;
use s2s_data::{
    Injection, InjectionKind, Sample, SampleMeta, ShardWriter, SpeakerInfo, WordTiming,
};

pub fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

pub fn random_layout(rng: &mut SplitMix64) -> StreamLayout {
    StreamLayout {
        text_slots: 1 + rng.below(4) as usize,
        codebooks: 1 + rng.below(4) as usize,
        user_text_delay: rng.below(4) as usize,
        acoustic_delay: rng.below(3) as usize,
    }
}

fn random_string(rng: &mut SplitMix64, max: usize) -> String {
    const PIECES: [&str; 10] = ["a", "Z", " ", "é", "日本", "🙂", "\"", "\\", "\n", "x9"];
    (0..rng.below(max as u64 + 1))
        .map(|_| PIECES[rng.below(PIECES.len() as u64) as usize])
        .collect()
}

/// A random but valid sample.
pub fn random_sample(rng: &mut SplitMix64, layout: StreamLayout) -> Sample {
    let frames = 1 + rng.below(40) as usize;
    let text_vocab = 10 + rng.below(200) as u32;
    let text_pad = rng.below(text_vocab as u64) as u32;
    let ns = layout.num_streams();
    let kinds = layout.kinds();
    let mut tokens = Vec::with_capacity(frames * ns);
    for _ in 0..frames {
        for k in &kinds {
            let limit = if k.is_text() { text_vocab } else { AUDIO_VOCAB };
            tokens.push(rng.below(limit as u64) as u32);
        }
    }
    let mut inj_frames: Vec<u32> = (0..rng.below(5))
        .map(|_| rng.below(frames as u64 + 1) as u32)
        .collect();
    inj_frames.sort_unstable();
    let kinds_inj = [
        InjectionKind::System,
        InjectionKind::ToolResult,
        InjectionKind::Text,
    ];
    let injections = inj_frames
        .into_iter()
        .map(|frame| Injection {
            frame,
            kind: kinds_inj[rng.below(3) as usize],
            tokens: (0..1 + rng.below(12))
                .map(|_| rng.below(text_vocab as u64) as u32)
                .collect(),
        })
        .collect();
    let words = (0..rng.below(6))
        .map(|_| {
            let start_ms = rng.below(10_000) as u32;
            WordTiming {
                stream: if rng.below(2) == 0 { 0 } else { 1 },
                start_ms,
                end_ms: start_ms + rng.below(800) as u32,
                text: random_string(rng, 6),
            }
        })
        .collect();
    let extra = if rng.below(2) == 0 {
        serde_json::Value::Null
    } else {
        serde_json::json!({"k": random_string(rng, 4), "n": rng.below(100)})
    };
    Sample {
        layout,
        frames,
        text_vocab,
        text_pad,
        tokens,
        injections,
        words,
        meta: SampleMeta {
            id: format!("sample-{}", rng.below(1_000_000)),
            source: random_string(rng, 8),
            license: ["CC-BY-4.0", "CC0", "LDC", ""][rng.below(4) as usize].to_string(),
            speakers: vec![SpeakerInfo {
                role: "user".into(),
                id: random_string(rng, 5),
            }],
            tokenizer: "test".into(),
            extra,
        },
    }
}

/// Write `samples` to `path` and return the path.
pub fn write_shard(path: &Path, samples: &[Sample]) -> PathBuf {
    let mut w = ShardWriter::create(path).unwrap();
    for s in samples {
        w.append(s).unwrap();
    }
    w.finish().unwrap();
    path.to_path_buf()
}
