//! CPU real-time factor of Mimi encode + decode (informational).
//!
//! ```text
//! cargo run -p s2s-mimi --example mimi_rtf            # full-size model, synthetic weights
//! S2S_MIMI_WEIGHTS=/path/to/tokenizer-e351c8d8-checkpoint125.safetensors \
//!     cargo run -p s2s-mimi --example mimi_rtf        # the published checkpoint
//! ```
//!
//! The workspace builds `s2s-mimi` and `s2s-kernels` at `opt-level = 3` even
//! in the dev profile, so no release build is needed. Single-threaded.

use std::collections::BTreeMap;
use std::time::Instant;

use s2s_core::SplitMix64;
use s2s_mimi::{Mimi, MimiConfig, TensorMap};

/// Full-size Mimi with deterministic random weights, using the parameter
/// names/shapes of Kyutai's default config (`fixtures/mimi/full_v0_1_names.json`).
fn synthetic() -> Mimi {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/mimi/full_v0_1_names.json"
    );
    let names: BTreeMap<String, Vec<usize>> =
        serde_json::from_str(&std::fs::read_to_string(path).expect("manifest")).expect("json");
    let mut rng = SplitMix64::new(2026);
    let mut m = TensorMap::new();
    for (name, shape) in names {
        let n: usize = shape.iter().product();
        let fan_in: usize = shape.iter().skip(1).product::<usize>().max(1);
        let leaf = name.rsplit('.').next().unwrap_or_default().to_string();
        let vals: Vec<f32> = (0..n)
            .map(|_| match leaf.as_str() {
                "cluster_usage" | "weight_g" | "_initialized" => 1.0,
                "scale" => 0.3,
                "weight" if name.contains(".norm") => 1.0,
                "bias" => 0.0,
                _ => (rng.uniform(-1.0, 1.0) / (fan_in as f64).sqrt()) as f32,
            })
            .collect();
        m.insert(name, shape, vals);
    }
    Mimi::from_source(&m, MimiConfig::v0_1()).expect("full-size names load")
}

fn main() {
    let mimi = match std::env::var("S2S_MIMI_WEIGHTS") {
        Ok(p) => Mimi::load(&p, MimiConfig::v0_1()).expect("load checkpoint"),
        Err(_) => synthetic(),
    };
    let fs = mimi.config().frame_size();
    let frames = 50; // 4 s
    let pcm: Vec<f32> = (0..frames * fs)
        .map(|i| (0.3 * (i as f64 * 2.0 * std::f64::consts::PI * 220.0 / 24_000.0).sin()) as f32)
        .collect();
    let secs = pcm.len() as f64 / mimi.config().sample_rate;

    let t0 = Instant::now();
    let codes = mimi.encode(&pcm).expect("encode");
    let t_enc = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let _ = mimi.decode(&codes).expect("decode");
    let t_dec = t0.elapsed().as_secs_f64();
    println!(
        "offline, {secs:.1} s of audio: encode {t_enc:.2} s (RTF {:.3}), decode {t_dec:.2} s (RTF {:.3}), total RTF {:.3}",
        t_enc / secs,
        t_dec / secs,
        (t_enc + t_dec) / secs
    );

    let mut enc = mimi.streaming_encoder();
    let mut dec = mimi.streaming_decoder();
    let (mut te, mut td) = (0.0, 0.0);
    for f in 0..frames {
        let t0 = Instant::now();
        let c = enc
            .push_frame(&pcm[f * fs..(f + 1) * fs])
            .expect("encode step");
        te += t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let _ = dec.push_frame(&c).expect("decode step");
        td += t0.elapsed().as_secs_f64();
    }
    let n = frames as f64;
    println!(
        "streaming, per 80 ms frame: encode {:.1} ms, decode {:.1} ms, total RTF {:.3}",
        1e3 * te / n,
        1e3 * td / n,
        (te + td) / secs
    );
}
