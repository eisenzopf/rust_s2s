//! GPU Kyutai LM vs. the CPU port.
//!
//! The fixture tests run the reduced Kyutai checkpoints in
//! `fixtures/kyutai-lm` (contexts of 5–6 frames, so the rotating KV cache
//! wraps) for more frames than fit the cache. With f32 weights the GPU must
//! match the CPU to f32 rounding and pick the same greedy tokens; with bf16
//! weights it must stay close. They need an sm_80+ GPU, so they are
//! `#[ignore]`d:
//!
//! ```text
//! cargo test -p s2s-gpu --features cuda -- --ignored
//! ```
//!
//! `real_moshi_matches_cpu` also needs `S2S_KYUTAI_LM_WEIGHTS` (Moshi's
//! `model.safetensors`, e.g. `kyutai/moshiko-candle-bf16`).

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::time::Instant;

use s2s_gpu::kyutai::{load_lm, GpuLmOptions, WeightDtype};
use s2s_model::kyutai::{Error, LmConfig, LmModel, Sampler};

fn fixture(name: &str) -> (LmConfig, PathBuf) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/kyutai-lm")
        .join(name);
    (
        LmConfig::from_path(dir.join("config.json")).unwrap(),
        dir.join("model.safetensors"),
    )
}

struct Frame {
    hidden: Vec<f32>,
    text_logits: Vec<f32>,
    dep_tokens: Vec<u32>,
    dep_logits: Vec<Vec<f32>>,
}

/// `frames` steps with deterministic inputs, greedy text and depformer.
fn run(model: &mut LmModel, frames: usize) -> Vec<Frame> {
    let cfg = model.config().clone();
    let mut text = model.text_start_token();
    let mut out = Vec::with_capacity(frames);
    for f in 0..frames {
        let audio: Vec<Option<u32>> = (0..model.in_audio_codebooks())
            .map(|c| Some(((f * 7 + c * 3 + 1) % cfg.audio_vocab_size) as u32))
            .collect();
        let step = model.step(Some(text), &audio, None).unwrap();
        // Depformer inputs are fixed too (text token and forced audio
        // tokens), so a near-tie flipping one greedy choice cannot make the
        // runs diverge; the sampled tokens are still compared.
        let dep_text = (f % cfg.text_out_vocab_size) as u32;
        let forced: Vec<Option<u32>> = (0..model.generated_audio_codebooks())
            .map(|s| Some(((f + 3 * s) % (cfg.audio_vocab_size - 1)) as u32))
            .collect();
        let dep = model
            .depformer_sample(
                &step.hidden,
                Some(dep_text),
                &forced,
                &mut Sampler::greedy(),
            )
            .unwrap();
        // Feed a fixed, in-vocabulary text token so CPU and GPU see the same
        // inputs even if a greedy choice differs.
        text = (f as u32 * 5 + 1) % cfg.text_in_vocab_size as u32;
        out.push(Frame {
            hidden: step.hidden,
            text_logits: step.text_logits,
            dep_tokens: dep.as_ref().map(|d| d.tokens.clone()).unwrap_or_default(),
            dep_logits: dep.map(|d| d.logits).unwrap_or_default(),
        });
    }
    out
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

/// Compare GPU and CPU runs; returns the worst relative error.
fn compare(name: &str, gpu: &[Frame], cpu: &[Frame], tol: f32, same_tokens: bool) -> f32 {
    let mut worst = 0f32;
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        let eh = max_rel(&g.hidden, &c.hidden);
        let el = max_rel(&g.text_logits, &c.text_logits);
        let ed = g
            .dep_logits
            .iter()
            .zip(&c.dep_logits)
            .fold(0f32, |m, (a, b)| m.max(max_rel(a, b)));
        worst = worst.max(eh).max(el).max(ed);
        assert!(
            eh <= tol && el <= tol && ed <= tol,
            "{name} frame {i}: rel err hidden {eh:e}, text logits {el:e}, depformer logits {ed:e} (tol {tol:e})"
        );
        if same_tokens {
            assert_eq!(
                g.dep_tokens, c.dep_tokens,
                "{name} frame {i}: depformer tokens"
            );
        }
    }
    println!(
        "{name}: {} frames, worst relative error {worst:e}",
        gpu.len()
    );
    worst
}

fn fixture_parity(name: &str) {
    let (cfg, path) = fixture(name);
    let frames = 3 * cfg.transformer.context + 2;
    let mut cpu = LmModel::load_safetensors(&cfg, &path).unwrap();
    let want = run(&mut cpu, frames);

    let f32_opts = GpuLmOptions {
        dtype: WeightDtype::F32,
        ..Default::default()
    };
    let mut gpu = load_lm(&cfg, &path, f32_opts).unwrap();
    assert!(gpu.is_accelerated());
    let got = run(&mut gpu, frames);
    compare(&format!("{name} f32"), &got, &want, 1e-4, true);

    // Reset: the same inputs give the same outputs again.
    gpu.reset_state();
    assert_eq!(gpu.current_seq_len(), 0);
    let again = run(&mut gpu, frames);
    compare(
        &format!("{name} f32 after reset"),
        &again,
        &want,
        1e-4,
        true,
    );

    let mut gpu = load_lm(&cfg, &path, GpuLmOptions::default()).unwrap();
    let got = run(&mut gpu, frames);
    compare(&format!("{name} bf16"), &got, &want, 5e-2, false);
}

#[test]
#[ignore = "needs an sm_80+ GPU"]
fn moshi_fixture_matches_cpu() {
    fixture_parity("moshi");
}

#[test]
#[ignore = "needs an sm_80+ GPU"]
fn stt_fixture_matches_cpu() {
    fixture_parity("stt");
}

#[test]
#[ignore = "needs an sm_80+ GPU"]
fn unsupported_config_is_rejected() {
    // LayerNorm, sinusoidal positions, biases, GELU.
    let (cfg, path) = fixture("variants");
    match load_lm(&cfg, &path, GpuLmOptions::default()) {
        Err(Error::Unsupported(m)) => println!("rejected as expected: {m}"),
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
#[ignore = "needs an sm_80+ GPU and S2S_KYUTAI_LM_WEIGHTS"]
fn real_moshi_matches_cpu() {
    let weights = std::env::var("S2S_KYUTAI_LM_WEIGHTS").expect("S2S_KYUTAI_LM_WEIGHTS");
    let cfg = LmConfig::v0_1_streaming(8);
    let t0 = Instant::now();
    let mut gpu = load_lm(&cfg, &weights, GpuLmOptions::default()).unwrap();
    println!("GPU load {:.1} s", t0.elapsed().as_secs_f64());
    // Warm up (module JIT, cuBLASLt heuristics), then time full frames.
    run(&mut gpu, 3);
    gpu.reset_state();
    let frames = 50;
    let t0 = Instant::now();
    let got = run(&mut gpu, frames);
    let per = t0.elapsed().as_secs_f64() / frames as f64;
    println!(
        "GPU: {:.1} ms per frame (temporal step + 8 depformer slices), real-time budget 80 ms",
        per * 1e3
    );

    let n = std::env::var("S2S_CPU_FRAMES").map_or(3, |v| v.parse().unwrap());
    let mut cpu = LmModel::load_safetensors(&cfg, &weights).unwrap();
    let want = run(&mut cpu, n);
    compare("moshi 7B bf16 vs CPU f32", &got[..n], &want, 5e-2, false);
}
