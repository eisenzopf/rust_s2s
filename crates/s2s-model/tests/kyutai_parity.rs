//! Parity with Kyutai's own implementation on fixtures from
//! `tools/kyutai-ref-lm` (Kyutai's `moshi` 0.6.4 crate, candle 0.9.1, CPU f32;
//! reduced configs of the real architectures with seeded random weights).
//!
//! Fixtures (`fixtures/kyutai-lm/<name>/`):
//! - `moshi`: Moshi v0.1 layout (text + 8 generated + 8 input codebooks,
//!   acoustic delay 2, depformer with 8 slices, RmsNorm, SiLU gating, RoPE).
//! - `stt`: DSM STT layout (no depformer, extra heads, continuous `delay`
//!   conditioner, RoPE with max_period 1e5), through the ASR driver.
//! - `variants`: LayerNorm, ungated GELU MLP, biases, layer scale, sinusoidal
//!   positions, depformer with RoPE / GELU gating / low-rank embeddings, a LUT
//!   conditioner.
//!
//! All contexts are shorter than the sequences, so the rotating KV cache wraps.
//! Logits must match within [`LOGIT_TOL`]; every sampled token must match
//! exactly.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use s2s_model::kyutai::{
    AsrConfig, AsrMsg, AsrState, Condition, LmConfig, LmModel, MultistreamConfig, MultistreamState,
    SafeTensors, Sampler, TensorMap, WeightSource,
};
use serde_json::Value;

/// Max absolute logit difference allowed (logits are O(10)).
const LOGIT_TOL: f32 = 1e-4;
/// Max absolute probability difference for extra heads.
const PROB_TOL: f32 = 1e-5;

fn fixture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/kyutai-lm")
        .join(name)
}

struct Fixture {
    lm_cfg: LmConfig,
    driver: Value,
    trace: Value,
    dir: PathBuf,
}

fn load_fixture(name: &str) -> Fixture {
    let dir = fixture_dir(name);
    let cfg: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let trace: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("trace.json")).unwrap()).unwrap();
    let lm_cfg = LmConfig::from_path(dir.join("config.json")).unwrap();
    assert_eq!(lm_cfg, serde_json::from_value(cfg["lm"].clone()).unwrap());
    Fixture {
        lm_cfg,
        driver: cfg["driver"].clone(),
        trace,
        dir,
    }
}

impl Fixture {
    fn model(&self) -> LmModel {
        LmModel::load_safetensors(&self.lm_cfg, self.dir.join("model.safetensors")).unwrap()
    }

    fn lut_condition(&self, model: &LmModel) -> Option<Condition> {
        let c = self.driver.get("lut_condition")?.as_array()?;
        let cp = model.condition_provider().unwrap();
        Some(
            cp.condition_lut(c[0].as_str().unwrap(), c[1].as_str().unwrap())
                .unwrap(),
        )
    }
}

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn f32s(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn opt_u32s(v: &Value) -> Vec<Option<u32>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().map(|t| t as u32))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Feed the recorded per-frame inputs to `LmModel::step` and
/// `depformer_sample`; compare logits and tokens. Returns (max text logit
/// error, max depformer logit error).
fn lm_parity(name: &str) -> (f32, f32) {
    let fx = load_fixture(name);
    let mut model = fx.model();
    let cond = fx.lut_condition(&model);
    let mut sampler = Sampler::greedy();
    let (mut text_err, mut dep_err) = (0.0f32, 0.0f32);
    for (i, f) in fx.trace["frames"].as_array().unwrap().iter().enumerate() {
        let text_in = f["text_in"].as_u64().map(|t| t as u32);
        let audio: Vec<Option<u32>> = u32s(&f["audio_in"]).into_iter().map(Some).collect();
        let out = model.step(text_in, &audio, cond.as_ref()).unwrap();
        text_err = text_err.max(max_abs_diff(&out.text_logits, &f32s(&f["text_logits"])));
        let text_token = sampler.sample(&out.text_logits).unwrap();
        assert_eq!(
            text_token,
            f["text_token"].as_u64().unwrap() as u32,
            "{name} frame {i}"
        );
        let dep = model
            .depformer_sample(
                &out.hidden,
                Some(text_token),
                &opt_u32s(&f["forced"]),
                &mut sampler,
            )
            .unwrap()
            .expect("depformer");
        assert_eq!(dep.tokens, u32s(&f["audio_tokens"]), "{name} frame {i}");
        for (ours, theirs) in dep
            .logits
            .iter()
            .zip(f["depformer_logits"].as_array().unwrap())
        {
            dep_err = dep_err.max(max_abs_diff(ours, &f32s(theirs)));
        }
    }
    assert_eq!(
        model.current_seq_len(),
        fx.trace["frames"].as_array().unwrap().len()
    );
    println!(
        "{name}: max |text logit err| = {text_err:.3e}, max |depformer logit err| = {dep_err:.3e}"
    );
    assert!(
        text_err < LOGIT_TOL,
        "{name}: text logits off by {text_err}"
    );
    assert!(
        dep_err < LOGIT_TOL,
        "{name}: depformer logits off by {dep_err}"
    );
    (text_err, dep_err)
}

/// Run `MultistreamState` like Kyutai's `State` and compare its outputs.
fn driver_parity(name: &str) {
    let fx = load_fixture(name);
    let gen: MultistreamConfig = serde_json::from_value(fx.driver["gen"].clone()).unwrap();
    let model = fx.model();
    let cond = fx.lut_condition(&model);
    let user_audio: Vec<Vec<u32>> = fx.trace["user_audio"]
        .as_array()
        .unwrap()
        .iter()
        .map(u32s)
        .collect();
    let frames = user_audio.len();
    let mut state = MultistreamState::new(
        model,
        frames + 1,
        Sampler::greedy(),
        Sampler::greedy(),
        gen.clone(),
    );
    let mut prev = gen.text_start_token;
    for (i, ua) in user_audio.iter().enumerate() {
        let out = state.step(Some(prev), ua, None, cond.as_ref()).unwrap();
        let f = &fx.trace["frames"][i];
        assert_eq!(
            out.audio_input,
            u32s(&f["audio_in"]),
            "{name} frame {i} inputs"
        );
        assert!(max_abs_diff(&out.text_logits, &f32s(&f["text_logits"])) < LOGIT_TOL);
        prev = out.text_token;
    }
    assert_eq!(
        state.text_tokens(false),
        u32s(&fx.trace["state_text_tokens"]).as_slice()
    );
    let grid: Vec<Vec<u32>> = fx.trace["state_audio_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(u32s)
        .collect();
    assert_eq!(state.audio_tokens(false), grid.as_slice());
    let last = fx.trace["state_last_audio_tokens"]
        .as_array()
        .map(|_| u32s(&fx.trace["state_last_audio_tokens"]));
    assert_eq!(state.last_audio_tokens(), last);
    assert!(
        last.is_some(),
        "fixture should end with a complete audio frame"
    );
}

#[test]
fn moshi_lm_matches_kyutai() {
    lm_parity("moshi");
}

#[test]
fn moshi_driver_matches_kyutai_state() {
    driver_parity("moshi");
}

#[test]
fn variants_lm_matches_kyutai() {
    lm_parity("variants");
}

#[test]
fn variants_driver_matches_kyutai_state() {
    driver_parity("variants");
}

#[test]
fn stt_lm_matches_kyutai() {
    let fx = load_fixture("stt");
    let mut model = fx.model();
    let delay = fx.driver["conditioning_delay"].as_f64().unwrap() as f32;
    let cond = model
        .condition_provider()
        .unwrap()
        .condition_cont("delay", -delay)
        .unwrap();
    let mut sampler = Sampler::greedy_first();
    let (mut text_err, mut prob_err) = (0.0f32, 0.0f32);
    for (i, f) in fx.trace["frames"].as_array().unwrap().iter().enumerate() {
        let audio: Vec<Option<u32>> = u32s(&f["audio_in"]).into_iter().map(Some).collect();
        let out = model
            .step(
                Some(f["text_in"].as_u64().unwrap() as u32),
                &audio,
                Some(&cond),
            )
            .unwrap();
        text_err = text_err.max(max_abs_diff(&out.text_logits, &f32s(&f["text_logits"])));
        assert_eq!(
            sampler.sample(&out.text_logits).unwrap(),
            f["text_token"].as_u64().unwrap() as u32,
            "frame {i}"
        );
        for (h, theirs) in model
            .extra_heads(&out.hidden)
            .iter()
            .zip(f["extra_heads"].as_array().unwrap())
        {
            let m = h.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let z: f32 = h.iter().map(|v| (v - m).exp()).sum();
            let p: Vec<f32> = h.iter().map(|v| (v - m).exp() / z).collect();
            prob_err = prob_err.max(max_abs_diff(&p, &f32s(theirs)));
        }
    }
    println!(
        "stt: max |text logit err| = {text_err:.3e}, max |extra-head prob err| = {prob_err:.3e}"
    );
    assert!(text_err < LOGIT_TOL);
    assert!(prob_err < PROB_TOL);
}

#[test]
fn stt_driver_matches_kyutai_asr_state() {
    let fx = load_fixture("stt");
    let asr_cfg = AsrConfig {
        asr_delay_in_tokens: fx.driver["asr_delay_in_tokens"].as_u64().unwrap() as usize,
        conditioning_delay: fx.driver["conditioning_delay"].as_f64().map(|d| d as f32),
        conditioning_learnt_padding: false,
        temperature: Some(0.0),
    };
    let mut state = AsrState::from_config(fx.model(), &asr_cfg).unwrap();
    let mut msgs = vec![];
    for (i, codes) in fx.trace["codes"].as_array().unwrap().iter().enumerate() {
        msgs.extend(state.step_tokens(&u32s(codes)).unwrap());
        assert_eq!(
            state.text_token(),
            fx.trace["frames"][i]["text_token"].as_u64().unwrap() as u32
        );
    }
    let theirs: Vec<AsrMsg> = serde_json::from_value(fx.trace["asr_msgs"].clone()).unwrap();
    assert_eq!(msgs.len(), theirs.len());
    let mut words = 0;
    for (a, b) in msgs.iter().zip(&theirs) {
        match (a, b) {
            (
                AsrMsg::Step {
                    step_idx: i,
                    prs: p,
                },
                AsrMsg::Step {
                    step_idx: j,
                    prs: q,
                },
            ) => {
                assert_eq!(i, j);
                assert_eq!(p.len(), q.len());
                for (x, y) in p.iter().zip(q) {
                    assert!(max_abs_diff(x, y) < PROB_TOL);
                }
            }
            _ => {
                assert_eq!(a, b);
                words += 1;
            }
        }
    }
    assert!(words >= 2, "fixture should exercise Word and EndWord");
    // Reset gives the same transcript again.
    state.reset();
    let mut again = vec![];
    for codes in fx.trace["codes"].as_array().unwrap() {
        again.extend(state.step_tokens(&u32s(codes)).unwrap());
    }
    assert_eq!(again, msgs);
}

#[test]
fn fixture_checkpoints_have_exactly_the_needed_tensors() {
    for name in ["moshi", "stt", "variants"] {
        let fx = load_fixture(name);
        let st = SafeTensors::open(fx.dir.join("model.safetensors")).unwrap();
        let shapes: BTreeMap<String, Vec<usize>> = st
            .tensors()
            .iter()
            .map(|(k, v)| (k.clone(), v.shape.clone()))
            .collect();
        let check = LmModel::check_tensor_names(&fx.lm_cfg, &shapes).unwrap();
        assert!(check.unused.is_empty(), "{name}: {:?}", check.unused);
        assert_eq!(check.used, shapes.len());
    }
}

#[test]
fn loader_reports_missing_and_misshaped_tensors() {
    let fx = load_fixture("moshi");
    let mut st = SafeTensors::open(fx.dir.join("model.safetensors")).unwrap();
    let mut map = TensorMap::default();
    for n in st.names() {
        let shape = st.shape(&n).unwrap();
        let data = st.read(&n).unwrap();
        map.0.insert(n, (shape, data));
    }
    // A complete map loads and gives the same logits as the file.
    let mut a = LmModel::load(&fx.lm_cfg, &mut map.clone()).unwrap();
    let mut b = fx.model();
    let audio = vec![Some(1); 16];
    assert_eq!(
        a.step(Some(2), &audio, None).unwrap(),
        b.step(Some(2), &audio, None).unwrap()
    );

    let mut missing = map.clone();
    missing.0.remove("depformer.3.linear_out.weight");
    let err = LmModel::load(&fx.lm_cfg, &mut missing).unwrap_err();
    assert!(
        err.to_string().contains("depformer.3.linear_out.weight"),
        "{err}"
    );

    let mut bad = map.clone();
    bad.0.get_mut("text_linear.weight").unwrap().0 = vec![31, 48];
    let err = LmModel::load(&fx.lm_cfg, &mut bad).unwrap_err();
    assert!(err.to_string().contains("text_linear.weight"), "{err}");

    // Extra tensors are ignored but reported by the name check.
    let mut extra = map;
    extra
        .0
        .insert("something.else".into(), (vec![1], vec![0.0]));
    LmModel::load(&fx.lm_cfg, &mut extra).unwrap();
    let shapes = extra
        .0
        .iter()
        .map(|(k, v)| (k.clone(), v.0.clone()))
        .collect();
    let check = LmModel::check_tensor_names(&fx.lm_cfg, &shapes).unwrap();
    assert_eq!(check.unused, vec!["something.else".to_string()]);

    // Bad inputs are errors, not panics.
    let mut m = fx.model();
    assert!(m.step(Some(33), &[], None).is_err());
    assert!(m.step(None, &[Some(17)], None).is_err());
    assert!(m.step(None, &[None; 17], None).is_err());
}

// ---------------------------------------------------------------------------
// Full-size configs and real checkpoints (not run by default)
// ---------------------------------------------------------------------------

/// Checks that the tensor names and shapes Kyutai's code requests for its
/// full-size configs are exactly what our loader requests.
///
/// Generate the manifest with
/// `cargo run --manifest-path tools/kyutai-ref-lm/Cargo.toml -- manifests /tmp/kyutai-manifests.json`
/// and run with `S2S_KYUTAI_MANIFESTS=/tmp/kyutai-manifests.json cargo test -p s2s-model -- --ignored full_size`.
#[test]
#[ignore = "needs S2S_KYUTAI_MANIFESTS (from tools/kyutai-ref-lm)"]
fn full_size_manifests_are_accepted() {
    let path = std::env::var("S2S_KYUTAI_MANIFESTS").expect("S2S_KYUTAI_MANIFESTS");
    let all: BTreeMap<String, Value> =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let presets: BTreeMap<&str, LmConfig> = [
        ("v0_1", LmConfig::v0_1()),
        ("v0_1_streaming_8", LmConfig::v0_1_streaming(8)),
        ("v0_1_asr", LmConfig::v0_1_asr()),
        ("s2s_v0_1", LmConfig::s2s_v0_1()),
        ("s2s_2b_16rvq_202501", LmConfig::s2s_2b_16rvq_202501()),
        ("asr_v0_1_1b", LmConfig::asr_v0_1_1b()),
        ("asr_300m_202501", LmConfig::asr_300m_202501()),
        ("tts_202501", LmConfig::tts_202501()),
    ]
    .into_iter()
    .collect();
    for (name, entry) in &all {
        let cfg: LmConfig = serde_json::from_value(entry["config"].clone()).unwrap();
        if let Some(p) = presets.get(name.as_str()) {
            assert_eq!(&cfg, p, "{name}: our preset differs from Kyutai's");
        }
        let shapes: BTreeMap<String, Vec<usize>> =
            serde_json::from_value(entry["tensors"].clone()).unwrap();
        let has_cross_attention = cfg.transformer.cross_attention.is_some();
        match LmModel::check_tensor_names(&cfg, &shapes) {
            Ok(check) => {
                assert!(!has_cross_attention);
                assert!(check.unused.is_empty(), "{name}: unused {:?}", check.unused);
                assert_eq!(check.used, shapes.len());
                println!("{name}: all {} tensors accepted", shapes.len());
            }
            Err(e) => {
                assert!(has_cross_attention, "{name}: {e}");
                println!("{name}: rejected as expected ({e})");
            }
        }
    }
}

/// Runs a real Kyutai checkpoint.
///
/// - `S2S_KYUTAI_LM_WEIGHTS`: a candle-format safetensors file, e.g.
///   `model.safetensors` from HF `kyutai/moshiko-candle-bf16` (7B: needs
///   ~31 GB RAM in f32), `kyutai/stt-1b-en_fr-candle` or
///   `kyutai/stt-2.6b-en-candle`.
/// - `S2S_KYUTAI_LM_CONFIG`: the LM config (JSON/TOML; a `moshi-server`
///   config such as `configs/config-stt-en_fr-hf.toml` from
///   `kyutai-labs/delayed-streams-modeling` works). If unset, Moshi v0.1
///   streaming (`LmConfig::v0_1_streaming(8)`) is assumed.
///
/// Checks that all tensors load, that a few frames of greedy decoding are
/// finite and deterministic.
#[test]
#[ignore = "needs S2S_KYUTAI_LM_WEIGHTS (downloaded Kyutai checkpoint)"]
fn real_checkpoint_runs() {
    let weights = std::env::var("S2S_KYUTAI_LM_WEIGHTS").expect("S2S_KYUTAI_LM_WEIGHTS");
    let cfg = match std::env::var("S2S_KYUTAI_LM_CONFIG") {
        Ok(p) => LmConfig::from_path(p).unwrap(),
        Err(_) => LmConfig::v0_1_streaming(8),
    };
    let st = SafeTensors::open(&weights).unwrap();
    let shapes = st
        .tensors()
        .iter()
        .map(|(k, v)| (k.clone(), v.shape.clone()))
        .collect();
    let check = LmModel::check_tensor_names(&cfg, &shapes).unwrap();
    println!("{} tensors used, unused: {:?}", check.used, check.unused);
    let mut model = LmModel::load_safetensors(&cfg, &weights).unwrap();
    let run = |model: &mut LmModel| {
        model.reset_state();
        let mut text = model.text_start_token();
        let pad = model.audio_pad_token();
        let mut tokens = vec![];
        for _ in 0..4 {
            let audio = vec![Some(pad); model.in_audio_codebooks()];
            let cond = model
                .condition_provider()
                .and_then(|cp| cp.learnt_padding(cp.names()[0]).ok());
            let out = model.step(Some(text), &audio, cond.as_ref()).unwrap();
            assert!(out.text_logits.iter().all(|v| v.is_finite()));
            text = Sampler::greedy().sample(&out.text_logits).unwrap();
            let dep = model
                .depformer_sample(&out.hidden, Some(text), &[], &mut Sampler::greedy())
                .unwrap();
            tokens.push((text, dep.map(|d| d.tokens)));
        }
        tokens
    };
    let a = run(&mut model);
    println!("greedy tokens: {a:?}");
    assert_eq!(a, run(&mut model));
}

/// Transcribes real Mimi codes with a real DSM STT checkpoint.
///
/// - `S2S_KYUTAI_STT_WEIGHTS`: `model.safetensors` from HF
///   `kyutai/stt-1b-en_fr-candle` (or `kyutai/stt-2.6b-en-candle`).
/// - `S2S_KYUTAI_STT_CONFIG`: the matching `moshi-server` TOML config from
///   `kyutai-labs/delayed-streams-modeling` (`configs/config-stt-en_fr-hf.toml`).
/// - `S2S_KYUTAI_STT_CODES`: JSON `[[u32; codebooks]; frames]` of Mimi codes
///   (e.g. from Kyutai's Mimi or `s2s-mimi`).
///
/// Prints the emitted word token ids; checks they are in vocabulary.
#[test]
#[ignore = "needs S2S_KYUTAI_STT_WEIGHTS, S2S_KYUTAI_STT_CONFIG, S2S_KYUTAI_STT_CODES"]
fn real_stt_transcribes_codes() {
    let weights = std::env::var("S2S_KYUTAI_STT_WEIGHTS").expect("S2S_KYUTAI_STT_WEIGHTS");
    let cfg_text = std::fs::read_to_string(
        std::env::var("S2S_KYUTAI_STT_CONFIG").expect("S2S_KYUTAI_STT_CONFIG"),
    )
    .unwrap();
    let codes: Vec<Vec<u32>> = serde_json::from_str(
        &std::fs::read_to_string(
            std::env::var("S2S_KYUTAI_STT_CODES").expect("S2S_KYUTAI_STT_CODES"),
        )
        .unwrap(),
    )
    .unwrap();
    let (asr_cfg, lm_cfg) = AsrConfig::from_moshi_server_toml(&cfg_text).unwrap();
    let model = LmModel::load_safetensors(&lm_cfg, &weights).unwrap();
    let mut state = AsrState::from_config(model, &asr_cfg).unwrap();
    for frame in &codes {
        for m in state.step_tokens(&frame[..lm_cfg.audio_codebooks]).unwrap() {
            if let AsrMsg::Word {
                tokens, start_time, ..
            } = &m
            {
                assert!(tokens
                    .iter()
                    .all(|&t| (t as usize) < lm_cfg.text_out_vocab_size));
                println!("{start_time:7.2}s {tokens:?}");
            }
        }
    }
}
