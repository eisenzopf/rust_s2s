//! Correctness tests for the multistream model.

use s2s_core::streams::{StreamKind, StreamLayout};
use s2s_core::SplitMix64;
use s2s_duplex::stream::SubstepHook;
use s2s_duplex::{DuplexBatch, DuplexConfig, DuplexModel};
use s2s_kernels::reference::loss::IGNORE_INDEX;
use s2s_nn::block::Checkpoint;
use s2s_nn::{Module, Param};

/// A small layout: 2 text slots, 2 codebooks -> S = 7 streams, K = 5 generated.
/// Audio keeps the real vocabulary (2048 codes + pad).
fn small_cfg() -> DuplexConfig {
    let layout = StreamLayout {
        text_slots: 2,
        codebooks: 2,
        user_text_delay: 1,
        acoustic_delay: 1,
    };
    let mut c = DuplexConfig::tiny(layout, 11, 0);
    c.d_model = 8;
    c.q_heads = 2;
    c.kv_heads = 1;
    c.mlp_hidden = 12;
    c.depth_d = 8;
    c.depth_heads = 2;
    c.depth_mlp_hidden = 10;
    c
}

fn random_batch(cfg: &DuplexConfig, b: usize, t: usize, seed: u64) -> DuplexBatch<f64> {
    let mut r = SplitMix64::new(seed);
    let (s, k) = (cfg.layout.num_streams(), cfg.layout.num_generated());
    let tokens: Vec<u32> = (0..b * t * s)
        .map(|i| r.below(cfg.stream_vocab(i % s) as u64) as u32)
        .collect();
    let targets: Vec<i64> = (0..b * t * k)
        .map(|i| {
            if r.below(7) == 0 {
                IGNORE_INDEX
            } else {
                r.below(cfg.stream_vocab(i % k) as u64) as i64
            }
        })
        .collect();
    let weights: Vec<f64> = (0..b * t * k).map(|_| r.uniform(0.2, 1.5)).collect();
    let mut is_injection = vec![0u8; b * t];
    is_injection[1] = 1;
    is_injection[t + 2] = 1;
    let positions = (0..b).flat_map(|_| 0..t as i64).collect();
    DuplexBatch {
        batch: b,
        window: t,
        tokens,
        targets,
        weights,
        is_injection,
        positions,
        segments: None,
    }
}

#[test]
fn every_parameter_tensor_matches_finite_differences() {
    let cfg = small_cfg();
    let batch = random_batch(&cfg, 2, 4, 1);
    let mut model = DuplexModel::<f64>::new(cfg, &mut SplitMix64::new(2));
    model.zero_grad();
    let (_, _, ctx) = model.forward_loss(&batch, Checkpoint::Off);
    model.backward(ctx);
    let mut analytic = Vec::new();
    model.visit_params(&mut |p: &Param<f64>| analytic.push((p.name.clone(), p.grad.clone())));

    let loss_of = |m: &DuplexModel<f64>| m.forward_loss(&batch, Checkpoint::Off).0;
    let mut pick = SplitMix64::new(3);
    let h = 1e-6;
    let mut checked = 0;
    for (pi, (name, g)) in analytic.iter().enumerate() {
        for _ in 0..4.min(g.len()) {
            let i = pick.below(g.len() as u64) as usize;
            let perturbed = |delta: f64| {
                let mut m = model.clone();
                let mut n = 0;
                m.visit_params_mut(&mut |p| {
                    if n == pi {
                        p.value[i] += delta;
                    }
                    n += 1;
                });
                loss_of(&m)
            };
            let numeric = (perturbed(h) - perturbed(-h)) / (2.0 * h);
            let tol = 1e-6 + 1e-4 * numeric.abs().max(g[i].abs());
            assert!(
                (numeric - g[i]).abs() <= tol,
                "{name}[{i}]: analytic {:.9e} vs numeric {numeric:.9e}",
                g[i]
            );
            checked += 1;
        }
    }
    assert!(checked > 80, "only {checked} entries checked");
    // The injection embedding actually received gradient.
    let inj = analytic.iter().find(|(n, _)| n == "embed.inject").unwrap();
    assert!(inj.1.iter().any(|&g| g != 0.0));
}

#[test]
fn checkpointing_gives_identical_gradients() {
    let cfg = small_cfg();
    let batch = random_batch(&cfg, 2, 5, 4);
    let base = DuplexModel::<f64>::new(cfg, &mut SplitMix64::new(5));
    let run = |ckpt| {
        let mut m = base.clone();
        m.zero_grad();
        let (l, _, ctx) = m.forward_loss(&batch, ckpt);
        m.backward(ctx);
        let mut g = Vec::new();
        m.visit_params(&mut |p: &Param<f64>| g.push(p.grad.clone()));
        (l, g)
    };
    assert_eq!(run(Checkpoint::Off), run(Checkpoint::On));
}

/// Records the logits it is shown and forces given tokens.
struct Forced {
    tokens: Vec<u32>,
    seen: Vec<Vec<f64>>,
}

impl SubstepHook<f64> for Forced {
    fn choose(&mut self, k: usize, logits: &mut [f64]) -> u32 {
        self.seen.push(logits.to_vec());
        self.tokens[k]
    }
}

#[test]
fn streaming_reproduces_teacher_forced_logits() {
    // One sequence: rows 0..6 with an injected token at position 3. Targets
    // are the next position's generated tokens (what streaming would have
    // sampled), except position 2, whose successor is the injection.
    let cfg = small_cfg();
    let l = cfg.layout;
    let (s, k) = (l.num_streams(), l.num_generated());
    let t = 7;
    let mut r = SplitMix64::new(9);
    let mut rows: Vec<Vec<u32>> = (0..t)
        .map(|_| {
            (0..s)
                .map(|i| r.below(cfg.stream_vocab(i) as u64) as u32)
                .collect()
        })
        .collect();
    let injected_token = 5u32;
    let mut inj_row = cfg.pad_row();
    inj_row[l.index(StreamKind::AssistantText(0))] = injected_token;
    rows[3] = inj_row;
    let mut targets = vec![IGNORE_INDEX; t * k];
    for p in 0..t - 1 {
        if p == 2 {
            continue; // the injection is never predicted
        }
        for j in 0..k {
            targets[p * k + j] = rows[p + 1][j] as i64;
        }
    }
    let mut is_injection = vec![0u8; t];
    is_injection[3] = 1;
    let batch = DuplexBatch {
        batch: 1,
        window: t,
        tokens: rows.concat(),
        targets: targets.clone(),
        weights: vec![1.0; t * k],
        is_injection,
        positions: (0..t as i64).collect(),
        segments: None,
    };
    let model = DuplexModel::<f64>::new(cfg, &mut SplitMix64::new(10));
    let (_, _, ctx) = model.forward_loss(&batch, Checkpoint::Off);

    let mut st = model.new_state();
    let mut compared = 0;
    for p in 0..t - 1 {
        if p == 3 {
            model.feed_injection(&mut st, &[injected_token]);
        } else {
            model.feed_row(&mut st, &rows[p]);
        }
        if p == 2 {
            continue; // the runtime injects before sampling the next row
        }
        let forced: Vec<u32> = (0..k).map(|j| targets[p * k + j] as u32).collect();
        let mut hook = Forced {
            tokens: forced.clone(),
            seen: Vec::new(),
        };
        let out = model.sample_next(&mut st, &mut hook);
        assert_eq!(out, forced);
        for j in 0..k {
            let v = model.heads[j].d_out();
            let full = &ctx.logits(j)[p * v..(p + 1) * v];
            for (a, b) in hook.seen[j].iter().zip(full) {
                assert!((a - b).abs() < 1e-10, "position {p} stream {j}: {a} vs {b}");
            }
            compared += 1;
        }
    }
    assert_eq!(compared, (t - 2) * k);
}

/// End to end on CPU: synthesize tool dialogues with `s2s-data`, write a
/// shard, load batches with the real loader, and train a tiny duplex model.
#[test]
fn trains_on_synthetic_tool_dialogues() {
    use s2s_data::{
        synthesize, AssemblyConfig, ByteTokenizer, Dataset, Loader, LoaderConfig, Script,
        ShardReader, ShardWriter, SpecialIds, SynthConfig, TextTokenizer, WindowMode,
    };
    use s2s_kernels::reference::optim::AdamWConfig;
    use s2s_train::optim::{default_groups, AdamW};

    let tok = ByteTokenizer;
    let sp = SpecialIds::from_tokenizer(&tok).unwrap();
    let dir = std::env::temp_dir().join(format!("s2s-duplex-train-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("synthetic.s2s");
    let mut writer = ShardWriter::create(&path).unwrap();
    let cities = ["Paris", "Oslo", "Lima"];
    for (i, city) in cities.iter().enumerate() {
        let script = Script::from_json(&format!(
            r#"{{
              "id": "w{i}", "seed": {i},
              "system": "Be brief.",
              "tools": [{{"name": "weather", "parameters": {{"type": "object",
                         "properties": {{"city": {{"type": "string"}}}}, "required": ["city"]}}}}],
              "turns": [
                {{"speaker": "user", "text": "weather in {city}?", "start": {{"at": 0.2}}, "duration": 1.0}},
                {{"speaker": "assistant", "text": "One sec.", "start": {{"after": 0.2}}, "duration": 0.6,
                  "tool_call": {{"name": "weather", "arguments": {{"city": "{city}"}},
                                "result": {{"temp": {t}}}, "latency": 0.5}}}},
                {{"speaker": "assistant", "text": "It is {t} degrees.", "start": {{"after_tool_result": 0.2}}, "duration": 1.2}}
              ]
            }}"#,
            t = 10 + 5 * i
        ))
        .unwrap();
        let syn = synthesize(&script, &tok, &SynthConfig::default()).unwrap();
        writer.append(&syn.sample).unwrap();
    }
    writer.finish().unwrap();

    let layout = StreamLayout::default();
    let dataset = std::sync::Arc::new(Dataset::new(vec![ShardReader::open(&path).unwrap()]));
    // Pack mode: windows hold several conversations; segment ids keep them
    // from attending to each other.
    let mut lcfg = LoaderConfig::new(AssemblyConfig::new(layout, sp.pad), 2, 96);
    lcfg.mode = WindowMode::Pack;
    let mut loader = Loader::new(dataset, lcfg).unwrap();

    let cfg = DuplexConfig::tiny(layout, tok.vocab_size() as usize, sp.pad);
    let mut model = DuplexModel::<f32>::new(cfg, &mut SplitMix64::new(1));
    let mut opt = AdamW::new(
        AdamWConfig {
            weight_decay: 0.0,
            ..Default::default()
        },
        Some(1.0),
        default_groups,
    );
    let mut losses = Vec::new();
    for _ in 0..120 {
        let batch = DuplexBatch::<f32>::from_data(&loader.next_batch().unwrap());
        model.zero_grad();
        let (_, report, ctx) = model.forward_loss(&batch, Checkpoint::Off);
        model.backward(ctx);
        opt.step(&mut model, 3e-3);
        losses.push(report.loss);
    }
    let first: f64 = losses[..5].iter().sum::<f64>() / 5.0;
    let last: f64 = losses[losses.len() - 5..].iter().sum::<f64>() / 5.0;
    assert!(last.is_finite() && first.is_finite());
    assert!(
        last < 0.5 * first,
        "loss only fell from {first:.3} to {last:.3}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn packed_segments_do_not_see_each_other() {
    // Window = [seq A (3 positions) | seq B (4 positions)]. B's logits must
    // equal running B alone, and A's must not depend on B.
    let cfg = small_cfg();
    let (s, k) = (cfg.layout.num_streams(), cfg.layout.num_generated());
    let mut r = SplitMix64::new(21);
    let rows = |n: usize, r: &mut SplitMix64| -> Vec<u32> {
        (0..n * s)
            .map(|i| r.below(cfg.stream_vocab(i % s) as u64) as u32)
            .collect()
    };
    let a = rows(3, &mut r);
    let b = rows(4, &mut r);
    let model = DuplexModel::<f64>::new(cfg, &mut SplitMix64::new(22));
    let targets = |n: usize| vec![IGNORE_INDEX; n * k];
    let packed = DuplexBatch {
        batch: 1,
        window: 7,
        tokens: [a.clone(), b.clone()].concat(),
        targets: targets(7),
        weights: vec![1.0; 7 * k],
        is_injection: vec![0; 7],
        positions: vec![0, 1, 2, 0, 1, 2, 3],
        segments: Some(vec![1, 1, 1, 2, 2, 2, 2]),
    };
    let alone = DuplexBatch {
        batch: 1,
        window: 4,
        tokens: b,
        targets: targets(4),
        weights: vec![1.0; 4 * k],
        is_injection: vec![0; 4],
        positions: vec![0, 1, 2, 3],
        segments: None,
    };
    let (_, _, cp) = model.forward_loss(&packed, Checkpoint::Off);
    let (_, _, ca) = model.forward_loss(&alone, Checkpoint::Off);
    for j in 0..k {
        let v = model.heads[j].d_out();
        let packed_b = &cp.logits(j)[3 * v..7 * v];
        for (x, y) in packed_b.iter().zip(ca.logits(j)) {
            assert!((x - y).abs() < 1e-10, "stream {j}: {x} vs {y}");
        }
    }
}
