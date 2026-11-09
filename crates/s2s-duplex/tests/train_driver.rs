//! The training driver: runs on shards and resumes bit-exactly.

use std::path::{Path, PathBuf};

use s2s_data::{synthesize, ByteTokenizer, Script, ShardWriter, SynthConfig};
use s2s_duplex::train::{run, Preset, TrainArgs};
use s2s_nn::{Module, Param};

fn make_shard(dir: &Path) -> PathBuf {
    let path = dir.join("d.s2s");
    let mut w = ShardWriter::create(&path).unwrap();
    for (i, city) in ["Rome", "Kyiv"].iter().enumerate() {
        let script = Script::from_json(&format!(
            r#"{{"id": "s{i}", "seed": {i}, "turns": [
                {{"speaker": "user", "text": "hi from {city}", "start": {{"at": 0.1}}, "duration": 0.8}},
                {{"speaker": "assistant", "text": "hello {city}", "start": {{"after": 0.2}}, "duration": 0.8}}
            ]}}"#
        ))
        .unwrap();
        w.append(
            &synthesize(&script, &ByteTokenizer, &SynthConfig::default())
                .unwrap()
                .sample,
        )
        .unwrap();
    }
    w.finish().unwrap();
    path
}

fn args(shard: PathBuf, stop_after: Option<u64>, ckpt: Option<PathBuf>) -> TrainArgs {
    TrainArgs {
        shards: vec![shard],
        preset: Preset::Tiny,
        steps: 8,
        stop_after,
        batch: 2,
        window: 32,
        peak_lr: 3e-3,
        min_lr: 3e-4,
        warmup: 2,
        max_grad_norm: 1.0,
        weight_decay: 0.1,
        seed: 5,
        checkpoint: ckpt,
        save_every: 0,
        activation_checkpointing: false,
    }
}

fn params(m: &impl Module<f32>) -> Vec<Vec<f32>> {
    let mut v = Vec::new();
    m.visit_params(&mut |p: &Param<f32>| v.push(p.value.clone()));
    v
}

#[test]
fn resume_is_bit_exact_including_the_loader_position() {
    let dir = std::env::temp_dir().join(format!("s2s-duplex-driver-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let shard = make_shard(&dir);

    let mut losses = Vec::new();
    let straight = run(&args(shard.clone(), None, None), &mut |s| {
        losses.push(s.loss)
    })
    .unwrap();
    assert_eq!((straight.start_step, straight.end_step), (0, 8));
    assert!(losses.iter().all(|l| l.is_finite()));

    let ckpt = dir.join("run.safetensors");
    let first = run(
        &args(shard.clone(), Some(4), Some(ckpt.clone())),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(first.end_step, 4);
    let second = run(&args(shard, None, Some(ckpt)), &mut |_| {}).unwrap();
    assert_eq!((second.start_step, second.end_step), (4, 8));
    let (a, b) = (params(&straight.model), params(&second.model));
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        if let Some(j) = x.iter().zip(y).position(|(p, q)| p != q) {
            panic!("resumed run diverged: param {i}[{j}] {} vs {}", x[j], y[j]);
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn windows_without_targets_do_not_cost_optimizer_steps() {
    // The weather dialogue starts with a long injected system prompt (tool
    // schema, byte tokenizer): small windows can fall entirely inside it.
    let dir = std::env::temp_dir().join(format!("s2s-duplex-empty-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let script = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../s2s-data/tests/data/weather.json"),
    )
    .unwrap();
    let path = dir.join("w.s2s");
    let mut w = ShardWriter::create(&path).unwrap();
    let syn = synthesize(
        &Script::from_json(&script).unwrap(),
        &ByteTokenizer,
        &SynthConfig::default(),
    )
    .unwrap();
    w.append(&syn.sample).unwrap();
    w.finish().unwrap();
    let mut a = args(path, None, None);
    a.window = 24;
    a.steps = 12;
    let mut logged = Vec::new();
    let summary = run(&a, &mut |s| logged.push((s.step, s.loss))).unwrap();
    assert_eq!(summary.end_step, 12);
    assert_eq!(logged.len(), 12);
    for (step, loss) in logged {
        assert!(loss > 0.0, "step {step} had no training signal");
    }
    std::fs::remove_dir_all(&dir).ok();
}
