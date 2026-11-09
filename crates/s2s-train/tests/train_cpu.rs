//! End-to-end CPU training checks for the trainer stack.

use s2s_core::SplitMix64;
use s2s_kernels::reference::optim::AdamWConfig;
use s2s_nn::block::Checkpoint;
use s2s_nn::lm::{LmConfig, TransformerLm};
use s2s_nn::{Module, Param};
use s2s_train::checkpoint::{self, CheckpointError};
use s2s_train::data::{CharCorpus, DEMO_TEXT};
use s2s_train::optim::{default_groups, AdamW};
use s2s_train::schedule::LrSchedule;
use s2s_train::trainer::{TrainConfig, Trainer};

fn trainer(seed: u64, ckpt: Checkpoint, total: u64) -> Trainer {
    let data = CharCorpus::new(DEMO_TEXT, seed);
    let model = TransformerLm::new(
        LmConfig::tiny(data.vocab_size()),
        &mut SplitMix64::new(seed),
    );
    Trainer {
        model,
        opt: AdamW::new(
            AdamWConfig {
                weight_decay: 0.01,
                ..Default::default()
            },
            Some(1.0),
            default_groups,
        ),
        data,
        cfg: TrainConfig {
            batch: 8,
            seq: 32,
            grad_accum: 1,
            schedule: LrSchedule::WarmupCosine {
                warmup: 10,
                total,
                peak: 1e-2,
                min: 1e-3,
            },
            checkpoint: ckpt,
        },
    }
}

fn params(m: &TransformerLm<f32>) -> Vec<Vec<f32>> {
    let mut v = Vec::new();
    m.visit_params(&mut |p: &Param<f32>| v.push(p.value.clone()));
    v
}

#[test]
fn tiny_lm_learns_the_corpus() {
    let mut t = trainer(1, Checkpoint::Off, 300);
    let first = t.step().loss;
    let mut last = first;
    for _ in 0..299 {
        last = t.step().loss;
    }
    let uniform = (t.data.vocab_size() as f32).ln();
    assert!(
        (first - uniform).abs() < 0.7,
        "initial loss {first} vs ln(V) {uniform}"
    );
    // A 2-layer, d=32 model memorizes much of this short text in 300 steps.
    assert!(last < 1.4, "loss only fell from {first} to {last}");
}

#[test]
fn resume_from_checkpoint_is_bit_exact() {
    let dir = std::env::temp_dir().join(format!("s2s-train-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("ckpt.safetensors");

    // Reference: 30 uninterrupted steps.
    let mut a = trainer(5, Checkpoint::On, 30);
    for _ in 0..30 {
        a.step();
    }

    // 15 steps, save, load into a *fresh* trainer (different init), 15 more.
    let mut b = trainer(5, Checkpoint::On, 30);
    for _ in 0..15 {
        b.step();
    }
    checkpoint::save(&path, &b.model, &b.opt, &b.manifest()).unwrap();
    let mut c = trainer(999, Checkpoint::On, 30);
    let manifest = checkpoint::load(&path, &mut c.model, &mut c.opt).unwrap();
    c.data.restore(manifest.data);
    assert_eq!(manifest.step, 15);
    for _ in 0..15 {
        c.step();
    }

    assert_eq!(params(&a.model), params(&c.model), "resumed run diverged");
    assert_eq!(a.opt.state, c.opt.state);
    assert_eq!(a.data.cursor(), c.data.cursor());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn checkpoint_rejects_a_different_architecture() {
    let t = trainer(3, Checkpoint::Off, 10);
    let bytes = checkpoint::to_safetensors(&t.model, &t.opt, &t.manifest()).unwrap();
    let mut cfg = LmConfig::tiny(t.data.vocab_size());
    cfg.d_model = 48;
    cfg.q_heads = 4;
    let mut other = TransformerLm::<f32>::new(cfg, &mut SplitMix64::new(0));
    let mut opt = AdamW::new(AdamWConfig::default(), None, default_groups);
    match checkpoint::from_safetensors(&bytes, &mut other, &mut opt) {
        Err(CheckpointError::Mismatch(_)) => {}
        other => panic!("expected a mismatch error, got {other:?}"),
    }
}

#[test]
fn gradient_accumulation_matches_one_big_batch() {
    // Two micro-batches of 4 == one batch of 8 drawn from the same sampler.
    let mut big = trainer(9, Checkpoint::Off, 10);
    big.cfg.batch = 8;
    let mut acc = trainer(9, Checkpoint::Off, 10);
    acc.cfg.batch = 4;
    acc.cfg.grad_accum = 2;
    let rb = big.step();
    let ra = acc.step();
    assert!(
        (rb.loss - ra.loss).abs() < 1e-5,
        "{} vs {}",
        rb.loss,
        ra.loss
    );
    assert!((rb.stats.grad_norm - ra.stats.grad_norm).abs() < 1e-4);
}
