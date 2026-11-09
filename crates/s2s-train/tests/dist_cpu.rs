//! Data-parallel training on CPU threads: DDP and ZeRO-1 against
//! single-process training, and sharded checkpoints across world sizes.
//!
//! Tolerances (f32 model, AdamW, 6 steps unless noted):
//! - DDP (N ranks, per-rank micro-batches) vs. one process on the
//!   concatenated batch: per-step loss within `LOSS_TOL`, parameters within
//!   `PARAM_TOL` (max abs). Not bitwise: one process sums each weight
//!   gradient over all rows in one accumulation chain and divides by the
//!   global token count, while DDP sums per-rank partial gradients.
//! - DDP with world size 1 vs. one process: **bitwise**.
//! - ZeRO-1 vs. DDP at the same world size: **bitwise** (parameters, losses,
//!   grad norms and the gathered optimizer state).
//! - Resume at the saving world size: **bitwise** vs. an uninterrupted run.
//!   Resume at another world size: the loaded state is bitwise the saved
//!   state; the continued run matches within the DDP tolerances.

use std::path::{Path, PathBuf};

use s2s_core::SplitMix64;
use s2s_kernels::reference::optim::AdamWConfig;
use s2s_nn::block::Checkpoint;
use s2s_nn::lm::{LmConfig, TransformerLm};
use s2s_nn::{Module, Param};
use s2s_train::checkpoint;
use s2s_train::comm::{run_ranks, Communicator};
use s2s_train::data::{CharCorpus, DEMO_TEXT};
use s2s_train::dist::{DistTrainer, Strategy};
use s2s_train::optim::{default_groups, AdamW, Moments};
use s2s_train::schedule::LrSchedule;
use s2s_train::trainer::{TrainConfig, Trainer};

/// Global batch: divisible by world sizes 1, 2, 3 and 6.
const GLOBAL_BATCH: usize = 6;
const LOSS_TOL: f32 = 1e-5;
const PARAM_TOL: f32 = 2e-5;
/// Small buckets, so the tiny model is split over several of them.
const DDP: Strategy = Strategy::Ddp { bucket_elems: 2048 };

fn trainer(seed: u64) -> Trainer {
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
            // Low enough that clipping is active in early steps.
            Some(0.5),
            default_groups,
        ),
        data,
        cfg: TrainConfig {
            batch: GLOBAL_BATCH,
            seq: 24,
            grad_accum: 1,
            schedule: LrSchedule::WarmupCosine {
                warmup: 3,
                total: 20,
                peak: 1e-2,
                min: 1e-3,
            },
            checkpoint: Checkpoint::Off,
        },
    }
}

fn params(m: &TransformerLm<f32>) -> Vec<Vec<f32>> {
    let mut v = Vec::new();
    m.visit_params(&mut |p: &Param<f32>| v.push(p.value.clone()));
    v
}

fn max_abs_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()))
        .fold(0.0, f32::max)
}

/// What one rank (or a single process) ends with.
#[derive(Debug, PartialEq)]
struct Run {
    losses: Vec<f32>,
    norms: Vec<f64>,
    params: Vec<Vec<f32>>,
}

fn single(seed: u64, steps: usize) -> Run {
    let mut t = trainer(seed);
    let mut losses = Vec::new();
    let mut norms = Vec::new();
    for _ in 0..steps {
        let r = t.step();
        losses.push(r.loss);
        norms.push(r.stats.grad_norm);
    }
    Run {
        losses,
        norms,
        params: params(&t.model),
    }
}

/// `steps` steps at `world` ranks; returns every rank's run and the full
/// optimizer state (moments gathered over ranks).
fn parallel(
    seed: u64,
    world: usize,
    strategy: Strategy,
    steps: usize,
) -> (Vec<Run>, Vec<(String, Moments<f32>)>) {
    let out = run_ranks(world, |comm| {
        // Different init per rank: DistTrainer::new broadcasts rank 0's.
        let mut t = trainer(seed);
        if comm.rank() > 0 {
            t.model = TransformerLm::new(*t.model.config(), &mut SplitMix64::new(1000));
        }
        let mut d = DistTrainer::new(t, comm, strategy).unwrap();
        let mut losses = Vec::new();
        let mut norms = Vec::new();
        for _ in 0..steps {
            let r = d.step().unwrap();
            losses.push(r.loss);
            norms.push(r.stats.grad_norm);
        }
        let state: Vec<_> = d.trainer.opt.state.clone().into_iter().collect();
        let run = Run {
            losses,
            norms,
            params: params(&d.trainer.model),
        };
        (run, state)
    });
    let mut state = Vec::new();
    let mut runs = Vec::new();
    for (run, s) in out {
        runs.push(run);
        state.extend(s);
    }
    state.sort_by(|a, b| a.0.cmp(&b.0));
    if matches!(strategy, Strategy::Ddp { .. }) {
        // Replicated: keep rank 0's copy.
        state.dedup_by(|a, b| a.0 == b.0);
    }
    (runs, state)
}

fn assert_ranks_agree(runs: &[Run]) {
    for r in &runs[1..] {
        assert_eq!(r, &runs[0], "ranks diverged");
    }
}

fn assert_close(what: &str, a: &Run, b: &Run) {
    let dl = a
        .losses
        .iter()
        .zip(&b.losses)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    let dp = max_abs_diff(&a.params, &b.params);
    eprintln!("{what}: max |loss diff| = {dl:.3e}, max |param diff| = {dp:.3e}");
    assert!(dl <= LOSS_TOL, "{what}: loss diff {dl} > {LOSS_TOL}");
    assert!(dp <= PARAM_TOL, "{what}: param diff {dp} > {PARAM_TOL}");
}

#[test]
fn ddp_matches_single_process_on_the_concatenated_batch() {
    let steps = 6;
    let reference = single(11, steps);
    let (one, _) = parallel(11, 1, DDP, steps);
    assert_eq!(one[0], reference, "world size 1 must be bitwise identical");
    for world in [2, 3, 6] {
        let (runs, _) = parallel(11, world, DDP, steps);
        assert_ranks_agree(&runs);
        assert_close(&format!("DDP x{world} vs single"), &runs[0], &reference);
    }
}

#[test]
fn zero1_is_bitwise_identical_to_ddp() {
    let steps = 6;
    for world in [2, 3] {
        let (ddp, ddp_state) = parallel(12, world, DDP, steps);
        let (zero, zero_state) = parallel(12, world, Strategy::Zero1, steps);
        assert_ranks_agree(&zero);
        assert_eq!(zero[0], ddp[0], "ZeRO-1 x{world} differs from DDP");
        assert_eq!(zero_state, ddp_state, "optimizer state differs");
    }
}

#[test]
fn zero1_shards_optimizer_state() {
    let out = run_ranks(3, |comm| {
        let mut d = DistTrainer::new(trainer(13), comm, Strategy::Zero1).unwrap();
        d.step().unwrap();
        let mine = d.partition().params_of(d.comm.rank());
        let names = d.trainer.model.param_names();
        let held: Vec<String> = d.trainer.opt.state.keys().cloned().collect();
        let mut expect: Vec<String> = names[mine].to_vec();
        expect.sort();
        assert_eq!(held, expect, "rank {} holds the wrong shard", d.comm.rank());
        held.len()
    });
    let total = trainer(13).model.param_names().len();
    assert_eq!(out.iter().sum::<usize>(), total);
    assert!(out.iter().all(|&n| n > 0), "unbalanced: {out:?}");
}

#[test]
fn gradient_accumulation_with_ddp() {
    // 2 ranks x 2 micro-batches of 3 == one process, 2 micro-batches of 6.
    let steps = 4;
    let mut t = trainer(14);
    t.cfg.grad_accum = 2;
    let mut losses = Vec::new();
    for _ in 0..steps {
        losses.push(t.step().loss);
    }
    let reference = Run {
        norms: vec![],
        losses,
        params: params(&t.model),
    };
    let out = run_ranks(2, |comm| {
        let mut t = trainer(14);
        t.cfg.grad_accum = 2;
        let mut d = DistTrainer::new(t, comm, Strategy::Zero1).unwrap();
        let losses: Vec<f32> = (0..steps).map(|_| d.step().unwrap().loss).collect();
        Run {
            norms: vec![],
            losses,
            params: params(&d.trainer.model),
        }
    });
    assert_close("ZeRO-1 x2, grad_accum 2 vs single", &out[0], &reference);
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("s2s-dist-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Resume from `path` at `world` ranks and train `steps` more.
fn resume(
    path: &Path,
    world: usize,
    strategy: Strategy,
    steps: usize,
    resave: Option<&Path>,
) -> Run {
    let runs = run_ranks(world, |comm| {
        // A different seed: everything must come from the checkpoint.
        let mut d = DistTrainer::new(trainer(999), comm, strategy).unwrap();
        let m = d.load(path).unwrap();
        assert_eq!(m.step, d.trainer.opt.step);
        if let Some(p) = resave {
            d.save(p).unwrap();
        }
        let mut losses = Vec::new();
        let mut norms = Vec::new();
        for _ in 0..steps {
            let r = d.step().unwrap();
            losses.push(r.loss);
            norms.push(r.stats.grad_norm);
        }
        Run {
            losses,
            norms,
            params: params(&d.trainer.model),
        }
    });
    assert_ranks_agree(&runs);
    runs.into_iter().next().unwrap()
}

#[test]
fn sharded_checkpoint_resumes_at_other_world_sizes() {
    let dir = temp_dir("resume");
    let ckpt = dir.join("w2.safetensors");
    let (k, rest) = (4, 4);

    // Uninterrupted ZeRO-1 run at world size 2 (the reference), saving a
    // checkpoint after k steps.
    let reference = run_ranks(2, |comm| {
        let mut d = DistTrainer::new(trainer(21), comm, Strategy::Zero1).unwrap();
        let mut losses = Vec::new();
        let mut norms = Vec::new();
        for i in 0..k + rest {
            if i == k {
                d.save(&ckpt).unwrap();
            }
            let r = d.step().unwrap();
            if i >= k {
                losses.push(r.loss);
                norms.push(r.stats.grad_norm);
            }
        }
        Run {
            losses,
            norms,
            params: params(&d.trainer.model),
        }
    });
    assert_ranks_agree(&reference);
    let reference = &reference[0];

    // The file is a regular single-process checkpoint with full moments.
    let mut single_t = trainer(5);
    let manifest = checkpoint::load(&ckpt, &mut single_t.model, &mut single_t.opt).unwrap();
    assert_eq!(manifest.step, k as u64);
    assert_eq!(single_t.opt.state.len(), single_t.model.param_names().len());

    // Same world size: bitwise.
    let same = resume(&ckpt, 2, Strategy::Zero1, rest, None);
    assert_eq!(&same, reference, "resume at world 2 is not bit-exact");

    // World size 3 (ZeRO-1): re-saving right after loading reproduces the
    // file byte for byte, so the loaded state is exactly the saved one.
    let resaved = dir.join("w3.safetensors");
    let three = resume(&ckpt, 3, Strategy::Zero1, rest, Some(&resaved));
    assert_eq!(
        std::fs::read(&resaved).unwrap(),
        std::fs::read(&ckpt).unwrap(),
        "world-3 re-save differs from the world-2 file"
    );
    assert_close("resume at ZeRO-1 x3 vs x2", &three, reference);

    // World size 1 (DDP): bitwise equal to a single-process trainer resumed
    // from the same file with checkpoint::load.
    let one = resume(&ckpt, 1, DDP, rest, None);
    single_t.data.restore(manifest.data);
    let mut losses = Vec::new();
    let mut norms = Vec::new();
    for _ in 0..rest {
        let r = single_t.step();
        losses.push(r.loss);
        norms.push(r.stats.grad_norm);
    }
    let single_resumed = Run {
        losses,
        norms,
        params: params(&single_t.model),
    };
    assert_eq!(
        one, single_resumed,
        "world-1 resume differs from single process"
    );
    assert_close("resume at DDP x1 vs ZeRO-1 x2", &one, reference);

    // World size 3 with DDP from the same ZeRO-written file: identical to the
    // ZeRO-1 x3 resume.
    let three_ddp = resume(&ckpt, 3, DDP, rest, None);
    assert_eq!(three_ddp, three);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn checkpoint_before_the_first_step_has_no_moments() {
    let dir = temp_dir("step0");
    let ckpt = dir.join("s0.safetensors");
    run_ranks(2, |comm| {
        let d = DistTrainer::new(trainer(31), comm, Strategy::Zero1).unwrap();
        d.save(&ckpt).unwrap();
    });
    let st = std::fs::read(&ckpt).unwrap();
    let mut t = trainer(1);
    let m = checkpoint::from_safetensors(&st, &mut t.model, &mut t.opt).unwrap();
    assert_eq!(m.step, 0);
    assert!(t.opt.state.is_empty());
    assert_eq!(params(&t.model), params(&trainer(31).model));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_errors_reach_every_rank() {
    let dir = temp_dir("missing");
    let missing = dir.join("nope.safetensors");
    let errs = run_ranks(3, |comm| {
        let mut d = DistTrainer::new(trainer(41), comm, DDP).unwrap();
        let e = d.load(&missing).unwrap_err().to_string();
        // The group is still usable afterwards.
        d.step().unwrap();
        e
    });
    assert!(errs[0].contains("I/O"), "{}", errs[0]);
    assert!(errs[1..].iter().all(|e| e.contains("rank 0 failed")));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn indivisible_batch_is_rejected() {
    let errs = run_ranks(4, |comm| DistTrainer::new(trainer(1), comm, DDP).err());
    assert!(errs
        .iter()
        .all(|e| e.as_ref().unwrap().0.contains("divisible")));
}
