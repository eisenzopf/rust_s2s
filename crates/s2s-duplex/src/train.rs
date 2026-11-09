//! Training driver: shards in, AdamW with a schedule, checkpoints with exact
//! resume (model, optimizer and loader position).
//!
//! The model is configured from the data: stream layout, text vocabulary and
//! pad id come from the shards' samples, so the same tokenizer that built the
//! data defines the model's text heads.

use std::path::PathBuf;
use std::sync::Arc;

use s2s_core::streams::{StreamKind, StreamLayout};
use s2s_core::SplitMix64;
use s2s_data::{AssemblyConfig, Cursor, Dataset, Loader, LoaderConfig, ShardReader, WindowMode};
use s2s_kernels::reference::optim::AdamWConfig;
use s2s_nn::block::Checkpoint;
use s2s_nn::Module;
use s2s_train::checkpoint::{self, Manifest};
use s2s_train::data::DataCursor;
use s2s_train::optim::{default_groups, AdamW};
use s2s_train::schedule::LrSchedule;

use crate::{DuplexBatch, DuplexConfig, DuplexModel};

/// Model size presets (plan §3.6). All run on CPU; anything beyond `Xs` is
/// only practical on GPU kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Preset {
    /// For tests and smoke runs (~0.3 M temporal parameters).
    Tiny,
    /// Plan's XS: 12 layers, width 768, depth transformer 4 × 512.
    Xs,
}

/// Model config for a preset over the data's layout and vocabulary.
pub fn config_for(
    preset: Preset,
    layout: StreamLayout,
    text_vocab: usize,
    text_pad: u32,
) -> DuplexConfig {
    let tiny = DuplexConfig::tiny(layout, text_vocab, text_pad);
    match preset {
        Preset::Tiny => tiny,
        Preset::Xs => DuplexConfig {
            d_model: 768,
            n_layers: 12,
            q_heads: 12,
            kv_heads: 4,
            mlp_hidden: 2048,
            depth_d: 512,
            depth_layers: 4,
            depth_heads: 8,
            depth_mlp_hidden: 1536,
            window: Some(3000),
            ..tiny
        },
    }
}

/// Training settings.
#[derive(Clone, Debug)]
pub struct TrainArgs {
    /// Shard files (`s2s-data` format).
    pub shards: Vec<PathBuf>,
    /// Model size.
    pub preset: Preset,
    /// Total optimizer steps of the run (sizes the LR schedule).
    pub steps: u64,
    /// Pause at this step (checkpoint and return) without changing the
    /// schedule; resume later with the same settings. `None` = run to `steps`.
    pub stop_after: Option<u64>,
    /// Windows per batch.
    pub batch: usize,
    /// Positions per window.
    pub window: usize,
    /// Peak learning rate.
    pub peak_lr: f64,
    /// Final learning rate.
    pub min_lr: f64,
    /// Warmup steps.
    pub warmup: u64,
    /// Global gradient-norm clip.
    pub max_grad_norm: f64,
    /// Decoupled weight decay.
    pub weight_decay: f64,
    /// Seed for init and data order.
    pub seed: u64,
    /// Checkpoint file: resumed from if it exists, written every
    /// `save_every` steps and at the end.
    pub checkpoint: Option<PathBuf>,
    /// Steps between checkpoints (0 = only at the end).
    pub save_every: u64,
    /// Recompute activations in backward (less memory).
    pub activation_checkpointing: bool,
}

/// One step's statistics.
#[derive(Clone, Debug, PartialEq)]
pub struct StepLog {
    /// Optimizer step just taken (1-based).
    pub step: u64,
    /// Weighted mean loss.
    pub loss: f64,
    /// Mean loss per stream kind: user text, assistant text, semantic audio,
    /// acoustic audio.
    pub per_kind: [f64; 4],
    /// Learning rate used.
    pub lr: f64,
    /// Gradient norm before clipping.
    pub grad_norm: f64,
}

/// Result of a run.
#[derive(Debug)]
pub struct TrainSummary {
    /// Step the run started from (0, or the resumed checkpoint's step).
    pub start_step: u64,
    /// Final step.
    pub end_step: u64,
    /// The trained model.
    pub model: DuplexModel<f32>,
}

type Error = Box<dyn std::error::Error>;

/// Run training. `log` is called after every step.
pub fn run(args: &TrainArgs, log: &mut dyn FnMut(&StepLog)) -> Result<TrainSummary, Error> {
    if args.shards.is_empty() {
        return Err("no shards given".into());
    }
    let readers = args
        .shards
        .iter()
        .map(ShardReader::open)
        .collect::<Result<Vec<_>, _>>()?;
    let first = readers[0].get(0)?;
    for r in &readers {
        let s = r.get(0)?;
        if (s.layout, s.text_vocab, s.text_pad) != (first.layout, first.text_vocab, first.text_pad)
        {
            return Err("shards disagree on layout or text vocabulary".into());
        }
    }
    let layout = first.layout;
    let dataset = Arc::new(Dataset::new(readers));

    let cfg = config_for(
        args.preset,
        layout,
        first.text_vocab as usize,
        first.text_pad,
    );
    let mut model = DuplexModel::<f32>::new(cfg, &mut SplitMix64::new(args.seed));
    let mut opt = AdamW::new(
        AdamWConfig {
            weight_decay: args.weight_decay,
            ..Default::default()
        },
        Some(args.max_grad_norm),
        default_groups,
    );
    let schedule = LrSchedule::WarmupCosine {
        warmup: args.warmup,
        total: args.steps,
        peak: args.peak_lr,
        min: args.min_lr,
    };
    let mut lcfg = LoaderConfig::new(
        AssemblyConfig::new(layout, first.text_pad),
        args.batch,
        args.window,
    );
    lcfg.mode = WindowMode::Pack;
    lcfg.seed = args.seed;

    let mut loader = match args.checkpoint.as_ref().filter(|p| p.exists()) {
        Some(path) => {
            let manifest = checkpoint::load(path, &mut model, &mut opt)?;
            let cursor: Cursor = serde_json::from_value(
                manifest
                    .config
                    .get("loader_cursor")
                    .cloned()
                    .ok_or("checkpoint has no loader cursor")?,
            )?;
            Loader::with_cursor(dataset, lcfg, cursor)?
        }
        None => Loader::new(dataset, lcfg)?,
    };
    let start_step = opt.step;
    let ckpt_mode = if args.activation_checkpointing {
        Checkpoint::On
    } else {
        Checkpoint::Off
    };
    let kinds: Vec<usize> = (0..layout.num_generated())
        .map(|k| match layout.kind(k) {
            StreamKind::UserText => 0,
            StreamKind::AssistantText(_) => 1,
            StreamKind::AssistantAudio(0) => 2,
            _ => 3,
        })
        .collect();

    let save = |model: &DuplexModel<f32>, opt: &AdamW<f32>, cursor: Cursor| -> Result<(), Error> {
        if let Some(path) = &args.checkpoint {
            let manifest = Manifest {
                version: 1,
                step: opt.step,
                data: DataCursor {
                    rng_state: 0,
                    batches: opt.step,
                },
                schedule,
                config: serde_json::json!({
                    "model": "DuplexModel",
                    "preset": format!("{:?}", args.preset),
                    "text_vocab": model.config().text_vocab,
                    "text_pad": model.config().text_pad,
                    "layout": {
                        "text_slots": layout.text_slots,
                        "codebooks": layout.codebooks,
                        "user_text_delay": layout.user_text_delay,
                        "acoustic_delay": layout.acoustic_delay,
                    },
                    "loader_cursor": serde_json::to_value(cursor)?,
                }),
            };
            checkpoint::save(path, model, opt, &manifest)?;
        }
        Ok(())
    };

    let stop = args.stop_after.unwrap_or(args.steps).min(args.steps);
    // Windows with no loss weight at all (e.g. entirely inside a long system
    // prompt injection) carry no signal: skip them rather than spend an
    // optimizer step (and its momentum / weight decay) on nothing.
    let mut empty_in_a_row = 0u32;
    while opt.step < stop {
        let batch = DuplexBatch::<f32>::from_data(&loader.next_batch()?);
        if batch.weights.iter().all(|&w| w == 0.0) {
            empty_in_a_row += 1;
            if empty_in_a_row > 10_000 {
                return Err(
                    "10000 batches in a row without any loss weight: no training targets?".into(),
                );
            }
            continue;
        }
        empty_in_a_row = 0;
        model.zero_grad();
        let (_, report, ctx) = model.forward_loss(&batch, ckpt_mode);
        model.backward(ctx);
        let lr = schedule.lr(opt.step);
        let stats = opt.step(&mut model, lr);
        let mut sums = [(0.0f64, 0.0f64); 4];
        for (k, &(s, w)) in report.per_stream.iter().enumerate() {
            sums[kinds[k]].0 += s;
            sums[kinds[k]].1 += w;
        }
        let per_kind = sums.map(|(s, w)| if w > 0.0 { s / w } else { 0.0 });
        log(&StepLog {
            step: opt.step,
            loss: report.loss,
            per_kind,
            lr,
            grad_norm: stats.grad_norm,
        });
        if args.save_every > 0 && opt.step.is_multiple_of(args.save_every) {
            save(&model, &opt, loader.cursor())?;
        }
    }
    save(&model, &opt, loader.cursor())?;
    Ok(TrainSummary {
        start_step,
        end_step: opt.step,
        model,
    })
}
