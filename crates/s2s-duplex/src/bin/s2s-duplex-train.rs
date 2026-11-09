//! Train the duplex model on `s2s-data` shards.
//!
//! ```text
//! s2s-duplex-train --shards data/*.s2s --steps 1000 --checkpoint run.safetensors
//! ```
//!
//! Re-running with the same `--checkpoint` resumes exactly where it stopped.

use std::path::PathBuf;

use clap::Parser;
use s2s_duplex::train::{run, Preset, TrainArgs};

#[derive(Parser, Debug)]
#[command(about = "Train the s2s duplex model on s2s-data shards")]
struct Cli {
    /// Shard files.
    #[arg(long, num_args = 1.., required = true)]
    shards: Vec<PathBuf>,
    /// Model size.
    #[arg(long, value_enum, default_value_t = Preset::Tiny)]
    preset: Preset,
    /// Total optimizer steps (sizes the LR schedule).
    #[arg(long, default_value_t = 1000)]
    steps: u64,
    /// Pause at this step (resume later with the same settings).
    #[arg(long)]
    stop_after: Option<u64>,
    /// Windows per batch.
    #[arg(long, default_value_t = 4)]
    batch: usize,
    /// Positions per window.
    #[arg(long, default_value_t = 256)]
    window: usize,
    /// Peak learning rate.
    #[arg(long, default_value_t = 3e-3)]
    lr: f64,
    /// Final learning rate.
    #[arg(long, default_value_t = 3e-4)]
    min_lr: f64,
    /// Warmup steps.
    #[arg(long, default_value_t = 50)]
    warmup: u64,
    /// Gradient-norm clip.
    #[arg(long, default_value_t = 1.0)]
    max_grad_norm: f64,
    /// Decoupled weight decay.
    #[arg(long, default_value_t = 0.1)]
    weight_decay: f64,
    /// Seed for init and data order.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Checkpoint file (resumed if it exists).
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    /// Steps between checkpoints (0 = only at the end).
    #[arg(long, default_value_t = 100)]
    save_every: u64,
    /// Recompute activations in backward.
    #[arg(long)]
    activation_checkpointing: bool,
    /// Print every N steps.
    #[arg(long, default_value_t = 10)]
    log_every: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let args = TrainArgs {
        shards: cli.shards,
        preset: cli.preset,
        steps: cli.steps,
        stop_after: cli.stop_after,
        batch: cli.batch,
        window: cli.window,
        peak_lr: cli.lr,
        min_lr: cli.min_lr,
        warmup: cli.warmup,
        max_grad_norm: cli.max_grad_norm,
        weight_decay: cli.weight_decay,
        seed: cli.seed,
        checkpoint: cli.checkpoint,
        save_every: cli.save_every,
        activation_checkpointing: cli.activation_checkpointing,
    };
    let every = cli.log_every.max(1);
    let summary = run(&args, &mut |s| {
        if s.step == 1 || s.step.is_multiple_of(every) {
            println!(
                "step {:6}  loss {:.4}  [Tu {:.3}  Tm {:.3}  sem {:.3}  ac {:.3}]  lr {:.2e}  |g| {:.3}",
                s.step, s.loss, s.per_kind[0], s.per_kind[1], s.per_kind[2], s.per_kind[3], s.lr, s.grad_norm
            );
        }
    })?;
    println!("done: steps {}..{}", summary.start_step, summary.end_step);
    Ok(())
}
