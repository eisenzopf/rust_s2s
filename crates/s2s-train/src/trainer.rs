//! A minimal training loop for the transformer LM on a character corpus.

use s2s_nn::block::Checkpoint;
use s2s_nn::lm::TransformerLm;
use s2s_nn::Module;

use crate::checkpoint::Manifest;
use crate::data::CharCorpus;
use crate::optim::{AdamW, StepStats};
use crate::schedule::LrSchedule;

/// Loop settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainConfig {
    /// Sequences per micro-batch.
    pub batch: usize,
    /// Tokens per sequence.
    pub seq: usize,
    /// Micro-batches accumulated per optimizer step.
    pub grad_accum: usize,
    /// Learning-rate schedule.
    pub schedule: LrSchedule,
    /// Activation checkpointing for every block.
    pub checkpoint: Checkpoint,
}

/// Result of one optimizer step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepReport {
    /// Mean loss over the accumulated micro-batches.
    pub loss: f32,
    /// Optimizer statistics.
    pub stats: StepStats,
}

/// Model, optimizer and data in one place.
#[derive(Debug)]
pub struct Trainer {
    /// The model.
    pub model: TransformerLm<f32>,
    /// The optimizer.
    pub opt: AdamW<f32>,
    /// The data.
    pub data: CharCorpus,
    /// Loop settings.
    pub cfg: TrainConfig,
}

impl Trainer {
    /// One optimizer step: `grad_accum` micro-batches, then AdamW.
    pub fn step(&mut self) -> StepReport {
        let c = self.cfg;
        self.model.zero_grad();
        let mut loss_sum = 0.0;
        for _ in 0..c.grad_accum {
            let (tokens, targets) = self.data.batch(c.batch, c.seq);
            let (loss, ctx) =
                self.model
                    .forward_loss(&tokens, &targets, c.batch, c.seq, c.checkpoint);
            self.model.backward(ctx, &targets);
            loss_sum += loss;
        }
        if c.grad_accum > 1 {
            // Each micro-batch's gradient is of its own mean loss; average them.
            let s = 1.0 / c.grad_accum as f32;
            self.model
                .visit_params_mut(&mut |p| p.grad.iter_mut().for_each(|g| *g *= s));
        }
        let lr = c.schedule.lr(self.opt.step);
        let stats = self.opt.step(&mut self.model, lr);
        StepReport {
            loss: loss_sum / c.grad_accum as f32,
            stats,
        }
    }

    /// The manifest describing the current state.
    pub fn manifest(&self) -> Manifest {
        let m = self.model.config();
        Manifest {
            version: 1,
            step: self.opt.step,
            data: self.data.cursor(),
            schedule: self.cfg.schedule,
            config: serde_json::json!({
                "model": "TransformerLm",
                "vocab": m.vocab,
                "d_model": m.d_model,
                "n_layers": m.n_layers,
                "q_heads": m.q_heads,
                "kv_heads": m.kv_heads,
                "mlp_hidden": m.mlp_hidden,
                "batch": self.cfg.batch,
                "seq": self.cfg.seq,
                "grad_accum": self.cfg.grad_accum,
            }),
        }
    }

    /// Greedy continuation of `prompt` for `n` tokens (full recompute per
    /// token; fine for tiny CPU demos).
    pub fn generate(&self, prompt: &[u32], n: usize) -> Vec<u32> {
        let mut seq = prompt.to_vec();
        let window = self.cfg.seq;
        let v = self.model.config().vocab;
        for _ in 0..n {
            let ctx: Vec<u32> = seq[seq.len().saturating_sub(window)..].to_vec();
            let logits = self.model.logits(&ctx, 1, ctx.len());
            let last = &logits[(ctx.len() - 1) * v..ctx.len() * v];
            let next = last
                .iter()
                .enumerate()
                .fold(
                    (0, f32::NEG_INFINITY),
                    |b, (i, &x)| if x > b.1 { (i, x) } else { b },
                )
                .0;
            seq.push(next as u32);
        }
        seq[prompt.len()..].to_vec()
    }
}
