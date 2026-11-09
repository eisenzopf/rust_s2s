//! Train the tiny transformer LM on the built-in corpus, on CPU.
//!
//! ```text
//! cargo run --release -p s2s-train --bin tiny-lm -- [steps] [checkpoint.safetensors]
//! ```

use s2s_core::SplitMix64;
use s2s_kernels::reference::optim::AdamWConfig;
use s2s_nn::block::Checkpoint;
use s2s_nn::lm::{LmConfig, TransformerLm};
use s2s_nn::Module;
use s2s_train::checkpoint;
use s2s_train::data::{CharCorpus, DEMO_TEXT};
use s2s_train::optim::{default_groups, AdamW};
use s2s_train::schedule::LrSchedule;
use s2s_train::trainer::{TrainConfig, Trainer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let steps: u64 = args.next().map(|s| s.parse()).transpose()?.unwrap_or(500);
    let ckpt_path = args.next().map(std::path::PathBuf::from);

    let data = CharCorpus::new(DEMO_TEXT, 1);
    let cfg = LmConfig {
        d_model: 64,
        n_layers: 3,
        q_heads: 4,
        kv_heads: 2,
        mlp_hidden: 128,
        ..LmConfig::tiny(data.vocab_size())
    };
    let model = TransformerLm::new(cfg, &mut SplitMix64::new(1));
    println!(
        "model: {} parameters, vocab {}",
        model.num_params(),
        cfg.vocab
    );
    let mut t = Trainer {
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
            batch: 16,
            seq: 64,
            grad_accum: 1,
            schedule: LrSchedule::WarmupCosine {
                warmup: 20,
                total: steps,
                peak: 5e-3,
                min: 5e-4,
            },
            checkpoint: Checkpoint::Off,
        },
    };
    if let Some(p) = ckpt_path.as_ref().filter(|p| p.exists()) {
        let m = checkpoint::load(p, &mut t.model, &mut t.opt)?;
        t.data.restore(m.data);
        println!("resumed from {} at step {}", p.display(), m.step);
    }
    while t.opt.step < steps {
        let r = t.step();
        if t.opt.step.is_multiple_of(50) || t.opt.step == 1 {
            println!(
                "step {:5}  loss {:.4}  lr {:.2e}  grad_norm {:.3}",
                t.opt.step, r.loss, r.stats.lr, r.stats.grad_norm
            );
        }
    }
    let prompt = t.data.encode("when the user ");
    println!(
        "sample: when the user {}",
        t.data.decode(&t.generate(&prompt, 80))
    );
    if let Some(p) = ckpt_path {
        checkpoint::save(&p, &t.model, &t.opt, &t.manifest())?;
        println!("saved {}", p.display());
    }
    Ok(())
}
