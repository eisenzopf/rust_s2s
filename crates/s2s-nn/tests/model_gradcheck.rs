//! Whole-model gradient checks for the transformer LM in `f64`.
//!
//! The kernels are checked op by op in `s2s-kernels`; this checks that the
//! layers compose correctly (residuals, norms, GQA head sharing, RoPE
//! placement, transposes, loss scaling) by comparing every parameter tensor's
//! backward-pass gradient against central differences of the actual loss.

use s2s_core::SplitMix64;
use s2s_kernels::reference::loss::IGNORE_INDEX;
use s2s_nn::block::Checkpoint;
use s2s_nn::lm::{LmConfig, TransformerLm};
use s2s_nn::{Module, Param};

fn tiny() -> LmConfig {
    LmConfig {
        vocab: 7,
        d_model: 8,
        n_layers: 2,
        q_heads: 4,
        kv_heads: 2,
        mlp_hidden: 12,
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
    }
}

fn data(seed: u64, cfg: &LmConfig, batch: usize, seq: usize) -> (Vec<u32>, Vec<i64>) {
    let mut r = SplitMix64::new(seed);
    let tokens: Vec<u32> = (0..batch * seq)
        .map(|_| r.below(cfg.vocab as u64) as u32)
        .collect();
    let mut targets: Vec<i64> = (0..batch * seq)
        .map(|_| r.below(cfg.vocab as u64) as i64)
        .collect();
    targets[3] = IGNORE_INDEX;
    (tokens, targets)
}

fn grads(model: &TransformerLm<f64>) -> Vec<(String, Vec<f64>)> {
    let mut out = Vec::new();
    model.visit_params(&mut |p: &Param<f64>| out.push((p.name.clone(), p.grad.clone())));
    out
}

#[test]
fn every_parameter_tensor_matches_finite_differences() {
    let cfg = tiny();
    let (batch, seq) = (2, 5);
    let mut model = TransformerLm::<f64>::new(cfg, &mut SplitMix64::new(11));
    let (tokens, targets) = data(12, &cfg, batch, seq);

    model.zero_grad();
    let (_, ctx) = model.forward_loss(&tokens, &targets, batch, seq, Checkpoint::Off);
    model.backward(ctx, &targets);
    let analytic = grads(&model);

    let loss_of = |m: &TransformerLm<f64>| {
        m.forward_loss(&tokens, &targets, batch, seq, Checkpoint::Off)
            .0
    };
    let mut pick = SplitMix64::new(13);
    let h = 1e-6;
    let mut checked = 0;
    for (pi, (name, g)) in analytic.iter().enumerate() {
        // Check up to 8 random entries of every parameter tensor.
        for _ in 0..8.min(g.len()) {
            let i = pick.below(g.len() as u64) as usize;
            let mut plus = model.clone();
            let mut minus = model.clone();
            let mut k = 0;
            plus.visit_params_mut(&mut |p| {
                if k == pi {
                    p.value[i] += h;
                }
                k += 1;
            });
            k = 0;
            minus.visit_params_mut(&mut |p| {
                if k == pi {
                    p.value[i] -= h;
                }
                k += 1;
            });
            let numeric = (loss_of(&plus) - loss_of(&minus)) / (2.0 * h);
            let tol = 1e-6 + 1e-4 * numeric.abs().max(g[i].abs());
            assert!(
                (numeric - g[i]).abs() <= tol,
                "{name}[{i}]: analytic {:.9e} vs numeric {numeric:.9e}",
                g[i]
            );
            checked += 1;
        }
    }
    assert!(checked > 100, "only {checked} entries checked");
}

#[test]
fn checkpointing_gives_identical_gradients() {
    let cfg = tiny();
    let (batch, seq) = (2, 6);
    let (tokens, targets) = data(21, &cfg, batch, seq);
    let base = TransformerLm::<f64>::new(cfg, &mut SplitMix64::new(22));

    let run = |ckpt| {
        let mut m = base.clone();
        m.zero_grad();
        let (loss, ctx) = m.forward_loss(&tokens, &targets, batch, seq, ckpt);
        m.backward(ctx, &targets);
        (loss, grads(&m))
    };
    let (l0, g0) = run(Checkpoint::Off);
    let (l1, g1) = run(Checkpoint::On);
    assert_eq!(l0, l1);
    assert_eq!(
        g0, g1,
        "recomputed activations must reproduce gradients exactly"
    );
}

#[test]
fn loss_starts_near_log_vocab() {
    let cfg = LmConfig::tiny(50);
    let model = TransformerLm::<f32>::new(cfg, &mut SplitMix64::new(1));
    let (tokens, targets) = {
        let mut r = SplitMix64::new(2);
        let t: Vec<u32> = (0..64).map(|_| r.below(50) as u32).collect();
        let y: Vec<i64> = (0..64).map(|_| r.below(50) as i64).collect();
        (t, y)
    };
    let (loss, _) = model.forward_loss(&tokens, &targets, 4, 16, Checkpoint::Off);
    let expected = (50f32).ln();
    assert!(
        (loss - expected).abs() < 0.6,
        "initial loss {loss} vs ln(V) {expected}"
    );
}

#[test]
fn streaming_step_matches_full_forward() {
    use s2s_nn::attention::KvCache;
    let mut cfg = tiny();
    cfg.n_layers = 2;
    let model = TransformerLm::<f64>::new(cfg, &mut SplitMix64::new(31));
    let seq = 9;
    let tokens: Vec<u32> = (0..seq as u32).map(|i| (i * 3) % 7).collect();
    let full = model.logits(&tokens, 1, seq);

    // Embed, then run blocks step by step with per-layer caches.
    let mut caches: Vec<KvCache<f64>> = (0..cfg.n_layers).map(|_| KvCache::new()).collect();
    for (t, &tok) in tokens.iter().enumerate() {
        let mut h = model.embed.forward(&[tok], 1).0;
        for (b, c) in model.blocks.iter().zip(caches.iter_mut()) {
            h = b.step(&h, t as i64, c);
        }
        let h = model.norm.forward(&h).0;
        let logits = model.head.forward(&h, 1).0;
        for (a, b) in logits.iter().zip(&full[t * cfg.vocab..(t + 1) * cfg.vocab]) {
            assert!((a - b).abs() < 1e-10, "position {t}: {a} vs {b}");
        }
    }
    assert_eq!(caches[0].len(), seq);
}
