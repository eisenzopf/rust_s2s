//! AdamW over a [`Module`], with parameter groups chosen by name.

use std::collections::BTreeMap;

use s2s_core::Float;
use s2s_kernels::reference::optim::{self, AdamWConfig};
use s2s_nn::{Module, Param};

/// Per-parameter settings decided by name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupSettings {
    /// Multiplier on the scheduled learning rate (e.g. lower for a pretrained
    /// backbone, higher for new embeddings; plan §8.1).
    pub lr_scale: f64,
    /// Whether decoupled weight decay applies.
    pub weight_decay: bool,
}

/// Default grouping: no weight decay on norms, biases and embeddings.
pub fn default_groups(name: &str) -> GroupSettings {
    let no_decay = name.contains("norm") || name.ends_with(".b") || name.starts_with("embed");
    GroupSettings {
        lr_scale: 1.0,
        weight_decay: !no_decay,
    }
}

/// First and second moments for one parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct Moments<T> {
    /// First moment.
    pub m: Vec<T>,
    /// Second moment.
    pub v: Vec<T>,
}

/// Statistics from one optimizer step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepStats {
    /// Global gradient L2 norm before clipping.
    pub grad_norm: f64,
    /// Scale applied to gradients by clipping (1.0 = not clipped).
    pub clip_scale: f64,
    /// Scheduled learning rate used.
    pub lr: f64,
}

/// AdamW optimizer state.
#[derive(Clone, Debug)]
pub struct AdamW<T> {
    /// Base hyper-parameters (`lr` is overridden per step by the schedule).
    pub cfg: AdamWConfig,
    /// Clip the global gradient norm to this value (`None` = no clipping).
    pub max_grad_norm: Option<f64>,
    /// Updates taken so far.
    pub step: u64,
    /// Moments by parameter name.
    pub state: BTreeMap<String, Moments<T>>,
    groups: fn(&str) -> GroupSettings,
}

impl<T: Float> AdamW<T> {
    /// New optimizer with the given parameter-group rule.
    pub fn new(
        cfg: AdamWConfig,
        max_grad_norm: Option<f64>,
        groups: fn(&str) -> GroupSettings,
    ) -> Self {
        Self {
            cfg,
            max_grad_norm,
            step: 0,
            state: BTreeMap::new(),
            groups,
        }
    }

    /// One update with learning rate `lr`, using the gradients accumulated in
    /// `model`. Does not zero them.
    pub fn step(&mut self, model: &mut dyn Module<T>, lr: f64) -> StepStats {
        let mut grads: Vec<Vec<T>> = Vec::new();
        model.visit_params(&mut |p: &Param<T>| grads.push(p.grad.clone()));
        let refs: Vec<&[T]> = grads.iter().map(|g| g.as_slice()).collect();
        let grad_norm = optim::global_grad_norm(&refs);
        let clip = self
            .max_grad_norm
            .map_or(1.0, |m| optim::clip_scale(grad_norm, m));
        self.step += 1;
        let step = self.step;
        let (base, groups) = (self.cfg, self.groups);
        let state = &mut self.state;
        model.visit_params_mut(&mut |p: &mut Param<T>| {
            let g = groups(&p.name);
            let cfg = AdamWConfig {
                lr: lr * g.lr_scale,
                weight_decay: if g.weight_decay {
                    base.weight_decay
                } else {
                    0.0
                },
                ..base
            };
            let mom = state.entry(p.name.clone()).or_insert_with(|| Moments {
                m: vec![T::zero(); p.numel()],
                v: vec![T::zero(); p.numel()],
            });
            optim::adamw_step(
                &mut p.value,
                &p.grad,
                &mut mom.m,
                &mut mom.v,
                step,
                &cfg,
                clip,
            );
        });
        StepStats {
            grad_norm,
            clip_scale: clip,
            lr,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use s2s_core::SplitMix64;
    use s2s_nn::linear::Linear;

    #[test]
    fn groups_and_state() {
        assert!(!default_groups("blocks.0.norm1.w").weight_decay);
        assert!(!default_groups("embed.table").weight_decay);
        assert!(!default_groups("head.b").weight_decay);
        assert!(default_groups("blocks.0.attn.wq.w").weight_decay);

        let mut lin = Linear::<f64>::new("lin", 3, 2, true, None, &mut SplitMix64::new(1));
        lin.w.grad.iter_mut().for_each(|g| *g = 1.0);
        lin.b
            .as_mut()
            .unwrap()
            .grad
            .iter_mut()
            .for_each(|g| *g = -1.0);
        let w0 = lin.w.value.clone();
        let mut opt = AdamW::new(
            AdamWConfig {
                weight_decay: 0.0,
                eps: 0.0,
                ..Default::default()
            },
            None,
            default_groups,
        );
        let stats = opt.step(&mut lin, 0.1);
        assert_eq!(opt.step, 1);
        assert_eq!(opt.state.len(), 2);
        // First AdamW step moves each weight by lr * sign(grad).
        for (a, b) in lin.w.value.iter().zip(&w0) {
            assert!((b - a - 0.1).abs() < 1e-12);
        }
        assert!(lin
            .b
            .as_ref()
            .unwrap()
            .value
            .iter()
            .all(|&x| (x - 0.1).abs() < 1e-12));
        assert!((stats.grad_norm - 8f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn clipping_scales_gradients() {
        let mut lin = Linear::<f64>::new("lin", 2, 1, false, None, &mut SplitMix64::new(2));
        lin.w.grad = vec![30.0, 40.0];
        let mut opt = AdamW::new(AdamWConfig::default(), Some(1.0), default_groups);
        let s = opt.step(&mut lin, 1e-3);
        assert!((s.grad_norm - 50.0).abs() < 1e-12);
        assert!((s.clip_scale - 1.0 / (50.0 + 1e-6)).abs() < 1e-9);
    }
}
