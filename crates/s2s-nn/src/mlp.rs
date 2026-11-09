//! SwiGLU feed-forward block: `down(silu(gate(x)) * up(x))`.

use s2s_core::{Float, SplitMix64};
use s2s_kernels::reference::activation;

use crate::linear::{Linear, LinearCtx};
use crate::{Module, Param};

/// SwiGLU MLP.
#[derive(Clone, Debug)]
pub struct SwiGluMlp<T> {
    /// Gate projection `[hidden, d]`.
    pub gate: Linear<T>,
    /// Up projection `[hidden, d]`.
    pub up: Linear<T>,
    /// Down projection `[d, hidden]`.
    pub down: Linear<T>,
}

/// Saved values for the backward pass.
#[derive(Clone, Debug)]
pub struct MlpCtx<T> {
    gate: LinearCtx<T>,
    up: LinearCtx<T>,
    down: LinearCtx<T>,
    g: Vec<T>,
    u: Vec<T>,
}

impl<T: Float> SwiGluMlp<T> {
    /// New MLP. `out_std` sets the down-projection init (often scaled by depth).
    pub fn new(
        name: &str,
        d: usize,
        hidden: usize,
        out_std: Option<f64>,
        rng: &mut SplitMix64,
    ) -> Self {
        Self {
            gate: Linear::new(&format!("{name}.gate"), d, hidden, false, None, rng),
            up: Linear::new(&format!("{name}.up"), d, hidden, false, None, rng),
            down: Linear::new(&format!("{name}.down"), hidden, d, false, out_std, rng),
        }
    }

    /// Forward over `x: [rows, d]`.
    pub fn forward(&self, x: &[T], rows: usize) -> (Vec<T>, MlpCtx<T>) {
        let (g, gate) = self.gate.forward(x, rows);
        let (u, up) = self.up.forward(x, rows);
        let mut h = vec![T::zero(); g.len()];
        activation::swiglu_fwd(&g, &u, &mut h);
        let (y, down) = self.down.forward(&h, rows);
        (
            y,
            MlpCtx {
                gate,
                up,
                down,
                g,
                u,
            },
        )
    }

    /// Backward: accumulates parameter gradients, returns `dx`.
    pub fn backward(&mut self, ctx: MlpCtx<T>, dy: &[T]) -> Vec<T> {
        let dh = self.down.backward(ctx.down, dy);
        let mut dg = vec![T::zero(); ctx.g.len()];
        let mut du = vec![T::zero(); ctx.u.len()];
        activation::swiglu_bwd(&ctx.g, &ctx.u, &dh, &mut dg, &mut du);
        let mut dx = self.gate.backward(ctx.gate, &dg);
        crate::util::add_assign(&mut dx, &self.up.backward(ctx.up, &du));
        dx
    }
}

impl<T: Float> Module<T> for SwiGluMlp<T> {
    fn visit_params(&self, f: &mut dyn FnMut(&Param<T>)) {
        self.gate.visit_params(f);
        self.up.visit_params(f);
        self.down.visit_params(f);
    }
    fn visit_params_mut(&mut self, f: &mut dyn FnMut(&mut Param<T>)) {
        self.gate.visit_params_mut(f);
        self.up.visit_params_mut(f);
        self.down.visit_params_mut(f);
    }
}
