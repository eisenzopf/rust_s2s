//! RMSNorm layer.

use s2s_core::Float;
use s2s_kernels::reference::norm;

use crate::{Module, Param};

/// RMSNorm over the last dimension with a learned scale.
#[derive(Clone, Debug)]
pub struct RmsNorm<T> {
    /// Scale, `[d]`, initialized to 1.
    pub w: Param<T>,
    eps: f64,
}

/// Saved values for the backward pass.
#[derive(Clone, Debug)]
pub struct RmsNormCtx<T> {
    x: Vec<T>,
    rstd: Vec<T>,
}

impl<T: Float> RmsNorm<T> {
    /// New layer over `d` features.
    pub fn new(name: &str, d: usize, eps: f64) -> Self {
        Self {
            w: Param::constant(format!("{name}.w"), &[d], 1.0),
            eps,
        }
    }

    fn d(&self) -> usize {
        self.w.numel()
    }

    /// Forward over `x: [rows, d]`.
    pub fn forward(&self, x: &[T]) -> (Vec<T>, RmsNormCtx<T>) {
        let d = self.d();
        let rows = x.len() / d;
        assert_eq!(rows * d, x.len());
        let mut y = vec![T::zero(); x.len()];
        let mut rstd = vec![T::zero(); rows];
        norm::rmsnorm_fwd(x, &self.w.value, self.eps, &mut y, &mut rstd, rows, d);
        (
            y,
            RmsNormCtx {
                x: x.to_vec(),
                rstd,
            },
        )
    }

    /// Backward: accumulates `dw`, returns `dx`.
    pub fn backward(&mut self, ctx: RmsNormCtx<T>, dy: &[T]) -> Vec<T> {
        let d = self.d();
        let rows = ctx.rstd.len();
        let mut dx = vec![T::zero(); ctx.x.len()];
        norm::rmsnorm_bwd(
            &ctx.x,
            &self.w.value,
            &ctx.rstd,
            dy,
            &mut dx,
            &mut self.w.grad,
            rows,
            d,
        );
        dx
    }
}

impl<T: Float> Module<T> for RmsNorm<T> {
    fn visit_params(&self, f: &mut dyn FnMut(&Param<T>)) {
        f(&self.w);
    }
    fn visit_params_mut(&mut self, f: &mut dyn FnMut(&mut Param<T>)) {
        f(&mut self.w);
    }
}
