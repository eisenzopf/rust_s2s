//! `y = x W^T (+ b)` with `W: [out, in]`.

use s2s_core::{Float, SplitMix64};
use s2s_kernels::reference::gemm;

use crate::{Module, Param};

/// A linear (fully connected) layer.
#[derive(Clone, Debug)]
pub struct Linear<T> {
    /// Weight, `[out, in]`.
    pub w: Param<T>,
    /// Optional bias, `[out]`.
    pub b: Option<Param<T>>,
    d_in: usize,
    d_out: usize,
}

/// Saved input for the backward pass.
#[derive(Clone, Debug)]
pub struct LinearCtx<T> {
    x: Vec<T>,
    rows: usize,
}

impl<T: Float> Linear<T> {
    /// New layer with `N(0, std^2)` weights (default `std = 1/sqrt(in)`) and zero bias.
    pub fn new(
        name: &str,
        d_in: usize,
        d_out: usize,
        bias: bool,
        std: Option<f64>,
        rng: &mut SplitMix64,
    ) -> Self {
        let std = std.unwrap_or(1.0 / (d_in as f64).sqrt());
        Self {
            w: Param::normal(format!("{name}.w"), &[d_out, d_in], std, rng),
            b: bias.then(|| Param::constant(format!("{name}.b"), &[d_out], 0.0)),
            d_in,
            d_out,
        }
    }

    /// Input features.
    pub fn d_in(&self) -> usize {
        self.d_in
    }

    /// Output features.
    pub fn d_out(&self) -> usize {
        self.d_out
    }

    /// Forward over `rows` rows of `x: [rows, in]`.
    pub fn forward(&self, x: &[T], rows: usize) -> (Vec<T>, LinearCtx<T>) {
        assert_eq!(
            x.len(),
            rows * self.d_in,
            "Linear {}: input length",
            self.w.name
        );
        let mut y = vec![T::zero(); rows * self.d_out];
        gemm::linear_fwd(
            x,
            &self.w.value,
            self.b.as_ref().map(|b| b.value.as_slice()),
            &mut y,
            rows,
            self.d_in,
            self.d_out,
        );
        (
            y,
            LinearCtx {
                x: x.to_vec(),
                rows,
            },
        )
    }

    /// Backward: accumulates parameter gradients, returns `dx`.
    pub fn backward(&mut self, ctx: LinearCtx<T>, dy: &[T]) -> Vec<T> {
        let mut dx = vec![T::zero(); ctx.rows * self.d_in];
        gemm::linear_bwd(
            &ctx.x,
            &self.w.value,
            dy,
            Some(&mut dx),
            Some(&mut self.w.grad),
            self.b.as_mut().map(|b| b.grad.as_mut_slice()),
            ctx.rows,
            self.d_in,
            self.d_out,
        );
        dx
    }
}

impl<T: Float> Module<T> for Linear<T> {
    fn visit_params(&self, f: &mut dyn FnMut(&Param<T>)) {
        f(&self.w);
        if let Some(b) = &self.b {
            f(b);
        }
    }
    fn visit_params_mut(&mut self, f: &mut dyn FnMut(&mut Param<T>)) {
        f(&mut self.w);
        if let Some(b) = &mut self.b {
            f(b);
        }
    }
}
