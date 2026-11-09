use s2s_core::{cast, Float, SplitMix64};

/// A trainable tensor: value, gradient and a hierarchical name.
#[derive(Clone, Debug, PartialEq)]
pub struct Param<T> {
    /// Hierarchical name, e.g. `blocks.0.attn.wq`.
    pub name: String,
    /// Row-major shape.
    pub shape: Vec<usize>,
    /// Values.
    pub value: Vec<T>,
    /// Accumulated gradient (same length as `value`).
    pub grad: Vec<T>,
}

impl<T: Float> Param<T> {
    /// A parameter filled with `fill`.
    pub fn constant(name: impl Into<String>, shape: &[usize], fill: f64) -> Self {
        let n = shape.iter().product();
        Self {
            name: name.into(),
            shape: shape.to_vec(),
            value: vec![cast(fill); n],
            grad: vec![T::zero(); n],
        }
    }

    /// A parameter drawn from `N(0, std^2)`.
    pub fn normal(
        name: impl Into<String>,
        shape: &[usize],
        std: f64,
        rng: &mut SplitMix64,
    ) -> Self {
        let n: usize = shape.iter().product();
        Self {
            name: name.into(),
            shape: shape.to_vec(),
            value: rng.normal_vec(n, std),
            grad: vec![T::zero(); n],
        }
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.value.len()
    }

    /// Reset the gradient to zero.
    pub fn zero_grad(&mut self) {
        self.grad.iter_mut().for_each(|g| *g = T::zero());
    }
}

/// Anything that owns parameters.
pub trait Module<T: Float> {
    /// Visit every parameter, in a fixed order.
    fn visit_params(&self, f: &mut dyn FnMut(&Param<T>));
    /// Visit every parameter mutably, in the same order.
    fn visit_params_mut(&mut self, f: &mut dyn FnMut(&mut Param<T>));

    /// Zero all gradients.
    fn zero_grad(&mut self) {
        self.visit_params_mut(&mut |p| p.zero_grad());
    }

    /// Total number of scalar parameters.
    fn num_params(&self) -> usize {
        let mut n = 0;
        self.visit_params(&mut |p| n += p.numel());
        n
    }

    /// Parameter names, in visiting order.
    fn param_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        self.visit_params(&mut |p| names.push(p.name.clone()));
        names
    }
}
