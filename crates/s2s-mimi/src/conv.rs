//! Causal conv layers (Kyutai's `StreamableConv1d` / `StreamableConvTranspose1d`).

use s2s_kernels::reference::conv1d::{causal_conv1d_fwd, CausalConv1dState, Conv1dParams, PadMode};
use s2s_kernels::reference::conv_transpose1d::{
    causal_conv_transpose1d_fwd, pack_weight, ConvTranspose1dParams, ConvTranspose1dState,
};

use crate::weights::Loader;
use crate::{Result, Signal};

/// Causal 1-D convolution.
#[derive(Clone, Debug)]
pub(crate) struct Conv1d {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
    c_in: usize,
    c_out: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    pad: PadMode,
}

/// Construction options for [`Conv1d::load`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct ConvSpec {
    pub c_in: usize,
    pub c_out: usize,
    pub kernel: usize,
    pub stride: usize,
    pub dilation: usize,
    pub bias: bool,
    pub pad: PadMode,
}

impl Conv1d {
    /// Load `<prefix>.weight` (or weight-norm pair) and `<prefix>.bias`.
    pub(crate) fn load(l: &Loader, prefix: &str, s: ConvSpec) -> Result<Self> {
        let w = l.conv_weight(prefix, [s.c_out, s.c_in, s.kernel])?;
        let b = if s.bias {
            Some(l.get(&format!("{prefix}.bias"), &[s.c_out])?)
        } else {
            None
        };
        Ok(Self {
            w,
            b,
            c_in: s.c_in,
            c_out: s.c_out,
            kernel: s.kernel,
            stride: s.stride,
            dilation: s.dilation,
            pad: s.pad,
        })
    }

    fn params(&self, t_in: usize) -> Conv1dParams {
        Conv1dParams {
            batch: 1,
            c_in: self.c_in,
            c_out: self.c_out,
            t_in,
            kernel: self.kernel,
            stride: self.stride,
            dilation: self.dilation,
            groups: 1,
        }
    }

    /// Offline causal forward.
    pub(crate) fn forward(&self, x: &Signal) -> Signal {
        assert_eq!(x.channels, self.c_in, "conv1d: input channels");
        let (y, t) = causal_conv1d_fwd(
            &self.params(x.len),
            &x.data,
            &self.w,
            self.b.as_deref(),
            self.pad,
        );
        Signal::new(self.c_out, t, y)
    }

    /// Streaming forward.
    pub(crate) fn step(&self, st: &mut CausalConv1dState<f32>, x: &Signal) -> Signal {
        assert_eq!(x.channels, self.c_in, "conv1d: input channels");
        let (y, t) = st.step(
            &self.params(x.len),
            &x.data,
            &self.w,
            self.b.as_deref(),
            self.pad,
        );
        Signal::new(self.c_out, t, y)
    }
}

/// Causal 1-D transposed convolution (output trimmed by `kernel - stride`).
#[derive(Clone, Debug)]
pub(crate) struct ConvTr1d {
    w_packed: Vec<f32>,
    b: Option<Vec<f32>>,
    c_in: usize,
    c_out: usize,
    kernel: usize,
    stride: usize,
    groups: usize,
}

impl ConvTr1d {
    /// Load a transposed conv whose weight is `[c_in, c_out / groups, kernel]`.
    /// Weight norm (`groups == 1` only, as in Kyutai's loader) is over dim 0.
    pub(crate) fn load(
        l: &Loader,
        prefix: &str,
        c_in: usize,
        c_out: usize,
        kernel: usize,
        stride: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let cog = c_out / groups;
        let w = if groups == 1 {
            l.conv_weight(prefix, [c_in, cog, kernel])?
        } else {
            l.get(&format!("{prefix}.weight"), &[c_in, cog, kernel])?
        };
        let b = if bias {
            Some(l.get(&format!("{prefix}.bias"), &[c_out])?)
        } else {
            None
        };
        Ok(Self {
            w_packed: pack_weight(&w, c_in, cog, kernel),
            b,
            c_in,
            c_out,
            kernel,
            stride,
            groups,
        })
    }

    fn params(&self, t_in: usize) -> ConvTranspose1dParams {
        ConvTranspose1dParams {
            batch: 1,
            c_in: self.c_in,
            c_out: self.c_out,
            t_in,
            kernel: self.kernel,
            stride: self.stride,
            groups: self.groups,
        }
    }

    /// Offline causal forward: `t_out = t_in * stride`.
    pub(crate) fn forward(&self, x: &Signal) -> Signal {
        assert_eq!(x.channels, self.c_in, "conv_tr1d: input channels");
        let y = causal_conv_transpose1d_fwd(
            &self.params(x.len),
            &x.data,
            &self.w_packed,
            self.b.as_deref(),
        );
        Signal::new(self.c_out, x.len * self.stride, y)
    }

    /// Streaming forward.
    pub(crate) fn step(&self, st: &mut ConvTranspose1dState<f32>, x: &Signal) -> Signal {
        assert_eq!(x.channels, self.c_in, "conv_tr1d: input channels");
        let y = st.step(
            &self.params(x.len),
            &x.data,
            &self.w_packed,
            self.b.as_deref(),
        );
        Signal::new(self.c_out, x.len * self.stride, y)
    }
}
