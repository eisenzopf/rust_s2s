//! SEANet encoder and decoder (Kyutai's `seanet.rs`).

use s2s_kernels::reference::conv1d::{CausalConv1dState, PadMode};
use s2s_kernels::reference::conv_transpose1d::ConvTranspose1dState;
use s2s_kernels::reference::elu::elu_inplace;

use crate::conv::{Conv1d, ConvSpec, ConvTr1d};
use crate::weights::Loader;
use crate::{MimiConfig, Result, Signal};

fn elu(x: &Signal, alpha: f32) -> Signal {
    let mut y = x.clone();
    elu_inplace(&mut y.data, alpha);
    y
}

fn add(a: &Signal, b: &Signal) -> Signal {
    assert!(
        a.channels == b.channels && a.len == b.len,
        "residual add: shape mismatch"
    );
    let data = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
    Signal::new(a.channels, a.len, data)
}

fn conv_spec(c_in: usize, c_out: usize, kernel: usize) -> ConvSpec {
    ConvSpec {
        c_in,
        c_out,
        kernel,
        stride: 1,
        dilation: 1,
        bias: true,
        pad: PadMode::Constant,
    }
}

/// `y = x + block(x)` with `block = [ELU, conv(k, dilation), ELU, conv(1)]`.
#[derive(Clone, Debug)]
struct ResBlock {
    convs: Vec<Conv1d>,
    shortcut: Option<Conv1d>,
}

#[derive(Clone, Debug, Default)]
struct ResBlockState {
    convs: Vec<CausalConv1dState<f32>>,
    shortcut: CausalConv1dState<f32>,
}

impl ResBlock {
    fn load(
        l: &Loader,
        prefix: &str,
        dim: usize,
        dilation: usize,
        cfg: &MimiConfig,
    ) -> Result<Self> {
        let hidden = dim / cfg.compress;
        let taps = [(cfg.residual_kernel_size, dilation), (1, 1)];
        let mut convs = Vec::with_capacity(taps.len());
        for (i, &(k, d)) in taps.iter().enumerate() {
            let c_in = if i == 0 { dim } else { hidden };
            let c_out = if i == taps.len() - 1 { dim } else { hidden };
            let spec = ConvSpec {
                dilation: d,
                ..conv_spec(c_in, c_out, k)
            };
            convs.push(Conv1d::load(
                l,
                &format!("{prefix}.block.{}.conv.conv", 2 * i + 1),
                spec,
            )?);
        }
        let shortcut = if cfg.true_skip {
            None
        } else {
            Some(Conv1d::load(
                l,
                &format!("{prefix}.shortcut.conv.conv"),
                conv_spec(dim, dim, 1),
            )?)
        };
        Ok(Self { convs, shortcut })
    }

    fn state(&self) -> ResBlockState {
        ResBlockState {
            convs: vec![CausalConv1dState::new(); self.convs.len()],
            shortcut: CausalConv1dState::new(),
        }
    }

    fn forward(&self, x: &Signal, alpha: f32) -> Signal {
        let mut y = x.clone();
        for c in &self.convs {
            y = c.forward(&elu(&y, alpha));
        }
        match &self.shortcut {
            None => add(&y, x),
            Some(s) => add(&y, &s.forward(x)),
        }
    }

    fn step(&self, st: &mut ResBlockState, x: &Signal, alpha: f32) -> Signal {
        let mut y = x.clone();
        for (c, s) in self.convs.iter().zip(&mut st.convs) {
            y = c.step(s, &elu(&y, alpha));
        }
        // Every conv here is causal with stride 1, so once its left padding
        // is applied it returns exactly as many steps as it receives.
        match &self.shortcut {
            None => add(&y, x),
            Some(c) => add(&y, &c.step(&mut st.shortcut, x)),
        }
    }
}

#[derive(Clone, Debug)]
struct EncoderLayer {
    residuals: Vec<ResBlock>,
    downsample: Conv1d,
}

/// SEANet encoder: `[channels, n] -> [dimension, n / hop]`.
#[derive(Clone, Debug)]
pub(crate) struct SeaNetEncoder {
    init: Conv1d,
    layers: Vec<EncoderLayer>,
    last: Conv1d,
    alpha: f32,
}

/// Streaming state of [`SeaNetEncoder`].
#[derive(Clone, Debug, Default)]
pub(crate) struct SeaNetEncoderState {
    init: CausalConv1dState<f32>,
    layers: Vec<(Vec<ResBlockState>, CausalConv1dState<f32>)>,
    last: CausalConv1dState<f32>,
}

impl SeaNetEncoder {
    pub(crate) fn load(l: &Loader, prefix: &str, cfg: &MimiConfig) -> Result<Self> {
        let nf = cfg.n_filters;
        let p = |i: usize| format!("{prefix}.model.{i}.conv.conv");
        let init = Conv1d::load(l, &p(0), conv_spec(cfg.channels, nf, cfg.kernel_size))?;
        let mut idx = 1;
        let mut mult = 1;
        let mut layers = Vec::with_capacity(cfg.ratios.len());
        for &ratio in cfg.ratios.iter().rev() {
            let dim = mult * nf;
            let mut residuals = Vec::with_capacity(cfg.n_residual_layers);
            for j in 0..cfg.n_residual_layers {
                let dilation = cfg.dilation_base.pow(j as u32);
                residuals.push(ResBlock::load(
                    l,
                    &format!("{prefix}.model.{idx}"),
                    dim,
                    dilation,
                    cfg,
                )?);
                idx += 1;
            }
            // Index `idx` is the ELU.
            let downsample = Conv1d::load(
                l,
                &p(idx + 1),
                ConvSpec {
                    stride: ratio,
                    ..conv_spec(dim, 2 * dim, 2 * ratio)
                },
            )?;
            idx += 2;
            layers.push(EncoderLayer {
                residuals,
                downsample,
            });
            mult *= 2;
        }
        let last = Conv1d::load(
            l,
            &p(idx + 1),
            conv_spec(mult * nf, cfg.dimension, cfg.last_kernel_size),
        )?;
        Ok(Self {
            init,
            layers,
            last,
            alpha: cfg.elu_alpha as f32,
        })
    }

    pub(crate) fn state(&self) -> SeaNetEncoderState {
        SeaNetEncoderState {
            init: CausalConv1dState::new(),
            layers: self
                .layers
                .iter()
                .map(|l| {
                    (
                        l.residuals.iter().map(ResBlock::state).collect(),
                        CausalConv1dState::new(),
                    )
                })
                .collect(),
            last: CausalConv1dState::new(),
        }
    }

    pub(crate) fn forward(&self, x: &Signal) -> Signal {
        let mut x = self.init.forward(x);
        for layer in &self.layers {
            for r in &layer.residuals {
                x = r.forward(&x, self.alpha);
            }
            x = layer.downsample.forward(&elu(&x, self.alpha));
        }
        self.last.forward(&elu(&x, self.alpha))
    }

    pub(crate) fn step(&self, st: &mut SeaNetEncoderState, x: &Signal) -> Signal {
        let mut x = self.init.step(&mut st.init, x);
        for (layer, (rs, ds)) in self.layers.iter().zip(&mut st.layers) {
            for (r, s) in layer.residuals.iter().zip(rs.iter_mut()) {
                x = r.step(s, &x, self.alpha);
            }
            x = layer.downsample.step(ds, &elu(&x, self.alpha));
        }
        self.last.step(&mut st.last, &elu(&x, self.alpha))
    }
}

#[derive(Clone, Debug)]
struct DecoderLayer {
    upsample: ConvTr1d,
    residuals: Vec<ResBlock>,
}

/// SEANet decoder: `[dimension, t] -> [channels, t * hop]`.
#[derive(Clone, Debug)]
pub(crate) struct SeaNetDecoder {
    init: Conv1d,
    layers: Vec<DecoderLayer>,
    last: Conv1d,
    alpha: f32,
}

/// Streaming state of [`SeaNetDecoder`].
#[derive(Clone, Debug, Default)]
pub(crate) struct SeaNetDecoderState {
    init: CausalConv1dState<f32>,
    layers: Vec<(ConvTranspose1dState<f32>, Vec<ResBlockState>)>,
    last: CausalConv1dState<f32>,
}

impl SeaNetDecoder {
    pub(crate) fn load(l: &Loader, prefix: &str, cfg: &MimiConfig) -> Result<Self> {
        let nf = cfg.n_filters;
        let p = |i: usize| format!("{prefix}.model.{i}");
        let mut mult = 1usize << cfg.ratios.len();
        let init = Conv1d::load(
            l,
            &format!("{}.conv.conv", p(0)),
            conv_spec(cfg.dimension, mult * nf, cfg.kernel_size),
        )?;
        let mut idx = 1;
        let mut layers = Vec::with_capacity(cfg.ratios.len());
        for &ratio in &cfg.ratios {
            let (c_in, c_out) = (mult * nf, mult * nf / 2);
            // Index `idx` is the ELU.
            let upsample = ConvTr1d::load(
                l,
                &format!("{}.convtr.convtr", p(idx + 1)),
                c_in,
                c_out,
                2 * ratio,
                ratio,
                1,
                true,
            )?;
            idx += 2;
            let mut residuals = Vec::with_capacity(cfg.n_residual_layers);
            for j in 0..cfg.n_residual_layers {
                let dilation = cfg.dilation_base.pow(j as u32);
                residuals.push(ResBlock::load(l, &p(idx), c_out, dilation, cfg)?);
                idx += 1;
            }
            layers.push(DecoderLayer {
                upsample,
                residuals,
            });
            mult /= 2;
        }
        let last = Conv1d::load(
            l,
            &format!("{}.conv.conv", p(idx + 1)),
            conv_spec(nf, cfg.channels, cfg.last_kernel_size),
        )?;
        Ok(Self {
            init,
            layers,
            last,
            alpha: cfg.elu_alpha as f32,
        })
    }

    pub(crate) fn state(&self) -> SeaNetDecoderState {
        SeaNetDecoderState {
            init: CausalConv1dState::new(),
            layers: self
                .layers
                .iter()
                .map(|l| {
                    (
                        ConvTranspose1dState::new(),
                        l.residuals.iter().map(ResBlock::state).collect(),
                    )
                })
                .collect(),
            last: CausalConv1dState::new(),
        }
    }

    pub(crate) fn forward(&self, x: &Signal) -> Signal {
        let mut x = self.init.forward(x);
        for layer in &self.layers {
            x = layer.upsample.forward(&elu(&x, self.alpha));
            for r in &layer.residuals {
                x = r.forward(&x, self.alpha);
            }
        }
        self.last.forward(&elu(&x, self.alpha))
    }

    pub(crate) fn step(&self, st: &mut SeaNetDecoderState, x: &Signal) -> Signal {
        let mut x = self.init.step(&mut st.init, x);
        for (layer, (us, rs)) in self.layers.iter().zip(&mut st.layers) {
            x = layer.upsample.step(us, &elu(&x, self.alpha));
            for (r, s) in layer.residuals.iter().zip(rs.iter_mut()) {
                x = r.step(s, &x, self.alpha);
            }
        }
        self.last.step(&mut st.last, &elu(&x, self.alpha))
    }
}
