//! 1-D transposed convolution with stride and groups, causal trimming and
//! streaming overlap-add state (plan §7.2 kernel #16, used by Mimi).
//!
//! Layouts (row-major):
//! - `x`: `[batch, c_in, t_in]`
//! - `w`: `[c_in, c_out / groups, kernel]` (PyTorch / safetensors convention)
//! - `y`: `[batch, c_out, (t_in - 1) * stride + kernel]` (untrimmed)
//!
//! Output position `u` receives `x[ci, t] * w[ci, o, u - t * stride]` from
//! every input step `t` whose window covers it. The contributions are summed
//! **in increasing `t`** (then increasing `ci`) starting from zero, and the
//! bias is added last. Streaming carries the partial sums of the last
//! `kernel - stride` positions to the next chunk and keeps accumulating into
//! them in the same order, so [`ConvTranspose1dState::step`] reproduces
//! [`causal_conv_transpose1d_fwd`] **bit for bit**.
//!
//! The inner loop runs over output channels, so the kernel works on a
//! *packed* weight `[c_in, kernel, c_out / groups]` ([`pack_weight`]). Callers
//! that run many steps pack once at load time.

use s2s_core::Float;

/// Transposed-convolution dimensions and options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvTranspose1dParams {
    /// Batch size.
    pub batch: usize,
    /// Input channels.
    pub c_in: usize,
    /// Output channels.
    pub c_out: usize,
    /// Input length.
    pub t_in: usize,
    /// Kernel size (taps).
    pub kernel: usize,
    /// Stride (upsampling factor).
    pub stride: usize,
    /// Groups. Must divide `c_in` and `c_out`.
    pub groups: usize,
}

impl ConvTranspose1dParams {
    /// Untrimmed output length `(t_in - 1) * stride + kernel` (0 for empty input).
    pub fn t_out(&self) -> usize {
        if self.t_in == 0 {
            0
        } else {
            (self.t_in - 1) * self.stride + self.kernel
        }
    }

    fn check(&self) {
        assert!(self.kernel > 0 && self.stride > 0);
        assert!(
            self.groups > 0
                && self.c_in.is_multiple_of(self.groups)
                && self.c_out.is_multiple_of(self.groups),
            "conv_transpose1d: groups must divide c_in and c_out"
        );
    }
}

/// Repack `w: [c_in, c_out / groups, kernel]` into `[c_in, kernel, c_out / groups]`.
pub fn pack_weight<T: Float>(w: &[T], c_in: usize, cog: usize, kernel: usize) -> Vec<T> {
    assert_eq!(
        w.len(),
        c_in * cog * kernel,
        "pack_weight: w has wrong length"
    );
    let mut out = vec![T::zero(); w.len()];
    for ci in 0..c_in {
        for o in 0..cog {
            for kk in 0..kernel {
                out[(ci * kernel + kk) * cog + o] = w[(ci * cog + o) * kernel + kk];
            }
        }
    }
    out
}

/// Add the contributions of `x` into a time-major accumulator
/// `acc: [batch, t_acc, c_out]` (`t_acc >= p.t_out()`), in the order
/// described in the module docs. `w_packed` is [`pack_weight`]'s layout.
pub fn conv_transpose1d_accumulate<T: Float>(
    p: &ConvTranspose1dParams,
    x: &[T],
    w_packed: &[T],
    acc: &mut [T],
    t_acc: usize,
) {
    p.check();
    let cig = p.c_in / p.groups;
    let cog = p.c_out / p.groups;
    assert_eq!(
        x.len(),
        p.batch * p.c_in * p.t_in,
        "conv_transpose1d: x has wrong length"
    );
    assert_eq!(
        w_packed.len(),
        p.c_in * p.kernel * cog,
        "conv_transpose1d: w has wrong length"
    );
    assert!(
        t_acc >= p.t_out(),
        "conv_transpose1d: accumulator too short"
    );
    assert_eq!(
        acc.len(),
        p.batch * t_acc * p.c_out,
        "conv_transpose1d: acc has wrong length"
    );
    for b in 0..p.batch {
        let acc_b = &mut acc[b * t_acc * p.c_out..(b + 1) * t_acc * p.c_out];
        for t in 0..p.t_in {
            for g in 0..p.groups {
                for cl in 0..cig {
                    let ci = g * cig + cl;
                    let xv = x[(b * p.c_in + ci) * p.t_in + t];
                    for kk in 0..p.kernel {
                        let u = t * p.stride + kk;
                        let wr = &w_packed[(ci * p.kernel + kk) * cog..][..cog];
                        let row = &mut acc_b[u * p.c_out + g * cog..][..cog];
                        for (r, &wv) in row.iter_mut().zip(wr) {
                            *r = *r + xv * wv;
                        }
                    }
                }
            }
        }
    }
}

/// Untrimmed transposed convolution. `w` is in PyTorch layout; `y` must have
/// length `batch * c_out * p.t_out()`.
pub fn conv_transpose1d_fwd<T: Float>(
    p: &ConvTranspose1dParams,
    x: &[T],
    w: &[T],
    bias: Option<&[T]>,
    y: &mut [T],
) {
    let wp = pack_weight(w, p.c_in, p.c_out / p.groups.max(1), p.kernel);
    conv_transpose1d_packed_fwd(p, x, &wp, bias, y);
}

/// [`conv_transpose1d_fwd`] with a pre-packed weight.
pub fn conv_transpose1d_packed_fwd<T: Float>(
    p: &ConvTranspose1dParams,
    x: &[T],
    w_packed: &[T],
    bias: Option<&[T]>,
    y: &mut [T],
) {
    let t_out = p.t_out();
    assert_eq!(
        y.len(),
        p.batch * p.c_out * t_out,
        "conv_transpose1d: y has wrong length"
    );
    let mut acc = vec![T::zero(); p.batch * t_out * p.c_out];
    conv_transpose1d_accumulate(p, x, w_packed, &mut acc, t_out);
    emit(&acc, t_out, 0, t_out, p.batch, p.c_out, bias, y);
}

/// Write `acc[b, from..to, o] + bias[o]` into channel-major `y: [batch, c_out, to - from]`.
fn emit<T: Float>(
    acc: &[T],
    t_acc: usize,
    from: usize,
    to: usize,
    batch: usize,
    c_out: usize,
    bias: Option<&[T]>,
    y: &mut [T],
) {
    let n = to - from;
    if let Some(bb) = bias {
        assert_eq!(bb.len(), c_out, "conv_transpose1d: bias has wrong length");
    }
    for b in 0..batch {
        for u in 0..n {
            let row = &acc[(b * t_acc + from + u) * c_out..][..c_out];
            for o in 0..c_out {
                let v = row[o];
                y[(b * c_out + o) * n + u] = match bias {
                    Some(bb) => v + bb[o],
                    None => v,
                };
            }
        }
    }
}

/// Causal transposed convolution over a whole sequence: the untrimmed output
/// with its last `kernel - stride` positions removed (Kyutai's
/// `trim_right_ratio = 1`). Returns `y: [batch, c_out, t_in * stride]`.
pub fn causal_conv_transpose1d_fwd<T: Float>(
    p: &ConvTranspose1dParams,
    x: &[T],
    w_packed: &[T],
    bias: Option<&[T]>,
) -> Vec<T> {
    assert!(
        p.kernel >= p.stride,
        "causal conv-transpose: kernel smaller than stride"
    );
    let t_full = p.t_out();
    let mut acc = vec![T::zero(); p.batch * t_full * p.c_out];
    conv_transpose1d_accumulate(p, x, w_packed, &mut acc, t_full);
    let keep = p.t_in * p.stride;
    let mut y = vec![T::zero(); p.batch * p.c_out * keep];
    emit(&acc, t_full, 0, keep, p.batch, p.c_out, bias, &mut y);
    y
}

/// Streaming state of a causal transposed convolution (Kyutai's
/// `StreamableConvTranspose1d::step`): the bias-free partial sums of the
/// `kernel - stride` positions that later inputs still contribute to.
#[derive(Clone, Debug, Default)]
pub struct ConvTranspose1dState<T> {
    /// `[batch, kernel - stride, c_out]`, time-major.
    partial: Vec<T>,
}

impl<T: Float> ConvTranspose1dState<T> {
    /// Fresh state (start of a stream).
    pub fn new() -> Self {
        Self {
            partial: Vec::new(),
        }
    }

    /// Forget everything (start a new stream).
    pub fn reset(&mut self) {
        self.partial.clear();
    }

    /// Process one chunk `x: [batch, c_in, p.t_in]`. Returns
    /// `y: [batch, c_out, p.t_in * stride]`; concatenating the outputs of
    /// consecutive steps equals [`causal_conv_transpose1d_fwd`] bit for bit.
    pub fn step(
        &mut self,
        p: &ConvTranspose1dParams,
        x: &[T],
        w_packed: &[T],
        bias: Option<&[T]>,
    ) -> Vec<T> {
        assert!(
            p.kernel >= p.stride,
            "causal conv-transpose: kernel smaller than stride"
        );
        if p.t_in == 0 {
            return Vec::new();
        }
        let tail = p.kernel - p.stride;
        let t_full = p.t_out();
        let mut acc = vec![T::zero(); p.batch * t_full * p.c_out];
        if !self.partial.is_empty() {
            for b in 0..p.batch {
                acc[b * t_full * p.c_out..][..tail * p.c_out]
                    .copy_from_slice(&self.partial[b * tail * p.c_out..][..tail * p.c_out]);
            }
        }
        conv_transpose1d_accumulate(p, x, w_packed, &mut acc, t_full);
        let keep = p.t_in * p.stride;
        let mut y = vec![T::zero(); p.batch * p.c_out * keep];
        emit(&acc, t_full, 0, keep, p.batch, p.c_out, bias, &mut y);
        let mut partial = vec![T::zero(); p.batch * tail * p.c_out];
        for b in 0..p.batch {
            partial[b * tail * p.c_out..][..tail * p.c_out]
                .copy_from_slice(&acc[(b * t_full + keep) * p.c_out..][..tail * p.c_out]);
        }
        self.partial = partial;
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use s2s_core::SplitMix64;

    fn rand_vec(n: usize, rng: &mut SplitMix64) -> Vec<f64> {
        (0..n).map(|_| rng.next_f64() * 2.0 - 1.0).collect()
    }

    /// Direct scatter definition.
    fn naive(p: &ConvTranspose1dParams, x: &[f64], w: &[f64], bias: Option<&[f64]>) -> Vec<f64> {
        let (cig, cog, t_out) = (p.c_in / p.groups, p.c_out / p.groups, p.t_out());
        let mut y = vec![0.0; p.batch * p.c_out * t_out];
        for b in 0..p.batch {
            for ci in 0..p.c_in {
                let g = ci / cig;
                for ol in 0..cog {
                    let o = g * cog + ol;
                    for t in 0..p.t_in {
                        for kk in 0..p.kernel {
                            y[(b * p.c_out + o) * t_out + t * p.stride + kk] += x
                                [(b * p.c_in + ci) * p.t_in + t]
                                * w[(ci * cog + ol) * p.kernel + kk];
                        }
                    }
                }
            }
            if let Some(bb) = bias {
                for o in 0..p.c_out {
                    for u in 0..t_out {
                        y[(b * p.c_out + o) * t_out + u] += bb[o];
                    }
                }
            }
        }
        y
    }

    #[test]
    fn known_small() {
        // x = [1 2], w = [1 10 100], stride 2 -> [1 10 102 20 200]
        let p = ConvTranspose1dParams {
            batch: 1,
            c_in: 1,
            c_out: 1,
            t_in: 2,
            kernel: 3,
            stride: 2,
            groups: 1,
        };
        let mut y = vec![0.0f64; 5];
        conv_transpose1d_fwd(&p, &[1.0, 2.0], &[1.0, 10.0, 100.0], None, &mut y);
        assert_eq!(y, [1.0, 10.0, 102.0, 20.0, 200.0]);
    }

    #[test]
    fn matches_naive() {
        let mut rng = SplitMix64::new(5);
        for &(c_in, c_out, kernel, stride, groups, t_in) in &[
            (3, 2, 4, 2, 1, 7),
            (4, 6, 16, 8, 2, 5),
            (5, 5, 4, 2, 5, 9), // depthwise, like Mimi's upsample
            (2, 3, 3, 1, 1, 6),
        ] {
            let p = ConvTranspose1dParams {
                batch: 2,
                c_in,
                c_out,
                t_in,
                kernel,
                stride,
                groups,
            };
            let x = rand_vec(2 * c_in * t_in, &mut rng);
            let w = rand_vec(c_in * c_out / groups * kernel, &mut rng);
            let b = rand_vec(c_out, &mut rng);
            let mut y = vec![0.0; 2 * c_out * p.t_out()];
            conv_transpose1d_fwd(&p, &x, &w, Some(&b), &mut y);
            for (a, e) in y.iter().zip(naive(&p, &x, &w, Some(&b))) {
                assert!((a - e).abs() < 1e-12, "{p:?}");
            }
        }
    }

    #[test]
    fn streaming_equals_offline_bit_exact() {
        let mut rng = SplitMix64::new(9);
        for &(kernel, stride, groups, chunk) in
            &[(4, 2, 1, 1), (16, 8, 1, 2), (4, 2, 3, 3), (3, 3, 1, 2)]
        {
            let (c_in, c_out, chunks) = (3usize, 6usize, 7usize);
            let t_in = chunk * chunks;
            let p = ConvTranspose1dParams {
                batch: 1,
                c_in,
                c_out,
                t_in,
                kernel,
                stride,
                groups,
            };
            let x: Vec<f32> = (0..c_in * t_in)
                .map(|_| (rng.next_f64() * 2.0 - 1.0) as f32)
                .collect();
            let w: Vec<f32> = (0..c_in * c_out / groups * kernel)
                .map(|_| (rng.next_f64() * 2.0 - 1.0) as f32)
                .collect();
            let b: Vec<f32> = (0..c_out).map(|_| rng.next_f64() as f32).collect();
            let wp = pack_weight(&w, c_in, c_out / groups, kernel);
            let y_off = causal_conv_transpose1d_fwd(&p, &x, &wp, Some(&b));
            let n_off = t_in * stride;
            let mut st = ConvTranspose1dState::new();
            let mut outs: Vec<Vec<f32>> = vec![Vec::new(); c_out];
            for c in 0..chunks {
                let mut xc = Vec::new();
                for ci in 0..c_in {
                    xc.extend_from_slice(&x[ci * t_in + c * chunk..ci * t_in + (c + 1) * chunk]);
                }
                let pc = ConvTranspose1dParams { t_in: chunk, ..p };
                let y = st.step(&pc, &xc, &wp, Some(&b));
                let n = chunk * stride;
                for o in 0..c_out {
                    outs[o].extend_from_slice(&y[o * n..(o + 1) * n]);
                }
            }
            for o in 0..c_out {
                assert_eq!(&outs[o][..], &y_off[o * n_off..(o + 1) * n_off]);
            }
        }
    }
}
