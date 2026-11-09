//! 1-D convolution with stride, dilation and groups, causal padding, folded
//! weight norm, and streaming state (plan §7.2 kernel #15, used by Mimi).
//!
//! Layouts (row-major):
//! - `x`: `[batch, c_in, t_in]`
//! - `w`: `[c_out, c_in / groups, kernel]` (PyTorch / safetensors convention)
//! - `y`: `[batch, c_out, t_out]`
//!
//! **Reduction order.** Every output element is a dot product over the
//! flattened `(c_in_of_group, tap)` index, computed by [`dot_lanes`]: [`LANES`]
//! interleaved partial sums combined in a fixed tree, then the bias is added.
//! The result for one output element therefore depends only on the inputs
//! it reads, never on `t_in`, the batch size or how the sequence is split
//! into chunks. That is what makes [`CausalConv1dState::step`] reproduce the
//! offline [`causal_conv1d_fwd`] **bit for bit**, and it lets the codec's
//! linear layers ([`linear_lanes_fwd`]) share the property.

use s2s_core::Float;

/// Number of interleaved partial sums used by [`dot_lanes`].
pub const LANES: usize = 8;

/// Rows processed together by [`linear_lanes_fwd`] so each weight row is
/// read from memory once per tile (does not change any result).
const ROW_TILE: usize = 4;

/// Dot product with [`LANES`] interleaved partial sums, combined by pairwise
/// halving (`acc[i] += acc[i + LANES / 2]`, ...), then the scalar tail.
/// Deterministic and vectorizable; the summation order depends only on
/// `a.len()`.
#[inline]
pub fn dot_lanes<T: Float>(a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len(), "dot_lanes: length mismatch");
    let mut acc = [T::zero(); LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (xa, xb) in (&mut ca).zip(&mut cb) {
        for l in 0..LANES {
            acc[l] = acc[l] + xa[l] * xb[l];
        }
    }
    let mut width = LANES / 2;
    while width > 0 {
        for l in 0..width {
            acc[l] = acc[l] + acc[l + width];
        }
        width /= 2;
    }
    let mut tail = T::zero();
    for (&x, &y) in ca.remainder().iter().zip(cb.remainder()) {
        tail = tail + x * y;
    }
    acc[0] + tail
}

/// `Linear` forward `y[rows, out] = x[rows, in] @ w[out, in]^T (+ bias)`, with
/// every element reduced by [`dot_lanes`] and the bias added last.
///
/// Same math as [`super::gemm::linear_fwd`], but with a fixed, row-count
/// independent reduction order (see the module docs).
pub fn linear_lanes_fwd<T: Float>(
    x: &[T],
    w: &[T],
    bias: Option<&[T]>,
    y: &mut [T],
    rows: usize,
    d_in: usize,
    d_out: usize,
) {
    assert_eq!(x.len(), rows * d_in, "linear_lanes_fwd: x has wrong length");
    assert_eq!(
        w.len(),
        d_out * d_in,
        "linear_lanes_fwd: w has wrong length"
    );
    assert_eq!(
        y.len(),
        rows * d_out,
        "linear_lanes_fwd: y has wrong length"
    );
    if let Some(b) = bias {
        assert_eq!(b.len(), d_out, "linear_lanes_fwd: bias has wrong length");
    }
    let mut r0 = 0;
    while r0 < rows {
        let r1 = (r0 + ROW_TILE).min(rows);
        for o in 0..d_out {
            let wo = &w[o * d_in..(o + 1) * d_in];
            for r in r0..r1 {
                let acc = dot_lanes(&x[r * d_in..(r + 1) * d_in], wo);
                y[r * d_out + o] = match bias {
                    Some(b) => acc + b[o],
                    None => acc,
                };
            }
        }
        r0 = r1;
    }
}

/// Convolution dimensions and options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv1dParams {
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
    /// Stride.
    pub stride: usize,
    /// Dilation.
    pub dilation: usize,
    /// Groups. Must divide `c_in` and `c_out`.
    pub groups: usize,
}

impl Conv1dParams {
    /// Kernel extent including dilation: `(kernel - 1) * dilation + 1`.
    pub fn effective_kernel(&self) -> usize {
        (self.kernel - 1) * self.dilation + 1
    }

    /// Output length of an unpadded ("valid") convolution; 0 if the input is
    /// shorter than the effective kernel.
    pub fn t_out(&self) -> usize {
        let k = self.effective_kernel();
        if self.t_in < k {
            0
        } else {
            (self.t_in - k) / self.stride + 1
        }
    }

    fn check(&self) {
        assert!(self.kernel > 0 && self.stride > 0 && self.dilation > 0);
        assert!(
            self.groups > 0
                && self.c_in.is_multiple_of(self.groups)
                && self.c_out.is_multiple_of(self.groups),
            "conv1d: groups must divide c_in and c_out"
        );
    }
}

/// Output positions processed per im2col block (bounds scratch memory).
const BLOCK_T: usize = 256;

/// Unpadded convolution. `y` must have length `batch * c_out * p.t_out()`.
pub fn conv1d_fwd<T: Float>(p: &Conv1dParams, x: &[T], w: &[T], bias: Option<&[T]>, y: &mut [T]) {
    p.check();
    let cig = p.c_in / p.groups;
    let cog = p.c_out / p.groups;
    let t_out = p.t_out();
    let ck = cig * p.kernel;
    assert_eq!(
        x.len(),
        p.batch * p.c_in * p.t_in,
        "conv1d: x has wrong length"
    );
    assert_eq!(w.len(), p.c_out * ck, "conv1d: w has wrong length");
    assert_eq!(
        y.len(),
        p.batch * p.c_out * t_out,
        "conv1d: y has wrong length"
    );
    if let Some(b) = bias {
        assert_eq!(b.len(), p.c_out, "conv1d: bias has wrong length");
    }
    let mut cols = vec![T::zero(); BLOCK_T.min(t_out.max(1)) * ck];
    let mut out = vec![T::zero(); BLOCK_T.min(t_out.max(1)) * cog];
    for b in 0..p.batch {
        for g in 0..p.groups {
            let wg = &w[g * cog * ck..(g + 1) * cog * ck];
            let bg = bias.map(|bb| &bb[g * cog..(g + 1) * cog]);
            let mut t0 = 0;
            while t0 < t_out {
                let nt = BLOCK_T.min(t_out - t0);
                // im2col: cols[t][ci * kernel + kk] = x[b, g*cig + ci, (t0+t)*stride + kk*dilation]
                for t in 0..nt {
                    let row = &mut cols[t * ck..(t + 1) * ck];
                    let start = (t0 + t) * p.stride;
                    for ci in 0..cig {
                        let xr = &x[(b * p.c_in + g * cig + ci) * p.t_in..][..p.t_in];
                        for kk in 0..p.kernel {
                            row[ci * p.kernel + kk] = xr[start + kk * p.dilation];
                        }
                    }
                }
                let out = &mut out[..nt * cog];
                linear_lanes_fwd(&cols[..nt * ck], wg, bg, out, nt, ck, cog);
                for o in 0..cog {
                    let yr = &mut y[(b * p.c_out + g * cog + o) * t_out..][..t_out];
                    for t in 0..nt {
                        yr[t0 + t] = out[t * cog + o];
                    }
                }
                t0 += nt;
            }
        }
    }
}

/// How padding values are produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadMode {
    /// Zeros.
    Constant,
    /// Repeat the edge sample.
    Replicate,
}

/// Pad every row of `x: [rows, t]` on the time axis. Returns `[rows, left + t + right]`.
///
/// `Replicate` needs `t > 0`.
pub fn pad1d<T: Float>(
    x: &[T],
    rows: usize,
    t: usize,
    left: usize,
    right: usize,
    mode: PadMode,
) -> Vec<T> {
    assert_eq!(x.len(), rows * t, "pad1d: x has wrong length");
    let tp = left + t + right;
    let mut out = vec![T::zero(); rows * tp];
    for r in 0..rows {
        let src = &x[r * t..(r + 1) * t];
        let dst = &mut out[r * tp..(r + 1) * tp];
        dst[left..left + t].copy_from_slice(src);
        if mode == PadMode::Replicate && (left > 0 || right > 0) {
            assert!(t > 0, "pad1d: replicate padding of an empty row");
            dst[..left].iter_mut().for_each(|v| *v = src[0]);
            dst[left + t..].iter_mut().for_each(|v| *v = src[t - 1]);
        }
    }
    out
}

/// Padding of a causal "same-rate" convolution, as in Kyutai's
/// `StreamableConv1d`: `(left, right)` where `left = k_eff - stride` and
/// `right` is the extra padding that makes the last (partial) window complete.
pub fn causal_padding(t_in: usize, effective_kernel: usize, stride: usize) -> (usize, usize) {
    assert!(
        effective_kernel >= stride,
        "causal conv: kernel smaller than stride"
    );
    let padding_total = effective_kernel - stride;
    // n_frames = ceil((t_in + padding_total - k) / stride) + 1, saturating.
    let span = (t_in + padding_total).saturating_sub(effective_kernel);
    let n_frames = span.div_ceil(stride) + 1;
    let ideal_len = ((n_frames - 1) * stride + effective_kernel).saturating_sub(padding_total);
    (padding_total, ideal_len.saturating_sub(t_in))
}

/// Causal convolution over a whole sequence: left padding `k_eff - stride`
/// plus [`causal_padding`]'s extra right padding, then [`conv1d_fwd`].
///
/// `p.t_in` is the unpadded length. Returns `(y, t_out)` with
/// `y: [batch, c_out, t_out]`; when `t_in` is a multiple of `stride`,
/// `t_out = t_in / stride`.
pub fn causal_conv1d_fwd<T: Float>(
    p: &Conv1dParams,
    x: &[T],
    w: &[T],
    bias: Option<&[T]>,
    mode: PadMode,
) -> (Vec<T>, usize) {
    let (left, right) = causal_padding(p.t_in, p.effective_kernel(), p.stride);
    let xp = pad1d(x, p.batch * p.c_in, p.t_in, left, right, mode);
    let pp = Conv1dParams {
        t_in: left + p.t_in + right,
        ..*p
    };
    let t_out = pp.t_out();
    let mut y = vec![T::zero(); p.batch * p.c_out * t_out];
    conv1d_fwd(&pp, &xp, w, bias, &mut y);
    (y, t_out)
}

/// Fold PyTorch weight normalization into a plain weight:
/// `w[o, ..] = g[o] * v[o, ..] / ||v[o, ..]||_2`, the norm taken over every
/// dimension but the first (`dim = 0`). `v` is `[d0, per]`, `g` is `[d0]`.
///
/// The squared norm is a plain sequential sum in `T` and the product is
/// computed as `(v * g) / norm`, the same operations in the same order as
/// candle's CPU backend in Kyutai's loader, so folded weights match theirs
/// bit for bit.
pub fn weight_norm_fold<T: Float>(g: &[T], v: &[T], d0: usize, per: usize) -> Vec<T> {
    assert_eq!(g.len(), d0, "weight_norm_fold: g has wrong length");
    assert_eq!(v.len(), d0 * per, "weight_norm_fold: v has wrong length");
    let mut w = vec![T::zero(); d0 * per];
    for o in 0..d0 {
        let row = &v[o * per..(o + 1) * per];
        let mut n2 = T::zero();
        for &a in row {
            n2 = n2 + a * a;
        }
        let norm = n2.sqrt();
        for (dst, &a) in w[o * per..(o + 1) * per].iter_mut().zip(row) {
            *dst = a * g[o] / norm;
        }
    }
    w
}

/// Streaming state of a causal convolution (Kyutai's `StreamableConv1d::step`).
///
/// Feed consecutive chunks of any length; the concatenated outputs equal
/// [`causal_conv1d_fwd`] on the concatenated input, bit for bit, except that
/// the offline version's extra right padding of a trailing partial window is
/// never applied (those outputs only appear once enough input has arrived).
#[derive(Clone, Debug, Default)]
pub struct CausalConv1dState<T> {
    /// Unconsumed input `[batch, c_in, prev_len]` (including left padding).
    prev: Vec<T>,
    prev_len: usize,
    left_pad_applied: bool,
}

impl<T: Float> CausalConv1dState<T> {
    /// Fresh state (start of a stream).
    pub fn new() -> Self {
        Self {
            prev: Vec::new(),
            prev_len: 0,
            left_pad_applied: false,
        }
    }

    /// Forget everything (start a new stream).
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Process one chunk `x: [batch, c_in, p.t_in]` (`p.t_in` is the chunk
    /// length). Returns `(y, t_out)` with `y: [batch, c_out, t_out]`, possibly
    /// empty while the state fills up.
    pub fn step(
        &mut self,
        p: &Conv1dParams,
        x: &[T],
        w: &[T],
        bias: Option<&[T]>,
        mode: PadMode,
    ) -> (Vec<T>, usize) {
        let rows = p.batch * p.c_in;
        assert_eq!(x.len(), rows * p.t_in, "conv1d step: x has wrong length");
        if p.t_in == 0 {
            return (Vec::new(), 0);
        }
        let k = p.effective_kernel();
        assert!(k >= p.stride, "causal conv: kernel smaller than stride");
        let (x, t_new) = if self.left_pad_applied {
            (x.to_vec(), p.t_in)
        } else {
            self.left_pad_applied = true;
            let left = k - p.stride;
            (pad1d(x, rows, p.t_in, left, 0, mode), left + p.t_in)
        };
        // Concatenate the carried-over input with the new chunk.
        let len = self.prev_len + t_new;
        let mut buf = vec![T::zero(); rows * len];
        for r in 0..rows {
            let dst = &mut buf[r * len..(r + 1) * len];
            dst[..self.prev_len]
                .copy_from_slice(&self.prev[r * self.prev_len..(r + 1) * self.prev_len]);
            dst[self.prev_len..].copy_from_slice(&x[r * t_new..(r + 1) * t_new]);
        }
        let n_frames = (len + p.stride).saturating_sub(k) / p.stride;
        if n_frames == 0 {
            self.prev = buf;
            self.prev_len = len;
            return (Vec::new(), 0);
        }
        let used = (n_frames - 1) * p.stride + k;
        let offset = n_frames * p.stride;
        let pp = Conv1dParams { t_in: used, ..*p };
        let mut xin = vec![T::zero(); rows * used];
        for r in 0..rows {
            xin[r * used..(r + 1) * used].copy_from_slice(&buf[r * len..r * len + used]);
        }
        let mut y = vec![T::zero(); p.batch * p.c_out * n_frames];
        conv1d_fwd(&pp, &xin, w, bias, &mut y);
        let keep = len - offset;
        let mut prev = vec![T::zero(); rows * keep];
        for r in 0..rows {
            prev[r * keep..(r + 1) * keep].copy_from_slice(&buf[r * len + offset..(r + 1) * len]);
        }
        self.prev = prev;
        self.prev_len = keep;
        (y, n_frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use s2s_core::SplitMix64;

    fn rand_vec(n: usize, rng: &mut SplitMix64) -> Vec<f64> {
        (0..n).map(|_| rng.next_f64() * 2.0 - 1.0).collect()
    }

    /// Direct definition, for checking the im2col implementation.
    fn conv_naive(p: &Conv1dParams, x: &[f64], w: &[f64], bias: Option<&[f64]>) -> Vec<f64> {
        let (cig, cog, t_out) = (p.c_in / p.groups, p.c_out / p.groups, p.t_out());
        let mut y = vec![0.0; p.batch * p.c_out * t_out];
        for b in 0..p.batch {
            for o in 0..p.c_out {
                let g = o / cog;
                for t in 0..t_out {
                    let mut acc = bias.map_or(0.0, |bb| bb[o]);
                    for ci in 0..cig {
                        for kk in 0..p.kernel {
                            let xi = x[(b * p.c_in + g * cig + ci) * p.t_in
                                + t * p.stride
                                + kk * p.dilation];
                            acc += w[(o * cig + ci) * p.kernel + kk] * xi;
                        }
                    }
                    y[(b * p.c_out + o) * t_out + t] = acc;
                }
            }
        }
        y
    }

    #[test]
    fn dot_lanes_matches_plain_sum() {
        let mut rng = SplitMix64::new(1);
        for n in [0, 1, 7, 8, 9, 31, 100] {
            let a = rand_vec(n, &mut rng);
            let b = rand_vec(n, &mut rng);
            let plain: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!((dot_lanes(&a, &b) - plain).abs() < 1e-12);
        }
    }

    #[test]
    fn known_small_conv() {
        // x = [1 2 3 4], w = [1 -1] (difference), stride 1 -> [-1 -1 -1]
        let p = Conv1dParams {
            batch: 1,
            c_in: 1,
            c_out: 1,
            t_in: 4,
            kernel: 2,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let mut y = vec![0.0f64; 3];
        conv1d_fwd(
            &p,
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, -1.0],
            Some(&[0.5]),
            &mut y,
        );
        assert_eq!(y, [-0.5, -0.5, -0.5]);
    }

    #[test]
    fn matches_naive_with_stride_dilation_groups() {
        let mut rng = SplitMix64::new(7);
        for &(c_in, c_out, kernel, stride, dilation, groups, t_in) in &[
            (3, 5, 3, 1, 1, 1, 11),
            (4, 6, 4, 2, 1, 2, 17),
            (4, 4, 3, 1, 2, 4, 9),
            (2, 3, 7, 3, 2, 1, 40),
            (6, 2, 1, 1, 1, 2, 300), // crosses an im2col block boundary
        ] {
            let p = Conv1dParams {
                batch: 2,
                c_in,
                c_out,
                t_in,
                kernel,
                stride,
                dilation,
                groups,
            };
            let x = rand_vec(2 * c_in * t_in, &mut rng);
            let w = rand_vec(c_out * c_in / groups * kernel, &mut rng);
            let b = rand_vec(c_out, &mut rng);
            let mut y = vec![0.0; 2 * c_out * p.t_out()];
            conv1d_fwd(&p, &x, &w, Some(&b), &mut y);
            let r = conv_naive(&p, &x, &w, Some(&b));
            for (a, e) in y.iter().zip(&r) {
                assert!((a - e).abs() < 1e-12, "{a} vs {e} for {p:?}");
            }
        }
    }

    #[test]
    fn causal_padding_matches_kyutai_formula() {
        // Kyutai: n_frames = (len + pt - k) / stride + 1 (as f64, ceil);
        // ideal = (ceil(n_frames) - 1) * stride + k - pt; extra = ideal - len.
        for k in 1..9usize {
            for stride in 1..=k {
                for len in 0..40usize {
                    let pt = k - stride;
                    let nf = (len + pt).saturating_sub(k) as f64 / stride as f64 + 1.0;
                    let ideal = ((nf.ceil() as usize - 1) * stride + k).saturating_sub(pt);
                    let extra = ideal.saturating_sub(len);
                    assert_eq!(causal_padding(len, k, stride), (pt, extra));
                }
            }
        }
    }

    #[test]
    fn causal_output_length_and_causality() {
        // A causal conv's output at t only depends on inputs <= its window end.
        let mut rng = SplitMix64::new(3);
        let p = Conv1dParams {
            batch: 1,
            c_in: 2,
            c_out: 3,
            t_in: 24,
            kernel: 8,
            stride: 4,
            dilation: 1,
            groups: 1,
        };
        let x = rand_vec(2 * 24, &mut rng);
        let w = rand_vec(3 * 2 * 8, &mut rng);
        let (y, t_out) = causal_conv1d_fwd(&p, &x, &w, None, PadMode::Constant);
        assert_eq!(t_out, 6);
        // Perturb the last 4 input samples: only the last output frame changes.
        let mut x2 = x.clone();
        for c in 0..2 {
            for t in 20..24 {
                x2[c * 24 + t] += 1.0;
            }
        }
        let (y2, _) = causal_conv1d_fwd(&p, &x2, &w, None, PadMode::Constant);
        for o in 0..3 {
            for t in 0..5 {
                assert_eq!(y[o * 6 + t], y2[o * 6 + t]);
            }
            assert_ne!(y[o * 6 + 5], y2[o * 6 + 5]);
        }
    }

    #[test]
    fn replicate_padding() {
        let x = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = pad1d(&x, 2, 3, 2, 1, PadMode::Replicate);
        assert_eq!(
            y,
            [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 6.0, 6.0]
        );
        let z = pad1d(&x, 2, 3, 1, 0, PadMode::Constant);
        assert_eq!(z, [0.0, 1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn weight_norm_fold_normalizes_rows() {
        let v = [3.0f64, 4.0, 0.0, 1.0, 0.0, 0.0];
        let w = weight_norm_fold(&[2.0, 5.0], &v, 2, 3);
        assert_eq!(w, [1.2, 1.6, 0.0, 5.0, 0.0, 0.0]);
    }

    #[test]
    fn streaming_equals_offline_bit_exact() {
        let mut rng = SplitMix64::new(11);
        for &(kernel, stride, dilation, chunk, mode) in &[
            (3, 1, 1, 1, PadMode::Constant),
            (3, 1, 2, 2, PadMode::Constant),
            (8, 4, 1, 4, PadMode::Constant),
            (8, 4, 1, 3, PadMode::Constant),
            (4, 2, 1, 2, PadMode::Replicate),
            (7, 1, 1, 5, PadMode::Replicate),
        ] {
            let (c_in, c_out, chunks) = (3usize, 2usize, 12usize);
            let t_in = chunk * chunks;
            let p = Conv1dParams {
                batch: 1,
                c_in,
                c_out,
                t_in,
                kernel,
                stride,
                dilation,
                groups: 1,
            };
            let x: Vec<f32> = (0..c_in * t_in)
                .map(|_| (rng.next_f64() * 2.0 - 1.0) as f32)
                .collect();
            let w: Vec<f32> = (0..c_out * c_in * kernel)
                .map(|_| (rng.next_f64() * 2.0 - 1.0) as f32)
                .collect();
            let b: Vec<f32> = (0..c_out).map(|_| rng.next_f64() as f32).collect();
            let (y_off, t_off) = causal_conv1d_fwd(&p, &x, &w, Some(&b), mode);
            let mut st = CausalConv1dState::new();
            let mut outs: Vec<Vec<f32>> = vec![Vec::new(); c_out];
            for c in 0..chunks {
                let mut xc = Vec::with_capacity(c_in * chunk);
                for ci in 0..c_in {
                    xc.extend_from_slice(&x[ci * t_in + c * chunk..ci * t_in + (c + 1) * chunk]);
                }
                let pc = Conv1dParams { t_in: chunk, ..p };
                let (y, n) = st.step(&pc, &xc, &w, Some(&b), mode);
                for o in 0..c_out {
                    outs[o].extend_from_slice(&y[o * n..(o + 1) * n]);
                }
            }
            // Streaming never applies the trailing extra padding.
            let n_stream = outs[0].len();
            assert!(
                n_stream <= t_off && t_off - n_stream <= 1,
                "{n_stream} vs {t_off}"
            );
            for o in 0..c_out {
                assert_eq!(
                    &outs[o][..],
                    &y_off[o * t_off..o * t_off + n_stream],
                    "kernel {kernel} stride {stride} chunk {chunk}"
                );
            }
        }
    }
}
