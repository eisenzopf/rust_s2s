//! Scaled dot-product attention with GQA and the masks the model needs.
//!
//! Layouts (row-major):
//! - `q`, `o`, `dq`, `do`: `[batch, q_heads, t, d]`
//! - `k`, `v`, `dk`, `dv`: `[batch, kv_heads, s, d]`
//! - `lse`: `[batch, q_heads, t]` (log-sum-exp per query row, saved for backward)
//!
//! Query row `i` sits at absolute position `i + q_offset` in the key sequence.
//! For training and prefill over a whole sequence, `t == s` and `q_offset == 0`.
//! For decoding one new frame against a KV cache of length `s`, `t == 1` and
//! `q_offset == s - 1`.
//!
//! The backward pass is the standard FlashAttention-style recomputation from
//! `lse`, so it needs only `q, k, v, o, lse, do`.

use s2s_core::{cast, Float};

/// Which keys each query may attend to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct AttnMask {
    /// Query at position `p` sees only keys at positions `<= p`.
    pub causal: bool,
    /// Sliding window: a query at `p` sees keys at positions `> p - window`
    /// (i.e. at most `window` keys including itself), except pinned ones.
    pub window: Option<usize>,
    /// Keys at positions `< pinned_prefix` are exempt from the sliding window
    /// (still subject to causality). Used to keep the system prompt and tool
    /// schemas visible for the whole session (plan §10.4).
    pub pinned_prefix: usize,
}

impl AttnMask {
    /// Whether a query at absolute position `qpos` may see the key at `kpos`.
    pub fn allows(&self, qpos: usize, kpos: usize) -> bool {
        if self.causal && kpos > qpos {
            return false;
        }
        if let Some(w) = self.window {
            if kpos >= self.pinned_prefix && qpos >= kpos + w {
                return false;
            }
        }
        true
    }
}

/// Problem dimensions and options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttnParams {
    /// Batch size.
    pub batch: usize,
    /// Number of query heads.
    pub q_heads: usize,
    /// Number of key/value heads. Must divide `q_heads` (GQA; equal = MHA).
    pub kv_heads: usize,
    /// Number of query rows.
    pub t: usize,
    /// Number of key/value rows.
    pub s: usize,
    /// Head dimension.
    pub d: usize,
    /// Score scale, normally `1 / sqrt(d)`.
    pub scale: f64,
    /// Absolute position of query row 0 within the key sequence.
    pub q_offset: usize,
    /// Mask.
    pub mask: AttnMask,
}

impl AttnParams {
    fn check(&self) {
        assert!(
            self.kv_heads > 0 && self.q_heads.is_multiple_of(self.kv_heads),
            "GQA: kv_heads must divide q_heads"
        );
        assert!(
            self.q_offset + self.t <= self.s,
            "queries extend past the key sequence"
        );
    }
    fn group(&self) -> usize {
        self.q_heads / self.kv_heads
    }
    fn q_idx(&self, b: usize, h: usize, i: usize) -> usize {
        ((b * self.q_heads + h) * self.t + i) * self.d
    }
    fn kv_idx(&self, b: usize, h: usize, j: usize) -> usize {
        ((b * self.kv_heads + h) * self.s + j) * self.d
    }
}

fn dot<T: Float>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Attention forward. Writes `o` and `lse`.
///
/// Rows with no visible key get `o = 0` and `lse = -inf`.
pub fn attention_fwd<T: Float>(
    p: &AttnParams,
    q: &[T],
    k: &[T],
    v: &[T],
    o: &mut [T],
    lse: &mut [T],
) {
    p.check();
    let (d, scale) = (p.d, cast::<T>(p.scale));
    assert_eq!(q.len(), p.batch * p.q_heads * p.t * d);
    assert_eq!(k.len(), p.batch * p.kv_heads * p.s * d);
    assert_eq!(v.len(), k.len());
    assert_eq!(o.len(), q.len());
    assert_eq!(lse.len(), p.batch * p.q_heads * p.t);
    let mut scores = vec![T::zero(); p.s];
    for b in 0..p.batch {
        for h in 0..p.q_heads {
            let kvh = h / p.group();
            for i in 0..p.t {
                let qi = &q[p.q_idx(b, h, i)..p.q_idx(b, h, i) + d];
                let qpos = i + p.q_offset;
                let mut m = T::neg_infinity();
                for j in 0..p.s {
                    scores[j] = if p.mask.allows(qpos, j) {
                        let kj = &k[p.kv_idx(b, kvh, j)..p.kv_idx(b, kvh, j) + d];
                        scale * dot(qi, kj)
                    } else {
                        T::neg_infinity()
                    };
                    m = m.max(scores[j]);
                }
                let out = &mut o[p.q_idx(b, h, i)..p.q_idx(b, h, i) + d];
                out.iter_mut().for_each(|x| *x = T::zero());
                let row = (b * p.q_heads + h) * p.t + i;
                if m == T::neg_infinity() {
                    lse[row] = T::neg_infinity();
                    continue;
                }
                let l: T = scores.iter().map(|&s| (s - m).exp()).sum();
                let row_lse = m + l.ln();
                lse[row] = row_lse;
                for j in 0..p.s {
                    if scores[j] == T::neg_infinity() {
                        continue;
                    }
                    let pj = (scores[j] - row_lse).exp();
                    let vj = &v[p.kv_idx(b, kvh, j)..p.kv_idx(b, kvh, j) + d];
                    for x in 0..d {
                        out[x] = out[x] + pj * vj[x];
                    }
                }
            }
        }
    }
}

/// Attention backward. Accumulates into `dq`, `dk`, `dv`.
///
/// With `P = softmax(S)`, `D_i = sum(dO_i * O_i)`:
/// `dV += P^T dO`, `dS = P * (dO V^T - D)`, `dQ += scale dS K`, `dK += scale dS^T Q`.
/// Gradients for KV heads shared by several query heads (GQA) are summed.
pub fn attention_bwd<T: Float>(
    p: &AttnParams,
    q: &[T],
    k: &[T],
    v: &[T],
    o: &[T],
    lse: &[T],
    d_o: &[T],
    dq: &mut [T],
    dk: &mut [T],
    dv: &mut [T],
) {
    p.check();
    let (d, scale) = (p.d, cast::<T>(p.scale));
    assert!(dq.len() == q.len() && dk.len() == k.len() && dv.len() == v.len());
    for b in 0..p.batch {
        for h in 0..p.q_heads {
            let kvh = h / p.group();
            for i in 0..p.t {
                let row = (b * p.q_heads + h) * p.t + i;
                if lse[row] == T::neg_infinity() {
                    continue;
                }
                let qo = p.q_idx(b, h, i);
                let qi = &q[qo..qo + d];
                let doi = &d_o[qo..qo + d];
                let di = dot(doi, &o[qo..qo + d]);
                let qpos = i + p.q_offset;
                for j in 0..p.s {
                    if !p.mask.allows(qpos, j) {
                        continue;
                    }
                    let ko = p.kv_idx(b, kvh, j);
                    let kj = &k[ko..ko + d];
                    let vj = &v[ko..ko + d];
                    let pij = (scale * dot(qi, kj) - lse[row]).exp();
                    let dp = dot(doi, vj);
                    let ds = pij * (dp - di);
                    for x in 0..d {
                        dv[ko + x] = dv[ko + x] + pij * doi[x];
                        dq[qo + x] = dq[qo + x] + scale * ds * kj[x];
                        dk[ko + x] = dk[ko + x] + scale * ds * qi[x];
                    }
                }
            }
        }
    }
}

/// Single-step decode: one query row per (batch, head) against a KV cache of
/// `cache_len` rows. Equivalent to [`attention_fwd`] with `t = 1` and
/// `q_offset = cache_len - 1`; kept as its own entry point because the GPU
/// version is a different kernel (split-K flash-decoding).
pub fn attention_decode<T: Float>(
    batch: usize,
    q_heads: usize,
    kv_heads: usize,
    cache_len: usize,
    d: usize,
    mask: AttnMask,
    q: &[T],
    k_cache: &[T],
    v_cache: &[T],
    o: &mut [T],
) {
    assert!(
        cache_len > 0,
        "decode needs at least one cached key (the current frame)"
    );
    let p = AttnParams {
        batch,
        q_heads,
        kv_heads,
        t: 1,
        s: cache_len,
        d,
        scale: 1.0 / (d as f64).sqrt(),
        q_offset: cache_len - 1,
        mask,
    };
    let mut lse = vec![T::zero(); batch * q_heads];
    attention_fwd(&p, q, k_cache, v_cache, o, &mut lse);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_rules() {
        let m = AttnMask {
            causal: true,
            window: Some(3),
            pinned_prefix: 2,
        };
        assert!(m.allows(10, 10));
        assert!(m.allows(10, 8));
        assert!(!m.allows(10, 7)); // outside window
        assert!(m.allows(10, 1)); // pinned
        assert!(!m.allows(10, 11)); // future
        assert!(!m.allows(0, 1));
    }

    #[test]
    fn single_key_returns_its_value() {
        let p = AttnParams {
            batch: 1,
            q_heads: 1,
            kv_heads: 1,
            t: 1,
            s: 1,
            d: 2,
            scale: 1.0,
            q_offset: 0,
            mask: AttnMask {
                causal: true,
                ..Default::default()
            },
        };
        let mut o = [0.0f64; 2];
        let mut lse = [0.0f64; 1];
        attention_fwd(&p, &[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0], &mut o, &mut lse);
        assert_eq!(o, [5.0, 6.0]);
        assert!((lse[0] - 11.0).abs() < 1e-12);
    }

    #[test]
    fn equal_scores_average_values() {
        // Zero query => uniform attention over visible keys.
        let p = AttnParams {
            batch: 1,
            q_heads: 1,
            kv_heads: 1,
            t: 2,
            s: 2,
            d: 1,
            scale: 1.0,
            q_offset: 0,
            mask: AttnMask {
                causal: true,
                ..Default::default()
            },
        };
        let mut o = [0.0f64; 2];
        let mut lse = [0.0f64; 2];
        attention_fwd(&p, &[0.0, 0.0], &[1.0, 1.0], &[2.0, 4.0], &mut o, &mut lse);
        assert_eq!(o, [2.0, 3.0]); // row 0 sees key 0 only; row 1 averages both
    }

    #[test]
    fn decode_matches_last_row_of_prefill() {
        let mut r = s2s_core::SplitMix64::new(3);
        let (hq, hkv, s, d) = (4, 2, 6, 8);
        let q: Vec<f64> = r.normal_vec(hq * s * d, 1.0);
        let k: Vec<f64> = r.normal_vec(hkv * s * d, 1.0);
        let v: Vec<f64> = r.normal_vec(hkv * s * d, 1.0);
        let mask = AttnMask {
            causal: true,
            window: Some(4),
            pinned_prefix: 1,
        };
        let p = AttnParams {
            batch: 1,
            q_heads: hq,
            kv_heads: hkv,
            t: s,
            s,
            d,
            scale: 1.0 / (d as f64).sqrt(),
            q_offset: 0,
            mask,
        };
        let mut o = vec![0.0; q.len()];
        let mut lse = vec![0.0; hq * s];
        attention_fwd(&p, &q, &k, &v, &mut o, &mut lse);

        let mut q_last = vec![0.0; hq * d];
        for h in 0..hq {
            q_last[h * d..(h + 1) * d].copy_from_slice(&q[(h * s + s - 1) * d..(h * s + s) * d]);
        }
        let mut o_dec = vec![0.0; hq * d];
        attention_decode(1, hq, hkv, s, d, mask, &q_last, &k, &v, &mut o_dec);
        for h in 0..hq {
            for x in 0..d {
                let a = o[(h * s + s - 1) * d + x];
                assert!((a - o_dec[h * d + x]).abs() < 1e-12);
            }
        }
    }
}
