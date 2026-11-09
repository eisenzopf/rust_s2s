//! Residual vector quantization, the core of the Mimi codec's bottleneck.
//!
//! Each level picks the codeword nearest (squared L2) to the current residual,
//! then subtracts it. Mimi uses one semantic level plus seven acoustic levels
//! of 2048 entries each; the projections in and out of the quantizer are
//! ordinary linear layers and live elsewhere.

use s2s_core::Float;

/// Encode `x: [n, d]` into `codes: [n, levels]`.
///
/// `codebooks[l]` is `[entries[l], d]`. Ties go to the lower index.
/// If `residual_out` is given (`[n, d]`) it receives the final residual
/// (quantization error), useful for commitment losses and diagnostics.
pub fn rvq_encode<T: Float>(
    x: &[T],
    codebooks: &[&[T]],
    entries: &[usize],
    n: usize,
    d: usize,
    codes: &mut [u32],
    residual_out: Option<&mut [T]>,
) {
    let levels = codebooks.len();
    assert_eq!(entries.len(), levels);
    assert_eq!(x.len(), n * d);
    assert_eq!(codes.len(), n * levels);
    let mut residual = x.to_vec();
    for r in 0..n {
        let res = &mut residual[r * d..(r + 1) * d];
        for l in 0..levels {
            let cb = codebooks[l];
            assert_eq!(cb.len(), entries[l] * d);
            let mut best = (0usize, T::infinity());
            for e in 0..entries[l] {
                let c = &cb[e * d..(e + 1) * d];
                let dist: T = res.iter().zip(c).map(|(&a, &b)| (a - b) * (a - b)).sum();
                if dist < best.1 {
                    best = (e, dist);
                }
            }
            codes[r * levels + l] = best.0 as u32;
            let c = &cb[best.0 * d..(best.0 + 1) * d];
            for x in 0..d {
                res[x] = res[x] - c[x];
            }
        }
    }
    if let Some(out) = residual_out {
        out.copy_from_slice(&residual);
    }
}

/// Decode `codes: [n, levels]` into `out: [n, d]` by summing codewords.
///
/// `levels_used` may be smaller than the number of codebooks to decode at a
/// lower bitrate (e.g. semantic level only).
pub fn rvq_decode<T: Float>(
    codes: &[u32],
    codebooks: &[&[T]],
    entries: &[usize],
    levels_used: usize,
    n: usize,
    d: usize,
    out: &mut [T],
) {
    let levels = codebooks.len();
    assert!(levels_used <= levels);
    assert_eq!(codes.len(), n * levels);
    assert_eq!(out.len(), n * d);
    out.iter_mut().for_each(|v| *v = T::zero());
    for r in 0..n {
        for l in 0..levels_used {
            let e = codes[r * levels + l] as usize;
            assert!(e < entries[l], "code {e} out of range at level {l}");
            let c = &codebooks[l][e * d..(e + 1) * d];
            for x in 0..d {
                out[r * d + x] = out[r * d + x] + c[x];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_sums_round_trip() {
        // Codebook 0 coarse, codebook 1 fine: any sum c0[i] + c1[j] is recovered.
        let cb0 = [0.0f64, 0.0, 10.0, 0.0, 0.0, 10.0]; // 3 entries, d=2
        let cb1 = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0, -1.0]; // 4 entries
        let cbs: [&[f64]; 2] = [&cb0, &cb1];
        let x = [11.0, 0.0, -1.0, 9.0];
        let mut codes = [0u32; 4];
        let mut res = [0.0; 4];
        rvq_encode(&x, &cbs, &[3, 4], 2, 2, &mut codes, Some(&mut res));
        assert_eq!(codes, [1, 1, 2, 3]);
        assert!(res.iter().all(|&r| r == 0.0));
        let mut out = [0.0; 4];
        rvq_decode(&codes, &cbs, &[3, 4], 2, 2, 2, &mut out);
        assert_eq!(out, x);
        // Decoding the first level only gives the coarse approximation.
        rvq_decode(&codes, &cbs, &[3, 4], 1, 2, 2, &mut out);
        assert_eq!(out, [10.0, 0.0, 0.0, 10.0]);
    }
}
