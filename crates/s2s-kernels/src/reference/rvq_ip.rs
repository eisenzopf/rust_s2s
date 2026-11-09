//! Residual vector quantization with the nearest codeword found in
//! inner-product form (plan §7.2 kernel #17, as Mimi / Kyutai compute it).
//!
//! `argmin_e ||r - c_e||^2 = argmin_e (||c_e||^2 / 2 - <r, c_e>)`: the
//! `||r||^2` term is the same for every entry, so the search is one GEMM
//! against the codebook plus precomputed half squared norms. Kyutai's Mimi
//! (`EuclideanCodebook::encode_slow`) uses exactly this score; using the same
//! form here keeps rare near-ties resolved the same way. Mathematically it is
//! [`super::rvq::rvq_encode`]; decoding is unchanged ([`super::rvq::rvq_decode`]).

use s2s_core::{cast, Float};

use super::conv1d::dot_lanes;

/// `||c_e||^2 / 2` for every entry of `codebook: [entries, d]`.
pub fn half_sq_norms<T: Float>(codebook: &[T], entries: usize, d: usize) -> Vec<T> {
    assert_eq!(
        codebook.len(),
        entries * d,
        "half_sq_norms: codebook has wrong length"
    );
    let half: T = cast(0.5);
    (0..entries)
        .map(|e| {
            let c = &codebook[e * d..(e + 1) * d];
            c.iter().map(|&v| v * v).sum::<T>() * half
        })
        .collect()
}

/// Index of the codeword minimizing `half_norms[e] - <x, c_e>`; ties go to
/// the lower index.
pub fn nearest_code_ip<T: Float>(
    x: &[T],
    codebook: &[T],
    half_norms: &[T],
    entries: usize,
    d: usize,
) -> usize {
    assert_eq!(x.len(), d, "nearest_code_ip: x has wrong length");
    assert_eq!(
        codebook.len(),
        entries * d,
        "nearest_code_ip: codebook has wrong length"
    );
    assert_eq!(
        half_norms.len(),
        entries,
        "nearest_code_ip: half_norms has wrong length"
    );
    let mut best = (0usize, T::infinity());
    for e in 0..entries {
        let score = half_norms[e] - dot_lanes(x, &codebook[e * d..(e + 1) * d]);
        if score < best.1 {
            best = (e, score);
        }
    }
    best.0
}

/// Encode `x: [n, d]` into `codes: [n, levels]`, level by level on the
/// residual. Same contract as [`super::rvq::rvq_encode`], plus
/// `half_norms[l]` = [`half_sq_norms`] of `codebooks[l]`.
pub fn rvq_encode_ip<T: Float>(
    x: &[T],
    codebooks: &[&[T]],
    half_norms: &[&[T]],
    entries: &[usize],
    n: usize,
    d: usize,
    codes: &mut [u32],
    residual_out: Option<&mut [T]>,
) {
    let levels = codebooks.len();
    assert!(entries.len() == levels && half_norms.len() == levels);
    assert_eq!(x.len(), n * d, "rvq_encode_ip: x has wrong length");
    assert_eq!(
        codes.len(),
        n * levels,
        "rvq_encode_ip: codes has wrong length"
    );
    let mut residual = x.to_vec();
    for r in 0..n {
        let res = &mut residual[r * d..(r + 1) * d];
        for l in 0..levels {
            let e = nearest_code_ip(res, codebooks[l], half_norms[l], entries[l], d);
            codes[r * levels + l] = e as u32;
            let c = &codebooks[l][e * d..(e + 1) * d];
            for (v, &cv) in res.iter_mut().zip(c) {
                *v = *v - cv;
            }
        }
    }
    if let Some(out) = residual_out {
        out.copy_from_slice(&residual);
    }
}

#[cfg(test)]
mod tests {
    use super::super::rvq::rvq_encode;
    use super::*;
    use s2s_core::SplitMix64;

    #[test]
    fn agrees_with_distance_form() {
        let mut rng = SplitMix64::new(42);
        let (n, d, levels, entries) = (20usize, 6usize, 3usize, 17usize);
        let cbs: Vec<Vec<f64>> = (0..levels)
            .map(|_| (0..entries * d).map(|_| rng.normal()).collect())
            .collect();
        let norms: Vec<Vec<f64>> = cbs.iter().map(|c| half_sq_norms(c, entries, d)).collect();
        let x: Vec<f64> = (0..n * d).map(|_| rng.normal() * 2.0).collect();
        let cb_refs: Vec<&[f64]> = cbs.iter().map(|c| c.as_slice()).collect();
        let norm_refs: Vec<&[f64]> = norms.iter().map(|c| c.as_slice()).collect();
        let ents = vec![entries; levels];
        let (mut c1, mut c2) = (vec![0u32; n * levels], vec![0u32; n * levels]);
        let (mut r1, mut r2) = (vec![0.0; n * d], vec![0.0; n * d]);
        rvq_encode_ip(
            &x,
            &cb_refs,
            &norm_refs,
            &ents,
            n,
            d,
            &mut c1,
            Some(&mut r1),
        );
        rvq_encode(&x, &cb_refs, &ents, n, d, &mut c2, Some(&mut r2));
        assert_eq!(c1, c2);
        assert_eq!(r1, r2);
    }

    #[test]
    fn ties_go_to_lower_index() {
        // Two identical codewords: the first wins.
        let cb = [1.0f64, 0.0, 1.0, 0.0, -1.0, 0.0];
        let norms = half_sq_norms(&cb, 3, 2);
        assert_eq!(norms, [0.5, 0.5, 0.5]);
        assert_eq!(nearest_code_ip(&[0.9, 0.1], &cb, &norms, 3, 2), 0);
        assert_eq!(nearest_code_ip(&[-0.9, 0.1], &cb, &norms, 3, 2), 2);
    }
}
