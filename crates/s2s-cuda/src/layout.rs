//! Mapping a row-major GEMM onto cuBLASLt's column-major convention.
//!
//! [`s2s_kernels::reference::gemm::gemm`] computes, all row-major,
//! `C[m, n] = alpha * op(A)[m, k] @ op(B)[k, n] + beta * C`. cuBLASLt is
//! column-major. A row-major `[r, c]` buffer is the same bytes as a
//! column-major `[c, r]` matrix (its transpose), so with
//! `Cᵀ = op(B)ᵀ @ op(A)ᵀ` the row-major product is a column-major product with
//! the **operands swapped**:
//!
//! - cuBLASLt `m, n, k` = our `n, m, k`;
//! - cuBLASLt's `A` operand is our `B` buffer, its `B` operand our `A`;
//! - each operand keeps its own transpose flag;
//! - `C` / `D` are column-major `[n, m]` with leading dimension `n`.
//!
//! The bias epilogue adds a vector of length cuBLASLt-`m` (= our `n`) to every
//! column-major column, i.e. `bias[j]` to every row-major row: exactly the
//! `Linear` bias of [`s2s_kernels::reference::gemm::linear_fwd`].
//!
//! This module is plain Rust (no CUDA), so the mapping is unit-tested on the
//! CPU against the reference GEMM.

use s2s_kernels::reference::gemm::Trans;

/// A column-major matrix layout as cuBLASLt describes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColMajorLayout {
    /// Rows of the stored matrix.
    pub rows: u64,
    /// Columns of the stored matrix.
    pub cols: u64,
    /// Leading dimension (elements between consecutive columns), `>= rows`.
    pub ld: i64,
}

impl ColMajorLayout {
    fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows: rows as u64,
            cols: cols as u64,
            ld: rows.max(1) as i64,
        }
    }

    /// Elements a buffer must hold for this layout.
    pub fn required_len(&self) -> usize {
        if self.rows == 0 || self.cols == 0 {
            0
        } else {
            self.ld as usize * (self.cols as usize - 1) + self.rows as usize
        }
    }
}

/// A cuBLASLt matmul problem, `D[m, n] = alpha * op(A)[m, k] op(B)[k, n] +
/// beta * C (+ bias[m])`, all column-major.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LtProblem {
    /// cuBLASLt `m` (our `n`).
    pub m: u64,
    /// cuBLASLt `n` (our `m`).
    pub n: u64,
    /// Inner dimension.
    pub k: u64,
    /// Transpose cuBLASLt's `A` (our `tb`).
    pub trans_a: bool,
    /// Transpose cuBLASLt's `B` (our `ta`).
    pub trans_b: bool,
    /// Layout of cuBLASLt's `A`, which is **our `B` buffer**.
    pub a: ColMajorLayout,
    /// Layout of cuBLASLt's `B`, which is **our `A` buffer**.
    pub b: ColMajorLayout,
    /// Layout of `C` and `D` (our row-major `C`).
    pub c: ColMajorLayout,
}

impl LtProblem {
    /// The cuBLASLt problem for the row-major
    /// `C[m, n] = op(A)[m, k] @ op(B)[k, n]` of
    /// [`reference::gemm::gemm`](s2s_kernels::reference::gemm::gemm).
    pub fn row_major(ta: Trans, tb: Trans, m: usize, n: usize, k: usize) -> Self {
        // Our B stored row-major is [k, n] (or [n, k] if transposed), i.e.
        // column-major [n, k] (or [k, n]).
        let (a, trans_a) = match tb {
            Trans::No => (ColMajorLayout::new(n, k), false),
            Trans::Yes => (ColMajorLayout::new(k, n), true),
        };
        // Our A stored row-major is [m, k] (or [k, m]): column-major [k, m]
        // (or [m, k]).
        let (b, trans_b) = match ta {
            Trans::No => (ColMajorLayout::new(k, m), false),
            Trans::Yes => (ColMajorLayout::new(m, k), true),
        };
        Self {
            m: n as u64,
            n: m as u64,
            k: k as u64,
            trans_a,
            trans_b,
            a,
            b,
            c: ColMajorLayout::new(n, m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use s2s_kernels::reference::gemm::gemm;

    /// What cuBLASLt computes for `p`, written naively.
    fn lt_reference(
        p: &LtProblem,
        alpha: f64,
        a: &[f64],
        b: &[f64],
        beta: f64,
        c: &mut [f64],
        bias: Option<&[f64]>,
    ) {
        assert!(a.len() >= p.a.required_len());
        assert!(b.len() >= p.b.required_len());
        assert!(c.len() >= p.c.required_len());
        let (lda, ldb, ldc) = (p.a.ld as usize, p.b.ld as usize, p.c.ld as usize);
        let op_a = |i: usize, l: usize| {
            if p.trans_a {
                a[l + i * lda]
            } else {
                a[i + l * lda]
            }
        };
        let op_b = |l: usize, j: usize| {
            if p.trans_b {
                b[j + l * ldb]
            } else {
                b[l + j * ldb]
            }
        };
        for j in 0..p.n as usize {
            for i in 0..p.m as usize {
                let mut acc = 0.0;
                for l in 0..p.k as usize {
                    acc += op_a(i, l) * op_b(l, j);
                }
                let prev = if beta == 0.0 {
                    0.0
                } else {
                    beta * c[i + j * ldc]
                };
                c[i + j * ldc] = alpha * acc + prev + bias.map_or(0.0, |v| v[i]);
            }
        }
    }

    fn data(n: usize, seed: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (((i * 7 + seed * 13) % 17) as f64 - 8.0) * 0.25)
            .collect()
    }

    #[test]
    fn row_major_mapping_matches_the_reference_gemm() {
        let (m, n, k) = (3, 5, 4);
        for ta in [Trans::No, Trans::Yes] {
            for tb in [Trans::No, Trans::Yes] {
                for (alpha, beta) in [(1.0, 0.0), (0.5, 2.0)] {
                    for with_bias in [false, true] {
                        let a = data(m * k, 1);
                        let b = data(k * n, 2);
                        let bias = data(n, 3);
                        let c0 = data(m * n, 4);

                        let mut want = c0.clone();
                        gemm(ta, tb, m, n, k, alpha, &a, &b, beta, &mut want);
                        if with_bias {
                            for r in 0..m {
                                for j in 0..n {
                                    want[r * n + j] += bias[j];
                                }
                            }
                        }

                        let p = LtProblem::row_major(ta, tb, m, n, k);
                        let mut got = c0.clone();
                        // Operands swapped: cuBLASLt A = our B.
                        let bias = with_bias.then_some(&bias[..]);
                        lt_reference(&p, alpha, &b, &a, beta, &mut got, bias);
                        assert_eq!(got, want, "ta={ta:?} tb={tb:?} alpha={alpha} beta={beta}");
                    }
                }
            }
        }
    }

    #[test]
    fn layouts_and_extents() {
        let p = LtProblem::row_major(Trans::Yes, Trans::No, 2, 3, 4);
        assert_eq!((p.m, p.n, p.k), (3, 2, 4));
        assert_eq!(
            p.a,
            ColMajorLayout {
                rows: 3,
                cols: 4,
                ld: 3
            }
        );
        assert_eq!(
            p.b,
            ColMajorLayout {
                rows: 2,
                cols: 4,
                ld: 2
            }
        );
        assert!(p.trans_b && !p.trans_a);
        assert_eq!(p.c.required_len(), 6);
        assert_eq!(ColMajorLayout::new(0, 5).required_len(), 0);
        assert_eq!(ColMajorLayout::new(0, 5).ld, 1);
    }
}
