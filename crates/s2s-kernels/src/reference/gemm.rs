//! General matrix multiply and the `Linear` layer built on it.

use s2s_core::Float;

/// Whether an operand is used as stored or transposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trans {
    /// Use the matrix as stored.
    No,
    /// Use the transpose of the stored matrix.
    Yes,
}

/// `C = alpha * op(A) @ op(B) + beta * C`, all row-major.
///
/// `op(A)` is `[m, k]` and `op(B)` is `[k, n]`. Stored shapes are therefore
/// `A: [m, k]` (or `[k, m]` if transposed) and `B: [k, n]` (or `[n, k]`).
/// Accumulation is done in `T`; GPU kernels accumulate bf16 inputs in f32.
pub fn gemm<T: Float>(
    ta: Trans,
    tb: Trans,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    b: &[T],
    beta: T,
    c: &mut [T],
) {
    assert_eq!(a.len(), m * k, "gemm: A has wrong length");
    assert_eq!(b.len(), k * n, "gemm: B has wrong length");
    assert_eq!(c.len(), m * n, "gemm: C has wrong length");
    let a_at = |i: usize, p: usize| match ta {
        Trans::No => a[i * k + p],
        Trans::Yes => a[p * m + i],
    };
    let b_at = |p: usize, j: usize| match tb {
        Trans::No => b[p * n + j],
        Trans::Yes => b[j * k + p],
    };
    for i in 0..m {
        for j in 0..n {
            let mut acc = T::zero();
            for p in 0..k {
                acc = acc + a_at(i, p) * b_at(p, j);
            }
            let prev = if beta == T::zero() {
                T::zero()
            } else {
                beta * c[i * n + j]
            };
            c[i * n + j] = alpha * acc + prev;
        }
    }
}

/// `Linear` forward: `y[rows, out] = x[rows, in] @ w[out, in]^T (+ bias[out])`.
///
/// Weights are stored `[out, in]` (PyTorch / safetensors convention).
pub fn linear_fwd<T: Float>(
    x: &[T],
    w: &[T],
    bias: Option<&[T]>,
    y: &mut [T],
    rows: usize,
    d_in: usize,
    d_out: usize,
) {
    gemm(
        Trans::No,
        Trans::Yes,
        rows,
        d_out,
        d_in,
        T::one(),
        x,
        w,
        T::zero(),
        y,
    );
    if let Some(b) = bias {
        assert_eq!(b.len(), d_out);
        for r in 0..rows {
            for o in 0..d_out {
                y[r * d_out + o] = y[r * d_out + o] + b[o];
            }
        }
    }
}

/// `Linear` backward. Accumulates into `dx`, `dw` and `dbias` when given.
///
/// - `dx[rows, in] += dy @ w`
/// - `dw[out, in] += dy^T @ x`
/// - `dbias[out] += sum_rows(dy)`
pub fn linear_bwd<T: Float>(
    x: &[T],
    w: &[T],
    dy: &[T],
    dx: Option<&mut [T]>,
    dw: Option<&mut [T]>,
    dbias: Option<&mut [T]>,
    rows: usize,
    d_in: usize,
    d_out: usize,
) {
    if let Some(dx) = dx {
        gemm(
            Trans::No,
            Trans::No,
            rows,
            d_in,
            d_out,
            T::one(),
            dy,
            w,
            T::one(),
            dx,
        );
    }
    if let Some(dw) = dw {
        gemm(
            Trans::Yes,
            Trans::No,
            d_out,
            d_in,
            rows,
            T::one(),
            dy,
            x,
            T::one(),
            dw,
        );
    }
    if let Some(db) = dbias {
        assert_eq!(db.len(), d_out);
        for r in 0..rows {
            for o in 0..d_out {
                db[o] = db[o] + dy[r * d_out + o];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_known_product() {
        // [1 2; 3 4] @ [5 6; 7 8] = [19 22; 43 50]
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0; 4];
        gemm(Trans::No, Trans::No, 2, 2, 2, 1.0, &a, &b, 0.0, &mut c);
        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);

        // A^T @ B with A stored as its transpose gives the same answer.
        let at = [1.0, 3.0, 2.0, 4.0];
        let mut c2 = [0.0; 4];
        gemm(Trans::Yes, Trans::No, 2, 2, 2, 1.0, &at, &b, 0.0, &mut c2);
        assert_eq!(c2, c);

        // A @ B^T with B stored as its transpose.
        let bt = [5.0, 7.0, 6.0, 8.0];
        let mut c3 = [0.0; 4];
        gemm(Trans::No, Trans::Yes, 2, 2, 2, 1.0, &a, &bt, 0.0, &mut c3);
        assert_eq!(c3, c);

        // beta accumulates.
        gemm(Trans::No, Trans::No, 2, 2, 2, 1.0, &a, &b, 1.0, &mut c3);
        assert_eq!(c3, [38.0, 44.0, 86.0, 100.0]);
    }

    #[test]
    fn non_square() {
        // [1 x 3] @ [3 x 2]
        let a = [1.0f64, 2.0, 3.0];
        let b = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut c = [0.0; 2];
        gemm(Trans::No, Trans::No, 1, 2, 3, 1.0, &a, &b, 0.0, &mut c);
        assert_eq!(c, [4.0, 5.0]);
    }
}
