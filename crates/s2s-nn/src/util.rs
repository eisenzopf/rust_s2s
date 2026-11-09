//! Layout helpers.

/// `[b, t, h, d]` -> `[b, h, t, d]`.
pub(crate) fn bthd_to_bhtd<T: Copy + Default>(
    x: &[T],
    b: usize,
    t: usize,
    h: usize,
    d: usize,
) -> Vec<T> {
    let mut out = vec![T::default(); x.len()];
    for bi in 0..b {
        for ti in 0..t {
            for hi in 0..h {
                let src = ((bi * t + ti) * h + hi) * d;
                let dst = ((bi * h + hi) * t + ti) * d;
                out[dst..dst + d].copy_from_slice(&x[src..src + d]);
            }
        }
    }
    out
}

/// `[b, h, t, d]` -> `[b, t, h, d]`.
pub(crate) fn bhtd_to_bthd<T: Copy + Default>(
    x: &[T],
    b: usize,
    h: usize,
    t: usize,
    d: usize,
) -> Vec<T> {
    let mut out = vec![T::default(); x.len()];
    for bi in 0..b {
        for hi in 0..h {
            for ti in 0..t {
                let src = ((bi * h + hi) * t + ti) * d;
                let dst = ((bi * t + ti) * h + hi) * d;
                out[dst..dst + d].copy_from_slice(&x[src..src + d]);
            }
        }
    }
    out
}

/// `a += b`, element-wise.
pub(crate) fn add_assign<T: s2s_core::Float>(a: &mut [T], b: &[T]) {
    assert_eq!(a.len(), b.len());
    for (x, &y) in a.iter_mut().zip(b) {
        *x = *x + y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_round_trip() {
        let x: Vec<u32> = (0..2 * 3 * 4 * 5).collect();
        let y = bthd_to_bhtd(&x, 2, 3, 4, 5);
        assert_eq!(bhtd_to_bthd(&y, 2, 4, 3, 5), x);
        // element [b=1, t=2, h=3, d=4] moves to [b=1, h=3, t=2, d=4]
        assert_eq!(y[((4 + 3) * 3 + 2) * 5 + 4], x[((3 + 2) * 4 + 3) * 5 + 4]);
    }
}
