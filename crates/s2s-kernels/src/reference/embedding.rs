//! Multi-stream embedding: each frame's input is the **sum** of one embedding
//! per stream slot (plan §3.4). Stream `s` has its own table.

use s2s_core::Float;

/// Token value meaning "this slot contributes nothing" (e.g. the first frame,
/// before any token exists, or a stream that is absent at an injection
/// position).
pub const NO_TOKEN: u32 = u32::MAX;

/// Gather-and-sum forward.
///
/// - `tables[s]`: `[vocab[s], d]`
/// - `tokens`: `[n, streams]`, where `streams == tables.len()`
/// - `out`: `[n, d]` (overwritten)
///
/// # Panics
/// If a token is out of range for its table (and is not [`NO_TOKEN`]).
pub fn embedding_sum_fwd<T: Float>(
    tables: &[&[T]],
    vocab: &[usize],
    tokens: &[u32],
    n: usize,
    d: usize,
    out: &mut [T],
) {
    let streams = tables.len();
    assert_eq!(vocab.len(), streams);
    assert_eq!(tokens.len(), n * streams);
    assert_eq!(out.len(), n * d);
    for (s, t) in tables.iter().enumerate() {
        assert_eq!(t.len(), vocab[s] * d, "table {s} has wrong length");
    }
    out.iter_mut().for_each(|x| *x = T::zero());
    for r in 0..n {
        for s in 0..streams {
            let tok = tokens[r * streams + s];
            if tok == NO_TOKEN {
                continue;
            }
            let tok = tok as usize;
            assert!(
                tok < vocab[s],
                "token {tok} out of range for stream {s} (vocab {})",
                vocab[s]
            );
            let row = &tables[s][tok * d..(tok + 1) * d];
            for x in 0..d {
                out[r * d + x] = out[r * d + x] + row[x];
            }
        }
    }
}

/// Scatter-add backward: `dtables[s][tokens[r, s]] += dout[r]`.
pub fn embedding_sum_bwd<T: Float>(
    dtables: &mut [&mut [T]],
    vocab: &[usize],
    tokens: &[u32],
    n: usize,
    d: usize,
    dout: &[T],
) {
    let streams = dtables.len();
    assert_eq!(vocab.len(), streams);
    assert_eq!(tokens.len(), n * streams);
    assert_eq!(dout.len(), n * d);
    for r in 0..n {
        for s in 0..streams {
            let tok = tokens[r * streams + s];
            if tok == NO_TOKEN {
                continue;
            }
            let tok = tok as usize;
            assert!(tok < vocab[s]);
            let row = &mut dtables[s][tok * d..(tok + 1) * d];
            for x in 0..d {
                row[x] = row[x] + dout[r * d + x];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_streams_and_skips_no_token() {
        let t0 = [1.0f64, 0.0, 0.0, 1.0]; // vocab 2, d 2
        let t1 = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]; // vocab 3
        let tokens = [1, 2, 0, NO_TOKEN];
        let mut out = [0.0; 4];
        embedding_sum_fwd(&[&t0, &t1], &[2, 3], &tokens, 2, 2, &mut out);
        assert_eq!(out, [50.0, 61.0, 1.0, 0.0]);

        let mut d0 = [0.0; 4];
        let mut d1 = [0.0; 6];
        embedding_sum_bwd(
            &mut [&mut d0, &mut d1],
            &[2, 3],
            &tokens,
            2,
            2,
            &[1.0, 2.0, 3.0, 4.0],
        );
        assert_eq!(d0, [3.0, 4.0, 1.0, 2.0]);
        assert_eq!(d1, [0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);
    }
}
