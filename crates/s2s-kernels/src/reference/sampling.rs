//! Token sampling with temperature, top-k, top-p and grammar masks.
//!
//! The random number is passed in (`u` uniform in `[0, 1)`), so sampling is a
//! pure function and GPU and CPU implementations can be compared exactly.

use s2s_core::Float;

/// Sampling settings for one stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParams {
    /// Softmax temperature. `0.0` means greedy (argmax).
    pub temperature: f64,
    /// Keep only the `k` most likely tokens.
    pub top_k: Option<usize>,
    /// Keep the smallest set of most-likely tokens whose probability reaches `p`.
    pub top_p: Option<f64>,
}

impl SamplingParams {
    /// Greedy decoding.
    pub const GREEDY: Self = Self {
        temperature: 0.0,
        top_k: None,
        top_p: None,
    };
}

/// Set logits of tokens not allowed by a bitmask to `-inf`, in place.
///
/// `allowed` is a little-endian bitset of `ceil(v / 32)` words: bit `i % 32`
/// of word `i / 32` set means token `i` is allowed. This is the format
/// grammar engines such as `llguidance` produce for JSON-schema-constrained
/// tool calls (plan §4.2).
pub fn apply_token_bitmask<T: Float>(logits: &mut [T], allowed: &[u32]) {
    assert!(allowed.len() * 32 >= logits.len(), "bitmask too short");
    for (i, l) in logits.iter_mut().enumerate() {
        if allowed[i / 32] >> (i % 32) & 1 == 0 {
            *l = T::neg_infinity();
        }
    }
}

/// Sample one token. Returns `None` if every token is masked out.
///
/// Order of operations: temperature, then top-k, then top-p over the
/// renormalized distribution, then inverse-CDF sampling with `u`. Ties are
/// broken by lower token id, so results are deterministic.
pub fn sample<T: Float>(logits: &[T], params: &SamplingParams, u: f64) -> Option<usize> {
    let vals: Vec<f64> = logits
        .iter()
        .map(|&x| x.to_f64().unwrap_or(f64::NEG_INFINITY))
        .collect();
    let mut order: Vec<usize> = (0..vals.len())
        .filter(|&i| vals[i] > f64::NEG_INFINITY)
        .collect();
    if order.is_empty() {
        return None;
    }
    // Descending by logit, ascending by id on ties.
    order.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]).then(a.cmp(&b)));
    if params.temperature <= 0.0 {
        return Some(order[0]);
    }
    if let Some(k) = params.top_k {
        order.truncate(k.max(1));
    }
    let max = vals[order[0]];
    let mut probs: Vec<f64> = order
        .iter()
        .map(|&i| ((vals[i] - max) / params.temperature).exp())
        .collect();
    let total: f64 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= total);
    if let Some(top_p) = params.top_p {
        let mut cum = 0.0;
        let mut keep = probs.len();
        for (n, p) in probs.iter().enumerate() {
            cum += p;
            if cum >= top_p {
                keep = n + 1;
                break;
            }
        }
        order.truncate(keep);
        probs.truncate(keep);
        let total: f64 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= total);
    }
    let target = u.clamp(0.0, 1.0 - f64::EPSILON);
    let mut cum = 0.0;
    for (n, p) in probs.iter().enumerate() {
        cum += p;
        if target < cum {
            return Some(order[n]);
        }
    }
    order.last().copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_ties() {
        assert_eq!(
            sample(&[1.0f32, 3.0, 3.0, 0.0], &SamplingParams::GREEDY, 0.5),
            Some(1)
        );
    }

    #[test]
    fn all_masked_is_none() {
        let mut l = [1.0f32, 2.0];
        apply_token_bitmask(&mut l, &[0]);
        assert_eq!(sample(&l, &SamplingParams::GREEDY, 0.0), None);
    }

    #[test]
    fn bitmask_restricts_choice() {
        let mut l = [5.0f32, 1.0, 0.0, 4.0];
        apply_token_bitmask(&mut l, &[0b0110]);
        assert_eq!(sample(&l, &SamplingParams::GREEDY, 0.0), Some(1));
    }

    #[test]
    fn top_k_one_is_greedy() {
        let p = SamplingParams {
            temperature: 1.0,
            top_k: Some(1),
            top_p: None,
        };
        for u in [0.0, 0.3, 0.999] {
            assert_eq!(sample(&[0.0f64, 2.0, 1.0], &p, u), Some(1));
        }
    }

    #[test]
    fn inverse_cdf_with_equal_logits() {
        let p = SamplingParams {
            temperature: 1.0,
            top_k: None,
            top_p: None,
        };
        let l = [0.0f64; 4];
        assert_eq!(sample(&l, &p, 0.10), Some(0));
        assert_eq!(sample(&l, &p, 0.30), Some(1));
        assert_eq!(sample(&l, &p, 0.60), Some(2));
        assert_eq!(sample(&l, &p, 0.99), Some(3));
    }

    #[test]
    fn top_p_cuts_tail() {
        // probs ~ [0.665, 0.245, 0.09]; top_p 0.6 keeps only the first.
        let p = SamplingParams {
            temperature: 1.0,
            top_k: None,
            top_p: Some(0.6),
        };
        let l = [2.0f64, 1.0, 0.0];
        for u in [0.0, 0.5, 0.99] {
            assert_eq!(sample(&l, &p, u), Some(0));
        }
    }
}
