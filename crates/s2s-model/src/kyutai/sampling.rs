//! Token sampling for the Kyutai drivers.
//!
//! Greedy decoding reproduces candle exactly, including its tie-breaking:
//! - `LogitsProcessor` with `Sampling::ArgMax` (text and depformer in the
//!   multistream driver) uses `Iterator::max_by`, which returns the **last**
//!   maximum: [`Sampler::greedy`].
//! - `Tensor::argmax` (the ASR driver at temperature 0) keeps the **first**
//!   maximum: [`Sampler::greedy_first`].
//!
//! Temperature / top-k / top-p sampling uses the reference kernel
//! [`s2s_kernels::reference::sampling::sample`] with a seeded
//! [`SplitMix64`]. It follows the same distribution as Kyutai's sampler but
//! not the same random stream (candle draws from `rand`'s StdRng / Gumbel
//! noise), so it is not bit-identical.

use s2s_core::SplitMix64;
use s2s_kernels::reference::sampling::{sample, SamplingParams};

use super::error::{Error, Result};

/// Which maximum greedy decoding picks on exact ties.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TieBreak {
    /// First maximal index (`Tensor::argmax`).
    First,
    /// Last maximal index (`LogitsProcessor` / `max_by`).
    Last,
}

/// A token sampler.
#[derive(Clone, Debug)]
pub struct Sampler {
    params: SamplingParams,
    rng: SplitMix64,
    ties: TieBreak,
}

impl Sampler {
    /// Greedy with `LogitsProcessor` semantics (last maximum on ties).
    pub fn greedy() -> Self {
        Self {
            params: SamplingParams::GREEDY,
            rng: SplitMix64::new(0),
            ties: TieBreak::Last,
        }
    }

    /// Greedy with `Tensor::argmax` semantics (first maximum on ties).
    pub fn greedy_first() -> Self {
        Self {
            ties: TieBreak::First,
            ..Self::greedy()
        }
    }

    /// Sampling with temperature / top-k / top-p; temperature `<= 0` is greedy
    /// ([`TieBreak::Last`]).
    pub fn new(params: SamplingParams, seed: u64) -> Self {
        Self {
            params,
            rng: SplitMix64::new(seed),
            ties: TieBreak::Last,
        }
    }

    /// Whether this sampler is greedy.
    pub fn is_greedy(&self) -> bool {
        self.params.temperature <= 0.0
    }

    /// Draw a token from `logits`.
    pub fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        if self.is_greedy() {
            return argmax(logits, self.ties)
                .ok_or_else(|| Error::Input("cannot sample from empty logits".into()));
        }
        let u = self.rng.next_f64();
        sample(logits, &self.params, u)
            .map(|t| t as u32)
            .ok_or_else(|| Error::Input("all logits are -inf".into()))
    }
}

/// Index of the maximum (NaN-free inputs), with the given tie-break.
pub fn argmax(logits: &[f32], ties: TieBreak) -> Option<u32> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in logits.iter().enumerate() {
        best = match best {
            None => Some((i, v)),
            Some((_, b)) if v > b || (ties == TieBreak::Last && v == b) => Some((i, v)),
            keep => keep,
        };
    }
    best.map(|(i, _)| i as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tie_breaking() {
        let l = [1.0, 3.0, 2.0, 3.0];
        assert_eq!(argmax(&l, TieBreak::First), Some(1));
        assert_eq!(argmax(&l, TieBreak::Last), Some(3));
        assert_eq!(Sampler::greedy().sample(&l).unwrap(), 3);
        assert_eq!(Sampler::greedy_first().sample(&l).unwrap(), 1);
        assert!(Sampler::greedy().sample(&[]).is_err());
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let p = SamplingParams {
            temperature: 1.0,
            top_k: Some(3),
            top_p: None,
        };
        let l = [0.5f32, 0.4, 0.3, -5.0];
        let draw = |seed| {
            let mut s = Sampler::new(p, seed);
            (0..20).map(|_| s.sample(&l).unwrap()).collect::<Vec<_>>()
        };
        assert_eq!(draw(7), draw(7));
        assert!(draw(7).iter().all(|&t| t < 3));
    }
}
