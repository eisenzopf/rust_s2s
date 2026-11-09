//! A tiny character-level corpus with a seeded, resumable batch sampler.

use s2s_core::SplitMix64;
use serde::{Deserialize, Serialize};

/// Where the sampler is; saved in checkpoints for exact resume.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataCursor {
    /// RNG state of the sampler.
    pub rng_state: u64,
    /// Batches drawn so far.
    pub batches: u64,
}

/// A short built-in English text for CPU bring-up runs and tests.
pub const DEMO_TEXT: &str = include_str!("corpus.txt");

/// Character-level corpus: each distinct character is one token.
#[derive(Clone, Debug)]
pub struct CharCorpus {
    vocab: Vec<char>,
    ids: Vec<u32>,
    rng: SplitMix64,
    batches: u64,
}

impl CharCorpus {
    /// Build from text with a sampler seed.
    pub fn new(text: &str, seed: u64) -> Self {
        let mut vocab: Vec<char> = text.chars().collect();
        vocab.sort_unstable();
        vocab.dedup();
        let ids = text
            .chars()
            .map(|c| vocab.binary_search(&c).expect("char in vocab") as u32)
            .collect();
        Self {
            vocab,
            ids,
            rng: SplitMix64::new(seed),
            batches: 0,
        }
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Token ids → text.
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter().map(|&i| self.vocab[i as usize]).collect()
    }

    /// Text → token ids (unknown characters are skipped).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .filter_map(|c| self.vocab.binary_search(&c).ok().map(|i| i as u32))
            .collect()
    }

    /// A batch of random windows: `(tokens [batch, seq], targets [batch, seq])`
    /// where targets are the next characters.
    pub fn batch(&mut self, batch: usize, seq: usize) -> (Vec<u32>, Vec<i64>) {
        assert!(self.ids.len() > seq + 1, "corpus shorter than a window");
        let mut tokens = Vec::with_capacity(batch * seq);
        let mut targets = Vec::with_capacity(batch * seq);
        for _ in 0..batch {
            let start = self.rng.below((self.ids.len() - seq - 1) as u64) as usize;
            tokens.extend_from_slice(&self.ids[start..start + seq]);
            targets.extend(
                self.ids[start + 1..start + seq + 1]
                    .iter()
                    .map(|&t| t as i64),
            );
        }
        self.batches += 1;
        (tokens, targets)
    }

    /// Current position, for checkpointing.
    pub fn cursor(&self) -> DataCursor {
        DataCursor {
            rng_state: self.rng.state(),
            batches: self.batches,
        }
    }

    /// Restore a position saved with [`cursor`](Self::cursor).
    pub fn restore(&mut self, c: DataCursor) {
        self.rng = SplitMix64::new(c.rng_state);
        self.batches = c.batches;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_and_resume() {
        let mut c = CharCorpus::new("hello world, hello rust", 7);
        assert_eq!(c.decode(&c.encode("hello")), "hello");
        let _ = c.batch(2, 4);
        let saved = c.cursor();
        let next = c.batch(2, 4);
        c.restore(saved);
        assert_eq!(c.batch(2, 4), next);
        let (t, y) = next;
        assert_eq!(t.len(), 8);
        // targets are tokens shifted by one within each window
        assert_eq!(t[1] as i64, y[0]);
    }
}
