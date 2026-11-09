//! Token embeddings: one table, or several summed (multi-stream, plan §3.4).

use s2s_core::{Float, SplitMix64};
use s2s_kernels::reference::embedding as emb;

use crate::{Module, Param};

/// Sum of per-stream embeddings: `out[r] = sum_s table_s[tokens[r, s]]`.
///
/// With one stream this is an ordinary embedding table. Tokens equal to
/// [`emb::NO_TOKEN`] contribute nothing.
#[derive(Clone, Debug)]
pub struct MultiEmbedding<T> {
    /// One table per stream, `[vocab_s, d]`.
    pub tables: Vec<Param<T>>,
    vocab: Vec<usize>,
    d: usize,
}

/// Saved tokens for the backward pass.
#[derive(Clone, Debug)]
pub struct EmbeddingCtx {
    tokens: Vec<u32>,
    rows: usize,
}

impl<T: Float> MultiEmbedding<T> {
    /// New embedding with one table per entry of `vocab`, init `N(0, std^2)`.
    pub fn new(name: &str, vocab: &[usize], d: usize, std: f64, rng: &mut SplitMix64) -> Self {
        let tables = vocab
            .iter()
            .enumerate()
            .map(|(s, &v)| {
                let n = if vocab.len() == 1 {
                    format!("{name}.table")
                } else {
                    format!("{name}.table{s}")
                };
                Param::normal(n, &[v, d], std, rng)
            })
            .collect();
        Self {
            tables,
            vocab: vocab.to_vec(),
            d,
        }
    }

    /// Embedding width.
    pub fn d(&self) -> usize {
        self.d
    }

    /// Forward. `tokens` is `[rows, streams]`; returns `[rows, d]`.
    pub fn forward(&self, tokens: &[u32], rows: usize) -> (Vec<T>, EmbeddingCtx) {
        let refs: Vec<&[T]> = self.tables.iter().map(|t| t.value.as_slice()).collect();
        let mut out = vec![T::zero(); rows * self.d];
        emb::embedding_sum_fwd(&refs, &self.vocab, tokens, rows, self.d, &mut out);
        (
            out,
            EmbeddingCtx {
                tokens: tokens.to_vec(),
                rows,
            },
        )
    }

    /// Backward: scatter-adds into the table gradients. Embeddings are the
    /// first layer, so there is no input gradient to return.
    pub fn backward(&mut self, ctx: EmbeddingCtx, dout: &[T]) {
        let mut grads: Vec<&mut [T]> = self
            .tables
            .iter_mut()
            .map(|t| t.grad.as_mut_slice())
            .collect();
        emb::embedding_sum_bwd(&mut grads, &self.vocab, &ctx.tokens, ctx.rows, self.d, dout);
    }
}

impl<T: Float> Module<T> for MultiEmbedding<T> {
    fn visit_params(&self, f: &mut dyn FnMut(&Param<T>)) {
        self.tables.iter().for_each(f);
    }
    fn visit_params_mut(&mut self, f: &mut dyn FnMut(&mut Param<T>)) {
        self.tables.iter_mut().for_each(f);
    }
}
