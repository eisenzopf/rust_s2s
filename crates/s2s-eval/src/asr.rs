//! Word and character error rates via Levenshtein alignment.
//!
//! `WER = (S + D + I) / N` where `N` is the number of reference words, `S`
//! substitutions, `D` deletions (reference words missing from the
//! hypothesis) and `I` insertions (extra hypothesis words), counted on a
//! minimum-edit-distance alignment with unit costs. CER is the same over
//! characters.
//!
//! When several alignments have the minimum cost, the backtrace (from the
//! end of both sequences) prefers the diagonal (match or substitution), then
//! a deletion, then an insertion, so the S/D/I split is deterministic.
//!
//! Both texts are normalized first ([`Normalizer`]). CER is computed on the
//! normalized text with words joined by single spaces; spaces count as
//! characters unless [`AsrScorer::cer_spaces`] is turned off.
//!
//! Corpus aggregates sum the counts over utterances (`ΣE / ΣN`, the usual
//! "corpus WER"), and also report the unweighted mean of per-utterance rates
//! and the sentence error rate (share of utterances with any word error).

use serde::Serialize;

use crate::normalize::Normalizer;
use crate::ratio;

/// One step of an alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EditOp {
    /// Reference and hypothesis items are equal.
    Match,
    /// The hypothesis item replaces a different reference item.
    Substitute,
    /// The reference item is missing from the hypothesis.
    Delete,
    /// The hypothesis item has no reference counterpart.
    Insert,
}

/// One aligned position: indices into the reference and hypothesis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct AlignedPair {
    /// What happened at this position.
    pub op: EditOp,
    /// Reference index (`None` for an insertion).
    pub reference: Option<usize>,
    /// Hypothesis index (`None` for a deletion).
    pub hypothesis: Option<usize>,
}

/// Minimum-edit-distance alignment of `reference` and `hypothesis` with
/// unit costs, in order (see the module docs for tie-breaking).
pub fn align<T: PartialEq>(reference: &[T], hypothesis: &[T]) -> Vec<AlignedPair> {
    let (n, m) = (reference.len(), hypothesis.len());
    let w = m + 1;
    // dp[i * w + j] = distance between reference[..i] and hypothesis[..j].
    let mut dp = vec![0u32; (n + 1) * w];
    for j in 0..=m {
        dp[j] = j as u32;
    }
    for i in 1..=n {
        dp[i * w] = i as u32;
        for j in 1..=m {
            let sub = dp[(i - 1) * w + j - 1] + u32::from(reference[i - 1] != hypothesis[j - 1]);
            let del = dp[(i - 1) * w + j] + 1;
            let ins = dp[i * w + j - 1] + 1;
            dp[i * w + j] = sub.min(del).min(ins);
        }
    }
    let mut out = Vec::with_capacity(n.max(m));
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        let here = dp[i * w + j];
        if i > 0 && j > 0 {
            let same = reference[i - 1] == hypothesis[j - 1];
            if dp[(i - 1) * w + j - 1] + u32::from(!same) == here {
                out.push(AlignedPair {
                    op: if same {
                        EditOp::Match
                    } else {
                        EditOp::Substitute
                    },
                    reference: Some(i - 1),
                    hypothesis: Some(j - 1),
                });
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && dp[(i - 1) * w + j] + 1 == here {
            out.push(AlignedPair {
                op: EditOp::Delete,
                reference: Some(i - 1),
                hypothesis: None,
            });
            i -= 1;
        } else {
            out.push(AlignedPair {
                op: EditOp::Insert,
                reference: None,
                hypothesis: Some(j - 1),
            });
            j -= 1;
        }
    }
    out.reverse();
    out
}

/// Hit / substitution / deletion / insertion counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ErrorCounts {
    /// Matching items.
    pub hits: usize,
    /// Substitutions.
    pub substitutions: usize,
    /// Deletions.
    pub deletions: usize,
    /// Insertions.
    pub insertions: usize,
}

impl ErrorCounts {
    /// Counts of an alignment.
    pub fn from_alignment(alignment: &[AlignedPair]) -> Self {
        let mut c = Self::default();
        for p in alignment {
            match p.op {
                EditOp::Match => c.hits += 1,
                EditOp::Substitute => c.substitutions += 1,
                EditOp::Delete => c.deletions += 1,
                EditOp::Insert => c.insertions += 1,
            }
        }
        c
    }

    /// Align and count.
    pub fn of<T: PartialEq>(reference: &[T], hypothesis: &[T]) -> Self {
        Self::from_alignment(&align(reference, hypothesis))
    }

    /// `S + D + I`.
    pub fn errors(&self) -> usize {
        self.substitutions + self.deletions + self.insertions
    }

    /// Reference length `N = H + S + D`.
    pub fn reference_len(&self) -> usize {
        self.hits + self.substitutions + self.deletions
    }

    /// Hypothesis length `H + S + I`.
    pub fn hypothesis_len(&self) -> usize {
        self.hits + self.substitutions + self.insertions
    }

    /// `(S + D + I) / N`; `None` for an empty reference (the rate is
    /// undefined; any insertion would make it infinite).
    pub fn rate(&self) -> Option<f64> {
        ratio(self.errors(), self.reference_len())
    }
}

impl std::ops::Add for ErrorCounts {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self {
            hits: self.hits + o.hits,
            substitutions: self.substitutions + o.substitutions,
            deletions: self.deletions + o.deletions,
            insertions: self.insertions + o.insertions,
        }
    }
}

impl std::ops::AddAssign for ErrorCounts {
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

/// Scores of one utterance.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct UtteranceResult {
    /// Normalized reference.
    pub reference: String,
    /// Normalized hypothesis.
    pub hypothesis: String,
    /// Word-level counts.
    pub words: ErrorCounts,
    /// Character-level counts.
    pub chars: ErrorCounts,
    /// Word error rate (`None` for an empty reference).
    pub wer: Option<f64>,
    /// Character error rate (`None` for an empty reference).
    pub cer: Option<f64>,
}

/// Corpus-level aggregates.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CorpusResult {
    /// Number of utterances.
    pub utterances: usize,
    /// Summed word counts.
    pub words: ErrorCounts,
    /// Summed character counts.
    pub chars: ErrorCounts,
    /// Corpus WER `ΣE / ΣN` (`None` if every reference is empty).
    pub wer: Option<f64>,
    /// Corpus CER.
    pub cer: Option<f64>,
    /// Mean of the per-utterance WERs that are defined.
    pub mean_utterance_wer: Option<f64>,
    /// Share of utterances with at least one word error.
    pub sentence_error_rate: Option<f64>,
}

/// Accumulates WER/CER over a corpus.
#[derive(Clone, Debug, Default)]
pub struct AsrScorer {
    /// Normalizer applied to both texts.
    pub normalizer: Normalizer,
    /// Count spaces between words as characters for CER (default `true`).
    pub cer_spaces: bool,
    utterances: Vec<UtteranceResult>,
}

impl AsrScorer {
    /// A scorer with `normalizer`, counting spaces in CER.
    pub fn new(normalizer: Normalizer) -> Self {
        Self {
            normalizer,
            cer_spaces: true,
            utterances: Vec::new(),
        }
    }

    /// Score one utterance without recording it.
    pub fn score(&self, reference: &str, hypothesis: &str) -> UtteranceResult {
        let r = self.normalizer.normalize(reference);
        let h = self.normalizer.normalize(hypothesis);
        let rw: Vec<&str> = r.split(' ').filter(|w| !w.is_empty()).collect();
        let hw: Vec<&str> = h.split(' ').filter(|w| !w.is_empty()).collect();
        let words = ErrorCounts::of(&rw, &hw);
        let chars_of =
            |s: &str| -> Vec<char> { s.chars().filter(|&c| self.cer_spaces || c != ' ').collect() };
        let chars = ErrorCounts::of(&chars_of(&r), &chars_of(&h));
        UtteranceResult {
            wer: words.rate(),
            cer: chars.rate(),
            reference: r,
            hypothesis: h,
            words,
            chars,
        }
    }

    /// Score and record one utterance.
    pub fn add(&mut self, reference: &str, hypothesis: &str) -> &UtteranceResult {
        let u = self.score(reference, hypothesis);
        self.utterances.push(u);
        self.utterances.last().expect("just pushed")
    }

    /// Recorded utterances, in order.
    pub fn utterances(&self) -> &[UtteranceResult] {
        &self.utterances
    }

    /// Corpus aggregates of the recorded utterances.
    pub fn summary(&self) -> CorpusResult {
        let mut words = ErrorCounts::default();
        let mut chars = ErrorCounts::default();
        let mut rates = Vec::new();
        let mut with_errors = 0;
        for u in &self.utterances {
            words += u.words;
            chars += u.chars;
            rates.extend(u.wer);
            with_errors += usize::from(u.words.errors() > 0);
        }
        CorpusResult {
            utterances: self.utterances.len(),
            words,
            chars,
            wer: words.rate(),
            cer: chars.rate(),
            mean_utterance_wer: (!rates.is_empty())
                .then(|| rates.iter().sum::<f64>() / rates.len() as f64),
            sentence_error_rate: ratio(with_errors, self.utterances.len()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(s: &str) -> Vec<&str> {
        s.split_whitespace().collect()
    }

    #[test]
    fn alignment_ops_and_tie_breaking() {
        use EditOp::*;
        let a = align(&w("the cat sat"), &w("the bat sat down"));
        let ops: Vec<EditOp> = a.iter().map(|p| p.op).collect();
        assert_eq!(ops, [Match, Substitute, Match, Insert]);
        assert_eq!(a[3].reference, None);
        assert_eq!(a[3].hypothesis, Some(3));

        // "a b" vs "b c": two substitutions and D+M+I both cost 2; the
        // diagonal is preferred.
        let ops: Vec<EditOp> = align(&w("a b"), &w("b c")).iter().map(|p| p.op).collect();
        assert_eq!(ops, [Substitute, Substitute]);

        let ops: Vec<EditOp> = align(&w("a b c"), &w("a c")).iter().map(|p| p.op).collect();
        assert_eq!(ops, [Match, Delete, Match]);
        assert!(align::<&str>(&[], &[]).is_empty());
        let ops: Vec<EditOp> = align(&[], &w("x y")).iter().map(|p| p.op).collect();
        assert_eq!(ops, [Insert, Insert]);
        let ops: Vec<EditOp> = align(&w("x y"), &[]).iter().map(|p| p.op).collect();
        assert_eq!(ops, [Delete, Delete]);
    }

    #[test]
    fn hand_computed_wer() {
        // ref: what is the weather in paris (6 words)
        // hyp: what's the weather in in paris today
        // Alignment: S(what->what's) D(is) M M M I(in) M I(today):
        // minimum cost 4 = 1 S + 1 D + 2 I.
        let c = ErrorCounts::of(
            &w("what is the weather in paris"),
            &w("what's the weather in in paris today"),
        );
        assert_eq!(c.reference_len(), 6);
        assert_eq!(c.hypothesis_len(), 7);
        assert_eq!(c.errors(), 4);
        assert_eq!(
            (c.substitutions, c.deletions, c.insertions, c.hits),
            (1, 1, 2, 4)
        );
        assert!((c.rate().unwrap() - 4.0 / 6.0).abs() < 1e-12);
        assert_eq!(ErrorCounts::of::<&str>(&[], &[]).rate(), None);
        assert_eq!(ErrorCounts::of(&w("a"), &w("a")).rate(), Some(0.0));
        // WER can exceed 1.
        assert_eq!(ErrorCounts::of(&w("a"), &w("b c d")).rate(), Some(3.0));
    }

    #[test]
    fn cer_and_corpus() {
        let mut s = AsrScorer::new(Normalizer::default());
        // "kitten" -> "sitting": 2 S + 1 I = 3 over 6 characters.
        let u = s.add("Kitten", "sitting").clone();
        assert_eq!(u.reference, "kitten");
        assert_eq!(
            (u.chars.substitutions, u.chars.deletions, u.chars.insertions),
            (2, 0, 1)
        );
        assert_eq!(u.cer, Some(0.5));
        assert_eq!(u.wer, Some(1.0));
        // Normalization makes these identical: 0 errors over 4 words.
        let u = s.add("It's 18 degrees!", "its 18 degrees").clone();
        assert_eq!(u.words.substitutions, 1, "it's vs its differ");
        let u = s.add("Hello, World.", "hello world").clone();
        assert_eq!(u.wer, Some(0.0));
        assert_eq!(u.cer, Some(0.0));
        // An empty reference with insertions.
        let u = s.add("", "uh").clone();
        assert_eq!(u.wer, None);
        assert_eq!(u.words.insertions, 1);

        let c = s.summary();
        assert_eq!(c.utterances, 4);
        // Words: kitten/sitting (1 S, N=1), its/it's (1 S, N=3), hello world
        // (N=2), "" vs uh (1 I, N=0): 3 errors over 6 words.
        assert_eq!(c.words.reference_len(), 6);
        assert_eq!(c.words.errors(), 3);
        assert_eq!(c.wer, Some(0.5));
        // Mean of the defined rates: (1 + 1/3 + 0) / 3.
        assert!((c.mean_utterance_wer.unwrap() - (4.0 / 3.0) / 3.0).abs() < 1e-12);
        assert_eq!(c.sentence_error_rate, Some(0.75));
        // Characters: 3 (kitten) + 1 (the apostrophe deleted) + 0 + 2 (`uh`
        // inserted) errors, over the normalized reference lengths.
        let n_chars = "kitten".len() + "it's eighteen degrees".len() + "hello world".len();
        assert_eq!(c.chars.reference_len(), n_chars);
        assert_eq!(c.chars.errors(), 3 + 1 + 2);

        // Without spaces, CER ignores them.
        let mut s = AsrScorer::new(Normalizer::default());
        s.cer_spaces = false;
        let u = s.add("a b", "ab");
        assert_eq!(u.cer, Some(0.0));
        assert_eq!(u.wer, Some(1.0));
    }
}
