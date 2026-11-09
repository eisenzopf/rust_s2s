//! Evaluation metrics for rust_s2s (plan §11). Everything runs on the CPU.
//!
//! - [`normalize`]: the configurable text normalizer applied before ASR
//!   scoring (case, punctuation, whitespace, numbers to words).
//! - [`asr`]: Levenshtein alignment, WER and CER with substitution /
//!   deletion / insertion counts, per utterance and over a corpus (STT on
//!   `Tu`, and TTS intelligibility via an external ASR).
//! - [`tools`]: tool-calling accuracy: tool selection, argument exact match,
//!   per-argument match, JSON validity, and precision/recall of "no call
//!   needed".
//! - [`latency`]: percentiles, turn-taking latency from timestamped turn
//!   events, tool round-trip time.
//! - [`audio`]: SI-SDR and a log-mel spectral distance. These are **proxies**
//!   for codec reconstruction quality (cheap, reference-based signal
//!   measures), not perceptual metrics such as ViSQOL or MOS ratings.
//!
//! The `s2s-eval` binary wraps them: `s2s-eval wer --ref refs.txt --hyp
//! hyps.txt`, `s2s-eval tools --ref ref.jsonl --hyp hyp.jsonl`, `s2s-eval
//! latency --events events.jsonl`, `s2s-eval audio --ref a.wav --est b.wav`.
//!
//! # Example
//!
//! ```
//! use s2s_eval::*;
//! use serde_json::json;
//!
//! // WER with S/D/I counts after normalization ("18" -> "eighteen").
//! let mut asr = AsrScorer::new(Normalizer::default());
//! let u = asr.add("It's 18 degrees in Paris.", "it's eighteen degrees in paris today");
//! assert_eq!((u.words.substitutions, u.words.deletions, u.words.insertions), (0, 0, 1));
//! assert_eq!(asr.summary().wer, Some(0.2));
//!
//! // Tool calling.
//! let mut tools = ToolEvaluator::new();
//! tools.add(&ToolExample {
//!     id: None,
//!     reference: vec![ToolCallRecord::new("get_weather", json!({"city": "Paris"}))],
//!     predicted: vec![PredictedCall::parse_span(
//!         r#"{"name": "get_weather", "arguments": {"city": "Paris"}}"#,
//!     )],
//! });
//! assert_eq!(tools.report().argument_exact_match, Some(1.0));
//!
//! // Latency percentiles.
//! assert_eq!(percentile(&[0.2, 0.3, 0.4], 50.0), Some(0.3));
//! ```

pub mod asr;
pub mod audio;
pub mod latency;
pub mod normalize;
pub mod tools;

pub use asr::{align, AlignedPair, AsrScorer, CorpusResult, EditOp, ErrorCounts, UtteranceResult};
pub use audio::{log_mel_distance, si_sdr, AudioError, LogMelDistance, StftConfig};
pub use latency::{
    percentile, tool_round_trips, turn_taking, LatencyStats, ToolRoundTrips, TurnEvent,
    TurnEventKind, TurnTaking,
};
pub use normalize::{Normalizer, NumberHandling};
pub use tools::{PredictedCall, ToolCallRecord, ToolEvaluator, ToolExample, ToolReport};

/// `num / den`, or `None` when `den` is zero.
pub(crate) fn ratio(num: usize, den: usize) -> Option<f64> {
    (den > 0).then(|| num as f64 / den as f64)
}
