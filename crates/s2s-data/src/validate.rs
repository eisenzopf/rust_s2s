//! Sample invariants.
//!
//! [`find_issues`] lists every violated invariant (empty means valid);
//! [`check`] turns the list into an [`Error::Invalid`]. The shard writer runs
//! [`check`] on every sample it appends.

use s2s_core::streams::AUDIO_VOCAB;

use crate::{Error, Result, Sample};

/// Report at most this many out-of-range tokens individually.
const MAX_TOKEN_REPORTS: usize = 8;

/// Every violated invariant of `s`, as human-readable messages:
///
/// - layout has at least one text slot and one codebook; `frames > 0`;
///   the grid has `frames * num_streams` entries;
/// - `text_pad < text_vocab`; text-stream tokens `< text_vocab`;
///   audio-stream tokens `< AUDIO_VOCAB` (codes or `AUDIO_PAD`);
/// - injections are non-empty, have tokens `< text_vocab`, frames in
///   `0..=frames`, and are sorted by frame;
/// - words name a text stream and have `start_ms <= end_ms`.
pub fn find_issues(s: &Sample) -> Vec<String> {
    let mut out = Vec::new();
    let l = &s.layout;
    if l.text_slots == 0 || l.codebooks == 0 {
        out.push(format!(
            "layout needs at least one text slot and one codebook (got {} and {})",
            l.text_slots, l.codebooks
        ));
        return out;
    }
    if s.frames == 0 {
        out.push("sample has no frames".into());
    }
    let ns = l.num_streams();
    if Some(s.tokens.len()) != s.frames.checked_mul(ns) {
        out.push(format!(
            "token grid has {} entries, expected {} frames x {ns} streams",
            s.tokens.len(),
            s.frames
        ));
        return out;
    }
    if s.text_pad >= s.text_vocab {
        out.push(format!(
            "text pad id {} is not below the text vocab {}",
            s.text_pad, s.text_vocab
        ));
    }
    let is_text: Vec<bool> = l.kinds().iter().map(|k| k.is_text()).collect();
    let mut bad = 0usize;
    for (i, &tok) in s.tokens.iter().enumerate() {
        let st = i % ns;
        let limit = if is_text[st] {
            s.text_vocab
        } else {
            AUDIO_VOCAB
        };
        if tok >= limit {
            if bad < MAX_TOKEN_REPORTS {
                out.push(format!(
                    "frame {} stream {st} ({:?}): token {tok} >= vocab {limit}",
                    i / ns,
                    l.kind(st)
                ));
            }
            bad += 1;
        }
    }
    if bad > MAX_TOKEN_REPORTS {
        out.push(format!(
            "... {} more out-of-range tokens",
            bad - MAX_TOKEN_REPORTS
        ));
    }
    let mut prev = 0u32;
    for (j, inj) in s.injections.iter().enumerate() {
        if inj.frame as usize > s.frames {
            out.push(format!(
                "injection {j} at frame {} is past the end ({} frames)",
                inj.frame, s.frames
            ));
        }
        if inj.frame < prev {
            out.push(format!(
                "injection {j} at frame {} comes after one at frame {prev} (not sorted)",
                inj.frame
            ));
        }
        prev = prev.max(inj.frame);
        if inj.tokens.is_empty() {
            out.push(format!("injection {j} is empty"));
        }
        if let Some(&t) = inj.tokens.iter().find(|&&t| t >= s.text_vocab) {
            out.push(format!(
                "injection {j}: token {t} >= text vocab {}",
                s.text_vocab
            ));
        }
    }
    for (j, w) in s.words.iter().enumerate() {
        let text_stream = (w.stream as usize) < ns && is_text[w.stream as usize];
        if !text_stream {
            out.push(format!(
                "word {j} ({:?}) names stream {}, which is not a text stream",
                w.text, w.stream
            ));
        }
        if w.start_ms > w.end_ms {
            out.push(format!(
                "word {j} ({:?}) ends ({} ms) before it starts ({} ms)",
                w.text, w.end_ms, w.start_ms
            ));
        }
    }
    out
}

/// `Ok(())` if [`find_issues`] finds nothing, else an [`Error::Invalid`]
/// listing the issues.
pub fn check(s: &Sample) -> Result<()> {
    let issues = find_issues(s);
    if issues.is_empty() {
        Ok(())
    } else {
        Err(Error::Invalid(issues.join("; ")))
    }
}
