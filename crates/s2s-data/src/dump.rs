//! Human-readable views of a sample, for debugging (plan Appendix A style).

use std::fmt::Write as _;

use s2s_core::streams::{special, StreamKind, AUDIO_PAD, FRAME_RATE};
use s2s_kernels::reference::delay::apply_delays;

use crate::tokenizer::TextTokenizer;
use crate::Sample;

/// What part of the grid to show, and how.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DumpOptions {
    /// First frame (row) shown.
    pub from: usize,
    /// One past the last frame shown (`None`: to the end).
    pub to: Option<usize>,
    /// Show the delayed grid (as the model sees it) instead of the aligned one.
    pub delayed: bool,
    /// Characters per column.
    pub col_width: usize,
}

impl Default for DumpOptions {
    fn default() -> Self {
        Self {
            from: 0,
            to: None,
            delayed: true,
            col_width: 6,
        }
    }
}

fn short_special(name: &str) -> &str {
    match name {
        special::PAD => "·",
        special::EPAD => "<ep>",
        special::TOOL_CALL => "<tc>",
        special::TOOL_CALL_END => "</tc>",
        special::TOOL_RESULT => "<tr>",
        special::TOOL_RESULT_END => "</tr>",
        special::TOOL_CANCEL => "<cx>",
        special::SYS => "<s>",
        special::SYS_END => "</s>",
        other => other,
    }
}

fn text_cell(id: u32, pad: u32, tok: Option<&dyn TextTokenizer>) -> String {
    if id == pad {
        return "·".into();
    }
    match tok {
        Some(t) => {
            for name in special::ALL {
                if t.special_id(name) == Some(id) {
                    return short_special(name).to_string();
                }
            }
            t.token_str(id)
        }
        None => id.to_string(),
    }
}

fn fit(s: &str, w: usize) -> String {
    let mut out: String = s.chars().take(w.saturating_sub(1)).collect();
    let n = out.chars().count();
    out.extend(std::iter::repeat_n(' ', w - n.min(w)));
    out
}

fn label(sample: &Sample, s: usize, delayed: bool) -> String {
    let l = &sample.layout;
    let base = match l.kind(s) {
        StreamKind::UserText => "Tu".to_string(),
        StreamKind::AssistantText(i) => format!("Tm{i}"),
        StreamKind::AssistantAudio(i) => format!("Am{i}"),
        StreamKind::UserAudio(i) => format!("Au{i}"),
    };
    let d = l.delay(s);
    if delayed && d > 0 {
        format!("{base} (+{d})")
    } else {
        base
    }
}

/// One-paragraph summary: id, source, license, length, layout, counts.
pub fn summary(s: &Sample) -> String {
    let l = &s.layout;
    format!(
        "id={:?} source={:?} license={:?} tokenizer={:?}\n\
         frames={} ({:.2} s) streams={} (text_slots={} codebooks={} d_u={} tau={})\n\
         text_vocab={} text_pad={} injections={} ({} tokens) words={}",
        s.meta.id,
        s.meta.source,
        s.meta.license,
        s.meta.tokenizer,
        s.frames,
        s.frames as f64 / FRAME_RATE,
        s.num_streams(),
        l.text_slots,
        l.codebooks,
        l.user_text_delay,
        l.acoustic_delay,
        s.text_vocab,
        s.text_pad,
        s.injections.len(),
        s.injection_tokens(),
        s.words.len(),
    )
}

/// Render the frame grid: rows are streams, columns are frames. Text tokens
/// are shown decoded (`·` = `<pad>`, `<ep>` = `<epad>`, `<tc>` = `<tool_call>`
/// ...) when a tokenizer is given, else as ids; audio codes are numbers and
/// `∅` is [`AUDIO_PAD`]. The `inject` row shows `+n` where `n` injected
/// tokens precede the frame; the segments are listed below the grid.
pub fn dump_grid(sample: &Sample, tok: Option<&dyn TextTokenizer>, opts: &DumpOptions) -> String {
    let l = &sample.layout;
    let ns = l.num_streams();
    let (grid, rows) = if opts.delayed {
        let d = apply_delays(
            &sample.tokens,
            sample.frames,
            &l.delays(),
            &l.pad_tokens(sample.text_pad),
        );
        (d, sample.frames + l.max_delay())
    } else {
        (sample.tokens.clone(), sample.frames)
    };
    let to = opts.to.unwrap_or(rows).min(rows);
    let from = opts.from.min(to);
    let w = opts.col_width.max(3);
    let lw = 12;

    let mut out = String::new();
    out.push_str(&fit("frame", lw));
    for f in from..to {
        out.push_str(&fit(&f.to_string(), w));
    }
    out.push('\n');
    out.push_str(&fit("inject", lw));
    for f in from..to {
        let n: usize = sample
            .injections
            .iter()
            .filter(|i| i.frame as usize == f)
            .map(|i| i.tokens.len())
            .sum();
        let cell = if n > 0 {
            format!("+{n}")
        } else {
            String::new()
        };
        out.push_str(&fit(&cell, w));
    }
    out.push('\n');
    for s in 0..ns {
        out.push_str(&fit(&label(sample, s, opts.delayed), lw));
        let is_text = l.kind(s).is_text();
        for f in from..to {
            let id = grid[f * ns + s];
            let cell = if is_text {
                text_cell(id, sample.text_pad, tok)
            } else if id == AUDIO_PAD {
                "∅".to_string()
            } else {
                id.to_string()
            };
            out.push_str(&fit(&cell, w));
        }
        out.push('\n');
    }
    for inj in &sample.injections {
        let text = match tok {
            Some(t) => t.decode(&inj.tokens),
            None => format!("{:?}", inj.tokens),
        };
        let text = text.replace('\n', "\\n");
        let mut shown: String = text.chars().take(160).collect();
        if text.chars().count() > 160 {
            shown.push('…');
        }
        let _ = writeln!(
            out,
            "injection before frame {} ({:?}, {} tokens): {shown}",
            inj.frame,
            inj.kind,
            inj.tokens.len()
        );
    }
    out
}
