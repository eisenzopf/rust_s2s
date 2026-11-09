//! Training-example assembly: delays, injection interleaving, shifted
//! targets and per-target loss weights.
//!
//! # Position sequence
//!
//! The Temporal Transformer consumes one **position** per step. From a sample
//! with `F` aligned frames and layout delays `d_s` (max `D`):
//!
//! 1. The grid is delayed with [`apply_delays`] and
//!    [`StreamLayout::pad_tokens`], giving `R = F + D` rows. Row `r` holds
//!    stream `s`'s token for frame `r - d_s`, or the stream's pad token where
//!    that frame does not exist (**delay padding**).
//! 2. If [`AssemblyConfig::start_frame`] is set (the default), position 0 is a
//!    virtual "frame −1" in which every stream holds its pad token (text
//!    `<pad>`, [`AUDIO_PAD`](s2s_core::streams::AUDIO_PAD)). It gives frame 0 a predecessor, so frame 0 is
//!    trained when no injection precedes it.
//! 3. For each row `r = 0..R`: first every injection segment whose `frame ==
//!    r` (in sample order), one position per token, then the row itself.
//!    Segments with `frame >= R` (only possible when `D == 0`) go last.
//!    Injection frames index rows of the delayed grid, which coincide with
//!    aligned frames for the delay-0 streams (`Tm`, `Am[0]`, `Au[0]`).
//! 4. An **injection position** carries its text token in
//!    [`AssemblyConfig::injection_stream`] (default `Tm` slot 0); every other
//!    text stream holds `<pad>` and every audio stream holds [`AUDIO_PAD`](s2s_core::streams::AUDIO_PAD).
//!    [`Example::is_injection`] marks it, so the model can add `e_inject` and
//!    use its `<no_audio>` embedding (plan §3.5).
//!
//! # Targets and weights
//!
//! Targets cover only the `K = layout.num_generated()` generated streams
//! (`Tu`, `Tm` slots, `Am` codebooks) and are **shifted by one position**:
//! `targets[p, k]` is stream `k`'s token at position `p + 1`, because the
//! output at position `p` predicts the next frame. Consequently:
//!
//! - if position `p + 1` is an injection position, `targets[p, ·]` is
//!   [`IGNORE_INDEX`] with weight 0 (injected text is never predicted);
//! - the last position of a segment run, whose next position is a frame, is
//!   trained normally: at inference the runtime prefills the segment and then
//!   samples the next frame from the last injected position's output;
//! - if `(p + 1, k)` is delay padding, the target is [`IGNORE_INDEX`] with
//!   weight 0; the runtime forces pads there;
//! - the last position has no successor: [`IGNORE_INDEX`], weight 0.
//!
//! Otherwise the weight is [`LossWeights`] for the stream kind, multiplied by
//! [`LossWeights::text_pad_factor`] when a text stream's target is `<pad>`.

use s2s_core::streams::{StreamKind, StreamLayout};
use s2s_kernels::reference::delay::{apply_delays, remove_delays};
use serde::{Deserialize, Serialize};

pub use s2s_kernels::reference::loss::IGNORE_INDEX;

use crate::{Error, Result, Sample};

/// Per-target loss weights by stream kind.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LossWeights {
    /// `Tu` (user transcript).
    pub user_text: f32,
    /// `Tm` slots (inner monologue, tool calls).
    pub assistant_text: f32,
    /// `Am[0]`, the semantic codebook.
    pub semantic: f32,
    /// `Am[1..]`, the acoustic codebooks.
    pub acoustic: f32,
    /// Multiplier for text targets equal to `<pad>` (plan §3.4: e.g. 0.5).
    pub text_pad_factor: f32,
}

impl Default for LossWeights {
    /// Text 1, semantic 1, acoustic 0.01 (Moshi's 100:1 semantic:acoustic
    /// ratio with text and semantic tokens on the same scale), `<pad>` 0.5.
    fn default() -> Self {
        Self {
            user_text: 1.0,
            assistant_text: 1.0,
            semantic: 1.0,
            acoustic: 0.01,
            text_pad_factor: 0.5,
        }
    }
}

impl LossWeights {
    /// Base weight for a stream kind (user audio, never a target, gets 0).
    pub fn for_kind(&self, kind: StreamKind) -> f32 {
        match kind {
            StreamKind::UserText => self.user_text,
            StreamKind::AssistantText(_) => self.assistant_text,
            StreamKind::AssistantAudio(0) => self.semantic,
            StreamKind::AssistantAudio(_) => self.acoustic,
            StreamKind::UserAudio(_) => 0.0,
        }
    }
}

/// How to assemble examples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AssemblyConfig {
    /// Model layout. Its `text_slots` and `codebooks` must match the sample;
    /// its delays are the ones applied (they may differ from the sample's).
    pub layout: StreamLayout,
    /// Id of the text `<pad>` token; must match the sample's.
    pub text_pad: u32,
    /// Loss weights.
    pub weights: LossWeights,
    /// Prepend a virtual all-pad "frame −1" position.
    pub start_frame: bool,
    /// The text stream that carries injected tokens.
    pub injection_stream: StreamKind,
}

impl AssemblyConfig {
    /// Defaults for a layout and pad id: default weights, start frame on,
    /// injections on `Tm` slot 0.
    pub fn new(layout: StreamLayout, text_pad: u32) -> Self {
        Self {
            layout,
            text_pad,
            weights: LossWeights::default(),
            start_frame: true,
            injection_stream: StreamKind::AssistantText(0),
        }
    }

    /// [`new`](Self::new) with the sample's own layout and pad id.
    pub fn for_sample(s: &Sample) -> Self {
        Self::new(s.layout, s.text_pad)
    }
}

/// One assembled training sequence of `positions` positions.
#[derive(Clone, Debug, PartialEq)]
pub struct Example {
    /// Streams per position (`S`).
    pub num_streams: usize,
    /// Generated streams (`K`), the width of `targets` and `weights`.
    pub num_generated: usize,
    /// Number of positions (`P`).
    pub positions: usize,
    /// Input tokens `[P, S]`.
    pub tokens: Vec<u32>,
    /// Targets `[P, K]` ([`IGNORE_INDEX`] = no loss).
    pub targets: Vec<i64>,
    /// Loss weights `[P, K]` (0 wherever the target is ignored).
    pub weights: Vec<f32>,
    /// `[P]`: 1 at injection positions.
    pub is_injection: Vec<u8>,
    /// `[P]`: delayed-grid row of a frame position, −1 for injection and
    /// start positions.
    pub frame_index: Vec<i32>,
}

impl Example {
    /// The delayed grid `[R, S]` recovered from the frame positions.
    pub fn frame_rows(&self) -> Vec<u32> {
        let s = self.num_streams;
        let mut out = Vec::new();
        for p in 0..self.positions {
            if self.frame_index[p] >= 0 {
                out.extend_from_slice(&self.tokens[p * s..(p + 1) * s]);
            }
        }
        out
    }

    /// The aligned grid `[frames, S]`, undoing the delays (for checks).
    pub fn aligned_frames(&self, layout: &StreamLayout, frames: usize) -> Vec<u32> {
        remove_delays(&self.frame_rows(), frames, &layout.delays())
    }
}

enum Pos {
    Start,
    Inject(u32),
    Row(usize),
}

/// Assemble a training example from a sample (see the [module docs](self)).
pub fn assemble(sample: &Sample, cfg: &AssemblyConfig) -> Result<Example> {
    let l = cfg.layout;
    if l.text_slots != sample.layout.text_slots || l.codebooks != sample.layout.codebooks {
        return Err(Error::Invalid(format!(
            "sample layout ({} text slots, {} codebooks) does not match the model ({}, {})",
            sample.layout.text_slots, sample.layout.codebooks, l.text_slots, l.codebooks
        )));
    }
    if cfg.text_pad != sample.text_pad {
        return Err(Error::Invalid(format!(
            "sample text pad id {} differs from the configured {}",
            sample.text_pad, cfg.text_pad
        )));
    }
    let ns = l.num_streams();
    let k = l.num_generated();
    let frames = sample.frames;
    if frames == 0 || sample.tokens.len() != frames * ns {
        return Err(Error::Invalid(format!(
            "token grid has {} entries for {frames} frames x {ns} streams",
            sample.tokens.len()
        )));
    }
    if sample
        .injections
        .windows(2)
        .any(|w| w[0].frame > w[1].frame)
        || sample.injections.iter().any(|i| i.frame as usize > frames)
    {
        return Err(Error::Invalid(
            "injections must be sorted with frames in 0..=frames".into(),
        ));
    }
    let inj_slot = match cfg.injection_stream {
        k @ (StreamKind::UserText | StreamKind::AssistantText(_)) => l.index(k),
        other => {
            return Err(Error::Invalid(format!(
                "injection stream must be a text stream, got {other:?}"
            )))
        }
    };

    let delays = l.delays();
    let pad = l.pad_tokens(cfg.text_pad);
    let rows = frames + l.max_delay();
    let delayed = apply_delays(&sample.tokens, frames, &delays, &pad);

    // Position order.
    let mut order = Vec::with_capacity(rows + sample.injection_tokens() + 1);
    if cfg.start_frame {
        order.push(Pos::Start);
    }
    let mut inj = sample.injections.iter().peekable();
    for r in 0..rows {
        while let Some(seg) = inj.next_if(|s| s.frame as usize == r) {
            order.extend(seg.tokens.iter().map(|&t| Pos::Inject(t)));
        }
        order.push(Pos::Row(r));
    }
    for seg in inj {
        order.extend(seg.tokens.iter().map(|&t| Pos::Inject(t)));
    }

    let p_len = order.len();
    let mut tokens = Vec::with_capacity(p_len * ns);
    let mut is_injection = Vec::with_capacity(p_len);
    let mut frame_index = Vec::with_capacity(p_len);
    for pos in &order {
        match *pos {
            Pos::Start => {
                tokens.extend_from_slice(&pad);
                is_injection.push(0);
                frame_index.push(-1);
            }
            Pos::Inject(t) => {
                let start = tokens.len();
                tokens.extend_from_slice(&pad);
                tokens[start + inj_slot] = t;
                is_injection.push(1);
                frame_index.push(-1);
            }
            Pos::Row(r) => {
                tokens.extend_from_slice(&delayed[r * ns..(r + 1) * ns]);
                is_injection.push(0);
                frame_index.push(r as i32);
            }
        }
    }

    let kinds: Vec<StreamKind> = (0..k).map(|i| l.kind(i)).collect();
    let mut targets = vec![IGNORE_INDEX; p_len * k];
    let mut weights = vec![0.0f32; p_len * k];
    for p in 0..p_len.saturating_sub(1) {
        let Pos::Row(r) = order[p + 1] else { continue };
        for s in 0..k {
            let real = r >= delays[s] && r - delays[s] < frames;
            if !real {
                continue;
            }
            let t = delayed[r * ns + s];
            let mut w = cfg.weights.for_kind(kinds[s]);
            if kinds[s].is_text() && t == cfg.text_pad {
                w *= cfg.weights.text_pad_factor;
            }
            targets[p * k + s] = t as i64;
            weights[p * k + s] = w;
        }
    }

    Ok(Example {
        num_streams: ns,
        num_generated: k,
        positions: p_len,
        tokens,
        targets,
        weights,
        is_injection,
        frame_index,
    })
}
