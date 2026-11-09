//! Synthetic tool-dialogue timeline assembly (plan §9.2 step 4).
//!
//! A [`Script`] (serde-loadable from JSON) describes a two-party dialogue:
//! turns with speaker, text, optional word timings, optional pre-computed
//! Mimi codes, tool calls with results and latencies, interruptions,
//! overlaps and backchannels. [`synthesize`] places it on a frame timeline
//! and produces a [`Sample`] in the shard format:
//!
//! - **Audio.** User turns go on `Au`, assistant turns on `Am`. A turn's codes
//!   are taken from [`Turn::codes`] (one `[codebooks]` row per frame, e.g. from
//!   Mimi-encoding TTS output) or, without TTS, generated as deterministic
//!   dummy codes from the script seed. Frames where a speaker is silent hold
//!   [`Script::silence_codes`] (default: a fixed dummy code per codebook).
//! - **User transcript (`Tu`).** Each word's tokens start at the word's start
//!   frame, one token per frame (the `d_u` delay is applied later, at
//!   assembly). Optionally preceded by `<epad>` ([`SynthConfig::user_epad`]).
//! - **Inner monologue (`Tm`).** Each assistant word is preceded by `<epad>`
//!   one frame before it, and its tokens start at the word's audio start frame
//!   (text leads audio once the acoustic delay is applied), using the first
//!   [`SynthConfig::monologue_slots`] slots of each frame. Words that would
//!   collide are pushed later; a warning is reported when text lags its audio
//!   by more than two frames.
//! - **Tool calls.** `<tool_call>{"name":…,"arguments":…}</tool_call>` is
//!   packed in reading order (frame by frame, slots `0..text_slots`), starting
//!   right after the assistant turn's own words (the spoken filler, whose audio
//!   continues meanwhile) or at [`ToolCallSpec::start`].
//! - **Tool results.** `<tool_result>{json}</tool_result>` becomes an
//!   [`Injection`] at frame `call_end + 1 + round(latency × 12.5)`, where
//!   `call_end` is the frame holding `</tool_call>`: the runtime dispatches
//!   when `</tool_call>` is sampled and injects the result before the first
//!   frame after it arrives.
//! - **Cancellation.** With [`ToolCallSpec::cancel_after`], `<tool_cancel>` is
//!   written to `Tm` at `call_end + 1 + round(cancel_after × 12.5)` (or the
//!   next free slot) and the result is dropped.
//! - **System prompt.** [`Script::system`] and [`Script::tools`] become a
//!   `<sys>…</sys>` injection at frame 0.
//!
//! Text is tokenized word by word with a leading space on every word except
//! the first on each stream, so the decoded stream reads naturally.

use s2s_core::streams::{StreamKind, StreamLayout, CODEBOOK_SIZE, FRAME_RATE};
use serde::{Deserialize, Serialize};

use crate::sample::{Injection, InjectionKind, Sample, SampleMeta, SpeakerInfo, WordTiming};
use crate::sampler::mix;
use crate::tokenizer::{SpecialIds, TextTokenizer};
use crate::{Error, Result};

/// Who speaks a turn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// The human (`Au`, `Tu`).
    #[default]
    User,
    /// The model (`Am`, `Tm`).
    Assistant,
}

/// When a turn starts. JSON: `{"after": 0.3}`, `{"at": 2.0}`,
/// `{"with_previous": 1.5}`, `{"after_tool_result": 0.4}`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Start {
    /// Seconds after the previous non-backchannel turn ends (negative
    /// overlaps it, e.g. a barge-in).
    After(f64),
    /// Absolute time in seconds.
    At(f64),
    /// Seconds after the previous non-backchannel turn **starts** (for
    /// backchannels and overlaps inside a long turn).
    WithPrevious(f64),
    /// Seconds after the most recent (non-cancelled) tool result arrived.
    AfterToolResult(f64),
}

/// A word with its time span, relative to the turn start (seconds).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimedWord {
    /// The word.
    pub word: String,
    /// Start (seconds from turn start).
    pub start: f64,
    /// End (seconds from turn start).
    pub end: f64,
}

/// A tool call the assistant makes during a turn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCallSpec {
    /// Optional call id (for parallel calls); included in call and result JSON.
    #[serde(default)]
    pub id: Option<String>,
    /// Tool name.
    pub name: String,
    /// Arguments (any JSON).
    #[serde(default)]
    pub arguments: serde_json::Value,
    /// The tool's result (any JSON).
    #[serde(default)]
    pub result: serde_json::Value,
    /// Seconds from `</tool_call>` until the result arrives.
    #[serde(default = "default_latency")]
    pub latency: f64,
    /// Earliest start of the call text, in seconds from the turn start.
    /// Default: right after the turn's words.
    #[serde(default)]
    pub start: Option<f64>,
    /// If set, the assistant emits `<tool_cancel>` this many seconds after
    /// `</tool_call>` and the result is dropped.
    #[serde(default)]
    pub cancel_after: Option<f64>,
}

fn default_latency() -> f64 {
    1.0
}

/// One turn.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    /// Speaker.
    pub speaker: Role,
    /// Utterance text (split on whitespace when `words` is absent).
    #[serde(default)]
    pub text: String,
    /// Start time; default `{"after": SynthConfig::default_gap}`.
    #[serde(default)]
    pub start: Option<Start>,
    /// Explicit word timings (relative to the turn start). Without them,
    /// words are spread uniformly over the duration.
    #[serde(default)]
    pub words: Option<Vec<TimedWord>>,
    /// Duration in seconds. Default: `codes.len()` frames, else the last word
    /// end, else `words × SynthConfig::seconds_per_word`.
    #[serde(default)]
    pub duration: Option<f64>,
    /// Pre-computed audio codes, one row of `codebooks` codes per frame.
    #[serde(default)]
    pub codes: Option<Vec<Vec<u32>>>,
    /// A backchannel ("mm-hmm"): does not become the anchor for later
    /// `after` / `with_previous` starts.
    #[serde(default)]
    pub backchannel: bool,
    /// Interruption: the turn stops this many seconds in (audio truncated,
    /// words starting at or after it dropped).
    #[serde(default)]
    pub cut_off_at: Option<f64>,
    /// A tool call made during this (assistant) turn.
    #[serde(default)]
    pub tool_call: Option<ToolCallSpec>,
}

/// A dialogue script.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Script {
    /// Sample id.
    #[serde(default)]
    pub id: String,
    /// Source tag (default `"synthetic"`).
    #[serde(default = "default_source")]
    pub source: String,
    /// License tag.
    #[serde(default)]
    pub license: String,
    /// User voice / speaker id.
    #[serde(default)]
    pub user_speaker: String,
    /// Assistant voice id.
    #[serde(default)]
    pub assistant_speaker: String,
    /// System prompt, injected at frame 0.
    #[serde(default)]
    pub system: Option<String>,
    /// Tool schemas (any JSON), appended to the system injection.
    #[serde(default)]
    pub tools: Vec<serde_json::Value>,
    /// Seed for dummy audio codes.
    #[serde(default)]
    pub seed: u64,
    /// Code per codebook used where a channel is silent.
    #[serde(default)]
    pub silence_codes: Option<Vec<u32>>,
    /// Turns, in chronological order per speaker.
    pub turns: Vec<Turn>,
}

fn default_source() -> String {
    "synthetic".to_string()
}

impl Script {
    /// Parse a script from JSON.
    pub fn from_json(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }
}

/// Timeline assembly settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynthConfig {
    /// Stream layout of the produced sample.
    pub layout: StreamLayout,
    /// `Tm` slots per frame used by inner-monologue words (tool calls use all).
    pub monologue_slots: usize,
    /// Put `<epad>` before user words on `Tu` too.
    pub user_epad: bool,
    /// Default word duration when nothing else gives one (seconds).
    pub seconds_per_word: f64,
    /// Default gap between turns (seconds).
    pub default_gap: f64,
    /// Trailing silence after the last event (seconds).
    pub tail: f64,
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self {
            layout: StreamLayout::default(),
            monologue_slots: 1,
            user_epad: false,
            seconds_per_word: 0.35,
            default_gap: 0.4,
            tail: 0.5,
        }
    }
}

/// Where a tool call ended up.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlacedToolCall {
    /// Index of the turn that made the call.
    pub turn: usize,
    /// The JSON written between `<tool_call>` and `</tool_call>`.
    pub call_json: String,
    /// Frame of `<tool_call>`.
    pub start_frame: usize,
    /// Frame of `</tool_call>`.
    pub end_frame: usize,
    /// Frame the result injection precedes (`None` if cancelled).
    pub result_frame: Option<usize>,
    /// Frame of `<tool_cancel>` (if cancelled).
    pub cancel_frame: Option<usize>,
}

/// Output of [`synthesize`].
#[derive(Clone, Debug, PartialEq)]
pub struct Synthesized {
    /// The sample.
    pub sample: Sample,
    /// Tool calls, in script order.
    pub tool_calls: Vec<PlacedToolCall>,
    /// Non-fatal oddities (e.g. text pushed far behind its audio).
    pub warnings: Vec<String>,
}

fn script_err(m: impl Into<String>) -> Error {
    Error::Script(m.into())
}

fn secs_to_frames(sec: f64) -> usize {
    (sec * FRAME_RATE).round().max(0.0) as usize
}

/// Places tokens on a text stream group (`Tu`: 1 slot, `Tm`: `text_slots`)
/// in reading order, never before the cursor.
struct Placer {
    slots: usize,
    cursor: usize,
    cells: Vec<(usize, usize, u32)>,
}

impl Placer {
    fn new(slots: usize) -> Self {
        Self {
            slots,
            cursor: 0,
            cells: Vec::new(),
        }
    }

    /// Move `pos` to the next position whose slot is below `limit`.
    fn norm(&self, pos: usize, limit: usize) -> usize {
        if pos % self.slots >= limit {
            (pos / self.slots + 1) * self.slots
        } else {
            pos
        }
    }

    /// Place `tokens` in order from frame `frame` (or the cursor, if later).
    /// Returns the frames of the first and last token.
    fn put_seq(&mut self, frame: usize, tokens: &[u32], limit: usize) -> (usize, usize) {
        let mut p = self.norm((frame * self.slots).max(self.cursor), limit);
        let first = p / self.slots;
        let mut last = first;
        for &t in tokens {
            p = self.norm(p, limit);
            self.cells.push((p / self.slots, p % self.slots, t));
            last = p / self.slots;
            p += 1;
        }
        self.cursor = p;
        (first, last)
    }

    /// Place a word whose audio starts at `frame`; with `epad`, the epad goes
    /// one frame earlier (or at the cursor) and the word starts on the next
    /// frame. Returns the frame of the word's first token.
    fn place_word(
        &mut self,
        frame: usize,
        tokens: &[u32],
        limit: usize,
        epad: Option<u32>,
    ) -> usize {
        match epad {
            Some(e) => {
                let (ep, _) = self.put_seq(frame.saturating_sub(1), &[e], limit);
                self.cursor = (ep + 1) * self.slots;
                self.put_seq(ep + 1, tokens, limit).0
            }
            None => self.put_seq(frame, tokens, limit).0,
        }
    }

    /// Frames needed to hold everything placed so far.
    fn frames_used(&self) -> usize {
        self.cursor.div_ceil(self.slots)
    }
}

struct AudioTurn {
    role: Role,
    turn: usize,
    start: usize,
    codes: Vec<Vec<u32>>,
}

fn call_json(c: &ToolCallSpec) -> Result<String> {
    let name = serde_json::to_string(&c.name)?;
    let args = serde_json::to_string(&c.arguments)?;
    Ok(match &c.id {
        Some(id) => format!(
            "{{\"id\":{},\"name\":{name},\"arguments\":{args}}}",
            serde_json::to_string(id)?
        ),
        None => format!("{{\"name\":{name},\"arguments\":{args}}}"),
    })
}

fn result_json(c: &ToolCallSpec) -> Result<String> {
    let res = serde_json::to_string(&c.result)?;
    Ok(match &c.id {
        Some(id) => format!("{{\"id\":{},\"result\":{res}}}", serde_json::to_string(id)?),
        None => res,
    })
}

fn check_secs(v: f64, what: &str, turn: usize) -> Result<f64> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(script_err(format!(
            "turn {turn}: {what} is not a finite number"
        )))
    }
}

/// Place `script` on a timeline and build a sample (see the [module docs](self)).
pub fn synthesize(
    script: &Script,
    tok: &dyn TextTokenizer,
    cfg: &SynthConfig,
) -> Result<Synthesized> {
    let sp = SpecialIds::from_tokenizer(tok)?;
    let l = cfg.layout;
    let q = l.codebooks;
    if l.text_slots == 0 || q == 0 {
        return Err(script_err("layout needs text slots and codebooks"));
    }
    if cfg.monologue_slots == 0 || cfg.monologue_slots > l.text_slots {
        return Err(script_err(format!(
            "monologue_slots must be in 1..={}",
            l.text_slots
        )));
    }
    let silence: Vec<u32> = match &script.silence_codes {
        Some(s) => {
            if s.len() != q || s.iter().any(|&c| c >= CODEBOOK_SIZE) {
                return Err(script_err(format!(
                    "silence_codes needs {q} codes below {CODEBOOK_SIZE}"
                )));
            }
            s.clone()
        }
        None => (0..q as u32)
            .map(|c| (c * 97 + 13) % CODEBOOK_SIZE)
            .collect(),
    };

    let mut tu = Placer::new(1);
    let mut tm = Placer::new(l.text_slots);
    let tm0 = l.index(StreamKind::AssistantText(0)) as u32;
    let mut audio: Vec<AudioTurn> = Vec::new();
    let mut words_table: Vec<WordTiming> = Vec::new();
    let mut injections: Vec<Injection> = Vec::new();
    let mut tool_calls: Vec<PlacedToolCall> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut pending_cancels: Vec<usize> = Vec::new();
    let mut first_word = [true, true]; // [Tu, Tm]
    let (mut prev_start, mut prev_end) = (0.0f64, 0.0f64);
    let mut last_result: Option<f64> = None;
    let mut last_start_frame = [0usize, 0usize];
    let mut result_frames_max = 0usize;

    // Flush cancels due at or before `frame` onto Tm.
    let flush = |tm: &mut Placer, pending: &mut Vec<usize>, frame: usize| {
        pending.sort_unstable();
        while let Some(&f) = pending.first() {
            if f > frame {
                break;
            }
            tm.put_seq(f, &[sp.tool_cancel], 1);
            pending.remove(0);
        }
    };

    for (ti, turn) in script.turns.iter().enumerate() {
        // Words with relative timings.
        let mut words: Vec<(String, f64, f64)> = match &turn.words {
            Some(ws) => ws
                .iter()
                .map(|w| {
                    let s = check_secs(w.start, "word start", ti)?;
                    let e = check_secs(w.end, "word end", ti)?;
                    if s < 0.0 || e < s {
                        return Err(script_err(format!(
                            "turn {ti}: word {:?} has a bad span {s}..{e}",
                            w.word
                        )));
                    }
                    Ok((w.word.clone(), s, e))
                })
                .collect::<Result<_>>()?,
            None => Vec::new(),
        };
        let duration = match (&turn.codes, turn.duration) {
            (Some(c), _) => c.len() as f64 / FRAME_RATE,
            (None, Some(d)) => check_secs(d, "duration", ti)?,
            (None, None) => match words.last() {
                Some(w) => w.2,
                None => turn.text.split_whitespace().count() as f64 * cfg.seconds_per_word,
            },
        };
        if turn.words.is_none() {
            let split: Vec<&str> = turn.text.split_whitespace().collect();
            let n = split.len().max(1) as f64;
            words = split
                .iter()
                .enumerate()
                .map(|(i, w)| {
                    (
                        w.to_string(),
                        duration * i as f64 / n,
                        duration * (i + 1) as f64 / n,
                    )
                })
                .collect();
        }
        if duration <= 0.0 {
            return Err(script_err(format!(
                "turn {ti} has no duration (no text, words, codes or duration)"
            )));
        }

        let start = match turn.start.unwrap_or(Start::After(cfg.default_gap)) {
            Start::After(g) => prev_end + check_secs(g, "start", ti)?,
            Start::At(t) => check_secs(t, "start", ti)?,
            Start::WithPrevious(g) => prev_start + check_secs(g, "start", ti)?,
            Start::AfterToolResult(g) => {
                last_result.ok_or_else(|| {
                    script_err(format!(
                        "turn {ti} starts after a tool result, but none arrived"
                    ))
                })? + check_secs(g, "start", ti)?
            }
        };
        if start < 0.0 {
            return Err(script_err(format!(
                "turn {ti} starts before time 0 ({start:.3} s)"
            )));
        }
        let start_frame = secs_to_frames(start);
        let ri = match turn.speaker {
            Role::User => 0,
            Role::Assistant => 1,
        };
        if start_frame < last_start_frame[ri] {
            return Err(script_err(format!(
                "turn {ti} starts before the previous {:?} turn (turns must be chronological per speaker)",
                turn.speaker
            )));
        }
        last_start_frame[ri] = start_frame;

        let mut n_frames = match &turn.codes {
            Some(c) => c.len(),
            None => secs_to_frames(duration).max(1),
        };
        if n_frames == 0 {
            return Err(script_err(format!("turn {ti} has no audio frames")));
        }
        if let Some(cut) = turn.cut_off_at {
            let cut = check_secs(cut, "cut_off_at", ti)?;
            n_frames = n_frames.min(secs_to_frames(cut).max(1));
            words.retain(|w| w.1 < cut);
        }
        let eff_dur = n_frames as f64 / FRAME_RATE;

        // Audio codes.
        let codes: Vec<Vec<u32>> = match &turn.codes {
            Some(c) => {
                for (f, row) in c.iter().enumerate() {
                    if row.len() != q || row.iter().any(|&v| v >= CODEBOOK_SIZE) {
                        return Err(script_err(format!(
                            "turn {ti} frame {f}: need {q} codes below {CODEBOOK_SIZE}"
                        )));
                    }
                }
                c[..n_frames].to_vec()
            }
            None => (0..n_frames)
                .map(|f| {
                    (0..q)
                        .map(|c| {
                            (mix(&[script.seed, ri as u64, ti as u64, f as u64, c as u64])
                                % CODEBOOK_SIZE as u64) as u32
                        })
                        .collect()
                })
                .collect(),
        };
        audio.push(AudioTurn {
            role: turn.speaker,
            turn: ti,
            start: start_frame,
            codes,
        });

        // Text.
        for (w, ws, we) in &words {
            let desired = start_frame + secs_to_frames(*ws);
            let text = if first_word[ri] {
                w.clone()
            } else {
                format!(" {w}")
            };
            let toks = tok.encode(&text);
            if toks.is_empty() {
                continue;
            }
            first_word[ri] = false;
            let placed = match turn.speaker {
                Role::User => tu.place_word(desired, &toks, 1, cfg.user_epad.then_some(sp.epad)),
                Role::Assistant => {
                    flush(&mut tm, &mut pending_cancels, desired);
                    tm.place_word(desired, &toks, cfg.monologue_slots, Some(sp.epad))
                }
            };
            if placed > desired + 2 {
                warnings.push(format!(
                    "turn {ti}: text of {w:?} placed at frame {placed}, {} frames after its audio",
                    placed - desired
                ));
            }
            words_table.push(WordTiming {
                stream: if ri == 0 { 0 } else { tm0 },
                start_ms: ((start + ws) * 1000.0).round() as u32,
                end_ms: ((start + we.min(eff_dur)) * 1000.0).round() as u32,
                text: w.clone(),
            });
        }

        // Tool call.
        if let Some(call) = &turn.tool_call {
            if turn.speaker != Role::Assistant {
                return Err(script_err(format!(
                    "turn {ti}: only the assistant calls tools"
                )));
            }
            let json = call_json(call)?;
            let req = start_frame
                + match call.start {
                    Some(s) => secs_to_frames(check_secs(s, "tool call start", ti)?),
                    None => 0,
                };
            flush(&mut tm, &mut pending_cancels, req);
            let mut toks = vec![sp.tool_call];
            toks.extend(tok.encode(&json));
            toks.push(sp.tool_call_end);
            let (first, last) = tm.put_seq(req, &toks, l.text_slots);
            first_word[1] = false;
            let latency = check_secs(call.latency, "latency", ti)?;
            let mut placed = PlacedToolCall {
                turn: ti,
                call_json: json,
                start_frame: first,
                end_frame: last,
                result_frame: None,
                cancel_frame: None,
            };
            match call.cancel_after {
                Some(c) => {
                    let f = last + 1 + secs_to_frames(check_secs(c, "cancel_after", ti)?);
                    pending_cancels.push(f);
                    placed.cancel_frame = Some(f);
                }
                None => {
                    let f = last + 1 + secs_to_frames(latency);
                    let mut toks = vec![sp.tool_result];
                    toks.extend(tok.encode(&result_json(call)?));
                    toks.push(sp.tool_result_end);
                    injections.push(Injection {
                        frame: f as u32,
                        kind: InjectionKind::ToolResult,
                        tokens: toks,
                    });
                    placed.result_frame = Some(f);
                    result_frames_max = result_frames_max.max(f + 1);
                    last_result = Some(f as f64 / FRAME_RATE);
                }
            }
            tool_calls.push(placed);
        }

        if !turn.backchannel {
            prev_start = start;
            prev_end = start + eff_dur;
        }
    }
    flush(&mut tm, &mut pending_cancels, usize::MAX);
    // Record the actual cancel frames (they may have been pushed later).
    let cancel_cells: Vec<usize> = tm
        .cells
        .iter()
        .filter(|c| c.2 == sp.tool_cancel)
        .map(|c| c.0)
        .collect();
    for (pc, f) in tool_calls
        .iter_mut()
        .filter(|c| c.cancel_frame.is_some())
        .zip(cancel_cells)
    {
        pc.cancel_frame = Some(f);
    }

    // Total frames.
    let mut frames = audio
        .iter()
        .map(|a| a.start + a.codes.len())
        .max()
        .unwrap_or(0)
        .max(tu.frames_used())
        .max(tm.frames_used())
        .max(result_frames_max)
        .max(1);
    frames += secs_to_frames(cfg.tail);

    // Grid.
    let ns = l.num_streams();
    let mut tokens = vec![0u32; frames * ns];
    let am0 = l.index(StreamKind::AssistantAudio(0));
    let au0 = l.index(StreamKind::UserAudio(0));
    for f in 0..frames {
        let row = &mut tokens[f * ns..(f + 1) * ns];
        for (s, cell) in row.iter_mut().enumerate() {
            *cell = match l.kind(s) {
                StreamKind::UserText | StreamKind::AssistantText(_) => sp.pad,
                StreamKind::AssistantAudio(c) | StreamKind::UserAudio(c) => silence[c],
            };
        }
    }
    let mut occupied = [vec![false; frames], vec![false; frames]];
    for a in &audio {
        let (base, ri) = match a.role {
            Role::User => (au0, 0),
            Role::Assistant => (am0, 1),
        };
        for (i, row) in a.codes.iter().enumerate() {
            let f = a.start + i;
            if occupied[ri][f] {
                return Err(script_err(format!(
                    "turn {} overlaps another {:?} turn at frame {f}",
                    a.turn, a.role
                )));
            }
            occupied[ri][f] = true;
            tokens[f * ns + base..f * ns + base + q].copy_from_slice(row);
        }
    }
    let tu_idx = l.index(StreamKind::UserText);
    for &(f, _, t) in &tu.cells {
        tokens[f * ns + tu_idx] = t;
    }
    for &(f, slot, t) in &tm.cells {
        tokens[f * ns + l.index(StreamKind::AssistantText(slot))] = t;
    }

    // System injection at frame 0.
    if script.system.is_some() || !script.tools.is_empty() {
        let mut text = script.system.clone().unwrap_or_default();
        for t in &script.tools {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&serde_json::to_string(t)?);
        }
        let mut toks = vec![sp.sys];
        toks.extend(tok.encode(&text));
        toks.push(sp.sys_end);
        injections.insert(
            0,
            Injection {
                frame: 0,
                kind: InjectionKind::System,
                tokens: toks,
            },
        );
    }
    injections.sort_by_key(|i| i.frame);
    words_table.sort_by_key(|w| (w.start_ms, w.stream));

    let extra = serde_json::json!({
        "tool_calls": tool_calls,
    });
    let sample = Sample {
        layout: l,
        frames,
        text_vocab: tok.vocab_size(),
        text_pad: sp.pad,
        tokens,
        injections,
        words: words_table,
        meta: SampleMeta {
            id: script.id.clone(),
            source: script.source.clone(),
            license: script.license.clone(),
            speakers: vec![
                SpeakerInfo {
                    role: "user".into(),
                    id: script.user_speaker.clone(),
                },
                SpeakerInfo {
                    role: "assistant".into(),
                    id: script.assistant_speaker.clone(),
                },
            ],
            tokenizer: tok.name(),
            extra,
        },
    };
    crate::validate::check(&sample)?;
    Ok(Synthesized {
        sample,
        tool_calls,
        warnings,
    })
}
