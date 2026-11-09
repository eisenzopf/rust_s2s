//! One conversation: the in-memory form of a shard record.

use s2s_core::streams::StreamLayout;
use serde::{Deserialize, Serialize};

/// What an injection segment carries (plan §3.5). Informational: assembly
/// treats every kind the same way.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InjectionKind {
    /// `<sys>…</sys>`: system prompt and tool schemas (normally at frame 0).
    System,
    /// `<tool_result>…</tool_result>`: a tool result arriving mid-dialogue.
    ToolResult,
    /// Anything else (e.g. text replay, plan §9.1 B2).
    Text,
}

impl InjectionKind {
    /// On-disk code.
    pub fn code(self) -> u32 {
        match self {
            InjectionKind::System => 0,
            InjectionKind::ToolResult => 1,
            InjectionKind::Text => 2,
        }
    }

    /// Inverse of [`code`](Self::code).
    pub fn from_code(c: u32) -> Option<Self> {
        match c {
            0 => Some(InjectionKind::System),
            1 => Some(InjectionKind::ToolResult),
            2 => Some(InjectionKind::Text),
            _ => None,
        }
    }
}

/// Text tokens inserted as text-only positions **before** frame `frame`
/// (i.e. between frame `frame - 1` and frame `frame`; `frame == 0` means before
/// the first frame).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Injection {
    /// Frame index the segment precedes, `0..=frames`.
    pub frame: u32,
    /// What the segment carries.
    pub kind: InjectionKind,
    /// Text token ids, including the delimiting specials (e.g. `<tool_result>`).
    pub tokens: Vec<u32>,
}

/// One word with its time span, for the optional word-timestamp side table.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WordTiming {
    /// Index of the text stream the word belongs to (0 = `Tu`, 1 = `Tm` slot 0).
    pub stream: u32,
    /// Start time in milliseconds from the start of the sample.
    pub start_ms: u32,
    /// End time in milliseconds (`>= start_ms`).
    pub end_ms: u32,
    /// The word.
    pub text: String,
}

/// A speaker in the conversation.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeakerInfo {
    /// Role: `"user"` or `"assistant"` (free-form for real data).
    pub role: String,
    /// Speaker / voice identifier.
    #[serde(default)]
    pub id: String,
}

/// Descriptive per-sample metadata, stored as JSON inside the record.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SampleMeta {
    /// Sample identifier (unique within a dataset, by convention).
    #[serde(default)]
    pub id: String,
    /// Where the sample came from (corpus name, generator, ...).
    #[serde(default)]
    pub source: String,
    /// License tag (plan §9.3), so a training mix can be filtered.
    #[serde(default)]
    pub license: String,
    /// Speakers, typically `[user, assistant]`.
    #[serde(default)]
    pub speakers: Vec<SpeakerInfo>,
    /// Name of the text tokenizer the text ids come from.
    #[serde(default)]
    pub tokenizer: String,
    /// Free-form extra metadata.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub extra: serde_json::Value,
}

/// One conversation.
///
/// `tokens` is the **aligned (undelayed)** frame-major grid
/// `[frames, layout.num_streams()]` of `u32` token ids: row `t` holds every
/// stream's token for frame `t`, in [`StreamLayout`] index order. Delays are
/// applied later, when a training example is assembled
/// ([`crate::assemble()`]), so `user_text_delay` and `acoustic_delay` can change
/// without rewriting shards; `layout` records the delays the data was prepared
/// for.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    /// Stream layout (slot and codebook counts must match the model).
    pub layout: StreamLayout,
    /// Number of frames.
    pub frames: usize,
    /// Text vocabulary size; every text token is below it.
    pub text_vocab: u32,
    /// Id of the text `<pad>` token.
    pub text_pad: u32,
    /// `[frames, num_streams]` token ids, frame-major.
    pub tokens: Vec<u32>,
    /// Injection segments, sorted by `frame` (stable for equal frames).
    pub injections: Vec<Injection>,
    /// Optional word-timestamp side table.
    pub words: Vec<WordTiming>,
    /// Descriptive metadata.
    pub meta: SampleMeta,
}

impl Sample {
    /// Streams per frame.
    pub fn num_streams(&self) -> usize {
        self.layout.num_streams()
    }

    /// Token of stream `s` at frame `t`.
    ///
    /// # Panics
    /// If out of range.
    pub fn token(&self, t: usize, s: usize) -> u32 {
        self.tokens[t * self.num_streams() + s]
    }

    /// One frame's row of tokens.
    pub fn frame(&self, t: usize) -> &[u32] {
        let s = self.num_streams();
        &self.tokens[t * s..(t + 1) * s]
    }

    /// The non-`<pad>` tokens of the assistant text slots (`Tm`), in reading
    /// order (frame by frame, slot 0 first). Includes `<epad>` and tool tokens.
    pub fn assistant_text(&self) -> Vec<u32> {
        let ns = self.num_streams();
        let slots = self.layout.text_slots;
        (0..self.frames)
            .flat_map(|t| (1..=slots).map(move |s| t * ns + s))
            .map(|i| self.tokens[i])
            .filter(|&t| t != self.text_pad)
            .collect()
    }

    /// The non-`<pad>` tokens of the user transcript (`Tu`), in order.
    pub fn user_text(&self) -> Vec<u32> {
        let ns = self.num_streams();
        (0..self.frames)
            .map(|t| self.tokens[t * ns])
            .filter(|&t| t != self.text_pad)
            .collect()
    }

    /// Total injected text tokens.
    pub fn injection_tokens(&self) -> usize {
        self.injections.iter().map(|i| i.tokens.len()).sum()
    }
}
