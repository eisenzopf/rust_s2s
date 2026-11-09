//! The per-frame token layout shared by the model, data pipeline and runtime.
//!
//! Every 80 ms frame carries one token per **stream** (plan §3.2):
//!
//! | Index | Stream | Count | Delay (frames) |
//! |---|---|---|---|
//! | 0 | `Tu` user text (transcript, STT) | 1 | `user_text_delay` (text lags audio) |
//! | 1.. | `Tm` assistant text slots (inner monologue, tool calls) | `text_slots` | 0 |
//! | .. | `Am` assistant audio codebooks (TTS) | `codebooks` | 0 for codebook 0, `acoustic_delay` for the rest |
//! | .. | `Au` user audio codebooks (input) | `codebooks` | 0 for codebook 0, `acoustic_delay` for the rest |
//!
//! The first `1 + text_slots + codebooks` streams are the ones the model
//! **generates** (in that order, which is also the Depth Transformer's
//! sub-step order). User audio is input only.

/// Sample rate of all audio, in Hz.
pub const SAMPLE_RATE: u32 = 24_000;
/// Samples per frame (80 ms at 24 kHz).
pub const FRAME_SAMPLES: usize = 1920;
/// Frames per second.
pub const FRAME_RATE: f64 = SAMPLE_RATE as f64 / FRAME_SAMPLES as f64;
/// Entries per Mimi codebook.
pub const CODEBOOK_SIZE: u32 = 2048;
/// Audio token used where no real code exists (before the first frame,
/// delay padding, text-only injection positions). Audio vocab is therefore
/// `CODEBOOK_SIZE + 1`.
pub const AUDIO_PAD: u32 = CODEBOOK_SIZE;
/// Audio vocabulary size including [`AUDIO_PAD`].
pub const AUDIO_VOCAB: u32 = CODEBOOK_SIZE + 1;

/// Names of the special text tokens (plan §3.2). Their ids come from the
/// tokenizer, which appends any that the base vocabulary lacks.
pub mod special {
    /// No text in this slot.
    pub const PAD: &str = "<pad>";
    /// End of padding: the next slot starts a word.
    pub const EPAD: &str = "<epad>";
    /// Opens a tool call in the assistant text stream.
    pub const TOOL_CALL: &str = "<tool_call>";
    /// Closes a tool call.
    pub const TOOL_CALL_END: &str = "</tool_call>";
    /// Opens an injected tool result (input only, never sampled).
    pub const TOOL_RESULT: &str = "<tool_result>";
    /// Closes an injected tool result.
    pub const TOOL_RESULT_END: &str = "</tool_result>";
    /// The model abandons an in-flight tool call.
    pub const TOOL_CANCEL: &str = "<tool_cancel>";
    /// Opens an injected system prompt / tool schema block.
    pub const SYS: &str = "<sys>";
    /// Closes an injected system block.
    pub const SYS_END: &str = "</sys>";
    /// All special tokens, in a fixed order.
    pub const ALL: [&str; 9] = [
        PAD,
        EPAD,
        TOOL_CALL,
        TOOL_CALL_END,
        TOOL_RESULT,
        TOOL_RESULT_END,
        TOOL_CANCEL,
        SYS,
        SYS_END,
    ];
}

/// What a stream carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StreamKind {
    /// User transcript (one slot).
    UserText,
    /// Assistant text slot `i` (`0..text_slots`).
    AssistantText(usize),
    /// Assistant audio codebook `i` (`0..codebooks`).
    AssistantAudio(usize),
    /// User audio codebook `i` (`0..codebooks`).
    UserAudio(usize),
}

impl StreamKind {
    /// Whether the stream holds text tokens (vs audio codes).
    pub fn is_text(self) -> bool {
        matches!(self, StreamKind::UserText | StreamKind::AssistantText(_))
    }
}

/// Stream layout and delays for one model configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamLayout {
    /// Assistant text slots per frame (`m` in the plan; 4 gives 50 tok/s).
    pub text_slots: usize,
    /// Mimi codebooks per audio stream (`Q`; 8).
    pub codebooks: usize,
    /// How many frames the user transcript lags the user's audio (`d_u`).
    pub user_text_delay: usize,
    /// How many frames acoustic codebooks (1..) lag the semantic one (`τ`).
    pub acoustic_delay: usize,
}

impl Default for StreamLayout {
    /// The plan's default: 4 text slots, 8 codebooks, `d_u = 6`, `τ = 1`.
    fn default() -> Self {
        Self {
            text_slots: 4,
            codebooks: 8,
            user_text_delay: 6,
            acoustic_delay: 1,
        }
    }
}

impl StreamLayout {
    /// Total streams per frame.
    pub fn num_streams(&self) -> usize {
        1 + self.text_slots + 2 * self.codebooks
    }

    /// Streams the model generates per frame (`K` in the plan; 13 by default).
    pub fn num_generated(&self) -> usize {
        1 + self.text_slots + self.codebooks
    }

    /// The kind of stream at index `i`.
    ///
    /// # Panics
    /// If `i >= num_streams()`.
    pub fn kind(&self, i: usize) -> StreamKind {
        let (t, q) = (self.text_slots, self.codebooks);
        match i {
            0 => StreamKind::UserText,
            i if i <= t => StreamKind::AssistantText(i - 1),
            i if i <= t + q => StreamKind::AssistantAudio(i - 1 - t),
            i if i <= t + 2 * q => StreamKind::UserAudio(i - 1 - t - q),
            _ => panic!(
                "stream index {i} out of range ({} streams)",
                self.num_streams()
            ),
        }
    }

    /// The index of a stream kind.
    ///
    /// # Panics
    /// If the slot/codebook number is out of range for this layout.
    pub fn index(&self, kind: StreamKind) -> usize {
        let (t, q) = (self.text_slots, self.codebooks);
        match kind {
            StreamKind::UserText => 0,
            StreamKind::AssistantText(s) => {
                assert!(s < t, "text slot {s} out of range");
                1 + s
            }
            StreamKind::AssistantAudio(c) => {
                assert!(c < q, "codebook {c} out of range");
                1 + t + c
            }
            StreamKind::UserAudio(c) => {
                assert!(c < q, "codebook {c} out of range");
                1 + t + q + c
            }
        }
    }

    /// All stream kinds, in index order.
    pub fn kinds(&self) -> Vec<StreamKind> {
        (0..self.num_streams()).map(|i| self.kind(i)).collect()
    }

    /// Delay of the stream at index `i`, in frames.
    pub fn delay(&self, i: usize) -> usize {
        match self.kind(i) {
            StreamKind::UserText => self.user_text_delay,
            StreamKind::AssistantText(_) => 0,
            StreamKind::AssistantAudio(c) | StreamKind::UserAudio(c) => {
                if c == 0 {
                    0
                } else {
                    self.acoustic_delay
                }
            }
        }
    }

    /// Delays for every stream, in index order (for `apply_delays`).
    pub fn delays(&self) -> Vec<usize> {
        (0..self.num_streams()).map(|i| self.delay(i)).collect()
    }

    /// Largest delay of any stream.
    pub fn max_delay(&self) -> usize {
        self.delays().into_iter().max().unwrap_or(0)
    }

    /// Per-stream pad tokens, given the text pad id (audio streams use
    /// [`AUDIO_PAD`]).
    pub fn pad_tokens(&self, text_pad: u32) -> Vec<u32> {
        self.kinds()
            .into_iter()
            .map(|k| if k.is_text() { text_pad } else { AUDIO_PAD })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_layout() {
        let l = StreamLayout::default();
        assert_eq!(l.num_streams(), 21);
        assert_eq!(l.num_generated(), 13);
        assert_eq!(l.kind(0), StreamKind::UserText);
        assert_eq!(l.kind(1), StreamKind::AssistantText(0));
        assert_eq!(l.kind(4), StreamKind::AssistantText(3));
        assert_eq!(l.kind(5), StreamKind::AssistantAudio(0));
        assert_eq!(l.kind(12), StreamKind::AssistantAudio(7));
        assert_eq!(l.kind(13), StreamKind::UserAudio(0));
        assert_eq!(l.kind(20), StreamKind::UserAudio(7));
        for i in 0..l.num_streams() {
            assert_eq!(l.index(l.kind(i)), i);
        }
        assert_eq!(l.delay(0), 6);
        assert_eq!(l.delay(5), 0);
        assert_eq!(l.delay(6), 1);
        assert_eq!(l.delay(13), 0);
        assert_eq!(l.delay(14), 1);
        assert_eq!(l.max_delay(), 6);
        assert!((FRAME_RATE - 12.5).abs() < 1e-12);
    }

    #[test]
    #[should_panic]
    fn out_of_range() {
        StreamLayout::default().kind(21);
    }
}
