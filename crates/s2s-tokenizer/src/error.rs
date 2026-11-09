/// Errors loading or building a tokenizer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The `tokenizer.json` could not be read or parsed, or the `tokenizers`
    /// library reported an error.
    Hf(String),
    /// The decoder type does not let token strings be mapped back to the
    /// bytes they stand for (WordPiece, CTC, suffix BPE).
    UnsupportedDecoder(String),
    /// A special token could not be added or resolved.
    Special(String),
    /// The vocabulary is empty or too large for `u32` ids.
    BadVocab(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hf(e) => write!(f, "tokenizers: {e}"),
            Self::UnsupportedDecoder(d) => write!(f, "unsupported tokenizer decoder: {d}"),
            Self::Special(e) => write!(f, "special token: {e}"),
            Self::BadVocab(e) => write!(f, "bad vocabulary: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl Error {
    pub(crate) fn hf(e: impl std::fmt::Display) -> Self {
        Self::Hf(e.to_string())
    }
}
