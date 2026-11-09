//! The tokenizer abstraction the grammar engine and token budgets use.
//!
//! The base LLM (and so its tokenizer) is not chosen yet (plan §15), so the
//! crate only depends on the small [`TokenizerInfo`] trait. A real tokenizer
//! (e.g. a Hugging Face `tokenizer.json` loader) implements it later; the
//! [`ByteTokenizer`] here is a tiny deterministic implementation for tests,
//! examples and bring-up.
//!
//! [`tok_env`] adapts any [`TokenizerInfo`] to the `toktrie::TokEnv` that
//! `llguidance` consumes.

use std::collections::HashMap;
use std::sync::Arc;

use s2s_core::streams::special;
use toktrie::{TokEnv, TokRxInfo, TokTrie, TokenizerEnv};

/// What a single token id decodes to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenPiece {
    /// An ordinary token: its raw bytes. For byte-level BPE vocabularies these
    /// are the *decoded* bytes (after undoing the GPT-2 byte-to-unicode map),
    /// and may be an incomplete UTF-8 sequence.
    Bytes(Vec<u8>),
    /// A special/control token, by name (for example `<tool_call>`).
    Special(String),
}

/// Minimal view of a tokenizer: enough to build grammar masks and count tokens.
///
/// Implementations must be cheap to query by id; [`tok_env`] calls
/// [`token`](Self::token) once per id when it builds its trie.
pub trait TokenizerInfo: Send + Sync {
    /// Number of token ids (`0..vocab_size`). Model logits may be longer
    /// (padded vocabularies); the extra ids are never allowed by a mask.
    fn vocab_size(&self) -> usize;

    /// What token `id` decodes to.
    ///
    /// # Panics
    /// May panic if `id >= vocab_size()`.
    fn token(&self, id: u32) -> TokenPiece;

    /// Tokenize plain text (special-token names are **not** interpreted).
    fn tokenize(&self, text: &[u8]) -> Vec<u32>;

    /// Whether [`tokenize`](Self::tokenize) returns the tokenizer's canonical
    /// tokenization (llguidance only forces tokens when this is true).
    fn tokenize_is_canonical(&self) -> bool {
        true
    }

    /// Id of a special token by name. The default scans the vocabulary.
    fn special_token_id(&self, name: &str) -> Option<u32> {
        (0..self.vocab_size() as u32)
            .find(|&id| matches!(self.token(id), TokenPiece::Special(ref n) if n == name))
    }

    /// Number of tokens `text` would take as plain text.
    fn count_tokens(&self, text: &str) -> usize {
        self.tokenize(text.as_bytes()).len()
    }
}

/// Errors building a token environment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenizerError {
    /// A required special token is missing from the vocabulary.
    MissingSpecialToken(&'static str),
    /// The vocabulary is empty or too large for `u32` ids.
    BadVocabSize(usize),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingSpecialToken(name) => {
                write!(f, "tokenizer has no special token {name}")
            }
            Self::BadVocabSize(n) => write!(f, "unsupported vocabulary size {n}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

/// `toktrie::TokenizerEnv` backed by a [`TokenizerInfo`].
struct TokEnvAdapter {
    trie: TokTrie,
    tokenizer: Arc<dyn TokenizerInfo>,
}

impl TokenizerEnv for TokEnvAdapter {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<u32> {
        self.tokenizer.tokenize(s)
    }

    fn tokenize_is_canonical(&self) -> bool {
        self.tokenizer.tokenize_is_canonical()
    }
}

/// Build a `toktrie` token environment whose end-of-sequence token is the
/// special token `eos_name`.
///
/// Special tokens are stored the way llguidance expects
/// (`0xFF` + name), so grammars never produce them as text. Ordinary tokens
/// whose bytes start with `0xFF` (impossible in valid UTF-8, so never part of
/// JSON) are left out of the trie and can therefore never be allowed.
///
/// The tool-call grammar uses `eos_name = "</tool_call>"`, so llguidance's
/// "grammar finished" token *is* the span terminator.
pub fn tok_env(
    tokenizer: Arc<dyn TokenizerInfo>,
    eos_name: &'static str,
) -> Result<TokEnv, TokenizerError> {
    let vocab = tokenizer.vocab_size();
    if vocab == 0 || u32::try_from(vocab).is_err() {
        return Err(TokenizerError::BadVocabSize(vocab));
    }
    let eos = tokenizer
        .special_token_id(eos_name)
        .ok_or(TokenizerError::MissingSpecialToken(eos_name))?;
    let words: Vec<Vec<u8>> = (0..vocab as u32)
        .map(|id| match tokenizer.token(id) {
            TokenPiece::Bytes(b) if b.first() == Some(&TokTrie::SPECIAL_TOKEN_MARKER) => Vec::new(),
            TokenPiece::Bytes(b) => b,
            TokenPiece::Special(name) => {
                let mut w = Vec::with_capacity(name.len() + 1);
                w.push(TokTrie::SPECIAL_TOKEN_MARKER);
                w.extend_from_slice(name.as_bytes());
                w
            }
        })
        .collect();
    let trie = TokTrie::from(&TokRxInfo::new(vocab as u32, eos), &words);
    Ok(Arc::new(TokEnvAdapter { trie, tokenizer }))
}

/// A tiny deterministic tokenizer: 256 byte tokens, then optional multi-byte
/// "merge" tokens, then the special tokens from [`special::ALL`].
///
/// Tokenization is greedy longest-match over the byte and merge tokens. It is
/// meant for tests, examples and bring-up, not for real models.
///
/// Ids: `0..256` are single bytes, `256..256 + merges.len()` are the merges
/// in the order given, and the special tokens follow.
#[derive(Clone, Debug)]
pub struct ByteTokenizer {
    merges: Vec<Vec<u8>>,
    specials: Vec<String>,
    /// Merge bytes -> id, for greedy matching.
    merge_ids: HashMap<Vec<u8>, u32>,
    max_merge_len: usize,
}

impl ByteTokenizer {
    /// A tokenizer with the given merges (duplicates and merges shorter than
    /// two bytes are ignored) and the standard special tokens.
    pub fn new<S: AsRef<[u8]>>(merges: &[S]) -> Self {
        let mut list: Vec<Vec<u8>> = Vec::new();
        let mut merge_ids = HashMap::new();
        for m in merges {
            let m = m.as_ref().to_vec();
            if m.len() < 2 || merge_ids.contains_key(&m) {
                continue;
            }
            merge_ids.insert(m.clone(), 256 + list.len() as u32);
            list.push(m);
        }
        let max_merge_len = list.iter().map(Vec::len).max().unwrap_or(1);
        Self {
            merges: list,
            specials: special::ALL.iter().map(|s| s.to_string()).collect(),
            merge_ids,
            max_merge_len,
        }
    }

    /// A tokenizer with merges for common JSON / tool-call fragments, so
    /// tokens frequently span grammar boundaries (a harder case for masks).
    pub fn with_json_merges() -> Self {
        Self::new(&[
            "{\"",
            "\":",
            "\": ",
            "\",",
            "\", \"",
            "\"}",
            "}}",
            "name",
            "\"name\": \"",
            "arguments",
            "\"arguments\": {",
            "true",
            "false",
            "null",
            "00",
            ", ",
            "the",
            "er",
            "in",
            " a",
        ])
    }

    /// Id of the special token `name`, if present.
    pub fn special(&self, name: &str) -> Option<u32> {
        self.specials
            .iter()
            .position(|s| s == name)
            .map(|i| (256 + self.merges.len() + i) as u32)
    }

    /// Decode a token sequence to text; special tokens are written by name.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            match self.token(id) {
                TokenPiece::Bytes(b) => bytes.extend_from_slice(&b),
                TokenPiece::Special(n) => bytes.extend_from_slice(n.as_bytes()),
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

impl TokenizerInfo for ByteTokenizer {
    fn vocab_size(&self) -> usize {
        256 + self.merges.len() + self.specials.len()
    }

    fn token(&self, id: u32) -> TokenPiece {
        let id = id as usize;
        let m = self.merges.len();
        if id < 256 {
            TokenPiece::Bytes(vec![id as u8])
        } else if id < 256 + m {
            TokenPiece::Bytes(self.merges[id - 256].clone())
        } else {
            TokenPiece::Special(self.specials[id - 256 - m].clone())
        }
    }

    fn tokenize(&self, text: &[u8]) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len());
        let mut i = 0;
        while i < text.len() {
            let longest = (2..=self.max_merge_len.min(text.len() - i))
                .rev()
                .find_map(|n| self.merge_ids.get(&text[i..i + n]).map(|&id| (id, n)));
            match longest {
                Some((id, n)) => {
                    out.push(id);
                    i += n;
                }
                None => {
                    out.push(u32::from(text[i]));
                    i += 1;
                }
            }
        }
        out
    }

    fn special_token_id(&self, name: &str) -> Option<u32> {
        self.special(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_tokenizer_round_trip_and_greedy_merges() {
        let t = ByteTokenizer::with_json_merges();
        let text = "{\"name\": \"get_weather\", \"arguments\": {}}";
        let ids = t.tokenize(text.as_bytes());
        assert!(ids.len() < text.len(), "merges should shorten the sequence");
        assert_eq!(t.decode(&ids), text);
        assert_eq!(t.count_tokens(text), ids.len());
        // Greedy longest match picks the 10-byte merge.
        assert_eq!(t.tokenize(b"\"name\": \"")[..], [256 + 8]);
    }

    #[test]
    fn specials_are_appended_and_found() {
        let t = ByteTokenizer::new::<&str>(&[]);
        assert_eq!(t.vocab_size(), 256 + special::ALL.len());
        for (i, name) in special::ALL.iter().enumerate() {
            let id = t.special(name).unwrap();
            assert_eq!(id as usize, 256 + i);
            assert_eq!(t.token(id), TokenPiece::Special(name.to_string()));
            // The trait's default scan agrees with the fast path.
            let scanned = (0..t.vocab_size() as u32)
                .find(|&j| t.token(j) == TokenPiece::Special(name.to_string()));
            assert_eq!(scanned, Some(id));
        }
        assert_eq!(t.special("<nope>"), None);
        // Special-token names in plain text are not interpreted.
        assert_eq!(t.tokenize(b"<pad>").len(), 5);
    }

    #[test]
    fn tok_env_marks_specials_and_eos() {
        let t = Arc::new(ByteTokenizer::new(&["ab"]));
        let end = t.special(special::TOOL_CALL_END).unwrap();
        let env = tok_env(t.clone(), special::TOOL_CALL_END).unwrap();
        let trie = env.tok_trie();
        assert_eq!(trie.vocab_size(), t.vocab_size());
        assert_eq!(trie.eos_token(), end);
        assert!(trie.is_special_token(end));
        assert!(!trie.is_special_token(u32::from(b'a')));
        assert_eq!(
            trie.get_special_token(special::TOOL_CALL),
            t.special(special::TOOL_CALL)
        );
        assert_eq!(trie.token(256), b"ab");
        // The 0xFF byte token is excluded (it would collide with the marker).
        assert!(trie.token(0xFF).is_empty());
    }

    #[test]
    fn tok_env_requires_eos_special() {
        #[derive(Debug)]
        struct NoSpecials;
        impl TokenizerInfo for NoSpecials {
            fn vocab_size(&self) -> usize {
                256
            }
            fn token(&self, id: u32) -> TokenPiece {
                TokenPiece::Bytes(vec![id as u8])
            }
            fn tokenize(&self, text: &[u8]) -> Vec<u32> {
                text.iter().map(|&b| u32::from(b)).collect()
            }
        }
        let err = tok_env(Arc::new(NoSpecials), special::TOOL_CALL_END)
            .err()
            .unwrap();
        assert_eq!(err, TokenizerError::MissingSpecialToken("</tool_call>"));
        assert!(err.to_string().contains("</tool_call>"));
    }
}
