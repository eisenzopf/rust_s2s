//! The text tokenizer of rust_s2s (plan §3.2, §6).
//!
//! [`HfTokenizer`] wraps a Hugging Face `tokenizer.json` (any base LLM: the
//! model is not chosen yet, plan §15), makes sure every stream special token
//! of [`s2s_core::streams::special::ALL`] exists (appending the missing ones
//! as added special tokens), and implements both tokenizer traits of the
//! workspace:
//!
//! - [`s2s_tools::TokenizerInfo`]: the view `llguidance` grammar masks and
//!   tool-result budgets use;
//! - [`s2s_data::TextTokenizer`]: the view the training-data synthesizer uses.
//!
//! Both views are the same tokenizer: same ids for the same text and
//! specials, same vocabulary size.
//!
//! # Example
//!
//! ```
//! use s2s_data::TextTokenizer;
//! use s2s_tokenizer::{testing, HfTokenizer};
//! use s2s_tools::TokenizerInfo;
//!
//! // A real model: `HfTokenizer::from_file("tokenizer.json")`. Here, a
//! // small byte-level BPE trained on an embedded corpus.
//! let tok: HfTokenizer = testing::byte_level_bpe(700).unwrap();
//! let sp = tok.special_ids();
//! assert_eq!(tok.special_id("<tool_call>"), Some(sp.tool_call));
//!
//! // Words of a stream: the first as is, the rest with a leading space.
//! let words = tok.encode_words(["Let", "me", "check."]);
//! assert_eq!(words.concat(), tok.encode("Let me check."));
//!
//! // Both trait views agree.
//! let text = "{\"city\": \"Paris\"}";
//! assert_eq!(TokenizerInfo::tokenize(&tok, text.as_bytes()), TextTokenizer::encode(&tok, text));
//! assert_eq!(TokenizerInfo::vocab_size(&tok), TextTokenizer::vocab_size(&tok) as usize);
//! assert!(tok.is_canonical());
//! ```
//!
//! # Encoding conventions for the text streams
//!
//! The model's text streams (`Tu`, `Tm`) and injection segments are built
//! from these rules, at training time (`s2s-data`) and at inference
//! (runtime, grammar masks) alike:
//!
//! 1. **Text is encoded exactly as given.** No BOS/EOS or other template
//!    tokens (the post-processor is never applied), no truncation or
//!    padding, and **no implicit prefix space**: on load, `Prepend`
//!    normalizers are removed, `Metaspace` pre-tokenizers get
//!    `prepend_scheme: never` and `ByteLevel` pre-tokenizers
//!    `add_prefix_space: false` (see [`HfTokenizer::adjustments`]). Spaces
//!    are therefore always explicit.
//! 2. **Special-token names in text are plain text.** Encoding
//!    `"<tool_call>"` yields ordinary tokens, never the special id: special
//!    ids only appear where the caller places them (the runtime when it
//!    samples or injects them, `s2s-data` when it renders markers).
//! 3. **Words.** Stream text is tokenized word by word: the first word on a
//!    stream as is, every later word with one leading space
//!    ([`HfTokenizer::encode_word`], [`HfTokenizer::encode_words`]), which is
//!    what `s2s_data::synthesize` does. For byte-level BPE tokenizers with
//!    GPT-2-style regex pre-tokenization (Llama 3, Qwen, GPT-NeoX, …),
//!    concatenating the word encodings of single-spaced text gives the same
//!    ids as encoding the whole string. Tool-call JSON and injected segments
//!    are encoded as one string.
//! 4. **Bytes and unknown characters.** [`HfTokenizer::encode_bytes`]
//!    accepts arbitrary bytes (llguidance tokenizes byte strings that may
//!    split a UTF-8 sequence): valid UTF-8 runs go through the tokenizer, and
//!    each invalid byte becomes its single-byte token. Byte-level
//!    vocabularies can represent every byte; SentencePiece-style
//!    vocabularies with `byte_fallback` encode characters outside the
//!    vocabulary as `<0xHH>` tokens. Without either, unknown characters map
//!    to the model's unknown token (lossy; such a tokenizer is reported as
//!    not canonical).
//! 5. **Decoding** concatenates the raw bytes of each token
//!    ([`s2s_tools::TokenPiece`]) and writes special tokens by name; byte
//!    sequences are converted to text lossily only at the end, so tokens
//!    that split a character decode correctly together.
//!
//! # Canonical tokenization
//!
//! llguidance forces tokens (e.g. the fixed `{"name": "` prefix of a tool
//! call) only when the tokenizer reports
//! [`tokenize_is_canonical`](s2s_tools::TokenizerInfo::tokenize_is_canonical):
//! it then allows exactly the tokenization the tokenizer gives for forced
//! bytes. That is right when this tokenization is what the model was trained
//! on, which holds by construction here (training data is encoded with the
//! same [`HfTokenizer::encode`]) provided encoding is deterministic and
//! byte-exact. [`HfTokenizer`] checks this on load: the vocabulary type is
//! known, BPE dropout is off, and a set of probe strings (JSON fragments,
//! spacing, digits, case, Unicode including NFKC-sensitive characters,
//! special-token names as text) round-trip exactly and never produce
//! special tokens. A tokenizer that fails (e.g. one with a lowercasing or
//! NFKC normalizer, or `<unk>` for unknown characters) reports `false`, and
//! llguidance then allows every tokenization of the grammar's bytes, which
//! is still correct, only without forcing. [`HfTokenizer::with_canonical`]
//! overrides the result.
//!
//! # Supported vocabularies
//!
//! Token bytes are recovered from the decoder type ([`VocabKind`]):
//! byte-level BPE (`ByteLevel` decoder), SentencePiece style
//! (`Metaspace`/`Replace("▁", " ")`/`ByteFallback` decoders), or no decoder
//! (token strings are the text). WordPiece, CTC and suffix-BPE decoders are
//! rejected, since their token strings are not the text they stand for.
//!
//! # Tests without downloads
//!
//! [`testing`] trains small tokenizers of both supported styles on an
//! embedded corpus, so the crate is tested without fetching a real
//! `tokenizer.json`. Set `S2S_TOKENIZER_JSON` to a real file and run the
//! ignored test (`cargo test -p s2s-tokenizer -- --ignored`) to check one.

mod adapters;
mod error;
mod hf;
pub mod testing;

pub use error::Error;
pub use hf::{HfTokenizer, VocabKind, CANONICAL_PROBES};

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;
