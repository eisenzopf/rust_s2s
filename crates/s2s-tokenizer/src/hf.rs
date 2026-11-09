//! [`HfTokenizer`]: a Hugging Face `tokenizer.json` with the stream special
//! tokens and the encoding conventions of the crate docs.

use std::collections::HashMap;
use std::path::Path;

use s2s_core::streams::special;
use s2s_data::SpecialIds;
use s2s_tools::TokenPiece;
use tokenizers::pre_tokenizers::metaspace::PrependScheme;
use tokenizers::{
    normalizers, pre_tokenizers, AddedToken, DecoderWrapper, Model, ModelWrapper,
    NormalizerWrapper, PreTokenizerWrapper, Tokenizer,
};

use crate::{Error, Result};

/// How token strings in the vocabulary map to the bytes they stand for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VocabKind {
    /// GPT-2-style byte-level BPE (Llama 3, Qwen, GPT-NeoX, Mistral Tekken,
    /// …): every character of a token string stands for one byte through the
    /// GPT-2 byte-to-unicode table.
    ByteLevel,
    /// SentencePiece style (Llama 2, Mistral v1, Gemma, …): `space` (usually
    /// `▁`) stands for a space, and byte-fallback tokens `<0xHH>` for the
    /// single byte `0xHH`.
    SentencePiece {
        /// The character that stands for a space.
        space: char,
    },
    /// No decoder: token strings are the text itself.
    Verbatim,
}

/// The probe strings [`HfTokenizer`] round-trips to decide whether its
/// tokenization is canonical (see the crate docs).
pub const CANONICAL_PROBES: &[&str] = &[
    "a",
    " a",
    "  a",
    "{\"",
    "\"}",
    "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"unit\": \"celsius\"}}",
    "{\"name\":\"set_timer\",\"arguments\":{\"seconds\":90,\"loud\":true,\"label\":null}}",
    "Let me check. It's 18 degrees and sunny in Paris!",
    "Hello World HELLO world",
    "0 1 12 123 1234 12345 3.14159 -42 1e-7 1,000",
    "tab\tnewline\ncarriage\r\n  two  spaces   three ",
    "é ü ß ñ € £ 日本語 中文 😀 🎉",
    // NFKC-sensitive (ligature, full-width, circled digit) and NFC-sensitive
    // (A + combining ring) text: normalizers would change these.
    "ﬁ ＡＢＣ ① A\u{30A}",
    // Special-token names as text must stay text.
    "<pad> <epad> <tool_call> </tool_call> <tool_cancel> <sys> </sys>",
    "\\u00e9 \\n \\\" \\\\",
    "a\u{0}b\u{7f}",
];

/// A Hugging Face tokenizer with the stream special tokens and the text
/// stream encoding conventions (see the [crate docs](crate)).
///
/// Cheap to share behind an `Arc`: it is `Send + Sync` and every method
/// takes `&self`.
pub struct HfTokenizer {
    hf: Tokenizer,
    kind: VocabKind,
    /// What every id `0..vocab_size` decodes to.
    pieces: Vec<TokenPiece>,
    /// Every special token by name (first id wins).
    special_by_name: HashMap<String, u32>,
    /// Ids of [`special::ALL`], in that order.
    stream_specials: [u32; special::ALL.len()],
    appended: Vec<&'static str>,
    /// Single-byte token per byte value, for bytes outside valid UTF-8.
    byte_ids: [Option<u32>; 256],
    unk: Option<u32>,
    canonical: bool,
    probes_ok: bool,
    adjustments: Vec<String>,
    label: String,
    fingerprint: u64,
}

impl std::fmt::Debug for HfTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HfTokenizer")
            .field("label", &self.label)
            .field("kind", &self.kind)
            .field("vocab_size", &self.pieces.len())
            .field("canonical", &self.canonical)
            .field("appended", &self.appended)
            .finish_non_exhaustive()
    }
}

/// GPT-2's byte-to-unicode table, inverted: printable Latin-1 characters
/// stand for themselves, the other 68 bytes for `U+0100..`.
fn gpt2_unicode_to_byte() -> HashMap<char, u8> {
    let self_mapped = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    let mut map = HashMap::with_capacity(256);
    let mut next = 0x100u32;
    for b in 0..=255u8 {
        if self_mapped(b) {
            map.insert(char::from(b), b);
        } else {
            map.insert(char::from_u32(next).expect("valid code point"), b);
            next += 1;
        }
    }
    map
}

/// `<0xHH>` -> `0xHH`.
fn byte_fallback_token(s: &str) -> Option<u8> {
    let hex = s.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

/// Remove `Prepend` normalizers (recursively). Returns the new normalizer
/// and whether anything changed.
fn strip_prepend(n: NormalizerWrapper) -> (Option<NormalizerWrapper>, bool) {
    match n {
        NormalizerWrapper::Prepend(_) => (None, true),
        NormalizerWrapper::Sequence(seq) => {
            let mut changed = false;
            let kept: Vec<NormalizerWrapper> = seq
                .as_ref()
                .iter()
                .filter_map(|n| {
                    let (n, c) = strip_prepend(n.clone());
                    changed |= c;
                    n
                })
                .collect();
            if !changed {
                (Some(NormalizerWrapper::Sequence(seq)), false)
            } else if kept.is_empty() {
                (None, true)
            } else {
                (
                    Some(NormalizerWrapper::Sequence(normalizers::Sequence::new(
                        kept,
                    ))),
                    true,
                )
            }
        }
        other => (Some(other), false),
    }
}

/// Turn off implicit prefix spaces in pre-tokenizers (recursively).
fn no_prefix_space(pt: PreTokenizerWrapper, notes: &mut Vec<String>) -> PreTokenizerWrapper {
    match pt {
        PreTokenizerWrapper::Metaspace(mut ms) if ms.prepend_scheme != PrependScheme::Never => {
            ms.prepend_scheme = PrependScheme::Never;
            notes.push("Metaspace pre-tokenizer: prepend_scheme set to never".into());
            PreTokenizerWrapper::Metaspace(ms)
        }
        PreTokenizerWrapper::ByteLevel(mut bl) if bl.add_prefix_space => {
            bl.add_prefix_space = false;
            notes.push("ByteLevel pre-tokenizer: add_prefix_space set to false".into());
            PreTokenizerWrapper::ByteLevel(bl)
        }
        PreTokenizerWrapper::Sequence(seq) => {
            PreTokenizerWrapper::Sequence(pre_tokenizers::sequence::Sequence::new(
                seq.as_ref()
                    .iter()
                    .map(|p| no_prefix_space(p.clone(), notes))
                    .collect(),
            ))
        }
        other => other,
    }
}

#[derive(Default)]
struct DecoderScan {
    byte_level: bool,
    sentencepiece: bool,
    space: Option<char>,
    unsupported: Option<String>,
}

fn scan_decoder(d: &DecoderWrapper, scan: &mut DecoderScan) {
    match d {
        DecoderWrapper::ByteLevel(_) => scan.byte_level = true,
        DecoderWrapper::ByteFallback(_) => scan.sentencepiece = true,
        DecoderWrapper::Metaspace(ms) => {
            scan.sentencepiece = true;
            scan.space = Some(ms.get_replacement());
        }
        DecoderWrapper::Replace(r) if r.content == " " => {
            // The pattern has no accessor; read it from the serialized form.
            let pattern = serde_json::to_value(r)
                .ok()
                .and_then(|v| v["pattern"]["String"].as_str().map(str::to_owned));
            if let Some(p) = pattern {
                let mut chars = p.chars();
                if let (Some(c), None) = (chars.next(), chars.next()) {
                    scan.sentencepiece = true;
                    scan.space = Some(c);
                }
            }
        }
        DecoderWrapper::Sequence(seq) => {
            for d in seq.get_decoders() {
                scan_decoder(d, scan);
            }
        }
        DecoderWrapper::Fuse(_) | DecoderWrapper::Strip(_) | DecoderWrapper::Replace(_) => {}
        DecoderWrapper::BPE(_) => scan.unsupported = Some("BPE (end-of-word suffix)".into()),
        DecoderWrapper::WordPiece(_) => scan.unsupported = Some("WordPiece".into()),
        DecoderWrapper::CTC(_) => scan.unsupported = Some("CTC".into()),
    }
}

fn detect_kind(d: Option<&DecoderWrapper>) -> Result<VocabKind> {
    let Some(d) = d else {
        return Ok(VocabKind::Verbatim);
    };
    let mut scan = DecoderScan::default();
    scan_decoder(d, &mut scan);
    if let Some(u) = scan.unsupported {
        return Err(Error::UnsupportedDecoder(u));
    }
    if scan.byte_level {
        Ok(VocabKind::ByteLevel)
    } else if scan.sentencepiece {
        Ok(VocabKind::SentencePiece {
            space: scan.space.unwrap_or('▁'),
        })
    } else {
        Ok(VocabKind::Verbatim)
    }
}

/// The model's unknown-token id, if it has one.
fn model_unk(model: &ModelWrapper) -> Option<u32> {
    match model {
        ModelWrapper::BPE(b) => b.unk_token.as_deref().and_then(|u| model.token_to_id(u)),
        ModelWrapper::WordPiece(w) => model.token_to_id(&w.unk_token),
        ModelWrapper::WordLevel(w) => model.token_to_id(&w.unk_token),
        ModelWrapper::Unigram(_) => serde_json::to_value(model)
            .ok()
            .and_then(|v| v["unk_id"].as_u64())
            .and_then(|id| u32::try_from(id).ok()),
    }
}

fn fnv1a(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= u64::from(b);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

impl HfTokenizer {
    /// Load a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| Error::Hf(format!("reading {}: {e}", path.display())))?;
        Self::from_bytes(&bytes)
    }

    /// Load a `tokenizer.json` from its bytes.
    pub fn from_bytes(json: &[u8]) -> Result<Self> {
        Self::from_hf(Tokenizer::from_bytes(json).map_err(Error::hf)?)
    }

    /// Wrap an already loaded Hugging Face tokenizer.
    ///
    /// Applies the conventions of the crate docs (no truncation, padding or
    /// implicit prefix spaces; special-token names encoded as text), appends
    /// the missing stream special tokens, maps every id to its bytes and
    /// decides whether tokenization is canonical.
    pub fn from_hf(mut hf: Tokenizer) -> Result<Self> {
        let mut adjustments = Vec::new();
        if hf.get_truncation().is_some() {
            adjustments.push("truncation disabled".to_string());
        }
        hf.with_truncation(None).map_err(Error::hf)?;
        if hf.get_padding().is_some() {
            adjustments.push("padding disabled".to_string());
        }
        hf.with_padding(None);
        if let Some(n) = hf.get_normalizer().cloned() {
            let (n, changed) = strip_prepend(n);
            if changed {
                adjustments.push("Prepend normalizer removed".to_string());
                hf.with_normalizer(n);
            }
        }
        if let Some(pt) = hf.get_pre_tokenizer().cloned() {
            let before = adjustments.len();
            let pt = no_prefix_space(pt, &mut adjustments);
            if adjustments.len() > before {
                hf.with_pre_tokenizer(Some(pt));
            }
        }
        let kind = detect_kind(hf.get_decoder())?;

        // Stream special tokens: reuse existing ones, append the rest.
        let appended: Vec<&'static str> = special::ALL
            .iter()
            .copied()
            .filter(|name| hf.token_to_id(name).is_none())
            .collect();
        let tokens: Vec<AddedToken> = special::ALL
            .iter()
            .map(|name| AddedToken::from(*name, true))
            .collect();
        hf.add_special_tokens(&tokens);
        hf.set_encode_special_tokens(true);
        let mut stream_specials = [0u32; special::ALL.len()];
        for (slot, name) in stream_specials.iter_mut().zip(special::ALL) {
            *slot = hf
                .token_to_id(name)
                .ok_or_else(|| Error::Special(format!("{name} could not be added")))?;
        }
        for (i, a) in stream_specials.iter().enumerate() {
            if stream_specials[..i].contains(a) {
                return Err(Error::Special(format!(
                    "{} shares id {a} with another special token",
                    special::ALL[i]
                )));
            }
        }

        // Every id's bytes.
        let added = hf.get_added_tokens_decoder();
        let model_vocab = hf.get_model().get_vocab();
        let vocab_size = model_vocab
            .values()
            .chain(added.keys())
            .map(|&id| id as usize + 1)
            .max()
            .unwrap_or(0);
        if vocab_size == 0 || u32::try_from(vocab_size).is_err() {
            return Err(Error::BadVocab(format!("vocabulary size {vocab_size}")));
        }
        let unk = model_unk(hf.get_model());
        let byte_map = (kind == VocabKind::ByteLevel).then(gpt2_unicode_to_byte);
        let mut pieces = vec![TokenPiece::Bytes(Vec::new()); vocab_size];
        let mut unmappable = 0usize;
        for (text, &id) in &model_vocab {
            if added.contains_key(&id) {
                continue;
            }
            pieces[id as usize] = if Some(id) == unk {
                TokenPiece::Special(text.clone())
            } else {
                match kind {
                    VocabKind::ByteLevel => {
                        let map = byte_map.as_ref().expect("byte map built");
                        match text.chars().map(|c| map.get(&c).copied()).collect() {
                            Some(bytes) => TokenPiece::Bytes(bytes),
                            None => {
                                unmappable += 1;
                                TokenPiece::Bytes(Vec::new())
                            }
                        }
                    }
                    VocabKind::SentencePiece { space } => match byte_fallback_token(text) {
                        Some(b) => TokenPiece::Bytes(vec![b]),
                        None => TokenPiece::Bytes(text.replace(space, " ").into_bytes()),
                    },
                    VocabKind::Verbatim => TokenPiece::Bytes(text.as_bytes().to_vec()),
                }
            };
        }
        if unmappable > 0 {
            adjustments.push(format!(
                "{unmappable} byte-level token(s) with characters outside the byte table \
                 decode to nothing"
            ));
        }
        for (&id, token) in &added {
            pieces[id as usize] = if token.special || special::ALL.contains(&token.content.as_str())
            {
                TokenPiece::Special(token.content.clone())
            } else {
                TokenPiece::Bytes(token.content.as_bytes().to_vec())
            };
        }

        let mut special_by_name = HashMap::new();
        let mut byte_ids = [None; 256];
        let mut fingerprint = 0xcbf2_9ce4_8422_2325u64;
        for (id, piece) in pieces.iter().enumerate() {
            let id = id as u32;
            match piece {
                TokenPiece::Special(name) => {
                    special_by_name.entry(name.clone()).or_insert(id);
                    fnv1a(&mut fingerprint, &[1]);
                    fnv1a(&mut fingerprint, name.as_bytes());
                }
                TokenPiece::Bytes(b) => {
                    if let [byte] = b[..] {
                        if !added.contains_key(&id) && byte_ids[byte as usize].is_none() {
                            byte_ids[byte as usize] = Some(id);
                        }
                    }
                    fnv1a(&mut fingerprint, &[0]);
                    fnv1a(&mut fingerprint, b);
                }
            }
            fnv1a(&mut fingerprint, &[0xFF]);
        }

        let dropout =
            matches!(hf.get_model(), ModelWrapper::BPE(b) if b.dropout.is_some_and(|d| d > 0.0));
        let mut tok = Self {
            hf,
            kind,
            pieces,
            special_by_name,
            stream_specials,
            appended,
            byte_ids,
            unk,
            canonical: false,
            probes_ok: false,
            adjustments,
            label: "hf".to_string(),
            fingerprint,
        };
        tok.probes_ok = !dropout && tok.probes_round_trip();
        tok.canonical = tok.probes_ok;
        Ok(tok)
    }

    /// Whether every [`CANONICAL_PROBES`] string encodes deterministically to
    /// ordinary tokens that decode back to exactly the same bytes.
    fn probes_round_trip(&self) -> bool {
        CANONICAL_PROBES.iter().all(|p| {
            let ids = self.encode(p);
            ids == self.encode(p)
                && ids
                    .iter()
                    .all(|&id| matches!(self.piece(id), Some(TokenPiece::Bytes(_))))
                && self.decode_bytes(&ids) == p.as_bytes()
        })
    }

    /// Override whether tokenization is reported as canonical to llguidance
    /// (see the crate docs). Only force `true` for a tokenizer whose
    /// [`encode`](Self::encode) is exactly what the model was trained on.
    pub fn with_canonical(mut self, canonical: bool) -> Self {
        self.canonical = canonical;
        self
    }

    /// Set the label used in [`name`](Self::name) (default `"hf"`), e.g. the
    /// base model's name.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Identifies the tokenizer in sample metadata: `"{label}-{fingerprint}"`,
    /// the fingerprint being a hash of every token's bytes or special name.
    pub fn name(&self) -> String {
        format!("{}-{:016x}", self.label, self.fingerprint)
    }

    /// 64-bit FNV-1a hash of the vocabulary (every id's piece, in order).
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// The wrapped Hugging Face tokenizer (with the adjustments and the
    /// appended special tokens).
    pub fn hf(&self) -> &Tokenizer {
        &self.hf
    }

    /// How token strings map to bytes.
    pub fn kind(&self) -> VocabKind {
        self.kind
    }

    /// Whether tokenization is reported as canonical to llguidance.
    pub fn is_canonical(&self) -> bool {
        self.canonical
    }

    /// Whether the load-time canonicality check passed (independent of
    /// [`with_canonical`](Self::with_canonical)).
    pub fn probes_passed(&self) -> bool {
        self.probes_ok
    }

    /// What was changed in the loaded tokenizer to apply the crate's
    /// encoding conventions, in human-readable form.
    pub fn adjustments(&self) -> &[String] {
        &self.adjustments
    }

    /// The stream special tokens that were missing and got appended.
    pub fn appended_specials(&self) -> &[&'static str] {
        &self.appended
    }

    /// Number of ids (`0..vocab_size`), special tokens included.
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// Ids of the stream special tokens ([`special::ALL`]).
    pub fn special_ids(&self) -> SpecialIds {
        let [pad, epad, tool_call, tool_call_end, tool_result, tool_result_end, tool_cancel, sys, sys_end] =
            self.stream_specials;
        SpecialIds {
            pad,
            epad,
            tool_call,
            tool_call_end,
            tool_result,
            tool_result_end,
            tool_cancel,
            sys,
            sys_end,
        }
    }

    /// Ids of the stream special tokens, in [`special::ALL`] order.
    pub fn stream_special_ids(&self) -> [u32; special::ALL.len()] {
        self.stream_specials
    }

    /// Id of any special token by name (the stream specials and the base
    /// tokenizer's own, such as `<|endoftext|>`).
    pub fn special_id(&self, name: &str) -> Option<u32> {
        self.special_by_name.get(name).copied()
    }

    /// The model's unknown token, if it has one.
    pub fn unk_id(&self) -> Option<u32> {
        self.unk
    }

    /// What `id` decodes to, or `None` if out of range.
    pub fn piece(&self, id: u32) -> Option<&TokenPiece> {
        self.pieces.get(id as usize)
    }

    /// Encode text exactly as given (see the crate docs). Never adds or
    /// interprets special tokens.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        self.encode_str_into(text, &mut out);
        out
    }

    /// Encode arbitrary bytes: valid UTF-8 runs as text, other bytes as their
    /// single-byte tokens (the unknown token, or nothing, if the vocabulary
    /// has none for that byte).
    pub fn encode_bytes(&self, bytes: &[u8]) -> Vec<u32> {
        let mut out = Vec::new();
        for chunk in bytes.utf8_chunks() {
            self.encode_str_into(chunk.valid(), &mut out);
            for &b in chunk.invalid() {
                self.push_byte(b, &mut out);
            }
        }
        out
    }

    /// Encode one word of stream text: with a leading space unless it is the
    /// first word on its stream.
    pub fn encode_word(&self, word: &str, leading_space: bool) -> Vec<u32> {
        if leading_space {
            self.encode(&format!(" {word}"))
        } else {
            self.encode(word)
        }
    }

    /// Encode the words of one stream: the first as is, the rest with a
    /// leading space. Returns one token list per word.
    pub fn encode_words<'a>(&self, words: impl IntoIterator<Item = &'a str>) -> Vec<Vec<u32>> {
        words
            .into_iter()
            .enumerate()
            .map(|(i, w)| self.encode_word(w, i > 0))
            .collect()
    }

    /// The bytes `ids` decode to; special tokens are written by name and
    /// out-of-range ids as `<unk:ID>`.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            match self.piece(id) {
                Some(TokenPiece::Bytes(b)) => out.extend_from_slice(b),
                Some(TokenPiece::Special(name)) => out.extend_from_slice(name.as_bytes()),
                None => out.extend_from_slice(format!("<unk:{id}>").as_bytes()),
            }
        }
        out
    }

    /// Decode to text ([`decode_bytes`](Self::decode_bytes), then lossy
    /// UTF-8).
    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    /// A short display form of one token for grid dumps: `␣` for spaces,
    /// `\xHH` for bytes that are not printable on their own, names for
    /// special tokens.
    pub fn token_str(&self, id: u32) -> String {
        match self.piece(id) {
            Some(TokenPiece::Bytes(b)) => {
                let mut out = String::new();
                for chunk in b.utf8_chunks() {
                    for c in chunk.valid().chars() {
                        match c {
                            ' ' => out.push('␣'),
                            c if c.is_control() => out.push_str(&format!("\\x{:02x}", c as u32)),
                            c => out.push(c),
                        }
                    }
                    for byte in chunk.invalid() {
                        out.push_str(&format!("\\x{byte:02x}"));
                    }
                }
                out
            }
            Some(TokenPiece::Special(name)) => name.clone(),
            None => format!("<unk:{id}>"),
        }
    }

    /// Serialize the tokenizer (adjusted, with the appended special tokens)
    /// as `tokenizer.json`, so training and serving load identical ids.
    pub fn to_json(&self, pretty: bool) -> Result<String> {
        self.hf.to_string(pretty).map_err(Error::hf)
    }

    /// Write [`to_json`](Self::to_json) to `path`.
    pub fn save(&self, path: impl AsRef<Path>, pretty: bool) -> Result<()> {
        let path = path.as_ref();
        std::fs::write(path, self.to_json(pretty)?)
            .map_err(|e| Error::Hf(format!("writing {}: {e}", path.display())))
    }

    fn encode_str_into(&self, s: &str, out: &mut Vec<u32>) {
        if s.is_empty() {
            return;
        }
        match self.hf.encode_fast(s, false) {
            Ok(enc) => out.extend_from_slice(enc.get_ids()),
            // Not expected with truncation and padding off; degrade to bytes
            // rather than lose text.
            Err(_) => s.bytes().for_each(|b| self.push_byte(b, out)),
        }
    }

    fn push_byte(&self, b: u8, out: &mut Vec<u32>) {
        if let Some(id) = self.byte_ids[b as usize].or(self.unk) {
            out.push(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt2_table_is_a_bijection() {
        let m = gpt2_unicode_to_byte();
        assert_eq!(m.len(), 256);
        let mut bytes: Vec<u8> = m.values().copied().collect();
        bytes.sort_unstable();
        bytes.dedup();
        assert_eq!(bytes.len(), 256);
        assert_eq!(m[&'Ġ'], b' ');
        assert_eq!(m[&'Ċ'], b'\n');
        assert_eq!(m[&'a'], b'a');
        assert_eq!(m[&'\u{100}'], 0);
    }

    #[test]
    fn byte_fallback_names() {
        assert_eq!(byte_fallback_token("<0x41>"), Some(0x41));
        assert_eq!(byte_fallback_token("<0xE2>"), Some(0xE2));
        assert_eq!(byte_fallback_token("<0x4>"), None);
        assert_eq!(byte_fallback_token("<0xZZ>"), None);
        assert_eq!(byte_fallback_token("0x41"), None);
    }
}
