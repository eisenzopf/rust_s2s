//! The text tokenizer interface the pipeline needs, and a tiny deterministic
//! byte-level tokenizer for tests and offline tooling.
//!
//! The real tokenizer (the base LLM's, via Hugging Face `tokenizers`) will live
//! in `s2s-tokenizer` and implement [`TextTokenizer`]; nothing here downloads
//! anything.

use s2s_core::streams::special;

use crate::{Error, Result};

/// Encodes and decodes text, and knows the ids of the special tokens
/// ([`s2s_core::streams::special`]).
pub trait TextTokenizer {
    /// Encode a string into token ids. Must not add BOS/EOS or other specials.
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Decode ids back into text. Special tokens decode to their names.
    fn decode(&self, ids: &[u32]) -> String;

    /// The id of a special token by name (e.g. `"<pad>"`), if present.
    fn special_id(&self, name: &str) -> Option<u32>;

    /// Total text vocabulary size, specials included. Every id is below this.
    fn vocab_size(&self) -> u32;

    /// A name identifying the tokenizer, recorded in sample metadata.
    fn name(&self) -> String {
        "unknown".to_string()
    }

    /// A short display form of one token, for grid dumps. Defaults to
    /// decoding it alone.
    fn token_str(&self, id: u32) -> String {
        self.decode(&[id])
    }
}

/// Resolved ids of every special text token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpecialIds {
    /// `<pad>`.
    pub pad: u32,
    /// `<epad>`.
    pub epad: u32,
    /// `<tool_call>`.
    pub tool_call: u32,
    /// `</tool_call>`.
    pub tool_call_end: u32,
    /// `<tool_result>`.
    pub tool_result: u32,
    /// `</tool_result>`.
    pub tool_result_end: u32,
    /// `<tool_cancel>`.
    pub tool_cancel: u32,
    /// `<sys>`.
    pub sys: u32,
    /// `</sys>`.
    pub sys_end: u32,
}

impl SpecialIds {
    /// Look every special token up in `tok`. Errors if one is missing.
    pub fn from_tokenizer(tok: &dyn TextTokenizer) -> Result<Self> {
        let get = |name: &str| {
            tok.special_id(name)
                .ok_or_else(|| Error::Invalid(format!("tokenizer lacks special token {name}")))
        };
        Ok(Self {
            pad: get(special::PAD)?,
            epad: get(special::EPAD)?,
            tool_call: get(special::TOOL_CALL)?,
            tool_call_end: get(special::TOOL_CALL_END)?,
            tool_result: get(special::TOOL_RESULT)?,
            tool_result_end: get(special::TOOL_RESULT_END)?,
            tool_cancel: get(special::TOOL_CANCEL)?,
            sys: get(special::SYS)?,
            sys_end: get(special::SYS_END)?,
        })
    }
}

/// A deterministic byte-level tokenizer: ids `0..9` are the special tokens in
/// [`special::ALL`] order, and id `9 + b` is the UTF-8 byte `b`. Vocabulary
/// size 265.
///
/// Every string round-trips exactly, which makes it ideal for tests (tool-call
/// JSON can be decoded back and compared) and for offline tooling before the
/// real tokenizer is wired in. It is far less compact than a BPE tokenizer,
/// so words occupy more text positions than they would in production.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    /// Number of special tokens before the byte range.
    pub const NUM_SPECIAL: u32 = special::ALL.len() as u32;
    /// Vocabulary size.
    pub const VOCAB: u32 = Self::NUM_SPECIAL + 256;
}

impl TextTokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| Self::NUM_SPECIAL + b as u32).collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        let mut bytes = Vec::new();
        let flush = |bytes: &mut Vec<u8>, out: &mut String| {
            out.push_str(&String::from_utf8_lossy(bytes));
            bytes.clear();
        };
        for &id in ids {
            if id < Self::NUM_SPECIAL {
                flush(&mut bytes, &mut out);
                out.push_str(special::ALL[id as usize]);
            } else if id < Self::VOCAB {
                bytes.push((id - Self::NUM_SPECIAL) as u8);
            } else {
                flush(&mut bytes, &mut out);
                out.push_str(&format!("<unk:{id}>"));
            }
        }
        flush(&mut bytes, &mut out);
        out
    }

    fn special_id(&self, name: &str) -> Option<u32> {
        special::ALL
            .iter()
            .position(|&s| s == name)
            .map(|i| i as u32)
    }

    fn vocab_size(&self) -> u32 {
        Self::VOCAB
    }

    fn name(&self) -> String {
        "bytes".to_string()
    }

    fn token_str(&self, id: u32) -> String {
        if (Self::NUM_SPECIAL..Self::VOCAB).contains(&id) {
            let b = (id - Self::NUM_SPECIAL) as u8;
            if b.is_ascii_graphic() {
                return (b as char).to_string();
            }
            if b == b' ' {
                return "␣".to_string();
            }
            return format!("x{b:02x}");
        }
        self.decode(&[id])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_tokenizer_round_trip_and_specials() {
        let t = ByteTokenizer;
        let s = "{\"city\": \"Paris\", \"é\": 1}";
        assert_eq!(t.decode(&t.encode(s)), s);
        let sp = SpecialIds::from_tokenizer(&t).unwrap();
        assert_eq!(sp.pad, 0);
        assert_eq!(sp.epad, 1);
        assert_eq!(sp.sys_end, 8);
        assert_eq!(
            t.decode(&[sp.tool_call, 9 + b'x' as u32, sp.tool_call_end]),
            "<tool_call>x</tool_call>"
        );
        assert!(t.encode(s).iter().all(|&i| i < t.vocab_size()));
        assert_eq!(t.token_str(9 + b' ' as u32), "␣");
        assert_eq!(t.token_str(sp.epad), "<epad>");
    }
}
