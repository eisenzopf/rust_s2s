//! The tokenizer the engine needs, and a byte-level one for development.
//!
//! The engine uses one tokenizer through two views: `s2s_tools::TokenizerInfo`
//! (grammar masks and incremental decoding) and `s2s_data::TextTokenizer`
//! (encoding injected text exactly as the training data does). The real
//! tokenizer (`s2s-tokenizer`) implements both; [`ByteTokenizer`] here wraps
//! `s2s_data::ByteTokenizer` (the id layout the synthetic training data uses)
//! with the `TokenizerInfo` view.

use s2s_core::streams::special;
use s2s_data::TextTokenizer;
use s2s_tools::{TokenPiece, TokenizerInfo};

/// Both tokenizer views, as the engine needs them.
pub trait Tokenizer: TokenizerInfo + TextTokenizer + Send + Sync + 'static {}

impl<T: TokenizerInfo + TextTokenizer + Send + Sync + 'static> Tokenizer for T {}

/// `s2s_data::ByteTokenizer` (special tokens, then 256 byte tokens) with the
/// `TokenizerInfo` view added.
#[derive(Clone, Copy, Debug, Default)]
pub struct ByteTokenizer;

impl TextTokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        s2s_data::ByteTokenizer.encode(text)
    }
    fn decode(&self, ids: &[u32]) -> String {
        s2s_data::ByteTokenizer.decode(ids)
    }
    fn special_id(&self, name: &str) -> Option<u32> {
        s2s_data::ByteTokenizer.special_id(name)
    }
    fn vocab_size(&self) -> u32 {
        s2s_data::ByteTokenizer.vocab_size()
    }
    fn name(&self) -> String {
        s2s_data::ByteTokenizer.name()
    }
}

impl TokenizerInfo for ByteTokenizer {
    fn vocab_size(&self) -> usize {
        s2s_data::ByteTokenizer::VOCAB as usize
    }

    fn token(&self, id: u32) -> TokenPiece {
        let n = s2s_data::ByteTokenizer::NUM_SPECIAL;
        if id < n {
            TokenPiece::Special(special::ALL[id as usize].to_string())
        } else {
            TokenPiece::Bytes(vec![(id - n) as u8])
        }
    }

    fn tokenize(&self, text: &[u8]) -> Vec<u32> {
        text.iter()
            .map(|&b| s2s_data::ByteTokenizer::NUM_SPECIAL + b as u32)
            .collect()
    }

    fn tokenize_is_canonical(&self) -> bool {
        true
    }

    fn special_token_id(&self, name: &str) -> Option<u32> {
        s2s_data::ByteTokenizer.special_id(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_agree() {
        let t = ByteTokenizer;
        let text = "héllo <x>";
        assert_eq!(
            TextTokenizer::encode(&t, text),
            TokenizerInfo::tokenize(&t, text.as_bytes())
        );
        for name in special::ALL {
            assert_eq!(
                TextTokenizer::special_id(&t, name),
                TokenizerInfo::special_token_id(&t, name)
            );
            let id = TextTokenizer::special_id(&t, name).unwrap();
            assert_eq!(t.token(id), TokenPiece::Special(name.to_string()));
        }
        assert_eq!(
            TextTokenizer::vocab_size(&t) as usize,
            TokenizerInfo::vocab_size(&t)
        );
    }
}
