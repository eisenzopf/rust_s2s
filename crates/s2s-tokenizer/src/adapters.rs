//! [`HfTokenizer`] as the workspace's two tokenizer traits.
//!
//! Both delegate to the same inherent methods, so they agree by construction:
//!
//! | | `s2s_tools::TokenizerInfo` | `s2s_data::TextTokenizer` |
//! |---|---|---|
//! | text → ids | `tokenize(&[u8])` = [`HfTokenizer::encode_bytes`] | `encode(&str)` = [`HfTokenizer::encode`] |
//! | specials | `special_token_id` = [`HfTokenizer::special_id`] | `special_id` = [`HfTokenizer::special_id`] |
//! | vocabulary | `vocab_size() -> usize` | `vocab_size() -> u32` |
//! | ids → text | `token(id)` (raw bytes / special name) | `decode` = [`HfTokenizer::decode`] |
//!
//! For valid UTF-8, `encode_bytes(s.as_bytes()) == encode(s)`.

use s2s_tools::{TokenPiece, TokenizerInfo};

use crate::HfTokenizer;

impl TokenizerInfo for HfTokenizer {
    fn vocab_size(&self) -> usize {
        HfTokenizer::vocab_size(self)
    }

    fn token(&self, id: u32) -> TokenPiece {
        match self.piece(id) {
            Some(p) => p.clone(),
            None => panic!(
                "token id {id} out of range (vocabulary {})",
                HfTokenizer::vocab_size(self)
            ),
        }
    }

    fn tokenize(&self, text: &[u8]) -> Vec<u32> {
        self.encode_bytes(text)
    }

    fn tokenize_is_canonical(&self) -> bool {
        self.is_canonical()
    }

    fn special_token_id(&self, name: &str) -> Option<u32> {
        HfTokenizer::special_id(self, name)
    }
}

impl s2s_data::TextTokenizer for HfTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        HfTokenizer::encode(self, text)
    }

    fn decode(&self, ids: &[u32]) -> String {
        HfTokenizer::decode(self, ids)
    }

    fn special_id(&self, name: &str) -> Option<u32> {
        HfTokenizer::special_id(self, name)
    }

    fn vocab_size(&self) -> u32 {
        HfTokenizer::vocab_size(self) as u32
    }

    fn name(&self) -> String {
        HfTokenizer::name(self)
    }

    fn token_str(&self, id: u32) -> String {
        HfTokenizer::token_str(self, id)
    }
}
