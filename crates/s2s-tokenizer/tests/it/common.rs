use std::sync::{Arc, OnceLock};

use s2s_tokenizer::{testing, HfTokenizer};

/// Byte-level BPE trained on the sample corpus (shared across tests).
pub fn byte_level() -> Arc<HfTokenizer> {
    static T: OnceLock<Arc<HfTokenizer>> = OnceLock::new();
    T.get_or_init(|| Arc::new(testing::byte_level_bpe(700).unwrap()))
        .clone()
}

/// SentencePiece-style BPE with byte fallback (shared across tests).
pub fn sentencepiece() -> Arc<HfTokenizer> {
    static T: OnceLock<Arc<HfTokenizer>> = OnceLock::new();
    T.get_or_init(|| Arc::new(testing::sentencepiece_bpe(500).unwrap()))
        .clone()
}

/// Both test tokenizers, labelled.
pub fn all() -> Vec<(&'static str, Arc<HfTokenizer>)> {
    vec![
        ("byte-level", byte_level()),
        ("sentencepiece", sentencepiece()),
    ]
}

/// Texts exercising spacing, JSON, digits, Unicode and special names.
pub const TEXTS: &[&str] = &[
    "",
    "Paris",
    " Paris",
    "what's the weather in Paris?",
    "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Zürich\"}}",
    "Set a timer for 90 seconds, then 3.5 minutes.",
    "  double  spaces\tand\nnewlines\r\n",
    "日本語 😀 €100",
    "<tool_call> <pad> </sys>",
];
