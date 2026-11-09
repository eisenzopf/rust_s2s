//! Small tokenizers trained on an embedded corpus, for tests and bring-up
//! without downloading a real `tokenizer.json`.
//!
//! - [`byte_level_bpe`]: GPT-2-style byte-level BPE with regex
//!   pre-tokenization (the Llama 3 / Qwen family's design).
//! - [`sentencepiece_bpe`]: SentencePiece-style BPE with `▁` for spaces and
//!   `<0xHH>` byte fallback (the Llama 2 / Mistral v1 family's design).
//!
//! Both are trained with the `tokenizers` crate's own BPE trainer, take
//! milliseconds, and are deterministic.

use std::collections::HashSet;

use serde_json::{json, Value};
use tokenizers::models::bpe::{BpeTrainer, BPE};
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::{AddedToken, Tokenizer};

use crate::{Error, HfTokenizer, Result};

/// Conversational and tool-calling text in the formats the model's text
/// streams use (`s2s_tools::render_tool_call`'s default JSON style).
pub const SAMPLE_CORPUS: &[&str] = &[
    "what's the weather in Paris?",
    "Let me check. It's 18 degrees and sunny in Paris.",
    "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"unit\": \"celsius\"}}",
    "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Berlin\"}}",
    "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\", \"unit\": \"fahrenheit\"}}",
    "{\"name\": \"set_timer\", \"arguments\": {\"seconds\": 90, \"label\": \"tea\", \"loud\": true}}",
    "{\"name\": \"set_timer\", \"arguments\": {\"seconds\": 300, \"loud\": false}}",
    "{\"name\": \"search\", \"arguments\": {\"query\": \"running shoes\", \"max_price\": 120.5, \"tags\": [\"sale\", \"new\"]}}",
    "{\"name\": \"search\", \"arguments\": {\"query\": \"lamp\", \"where\": {\"near\": true, \"lat\": 48.85}}}",
    "{\"call_id\": \"c_1\", \"name\": \"get_weather\", \"status\": \"ok\", \"result\": {\"temp\": 18, \"conditions\": \"sunny\"}}",
    "{\"call_id\": \"c_2\", \"name\": \"set_timer\", \"status\": \"cancelled\"}",
    "{\"call_id\": \"c_3\", \"name\": \"search\", \"status\": \"error\", \"error\": \"timeout\"}",
    "Set a timer for ten minutes please.",
    "Sure, a timer for ten minutes is running now.",
    "Can you find me some running shoes under one hundred dollars?",
    "I found three pairs on sale. The first one costs eighty nine dollars.",
    "Actually, never mind, cancel that.",
    "Okay, I cancelled the search. Anything else?",
    "How long does it take to drive from Berlin to Munich?",
    "It takes about six hours by car, depending on traffic.",
    "Remind me to call my mother tomorrow at nine in the morning.",
    "Done. I will remind you tomorrow at nine.",
    "mm-hmm yeah right okay sure",
    "The quick brown fox jumps over the lazy dog.",
    "You are a helpful voice assistant. Answer briefly and naturally.",
    "## get_weather\nCurrent weather for a city.\nParameters: {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}}",
    "## set_timer\nStart a timer.\nParameters: {\"type\": \"object\", \"properties\": {\"seconds\": {\"type\": \"integer\"}}}",
    "Numbers like 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 42, 100, 2024 and 3.14 appear in answers.",
    "Is it going to rain in London this weekend? It might rain on Saturday.",
    "Please turn the lights off in the living room.",
    "What time is it in New York right now? It's a quarter past three.",
    "Thanks, that's all for now. You're welcome, have a nice day!",
    "Café, naïve, Zürich, São Paulo and Kraków are city names with accents.",
    "true false null [] {} \"\" : , . ? ! ' - _",
];

fn hf_err(e: impl std::fmt::Display) -> Error {
    Error::Hf(e.to_string())
}

/// Train a byte-level BPE (`ByteLevel` pre-tokenizer with the GPT-2 regex,
/// all 256 byte tokens, `ByteLevel` decoder) on `corpus`.
///
/// `base_specials` become the base tokenizer's own special tokens (ids
/// `0..`), e.g. `["<|endoftext|>"]`; stream specials named here are
/// therefore reused rather than appended by [`HfTokenizer`]. The returned
/// tokenizer is the raw Hugging Face one.
pub fn train_byte_level_bpe(
    corpus: &[&str],
    vocab_size: usize,
    base_specials: &[&str],
) -> Result<Tokenizer> {
    let mut tok = Tokenizer::new(BPE::default());
    tok.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
    tok.with_decoder(Some(ByteLevel::default()));
    let mut trainer: TrainerWrapper = BpeTrainer::builder()
        .vocab_size(vocab_size)
        .min_frequency(2)
        .show_progress(false)
        .initial_alphabet(ByteLevel::alphabet().into_iter().collect::<HashSet<char>>())
        .special_tokens(
            base_specials
                .iter()
                .map(|s| AddedToken::from(*s, true))
                .collect(),
        )
        .build()
        .into();
    tok.train(&mut trainer, corpus.iter().copied())
        .map_err(hf_err)?;
    Ok(tok)
}

/// [`train_byte_level_bpe`] on [`SAMPLE_CORPUS`] with base specials
/// `<|endoftext|>` and `<pad>` (so `<pad>` is reused and the other stream
/// specials are appended), wrapped in an [`HfTokenizer`].
pub fn byte_level_bpe(vocab_size: usize) -> Result<HfTokenizer> {
    HfTokenizer::from_hf(train_byte_level_bpe(
        SAMPLE_CORPUS,
        vocab_size,
        &["<|endoftext|>", "<pad>"],
    )?)
}

/// Train a SentencePiece-style BPE on `corpus`: `Metaspace` pre-tokenizer
/// (`▁` for spaces, prefix space on the first word, as in the original
/// models), `<unk>`/`<s>`/`</s>` specials, `<0x00>`..`<0xFF>` byte-fallback
/// tokens appended to the model vocabulary, and the Llama 2 decoder
/// (`Replace("▁", " ")`, `ByteFallback`, `Fuse`, `Strip`).
///
/// Characters outside the trained alphabet encode as byte-fallback tokens.
pub fn train_sentencepiece_bpe(corpus: &[&str], vocab_size: usize) -> Result<Tokenizer> {
    let mut tok = Tokenizer::new(BPE::default());
    tok.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::First, true)));
    let mut trainer: TrainerWrapper = BpeTrainer::builder()
        .vocab_size(vocab_size)
        .min_frequency(2)
        .show_progress(false)
        .special_tokens(vec![
            AddedToken::from("<unk>", true),
            AddedToken::from("<s>", true),
            AddedToken::from("</s>", true),
        ])
        .build()
        .into();
    tok.train(&mut trainer, corpus.iter().copied())
        .map_err(hf_err)?;

    let mut v: Value =
        serde_json::from_str(&tok.to_string(false).map_err(hf_err)?).map_err(hf_err)?;
    let model = &mut v["model"];
    model["byte_fallback"] = json!(true);
    model["unk_token"] = json!("<unk>");
    let vocab = model["vocab"]
        .as_object_mut()
        .ok_or_else(|| Error::Hf("BPE model without vocab".into()))?;
    let mut next = vocab.values().filter_map(Value::as_u64).max().unwrap_or(0) + 1;
    for b in 0..=255u8 {
        let name = format!("<0x{b:02X}>");
        if !vocab.contains_key(&name) {
            vocab.insert(name, json!(next));
            next += 1;
        }
    }
    v["decoder"] = json!({
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0}
        ]
    });
    Tokenizer::from_bytes(v.to_string().as_bytes()).map_err(hf_err)
}

/// [`train_sentencepiece_bpe`] on [`SAMPLE_CORPUS`], wrapped in an
/// [`HfTokenizer`] (which turns the prefix space off).
pub fn sentencepiece_bpe(vocab_size: usize) -> Result<HfTokenizer> {
    HfTokenizer::from_hf(train_sentencepiece_bpe(SAMPLE_CORPUS, vocab_size)?)
}
