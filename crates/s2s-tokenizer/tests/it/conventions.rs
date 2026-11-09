use s2s_core::streams::special;
use s2s_data::{SpecialIds, TextTokenizer};
use s2s_tokenizer::{testing, Error, HfTokenizer, VocabKind};
use s2s_tools::{TokenPiece, TokenizerInfo};
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::utils::Lowercase;
use tokenizers::Tokenizer;

use crate::common::{all, byte_level, sentencepiece, TEXTS};

fn is_special(tok: &HfTokenizer, id: u32) -> bool {
    matches!(tok.piece(id), Some(TokenPiece::Special(_)))
}

#[test]
fn stream_specials_exist_reused_or_appended() {
    let t = byte_level();
    // `<|endoftext|>` and `<pad>` were trainer specials (ids 0 and 1).
    assert_eq!(t.special_id("<|endoftext|>"), Some(0));
    assert_eq!(t.special_id(special::PAD), Some(1));
    let appended: Vec<&str> = special::ALL[1..].to_vec();
    assert_eq!(t.appended_specials(), &appended[..]);
    let ids = t.stream_special_ids();
    for (i, (&id, name)) in ids.iter().zip(special::ALL).enumerate() {
        assert_eq!(t.piece(id), Some(&TokenPiece::Special(name.to_string())));
        assert_eq!(t.special_id(name), Some(id));
        if i > 0 {
            // Appended after the trained vocabulary, in `ALL` order.
            assert_eq!(id, ids[1] + i as u32 - 1);
        }
    }
    assert_eq!(ids[8] as usize + 1, t.vocab_size());
    let sp = t.special_ids();
    assert_eq!(
        [
            sp.pad,
            sp.epad,
            sp.tool_call,
            sp.tool_call_end,
            sp.tool_result,
            sp.tool_result_end,
            sp.tool_cancel,
            sp.sys,
            sp.sys_end
        ],
        ids
    );

    let s = sentencepiece();
    assert_eq!(s.appended_specials(), &special::ALL[..]);
    assert_eq!(s.special_id("<unk>"), Some(0));
    assert_eq!(s.unk_id(), Some(0));
    assert_eq!(s.special_id("</s>"), Some(2));
}

#[test]
fn both_trait_views_agree() {
    for (label, t) in all() {
        let info: &dyn TokenizerInfo = &*t;
        let text: &dyn TextTokenizer = &*t;
        assert_eq!(info.vocab_size(), text.vocab_size() as usize, "{label}");
        for text_str in TEXTS {
            let a = info.tokenize(text_str.as_bytes());
            let b = text.encode(text_str);
            assert_eq!(a, b, "{label}: {text_str:?}");
            assert_eq!(info.count_tokens(text_str), b.len());
            assert!(b.iter().all(|&id| (id as usize) < info.vocab_size()));
            // TextTokenizer::decode is the concatenation of TokenizerInfo pieces.
            let mut bytes = Vec::new();
            for &id in &a {
                match info.token(id) {
                    TokenPiece::Bytes(x) => bytes.extend_from_slice(&x),
                    TokenPiece::Special(n) => bytes.extend_from_slice(n.as_bytes()),
                }
            }
            assert_eq!(text.decode(&b).as_bytes(), &bytes[..], "{label}");
            assert_eq!(text.decode(&b), *text_str, "{label}: exact round trip");
        }
        for name in special::ALL
            .iter()
            .chain(&["<|endoftext|>", "<unk>", "<nope>"])
        {
            let id = text.special_id(name);
            assert_eq!(info.special_token_id(name), id, "{label}: {name}");
            // The trait's default scan over `token()` finds the same id.
            let scanned = (0..info.vocab_size() as u32)
                .find(|&i| info.token(i) == TokenPiece::Special(name.to_string()));
            assert_eq!(scanned, id, "{label}: {name}");
        }
        // s2s-data's resolver sees the same ids as the inherent accessor.
        assert_eq!(SpecialIds::from_tokenizer(text).unwrap(), t.special_ids());
        let sp = t.special_ids();
        assert_eq!(
            text.decode(&[sp.tool_call, sp.tool_call_end]),
            "<tool_call></tool_call>"
        );
        assert_eq!(text.name(), t.name());
        assert!(info.tokenize_is_canonical(), "{label}");
    }
}

#[test]
fn special_names_in_text_are_plain_text() {
    for (label, t) in all() {
        for name in special::ALL
            .iter()
            .chain(&["<|endoftext|>", "<unk>", "<s>"])
        {
            let ids = t.encode(name);
            assert!(!ids.iter().any(|&i| is_special(&t, i)), "{label}: {name}");
            assert_eq!(t.decode(&ids), *name);
        }
    }
}

#[test]
fn no_implicit_prefix_space_and_adjustments() {
    for (label, t) in all() {
        assert_eq!(t.decode(&t.encode("a")), "a", "{label}");
        assert_eq!(t.decode(&t.encode(" a")), " a", "{label}");
        assert_ne!(t.encode("Paris"), t.encode(" Paris"), "{label}");
        assert!(t.probes_passed() && t.is_canonical(), "{label}");
    }
    let s = sentencepiece();
    assert_eq!(s.kind(), VocabKind::SentencePiece { space: '▁' });
    assert_eq!(
        s.adjustments(),
        ["Metaspace pre-tokenizer: prepend_scheme set to never"]
    );
    let b = byte_level();
    assert_eq!(b.kind(), VocabKind::ByteLevel);
    assert!(b.adjustments().is_empty(), "{:?}", b.adjustments());
}

#[test]
fn word_encodings_concatenate_to_the_whole_string() {
    let sentences = [
        "what's the weather in Paris?",
        "Let me check. It's 18 degrees and sunny in Paris.",
        "Set a timer for 90 seconds, please!",
        "Café in Zürich costs 4.50 euros.",
    ];
    for (label, t) in all() {
        for s in sentences {
            let words: Vec<&str> = s.split(' ').collect();
            let per_word = t.encode_words(words.iter().copied());
            assert_eq!(per_word[0], t.encode_word(words[0], false));
            assert_eq!(per_word[1], t.encode(&format!(" {}", words[1])));
            let flat: Vec<u32> = per_word.concat();
            assert_eq!(flat, t.encode(s), "{label}: {s:?}");
        }
    }
}

#[test]
fn bytes_and_unknown_characters() {
    // SentencePiece style: characters outside the vocabulary fall back to
    // `<0xHH>` byte tokens.
    let s = sentencepiece();
    let euro = s.encode("€");
    assert_eq!(euro.len(), 3);
    assert_eq!(
        euro.iter()
            .map(|&id| s.hf().id_to_token(id).unwrap())
            .collect::<Vec<_>>(),
        ["<0xE2>", "<0x82>", "<0xAC>"]
    );
    assert_eq!(s.decode(&euro), "€");
    // Byte-level: every byte is representable.
    let b = byte_level();
    assert_eq!(b.decode(&b.encode("€ 😀")), "€ 😀");

    for (label, t) in all() {
        // Invalid UTF-8 (a truncated `€`, a stray continuation byte, 0xFF)
        // becomes single-byte tokens and decodes back byte-exactly.
        let raw: &[u8] = b"ok \xE2\x82 x\x80\xFF";
        let ids = t.encode_bytes(raw);
        assert_eq!(t.decode_bytes(&ids), raw, "{label}");
        assert_eq!(&ids[..t.encode("ok ").len()], &t.encode("ok ")[..]);
        // Valid UTF-8 goes through the tokenizer unchanged.
        assert_eq!(t.encode_bytes("ok €".as_bytes()), t.encode("ok €"));
        // Out-of-range ids decode visibly.
        let big = t.vocab_size() as u32 + 5;
        assert_eq!(t.decode(&[big]), format!("<unk:{big}>"));
        assert_eq!(t.token_str(big), format!("<unk:{big}>"));
    }
}

#[test]
fn reloading_is_idempotent_and_saved_json_round_trips() {
    let t = byte_level();
    let again = HfTokenizer::from_hf(t.hf().clone()).unwrap();
    assert!(again.appended_specials().is_empty());
    assert_eq!(again.stream_special_ids(), t.stream_special_ids());
    assert_eq!(again.fingerprint(), t.fingerprint());

    let dir = std::env::temp_dir().join(format!("s2s-tokenizer-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tokenizer.json");
    t.save(&path, false).unwrap();
    let loaded = HfTokenizer::from_file(&path).unwrap().with_label("test");
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(loaded.appended_specials().is_empty());
    assert_eq!(loaded.vocab_size(), t.vocab_size());
    assert_eq!(loaded.fingerprint(), t.fingerprint());
    assert_eq!(loaded.name(), format!("test-{:016x}", t.fingerprint()));
    for text in TEXTS {
        assert_eq!(loaded.encode(text), t.encode(text));
    }
    assert!(matches!(
        HfTokenizer::from_file(dir.join("missing.json")),
        Err(Error::Hf(_))
    ));
    assert!(matches!(
        HfTokenizer::from_bytes(b"{not json"),
        Err(Error::Hf(_))
    ));
}

#[test]
fn normalizers_that_change_text_are_not_canonical() {
    let mut hf: Tokenizer = byte_level().hf().clone();
    hf.with_normalizer(Some(Lowercase));
    let t = HfTokenizer::from_hf(hf).unwrap();
    assert!(!t.probes_passed());
    assert!(!t.is_canonical());
    assert!(!TokenizerInfo::tokenize_is_canonical(&t));
    assert_eq!(t.decode(&t.encode("Paris")), "paris");
    // The override is explicit.
    let forced = t.with_canonical(true);
    assert!(forced.is_canonical() && !forced.probes_passed());
}

#[test]
fn unsupported_decoders_are_rejected() {
    let mut hf = Tokenizer::new(WordPiece::default());
    hf.with_decoder(Some(tokenizers::decoders::wordpiece::WordPiece::default()));
    assert_eq!(
        HfTokenizer::from_hf(hf).unwrap_err(),
        Error::UnsupportedDecoder("WordPiece".into())
    );
}

#[test]
fn token_display_and_names() {
    let t = byte_level();
    let sp = t.special_ids();
    assert_eq!(t.token_str(sp.epad), "<epad>");
    let space = t.encode(" ");
    assert_eq!(space.len(), 1);
    assert_eq!(t.token_str(space[0]), "␣");
    let nl = t.encode("\n");
    assert_eq!(t.token_str(nl[0]), "\\x0a");
    assert_eq!(TextTokenizer::token_str(&*t, sp.pad), "<pad>");
    assert!(t.name().starts_with("hf-"));
    assert_eq!(t.name().len(), "hf-".len() + 16);
    // A different vocabulary has a different fingerprint.
    let other = testing::byte_level_bpe(300).unwrap();
    assert_ne!(other.fingerprint(), t.fingerprint());
    assert!(format!("{t:?}").contains("ByteLevel"));
}
