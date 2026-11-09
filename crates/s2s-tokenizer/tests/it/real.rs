//! A real base-LLM tokenizer, when one is available locally.
//!
//! ```text
//! S2S_TOKENIZER_JSON=/path/to/tokenizer.json cargo test -p s2s-tokenizer -- --ignored --nocapture
//! ```

use std::sync::Arc;

use s2s_core::streams::special;
use s2s_core::SplitMix64;
use s2s_data::{synthesize, Script, SpecialIds, SynthConfig, TextTokenizer};
use s2s_kernels::reference::sampling::{apply_token_bitmask, sample, SamplingParams};
use s2s_tokenizer::{HfTokenizer, CANONICAL_PROBES};
use s2s_tools::{
    GrammarConfig, ParseEvent, SpanState, TokenizerInfo, ToolCallConstraint, ToolCallGrammar,
    ToolCallParser, ToolRegistry,
};

use crate::grammar::{tools, walk};

#[test]
#[ignore = "needs a real tokenizer.json: set S2S_TOKENIZER_JSON"]
fn real_tokenizer_from_env() {
    let path = std::env::var("S2S_TOKENIZER_JSON")
        .expect("set S2S_TOKENIZER_JSON to a tokenizer.json path");
    let tok = Arc::new(HfTokenizer::from_file(&path).unwrap());
    println!(
        "{path}: kind {:?}, vocab {}, canonical {}, appended {:?}, adjustments {:?}",
        tok.kind(),
        tok.vocab_size(),
        tok.is_canonical(),
        tok.appended_specials(),
        tok.adjustments()
    );

    // Specials and the two trait views.
    let ids = SpecialIds::from_tokenizer(&*tok).unwrap();
    assert_eq!(ids, tok.special_ids());
    for name in special::ALL {
        assert_eq!(
            TokenizerInfo::special_token_id(&*tok, name),
            TextTokenizer::special_id(&*tok, name)
        );
    }
    assert_eq!(
        TokenizerInfo::vocab_size(&*tok),
        TextTokenizer::vocab_size(&*tok) as usize
    );
    for p in CANONICAL_PROBES {
        assert_eq!(
            TokenizerInfo::tokenize(&*tok, p.as_bytes()),
            TextTokenizer::encode(&*tok, p)
        );
        if tok.is_canonical() {
            assert_eq!(TextTokenizer::decode(&*tok, &tok.encode(p)), *p);
        }
    }

    // Grammar: rendered calls are admissible; sampled calls are valid.
    let reg = Arc::new(ToolRegistry::from_specs(tools()).unwrap());
    let g = Arc::new(
        ToolCallGrammar::new(
            &reg,
            tok.clone(),
            GrammarConfig {
                allow_cancel: false,
                ..GrammarConfig::default()
            },
        )
        .unwrap(),
    );
    walk(
        &g,
        &tok,
        r#"{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}"#,
    );
    let mut rng = SplitMix64::new(3);
    let params = SamplingParams {
        temperature: 1.0,
        top_k: None,
        top_p: None,
    };
    for _ in 0..10 {
        let mut c = ToolCallConstraint::new(g.clone());
        c.observe(g.special_ids().tool_call).unwrap();
        let mut span = vec![g.special_ids().tool_call];
        loop {
            let mask = c.mask().unwrap().unwrap();
            let mut logits: Vec<f32> = (0..g.vocab_size())
                .map(|_| rng.next_f64() as f32 * 8.0)
                .collect();
            apply_token_bitmask(&mut logits, &mask);
            let t = sample(&logits, &params, rng.next_f64()).unwrap() as u32;
            span.push(t);
            if c.observe(t).unwrap() != Some(SpanState::Open) {
                break;
            }
            assert!(span.len() < 2000, "span too long");
        }
        let mut p = ToolCallParser::new(reg.clone());
        let mut ev = p.push_str(&TextTokenizer::decode(&*tok, &span));
        ev.extend(p.finish());
        assert!(matches!(ev.last(), Some(ParseEvent::ToolCall(_))), "{ev:?}");
    }

    // The synthesizer works with it.
    let script = Script::from_json(
        r#"{"turns": [
            {"speaker": "user", "text": "hello there", "start": {"at": 0.2}},
            {"speaker": "assistant", "text": "Hi! How can I help?", "start": {"after": 0.3}}
        ]}"#,
    )
    .unwrap();
    let syn = synthesize(&script, &*tok, &SynthConfig::default()).unwrap();
    let text = TextTokenizer::decode(&*tok, &syn.sample.assistant_text());
    assert_eq!(text.replace(special::EPAD, ""), "Hi! How can I help?");
}
