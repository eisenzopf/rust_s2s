//! Offline Mimi tokenization (plan §9.3).
//!
//! ```text
//! mimi-tokenize encode --weights tokenizer.safetensors [--config mimi.json] in.wav out.codes.json
//! mimi-tokenize decode --weights tokenizer.safetensors [--config mimi.json] in.codes.json out.wav
//! ```
//!
//! Without `--config`, Kyutai's default Mimi config (8 codebooks) is used.

use s2s_mimi::{tokenize, Mimi, MimiConfig};

fn usage() -> ! {
    eprintln!(
        "usage: mimi-tokenize <encode|decode> --weights W.safetensors [--config C.json] <input> <output>"
    );
    std::process::exit(2)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| usage());
    let (mut weights, mut config, mut pos) = (None, None, Vec::new());
    while let Some(a) = args.next() {
        match a.as_str() {
            "--weights" => weights = Some(args.next().unwrap_or_else(|| usage())),
            "--config" => config = Some(args.next().unwrap_or_else(|| usage())),
            _ => pos.push(a),
        }
    }
    let (Some(weights), [input, output]) = (weights, pos.as_slice()) else {
        usage()
    };
    let cfg = match config {
        Some(p) => MimiConfig::from_json(&std::fs::read_to_string(p)?)?,
        None => MimiConfig::v0_1(),
    };
    let mimi = Mimi::load(&weights, cfg)?;
    match cmd.as_str() {
        "encode" => {
            let f = tokenize::encode_wav(&mimi, input, output)?;
            eprintln!(
                "{} frames × {} codebooks -> {output}",
                f.frames.len(),
                f.n_q
            );
        }
        "decode" => {
            let n = tokenize::decode_to_wav(&mimi, input, output)?;
            eprintln!("{n} frames -> {output}");
        }
        _ => usage(),
    }
    Ok(())
}
