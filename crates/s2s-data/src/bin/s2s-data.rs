//! `s2s-data`: build, inspect and validate training shards.
//!
//! ```text
//! s2s-data synth --out dialogues.s2s script.json [more.json ...]
//! s2s-data inspect dialogues.s2s [--sample 0 --dump --from 0 --to 40 --aligned]
//! s2s-data validate dialogues.s2s [more.s2s ...]
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use s2s_core::streams::{StreamLayout, FRAME_RATE};
use s2s_data::dump::{dump_grid, summary, DumpOptions};
use s2s_data::{
    synthesize, validate, ByteTokenizer, Error, Result, Script, ShardReader, ShardWriter,
    SynthConfig, TextTokenizer,
};

#[derive(Parser, Debug)]
#[command(
    name = "s2s-data",
    about = "Build, inspect and validate rust_s2s training shards"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Assemble synthetic dialogue scripts (JSON) into a shard.
    Synth {
        /// Output shard path (the index is written next to it as `<out>.idx`).
        #[arg(long)]
        out: PathBuf,
        /// Script files: each holds one script object or an array of them.
        #[arg(required = true)]
        scripts: Vec<PathBuf>,
        /// Text tokenizer (only `bytes` is built in).
        #[arg(long, default_value = "bytes")]
        tokenizer: String,
        /// Assistant text slots per frame.
        #[arg(long, default_value_t = 4)]
        text_slots: usize,
        /// Mimi codebooks per audio stream.
        #[arg(long, default_value_t = 8)]
        codebooks: usize,
        /// User transcript delay in frames (recorded; applied at assembly).
        #[arg(long, default_value_t = 6)]
        user_text_delay: usize,
        /// Acoustic codebook delay in frames (recorded; applied at assembly).
        #[arg(long, default_value_t = 1)]
        acoustic_delay: usize,
        /// Assistant text slots used by inner-monologue words.
        #[arg(long, default_value_t = 1)]
        monologue_slots: usize,
    },
    /// Print a shard summary, or one sample (optionally its frame grid).
    Inspect {
        /// Shard path.
        shard: PathBuf,
        /// Sample to show.
        #[arg(long)]
        sample: Option<usize>,
        /// Dump the frame grid of the sample.
        #[arg(long)]
        dump: bool,
        /// First frame of the dump.
        #[arg(long, default_value_t = 0)]
        from: usize,
        /// One past the last frame of the dump.
        #[arg(long)]
        to: Option<usize>,
        /// Show the aligned grid instead of the delayed one.
        #[arg(long)]
        aligned: bool,
    },
    /// Check shard integrity (CRCs, lengths) and every sample's invariants.
    Validate {
        /// Shard paths.
        #[arg(required = true)]
        shards: Vec<PathBuf>,
    },
}

fn tokenizer(name: &str) -> Result<Box<dyn TextTokenizer>> {
    match name {
        "bytes" => Ok(Box::new(ByteTokenizer)),
        other => Err(Error::Invalid(format!(
            "unknown tokenizer {other:?} (built in: bytes)"
        ))),
    }
}

fn load_scripts(path: &Path) -> Result<Vec<Script>> {
    let text = std::fs::read_to_string(path).map_err(|e| Error::Io {
        context: path.display().to_string(),
        source: e,
    })?;
    let v: serde_json::Value = serde_json::from_str(&text)?;
    Ok(match v {
        serde_json::Value::Array(items) => items
            .into_iter()
            .map(serde_json::from_value)
            .collect::<std::result::Result<_, _>>()?,
        other => vec![serde_json::from_value(other)?],
    })
}

fn run(cli: Cli) -> Result<bool> {
    match cli.cmd {
        Cmd::Synth {
            out,
            scripts,
            tokenizer: tok_name,
            text_slots,
            codebooks,
            user_text_delay,
            acoustic_delay,
            monologue_slots,
        } => {
            let tok = tokenizer(&tok_name)?;
            let cfg = SynthConfig {
                layout: StreamLayout {
                    text_slots,
                    codebooks,
                    user_text_delay,
                    acoustic_delay,
                },
                monologue_slots,
                ..SynthConfig::default()
            };
            let mut w = ShardWriter::create(&out)?;
            for path in &scripts {
                for (i, script) in load_scripts(path)?.iter().enumerate() {
                    let syn = synthesize(script, tok.as_ref(), &cfg).map_err(|e| {
                        Error::Script(format!("{} script {i}: {e}", path.display()))
                    })?;
                    for warn in &syn.warnings {
                        eprintln!("warning: {} script {i}: {warn}", path.display());
                    }
                    w.append(&syn.sample)?;
                }
            }
            let info = w.finish()?;
            println!(
                "wrote {} samples, {} frames ({:.1} s), {} bytes to {} (+ {})",
                info.num_samples,
                info.total_frames,
                info.total_frames as f64 / FRAME_RATE,
                info.data_len,
                info.path.display(),
                info.index_path.display()
            );
            Ok(true)
        }
        Cmd::Inspect {
            shard,
            sample,
            dump,
            from,
            to,
            aligned,
        } => {
            let r = ShardReader::open(&shard)?;
            match sample {
                None => {
                    let frames: u64 = r.entries().iter().map(|e| e.frames as u64).sum();
                    println!(
                        "{}: {} samples, {} frames ({:.1} s)",
                        shard.display(),
                        r.len(),
                        frames,
                        frames as f64 / FRAME_RATE
                    );
                    for i in 0..r.len() {
                        let m = r.meta(i)?;
                        let e = r.entries()[i];
                        println!(
                            "  [{i}] id={:?} source={:?} license={:?} frames={} bytes={}",
                            m.id, m.source, m.license, e.frames, e.len
                        );
                    }
                }
                Some(i) => {
                    let s = r.get(i)?;
                    println!("{}", summary(&s));
                    if dump {
                        let tok = (s.meta.tokenizer == "bytes").then_some(ByteTokenizer);
                        let opts = DumpOptions {
                            from,
                            to,
                            delayed: !aligned,
                            ..DumpOptions::default()
                        };
                        print!(
                            "{}",
                            dump_grid(&s, tok.as_ref().map(|t| t as &dyn TextTokenizer), &opts)
                        );
                    }
                }
            }
            Ok(true)
        }
        Cmd::Validate { shards } => {
            let mut ok = true;
            for path in &shards {
                let r = match ShardReader::open(path) {
                    Ok(r) => r,
                    Err(e) => {
                        println!("{}: FAILED to open: {e}", path.display());
                        ok = false;
                        continue;
                    }
                };
                let mut bad = 0usize;
                for i in 0..r.len() {
                    let issues = match r.get(i) {
                        Ok(s) => validate::find_issues(&s),
                        Err(e) => vec![e.to_string()],
                    };
                    for issue in &issues {
                        println!("{} sample {i}: {issue}", path.display());
                    }
                    bad += usize::from(!issues.is_empty());
                }
                println!(
                    "{}: {} samples, {}",
                    path.display(),
                    r.len(),
                    if bad == 0 {
                        "OK".to_string()
                    } else {
                        format!("{bad} invalid")
                    }
                );
                ok &= bad == 0;
            }
            Ok(ok)
        }
    }
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
