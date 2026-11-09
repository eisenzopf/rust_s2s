//! `s2s-eval`: evaluation metrics from the command line.
//!
//! ```text
//! s2s-eval wer --ref refs.txt --hyp hyps.txt [--per-utt] [--json]
//! s2s-eval tools --ref ref.jsonl --hyp hyp.jsonl [--json]
//! s2s-eval latency --events events.jsonl [--json]
//! s2s-eval audio --ref reference.wav --est decoded.wav [--json]
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand, ValueEnum};
use s2s_eval::{
    log_mel_distance, si_sdr, tool_round_trips, turn_taking, AsrScorer, LatencyStats, Normalizer,
    NumberHandling, PredictedCall, StftConfig, ToolEvaluator, ToolExample, TurnEvent,
};
use serde_json::{json, Value};

#[derive(Parser)]
#[command(name = "s2s-eval", version, about = "Evaluation metrics for rust_s2s")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Word and character error rates of line-aligned transcripts.
    Wer(WerArgs),
    /// Tool-calling accuracy of line-aligned JSONL examples.
    Tools(ToolsArgs),
    /// Turn-taking latency and tool round trips from a JSONL event log.
    Latency(LatencyArgs),
    /// SI-SDR and log-mel distance of a decoded WAV against its reference.
    Audio(AudioArgs),
}

#[derive(Clone, Copy, ValueEnum)]
enum Numbers {
    /// Spell numbers out in words.
    Words,
    /// Leave digits alone.
    Keep,
}

#[derive(Args)]
struct WerArgs {
    /// Reference transcripts, one utterance per line.
    #[arg(long = "ref")]
    reference: PathBuf,
    /// Hypothesis transcripts, one utterance per line (same order).
    #[arg(long = "hyp")]
    hypothesis: PathBuf,
    /// Keep case.
    #[arg(long)]
    keep_case: bool,
    /// Keep punctuation.
    #[arg(long)]
    keep_punctuation: bool,
    /// Number handling.
    #[arg(long, value_enum, default_value = "words")]
    numbers: Numbers,
    /// Ignore spaces in CER.
    #[arg(long)]
    cer_no_spaces: bool,
    /// Also print every utterance.
    #[arg(long)]
    per_utt: bool,
    /// Print JSON instead of text.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct ToolsArgs {
    /// Reference examples (`{"id": …, "calls": [{"name": …, "arguments": {…}}]}` per line).
    #[arg(long = "ref")]
    reference: PathBuf,
    /// Predicted examples (same format; a call may be `{"raw": "<span text>"}`).
    #[arg(long = "hyp")]
    hypothesis: PathBuf,
    /// Print JSON instead of text.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct LatencyArgs {
    /// Events (`{"t": seconds, "event": "user_stop", "session": …}` per line).
    #[arg(long)]
    events: PathBuf,
    /// Print JSON instead of text.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct AudioArgs {
    /// Reference WAV (mono, or the first channel is used).
    #[arg(long = "ref")]
    reference: PathBuf,
    /// Estimate WAV (same sample rate).
    #[arg(long = "est")]
    estimate: PathBuf,
    /// FFT size.
    #[arg(long, default_value_t = 1024)]
    n_fft: usize,
    /// Hop size.
    #[arg(long, default_value_t = 256)]
    hop: usize,
    /// Mel bands.
    #[arg(long, default_value_t = 80)]
    n_mels: usize,
    /// Print JSON instead of text.
    #[arg(long)]
    json: bool,
}

type Result<T> = std::result::Result<T, String>;

fn read(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn lines(path: &Path) -> Result<Vec<String>> {
    Ok(read(path)?.lines().map(str::to_string).collect())
}

fn jsonl(path: &Path) -> Result<Vec<Value>> {
    read(path)?
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            serde_json::from_str(l).map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))
        })
        .collect()
}

fn pct(x: Option<f64>) -> String {
    x.map_or("n/a".into(), |v| format!("{:.2}%", 100.0 * v))
}

fn print_json(v: &impl serde::Serialize) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(v).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn wer(a: &WerArgs) -> Result<()> {
    let refs = lines(&a.reference)?;
    let hyps = lines(&a.hypothesis)?;
    if refs.len() != hyps.len() {
        return Err(format!(
            "{} has {} lines but {} has {}",
            a.reference.display(),
            refs.len(),
            a.hypothesis.display(),
            hyps.len()
        ));
    }
    let mut scorer = AsrScorer::new(Normalizer {
        lowercase: !a.keep_case,
        remove_punctuation: !a.keep_punctuation,
        keep_apostrophes: true,
        numbers: match a.numbers {
            Numbers::Words => NumberHandling::Words,
            Numbers::Keep => NumberHandling::Keep,
        },
    });
    scorer.cer_spaces = !a.cer_no_spaces;
    for (r, h) in refs.iter().zip(&hyps) {
        scorer.add(r, h);
    }
    let sum = scorer.summary();
    if a.json {
        return print_json(&json!({
            "corpus": sum,
            "utterances": if a.per_utt { json!(scorer.utterances()) } else { Value::Null },
        }));
    }
    if a.per_utt {
        for (i, u) in scorer.utterances().iter().enumerate() {
            println!(
                "{:>5}  WER {:>8}  S {} D {} I {}  REF: {}  HYP: {}",
                i + 1,
                pct(u.wer),
                u.words.substitutions,
                u.words.deletions,
                u.words.insertions,
                u.reference,
                u.hypothesis
            );
        }
    }
    let w = sum.words;
    let c = sum.chars;
    println!("utterances: {}", sum.utterances);
    println!(
        "WER: {} (N={} S={} D={} I={})",
        pct(sum.wer),
        w.reference_len(),
        w.substitutions,
        w.deletions,
        w.insertions
    );
    println!(
        "CER: {} (N={} S={} D={} I={})",
        pct(sum.cer),
        c.reference_len(),
        c.substitutions,
        c.deletions,
        c.insertions
    );
    println!("mean utterance WER: {}", pct(sum.mean_utterance_wer));
    println!("sentence error rate: {}", pct(sum.sentence_error_rate));
    Ok(())
}

fn example(v: &Value, predicted: bool, where_: &str) -> Result<ToolExample> {
    let calls = match v.get("calls") {
        Some(Value::Array(c)) => c.as_slice(),
        None | Some(Value::Null) => &[],
        Some(_) => return Err(format!("{where_}: `calls` must be an array")),
    };
    let id = v.get("id").map(|i| match i {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    });
    let mut ex = ToolExample {
        id,
        ..ToolExample::default()
    };
    if predicted {
        ex.predicted = calls.iter().map(PredictedCall::from_json).collect();
    } else {
        for c in calls {
            match PredictedCall::from_json(c) {
                PredictedCall::Valid(call) => ex.reference.push(call),
                PredictedCall::Invalid { raw } => {
                    return Err(format!("{where_}: invalid reference call {raw}"))
                }
            }
        }
    }
    Ok(ex)
}

fn tools(a: &ToolsArgs) -> Result<()> {
    let refs = jsonl(&a.reference)?;
    let hyps = jsonl(&a.hypothesis)?;
    if refs.len() != hyps.len() {
        return Err(format!(
            "{} has {} examples but {} has {}",
            a.reference.display(),
            refs.len(),
            a.hypothesis.display(),
            hyps.len()
        ));
    }
    let mut ev = ToolEvaluator::new();
    for (i, (r, h)) in refs.iter().zip(&hyps).enumerate() {
        let mut ex = example(
            r,
            false,
            &format!("{} example {}", a.reference.display(), i + 1),
        )?;
        let hx = example(
            h,
            true,
            &format!("{} example {}", a.hypothesis.display(), i + 1),
        )?;
        if let (Some(ri), Some(hi)) = (&ex.id, &hx.id) {
            if ri != hi {
                return Err(format!("example {}: id {ri:?} vs {hi:?}", i + 1));
            }
        }
        ex.predicted = hx.predicted;
        ev.add(&ex);
    }
    let r = ev.report();
    if a.json {
        return print_json(&r);
    }
    let c = r.counts;
    println!(
        "examples: {}  reference calls: {}  predicted calls: {}",
        c.examples, c.reference_calls, c.predicted_calls
    );
    println!(
        "tool selection accuracy: {}",
        pct(r.tool_selection_accuracy)
    );
    println!("argument exact match: {}", pct(r.argument_exact_match));
    println!(
        "argument exact match | right tool: {}",
        pct(r.argument_exact_match_given_name)
    );
    println!(
        "per-argument accuracy / precision: {} / {}",
        pct(r.per_argument_accuracy),
        pct(r.per_argument_precision)
    );
    println!("JSON validity rate: {}", pct(r.json_validity_rate));
    println!(
        "no-call precision / recall: {} / {}",
        pct(r.no_call_precision),
        pct(r.no_call_recall)
    );
    println!("example exact match: {}", pct(r.example_exact_match));
    println!(
        "missed calls: {}  spurious calls: {}",
        c.missed_calls, c.spurious_calls
    );
    for (name, s) in &r.per_argument {
        println!("  {name}: {}/{}", s.matched, s.total);
    }
    Ok(())
}

fn stats_line(label: &str, s: Option<LatencyStats>) {
    match s {
        Some(s) => println!(
            "{label}: n={} mean={:.3}s p50={:.3}s p90={:.3}s p99={:.3}s min={:.3}s max={:.3}s",
            s.count, s.mean, s.p50, s.p90, s.p99, s.min, s.max
        ),
        None => println!("{label}: no samples"),
    }
}

fn latency(a: &LatencyArgs) -> Result<()> {
    let events: Vec<TurnEvent> = jsonl(&a.events)?
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            serde_json::from_value(v)
                .map_err(|e| format!("{} event {}: {e}", a.events.display(), i + 1))
        })
        .collect::<Result<_>>()?;
    let tt = turn_taking(&events);
    let rt = tool_round_trips(&events);
    if a.json {
        return print_json(&json!({
            "turn_taking": {"stats": tt.stats(), "unanswered": tt.unanswered, "overlaps": tt.overlaps},
            "tool_round_trip": {"stats": rt.stats(), "unmatched_calls": rt.unmatched_calls,
                                "orphan_results": rt.orphan_results},
        }));
    }
    stats_line("turn-taking latency", tt.stats());
    println!(
        "unanswered turns: {}  overlaps: {}",
        tt.unanswered, tt.overlaps
    );
    stats_line("tool round trip", rt.stats());
    println!(
        "unmatched calls: {}  orphan results: {}",
        rt.unmatched_calls, rt.orphan_results
    );
    Ok(())
}

/// First channel of a WAV file as f32 in [-1, 1], and its sample rate.
fn read_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let err = |e: hound::Error| format!("{}: {e}", path.display());
    let mut r = hound::WavReader::open(path).map_err(err)?;
    let spec = r.spec();
    let ch = usize::from(spec.channels.max(1));
    let all: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<std::result::Result<_, _>>(),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<std::result::Result<_, _>>()
        }
    }
    .map_err(err)?;
    Ok((all.into_iter().step_by(ch).collect(), spec.sample_rate))
}

fn audio(a: &AudioArgs) -> Result<()> {
    let (r, sr) = read_wav(&a.reference)?;
    let (e, sr2) = read_wav(&a.estimate)?;
    if sr != sr2 {
        return Err(format!("sample rates differ: {sr} vs {sr2}"));
    }
    let len = r.len().min(e.len());
    let sdr = si_sdr(&r[..len], &e[..len]).map_err(|x| x.to_string())?;
    let cfg = StftConfig {
        sample_rate: sr,
        n_fft: a.n_fft,
        hop: a.hop,
        n_mels: a.n_mels,
        ..StftConfig::default()
    };
    let lm = log_mel_distance(&r, &e, &cfg).map_err(|x| x.to_string())?;
    if a.json {
        return print_json(&json!({"si_sdr_db": sdr, "log_mel": lm, "stft": cfg}));
    }
    println!("samples compared: {len} at {sr} Hz");
    println!("SI-SDR: {sdr:.2} dB");
    println!(
        "log-mel distance: L1 {:.4}  RMSE {:.4} (log10 units, {} frames)",
        lm.l1, lm.rmse, lm.frames
    );
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let res = match &cli.cmd {
        Cmd::Wer(a) => wer(a),
        Cmd::Tools(a) => tools(a),
        Cmd::Latency(a) => latency(a),
        Cmd::Audio(a) => audio(a),
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("s2s-eval: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use s2s_eval::ToolCallRecord;

    #[test]
    fn reference_examples_parse() {
        let ex = example(
            &json!({"id": 3, "calls": [{"name": "f", "arguments": {"x": 1}}]}),
            false,
            "t",
        )
        .unwrap();
        assert_eq!(ex.id.as_deref(), Some("3"));
        assert_eq!(ex.reference, [ToolCallRecord::new("f", json!({"x": 1}))]);
        assert!(example(&json!({"calls": [{"raw": "{"}]}), false, "t").is_err());
        assert!(example(&json!({"calls": 5}), true, "t").is_err());
        let ex = example(&json!({}), true, "t").unwrap();
        assert!(ex.predicted.is_empty());
    }
}
