//! The `s2s-eval` binary end to end.

use std::path::Path;
use std::process::Command;

use serde_json::Value;

fn run(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_s2s-eval"))
        .args(args)
        .output()
        .unwrap();
    (
        out.status.success(),
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
    )
}

fn write(dir: &Path, name: &str, body: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    p.to_str().unwrap().to_string()
}

#[test]
fn wer_text_and_json() {
    let dir = tempfile::tempdir().unwrap();
    let r = write(
        dir.path(),
        "refs.txt",
        "what is the weather in Paris\nIt's 18 degrees.\n",
    );
    let h = write(
        dir.path(),
        "hyps.txt",
        "what's the weather in in paris today\nit's eighteen degrees\n",
    );
    let (ok, out, err) = run(&["wer", "--ref", &r, "--hyp", &h]);
    assert!(ok, "{err}");
    // 4 errors over 6 + 3 words.
    assert!(out.contains("WER: 44.44% (N=9 S=1 D=1 I=2)"), "{out}");
    let (ok, out, _) = run(&["wer", "--ref", &r, "--hyp", &h, "--json", "--per-utt"]);
    assert!(ok);
    let v: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["corpus"]["words"]["insertions"], 2);
    assert_eq!(v["utterances"][1]["wer"], 0.0);
    // Digits kept: "18" vs "eighteen" is now a substitution.
    let (_, out, _) = run(&["wer", "--ref", &r, "--hyp", &h, "--numbers", "keep"]);
    assert!(out.contains("S=2"), "{out}");

    let short = write(dir.path(), "short.txt", "one line\n");
    let (ok, _, err) = run(&["wer", "--ref", &r, "--hyp", &short]);
    assert!(!ok);
    assert!(err.contains("lines"), "{err}");
}

#[test]
fn tools_report() {
    let dir = tempfile::tempdir().unwrap();
    let r = write(
        dir.path(),
        "ref.jsonl",
        concat!(
            r#"{"id": "a", "calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}"#,
            "\n",
            r#"{"id": "b", "calls": []}"#,
            "\n",
            r#"{"id": "c", "calls": [{"name": "set_timer", "arguments": {"seconds": 60}}]}"#,
            "\n"
        ),
    );
    let h = write(
        dir.path(),
        "hyp.jsonl",
        concat!(
            r#"{"id": "a", "calls": [{"raw": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}"}]}"#,
            "\n",
            r#"{"id": "b"}"#,
            "\n",
            r#"{"id": "c", "calls": [{"raw": "{\"name\": \"set_ti"}]}"#,
            "\n"
        ),
    );
    let (ok, out, err) = run(&["tools", "--ref", &r, "--hyp", &h, "--json"]);
    assert!(ok, "{err}");
    let v: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["tool_selection_accuracy"], 0.5);
    assert_eq!(v["argument_exact_match"], 0.5);
    assert_eq!(v["json_validity_rate"], 0.5);
    assert_eq!(v["no_call_precision"], 1.0);
    assert_eq!(v["no_call_recall"], 1.0);
    let (ok, out, _) = run(&["tools", "--ref", &r, "--hyp", &h]);
    assert!(ok);
    assert!(out.contains("JSON validity rate: 50.00%"), "{out}");

    let bad = write(
        dir.path(),
        "bad.jsonl",
        "{\"id\": \"a\"}\n{\"id\": \"x\"}\n{\"id\": \"c\"}\n",
    );
    let (ok, _, err) = run(&["tools", "--ref", &r, "--hyp", &bad]);
    assert!(!ok);
    assert!(err.contains("id"), "{err}");
}

#[test]
fn latency_report() {
    let dir = tempfile::tempdir().unwrap();
    let e = write(
        dir.path(),
        "events.jsonl",
        concat!(
            r#"{"t": 0.0, "event": "user_start"}"#,
            "\n",
            r#"{"t": 1.0, "event": "user_stop"}"#,
            "\n",
            r#"{"t": 1.25, "event": "model_start"}"#,
            "\n",
            r#"{"t": 1.5, "event": "tool_call", "call_id": "c_1"}"#,
            "\n",
            r#"{"t": 2.5, "event": "tool_result", "call_id": "c_1"}"#,
            "\n"
        ),
    );
    let (ok, out, err) = run(&["latency", "--events", &e, "--json"]);
    assert!(ok, "{err}");
    let v: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["turn_taking"]["stats"]["p50"], 0.25);
    assert_eq!(v["tool_round_trip"]["stats"]["mean"], 1.0);
    let (ok, out, _) = run(&["latency", "--events", &e]);
    assert!(ok);
    assert!(out.contains("turn-taking latency: n=1"), "{out}");
}

#[test]
fn audio_report() {
    let dir = tempfile::tempdir().unwrap();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let write_wav = |name: &str, f: &dyn Fn(usize) -> f64| {
        let p = dir.path().join(name);
        let mut w = hound::WavWriter::create(&p, spec).unwrap();
        for i in 0..12_000 {
            w.write_sample((f(i) * 16_000.0) as i16).unwrap();
        }
        w.finalize().unwrap();
        p.to_str().unwrap().to_string()
    };
    let tone = |i: usize| (i as f64 * 0.07).sin() + 0.5 * (i as f64 * 0.31).sin();
    let r = write_wav("ref.wav", &tone);
    let e = write_wav("est.wav", &|i| 0.5 * tone(i));
    let (ok, out, err) = run(&["audio", "--ref", &r, "--est", &e, "--json"]);
    assert!(ok, "{err}");
    let v: Value = serde_json::from_str(&out).unwrap();
    // Half amplitude: SI-SDR is limited only by 16-bit quantization; the
    // log-mel bands drop by log10(4) ≈ 0.602 where the tones have energy.
    assert!(v["si_sdr_db"].as_f64().unwrap() > 40.0, "{out}");
    assert_eq!(v["log_mel"]["samples"], 12_000);
    let (ok, out, _) = run(&["audio", "--ref", &r, "--est", &e]);
    assert!(ok);
    assert!(out.contains("SI-SDR:"), "{out}");
}
