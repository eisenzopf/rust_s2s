use std::process::Command;

use crate::common::fixture;

fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_s2s-data"))
}

#[test]
fn synth_inspect_validate_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let shard = dir.path().join("d.s2s");
    // Two scripts in one shard: the fixture file twice (object form) plus an
    // array file.
    let arr = dir.path().join("arr.json");
    let one = std::fs::read_to_string(fixture("weather.json")).unwrap();
    std::fs::write(&arr, format!("[{one}, {one}]")).unwrap();

    let out = cli()
        .arg("synth")
        .arg("--out")
        .arg(&shard)
        .arg(fixture("weather.json"))
        .arg(&arr)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("wrote 3 samples"));

    let out = cli().arg("inspect").arg(&shard).output().unwrap();
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("3 samples") && text.contains("weather-001"));

    let out = cli()
        .args([
            "inspect", "--sample", "0", "--dump", "--from", "20", "--to", "45",
        ])
        .arg(&shard)
        .output()
        .unwrap();
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    for needle in [
        "frame",
        "Tu (+6)",
        "Tm0",
        "Am1 (+1)",
        "<ep>",
        "<tc>",
        "injection before frame 0",
    ] {
        assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
    }

    let out = cli().arg("validate").arg(&shard).output().unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("OK"));

    // Corrupt one byte of the first record: validate must fail.
    let mut bytes = std::fs::read(&shard).unwrap();
    bytes[200] ^= 0xFF;
    std::fs::write(&shard, &bytes).unwrap();
    let out = cli().arg("validate").arg(&shard).output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("CRC"));

    // Missing files and bad scripts are reported, not panics.
    let out = cli()
        .arg("validate")
        .arg(dir.path().join("nope.s2s"))
        .output()
        .unwrap();
    assert!(!out.status.success());
    std::fs::write(
        &arr,
        "{\"turns\": [{\"speaker\": \"user\", \"text\": \"\"}]}",
    )
    .unwrap();
    let out = cli()
        .arg("synth")
        .arg("--out")
        .arg(dir.path().join("e.s2s"))
        .arg(&arr)
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("invalid script"));
}
