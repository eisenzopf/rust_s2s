//! Transcribe a WAV file with a Kyutai DSM speech-to-text checkpoint:
//! Mimi encode (`s2s-mimi`) → ASR driver (`s2s-model`) → text
//! (`s2s-tokenizer`). The LM runs on GPU 0 when built with feature `cuda`
//! (unless `--cpu`), else on the CPU.
//!
//! ```text
//! cargo run --release -p s2s-infer [--features cuda] --example transcribe -- \
//!     --lm stt-1b-en_fr-candle/model.safetensors \
//!     --config config-stt-en_fr-hf.toml \
//!     --mimi tokenizer-e351c8d8-checkpoint125.safetensors \
//!     --tokenizer stt-tokenizer.json  speech.wav
//! ```
//!
//! `--config` is the `moshi-server` TOML from
//! `kyutai-labs/delayed-streams-modeling`; `--tokenizer` is the model's
//! SentencePiece tokenizer converted with `scripts/spm-to-tokenizer-json.py`.
//! The input must be mono 24 kHz. Prints each word with its start time,
//! then the whole transcript on a line starting with `TRANSCRIPT:`.
//!
//! `scripts/e2e-speech.sh` uses this to check what a speech-to-speech
//! model actually said.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use s2s_mimi::audio::read_wav_mono_24k;
use s2s_mimi::{Mimi, MimiConfig};
use s2s_model::kyutai::{AsrConfig, AsrMsg, AsrState, LmConfig, LmModel};
use s2s_tokenizer::HfTokenizer;

const USAGE: &str =
    "usage: transcribe --lm LM.safetensors --config STT.toml --mimi MIMI.safetensors \
--tokenizer tokenizer.json [--cpu] IN.wav";

/// Trailing silence fed after the audio so the delayed text stream (and the
/// last word) comes out.
const FLUSH_SECONDS: f32 = 2.0;

fn load_lm(cfg: &LmConfig, path: &PathBuf, cpu: bool) -> Result<LmModel, String> {
    #[cfg(feature = "cuda")]
    if !cpu {
        return s2s_gpu::kyutai::load_lm(cfg, path, Default::default())
            .map_err(|e| format!("loading {} on the GPU: {e}", path.display()));
    }
    let _ = cpu;
    LmModel::load_safetensors(cfg, path).map_err(|e| format!("loading {}: {e}", path.display()))
}

fn run() -> Result<(), String> {
    let (mut lm, mut cfg, mut mimi, mut tok, mut wav, mut cpu) =
        (None, None, None, None, None, false);
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut value = || {
            it.next()
                .ok_or_else(|| format!("{a} needs a value\n{USAGE}"))
        };
        match a.as_str() {
            "--lm" => lm = Some(PathBuf::from(value()?)),
            "--config" => cfg = Some(PathBuf::from(value()?)),
            "--mimi" => mimi = Some(PathBuf::from(value()?)),
            "--tokenizer" => tok = Some(PathBuf::from(value()?)),
            "--cpu" => cpu = true,
            "-h" | "--help" => return Err(USAGE.into()),
            s if !s.starts_with("--") && wav.is_none() => wav = Some(PathBuf::from(s)),
            other => return Err(format!("unexpected argument {other:?}\n{USAGE}")),
        }
    }
    let (Some(lm_path), Some(cfg_path), Some(mimi_path), Some(tok_path), Some(wav)) =
        (lm, cfg, mimi, tok, wav)
    else {
        return Err(USAGE.into());
    };

    let toml = std::fs::read_to_string(&cfg_path)
        .map_err(|e| format!("reading {}: {e}", cfg_path.display()))?;
    let (asr_cfg, lm_cfg) = AsrConfig::from_moshi_server_toml(&toml)
        .map_err(|e| format!("{}: {e}", cfg_path.display()))?;
    let n_q = lm_cfg.audio_codebooks;
    let t0 = Instant::now();
    let lm = load_lm(&lm_cfg, &lm_path, cpu)?;
    let mut asr = AsrState::from_config(lm, &asr_cfg).map_err(|e| e.to_string())?;
    let mimi = Mimi::load(&mimi_path, MimiConfig::v0_1().with_n_q(n_q))
        .map_err(|e| format!("loading Mimi {}: {e}", mimi_path.display()))?;
    let tok = HfTokenizer::from_file(&tok_path)
        .map_err(|e| format!("loading {}: {e}", tok_path.display()))?;
    eprintln!("models loaded in {:.1} s", t0.elapsed().as_secs_f64());

    let mut pcm = read_wav_mono_24k(&wav).map_err(|e| format!("{}: {e}", wav.display()))?;
    let audio_s = pcm.len() as f32 / 24_000.0;
    pcm.extend(std::iter::repeat_n(
        0.0,
        (FLUSH_SECONDS * 24_000.0) as usize,
    ));
    let t0 = Instant::now();
    let codes = mimi.encode(&pcm).map_err(|e| e.to_string())?;
    let mut all = Vec::new();
    for f in 0..codes.frames() {
        for msg in asr
            .step_tokens(&codes.frame(f))
            .map_err(|e| e.to_string())?
        {
            if let AsrMsg::Word {
                tokens, start_time, ..
            } = msg
            {
                println!("{start_time:7.2}s {}", tok.decode(&tokens).trim());
                all.extend(tokens);
            }
        }
    }
    eprintln!(
        "{audio_s:.1} s of audio transcribed in {:.1} s",
        t0.elapsed().as_secs_f64()
    );
    println!("TRANSCRIPT: {}", tok.decode(&all).trim());
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
