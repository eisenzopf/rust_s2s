//! `s2s-kyutai-server`: serve a Kyutai checkpoint (Moshi full duplex, or a
//! DSM speech-to-text model) through `s2s-server`, with the browser client
//! at `/` (plan milestone M4).
//!
//! ```text
//! s2s-kyutai-server --lm model.safetensors --mimi tokenizer-e351c8d8-checkpoint125.safetensors \
//!     --tokenizer tokenizer.json [--stt-config config-stt-en_fr-hf.toml] \
//!     [--bind 127.0.0.1:8080] [--cpu] [--seed N] [--greedy]
//! ```
//!
//! - `--lm`: the LM checkpoint (`kyutai/moshiko-candle-bf16/model.safetensors`,
//!   or `kyutai/stt-*-candle/model.safetensors` with `--stt-config`).
//! - `--mimi`: Kyutai's Mimi checkpoint.
//! - `--tokenizer`: the text tokenizer as `tokenizer.json`
//!   (`scripts/spm-to-tokenizer-json.py` converts Kyutai's SentencePiece
//!   model).
//! - `--stt-config`: serve speech-to-text with this `moshi-server` TOML
//!   (from `kyutai-labs/delayed-streams-modeling`) instead of Moshi.
//! - `--cpu`: run the LM on the CPU (far slower than real time for Moshi);
//!   by default it runs on GPU 0 when built with feature `cuda`.
//!
//! Logging is controlled by `RUST_LOG` (default `info`).

use std::io::IsTerminal;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use s2s_infer::{FrameCodec, KyutaiEngine, KyutaiEngineConfig, KyutaiModel, MimiCodec};
use s2s_mimi::{Mimi, MimiConfig};
use s2s_model::kyutai::{AsrConfig, LmConfig, LmModel, MultistreamConfig};
use s2s_server::ServerConfig;
use s2s_tokenizer::HfTokenizer;
use tracing_subscriber::EnvFilter;

const USAGE: &str = "usage: s2s-kyutai-server --lm LM.safetensors --mimi MIMI.safetensors \
--tokenizer tokenizer.json [--stt-config STT.toml] [--bind ADDR] [--cpu] [--seed N] [--greedy]";

#[derive(Debug, Default)]
struct Args {
    lm: Option<PathBuf>,
    mimi: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    stt_config: Option<PathBuf>,
    bind: Option<String>,
    cpu: bool,
    seed: u64,
    greedy: bool,
}

fn parse_args(mut it: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut a = Args::default();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--lm" => a.lm = Some(value("--lm")?.into()),
            "--mimi" => a.mimi = Some(value("--mimi")?.into()),
            "--tokenizer" => a.tokenizer = Some(value("--tokenizer")?.into()),
            "--stt-config" => a.stt_config = Some(value("--stt-config")?.into()),
            "--bind" => a.bind = Some(value("--bind")?),
            "--seed" => {
                let v = value("--seed")?;
                a.seed = v.parse().map_err(|e| format!("bad --seed {v:?}: {e}"))?;
            }
            "--cpu" => a.cpu = true,
            "--greedy" => a.greedy = true,
            "-h" | "--help" => return Err(USAGE.to_owned()),
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        }
    }
    if a.lm.is_none() || a.mimi.is_none() || a.tokenizer.is_none() {
        return Err(format!(
            "--lm, --mimi and --tokenizer are required\n{USAGE}"
        ));
    }
    Ok(a)
}

fn load_lm(cfg: &LmConfig, path: &PathBuf, cpu: bool) -> Result<LmModel, String> {
    #[cfg(feature = "cuda")]
    if !cpu {
        tracing::info!("loading the LM on GPU 0 (bf16)");
        return s2s_gpu::kyutai::load_lm(cfg, path, Default::default())
            .map_err(|e| format!("loading {} on the GPU: {e}", path.display()));
    }
    if !cpu {
        tracing::warn!("built without feature `cuda`: the LM runs on the CPU");
    }
    tracing::info!("loading the LM on the CPU (f32)");
    LmModel::load_safetensors(cfg, path).map_err(|e| format!("loading {}: {e}", path.display()))
}

fn build_engine(a: &Args) -> Result<KyutaiEngine, String> {
    let (lm_path, mimi_path, tok_path) = (
        a.lm.as_ref().ok_or(USAGE)?,
        a.mimi.as_ref().ok_or(USAGE)?,
        a.tokenizer.as_ref().ok_or(USAGE)?,
    );
    let t0 = Instant::now();
    let (model, n_q) = match &a.stt_config {
        Some(toml) => {
            let text = std::fs::read_to_string(toml)
                .map_err(|e| format!("reading {}: {e}", toml.display()))?;
            let (asr, lm_cfg) = AsrConfig::from_moshi_server_toml(&text)
                .map_err(|e| format!("{}: {e}", toml.display()))?;
            let n_q = lm_cfg.audio_codebooks;
            let lm = load_lm(&lm_cfg, lm_path, a.cpu)?;
            (KyutaiModel::Stt { lm, asr }, n_q)
        }
        None => {
            let lm_cfg = LmConfig::v0_1_streaming(8);
            let driver = MultistreamConfig::v0_1();
            let n_q = driver.input_audio_codebooks;
            let lm = load_lm(&lm_cfg, lm_path, a.cpu)?;
            (KyutaiModel::Moshi { lm, driver }, n_q)
        }
    };
    tracing::info!("LM ready in {:.1} s", t0.elapsed().as_secs_f64());
    let mimi = Mimi::load(mimi_path, MimiConfig::v0_1().with_n_q(n_q))
        .map_err(|e| format!("loading Mimi {}: {e}", mimi_path.display()))?;
    let mimi = Arc::new(mimi);
    let tok = HfTokenizer::from_file(tok_path)
        .map_err(|e| format!("loading {}: {e}", tok_path.display()))?;
    let mut cfg = KyutaiEngineConfig {
        seed: a.seed,
        ..Default::default()
    };
    // CPU Mimi next to the GPU LM: overlap decoding with the next step.
    cfg.pipeline_decode = true;
    if a.greedy {
        cfg.text_sampling.temperature = 0.0;
        cfg.audio_sampling.temperature = 0.0;
    }
    KyutaiEngine::new(
        model,
        Arc::new(tok),
        move || Box::new(MimiCodec::new(mimi.clone())) as Box<dyn FrameCodec>,
        cfg,
    )
    .map_err(|e| format!("building the engine: {e}"))
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .init();
    let args = match parse_args(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::from(2);
        }
    };
    let engine = match tokio::task::block_in_place(|| build_engine(&args)) {
        Ok(e) => e,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::FAILURE;
        }
    };
    let mut config = ServerConfig {
        // One model, one KV cache: one session at a time.
        max_sessions: 1,
        ..Default::default()
    };
    if let Some(b) = &args.bind {
        match b.parse() {
            Ok(addr) => config.bind = addr,
            Err(e) => {
                eprintln!("bad --bind {b:?}: {e}");
                return ExitCode::from(2);
            }
        }
    }
    let server = match s2s_server::start(config, engine).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to start: {e}");
            return ExitCode::FAILURE;
        }
    };
    let addr = server.local_addr();
    tracing::info!("web client at http://{addr}/ (WebSocket at ws://{addr}/ws; Ctrl-C to stop)");
    let token = server.shutdown_token();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            tracing::info!("shutting down");
            token.cancel();
        }
    });
    match server.wait().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("server error: {e}");
            ExitCode::FAILURE
        }
    }
}
