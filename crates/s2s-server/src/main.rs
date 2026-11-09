//! `s2s-server`: runs the streaming server with the development
//! [`FakeEngine`] and serves the browser client at `/`.
//!
//! ```text
//! s2s-server [--bind ADDR] [--max-sessions N] [--codec opus|pcm_s16le|pcm_f32le]
//!            [--echo-delay-frames N] [--no-web-client]
//! ```
//!
//! Logging is controlled by `RUST_LOG` (default `info`).

use std::io::IsTerminal;
use std::process::ExitCode;

use s2s_server::{AudioFormat, FakeEngine, FakeEngineConfig, ServerConfig};
use tracing_subscriber::EnvFilter;

const USAGE: &str = "usage: s2s-server [--bind ADDR] [--max-sessions N] \
[--codec opus|pcm_s16le|pcm_f32le] [--echo-delay-frames N] [--no-web-client]";

fn parse_args(
    mut args: impl Iterator<Item = String>,
) -> Result<(ServerConfig, FakeEngineConfig), String> {
    let mut config = ServerConfig::default();
    let mut engine = FakeEngineConfig::default();
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--bind" => {
                let v = value("--bind")?;
                config.bind = v.parse().map_err(|e| format!("bad --bind {v:?}: {e}"))?;
            }
            "--max-sessions" => {
                let v = value("--max-sessions")?;
                config.max_sessions = v
                    .parse()
                    .map_err(|e| format!("bad --max-sessions {v:?}: {e}"))?;
            }
            "--codec" => {
                let v = value("--codec")?;
                config.default_output_format =
                    AudioFormat::from_name(&v).ok_or_else(|| format!("unknown codec {v:?}"))?;
            }
            "--echo-delay-frames" => {
                let v = value("--echo-delay-frames")?;
                engine.echo_delay_frames = v
                    .parse()
                    .map_err(|e| format!("bad --echo-delay-frames {v:?}: {e}"))?;
            }
            "--no-web-client" => config.serve_web_client = false,
            "-h" | "--help" => return Err(USAGE.to_owned()),
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        }
    }
    Ok((config, engine))
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

    let (config, engine_config) = match parse_args(std::env::args().skip(1)) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::from(2);
        }
    };
    let web = config.serve_web_client;
    let server = match s2s_server::start(config, FakeEngine::new(engine_config)).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to start: {e}");
            return ExitCode::FAILURE;
        }
    };
    let addr = server.local_addr();
    if web {
        tracing::info!("web client at http://{addr}/");
    }
    tracing::info!("WebSocket endpoint at ws://{addr}/ws (Ctrl-C to stop)");

    let token = server.shutdown_token();
    tokio::spawn(async move {
        match tokio::signal::ctrl_c().await {
            Ok(()) => {
                tracing::info!("shutting down");
                token.cancel();
            }
            Err(e) => tracing::warn!("cannot listen for Ctrl-C: {e}"),
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
