//! The axum HTTP/WebSocket front end.
//!
//! Routes:
//!
//! - `GET /ws`: the session WebSocket (protocol in [`crate::protocol`]).
//! - `GET /`: the bundled web client (if enabled).
//! - `GET /healthz`: `ok`.
//!
//! Each connection runs two tasks: a **reader** (the handler itself) that
//! decodes client audio into the session's jitter buffer and turns control
//! messages into injections, and a **writer** that drains the session's
//! [`OutboundQueue`], encodes audio and sends it. The frame clock in between
//! is shared by all sessions (see [`crate::clock`]).

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{close_code, CloseFrame, Message, Utf8Bytes, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::serve::ListenerExt;
use axum::Router;
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, info_span, warn, Instrument};

use crate::codec::{AudioDecoder, AudioEncoder, DEFAULT_OPUS_BITRATE};
use crate::engine::{Engine, Injection, SessionConfig};
use crate::protocol::{
    AudioChunk, AudioFormat, ClientMessage, ErrorCode, Mode, ServerMessage, SessionStart,
};
use crate::session::{Outbound, OutboundQueue, SessionHandle, SessionLimits, SessionManager};

/// The bundled browser client.
pub const WEB_CLIENT_HTML: &str = include_str!("../static/index.html");

/// Server settings.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// Address to listen on (port 0 picks a free port).
    pub bind: SocketAddr,
    /// Most concurrent sessions; more are refused with
    /// `error {session_limit}`.
    pub max_sessions: usize,
    /// Server → client audio format when `session.start` does not choose.
    pub default_output_format: AudioFormat,
    /// Opus bitrate for server → client audio, bits per second.
    pub opus_bitrate: i32,
    /// Per-session queue limits and jitter buffer tuning.
    pub limits: SessionLimits,
    /// How long a new socket may take to send `session.start`.
    pub start_timeout: Duration,
    /// Longest a single socket send may block before the client is
    /// disconnected as too slow.
    pub write_timeout: Duration,
    /// Largest accepted WebSocket message, in bytes.
    pub max_message_bytes: usize,
    /// Serve the web client at `/`.
    pub serve_web_client: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: SocketAddr::from(([127, 0, 0, 1], 8080)),
            max_sessions: 16,
            default_output_format: AudioFormat::Opus,
            opus_bitrate: DEFAULT_OPUS_BITRATE,
            limits: SessionLimits::default(),
            start_timeout: Duration::from_secs(10),
            write_timeout: Duration::from_secs(5),
            max_message_bytes: 1 << 20,
            serve_web_client: true,
        }
    }
}

#[derive(Debug)]
struct AppState {
    config: ServerConfig,
    sessions: SessionManager,
    shutdown: CancellationToken,
}

/// A server running in the background.
#[derive(Debug)]
pub struct RunningServer {
    local_addr: SocketAddr,
    sessions: SessionManager,
    shutdown: CancellationToken,
    http: JoinHandle<io::Result<()>>,
    clock: JoinHandle<()>,
}

impl RunningServer {
    /// The address actually bound.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// The session manager (for stats).
    pub fn sessions(&self) -> &SessionManager {
        &self.sessions
    }

    /// A token that stops the server when cancelled.
    pub fn shutdown_token(&self) -> CancellationToken {
        self.shutdown.clone()
    }

    /// Stops accepting connections, ends every session (clients get
    /// `error {shutting_down}` and a close frame) and waits for the HTTP
    /// server and frame clock to finish.
    pub async fn shutdown(self) -> io::Result<()> {
        self.shutdown.cancel();
        self.wait().await
    }

    /// Waits until the server stops (after its shutdown token is cancelled).
    pub async fn wait(self) -> io::Result<()> {
        let http = self.http.await.map_err(io::Error::other)?;
        self.clock.await.map_err(io::Error::other)?;
        http
    }
}

/// Binds `config.bind`, starts the frame clock with `engine`, and serves in
/// the background.
pub async fn start<E: Engine>(config: ServerConfig, engine: E) -> io::Result<RunningServer> {
    let listener = TcpListener::bind(config.bind).await?;
    let local_addr = listener.local_addr()?;
    // Audio frames are small and latency-critical: disable Nagle's
    // algorithm, which would otherwise hold them back waiting for ACKs.
    let listener = listener.tap_io(|tcp| {
        if let Err(e) = tcp.set_nodelay(true) {
            warn!(error = %e, "failed to set TCP_NODELAY");
        }
    });
    let shutdown = CancellationToken::new();
    let (sessions, clock) =
        SessionManager::start(engine, config.max_sessions, config.limits, shutdown.clone());
    let state = Arc::new(AppState {
        config,
        sessions: sessions.clone(),
        shutdown: shutdown.clone(),
    });
    let app = router(state);
    let http = tokio::spawn({
        let shutdown = shutdown.clone();
        async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown.cancelled_owned())
                .await
        }
    });
    info!(%local_addr, "s2s-server listening");
    Ok(RunningServer {
        local_addr,
        sessions,
        shutdown,
        http,
        clock,
    })
}

fn router(state: Arc<AppState>) -> Router {
    let mut app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/healthz", get(|| async { "ok" }));
    if state.config.serve_web_client {
        app = app.route("/", get(|| async { Html(WEB_CLIENT_HTML) }));
    }
    app.with_state(state)
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Arc<AppState>>) -> Response {
    ws.max_message_size(state.config.max_message_bytes)
        .on_upgrade(move |socket| handle_socket(socket, state))
        .into_response()
}

type WsSink = SplitSink<WebSocket, Message>;
type WsStream = SplitStream<WebSocket>;

fn text_message(msg: &ServerMessage) -> Option<Message> {
    match serde_json::to_string(msg) {
        Ok(s) => Some(Message::Text(s.into())),
        Err(e) => {
            warn!(error = %e, "failed to serialize server message");
            None
        }
    }
}

/// Sends with a timeout; `false` if the send failed or timed out.
async fn send_timed(sink: &mut WsSink, msg: Message, limit: Duration) -> bool {
    matches!(
        tokio::time::timeout(limit, sink.send(msg)).await,
        Ok(Ok(()))
    )
}

async fn send_close(sink: &mut WsSink, code: u16, reason: &'static str, limit: Duration) {
    let frame = CloseFrame {
        code,
        reason: Utf8Bytes::from_static(reason),
    };
    let _ = send_timed(sink, Message::Close(Some(frame)), limit).await;
}

/// Sends an error and closes; used before a session exists.
async fn reject(sink: &mut WsSink, code: ErrorCode, message: String, limit: Duration) {
    debug!(?code, %message, "rejecting connection");
    if let Some(m) = text_message(&ServerMessage::error(code, message)) {
        let _ = send_timed(sink, m, limit).await;
    }
    let close = match code {
        ErrorCode::SessionLimit => close_code::AGAIN,
        ErrorCode::ShuttingDown => close_code::AWAY,
        ErrorCode::Internal | ErrorCode::EngineFailure => close_code::ERROR,
        _ => close_code::POLICY,
    };
    send_close(sink, close, "", limit).await;
}

/// Waits for `session.start`. `Err(None)` means the client went away.
async fn await_start(
    stream: &mut WsStream,
    state: &AppState,
) -> Result<SessionStart, Option<(ErrorCode, String)>> {
    let wait = async {
        while let Some(msg) = stream.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    return match serde_json::from_str::<ClientMessage>(&text) {
                        Ok(ClientMessage::SessionStart(s)) => Ok(s),
                        Ok(_) => Err(Some((
                            ErrorCode::ProtocolViolation,
                            "the first message must be session.start".to_owned(),
                        ))),
                        Err(e) => Err(Some((ErrorCode::BadMessage, e.to_string()))),
                    };
                }
                Ok(Message::Binary(_)) => {
                    return Err(Some((
                        ErrorCode::ProtocolViolation,
                        "audio before session.start".to_owned(),
                    )))
                }
                Ok(Message::Close(_)) | Err(_) => return Err(None),
                Ok(Message::Ping(_) | Message::Pong(_)) => {}
            }
        }
        Err(None)
    };
    tokio::select! {
        r = tokio::time::timeout(state.config.start_timeout, wait) => match r {
            Ok(r) => r,
            Err(_) => Err(Some((ErrorCode::Timeout, "no session.start received in time".to_owned()))),
        },
        _ = state.shutdown.cancelled() => Err(Some((ErrorCode::ShuttingDown, "server is shutting down".to_owned()))),
    }
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sink, mut stream) = socket.split();
    let limit = state.config.write_timeout;

    let start = match await_start(&mut stream, &state).await {
        Ok(s) => s,
        Err(Some((code, message))) => return reject(&mut sink, code, message, limit).await,
        Err(None) => return,
    };
    let Some(admission) = state.sessions.try_admit() else {
        let max = state.sessions.max_sessions();
        warn!(max, "session limit reached; refusing session");
        let message = format!("server is at its limit of {max} sessions");
        return reject(&mut sink, ErrorCode::SessionLimit, message, limit).await;
    };
    let output_format = start
        .output_format
        .unwrap_or(state.config.default_output_format);
    let encoder = match AudioEncoder::new(output_format, state.config.opus_bitrate) {
        Ok(e) => e,
        Err(e) => return reject(&mut sink, ErrorCode::Internal, e.to_string(), limit).await,
    };
    let handle = match state
        .sessions
        .join(admission, SessionConfig::from(&start))
        .await
    {
        Ok(h) => h,
        Err(e) => {
            return reject(&mut sink, ErrorCode::EngineFailure, e.to_string(), limit).await;
        }
    };
    let span = info_span!("session", id = handle.id());
    run_session(sink, stream, handle, encoder, start.mode, state)
        .instrument(span)
        .await;
}

async fn run_session(
    mut sink: WsSink,
    mut stream: WsStream,
    mut handle: SessionHandle,
    encoder: AudioEncoder,
    mode: Mode,
    state: Arc<AppState>,
) {
    let limit = state.config.write_timeout;
    let started = ServerMessage::started(handle.id(), mode, encoder.format());
    if let Some(m) = text_message(&started) {
        if !send_timed(&mut sink, m, limit).await {
            return;
        }
    }
    info!(?mode, output_format = %encoder.format(), "session started");

    let kill = handle.kill_token().clone();
    let writer = tokio::spawn(
        write_loop(
            sink,
            Arc::clone(handle.outbound()),
            encoder,
            kill.clone(),
            state.shutdown.clone(),
            limit,
        )
        .in_current_span(),
    );

    let mut decoder = AudioDecoder::new();
    let mut pcm = Vec::new();
    let mut ignored_audio = false;
    let reply = |h: &SessionHandle, code: ErrorCode, msg: String| {
        debug!(?code, %msg, "client error");
        // A full queue means the client is not reading anyway.
        let _ = h.outbound().push_message(ServerMessage::error(code, msg));
    };
    loop {
        let msg = tokio::select! {
            _ = kill.cancelled() => break,
            msg = stream.next() => msg,
        };
        let msg = match msg {
            Some(Ok(m)) => m,
            Some(Err(e)) => {
                debug!(error = %e, "socket error");
                break;
            }
            None => break,
        };
        match msg {
            Message::Binary(bytes) => {
                if mode == Mode::Tts {
                    if !ignored_audio {
                        ignored_audio = true;
                        debug!("ignoring client audio in TTS mode");
                    }
                    continue;
                }
                let chunk = match AudioChunk::from_bytes(&bytes) {
                    Ok(c) => c,
                    Err(e) => {
                        reply(&handle, ErrorCode::BadAudio, e.to_string());
                        continue;
                    }
                };
                pcm.clear();
                match decoder.decode(&chunk, &mut pcm) {
                    Ok(_) => handle.push_audio(&pcm),
                    Err(e) => reply(&handle, ErrorCode::BadAudio, e.to_string()),
                }
            }
            Message::Text(text) => {
                let injection = match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(ClientMessage::SessionEnd) => {
                        info!("client ended the session");
                        break;
                    }
                    Ok(ClientMessage::SessionStart(_)) => {
                        reply(
                            &handle,
                            ErrorCode::ProtocolViolation,
                            "session already started".into(),
                        );
                        continue;
                    }
                    Ok(ClientMessage::TextInput { text }) => Injection::Text(text),
                    Ok(ClientMessage::ToolResult { call_id, json }) => {
                        Injection::ToolResult { call_id, json }
                    }
                    Ok(ClientMessage::CancelTool { call_id }) => Injection::CancelTool { call_id },
                    Err(e) => {
                        reply(&handle, ErrorCode::BadMessage, e.to_string());
                        continue;
                    }
                };
                if handle.inject(injection).is_err() {
                    reply(
                        &handle,
                        ErrorCode::RateLimited,
                        "too many pending control messages; message dropped".into(),
                    );
                }
            }
            Message::Close(_) => break,
            // axum answers pings itself.
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }

    // Leave the clock, let the writer flush what is queued and close.
    handle.leave();
    handle.outbound().close();
    let flush = limit * 2;
    match tokio::time::timeout(flush, writer).await {
        Ok(_) => {}
        Err(_) => {
            kill.cancel();
            debug!("writer did not finish flushing in time");
        }
    }
    kill.cancel();
    let j = handle.inbound().jitter_stats();
    let o = handle.outbound().stats();
    info!(
        underruns = j.underruns,
        input_overruns = j.overruns,
        dropped_output_frames = o.dropped_audio,
        "session ended"
    );
    // Dropping the handle frees the session slot.
    drop(handle);
}

async fn write_loop(
    mut sink: WsSink,
    outbound: Arc<OutboundQueue>,
    mut encoder: AudioEncoder,
    kill: CancellationToken,
    shutdown: CancellationToken,
    limit: Duration,
) {
    let mut healthy = true;
    loop {
        let item = tokio::select! {
            biased;
            _ = kill.cancelled() => None,
            item = outbound.pop() => item,
        };
        let Some(item) = item else { break };
        let msg = match item {
            Outbound::Audio(frame) => match encoder.encode_frame(&frame[..]) {
                Ok(chunk) => Message::Binary(chunk.to_bytes().into()),
                Err(e) => {
                    warn!(error = %e, "audio encode failed; frame skipped");
                    continue;
                }
            },
            Outbound::Message(m) => match text_message(&m) {
                Some(m) => m,
                None => continue,
            },
        };
        if !send_timed(&mut sink, msg, limit).await {
            warn!("socket send failed or timed out; disconnecting");
            healthy = false;
            break;
        }
    }
    if healthy {
        let short = limit.min(Duration::from_secs(1));
        if shutdown.is_cancelled() {
            if let Some(m) = text_message(&ServerMessage::error(
                ErrorCode::ShuttingDown,
                "server is shutting down",
            )) {
                let _ = send_timed(&mut sink, m, short).await;
            }
        }
        // Final messages (e.g. an error queued with `fail`), not audio.
        while let Some(item) = outbound.try_pop() {
            if let Outbound::Message(m) = item {
                if let Some(m) = text_message(&m) {
                    if !send_timed(&mut sink, m, short).await {
                        break;
                    }
                }
            }
        }
        let code = if shutdown.is_cancelled() {
            close_code::AWAY
        } else {
            close_code::NORMAL
        };
        send_close(&mut sink, code, "session ended", short).await;
    }
    // Stop the reader too.
    kill.cancel();
}
