//! End-to-end tests: a real server on an ephemeral port, real WebSocket
//! clients, the `FakeEngine`.
//!
//! The tests run on tokio's paused clock, so seconds of streamed audio take
//! milliseconds and the frame clock is deterministic. Because the sockets
//! are real, a heartbeat task keeps virtual time advancing in 1 ms steps:
//! the runtime then polls the sockets between steps instead of jumping
//! straight to the next (possibly far-off) timer while bytes are in flight.

use std::net::SocketAddr;
use std::time::Duration;

use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use s2s_core::streams::{FRAME_SAMPLES, SAMPLE_RATE};
use s2s_server::codec::{f32_to_f32le, AudioDecoder, OpusEncoder};
use s2s_server::protocol::{ErrorCode, ToolState, TurnEvent};
use s2s_server::{
    AudioChunk, AudioFormat, FakeEngine, Mode, RunningServer, ServerConfig, ServerMessage,
};
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async_with_config, MaybeTlsStream, WebSocketStream};

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;
type Tx = SplitSink<Ws, Message>;
type Rx = SplitStream<Ws>;

const FRAME: Duration = Duration::from_millis(80);
const TIMEOUT: Duration = Duration::from_secs(30);

fn heartbeat() -> JoinHandle<()> {
    tokio::spawn(async {
        loop {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
}

async fn start_server(max_sessions: usize) -> RunningServer {
    let config = ServerConfig {
        bind: SocketAddr::from(([127, 0, 0, 1], 0)),
        max_sessions,
        ..ServerConfig::default()
    };
    s2s_server::start(config, FakeEngine::default())
        .await
        .expect("server starts")
}

async fn connect(addr: SocketAddr) -> (Tx, Rx) {
    // Disable Nagle, as browsers do for WebSockets.
    let (ws, _) = connect_async_with_config(format!("ws://{addr}/ws"), None, true)
        .await
        .expect("websocket connects");
    ws.split()
}

async fn send_json(tx: &mut Tx, v: Value) {
    tx.send(Message::text(v.to_string())).await.unwrap();
}

#[derive(Debug)]
enum Event {
    Audio(AudioChunk),
    Msg(ServerMessage),
    Closed,
}

async fn next_event(rx: &mut Rx) -> Event {
    loop {
        let msg = tokio::time::timeout(TIMEOUT, rx.next())
            .await
            .expect("server responded in time");
        match msg {
            None | Some(Err(_)) | Some(Ok(Message::Close(_))) => return Event::Closed,
            Some(Ok(Message::Text(t))) => {
                return Event::Msg(serde_json::from_str(&t).expect("valid server message"))
            }
            Some(Ok(Message::Binary(b))) => {
                return Event::Audio(AudioChunk::from_bytes(&b).expect("valid audio frame"))
            }
            Some(Ok(_)) => {}
        }
    }
}

/// Next control message, skipping audio. Panics on close.
async fn next_msg(rx: &mut Rx) -> ServerMessage {
    loop {
        match next_event(rx).await {
            Event::Msg(m) => return m,
            Event::Audio(_) => {}
            Event::Closed => panic!("socket closed while waiting for a message"),
        }
    }
}

/// Collects everything until the server closes the socket.
async fn collect_until_closed(rx: &mut Rx) -> (Vec<AudioChunk>, Vec<ServerMessage>) {
    let mut audio = Vec::new();
    let mut msgs = Vec::new();
    loop {
        match next_event(rx).await {
            Event::Audio(a) => audio.push(a),
            Event::Msg(m) => msgs.push(m),
            Event::Closed => return (audio, msgs),
        }
    }
}

async fn start_session(rx: &mut Rx, tx: &mut Tx, start: Value) -> u64 {
    send_json(tx, start).await;
    match next_msg(rx).await {
        ServerMessage::SessionStarted {
            session_id,
            sample_rate,
            frame_samples,
            ..
        } => {
            assert_eq!(sample_rate, SAMPLE_RATE);
            assert_eq!(frame_samples, FRAME_SAMPLES);
            session_id
        }
        other => panic!("expected session.started, got {other:?}"),
    }
}

fn sine(start: usize, len: usize) -> Vec<f32> {
    (start..start + len)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
        .collect()
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

fn decode_f32(chunk: &AudioChunk) -> Vec<f32> {
    let mut out = Vec::new();
    AudioDecoder::new().decode(chunk, &mut out).unwrap();
    out
}

fn user_text(msgs: &[ServerMessage]) -> String {
    msgs.iter()
        .filter_map(|m| match m {
            ServerMessage::UserTranscriptDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

fn assistant_text(msgs: &[ServerMessage]) -> String {
    msgs.iter()
        .filter_map(|m| match m {
            ServerMessage::AssistantTranscriptDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

fn turns(msgs: &[ServerMessage]) -> Vec<TurnEvent> {
    msgs.iter()
        .filter_map(|m| match m {
            ServerMessage::TurnEvent { event, .. } => Some(*event),
            _ => None,
        })
        .collect()
}

/// 2 s of a 440 Hz sine then 1.2 s of silence, as raw f32 PCM in 20 ms
/// chunks, in real (virtual) time; then `session.end`.
async fn stream_pcm(mut tx: Tx) {
    let voiced = 25 * FRAME_SAMPLES;
    let total = 40 * FRAME_SAMPLES;
    let chunk = FRAME_SAMPLES / 4;
    let mut pos = 0;
    while pos < total {
        let pcm = if pos < voiced {
            sine(pos, chunk)
        } else {
            vec![0.0; chunk]
        };
        let mut payload = Vec::new();
        f32_to_f32le(&pcm, &mut payload);
        let bytes = AudioChunk {
            format: AudioFormat::PcmF32le,
            payload,
        }
        .to_bytes();
        tx.send(Message::binary(bytes)).await.unwrap();
        pos += chunk;
        tokio::time::sleep(FRAME / 4).await;
    }
    send_json(&mut tx, json!({"type": "session.end"})).await;
}

#[tokio::test(start_paused = true)]
async fn s2s_pcm_echo_and_transcripts() {
    let _hb = heartbeat();
    let server = start_server(4).await;
    let (mut tx, mut rx) = connect(server.local_addr()).await;
    start_session(
        &mut rx,
        &mut tx,
        json!({"type": "session.start", "system_prompt": "be brief", "tools": [], "mode": "s2s", "output_format": "pcm_f32le"}),
    )
    .await;

    let sender = tokio::spawn(stream_pcm(tx));
    let (audio, msgs) = collect_until_closed(&mut rx).await;
    sender.await.unwrap();

    // One 80 ms frame per tick, in the requested format.
    assert!(audio.len() >= 35, "got {} audio frames", audio.len());
    for chunk in &audio {
        assert_eq!(chunk.format, AudioFormat::PcmF32le);
        assert_eq!(chunk.payload.len(), 4 * FRAME_SAMPLES);
    }
    // The fake engine echoes the sine back: about 25 loud frames.
    let levels: Vec<f32> = audio.iter().map(|c| rms(&decode_f32(c))).collect();
    let loud = levels.iter().filter(|&&l| (l - 0.354).abs() < 0.01).count();
    assert!(
        (23..=26).contains(&loud),
        "loud frames: {loud}, levels {levels:?}"
    );
    let first_loud = levels.iter().position(|&l| l > 0.1).unwrap();
    assert!(
        first_loud >= 2,
        "echo is delayed (first loud frame {first_loud})"
    );
    // The echoed samples are the sine itself (phase-continuous PCM).
    let echoed = decode_f32(&audio[first_loud + 1]);
    assert!(
        (0..25).any(|f| sine(f * FRAME_SAMPLES, FRAME_SAMPLES) == echoed),
        "echoed frame is bit-identical to a sent frame"
    );

    // Synthetic transcripts and turn events.
    assert_eq!(user_text(&msgs), "u1 u2 u3 u4 u5 ");
    assert_eq!(assistant_text(&msgs), "a1 a2 a3 a4 a5 ");
    assert_eq!(
        turns(&msgs),
        vec![
            TurnEvent::UserStarted,
            TurnEvent::AssistantStarted,
            TurnEvent::UserStopped,
            TurnEvent::AssistantStopped
        ]
    );
    let times: Vec<f64> = msgs
        .iter()
        .filter_map(|m| match m {
            ServerMessage::UserTranscriptDelta { t, .. } => Some(*t),
            _ => None,
        })
        .collect();
    assert!(times.windows(2).all(|w| w[1] - w[0] > 0.39), "{times:?}");
    assert!(!msgs
        .iter()
        .any(|m| matches!(m, ServerMessage::Error { .. })));

    // The slot is released after the session ends.
    tokio::time::sleep(FRAME * 2).await;
    assert_eq!(server.sessions().active(), 0);
    server.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn s2s_opus_round_trip() {
    let _hb = heartbeat();
    let server = start_server(4).await;
    let (mut tx, mut rx) = connect(server.local_addr()).await;
    start_session(
        &mut rx,
        &mut tx,
        json!({"type": "session.start", "output_format": "opus"}),
    )
    .await;

    let sender = tokio::spawn(async move {
        // 20 ms Opus packets, as a browser would send.
        let mut enc = OpusEncoder::new(48_000).unwrap();
        let packet = FRAME_SAMPLES / 4;
        for i in 0..(30 * 4) {
            let pcm = if i < 25 * 4 {
                sine(i * packet, packet)
            } else {
                vec![0.0; packet]
            };
            let chunk = AudioChunk {
                format: AudioFormat::Opus,
                payload: enc.encode(&pcm).unwrap(),
            };
            tx.send(Message::binary(chunk.to_bytes())).await.unwrap();
            tokio::time::sleep(FRAME / 4).await;
        }
        send_json(&mut tx, json!({"type": "session.end"})).await;
    });
    let (audio, msgs) = collect_until_closed(&mut rx).await;
    sender.await.unwrap();

    assert!(audio.len() >= 25, "got {} audio frames", audio.len());
    let mut dec = AudioDecoder::new();
    let mut loud = 0;
    for chunk in &audio {
        assert_eq!(chunk.format, AudioFormat::Opus);
        let mut pcm = Vec::new();
        assert_eq!(dec.decode(chunk, &mut pcm).unwrap(), FRAME_SAMPLES);
        if rms(&pcm) > 0.25 {
            loud += 1;
        }
    }
    assert!(loud >= 20, "loud frames after two Opus hops: {loud}");
    assert!(user_text(&msgs).starts_with("u1 u2 u3"));
    server.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn text_and_tool_round_trip() {
    let _hb = heartbeat();
    let server = start_server(4).await;
    let (mut tx, mut rx) = connect(server.local_addr()).await;
    start_session(
        &mut rx,
        &mut tx,
        json!({
            "type": "session.start",
            "tools": [{"name": "get_weather", "parameters": {"type": "object"}}],
            "output_format": "pcm_s16le"
        }),
    )
    .await;

    // Silence keeps flowing back while nothing is said.
    match next_event(&mut rx).await {
        Event::Audio(chunk) => {
            assert_eq!(chunk.format, AudioFormat::PcmS16le);
            assert_eq!(chunk.payload.len(), 2 * FRAME_SAMPLES);
            assert!(chunk.payload.iter().all(|&b| b == 0));
        }
        other => panic!("expected audio, got {other:?}"),
    }

    send_json(&mut tx, json!({"type": "text.input", "text": "hello"})).await;
    assert!(matches!(next_msg(&mut rx).await,
        ServerMessage::UserTranscriptDelta { text, .. } if text == "hello"));
    assert!(matches!(next_msg(&mut rx).await,
        ServerMessage::AssistantTranscriptDelta { text, .. } if text == "You said: hello"));

    send_json(
        &mut tx,
        json!({"type": "text.input", "text": "/tool get_weather {\"city\":\"Paris\"}"}),
    )
    .await;
    assert_eq!(
        next_msg(&mut rx).await,
        ServerMessage::ToolCall {
            call_id: "c_1".into(),
            name: "get_weather".into(),
            arguments: json!({"city": "Paris"})
        }
    );
    assert_eq!(
        next_msg(&mut rx).await,
        ServerMessage::ToolStatus {
            call_id: "c_1".into(),
            state: ToolState::Dispatched
        }
    );
    send_json(
        &mut tx,
        json!({"type": "tool.result", "call_id": "c_1", "json": {"temp": 18, "conditions": "sunny"}}),
    )
    .await;
    assert_eq!(
        next_msg(&mut rx).await,
        ServerMessage::ToolStatus {
            call_id: "c_1".into(),
            state: ToolState::Injected
        }
    );
    // Build the expected text with the same serializer: key order depends on
    // whether serde_json's `preserve_order` feature is enabled in the build
    // (another workspace crate enables it, and Cargo unifies features).
    let expected = format!(
        "The tool returned {}.",
        json!({"temp": 18, "conditions": "sunny"})
    );
    assert!(matches!(next_msg(&mut rx).await,
        ServerMessage::AssistantTranscriptDelta { text, .. }
            if text == expected));

    // Cancel an in-flight call.
    send_json(
        &mut tx,
        json!({"type": "text.input", "text": "/tool slow_thing"}),
    )
    .await;
    assert!(
        matches!(next_msg(&mut rx).await, ServerMessage::ToolCall { call_id, .. } if call_id == "c_2")
    );
    assert!(matches!(
        next_msg(&mut rx).await,
        ServerMessage::ToolStatus {
            state: ToolState::Dispatched,
            ..
        }
    ));
    send_json(
        &mut tx,
        json!({"type": "session.cancel_tool", "call_id": "c_2"}),
    )
    .await;
    assert_eq!(
        next_msg(&mut rx).await,
        ServerMessage::ToolStatus {
            call_id: "c_2".into(),
            state: ToolState::Cancelled
        }
    );

    // A bad message is reported but does not end the session.
    tx.send(Message::text("{not json")).await.unwrap();
    assert!(matches!(
        next_msg(&mut rx).await,
        ServerMessage::Error {
            code: ErrorCode::BadMessage,
            ..
        }
    ));
    send_json(&mut tx, json!({"type": "session.start"})).await;
    assert!(matches!(
        next_msg(&mut rx).await,
        ServerMessage::Error {
            code: ErrorCode::ProtocolViolation,
            ..
        }
    ));
    tx.send(Message::binary(vec![0x09, 1, 2])).await.unwrap();
    assert!(matches!(
        next_msg(&mut rx).await,
        ServerMessage::Error {
            code: ErrorCode::BadAudio,
            ..
        }
    ));

    send_json(&mut tx, json!({"type": "session.end"})).await;
    let (_, rest) = collect_until_closed(&mut rx).await;
    assert!(!rest
        .iter()
        .any(|m| matches!(m, ServerMessage::Error { .. })));
    server.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn tts_and_stt_modes() {
    let _hb = heartbeat();
    let server = start_server(4).await;

    // TTS: words become tones, client audio is ignored.
    let (mut tx, mut rx) = connect(server.local_addr()).await;
    start_session(
        &mut rx,
        &mut tx,
        json!({"type": "session.start", "mode": "tts", "output_format": "pcm_s16le"}),
    )
    .await;
    send_json(
        &mut tx,
        json!({"type": "text.input", "text": "hello brave new world"}),
    )
    .await;
    let mut said = String::new();
    let mut voiced = 0;
    let mut frames = 0;
    while frames < 20 {
        match next_event(&mut rx).await {
            Event::Audio(chunk) => {
                frames += 1;
                if rms(&decode_f32(&chunk)) > 0.1 {
                    voiced += 1;
                }
            }
            Event::Msg(ServerMessage::AssistantTranscriptDelta { text, .. }) => said += &text,
            Event::Msg(_) => {}
            Event::Closed => panic!("closed"),
        }
    }
    assert_eq!(said, "hello brave new world ");
    assert_eq!(voiced, 12, "3 tone frames per word");
    send_json(&mut tx, json!({"type": "session.end"})).await;
    collect_until_closed(&mut rx).await;

    // STT: transcripts only, no audio back.
    let (mut tx, mut rx) = connect(server.local_addr()).await;
    let id = start_session(
        &mut rx,
        &mut tx,
        json!({"type": "session.start", "mode": "stt"}),
    )
    .await;
    assert!(id > 1);
    let sender = tokio::spawn(stream_pcm(tx));
    let (audio, msgs) = collect_until_closed(&mut rx).await;
    sender.await.unwrap();
    assert!(audio.is_empty(), "STT sends no audio");
    assert_eq!(user_text(&msgs), "u1 u2 u3 u4 u5 ");
    assert_eq!(assistant_text(&msgs), "");
    assert_eq!(
        turns(&msgs),
        vec![TurnEvent::UserStarted, TurnEvent::UserStopped]
    );
    server.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn admission_limit() {
    let _hb = heartbeat();
    let server = start_server(1).await;
    let (mut tx_a, mut rx_a) = connect(server.local_addr()).await;
    start_session(&mut rx_a, &mut tx_a, json!({"type": "session.start"})).await;

    let (mut tx_b, mut rx_b) = connect(server.local_addr()).await;
    send_json(&mut tx_b, json!({"type": "session.start"})).await;
    let (_, msgs) = collect_until_closed(&mut rx_b).await;
    assert!(
        matches!(
            &msgs[..],
            [ServerMessage::Error {
                code: ErrorCode::SessionLimit,
                ..
            }]
        ),
        "{msgs:?}"
    );

    send_json(&mut tx_a, json!({"type": "session.end"})).await;
    collect_until_closed(&mut rx_a).await;

    let (mut tx_c, mut rx_c) = connect(server.local_addr()).await;
    start_session(&mut rx_c, &mut tx_c, json!({"type": "session.start"})).await;
    assert_eq!(server.sessions().active(), 1);
    server.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn protocol_violations_before_start() {
    let _hb = heartbeat();
    let server = start_server(2).await;

    let (mut tx, mut rx) = connect(server.local_addr()).await;
    tx.send(Message::binary(vec![0x02, 0, 0])).await.unwrap();
    let (_, msgs) = collect_until_closed(&mut rx).await;
    assert!(matches!(
        &msgs[..],
        [ServerMessage::Error {
            code: ErrorCode::ProtocolViolation,
            ..
        }]
    ));

    let (mut tx, mut rx) = connect(server.local_addr()).await;
    send_json(&mut tx, json!({"type": "text.input", "text": "hi"})).await;
    let (_, msgs) = collect_until_closed(&mut rx).await;
    assert!(matches!(
        &msgs[..],
        [ServerMessage::Error {
            code: ErrorCode::ProtocolViolation,
            ..
        }]
    ));

    // Silence until the start timeout.
    let (_tx, mut rx) = connect(server.local_addr()).await;
    let (_, msgs) = collect_until_closed(&mut rx).await;
    assert!(matches!(
        &msgs[..],
        [ServerMessage::Error {
            code: ErrorCode::Timeout,
            ..
        }]
    ));
    assert_eq!(server.sessions().active(), 0);
    server.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn shutdown_ends_live_sessions() {
    let _hb = heartbeat();
    let server = start_server(2).await;
    let (mut tx, mut rx) = connect(server.local_addr()).await;
    start_session(
        &mut rx,
        &mut tx,
        json!({"type": "session.start", "mode": Mode::S2s}),
    )
    .await;
    // Let a few frames flow.
    for _ in 0..3 {
        assert!(matches!(next_event(&mut rx).await, Event::Audio(_)));
    }
    let token = server.shutdown_token();
    let done = tokio::spawn(server.shutdown());
    let (_, msgs) = collect_until_closed(&mut rx).await;
    assert!(token.is_cancelled());
    assert!(
        msgs.iter().any(|m| matches!(
            m,
            ServerMessage::Error {
                code: ErrorCode::ShuttingDown,
                ..
            }
        )),
        "{msgs:?}"
    );
    done.await.unwrap().unwrap();
}

#[tokio::test]
async fn serves_web_client_and_health() {
    let server = start_server(1).await;
    let addr = server.local_addr();
    let get = |path: &'static str| async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut s = TcpStream::connect(addr).await.unwrap();
        let req = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
        s.write_all(req.as_bytes()).await.unwrap();
        let mut body = String::new();
        s.read_to_string(&mut body).await.unwrap();
        body
    };
    let page = get("/").await;
    assert!(page.starts_with("HTTP/1.1 200"), "{page:.80}");
    assert!(page.contains("session.start") && page.contains("AudioWorkletProcessor"));
    let health = get("/healthz").await;
    assert!(health.starts_with("HTTP/1.1 200") && health.ends_with("ok"));
    server.shutdown().await.unwrap();
}
