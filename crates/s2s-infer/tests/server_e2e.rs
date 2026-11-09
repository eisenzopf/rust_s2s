//! The duplex engine inside the real server: frame clock, blocking-thread
//! steps, WebSocket protocol and audio framing, end to end.

use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use s2s_core::streams::{special, StreamLayout, FRAME_SAMPLES};
use s2s_core::SplitMix64;
use s2s_data::TextTokenizer;
use s2s_duplex::{DuplexConfig, DuplexModel};
use s2s_infer::tokenizer::ByteTokenizer;
use s2s_infer::{DuplexEngine, EngineConfig, ToneCodec};
use s2s_server::protocol::{AudioChunk, AudioFormat, ServerMessage};
use s2s_server::ServerConfig;
use serde_json::json;
use tokio_tungstenite::tungstenite::Message;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn duplex_engine_serves_a_websocket_session() {
    let tok = ByteTokenizer;
    let pad = TextTokenizer::special_id(&tok, special::PAD).unwrap();
    let cfg = DuplexConfig::tiny(
        StreamLayout::default(),
        TextTokenizer::vocab_size(&tok) as usize,
        pad,
    );
    let model = Arc::new(DuplexModel::new(cfg, &mut SplitMix64::new(3)));
    let engine = DuplexEngine::new(
        model,
        Arc::new(ByteTokenizer),
        || Box::new(ToneCodec::new(8)),
        EngineConfig::default(),
    )
    .unwrap();
    let server = s2s_server::start(
        ServerConfig {
            bind: "127.0.0.1:0".parse().unwrap(),
            serve_web_client: false,
            ..ServerConfig::default()
        },
        engine,
    )
    .await
    .unwrap();
    let addr = server.local_addr();

    let (ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws"))
        .await
        .expect("websocket connects");
    let (mut tx, mut rx) = ws.split();
    tx.send(Message::text(
        json!({"type": "session.start", "system_prompt": "Be brief.", "tools": [],
               "mode": "s2s", "output_format": "pcm_f32le"})
        .to_string(),
    ))
    .await
    .unwrap();

    // Stream user audio (PCM s16le, 80 ms per message) in the background.
    let sender = tokio::spawn(async move {
        for i in 0..40 {
            let mut payload = vec![0x02u8];
            for n in 0..FRAME_SAMPLES {
                let s = ((n + i * FRAME_SAMPLES) as f32 * 0.05).sin() * 8000.0;
                payload.extend_from_slice(&(s as i16).to_le_bytes());
            }
            if tx.send(Message::binary(payload)).await.is_err() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(80)).await;
        }
        tx
    });

    let mut started = false;
    let mut frames = 0;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    while frames < 10 {
        let msg = tokio::time::timeout_at(deadline, rx.next())
            .await
            .expect("audio within the deadline")
            .expect("socket open")
            .unwrap();
        match msg {
            Message::Text(t) => match serde_json::from_str::<ServerMessage>(&t).unwrap() {
                ServerMessage::SessionStarted { frame_samples, .. } => {
                    assert_eq!(frame_samples, FRAME_SAMPLES);
                    started = true;
                }
                ServerMessage::Error { code, message } => {
                    panic!("server error {code:?}: {message}")
                }
                _ => {}
            },
            Message::Binary(b) => {
                let chunk = AudioChunk::from_bytes(&b).unwrap();
                assert_eq!(chunk.format, AudioFormat::PcmF32le);
                assert_eq!(
                    chunk.payload.len(),
                    FRAME_SAMPLES * 4,
                    "one 80 ms frame per tick"
                );
                frames += 1;
            }
            _ => {}
        }
    }
    assert!(started);
    let mut tx = sender.await.unwrap();
    tx.send(Message::text(json!({"type": "session.end"}).to_string()))
        .await
        .ok();
    server.shutdown().await.unwrap();
}
