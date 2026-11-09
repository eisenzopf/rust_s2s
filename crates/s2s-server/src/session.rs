//! Session management: admission, the per-session queues shared between a
//! connection's tasks and the frame clock, and session handles.
//!
//! # Data flow and backpressure
//!
//! ```text
//!  WebSocket reader ──decode──▶ SessionInbound ──(each tick)──▶ frame clock ──▶ Engine::step
//!                               (jitter buffer,                     │
//!                                injection queue)                   ▼
//!  WebSocket writer ◀──encode── OutboundQueue ◀──────────────── route outputs
//! ```
//!
//! The frame clock never waits on a client. Every queue between them is
//! bounded, with these policies:
//!
//! - **Inbound audio**: the [`JitterBuffer`] drops the oldest samples when
//!   the client gets more than `max_samples` ahead (see [`crate::jitter`]).
//! - **Inbound control** (`text.input`, `tool.result`, `session.cancel_tool`):
//!   at most `max_pending_injections` wait for the next tick; beyond that the
//!   message is dropped and the client gets `error {rate_limited}`.
//! - **Outbound audio**: at most `max_outbound_audio_frames` frames wait for
//!   the writer; when full, the **oldest** queued frame is dropped, so a
//!   client that falls behind hears the freshest audio (bounded latency, gaps
//!   instead of drift).
//! - **Outbound events** (transcripts, tool calls, turn events): never dropped
//!   silently, because losing a `tool.call` breaks the conversation. If more
//!   than `max_outbound_messages` are queued, the client is hopelessly
//!   behind: the session ends with `error {slow_client}`.
//! - **Socket writes**: each send has a timeout (`write_timeout`); a client
//!   whose socket stays blocked that long is disconnected.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use tokio::sync::{mpsc, oneshot, Notify, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::clock;
use crate::engine::{Engine, EngineError, Frame, Injection, SessionConfig, SessionId};
use crate::jitter::{FrameFill, JitterBuffer, JitterConfig, JitterStats};
use crate::protocol::{Mode, ServerMessage};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A bounded queue was full; the item was not queued.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("queue full")]
pub struct QueueFull;

/// An item for a session's WebSocket writer.
#[derive(Debug)]
pub enum Outbound {
    /// One 80 ms frame of assistant audio, still PCM (the writer encodes).
    Audio(Box<Frame>),
    /// A JSON control message.
    Message(ServerMessage),
}

/// Counters of an [`OutboundQueue`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OutboundStats {
    /// Audio frames waiting.
    pub queued_audio: usize,
    /// Messages waiting.
    pub queued_messages: usize,
    /// Audio frames dropped because the queue was full.
    pub dropped_audio: u64,
}

#[derive(Debug, Default)]
struct OutboundState {
    items: VecDeque<Outbound>,
    audio: usize,
    messages: usize,
    dropped_audio: u64,
    closed: bool,
}

/// Single-consumer queue from the frame clock (and the connection's reader)
/// to the connection's writer, with the drop policies in the
/// [module docs](self). Order is preserved among the items that are kept.
#[derive(Debug)]
pub struct OutboundQueue {
    state: Mutex<OutboundState>,
    notify: Notify,
    max_audio: usize,
    max_messages: usize,
}

impl OutboundQueue {
    /// A queue holding at most `max_audio` frames and `max_messages`
    /// messages.
    pub fn new(max_audio: usize, max_messages: usize) -> Self {
        Self {
            state: Mutex::new(OutboundState::default()),
            notify: Notify::new(),
            max_audio: max_audio.max(1),
            max_messages: max_messages.max(1),
        }
    }

    /// Queues an audio frame, dropping the oldest queued frame if full.
    /// Returns `true` if a frame was dropped. Ignored once closed.
    pub fn push_audio(&self, frame: Box<Frame>) -> bool {
        let mut st = lock(&self.state);
        if st.closed {
            return false;
        }
        let mut dropped = false;
        if st.audio >= self.max_audio {
            if let Some(pos) = st
                .items
                .iter()
                .position(|i| matches!(i, Outbound::Audio(_)))
            {
                st.items.remove(pos);
                st.audio -= 1;
                st.dropped_audio += 1;
                dropped = true;
            }
        }
        st.items.push_back(Outbound::Audio(frame));
        st.audio += 1;
        drop(st);
        self.notify.notify_one();
        dropped
    }

    /// Queues a message, or fails if `max_messages` are already queued.
    /// Ignored (`Ok`) once closed.
    pub fn push_message(&self, msg: ServerMessage) -> Result<(), QueueFull> {
        let mut st = lock(&self.state);
        if st.closed {
            return Ok(());
        }
        if st.messages >= self.max_messages {
            return Err(QueueFull);
        }
        st.items.push_back(Outbound::Message(msg));
        st.messages += 1;
        drop(st);
        self.notify.notify_one();
        Ok(())
    }

    /// Ends the stream with a final message: discards queued audio, queues
    /// `msg` regardless of limits, and closes the queue.
    pub fn fail(&self, msg: ServerMessage) {
        let mut st = lock(&self.state);
        if st.closed {
            return;
        }
        st.items.retain(|i| matches!(i, Outbound::Message(_)));
        st.audio = 0;
        st.items.push_back(Outbound::Message(msg));
        st.messages += 1;
        st.closed = true;
        drop(st);
        self.notify.notify_one();
    }

    /// Closes the queue: no more items are accepted, and [`pop`](Self::pop)
    /// returns `None` once the queued items are drained.
    pub fn close(&self) {
        lock(&self.state).closed = true;
        self.notify.notify_one();
    }

    /// Whether the queue is closed.
    pub fn is_closed(&self) -> bool {
        lock(&self.state).closed
    }

    /// Takes the next item without waiting.
    pub fn try_pop(&self) -> Option<Outbound> {
        let mut st = lock(&self.state);
        let item = st.items.pop_front()?;
        match item {
            Outbound::Audio(_) => st.audio -= 1,
            Outbound::Message(_) => st.messages -= 1,
        }
        Some(item)
    }

    /// Waits for the next item; `None` when closed and drained. Only one
    /// task may wait at a time.
    pub async fn pop(&self) -> Option<Outbound> {
        loop {
            if let Some(item) = self.try_pop() {
                return Some(item);
            }
            if self.is_closed() {
                return None;
            }
            // `notify_one` stores a permit if nobody is waiting, so a push
            // between the checks above and this await is not lost.
            self.notify.notified().await;
        }
    }

    /// Current counters.
    pub fn stats(&self) -> OutboundStats {
        let st = lock(&self.state);
        OutboundStats {
            queued_audio: st.audio,
            queued_messages: st.messages,
            dropped_audio: st.dropped_audio,
        }
    }
}

#[derive(Debug)]
struct InboundState {
    jitter: JitterBuffer,
    injections: VecDeque<Injection>,
}

/// Client input waiting for the next tick: decoded audio in a jitter buffer
/// and a bounded queue of injections.
#[derive(Debug)]
pub struct SessionInbound {
    state: Mutex<InboundState>,
    max_injections: usize,
}

impl SessionInbound {
    /// An empty inbound side.
    pub fn new(jitter: JitterConfig, max_injections: usize) -> Self {
        Self {
            state: Mutex::new(InboundState {
                jitter: JitterBuffer::new(jitter),
                injections: VecDeque::new(),
            }),
            max_injections: max_injections.max(1),
        }
    }

    /// Queues decoded client audio.
    pub fn push_audio(&self, pcm: &[f32]) {
        lock(&self.state).jitter.push(pcm);
    }

    /// Queues an injection for the next tick.
    pub fn inject(&self, injection: Injection) -> Result<(), QueueFull> {
        let mut st = lock(&self.state);
        if st.injections.len() >= self.max_injections {
            return Err(QueueFull);
        }
        st.injections.push_back(injection);
        Ok(())
    }

    /// Takes one frame of audio into `audio` and all pending injections
    /// into `injections` (called by the frame clock once per tick).
    pub fn take_frame(&self, audio: &mut Frame, injections: &mut Vec<Injection>) -> FrameFill {
        let mut st = lock(&self.state);
        injections.extend(st.injections.drain(..));
        st.jitter.pop_frame(&mut audio[..])
    }

    /// Jitter buffer counters.
    pub fn jitter_stats(&self) -> JitterStats {
        lock(&self.state).jitter.stats()
    }
}

/// Queue limits applied to every session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionLimits {
    /// Input jitter buffer tuning.
    pub jitter: JitterConfig,
    /// Control messages that may wait for the next tick.
    pub max_pending_injections: usize,
    /// Audio frames that may wait for the writer (25 = 2 s).
    pub max_outbound_audio_frames: usize,
    /// Messages that may wait for the writer before the client counts as
    /// too slow.
    pub max_outbound_messages: usize,
}

impl Default for SessionLimits {
    fn default() -> Self {
        Self {
            jitter: JitterConfig::default(),
            max_pending_injections: 64,
            max_outbound_audio_frames: 25,
            max_outbound_messages: 512,
        }
    }
}

/// Commands from session handles to the frame clock, applied between ticks.
#[derive(Debug)]
pub(crate) enum ClockCommand {
    Join {
        id: SessionId,
        config: SessionConfig,
        inbound: Arc<SessionInbound>,
        outbound: Arc<OutboundQueue>,
        reply: oneshot::Sender<Result<(), EngineError>>,
    },
    Leave {
        id: SessionId,
    },
}

/// Why [`SessionManager::join`] failed.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum JoinError {
    /// The engine refused the session.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// The frame clock is no longer running (shutdown or engine crash).
    #[error("frame clock is not running")]
    ClockStopped,
}

/// Proof that a session slot is free; obtained from
/// [`SessionManager::try_admit`] and consumed by [`SessionManager::join`].
/// Dropping it frees the slot.
#[derive(Debug)]
pub struct Admission {
    permit: OwnedSemaphorePermit,
}

#[derive(Debug)]
struct ManagerInner {
    cmd_tx: mpsc::UnboundedSender<ClockCommand>,
    permits: Arc<Semaphore>,
    max_sessions: usize,
    next_id: AtomicU64,
    limits: SessionLimits,
    shutdown: CancellationToken,
}

/// Admits sessions and connects them to the frame clock. Cheap to clone.
#[derive(Clone, Debug)]
pub struct SessionManager {
    inner: Arc<ManagerInner>,
}

impl SessionManager {
    /// Spawns the frame clock driving `engine` and returns a manager for
    /// it. The clock runs until `shutdown` is cancelled (or the engine
    /// panics); the returned handle completes then.
    pub fn start<E: Engine>(
        engine: E,
        max_sessions: usize,
        limits: SessionLimits,
        shutdown: CancellationToken,
    ) -> (Self, JoinHandle<()>) {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let task = tokio::spawn(clock::run(engine, cmd_rx, shutdown.clone()));
        let inner = ManagerInner {
            cmd_tx,
            permits: Arc::new(Semaphore::new(max_sessions)),
            max_sessions,
            next_id: AtomicU64::new(1),
            limits,
            shutdown,
        };
        (
            Self {
                inner: Arc::new(inner),
            },
            task,
        )
    }

    /// Reserves a session slot, or `None` at the session limit.
    pub fn try_admit(&self) -> Option<Admission> {
        let permit = Arc::clone(&self.inner.permits).try_acquire_owned().ok()?;
        Some(Admission { permit })
    }

    /// Sessions currently admitted (including ones still joining or
    /// flushing).
    pub fn active(&self) -> usize {
        self.inner.max_sessions - self.inner.permits.available_permits()
    }

    /// The session limit.
    pub fn max_sessions(&self) -> usize {
        self.inner.max_sessions
    }

    /// The per-session queue limits.
    pub fn limits(&self) -> SessionLimits {
        self.inner.limits
    }

    /// Opens a session in the engine and adds it to the frame clock. It
    /// takes part from the next tick on.
    pub async fn join(
        &self,
        admission: Admission,
        config: SessionConfig,
    ) -> Result<SessionHandle, JoinError> {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let limits = self.inner.limits;
        let inbound = Arc::new(SessionInbound::new(
            limits.jitter,
            limits.max_pending_injections,
        ));
        let outbound = Arc::new(OutboundQueue::new(
            limits.max_outbound_audio_frames,
            limits.max_outbound_messages,
        ));
        let kill = self.inner.shutdown.child_token();
        let mode = config.mode;
        let (reply, reply_rx) = oneshot::channel();
        self.inner
            .cmd_tx
            .send(ClockCommand::Join {
                id,
                config,
                inbound: Arc::clone(&inbound),
                outbound: Arc::clone(&outbound),
                reply,
            })
            .map_err(|_| JoinError::ClockStopped)?;
        reply_rx.await.map_err(|_| JoinError::ClockStopped)??;
        Ok(SessionHandle {
            id,
            mode,
            inbound,
            outbound,
            kill,
            cmd_tx: self.inner.cmd_tx.clone(),
            left: false,
            _permit: admission.permit,
        })
    }
}

/// A joined session, held by its connection. Dropping it removes the
/// session from the frame clock (before the next tick) and frees its slot.
#[derive(Debug)]
pub struct SessionHandle {
    id: SessionId,
    mode: Mode,
    inbound: Arc<SessionInbound>,
    outbound: Arc<OutboundQueue>,
    kill: CancellationToken,
    cmd_tx: mpsc::UnboundedSender<ClockCommand>,
    left: bool,
    _permit: OwnedSemaphorePermit,
}

impl SessionHandle {
    /// The session id.
    pub fn id(&self) -> SessionId {
        self.id
    }

    /// The session's mode.
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Queues decoded client audio.
    pub fn push_audio(&self, pcm: &[f32]) {
        self.inbound.push_audio(pcm);
    }

    /// Queues an injection for the next tick.
    pub fn inject(&self, injection: Injection) -> Result<(), QueueFull> {
        self.inbound.inject(injection)
    }

    /// The queue the writer drains.
    pub fn outbound(&self) -> &Arc<OutboundQueue> {
        &self.outbound
    }

    /// The inbound side (for stats).
    pub fn inbound(&self) -> &Arc<SessionInbound> {
        &self.inbound
    }

    /// Cancelled when the session must stop at once: server shutdown, or
    /// the connection's writer giving up. The connection also cancels it
    /// itself to stop its other task.
    pub fn kill_token(&self) -> &CancellationToken {
        &self.kill
    }

    /// Removes the session from the frame clock now (also done on drop).
    pub fn leave(&mut self) {
        if !self.left {
            self.left = true;
            // The clock being gone means the session is gone too.
            let _ = self.cmd_tx.send(ClockCommand::Leave { id: self.id });
        }
    }
}

impl Drop for SessionHandle {
    fn drop(&mut self) {
        self.leave();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::silent_frame;
    use crate::protocol::ErrorCode;

    fn frame(v: f32) -> Box<Frame> {
        let mut f = silent_frame();
        f[0] = v;
        f
    }

    fn audio_value(item: Option<Outbound>) -> f32 {
        match item {
            Some(Outbound::Audio(f)) => f[0],
            other => panic!("expected audio, got {other:?}"),
        }
    }

    #[test]
    fn outbound_drops_oldest_audio_keeps_order() {
        let q = OutboundQueue::new(2, 10);
        assert!(!q.push_audio(frame(1.0)));
        q.push_message(ServerMessage::error(ErrorCode::Internal, "m"))
            .unwrap();
        assert!(!q.push_audio(frame(2.0)));
        assert!(q.push_audio(frame(3.0)), "full: oldest dropped");
        let s = q.stats();
        assert_eq!(
            (s.queued_audio, s.queued_messages, s.dropped_audio),
            (2, 1, 1)
        );
        assert!(matches!(q.try_pop(), Some(Outbound::Message(_))));
        assert_eq!(audio_value(q.try_pop()), 2.0);
        assert_eq!(audio_value(q.try_pop()), 3.0);
        assert!(q.try_pop().is_none());
    }

    #[test]
    fn outbound_message_limit_and_fail() {
        let q = OutboundQueue::new(4, 2);
        q.push_audio(frame(1.0));
        q.push_message(ServerMessage::error(ErrorCode::Internal, "a"))
            .unwrap();
        q.push_message(ServerMessage::error(ErrorCode::Internal, "b"))
            .unwrap();
        assert_eq!(
            q.push_message(ServerMessage::error(ErrorCode::Internal, "c")),
            Err(QueueFull)
        );
        q.fail(ServerMessage::error(ErrorCode::SlowClient, "bye"));
        assert!(q.is_closed());
        // Audio discarded, messages kept, final message last.
        let mut codes = Vec::new();
        while let Some(item) = q.try_pop() {
            match item {
                Outbound::Message(ServerMessage::Error { code, .. }) => codes.push(code),
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(
            codes,
            vec![
                ErrorCode::Internal,
                ErrorCode::Internal,
                ErrorCode::SlowClient
            ]
        );
        // Closed: pushes are ignored.
        assert!(!q.push_audio(frame(1.0)));
        assert!(q
            .push_message(ServerMessage::error(ErrorCode::Internal, "x"))
            .is_ok());
        assert!(q.try_pop().is_none());
    }

    #[tokio::test]
    async fn outbound_pop_waits_and_ends_on_close() {
        let q = Arc::new(OutboundQueue::new(4, 4));
        let q2 = Arc::clone(&q);
        let consumer = tokio::spawn(async move {
            let mut n = 0;
            while let Some(item) = q2.pop().await {
                assert!(matches!(item, Outbound::Audio(_)));
                n += 1;
            }
            n
        });
        for i in 0..3 {
            tokio::task::yield_now().await;
            q.push_audio(frame(i as f32));
        }
        q.close();
        assert_eq!(consumer.await.unwrap(), 3);
    }

    #[test]
    fn inbound_injection_limit_and_take() {
        let inb = SessionInbound::new(JitterConfig::default(), 2);
        inb.inject(Injection::Text("a".into())).unwrap();
        inb.inject(Injection::Text("b".into())).unwrap();
        assert_eq!(inb.inject(Injection::Text("c".into())), Err(QueueFull));
        inb.push_audio(&[0.5; 1920]);
        let mut audio = [0.0; 1920];
        let mut inj = Vec::new();
        assert_eq!(inb.take_frame(&mut audio, &mut inj), FrameFill::Full);
        assert_eq!(inj.len(), 2);
        assert!(audio.iter().all(|&x| x == 0.5));
        assert_eq!(inb.take_frame(&mut audio, &mut inj), FrameFill::Silence);
        assert_eq!(inj.len(), 2);
    }
}
