//! The lockstep frame clock (plan §10.1, §10.3).
//!
//! One task owns the engine and every active session. Each 80 ms tick it:
//!
//! 1. applies queued joins and leaves (sessions change only between ticks),
//! 2. takes one 1920-sample frame (jitter buffer; silence if missing) and
//!    the pending injections from every session,
//! 3. calls [`Engine::step`] once for the whole batch, on a blocking thread,
//! 4. routes each session's audio and events to its outbound queue, never
//!    waiting on a client (see [`crate::session`] for the drop policy).
//!
//! With no sessions the clock sleeps until one joins. If a step overruns
//! the 80 ms budget, missed ticks are skipped (not bursted) and a warning is
//! logged; the jitter buffers absorb the slip.

use std::sync::Arc;
use std::time::Duration;

use s2s_core::streams::{FRAME_SAMPLES, SAMPLE_RATE};
use tokio::sync::mpsc;
use tokio::time::{self, Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::engine::{silent_frame, Engine, FrameInput, FrameOutput, SessionId};
use crate::protocol::{ErrorCode, Mode, ServerMessage};
use crate::session::{ClockCommand, OutboundQueue, SessionInbound};

/// Wall-clock length of one frame: 1920 samples at 24 kHz = 80 ms.
pub const FRAME_PERIOD: Duration =
    Duration::from_nanos(FRAME_SAMPLES as u64 * 1_000_000_000 / SAMPLE_RATE as u64);

/// Ticks between periodic stats log lines (10 s).
const STATS_EVERY: u64 = 125;

#[derive(Debug)]
struct Active {
    id: SessionId,
    mode: Mode,
    inbound: Arc<SessionInbound>,
    outbound: Arc<OutboundQueue>,
    frame_index: u64,
    failed: bool,
}

/// Session time in seconds at the end of frame `frame_index`, rounded to
/// the millisecond.
fn session_time(frame_index: u64) -> f64 {
    let t = (frame_index + 1) as f64 * FRAME_PERIOD.as_secs_f64();
    (t * 1000.0).round() / 1000.0
}

fn apply<E: Engine>(engine: &mut E, sessions: &mut Vec<Active>, cmd: ClockCommand) {
    match cmd {
        ClockCommand::Join {
            id,
            config,
            inbound,
            outbound,
            reply,
        } => {
            let result = engine.open_session(id, &config);
            match &result {
                Ok(()) => {
                    info!(session = id, mode = ?config.mode, batch = sessions.len() + 1, "session joined");
                    sessions.push(Active {
                        id,
                        mode: config.mode,
                        inbound,
                        outbound,
                        frame_index: 0,
                        failed: false,
                    });
                }
                Err(e) => warn!(session = id, error = %e, "engine rejected session"),
            }
            if reply.send(result.clone()).is_err() && result.is_ok() {
                // The connection gave up while joining.
                engine.close_session(id);
                sessions.retain(|s| s.id != id);
            }
        }
        ClockCommand::Leave { id } => {
            if let Some(pos) = sessions.iter().position(|s| s.id == id) {
                let s = sessions.swap_remove(pos);
                engine.close_session(s.id);
                s.outbound.close();
                info!(session = id, batch = sessions.len(), "session left");
            }
        }
    }
}

/// Ends every session (engine failure or clock shutdown).
fn fail_all<E: Engine>(
    engine: Option<&mut E>,
    sessions: &mut Vec<Active>,
    msg: Option<ServerMessage>,
) {
    let mut engine = engine;
    for s in sessions.drain(..) {
        if let Some(e) = engine.as_deref_mut() {
            e.close_session(s.id);
        }
        match &msg {
            Some(m) => s.outbound.fail(m.clone()),
            None => s.outbound.close(),
        }
    }
}

#[derive(Debug, Default)]
struct TickStats {
    ticks: u64,
    sum_step: Duration,
    max_step: Duration,
    overruns: u64,
}

/// Runs the frame clock until `shutdown` is cancelled, the command channel
/// closes, or the engine panics.
pub(crate) async fn run<E: Engine>(
    engine: E,
    mut commands: mpsc::UnboundedReceiver<ClockCommand>,
    shutdown: CancellationToken,
) {
    let mut engine = Some(engine);
    let mut sessions: Vec<Active> = Vec::new();
    let mut interval = time::interval(FRAME_PERIOD);
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut stats = TickStats::default();
    info!(
        period_ms = FRAME_PERIOD.as_millis() as u64,
        "frame clock started"
    );

    'outer: while let Some(eng) = engine.as_mut() {
        if sessions.is_empty() {
            // Idle: sleep until someone joins.
            let cmd = tokio::select! {
                _ = shutdown.cancelled() => break,
                cmd = commands.recv() => cmd,
            };
            let Some(cmd) = cmd else { break };
            apply(eng, &mut sessions, cmd);
            if !sessions.is_empty() {
                // First tick one frame after the first join, so the jitter
                // buffer has had a chance to fill.
                interval.reset();
            }
            continue;
        }

        tokio::select! {
            _ = shutdown.cancelled() => break,
            _ = interval.tick() => {}
        }
        loop {
            match commands.try_recv() {
                Ok(cmd) => apply(eng, &mut sessions, cmd),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => break 'outer,
            }
        }
        if sessions.is_empty() {
            continue;
        }

        // Gather one frame per session.
        let mut inputs: Vec<FrameInput> = sessions
            .iter()
            .map(|s| {
                let mut audio = silent_frame();
                let mut injections = Vec::new();
                s.inbound.take_frame(&mut audio, &mut injections);
                FrameInput {
                    session: s.id,
                    frame_index: s.frame_index,
                    audio,
                    injections,
                }
            })
            .collect();
        let mut outputs: Vec<FrameOutput> = sessions.iter().map(|_| FrameOutput::new()).collect();

        // Step the whole batch on a blocking thread.
        let Some(mut eng) = engine.take() else { break };
        let started = Instant::now();
        let joined = tokio::task::spawn_blocking(move || {
            let result = eng.step(&mut inputs, &mut outputs);
            (eng, outputs, result)
        })
        .await;
        let elapsed = started.elapsed();
        let (eng, outputs, result) = match joined {
            Ok(r) => r,
            Err(e) => {
                error!(error = %e, "engine panicked; frame clock stopping");
                fail_all::<E>(
                    None,
                    &mut sessions,
                    Some(ServerMessage::error(
                        ErrorCode::EngineFailure,
                        "engine crashed",
                    )),
                );
                return;
            }
        };
        let eng = engine.insert(eng);

        stats.ticks += 1;
        stats.sum_step += elapsed;
        stats.max_step = stats.max_step.max(elapsed);
        if elapsed > FRAME_PERIOD {
            stats.overruns += 1;
            warn!(
                step_ms = elapsed.as_secs_f64() * 1e3,
                batch = sessions.len(),
                "engine step overran the frame budget"
            );
        }
        if stats.ticks % STATS_EVERY == 0 {
            info!(
                batch = sessions.len(),
                mean_step_ms = stats.sum_step.as_secs_f64() * 1e3 / STATS_EVERY as f64,
                max_step_ms = stats.max_step.as_secs_f64() * 1e3,
                overruns = stats.overruns,
                "frame clock stats"
            );
            stats.sum_step = Duration::ZERO;
            stats.max_step = Duration::ZERO;
        }

        if let Err(e) = result {
            error!(error = %e, batch = sessions.len(), "engine step failed; ending the batch");
            fail_all(
                Some(eng),
                &mut sessions,
                Some(ServerMessage::error(
                    ErrorCode::EngineFailure,
                    e.to_string(),
                )),
            );
            continue;
        }

        // Route outputs.
        for (s, out) in sessions.iter_mut().zip(outputs) {
            let t = session_time(s.frame_index);
            s.frame_index += 1;
            for ev in out.events {
                if s.outbound.push_message(ev.into_message(t)).is_err() {
                    warn!(
                        session = s.id,
                        "client is not reading events; disconnecting"
                    );
                    s.outbound.fail(ServerMessage::error(
                        ErrorCode::SlowClient,
                        "client fell too far behind the event stream",
                    ));
                    s.failed = true;
                    break;
                }
            }
            if !s.failed && s.mode != Mode::Stt && s.outbound.push_audio(out.audio) {
                let dropped = s.outbound.stats().dropped_audio;
                if dropped.is_power_of_two() {
                    debug!(
                        session = s.id,
                        dropped, "client is slow; dropped oldest audio frame"
                    );
                }
            }
            if s.frame_index % STATS_EVERY == 0 {
                let j = s.inbound.jitter_stats();
                debug!(
                    session = s.id,
                    underruns = j.underruns,
                    overruns = j.overruns,
                    queued = j.queued,
                    dropped_out = s.outbound.stats().dropped_audio,
                    "session stats"
                );
            }
        }
        // Remove sessions that failed this tick.
        let mut i = 0;
        while i < sessions.len() {
            if sessions[i].failed {
                let s = sessions.swap_remove(i);
                eng.close_session(s.id);
                debug!(session = s.id, "session removed");
            } else {
                i += 1;
            }
        }
    }

    fail_all(engine.as_mut(), &mut sessions, None);
    info!("frame clock stopped");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineError, EngineEvent, Injection, SessionConfig};
    use crate::fake::{FakeEngine, FakeEngineConfig};
    use crate::session::{Outbound, SessionLimits, SessionManager};
    use std::sync::atomic::Ordering;

    #[test]
    fn frame_period_is_80ms() {
        assert_eq!(FRAME_PERIOD, Duration::from_millis(80));
        assert_eq!(session_time(0), 0.08);
        assert_eq!(session_time(30), 2.48);
    }

    fn drain(q: &OutboundQueue) -> (usize, Vec<ServerMessage>) {
        let mut audio = 0;
        let mut msgs = Vec::new();
        while let Some(item) = q.try_pop() {
            match item {
                Outbound::Audio(_) => audio += 1,
                Outbound::Message(m) => msgs.push(m),
            }
        }
        (audio, msgs)
    }

    #[tokio::test(start_paused = true)]
    async fn batches_sessions_in_lockstep() {
        let engine = FakeEngine::new(FakeEngineConfig::default());
        let stats = engine.stats();
        let shutdown = CancellationToken::new();
        let (mgr, clock) =
            SessionManager::start(engine, 3, SessionLimits::default(), shutdown.clone());

        let a = mgr
            .join(mgr.try_admit().unwrap(), SessionConfig::default())
            .await
            .unwrap();
        let b = mgr
            .join(mgr.try_admit().unwrap(), SessionConfig::default())
            .await
            .unwrap();
        let mut c = mgr
            .join(
                mgr.try_admit().unwrap(),
                SessionConfig {
                    mode: Mode::Stt,
                    ..SessionConfig::default()
                },
            )
            .await
            .unwrap();
        assert!(mgr.try_admit().is_none(), "at the session limit");
        assert_eq!(mgr.active(), 3);

        // Let the in-flight tick settle, then start from empty queues.
        time::sleep(Duration::from_millis(1)).await;
        for h in [&a, &b, &c] {
            drain(h.outbound());
        }
        let steps0 = stats.steps.load(Ordering::Relaxed);
        b.inject(Injection::Text("hello".into())).unwrap();
        for _ in 0..10 {
            a.push_audio(&[0.5; FRAME_SAMPLES]);
            time::sleep(FRAME_PERIOD).await;
        }
        assert_eq!(stats.steps.load(Ordering::Relaxed) - steps0, 10);
        assert_eq!(stats.last_batch.load(Ordering::Relaxed), 3);

        let (audio_a, msgs_a) = drain(a.outbound());
        assert_eq!(audio_a, 10);
        let user_text: Vec<_> = msgs_a
            .iter()
            .filter_map(|m| match m {
                ServerMessage::UserTranscriptDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(user_text, vec!["u1 ", "u2 "]);
        let (audio_b, msgs_b) = drain(b.outbound());
        assert_eq!(audio_b, 10);
        assert!(matches!(
            &msgs_b[0],
            ServerMessage::UserTranscriptDelta { text, .. } if text == "hello"
        ));
        let (audio_c, _) = drain(c.outbound());
        assert_eq!(audio_c, 0, "STT sessions get no audio");

        // Leaving takes effect before the next tick and frees the slot.
        c.leave();
        drop(c);
        assert!(mgr.try_admit().is_some());
        time::sleep(FRAME_PERIOD).await;
        assert_eq!(stats.last_batch.load(Ordering::Relaxed), 2);
        drop(a);
        drop(b);
        time::sleep(FRAME_PERIOD * 3).await;
        let steps = stats.steps.load(Ordering::Relaxed);
        time::sleep(FRAME_PERIOD * 3).await;
        assert_eq!(
            stats.steps.load(Ordering::Relaxed),
            steps,
            "idle clock does not step"
        );
        assert_eq!(stats.closed.load(Ordering::Relaxed), 3);

        shutdown.cancel();
        clock.await.unwrap();
        assert!(mgr
            .join(mgr.try_admit().unwrap(), SessionConfig::default())
            .await
            .is_err());
    }

    /// An engine whose step fails on its third call.
    #[derive(Debug, Default)]
    struct Flaky {
        steps: u32,
    }

    impl Engine for Flaky {
        fn open_session(&mut self, _: SessionId, c: &SessionConfig) -> Result<(), EngineError> {
            if c.system_prompt == "reject me" {
                Err(EngineError::Rejected("no".into()))
            } else {
                Ok(())
            }
        }
        fn close_session(&mut self, _: SessionId) {}
        fn step(
            &mut self,
            _: &mut [FrameInput],
            out: &mut [FrameOutput],
        ) -> Result<(), EngineError> {
            self.steps += 1;
            if self.steps == 3 {
                return Err(EngineError::StepFailed("boom".into()));
            }
            for o in out {
                o.events.push(EngineEvent::AssistantText("x".into()));
            }
            Ok(())
        }
    }

    #[tokio::test(start_paused = true)]
    async fn engine_rejection_and_failure() {
        let shutdown = CancellationToken::new();
        let (mgr, _clock) = SessionManager::start(
            Flaky::default(),
            2,
            SessionLimits::default(),
            shutdown.clone(),
        );
        let rejected = mgr
            .join(
                mgr.try_admit().unwrap(),
                SessionConfig {
                    system_prompt: "reject me".into(),
                    ..SessionConfig::default()
                },
            )
            .await;
        assert!(matches!(
            rejected,
            Err(crate::session::JoinError::Engine(EngineError::Rejected(_)))
        ));
        assert_eq!(mgr.active(), 0, "rejected admission is released");

        let s = mgr
            .join(mgr.try_admit().unwrap(), SessionConfig::default())
            .await
            .unwrap();
        time::sleep(FRAME_PERIOD * 5).await;
        let (audio, msgs) = drain(s.outbound());
        assert_eq!(audio, 0, "queued audio is discarded when the session fails");
        assert_eq!(msgs.len(), 3, "two events, then the error");
        assert!(matches!(
            msgs.last(),
            Some(ServerMessage::Error {
                code: ErrorCode::EngineFailure,
                ..
            })
        ));
        assert!(s.outbound().is_closed());
        shutdown.cancel();
    }

    #[tokio::test(start_paused = true)]
    async fn slow_client_is_disconnected() {
        let limits = SessionLimits {
            max_outbound_messages: 3,
            max_outbound_audio_frames: 2,
            ..SessionLimits::default()
        };
        let shutdown = CancellationToken::new();
        let (mgr, _clock) =
            SessionManager::start(FakeEngine::default(), 1, limits, shutdown.clone());
        let s = mgr
            .join(mgr.try_admit().unwrap(), SessionConfig::default())
            .await
            .unwrap();
        // Nobody drains the queue. Each text input yields two events, so
        // the first tick overflows the 3-message limit.
        for _ in 0..4 {
            s.inject(Injection::Text("hi".into())).unwrap();
        }
        time::sleep(FRAME_PERIOD * 4).await;
        assert!(s.outbound().is_closed());
        let (audio, msgs) = drain(s.outbound());
        assert_eq!(audio, 0, "queued audio discarded on failure");
        assert!(matches!(
            msgs.last(),
            Some(ServerMessage::Error {
                code: ErrorCode::SlowClient,
                ..
            })
        ));
        shutdown.cancel();
    }
}
