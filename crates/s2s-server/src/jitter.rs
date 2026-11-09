//! Input jitter buffer: turns client audio chunks of any size and arrival
//! time into exactly one 1920-sample frame per frame clock tick.
//!
//! # Policy
//!
//! - **Prefill.** The buffer starts (and restarts after an underrun) in a
//!   *buffering* state and yields silence until at least `prefill_samples`
//!   are queued. A larger prefill absorbs more network jitter at the cost of
//!   latency.
//! - **Underrun.** If fewer than a full frame is queued at a tick, the tick
//!   gets silence and the buffer goes back to buffering. Partial frames are
//!   not split across silence, which would sound choppy for clients that
//!   send small chunks. A leftover partial frame is flushed (padded with
//!   silence) only after `flush_after_ticks` starved ticks in a row, so the
//!   tail of an utterance is not stuck forever.
//! - **Overrun.** If more than `max_samples` are queued (the client sends
//!   faster than real time, or the clock fell behind), the oldest samples
//!   are dropped down to the prefill level, bounding input latency.

use std::collections::VecDeque;

use s2s_core::streams::FRAME_SAMPLES;

/// Jitter buffer tuning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JitterConfig {
    /// Samples to queue before playout starts (at least one frame is always
    /// required).
    pub prefill_samples: usize,
    /// Queue length above which the oldest samples are dropped.
    pub max_samples: usize,
    /// Starved ticks after which a leftover partial frame is flushed.
    pub flush_after_ticks: u32,
}

impl Default for JitterConfig {
    /// One frame of prefill (80 ms), 480 ms maximum, flush after 2 ticks.
    fn default() -> Self {
        Self {
            prefill_samples: FRAME_SAMPLES,
            max_samples: 6 * FRAME_SAMPLES,
            flush_after_ticks: 2,
        }
    }
}

/// What [`JitterBuffer::pop_frame`] produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameFill {
    /// A full frame of client audio.
    Full,
    /// This many client samples, then silence.
    Partial(usize),
    /// Silence only.
    Silence,
}

/// Counters for monitoring.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JitterStats {
    /// Times playout ran dry.
    pub underruns: u64,
    /// Times the queue overflowed.
    pub overruns: u64,
    /// Samples dropped by overruns.
    pub dropped_samples: u64,
    /// Samples currently queued.
    pub queued: usize,
}

/// See the [module docs](self).
#[derive(Clone, Debug)]
pub struct JitterBuffer {
    config: JitterConfig,
    queue: VecDeque<f32>,
    playing: bool,
    starved_ticks: u32,
    stats: JitterStats,
}

impl JitterBuffer {
    /// An empty buffer.
    pub fn new(config: JitterConfig) -> Self {
        Self {
            config,
            queue: VecDeque::with_capacity(config.max_samples + FRAME_SAMPLES),
            playing: false,
            starved_ticks: 0,
            stats: JitterStats::default(),
        }
    }

    fn prefill(&self) -> usize {
        self.config.prefill_samples.max(FRAME_SAMPLES)
    }

    /// Queues decoded client samples.
    pub fn push(&mut self, pcm: &[f32]) {
        self.queue.extend(pcm.iter().copied());
        let max = self.config.max_samples.max(self.prefill());
        if self.queue.len() > max {
            let drop = self.queue.len() - self.prefill();
            self.queue.drain(..drop);
            self.stats.overruns += 1;
            self.stats.dropped_samples += drop as u64;
        }
    }

    /// Fills `out` (exactly [`FRAME_SAMPLES`] long) for one tick.
    ///
    /// # Panics
    /// If `out.len() != FRAME_SAMPLES`.
    pub fn pop_frame(&mut self, out: &mut [f32]) -> FrameFill {
        assert_eq!(out.len(), FRAME_SAMPLES, "jitter buffer output frame size");
        let queued = self.queue.len();
        if self.playing && queued < FRAME_SAMPLES {
            self.playing = false;
            self.stats.underruns += 1;
        }
        if !self.playing && queued >= self.prefill() {
            self.playing = true;
        }
        if self.playing {
            self.starved_ticks = 0;
            for (o, s) in out.iter_mut().zip(self.queue.drain(..FRAME_SAMPLES)) {
                *o = s;
            }
            return FrameFill::Full;
        }
        if queued > 0 && queued < FRAME_SAMPLES {
            self.starved_ticks += 1;
            if self.starved_ticks >= self.config.flush_after_ticks {
                self.starved_ticks = 0;
                for (o, s) in out.iter_mut().zip(self.queue.drain(..)) {
                    *o = s;
                }
                out[queued..].fill(0.0);
                return FrameFill::Partial(queued);
            }
        } else {
            self.starved_ticks = 0;
        }
        out.fill(0.0);
        FrameFill::Silence
    }

    /// Current counters.
    pub fn stats(&self) -> JitterStats {
        JitterStats {
            queued: self.queue.len(),
            ..self.stats
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(start: usize, len: usize) -> Vec<f32> {
        (start..start + len).map(|i| i as f32).collect()
    }

    #[test]
    fn assembles_exact_frames_from_odd_chunks() {
        let mut jb = JitterBuffer::new(JitterConfig::default());
        let mut out = vec![0.0; FRAME_SAMPLES];
        // Push 3 frames' worth in uneven chunks.
        let total = 3 * FRAME_SAMPLES;
        let mut pos = 0;
        for size in [7, 500, 1913, 960, 1, 2379].iter().cycle() {
            if pos >= total {
                break;
            }
            let n = (*size).min(total - pos);
            jb.push(&ramp(pos, n));
            pos += n;
        }
        for f in 0..3 {
            assert_eq!(jb.pop_frame(&mut out), FrameFill::Full);
            assert_eq!(out, ramp(f * FRAME_SAMPLES, FRAME_SAMPLES));
        }
        assert_eq!(jb.pop_frame(&mut out), FrameFill::Silence);
        assert!(out.iter().all(|&x| x == 0.0));
        assert_eq!(jb.stats().underruns, 1);
    }

    #[test]
    fn prefill_and_underrun() {
        let cfg = JitterConfig {
            prefill_samples: 2 * FRAME_SAMPLES,
            ..JitterConfig::default()
        };
        let mut jb = JitterBuffer::new(cfg);
        let mut out = vec![0.0; FRAME_SAMPLES];
        jb.push(&ramp(0, FRAME_SAMPLES));
        // One frame queued, prefill wants two: silence.
        assert_eq!(jb.pop_frame(&mut out), FrameFill::Silence);
        jb.push(&ramp(FRAME_SAMPLES, FRAME_SAMPLES));
        assert_eq!(jb.pop_frame(&mut out), FrameFill::Full);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 1.0);
        assert_eq!(jb.pop_frame(&mut out), FrameFill::Full);
        assert_eq!(out[0], FRAME_SAMPLES as f32);
        // Dry: silence, then needs a full prefill again.
        assert_eq!(jb.pop_frame(&mut out), FrameFill::Silence);
        jb.push(&ramp(0, FRAME_SAMPLES));
        assert_eq!(jb.pop_frame(&mut out), FrameFill::Silence);
        assert_eq!(jb.stats().underruns, 1);
        assert_eq!(jb.stats().queued, FRAME_SAMPLES);
    }

    #[test]
    fn partial_frame_waits_then_flushes() {
        let mut jb = JitterBuffer::new(JitterConfig::default());
        let mut out = vec![1.0; FRAME_SAMPLES];
        jb.push(&ramp(1, 1000));
        assert_eq!(jb.pop_frame(&mut out), FrameFill::Silence);
        assert!(out.iter().all(|&x| x == 0.0));
        // More arrives in time to complete the frame: no flush.
        jb.push(&ramp(1001, 920));
        assert_eq!(jb.pop_frame(&mut out), FrameFill::Full);
        assert_eq!(out, ramp(1, FRAME_SAMPLES));
        // A tail that never completes is flushed after two starved ticks.
        jb.push(&ramp(1, 300));
        assert_eq!(jb.pop_frame(&mut out), FrameFill::Silence);
        assert_eq!(jb.pop_frame(&mut out), FrameFill::Partial(300));
        assert_eq!(&out[..300], &ramp(1, 300)[..]);
        assert!(out[300..].iter().all(|&x| x == 0.0));
        assert_eq!(jb.stats().queued, 0);
    }

    #[test]
    fn overrun_drops_oldest() {
        let cfg = JitterConfig::default();
        let mut jb = JitterBuffer::new(cfg);
        let mut out = vec![0.0; FRAME_SAMPLES];
        let total = cfg.max_samples + 10;
        jb.push(&ramp(0, total));
        let stats = jb.stats();
        assert_eq!(stats.overruns, 1);
        assert_eq!(stats.queued, FRAME_SAMPLES);
        assert_eq!(stats.dropped_samples, (total - FRAME_SAMPLES) as u64);
        // What is left is the newest audio.
        assert_eq!(jb.pop_frame(&mut out), FrameFill::Full);
        assert_eq!(out, ramp(total - FRAME_SAMPLES, FRAME_SAMPLES));
    }

    #[test]
    #[should_panic(expected = "frame size")]
    fn wrong_output_size_panics() {
        JitterBuffer::new(JitterConfig::default()).pop_frame(&mut [0.0; 10]);
    }
}
