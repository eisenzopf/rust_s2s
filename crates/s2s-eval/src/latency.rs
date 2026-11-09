//! Latency metrics: percentiles, turn-taking latency, tool round trips.
//!
//! # Percentiles
//!
//! [`percentile`] uses linear interpolation between closest ranks (NumPy's
//! default): for sorted samples `x[0..n]`, `p` in `[0, 100]` maps to rank
//! `r = p / 100 × (n − 1)` and the result is `x[⌊r⌋] + (r − ⌊r⌋) ×
//! (x[⌊r⌋+1] − x[⌊r⌋])`.
//!
//! # Turn events
//!
//! A session log is a list of timestamped [`TurnEvent`]s (seconds), e.g.
//! from voice-activity detection on both channels plus runtime tool events.
//! Events are grouped by [`TurnEvent::session`] and processed in time order
//! (stable for equal times).
//!
//! **Turn-taking latency** ([`turn_taking`]): when the user stops speaking
//! (`user_stop` at `t0`), the next `model_start` at `t1` closes the turn with
//! latency `t1 − t0`. If the user starts speaking again first, the stop was
//! a pause, not a turn end, and nothing is counted. A `user_stop` with no
//! later `model_start` (or user restart) is *unanswered*. A `model_start`
//! while the user is speaking is an *overlap* (barge-in or backchannel) and
//! yields no latency sample.
//!
//! **Tool round trip** ([`tool_round_trips`]): from `tool_call` (the runtime
//! dispatches when `</tool_call>` is sampled) to the `tool_result` with the
//! same `call_id` (the result is injected). Calls without a result are
//! *unmatched*.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

/// The `p`-th percentile (`p` in `[0, 100]`) of `samples` with linear
/// interpolation (see the [module docs](self)). `None` for no samples, a
/// `p` outside `[0, 100]` or NaN samples.
pub fn percentile(samples: &[f64], p: f64) -> Option<f64> {
    if samples.is_empty() || !(0.0..=100.0).contains(&p) || samples.iter().any(|x| x.is_nan()) {
        return None;
    }
    let mut s = samples.to_vec();
    s.sort_by(f64::total_cmp);
    Some(sorted_percentile(&s, p))
}

fn sorted_percentile(s: &[f64], p: f64) -> f64 {
    let r = p / 100.0 * (s.len() - 1) as f64;
    let lo = r.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    s[lo] + (r - lo as f64) * (s[hi] - s[lo])
}

/// Summary statistics of latency samples (same unit as the samples).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct LatencyStats {
    /// Number of samples.
    pub count: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// Smallest sample.
    pub min: f64,
    /// Largest sample.
    pub max: f64,
    /// Median.
    pub p50: f64,
    /// 90th percentile.
    pub p90: f64,
    /// 99th percentile.
    pub p99: f64,
}

impl LatencyStats {
    /// Statistics of `samples`; `None` if empty or any sample is NaN.
    pub fn from_samples(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() || samples.iter().any(|x| x.is_nan()) {
            return None;
        }
        let mut s = samples.to_vec();
        s.sort_by(f64::total_cmp);
        Some(Self {
            count: s.len(),
            mean: s.iter().sum::<f64>() / s.len() as f64,
            min: s[0],
            max: s[s.len() - 1],
            p50: sorted_percentile(&s, 50.0),
            p90: sorted_percentile(&s, 90.0),
            p99: sorted_percentile(&s, 99.0),
        })
    }
}

/// What happened at a [`TurnEvent`]. JSON: `{"t": 1.2, "event": "user_stop"}`,
/// `{"t": 2.0, "event": "tool_call", "call_id": "c_1"}`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum TurnEventKind {
    /// The user starts speaking.
    UserStart,
    /// The user stops speaking.
    UserStop,
    /// The model starts speaking.
    ModelStart,
    /// The model stops speaking.
    ModelStop,
    /// A tool call is dispatched.
    ToolCall {
        /// Call id.
        call_id: String,
    },
    /// A tool result is injected.
    ToolResult {
        /// Call id.
        call_id: String,
    },
}

/// A timestamped event in a session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnEvent {
    /// Time in seconds.
    pub t: f64,
    /// What happened.
    #[serde(flatten)]
    pub kind: TurnEventKind,
    /// Session the event belongs to (default: one unnamed session).
    #[serde(default)]
    pub session: String,
}

impl TurnEvent {
    /// An event in the default session.
    pub fn new(t: f64, kind: TurnEventKind) -> Self {
        Self {
            t,
            kind,
            session: String::new(),
        }
    }
}

/// Events grouped by session, each sorted by time (stable).
fn by_session(events: &[TurnEvent]) -> BTreeMap<&str, Vec<&TurnEvent>> {
    let mut map: BTreeMap<&str, Vec<&TurnEvent>> = BTreeMap::new();
    for e in events {
        map.entry(e.session.as_str()).or_default().push(e);
    }
    for v in map.values_mut() {
        v.sort_by(|a, b| a.t.total_cmp(&b.t));
    }
    map
}

/// Turn-taking results (see the [module docs](self)).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct TurnTaking {
    /// Latency of every answered user turn, in seconds.
    pub latencies: Vec<f64>,
    /// User turn ends the model never answered.
    pub unanswered: usize,
    /// Model starts while the user was speaking.
    pub overlaps: usize,
}

impl TurnTaking {
    /// Statistics of [`latencies`](Self::latencies).
    pub fn stats(&self) -> Option<LatencyStats> {
        LatencyStats::from_samples(&self.latencies)
    }
}

/// Turn-taking latency from turn events (see the [module docs](self)).
pub fn turn_taking(events: &[TurnEvent]) -> TurnTaking {
    let mut out = TurnTaking::default();
    for session in by_session(events).values() {
        let mut user_speaking = false;
        let mut pending: Option<f64> = None;
        for e in session {
            match e.kind {
                TurnEventKind::UserStart => {
                    user_speaking = true;
                    pending = None;
                }
                TurnEventKind::UserStop => {
                    user_speaking = false;
                    pending = Some(e.t);
                }
                TurnEventKind::ModelStart => {
                    if let Some(t0) = pending.take() {
                        out.latencies.push(e.t - t0);
                    } else if user_speaking {
                        out.overlaps += 1;
                    }
                }
                _ => {}
            }
        }
        out.unanswered += usize::from(pending.is_some());
    }
    out
}

/// Tool round-trip results.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ToolRoundTrips {
    /// Round-trip time of every completed call, in seconds.
    pub round_trips: Vec<f64>,
    /// Calls without a result.
    pub unmatched_calls: usize,
    /// Results without a preceding call.
    pub orphan_results: usize,
}

impl ToolRoundTrips {
    /// Statistics of [`round_trips`](Self::round_trips).
    pub fn stats(&self) -> Option<LatencyStats> {
        LatencyStats::from_samples(&self.round_trips)
    }
}

/// Tool round-trip times from turn events (see the [module docs](self)).
pub fn tool_round_trips(events: &[TurnEvent]) -> ToolRoundTrips {
    let mut out = ToolRoundTrips::default();
    for session in by_session(events).values() {
        let mut open: HashMap<&str, f64> = HashMap::new();
        for e in session {
            match &e.kind {
                TurnEventKind::ToolCall { call_id } => {
                    open.insert(call_id, e.t);
                }
                TurnEventKind::ToolResult { call_id } => match open.remove(call_id.as_str()) {
                    Some(t0) => out.round_trips.push(e.t - t0),
                    None => out.orphan_results += 1,
                },
                _ => {}
            }
        }
        out.unmatched_calls += open.len();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use TurnEventKind::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn percentiles_hand_computed() {
        let s = [15.0, 20.0, 35.0, 40.0, 50.0];
        assert_eq!(percentile(&s, 0.0), Some(15.0));
        assert_eq!(percentile(&s, 100.0), Some(50.0));
        assert_eq!(percentile(&s, 50.0), Some(35.0));
        // r = 0.4 * 4 = 1.6 → 20 + 0.6 * 15 = 29.
        assert!(close(percentile(&s, 40.0).unwrap(), 29.0));
        // r = 0.9 * 4 = 3.6 → 40 + 0.6 * 10 = 46.
        assert!(close(
            percentile(&[50.0, 15.0, 40.0, 35.0, 20.0], 90.0).unwrap(),
            46.0
        ));
        assert_eq!(percentile(&[7.0], 99.0), Some(7.0));
        assert_eq!(percentile(&[], 50.0), None);
        assert_eq!(percentile(&s, 101.0), None);
        assert_eq!(percentile(&[1.0, f64::NAN], 50.0), None);

        // 1..=100: p99 → r = 98.01 → 99 + 0.01 = 99.01.
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        let st = LatencyStats::from_samples(&v).unwrap();
        assert_eq!(st.count, 100);
        assert!(close(st.mean, 50.5));
        assert!(close(st.p50, 50.5));
        assert!(close(st.p90, 90.1));
        assert!(close(st.p99, 99.01));
        assert_eq!((st.min, st.max), (1.0, 100.0));
        assert_eq!(LatencyStats::from_samples(&[]), None);
    }

    #[test]
    fn turn_taking_hand_computed() {
        let ev = vec![
            // Turn 1: answered after 0.30 s.
            TurnEvent::new(0.0, UserStart),
            TurnEvent::new(1.0, UserStop),
            TurnEvent::new(1.3, ModelStart),
            TurnEvent::new(2.0, ModelStop),
            // A pause: the user resumes before the model speaks.
            TurnEvent::new(2.5, UserStart),
            TurnEvent::new(3.0, UserStop),
            TurnEvent::new(3.2, UserStart),
            // The model barges in while the user speaks (overlap).
            TurnEvent::new(3.5, ModelStart),
            TurnEvent::new(3.6, ModelStop),
            TurnEvent::new(4.0, UserStop),
            // Answered after 0.45 s (events given out of order).
            TurnEvent::new(4.45, ModelStart),
            TurnEvent::new(5.0, ModelStop),
            // Unanswered.
            TurnEvent::new(6.0, UserStart),
            TurnEvent::new(7.0, UserStop),
        ];
        let mut shuffled = ev.clone();
        shuffled.swap(9, 10);
        let tt = turn_taking(&shuffled);
        assert_eq!(tt.latencies.len(), 2);
        assert!(close(tt.latencies[0], 0.3));
        assert!(close(tt.latencies[1], 0.45));
        assert_eq!(tt.unanswered, 1);
        assert_eq!(tt.overlaps, 1);
        assert!(close(tt.stats().unwrap().p50, 0.375));

        // Sessions are independent.
        let mut two = ev[..3].to_vec();
        two.push(TurnEvent {
            t: 0.5,
            kind: UserStop,
            session: "b".into(),
        });
        two.push(TurnEvent {
            t: 1.5,
            kind: ModelStart,
            session: "b".into(),
        });
        let tt = turn_taking(&two);
        let mut l = tt.latencies.clone();
        l.sort_by(f64::total_cmp);
        assert!(close(l[0], 0.3) && close(l[1], 1.0));
    }

    #[test]
    fn tool_round_trips_hand_computed() {
        let ev = vec![
            TurnEvent::new(
                1.0,
                ToolCall {
                    call_id: "c_1".into(),
                },
            ),
            TurnEvent::new(
                1.5,
                ToolCall {
                    call_id: "c_2".into(),
                },
            ),
            TurnEvent::new(
                2.2,
                ToolResult {
                    call_id: "c_2".into(),
                },
            ),
            TurnEvent::new(
                3.0,
                ToolResult {
                    call_id: "c_1".into(),
                },
            ),
            TurnEvent::new(
                4.0,
                ToolCall {
                    call_id: "c_3".into(),
                },
            ),
            TurnEvent::new(
                4.5,
                ToolResult {
                    call_id: "c_9".into(),
                },
            ),
        ];
        let rt = tool_round_trips(&ev);
        assert_eq!(rt.round_trips.len(), 2);
        assert!(close(rt.round_trips[0], 0.7));
        assert!(close(rt.round_trips[1], 2.0));
        assert_eq!(rt.unmatched_calls, 1);
        assert_eq!(rt.orphan_results, 1);
        assert!(close(rt.stats().unwrap().mean, 1.35));
    }

    #[test]
    fn events_json() {
        let e: TurnEvent =
            serde_json::from_str(r#"{"t": 2.0, "event": "tool_call", "call_id": "c_1"}"#).unwrap();
        assert_eq!(
            e,
            TurnEvent::new(
                2.0,
                ToolCall {
                    call_id: "c_1".into()
                }
            )
        );
        let e: TurnEvent =
            serde_json::from_str(r#"{"t": 1, "event": "user_stop", "session": "s"}"#).unwrap();
        assert_eq!(e.kind, UserStop);
        assert_eq!(e.session, "s");
    }
}
