//! Per-stream delays on frame-major token grids (plan §3.3).
//!
//! A grid is `[frames, streams]`. Delaying stream `s` by `d_s` frames moves its
//! token for frame `t` to row `t + d_s`. The delayed grid has
//! `frames + max_delay` rows so nothing is dropped, and the vacated cells hold
//! that stream's pad token. This is how DSM expresses "text leads audio"
//! (assistant text, delay 0, acoustic codebooks delayed by τ) and "text lags
//! audio" (user transcript delayed by `d_u`).

/// Delay each stream. Returns a `[frames + max(delays), streams]` grid.
pub fn apply_delays(tokens: &[u32], frames: usize, delays: &[usize], pad: &[u32]) -> Vec<u32> {
    let streams = delays.len();
    assert_eq!(pad.len(), streams);
    assert_eq!(tokens.len(), frames * streams);
    let max_d = delays.iter().copied().max().unwrap_or(0);
    let out_frames = frames + max_d;
    let mut out = vec![0u32; out_frames * streams];
    for t in 0..out_frames {
        for s in 0..streams {
            out[t * streams + s] = match t.checked_sub(delays[s]) {
                Some(src) if src < frames => tokens[src * streams + s],
                _ => pad[s],
            };
        }
    }
    out
}

/// Inverse of [`apply_delays`]: recover the `[frames, streams]` grid.
pub fn remove_delays(delayed: &[u32], frames: usize, delays: &[usize]) -> Vec<u32> {
    let streams = delays.len();
    let max_d = delays.iter().copied().max().unwrap_or(0);
    assert_eq!(delayed.len(), (frames + max_d) * streams);
    let mut out = vec![0u32; frames * streams];
    for t in 0..frames {
        for s in 0..streams {
            out[t * streams + s] = delayed[(t + delays[s]) * streams + s];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_layout() {
        // 3 frames, 3 streams: text (d=0), semantic (d=0), acoustic (d=1).
        let tokens = [1, 10, 100, 2, 20, 200, 3, 30, 300];
        let delays = [0, 0, 1];
        let pad = [0, 0, 999];
        let d = apply_delays(&tokens, 3, &delays, &pad);
        assert_eq!(d, vec![1, 10, 999, 2, 20, 100, 3, 30, 200, 0, 0, 300]);
        assert_eq!(remove_delays(&d, 3, &delays), tokens.to_vec());
    }

    #[test]
    fn user_text_lag() {
        // Stream 1 lags by 2 frames (like the user transcript, d_u).
        let tokens = [7, 1, 8, 2, 9, 3];
        let d = apply_delays(&tokens, 3, &[0, 2], &[0, 5]);
        assert_eq!(d, vec![7, 5, 8, 5, 9, 1, 0, 2, 0, 3]);
        assert_eq!(remove_delays(&d, 3, &[0, 2]), tokens.to_vec());
    }
}
