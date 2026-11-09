//! SIMT GPU kernels for rust_s2s, written with NVIDIA cuda-oxide.
//!
//! NVIDIA's guidance for CUDA Rust: use the Tile model (cuTile, see
//! `s2s-kernels` feature `cuda`) for dense math, and the SIMT model
//! (cuda-oxide) for sparse, irregular or pointer-chasing work. These are the
//! irregular kernels from plan §7.2:
//!
//! | Kernel | Plan # | CPU reference |
//! |---|---|---|
//! | [`kernels::apply_token_bitmask`] | 14 | `s2s_kernels::reference::sampling::apply_token_bitmask` |
//! | [`kernels::embedding_sum_fwd`] | 7 | `s2s_kernels::reference::embedding::embedding_sum_fwd` |
//! | [`kernels::embedding_sum_bwd`] | 7 | `s2s_kernels::reference::embedding::embedding_sum_bwd` |
//! | [`kernels::apply_delays`] | 20 | `s2s_kernels::reference::delay::apply_delays` |
//! | [`kernels::argmax_rows`] | 13 | greedy `s2s_kernels::reference::sampling::sample` |
//! | [`kernels::kv_ring_write`] | 12 | (inline in the self-test) |
//!
//! **Status: M0.** These compile to PTX through cuda-oxide and assemble with
//! `ptxas` for every target without a GPU. They have not yet run on a GPU;
//! `cargo oxide run` (the `s2s-simt-selftest` binary) checks them against the
//! CPU references on a machine that has one.
//!
//! Multi-stream embedding tables are passed as **one concatenated buffer**
//! plus a per-stream row offset (`table_row_offset[s]` = first row of stream
//! `s`), because kernels take flat slices rather than slices of slices.

use cuda_host::cuda_module;

mod infer_kernels;
pub use infer_kernels::infer;

/// Threads per block for every kernel here (one thread per output element).
pub const BLOCK: u32 = 256;

/// Grid size for `n` elements at [`BLOCK`] threads per block.
pub fn blocks_for(n: usize) -> u32 {
    (n as u32).div_ceil(BLOCK)
}

/// Token value meaning "this slot contributes nothing" (matches
/// `s2s_kernels::reference::embedding::NO_TOKEN`).
pub const NO_TOKEN: u32 = u32::MAX;

#[cuda_module]
pub mod kernels {
    use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
    use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};

    /// Set logits of tokens not allowed by a bitmask to `-inf`.
    ///
    /// `allowed` is a little-endian bitset (`ceil(v / 32)` words). Used for
    /// JSON-schema-constrained tool calls (plan §4.2). One thread per logit.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn apply_token_bitmask(allowed: &[u32], mut logits: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(l) = logits.get_mut(idx) {
            if (allowed[i / 32] >> (i % 32)) & 1 == 0 {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    /// Multi-stream embedding gather-and-sum.
    ///
    /// `out[r, x] = sum_s tables[(table_row_offset[s] + tokens[r, s]) * d + x]`,
    /// skipping `NO_TOKEN`. `tokens` is `[n, streams]`, `out` is `[n, d]`.
    /// One thread per output element.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn embedding_sum_fwd(
        tables: &[f32],
        table_row_offset: &[u32],
        tokens: &[u32],
        streams: u32,
        d: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let (streams, d) = (streams as usize, d as usize);
        if let Some(o) = out.get_mut(idx) {
            let (r, x) = (i / d, i % d);
            let mut acc = 0.0f32;
            for s in 0..streams {
                let tok = tokens[r * streams + s];
                if tok != u32::MAX {
                    let row = table_row_offset[s] as usize + tok as usize;
                    acc += tables[row * d + x];
                }
            }
            *o = acc;
        }
    }

    /// Embedding backward: scatter-add `dout` rows into the (concatenated)
    /// table gradients with float atomics, since several frames can use the
    /// same token. `n_elems = n * d`. One thread per element of `dout`.
    ///
    /// Atomic accumulation order is nondeterministic, so results match the
    /// CPU reference to float rounding, not bit-exactly.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn embedding_sum_bwd(
        dout: &[f32],
        tokens: &[u32],
        table_row_offset: &[u32],
        streams: u32,
        d: u32,
        n_elems: u32,
        dtables: &[DeviceAtomicF32],
        mut done: DisjointSlice<u32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let (streams, d) = (streams as usize, d as usize);
        if i < n_elems as usize {
            let (r, x) = (i / d, i % d);
            let g = dout[i];
            for s in 0..streams {
                let tok = tokens[r * streams + s];
                if tok != u32::MAX {
                    let row = table_row_offset[s] as usize + tok as usize;
                    dtables[row * d + x].fetch_add(g, AtomicOrdering::Relaxed);
                }
            }
        }
        // One flag per thread so the launch has a disjoint output domain.
        if let Some(f) = done.get_mut(idx) {
            *f = 1;
        }
    }

    /// Per-stream delays on a `[frames, streams]` token grid (plan §3.3).
    ///
    /// `out` is `[frames + max_delay, streams]`; `out[t, s] = tokens[t - delays[s], s]`
    /// when that frame exists, else `pad[s]`. One thread per output element.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn apply_delays(
        tokens: &[u32],
        delays: &[u32],
        pad: &[u32],
        frames: u32,
        streams: u32,
        mut out: DisjointSlice<u32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let streams = streams as usize;
        if let Some(o) = out.get_mut(idx) {
            let (t, s) = (i / streams, i % streams);
            let delay = delays[s] as usize;
            *o = if t >= delay && t - delay < frames as usize {
                tokens[(t - delay) * streams + s]
            } else {
                pad[s]
            };
        }
    }

    /// Greedy decoding: index of the largest logit in each row, one warp
    /// per row. Lanes scan strided columns, then a shuffle-xor butterfly
    /// reduces `(value, index)` pairs; ties go to the lower index, matching
    /// the CPU reference. A row where every logit is `-inf` (fully masked)
    /// yields `u32::MAX`. Launch `rows * 32` threads (rounded up to whole
    /// blocks); `out` has one slot per row.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn argmax_rows(
        logits: &[f32],
        rows: u32,
        v: u32,
        mut out: DisjointSlice<u32, thread::WarpIndex>,
    ) {
        let gid = thread::index_1d().get();
        let (row, lane) = (gid / 32, (gid % 32) as u32);
        let (rows, v) = (rows as usize, v as usize);
        let mut best = f32::NEG_INFINITY;
        let mut idx = u32::MAX;
        if row < rows {
            let mut i = lane as usize;
            while i < v {
                let x = logits[row * v + i];
                if x > best {
                    best = x;
                    idx = i as u32;
                }
                i += 32;
            }
        }
        // Every lane of every (full) warp reaches the shuffles.
        let mut offset = 16u32;
        while offset > 0 {
            let other_best = warp::shuffle_xor_f32(best, offset);
            let other_idx = warp::shuffle_xor(idx, offset);
            if other_best > best || (other_best == best && other_idx < idx) {
                best = other_best;
                idx = other_idx;
            }
            offset /= 2;
        }
        if let Some(w) = thread::warp_index()
            && let Some(slot) = out.get_mut(w)
        {
            *slot = idx;
        }
    }

    /// KV-cache ring-buffer write: copy `t` new rows of `width` floats into a
    /// cache of `capacity` rows, starting at slot `start` and wrapping
    /// around (sliding-window attention, plan §10.4).
    ///
    /// Written gather-style, one thread per **cache** element, so every write
    /// goes through the thread's own `DisjointSlice` slot with no `unsafe`;
    /// threads whose slot receives no new row leave it untouched. That costs
    /// one launch over the whole cache per write: fine for M0, and the place
    /// to optimize (with a launch over just the new rows) once it shows up in
    /// profiles.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn kv_ring_write(
        new_rows: &[f32],
        t: u32,
        start: u32,
        capacity: u32,
        width: u32,
        mut cache: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let (t, start, capacity, width) = (
            t as usize,
            start as usize,
            capacity as usize,
            width as usize,
        );
        if let Some(c) = cache.get_mut(idx) {
            let (slot, col) = (i / width, i % width);
            let k = (slot + capacity - start % capacity) % capacity;
            if k < t {
                *c = new_rows[k * width + col];
            }
        }
    }
}
