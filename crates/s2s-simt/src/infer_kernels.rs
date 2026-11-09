//! Inference kernels for streaming decode (batch 1, one frame per step).
//!
//! The host side is `crates/s2s-gpu`, which loads this crate's PTX with
//! cudarc and launches these kernels on the same stream as its cuBLASLt
//! GEMMs. Everything in one decode step is therefore queued without host
//! synchronization, which matters more than per-kernel speed at batch 1
//! (see ADR 0002).
//!
//! Kernels take raw device pointers (one `u64` PTX parameter each) and `u32`
//! sizes, so any cudarc buffer can be passed. Activations are `f32`; weights
//! live in the GEMMs. The numerics follow `s2s_kernels::reference` (and so
//! Kyutai's): RMSNorm `x * rstd * alpha`, interleaved RoPE, attention scaled
//! by `1/sqrt(head_dim)`, gated MLP `silu(gate) * up`.
//!
//! Launch shapes are part of each kernel's contract and are documented on it;
//! the reductions assume the stated block size.

use cuda_host::cuda_module;

#[cuda_module]
pub mod infer {
    use cuda_device::shared::SharedArray;
    use cuda_device::{kernel, thread, warp};

    /// Keys per attention split (one block); also the scores buffer size.
    const ATTN_CHUNK: u32 = 256;
    /// Largest head dimension the attention kernels support.
    const MAX_HEAD_DIM: u32 = 256;

    #[inline(always)]
    fn warp_sum(mut v: f32) -> f32 {
        let mut m = 16u32;
        while m > 0 {
            v += warp::shuffle_xor_f32_sync(u32::MAX, v, m);
            m >>= 1;
        }
        v
    }

    #[inline(always)]
    fn warp_max(mut v: f32) -> f32 {
        let mut m = 16u32;
        while m > 0 {
            let o = warp::shuffle_xor_f32_sync(u32::MAX, v, m);
            if o > v {
                v = o;
            }
            m >>= 1;
        }
        v
    }

    #[inline(always)]
    fn global_index() -> usize {
        (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize
    }

    /// RMSNorm of each row: `y = x * alpha / sqrt(mean(x^2) + eps)`.
    ///
    /// Launch: grid = rows, block = 256. `x`, `y`: `[rows, d]`; `alpha`: `[d]`.
    /// `x` and `y` may alias.
    #[kernel]
    pub fn rms_norm(x: *const f32, alpha: *const f32, y: *mut f32, d: u32, eps: f32) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x();
        let nt = thread::blockDim_x();
        let d = d as usize;
        let base = thread::blockIdx_x() as usize * d;
        let mut acc = 0.0f32;
        let mut i = tid as usize;
        while i < d {
            let v = unsafe { *x.add(base + i) };
            acc += v * v;
            i += nt as usize;
        }
        acc = warp_sum(acc);
        let (lane, w) = (tid % 32, tid / 32);
        if lane == 0 {
            unsafe { RED[w as usize] = acc };
        }
        thread::sync_threads();
        if w == 0 {
            let nw = nt.div_ceil(32);
            let mut t = if lane < nw {
                unsafe { RED[lane as usize] }
            } else {
                0.0
            };
            t = warp_sum(t);
            if lane == 0 {
                unsafe { RED[0] = t };
            }
        }
        thread::sync_threads();
        let total = unsafe { RED[0] };
        let rs = 1.0f32 / (total / d as f32 + eps).sqrt();
        let mut i = tid as usize;
        while i < d {
            unsafe { *y.add(base + i) = *x.add(base + i) * rs * *alpha.add(i) };
            i += nt as usize;
        }
    }

    /// Interleaved RoPE on `npairs` consecutive `(x[2p], x[2p + 1])` pairs, in
    /// place; pair `p` uses angle index `p % half` (`half = head_dim / 2`), so
    /// one launch rotates q and k of every head at one position.
    ///
    /// Launch: one thread per pair, block 256.
    #[kernel]
    pub fn rope_interleaved(x: *mut f32, cos: *const f32, sin: *const f32, npairs: u32, half: u32) {
        let p = global_index();
        if p < npairs as usize {
            let i = p % half as usize;
            unsafe {
                let a = *x.add(2 * p);
                let b = *x.add(2 * p + 1);
                let c = *cos.add(i);
                let s = *sin.add(i);
                *x.add(2 * p) = a * c - b * s;
                *x.add(2 * p + 1) = a * s + b * c;
            }
        }
    }

    /// Copy this position's k and v (from a fused `qkv: [q | k | v]`, each
    /// `[heads, head_dim]`, width `d`) into slot `slot` of caches laid out
    /// `[heads, cap, head_dim]`.
    ///
    /// Launch: one thread per element of `[0, d)`, block 256.
    #[kernel]
    pub fn kv_append(
        qkv: *const f32,
        kc: *mut f32,
        vc: *mut f32,
        d: u32,
        head_dim: u32,
        cap: u32,
        slot: u32,
    ) {
        let i = global_index();
        let (d, hd) = (d as usize, head_dim as usize);
        if i < d {
            let (h, j) = (i / hd, i % hd);
            let dst = (h * cap as usize + slot as usize) * hd + j;
            unsafe {
                *kc.add(dst) = *qkv.add(d + i);
                *vc.add(dst) = *qkv.add(2 * d + i);
            }
        }
    }

    /// Split-K decode attention, pass 1 (flash-decoding). For one query
    /// position, head `h = block / nsplit` and key chunk `s = block % nsplit`
    /// (keys `[256 s, min(n, 256 (s + 1)))` of the first `n` cache slots):
    /// writes the chunk's unnormalized output `po[block, :]`, and its running
    /// max and sum `pml[block] = (m, l)`.
    ///
    /// Every one of the first `n` slots is visible (Kyutai's rotating cache
    /// holds exactly the attendable keys, in slot order).
    ///
    /// Launch: grid = heads * nsplit with `nsplit = ceil(n / 256)`, block 256.
    /// `head_dim <= 256`. `q`: `[heads, head_dim]`; caches as in [`kv_append`].
    #[kernel]
    pub fn attn_decode_partial(
        q: *const f32,
        kc: *const f32,
        vc: *const f32,
        po: *mut f32,
        pml: *mut f32,
        head_dim: u32,
        cap: u32,
        n: u32,
        nsplit: u32,
        scale: f32,
    ) {
        static mut QS: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut SC: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x();
        let b = thread::blockIdx_x();
        let (h, s) = (b / nsplit, b % nsplit);
        let hd = head_dim as usize;
        let start = s * ATTN_CHUNK;
        let end = if n < start + ATTN_CHUNK {
            n
        } else {
            start + ATTN_CHUNK
        };
        let cnt = end - start;
        if tid < head_dim && tid < MAX_HEAD_DIM {
            unsafe { QS[tid as usize] = *q.add(h as usize * hd + tid as usize) };
        }
        thread::sync_threads();

        // Scores: warp w takes keys w, w + 8, ...
        let (lane, w) = (tid % 32, tid / 32);
        let row0 = h as usize * cap as usize + start as usize;
        let mut j = w;
        while j < cnt {
            let kb = (row0 + j as usize) * hd;
            let mut dot = 0.0f32;
            let mut t = lane as usize;
            while t < hd {
                dot += unsafe { QS[t] * *kc.add(kb + t) };
                t += 32;
            }
            dot = warp_sum(dot);
            if lane == 0 {
                unsafe { SC[j as usize] = dot * scale };
            }
            j += 8;
        }
        thread::sync_threads();

        // Block max.
        let mine = if tid < cnt {
            unsafe { SC[tid as usize] }
        } else {
            f32::NEG_INFINITY
        };
        let wm = warp_max(mine);
        if lane == 0 {
            unsafe { RED[w as usize] = wm };
        }
        thread::sync_threads();
        if w == 0 {
            let mut t = if lane < 8 {
                unsafe { RED[lane as usize] }
            } else {
                f32::NEG_INFINITY
            };
            t = warp_max(t);
            if lane == 0 {
                unsafe { RED[0] = t };
            }
        }
        thread::sync_threads();
        let m = unsafe { RED[0] };
        thread::sync_threads();

        // Probabilities (in place) and their sum.
        let p = if tid < cnt { (mine - m).exp() } else { 0.0 };
        if tid < cnt {
            unsafe { SC[tid as usize] = p };
        }
        let ws = warp_sum(p);
        if lane == 0 {
            unsafe { RED[w as usize] = ws };
        }
        thread::sync_threads();
        if w == 0 {
            let mut t = if lane < 8 {
                unsafe { RED[lane as usize] }
            } else {
                0.0
            };
            t = warp_sum(t);
            if lane == 0 {
                unsafe { RED[0] = t };
            }
        }
        thread::sync_threads();
        let l = unsafe { RED[0] };

        // Unnormalized output: thread t owns output dim t.
        let mut t = tid as usize;
        while t < hd {
            let mut o = 0.0f32;
            let mut jj = 0usize;
            while jj < cnt as usize {
                o += unsafe { SC[jj] * *vc.add((row0 + jj) * hd + t) };
                jj += 1;
            }
            unsafe { *po.add(b as usize * hd + t) = o };
            t += thread::blockDim_x() as usize;
        }
        if tid == 0 {
            unsafe {
                *pml.add(2 * b as usize) = m;
                *pml.add(2 * b as usize + 1) = l;
            }
        }
    }

    /// Split-K decode attention, pass 2: merge the `nsplit` chunks of each
    /// head into `out: [heads, head_dim]`.
    ///
    /// Launch: grid = heads, block >= head_dim (at most 256).
    #[kernel]
    pub fn attn_decode_combine(
        po: *const f32,
        pml: *const f32,
        out: *mut f32,
        head_dim: u32,
        nsplit: u32,
    ) {
        let h = thread::blockIdx_x() as usize;
        let t = thread::threadIdx_x() as usize;
        let (hd, ns) = (head_dim as usize, nsplit as usize);
        if t < hd {
            let mut mx = f32::NEG_INFINITY;
            let mut s = 0usize;
            while s < ns {
                let m = unsafe { *pml.add(2 * (h * ns + s)) };
                if m > mx {
                    mx = m;
                }
                s += 1;
            }
            let (mut l, mut o) = (0.0f32, 0.0f32);
            let mut s = 0usize;
            while s < ns {
                let b = h * ns + s;
                let wgt = (unsafe { *pml.add(2 * b) } - mx).exp();
                l += unsafe { *pml.add(2 * b + 1) } * wgt;
                o += unsafe { *po.add(b * hd + t) } * wgt;
                s += 1;
            }
            unsafe { *out.add(h * hd + t) = o / l };
        }
    }

    /// Gated MLP activation: `h[i] = silu(y[i]) * y[hidden + i]` for a
    /// `linear_in` output `y = [gate | up]`.
    ///
    /// Launch: one thread per element of `h`, block 256.
    #[kernel]
    pub fn glu_silu(y: *const f32, h: *mut f32, hidden: u32) {
        let i = global_index();
        if i < hidden as usize {
            unsafe {
                let g = *y.add(i);
                *h.add(i) = g / (1.0f32 + (-g).exp()) * *y.add(hidden as usize + i);
            }
        }
    }

    /// `x[i] += a[i] * scale[i]` (`scale` used only when `has_scale != 0`).
    ///
    /// Launch: one thread per element, block 256.
    #[kernel]
    pub fn add_inplace(x: *mut f32, a: *const f32, scale: *const f32, n: u32, has_scale: u32) {
        let i = global_index();
        if i < n as usize {
            unsafe {
                let mut v = *a.add(i);
                if has_scale != 0 {
                    v *= *scale.add(i);
                }
                *x.add(i) += v;
            }
        }
    }

    /// `f32` to `bf16` bits, round to nearest even (NaN stays NaN).
    ///
    /// Launch: one thread per element, block 256.
    #[kernel]
    pub fn f32_to_bf16(x: *const f32, y: *mut u16, n: u32) {
        let i = global_index();
        if i < n as usize {
            let b = unsafe { *x.add(i) }.to_bits();
            let out = if (b & 0x7fff_ffff) > 0x7f80_0000 {
                0x7fc0u16
            } else {
                ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
            };
            unsafe { *y.add(i) = out };
        }
    }
}
