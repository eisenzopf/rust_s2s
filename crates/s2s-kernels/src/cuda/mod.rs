//! GPU kernels written in NVIDIA's cuTile Rust (feature `cuda`).
//!
//! **Status: milestone M0.** These kernels build, lower to Tile IR, and
//! assemble to cubins for sm_80/89/90/100 in CI (`tests/cuda_compile.rs`;
//! needs CUDA headers and `tileiras`, but **no GPU**). They have **not yet
//! been run on a GPU**; `tests/cuda_parity.rs` compares them against
//! [`crate::reference`] and is `#[ignore]`d until it runs on real hardware
//! (`cargo test -p s2s-kernels --features cuda -- --ignored`).
//!
//! These are the dense, tile-shaped kernels. Sparse and irregular kernels
//! (masks, embedding scatter-add, delays) are written in cuda-oxide in
//! `crates/s2s-simt` (ADR 0001).
//!
//! Kernels are f32 for now; bf16 variants follow once the f32 versions are
//! validated on hardware.
//!
//! Layout notes for callers:
//! - [`rms_norm_fwd_f32`]: `x` is `[rows, N]`; `y` must be partitioned into
//!   `[1, N]` tiles and `rstd` (shape `[rows]`) into `[1]` tiles, so the launch
//!   grid has one tile block per row. `N` must be a multiple of `BLOCK`.
//! - [`swiglu_fwd_f32`]: element-wise over 1-D tensors; the output partition
//!   size `B` must divide the length.

pub mod ops;

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// `z = x + y`. The M0 "hello kernel".
    #[cutile::entry()]
    fn vec_add_f32<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = x.load_like(z);
        let ty = y.load_like(z);
        z.store(tx + ty);
    }

    /// RMSNorm forward over rows of length `N`, one tile block per row.
    ///
    /// Writes `y = x * rstd * w` and `rstd = 1 / sqrt(mean(x^2) + eps)`
    /// (saved for the backward pass), matching
    /// [`crate::reference::norm::rmsnorm_fwd`].
    #[cutile::entry()]
    fn rms_norm_fwd_f32<const N: i32, const BLOCK: i32>(
        x: &Tensor<f32, { [-1, N] }>,
        w: &Tensor<f32, { [N] }>,
        y: &mut Tensor<f32, { [1, N] }>,
        rstd: &mut Tensor<f32, { [1] }>,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK] }> = const_shape![1, BLOCK];
        let num_tiles: i32 = N / BLOCK;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let x_part: Partition<f32, { [1, BLOCK] }> = x.partition(tile_shape);
        let mut sumsq: Tile<f32, { [1, BLOCK] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tx: Tile<f32, { [1, BLOCK] }> = x_part.load([row, j]);
            sumsq = sumsq + tx * tx;
        }
        let sumsq: Tile<f32, { [1] }> = reduce_sum(sumsq, 1i32);
        let sumsq: Tile<f32, { [] }> = sumsq.reshape(const_shape![]);
        let sumsq: f32 = tile_to_scalar(sumsq);
        let n: f32 = convert_scalar(N);
        let ms_eps: f32 = sumsq / n + eps;
        let r: Tile<f32, { [] }> = rsqrt(scalar_to_tile(ms_eps), ftz::Disabled);
        let r: f32 = tile_to_scalar(r);

        let r_out: Tile<f32, { [1] }> = r.broadcast(const_shape![1]);
        rstd.store(r_out);

        let r_tile: Tile<f32, { [1, BLOCK] }> = r.broadcast(tile_shape);
        let w_part: Partition<f32, { [BLOCK] }> = w.partition(const_shape![BLOCK]);
        let mut y_part: PartitionMut<f32, { [1, BLOCK] }> = y.partition_mut(tile_shape);
        for j in 0i32..num_tiles {
            let tx: Tile<f32, { [1, BLOCK] }> = x_part.load([row, j]);
            let tw: Tile<f32, { [1, BLOCK] }> = w_part.load([j]).reshape(tile_shape);
            y_part.store(tx * r_tile * tw, [0i32, j]);
        }
    }

    /// RMSNorm backward, one tile block per row.
    ///
    /// Writes `dx = rstd * (w*dy - x * rstd^2 * mean(w*dy*x))` and the row's
    /// contribution to the weight gradient, `dw_partial[row] = dy * x * rstd`;
    /// [`column_sum_f32`] then reduces `dw_partial` over rows. Matches
    /// [`crate::reference::norm::rmsnorm_bwd`].
    #[cutile::entry()]
    fn rms_norm_bwd_f32<const N: i32, const BLOCK: i32>(
        x: &Tensor<f32, { [-1, N] }>,
        w: &Tensor<f32, { [N] }>,
        rstd: &Tensor<f32, { [-1] }>,
        dy: &Tensor<f32, { [-1, N] }>,
        dx: &mut Tensor<f32, { [1, N] }>,
        dw_partial: &mut Tensor<f32, { [1, N] }>,
    ) {
        let tile_shape: Shape<{ [1, BLOCK] }> = const_shape![1, BLOCK];
        let num_tiles: i32 = N / BLOCK;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let rs_tile: Tile<f32, { [1] }> = rstd.partition(const_shape![1]).load([row]);
        let rs: f32 = tile_to_scalar(rs_tile.reshape(const_shape![]));

        let x_part: Partition<f32, { [1, BLOCK] }> = x.partition(tile_shape);
        let dy_part: Partition<f32, { [1, BLOCK] }> = dy.partition(tile_shape);
        let w_part: Partition<f32, { [BLOCK] }> = w.partition(const_shape![BLOCK]);

        let mut acc: Tile<f32, { [1, BLOCK] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tx: Tile<f32, { [1, BLOCK] }> = x_part.load([row, j]);
            let tdy: Tile<f32, { [1, BLOCK] }> = dy_part.load([row, j]);
            let tw: Tile<f32, { [1, BLOCK] }> = w_part.load([j]).reshape(tile_shape);
            acc = acc + tw * tdy * tx;
        }
        let dot: Tile<f32, { [1] }> = reduce_sum(acc, 1i32);
        let dot: f32 = tile_to_scalar(dot.reshape(const_shape![]));
        let n: f32 = convert_scalar(N);
        let coef: f32 = rs * rs * dot / n;

        let rs_t: Tile<f32, { [1, BLOCK] }> = rs.broadcast(tile_shape);
        let coef_t: Tile<f32, { [1, BLOCK] }> = coef.broadcast(tile_shape);
        let mut dx_part: PartitionMut<f32, { [1, BLOCK] }> = dx.partition_mut(tile_shape);
        let mut dw_part: PartitionMut<f32, { [1, BLOCK] }> = dw_partial.partition_mut(tile_shape);
        for j in 0i32..num_tiles {
            let tx: Tile<f32, { [1, BLOCK] }> = x_part.load([row, j]);
            let tdy: Tile<f32, { [1, BLOCK] }> = dy_part.load([row, j]);
            let tw: Tile<f32, { [1, BLOCK] }> = w_part.load([j]).reshape(tile_shape);
            dx_part.store(rs_t * (tw * tdy - tx * coef_t), [0i32, j]);
            dw_part.store(tdy * tx * rs_t, [0i32, j]);
        }
    }

    /// Column sums of a `[rows, N]` matrix: `out[c] = sum_r partial[r, c]`.
    /// One tile block per `BLOCK` columns; `out` must be partitioned into
    /// `[BLOCK]` tiles.
    #[cutile::entry()]
    fn column_sum_f32<const N: i32, const BLOCK: i32>(
        partial: &Tensor<f32, { [-1, N] }>,
        out: &mut Tensor<f32, { [BLOCK] }>,
        rows: i32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK] }> = const_shape![1, BLOCK];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let col = pid.0;
        let part: Partition<f32, { [1, BLOCK] }> = partial.partition(tile_shape);
        let mut acc: Tile<f32, { [1, BLOCK] }> = constant(0.0, tile_shape);
        for r in 0i32..rows {
            let t: Tile<f32, { [1, BLOCK] }> = part.load([r, col]);
            acc = acc + t;
        }
        out.store(acc.reshape(const_shape![BLOCK]));
    }

    /// Row-wise softmax over rows of length `N`, one tile block per row.
    /// Matches [`crate::reference::loss::softmax_fwd`].
    #[cutile::entry()]
    fn softmax_f32<const N: i32, const BLOCK: i32>(
        x: &Tensor<f32, { [-1, N] }>,
        y: &mut Tensor<f32, { [1, N] }>,
    ) {
        let tile_shape: Shape<{ [1, BLOCK] }> = const_shape![1, BLOCK];
        let num_tiles: i32 = N / BLOCK;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let x_part: Partition<f32, { [1, BLOCK] }> = x.partition(tile_shape);

        let big: Tile<f32, { [1, BLOCK] }> = constant(1.0e30f32, tile_shape);
        let zero: Tile<f32, { [1, BLOCK] }> = constant(0.0f32, tile_shape);
        let mut m_acc: Tile<f32, { [1, BLOCK] }> = zero - big;
        for j in 0i32..num_tiles {
            let tx: Tile<f32, { [1, BLOCK] }> = x_part.load([row, j]);
            m_acc = max_tile(m_acc, tx);
        }
        let mx: Tile<f32, { [1] }> = reduce_max(m_acc, 1i32);
        let mx: f32 = tile_to_scalar(mx.reshape(const_shape![]));
        let mx_t: Tile<f32, { [1, BLOCK] }> = mx.broadcast(tile_shape);

        let mut s_acc: Tile<f32, { [1, BLOCK] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tx: Tile<f32, { [1, BLOCK] }> = x_part.load([row, j]);
            s_acc = s_acc + exp(tx - mx_t);
        }
        let s: Tile<f32, { [1] }> = reduce_sum(s_acc, 1i32);
        let s: f32 = tile_to_scalar(s.reshape(const_shape![]));
        let inv_t: Tile<f32, { [1, BLOCK] }> = (1.0f32 / s).broadcast(tile_shape);

        let mut y_part: PartitionMut<f32, { [1, BLOCK] }> = y.partition_mut(tile_shape);
        for j in 0i32..num_tiles {
            let tx: Tile<f32, { [1, BLOCK] }> = x_part.load([row, j]);
            y_part.store(exp(tx - mx_t) * inv_t, [0i32, j]);
        }
    }

    /// Fused weighted cross-entropy forward + backward, one tile block per row.
    ///
    /// Writes `loss[row] = w * (lse - logits[target])` and
    /// `dlogits[row] = scale * w * (softmax(logits) - onehot(target))`.
    /// Rows to ignore must have weight 0 (and any in-range target); the host
    /// wrapper maps `IGNORE_INDEX` that way. Matches
    /// [`crate::reference::loss::cross_entropy_fwd`] / `_bwd`.
    #[cutile::entry()]
    fn cross_entropy_f32<const N: i32, const BLOCK: i32>(
        logits: &Tensor<f32, { [-1, N] }>,
        targets: &Tensor<i32, { [-1] }>,
        weights: &Tensor<f32, { [-1] }>,
        loss: &mut Tensor<f32, { [1] }>,
        dlogits: &mut Tensor<f32, { [1, N] }>,
        scale: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK] }> = const_shape![1, BLOCK];
        let num_tiles: i32 = N / BLOCK;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let l_part: Partition<f32, { [1, BLOCK] }> = logits.partition(tile_shape);

        let t_tile: Tile<i32, { [1] }> = targets.partition(const_shape![1]).load([row]);
        let target: i32 = tile_to_scalar(t_tile.reshape(const_shape![]));
        let w_tile: Tile<f32, { [1] }> = weights.partition(const_shape![1]).load([row]);
        let w: f32 = tile_to_scalar(w_tile.reshape(const_shape![]));

        let big: Tile<f32, { [1, BLOCK] }> = constant(1.0e30f32, tile_shape);
        let zero: Tile<f32, { [1, BLOCK] }> = constant(0.0f32, tile_shape);
        let one: Tile<f32, { [1, BLOCK] }> = constant(1.0f32, tile_shape);
        let target_t: Tile<i32, { [1, BLOCK] }> = target.broadcast(tile_shape);
        let offs: Tile<i32, { [BLOCK] }> = iota(const_shape![BLOCK]);
        let offs: Tile<i32, { [1, BLOCK] }> = offs.reshape(tile_shape);

        // Pass 1: row max and the target logit.
        let mut m_acc: Tile<f32, { [1, BLOCK] }> = zero - big;
        let mut t_acc: Tile<f32, { [1, BLOCK] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tx: Tile<f32, { [1, BLOCK] }> = l_part.load([row, j]);
            m_acc = max_tile(m_acc, tx);
            let base: Tile<i32, { [1, BLOCK] }> = (j * BLOCK).broadcast(tile_shape);
            let is_t: Tile<bool, { [1, BLOCK] }> = eq_tile(base + offs, target_t);
            t_acc = t_acc + select(is_t, tx, zero);
        }
        let mx: Tile<f32, { [1] }> = reduce_max(m_acc, 1i32);
        let mx: f32 = tile_to_scalar(mx.reshape(const_shape![]));
        let mx_t: Tile<f32, { [1, BLOCK] }> = mx.broadcast(tile_shape);
        let lt: Tile<f32, { [1] }> = reduce_sum(t_acc, 1i32);
        let logit_t: f32 = tile_to_scalar(lt.reshape(const_shape![]));

        // Pass 2: log-sum-exp.
        let mut s_acc: Tile<f32, { [1, BLOCK] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tx: Tile<f32, { [1, BLOCK] }> = l_part.load([row, j]);
            s_acc = s_acc + exp(tx - mx_t);
        }
        let s: Tile<f32, { [1] }> = reduce_sum(s_acc, 1i32);
        let s_t: Tile<f32, { [] }> = s.reshape(const_shape![]);
        let lse_t: Tile<f32, { [] }> = log(s_t);
        let log_s: f32 = tile_to_scalar(lse_t);
        let lse: f32 = mx + log_s;

        let row_loss: f32 = w * (lse - logit_t);
        let row_loss: Tile<f32, { [1] }> = row_loss.broadcast(const_shape![1]);
        loss.store(row_loss);

        // Pass 3: gradient.
        let lse_bt: Tile<f32, { [1, BLOCK] }> = lse.broadcast(tile_shape);
        let g_t: Tile<f32, { [1, BLOCK] }> = (scale * w).broadcast(tile_shape);
        let mut d_part: PartitionMut<f32, { [1, BLOCK] }> = dlogits.partition_mut(tile_shape);
        for j in 0i32..num_tiles {
            let tx: Tile<f32, { [1, BLOCK] }> = l_part.load([row, j]);
            let base: Tile<i32, { [1, BLOCK] }> = (j * BLOCK).broadcast(tile_shape);
            let is_t: Tile<bool, { [1, BLOCK] }> = eq_tile(base + offs, target_t);
            let onehot: Tile<f32, { [1, BLOCK] }> = select(is_t, one, zero);
            d_part.store(g_t * (exp(tx - lse_bt) - onehot), [0i32, j]);
        }
    }

    /// Fused AdamW step, element-wise. `param`, `m` and `v` are updated in
    /// place and must share one partition of `[B]`. `bc1`/`bc2` are the bias
    /// corrections `1 - beta^step`; `grad_scale` fuses gradient clipping.
    /// Matches [`crate::reference::optim::adamw_step`].
    #[cutile::entry()]
    fn adamw_f32<const B: i32>(
        param: &mut Tensor<f32, { [B] }>,
        m: &mut Tensor<f32, { [B] }>,
        v: &mut Tensor<f32, { [B] }>,
        grad: &Tensor<f32, { [-1] }>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bc1: f32,
        bc2: f32,
        grad_scale: f32,
    ) {
        let shape: Shape<{ [B] }> = const_shape![B];
        let g: Tile<f32, { [B] }> = grad.load_like(param) * grad_scale.broadcast(shape);
        let p0: Tile<f32, { [B] }> = load_tile_mut(param);
        let m0: Tile<f32, { [B] }> = load_tile_mut(m);
        let v0: Tile<f32, { [B] }> = load_tile_mut(v);
        let b1: Tile<f32, { [B] }> = beta1.broadcast(shape);
        let b2: Tile<f32, { [B] }> = beta2.broadcast(shape);
        let one: Tile<f32, { [B] }> = constant(1.0f32, shape);
        let m1: Tile<f32, { [B] }> = b1 * m0 + (one - b1) * g;
        let v1: Tile<f32, { [B] }> = b2 * v0 + (one - b2) * g * g;
        let mhat: Tile<f32, { [B] }> = m1 / bc1.broadcast(shape);
        let vhat: Tile<f32, { [B] }> = v1 / bc2.broadcast(shape);
        let denom: Tile<f32, { [B] }> =
            sqrt(vhat, rounding::NearestEven, ftz::Disabled) + eps.broadcast(shape);
        let update: Tile<f32, { [B] }> = mhat / denom + weight_decay.broadcast(shape) * p0;
        param.store(p0 - lr.broadcast(shape) * update);
        m.store(m1);
        v.store(v1);
    }

    /// SwiGLU forward: `h = gate * sigmoid(gate) * up`, element-wise.
    #[cutile::entry()]
    fn swiglu_fwd_f32<const B: i32>(
        h: &mut Tensor<f32, { [B] }>,
        gate: &Tensor<f32, { [-1] }>,
        up: &Tensor<f32, { [-1] }>,
    ) {
        let g: Tile<f32, { [B] }> = gate.load_like(h);
        let u: Tile<f32, { [B] }> = up.load_like(h);
        let ones: Tile<f32, { [B] }> = broadcast_scalar(1.0f32, h.shape());
        let zeros: Tile<f32, { [B] }> = broadcast_scalar(0.0f32, h.shape());
        let sig: Tile<f32, { [B] }> = ones / (ones + exp(zeros - g));
        h.store(g * sig * u);
    }
}
