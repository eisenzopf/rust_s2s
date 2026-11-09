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
