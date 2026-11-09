//! Host-side wrappers that run the cuTile kernels on host data.
//!
//! These copy inputs to the device, launch, and copy results back. They exist
//! for parity tests and bring-up, not for the training or inference hot path,
//! which keeps tensors resident on the device.

use std::sync::Arc;

use cutile::api::{copy_host_vec_to_device, zeros};
use cutile::error::Error;
use cutile::prelude::DeviceOp;
use cutile::tensor::{IntoPartition, Reshape, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;

use super::kernels;

fn upload(v: &[f32]) -> Result<Tensor<f32>, Error> {
    Ok(copy_host_vec_to_device(&Arc::new(v.to_vec())).sync()?)
}

/// `x + y` on the GPU. Lengths must match and be a multiple of `block`.
pub fn vec_add(x: &[f32], y: &[f32], block: usize) -> Result<Vec<f32>, Error> {
    assert_eq!(x.len(), y.len());
    assert!(
        x.len().is_multiple_of(block),
        "length must be a multiple of block"
    );
    let z: Tensor<f32> = zeros(&[x.len()]).sync()?;
    let (z, _x, _y) = kernels::vec_add_f32(
        z.partition([block]),
        Arc::new(upload(x)?),
        Arc::new(upload(y)?),
    )
    .sync()?;
    Ok(z.unpartition().to_host_vec().sync()?)
}

/// RMSNorm forward on the GPU. Returns `(y, rstd)`.
///
/// `d` must be a multiple of `block`.
pub fn rms_norm_fwd(
    x: &[f32],
    w: &[f32],
    eps: f32,
    rows: usize,
    d: usize,
    block: usize,
) -> Result<(Vec<f32>, Vec<f32>), Error> {
    assert_eq!(x.len(), rows * d);
    assert_eq!(w.len(), d);
    assert!(
        d.is_multiple_of(block),
        "row length must be a multiple of block"
    );
    let x = Arc::new(upload(x)?.reshape(&[rows, d])?);
    let w = Arc::new(upload(w)?);
    let y: Tensor<f32> = zeros(&[rows, d]).sync()?;
    let rstd: Tensor<f32> = zeros(&[rows]).sync()?;
    let (_x, _w, y, rstd, _eps) =
        kernels::rms_norm_fwd_f32(x, w, y.partition([1, d]), rstd.partition([1]), eps)
            .generics(vec![d.to_string(), block.to_string()])
            .sync()?;
    Ok((
        y.unpartition().to_host_vec().sync()?,
        rstd.unpartition().to_host_vec().sync()?,
    ))
}

/// SwiGLU forward on the GPU. Length must be a multiple of `block`.
pub fn swiglu_fwd(gate: &[f32], up: &[f32], block: usize) -> Result<Vec<f32>, Error> {
    assert_eq!(gate.len(), up.len());
    assert!(
        gate.len().is_multiple_of(block),
        "length must be a multiple of block"
    );
    let h: Tensor<f32> = zeros(&[gate.len()]).sync()?;
    let (h, _g, _u) = kernels::swiglu_fwd_f32(
        h.partition([block]),
        Arc::new(upload(gate)?),
        Arc::new(upload(up)?),
    )
    .sync()?;
    Ok(h.unpartition().to_host_vec().sync()?)
}

fn upload_i32(v: &[i32]) -> Result<Tensor<i32>, Error> {
    Ok(copy_host_vec_to_device(&Arc::new(v.to_vec())).sync()?)
}

/// RMSNorm backward on the GPU. Returns `(dx, dw)`; `dw` is the column sum
/// of the per-row partial gradients (a second kernel).
pub fn rms_norm_bwd(
    x: &[f32],
    w: &[f32],
    rstd: &[f32],
    dy: &[f32],
    rows: usize,
    d: usize,
    block: usize,
) -> Result<(Vec<f32>, Vec<f32>), Error> {
    assert!(
        d.is_multiple_of(block),
        "row length must be a multiple of block"
    );
    let generics = vec![d.to_string(), block.to_string()];
    let x = Arc::new(upload(x)?.reshape(&[rows, d])?);
    let dy = Arc::new(upload(dy)?.reshape(&[rows, d])?);
    let dx: Tensor<f32> = zeros(&[rows, d]).sync()?;
    let dwp: Tensor<f32> = zeros(&[rows, d]).sync()?;
    let (_x, _w, _r, _dy, dx, dwp) = kernels::rms_norm_bwd_f32(
        x,
        Arc::new(upload(w)?),
        Arc::new(upload(rstd)?),
        dy,
        dx.partition([1, d]),
        dwp.partition([1, d]),
    )
    .generics(generics.clone())
    .sync()?;
    let dw: Tensor<f32> = zeros(&[d]).sync()?;
    let (_p, dw, _rows) = kernels::column_sum_f32(
        Arc::new(dwp.unpartition()),
        dw.partition([block]),
        rows as i32,
    )
    .generics(generics)
    .sync()?;
    Ok((
        dx.unpartition().to_host_vec().sync()?,
        dw.unpartition().to_host_vec().sync()?,
    ))
}

/// Row-wise softmax on the GPU. `v` must be a multiple of `block`.
pub fn softmax(x: &[f32], rows: usize, v: usize, block: usize) -> Result<Vec<f32>, Error> {
    assert!(
        v.is_multiple_of(block),
        "row length must be a multiple of block"
    );
    let x = Arc::new(upload(x)?.reshape(&[rows, v])?);
    let y: Tensor<f32> = zeros(&[rows, v]).sync()?;
    let (_x, y) = kernels::softmax_f32(x, y.partition([1, v]))
        .generics(vec![v.to_string(), block.to_string()])
        .sync()?;
    Ok(y.unpartition().to_host_vec().sync()?)
}

/// Fused weighted cross-entropy forward + backward on the GPU.
///
/// Same contract as the CPU reference: `targets` may hold
/// `reference::loss::IGNORE_INDEX`; returns `(weighted_sum, total_weight,
/// dlogits)` where `dlogits = scale * w * (softmax - onehot)`.
pub fn cross_entropy(
    logits: &[f32],
    targets: &[i64],
    weights: Option<&[f32]>,
    scale: f32,
    rows: usize,
    v: usize,
    block: usize,
) -> Result<(f32, f32, Vec<f32>), Error> {
    use crate::reference::loss::IGNORE_INDEX;
    assert!(
        v.is_multiple_of(block),
        "row length must be a multiple of block"
    );
    // Ignored rows become (target 0, weight 0): zero loss and zero gradient.
    let mut t32 = Vec::with_capacity(rows);
    let mut w = Vec::with_capacity(rows);
    for r in 0..rows {
        let wr = weights.map_or(1.0, |ws| ws[r]);
        if targets[r] == IGNORE_INDEX {
            t32.push(0);
            w.push(0.0);
        } else {
            assert!(
                targets[r] >= 0 && (targets[r] as usize) < v,
                "target out of range"
            );
            t32.push(targets[r] as i32);
            w.push(wr);
        }
    }
    let total: f32 = w.iter().sum();
    let logits = Arc::new(upload(logits)?.reshape(&[rows, v])?);
    let loss: Tensor<f32> = zeros(&[rows]).sync()?;
    let dl: Tensor<f32> = zeros(&[rows, v]).sync()?;
    let (_l, _t, _w, loss, dl, _s) = kernels::cross_entropy_f32(
        logits,
        Arc::new(upload_i32(&t32)?),
        Arc::new(upload(&w)?),
        loss.partition([1]),
        dl.partition([1, v]),
        scale,
    )
    .generics(vec![v.to_string(), block.to_string()])
    .sync()?;
    let loss_rows = loss.unpartition().to_host_vec().sync()?;
    Ok((
        loss_rows.iter().sum(),
        total,
        dl.unpartition().to_host_vec().sync()?,
    ))
}

/// Updated `(param, m, v)` after an AdamW step.
pub type AdamState = (Vec<f32>, Vec<f32>, Vec<f32>);

/// One fused AdamW step on the GPU. `step` is 1-based. Returns the updated
/// `(param, m, v)`.
pub fn adamw_step(
    param: &[f32],
    grad: &[f32],
    m: &[f32],
    v: &[f32],
    step: u64,
    cfg: &crate::reference::optim::AdamWConfig,
    grad_scale: f32,
    block: usize,
) -> Result<AdamState, Error> {
    let n = param.len();
    assert!(
        n.is_multiple_of(block),
        "length must be a multiple of block"
    );
    let bc1 = (1.0 - cfg.beta1.powi(step as i32)) as f32;
    let bc2 = (1.0 - cfg.beta2.powi(step as i32)) as f32;
    let (p, m, v, ..) = kernels::adamw_f32(
        upload(param)?.partition([block]),
        upload(m)?.partition([block]),
        upload(v)?.partition([block]),
        Arc::new(upload(grad)?),
        cfg.lr as f32,
        cfg.beta1 as f32,
        cfg.beta2 as f32,
        cfg.eps as f32,
        cfg.weight_decay as f32,
        bc1,
        bc2,
        grad_scale,
    )
    .sync()?;
    Ok((
        p.unpartition().to_host_vec().sync()?,
        m.unpartition().to_host_vec().sync()?,
        v.unpartition().to_host_vec().sync()?,
    ))
}
