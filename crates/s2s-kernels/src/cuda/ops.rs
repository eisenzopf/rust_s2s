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
