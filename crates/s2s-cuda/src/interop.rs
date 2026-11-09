//! Zero-copy hand-off of cudarc buffers to cuTile tensors.
//!
//! This follows NVIDIA's documented pattern (`cudarc_interop.rs` in the
//! cuTile examples): the foreign buffer implements
//! `cuda_core::DeviceAllocation`, and `Tensor::from_foreign` wraps it without
//! a copy, holding an `Arc` to it so the memory outlives every cuTile use.
//!
//! The API here is safe because it enforces the one obligation
//! `from_foreign` cannot check, exclusive access while cuTile uses the
//! memory:
//!
//! - [`TileContext::adopt`] **moves** the [`CudaSlice`] in, finishes cudarc's
//!   pending work on it, and returns the only tensor over it;
//! - the slice comes back only through [`ForeignBuffer::try_into_slice`],
//!   which succeeds once that tensor (and everything cuTile derived from it)
//!   has been dropped, after waiting for the device to go idle.
//!
//! Both libraries must use the device's **primary context** (cudarc's
//! `CudaContext::new` and cuTile's `Context::new` both retain it), which
//! [`TileContext::new`] checks.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, DriverError};
use cutile::cuda_core;
use cutile::tensor::Tensor;

use crate::raw::{self, CudarcAllocation};

/// Interop failure.
#[derive(Debug)]
pub enum InteropError {
    /// The shape does not match the buffer, or does not fit cuTile's `i32`
    /// dimensions.
    Shape(String),
    /// The cudarc context is not the primary context cuTile uses.
    Context(String),
    /// cudarc driver error.
    Driver(DriverError),
    /// cuTile / `cuda_core` driver error.
    Tile(cuda_core::DriverError),
}

impl std::fmt::Display for InteropError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InteropError::Shape(s) => write!(f, "interop shape: {s}"),
            InteropError::Context(s) => write!(f, "interop context: {s}"),
            InteropError::Driver(e) => write!(f, "CUDA driver (cudarc): {e:?}"),
            InteropError::Tile(e) => write!(f, "CUDA driver (cuTile): {e:?}"),
        }
    }
}

impl std::error::Error for InteropError {}

impl From<DriverError> for InteropError {
    fn from(e: DriverError) -> Self {
        InteropError::Driver(e)
    }
}

impl From<cuda_core::DriverError> for InteropError {
    fn from(e: cuda_core::DriverError) -> Self {
        InteropError::Tile(e)
    }
}

mod sealed {
    pub trait Sealed {}
    macro_rules! sealed {
        ($($t:ty),*) => { $(impl Sealed for $t {})* };
    }
    sealed!(f32, f64, half::bf16, half::f16, i8, u8, i32, u32, i64, u64);
}

/// Element types that can be shared: plain numbers, for which every bit
/// pattern is a valid value (so cuTile may read memory cudarc wrote, or never
/// initialized). Sealed.
pub trait InteropElem: sealed::Sealed + cuda_core::DType + Send + Sync + 'static {}
impl<T: sealed::Sealed + cuda_core::DType + Send + Sync + 'static> InteropElem for T {}

/// cuTile's handle to the device a cudarc context runs on.
#[derive(Debug, Clone)]
pub struct TileContext {
    tile: Arc<cuda_core::Context>,
    cu_ctx: usize,
}

impl TileContext {
    /// Retain the primary context of `ctx`'s device for cuTile, after
    /// checking that `ctx` is that primary context.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, InteropError> {
        if !ctx.is_primary() {
            return Err(InteropError::Context(
                "cudarc context is not the device's primary context".into(),
            ));
        }
        let tile = cuda_core::Context::new(ctx.ordinal())?;
        let (ours, theirs) = (tile.cu_ctx() as usize, ctx.cu_ctx() as usize);
        if ours != theirs {
            return Err(InteropError::Context(format!(
                "cuTile primary context {ours:#x} differs from cudarc's {theirs:#x}"
            )));
        }
        Ok(Self { tile, cu_ctx: ours })
    }

    /// cuTile's context (for streams: `context().new_stream()`).
    pub fn context(&self) -> &Arc<cuda_core::Context> {
        &self.tile
    }

    /// Lend `slice` to cuTile as a contiguous row-major tensor of `shape`,
    /// with no copy. cudarc's pending work on the slice is finished first.
    ///
    /// Returns the tensor and a [`ForeignBuffer`] to take the slice back
    /// once the tensor is gone.
    pub fn adopt<T: InteropElem>(
        &self,
        slice: CudaSlice<T>,
        shape: &[usize],
    ) -> Result<(Tensor<T>, ForeignBuffer<T>), InteropError> {
        let theirs = slice.context().cu_ctx() as usize;
        if theirs != self.cu_ctx {
            return Err(InteropError::Context(format!(
                "buffer lives in context {theirs:#x}, not this device's primary context {:#x}",
                self.cu_ctx
            )));
        }
        let (tensor, alloc) = raw::adopt(self.tile.clone(), slice, shape)?;
        Ok((tensor, ForeignBuffer(alloc)))
    }
}

/// A cudarc buffer currently lent to cuTile (see [`TileContext::adopt`]).
#[derive(Debug)]
pub struct ForeignBuffer<T>(Arc<CudarcAllocation<T>>);

impl<T> ForeignBuffer<T> {
    /// Elements in the buffer.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether cuTile still holds the buffer.
    pub fn in_use(&self) -> bool {
        Arc::strong_count(&self.0) > 1
    }

    /// The cudarc slice back, once cuTile no longer holds it; otherwise
    /// `self` unchanged. Waits for the device to finish outstanding work.
    pub fn try_into_slice(self) -> Result<CudaSlice<T>, Self> {
        CudarcAllocation::into_slice(self.0).map_err(ForeignBuffer)
    }
}
