//! cuBLASLt GEMMs in the row-major layouts of
//! [`s2s_kernels::reference::gemm`].
//!
//! [`Gemm::gemm`] has the reference signature
//! (`C = alpha * op(A) @ op(B) + beta * C`, row-major, [`Trans`] per operand)
//! plus an optional bias added to every row (the cuBLASLt `BIAS` epilogue).
//! Supported element types, all with f32 accumulation and f32 `alpha`/`beta`:
//!
//! | A, B | C | Compute |
//! |---|---|---|
//! | `f32` | `f32` | `CUBLAS_COMPUTE_32F` ([`F32Math::Strict`], default) or `32F_FAST_TF32` ([`F32Math::Tf32`]) |
//! | `bf16` | `f32` | `CUBLAS_COMPUTE_32F` |
//! | `bf16` | `bf16` | `CUBLAS_COMPUTE_32F` |
//!
//! The bias has the element type of `C`. See [`crate::layout`] for how the
//! row-major problem maps onto cuBLASLt's column-major one.

use std::sync::Arc;

use cudarc::cublaslt::result::CublasError;
use cudarc::cublaslt::sys;
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, DriverError};
use half::bf16;

pub use s2s_kernels::reference::gemm::Trans;

use crate::layout::LtProblem;
use crate::raw::{self, LtHandle};

/// GEMM failure.
#[derive(Debug)]
pub enum GemmError {
    /// Buffer lengths or dimensions do not fit the problem.
    Shape(String),
    /// cuBLASLt returned an error (including "no algorithm for this problem").
    Cublas(CublasError),
    /// CUDA driver error.
    Driver(DriverError),
}

impl std::fmt::Display for GemmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GemmError::Shape(s) => write!(f, "gemm shape: {s}"),
            GemmError::Cublas(e) => write!(f, "cuBLASLt: {e:?}"),
            GemmError::Driver(e) => write!(f, "CUDA driver: {e:?}"),
        }
    }
}

impl std::error::Error for GemmError {}

impl From<CublasError> for GemmError {
    fn from(e: CublasError) -> Self {
        GemmError::Cublas(e)
    }
}

impl From<DriverError> for GemmError {
    fn from(e: DriverError) -> Self {
        GemmError::Driver(e)
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for half::bf16 {}
}

/// An element type cuBLASLt reads or writes here (`f32`, `bf16`). Sealed.
pub trait GemmElem: sealed::Sealed + DeviceRepr {
    /// The cuBLASLt data type.
    const DATA_TYPE: sys::cudaDataType;
}

impl GemmElem for f32 {
    const DATA_TYPE: sys::cudaDataType = sys::cudaDataType::CUDA_R_32F;
}

impl GemmElem for bf16 {
    const DATA_TYPE: sys::cudaDataType = sys::cudaDataType::CUDA_R_16BF;
}

/// Input element types allowed with output type `O`.
pub trait GemmInput<O: GemmElem>: GemmElem {}
impl GemmInput<f32> for f32 {}
impl GemmInput<f32> for bf16 {}
impl GemmInput<bf16> for bf16 {}

/// Arithmetic for f32 inputs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum F32Math {
    /// Full f32 multiplies (`CUBLAS_COMPUTE_32F`); matches the CPU reference
    /// to f32 rounding.
    #[default]
    Strict,
    /// TF32 tensor cores (`CUBLAS_COMPUTE_32F_FAST_TF32`): ~10-bit mantissa
    /// products, f32 accumulation. Much faster on sm_80+.
    Tf32,
}

/// A cuBLASLt handle, a workspace and the stream GEMMs run on.
#[derive(Debug)]
pub struct Gemm {
    handle: LtHandle,
    workspace: CudaSlice<u8>,
    stream: Arc<CudaStream>,
    f32_math: F32Math,
}

impl Gemm {
    /// A handle for `stream`, with NVIDIA's recommended workspace (32 MiB on
    /// Hopper and newer, 4 MiB otherwise).
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, GemmError> {
        let ctx = stream.context();
        ctx.bind_to_thread()?;
        let major =
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let bytes = if major >= 9 { 32 << 20 } else { 4 << 20 };
        let workspace = stream.alloc_zeros::<u8>(bytes)?;
        Ok(Self {
            handle: LtHandle::new()?,
            workspace,
            stream,
            f32_math: F32Math::default(),
        })
    }

    /// Choose the arithmetic for f32 inputs.
    pub fn with_f32_math(mut self, math: F32Math) -> Self {
        self.f32_math = math;
        self
    }

    /// The stream GEMMs are enqueued on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// `C = alpha * op(A) @ op(B) + beta * C (+ bias)`, row-major, enqueued
    /// on [`stream`](Self::stream).
    ///
    /// Shapes as in [`reference::gemm::gemm`](s2s_kernels::reference::gemm::gemm):
    /// `op(A)` is `[m, k]`, `op(B)` is `[k, n]`, `C` is `[m, n]`. `bias`, if
    /// given, has `n` elements and is added to every row of `C` after the
    /// `alpha`/`beta` combination. `m == 0` or `n == 0` is a no-op; `k` must
    /// be positive.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm<I: GemmInput<O>, O: GemmElem>(
        &mut self,
        ta: Trans,
        tb: Trans,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &impl DevicePtr<I>,
        b: &impl DevicePtr<I>,
        beta: f32,
        c: &mut impl DevicePtrMut<O>,
        bias: Option<&CudaSlice<O>>,
    ) -> Result<(), GemmError> {
        let check = |what: &str, len: usize, want: usize| {
            if len == want {
                Ok(())
            } else {
                Err(GemmError::Shape(format!(
                    "{what} has {len} elements, expected {want}"
                )))
            }
        };
        check("A", a.len(), m * k)?;
        check("B", b.len(), k * n)?;
        check("C", c.len(), m * n)?;
        if let Some(bias) = bias {
            check("bias", bias.len(), n)?;
        }
        if m == 0 || n == 0 {
            return Ok(());
        }
        if k == 0 {
            return Err(GemmError::Shape("k must be positive".into()));
        }
        let compute =
            if I::DATA_TYPE == sys::cudaDataType::CUDA_R_32F && self.f32_math == F32Math::Tf32 {
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
            } else {
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
            };
        let p = LtProblem::row_major(ta, tb, m, n, k);
        // cuBLASLt's A operand is our B (see `crate::layout`).
        raw::lt_matmul(
            &self.handle,
            &self.stream,
            &mut self.workspace,
            &p,
            compute,
            alpha,
            beta,
            b,
            a,
            c,
            bias,
        )
    }

    /// `Linear` forward like
    /// [`reference::gemm::linear_fwd`](s2s_kernels::reference::gemm::linear_fwd):
    /// `y[rows, out] = x[rows, in] @ w[out, in]^T (+ bias[out])`, one fused
    /// cuBLASLt call.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_fwd<I: GemmInput<O>, O: GemmElem>(
        &mut self,
        x: &impl DevicePtr<I>,
        w: &impl DevicePtr<I>,
        bias: Option<&CudaSlice<O>>,
        y: &mut impl DevicePtrMut<O>,
        rows: usize,
        d_in: usize,
        d_out: usize,
    ) -> Result<(), GemmError> {
        self.gemm(
            Trans::No,
            Trans::Yes,
            rows,
            d_out,
            d_in,
            1.0,
            x,
            w,
            0.0,
            y,
            bias,
        )
    }
}
