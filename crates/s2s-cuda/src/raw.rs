//! The crate's only `unsafe` code: cuBLASLt FFI calls and the cuTile
//! [`DeviceAllocation`] impl for cudarc buffers. Every function exported from
//! here is safe; the checks that make it so are next to each `unsafe` block.

use std::mem::size_of;
use std::sync::Arc;

use cudarc::cublaslt::result as lt;
use cudarc::cublaslt::sys;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use cutile::cuda_core::{self, DeviceAllocation};
use cutile::tensor::Tensor;

use crate::gemm::{GemmElem, GemmError};
use crate::interop::{InteropElem, InteropError};
use crate::layout::{ColMajorLayout, LtProblem};

// ---------------------------------------------------------------- cuBLASLt

/// An owned `cublasLtHandle_t`.
#[derive(Debug)]
pub(crate) struct LtHandle(sys::cublasLtHandle_t);

// SAFETY: a cuBLASLt handle is a heap object owned by this wrapper; cuBLASLt
// allows using it from any host thread. `LtHandle` is not `Sync`, so calls on
// one handle never overlap.
unsafe impl Send for LtHandle {}

impl LtHandle {
    pub(crate) fn new() -> Result<Self, GemmError> {
        Ok(Self(lt::create_handle()?))
    }
}

impl Drop for LtHandle {
    fn drop(&mut self) {
        // SAFETY: the handle came from `cublasLtCreate` and is destroyed once.
        // A destructor cannot report a failure, so it is ignored.
        let _ = unsafe { lt::destroy_handle(self.0) };
    }
}

struct Layout(sys::cublasLtMatrixLayout_t);

impl Layout {
    fn new(dtype: sys::cudaDataType, l: &ColMajorLayout) -> Result<Self, GemmError> {
        Ok(Self(lt::create_matrix_layout(dtype, l.rows, l.cols, l.ld)?))
    }
}

impl Drop for Layout {
    fn drop(&mut self) {
        // SAFETY: created by `cublasLtMatrixLayoutCreate`, destroyed once.
        let _ = unsafe { lt::destroy_matrix_layout(self.0) };
    }
}

struct Desc(sys::cublasLtMatmulDesc_t);

impl Desc {
    fn set<V>(
        &self,
        attr: sys::cublasLtMatmulDescAttributes_t,
        value: &V,
    ) -> Result<(), GemmError> {
        // SAFETY: `self.0` is a live descriptor; the buffer is a `V` that
        // lives across the call, and `size_of::<V>()` is its size. cuBLASLt
        // copies the value. Callers pass the attribute's documented type.
        unsafe {
            lt::set_matmul_desc_attribute(self.0, attr, (value as *const V).cast(), size_of::<V>())?
        };
        Ok(())
    }
}

impl Drop for Desc {
    fn drop(&mut self) {
        // SAFETY: created by `cublasLtMatmulDescCreate`, destroyed once.
        let _ = unsafe { lt::destroy_matmul_desc(self.0) };
    }
}

struct Pref(sys::cublasLtMatmulPreference_t);

impl Drop for Pref {
    fn drop(&mut self) {
        // SAFETY: created by `cublasLtMatmulPreferenceCreate`, destroyed once.
        let _ = unsafe { lt::destroy_matmul_pref(self.0) };
    }
}

fn dtype_bytes(dt: sys::cudaDataType) -> usize {
    match dt {
        sys::cudaDataType::CUDA_R_32F => 4,
        sys::cudaDataType::CUDA_R_16BF | sys::cudaDataType::CUDA_R_16F => 2,
        _ => 0,
    }
}

fn check_operand<T: GemmElem>(what: &str, len: usize, l: &ColMajorLayout) -> Result<(), GemmError> {
    if size_of::<T>() != dtype_bytes(T::DATA_TYPE) {
        return Err(GemmError::Shape(format!(
            "{what}: element type does not match its cuBLASLt data type"
        )));
    }
    if l.ld < l.rows.max(1) as i64 {
        return Err(GemmError::Shape(format!(
            "{what}: leading dimension below rows"
        )));
    }
    if len < l.required_len() {
        return Err(GemmError::Shape(format!(
            "{what}: buffer has {len} elements, layout needs {}",
            l.required_len()
        )));
    }
    Ok(())
}

/// Run `D = alpha * op(A) op(B) + beta * C (+ bias)` on `stream`, with `D`
/// written in place over `C`. `a` and `b` are cuBLASLt's operands (see
/// [`LtProblem`]); `bias` has cuBLASLt-`m` elements.
///
/// Every buffer extent is checked against its layout before the call, so
/// cuBLASLt never addresses memory outside the given buffers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lt_matmul<I: GemmElem, O: GemmElem>(
    handle: &LtHandle,
    stream: &Arc<CudaStream>,
    workspace: &mut CudaSlice<u8>,
    p: &LtProblem,
    compute: sys::cublasComputeType_t,
    alpha: f32,
    beta: f32,
    a: &impl DevicePtr<I>,
    b: &impl DevicePtr<I>,
    c: &mut impl DevicePtrMut<O>,
    bias: Option<&CudaSlice<O>>,
) -> Result<(), GemmError> {
    check_operand::<I>("A operand", a.len(), &p.a)?;
    check_operand::<I>("B operand", b.len(), &p.b)?;
    check_operand::<O>("C", c.len(), &p.c)?;
    // The A/B/C layouts must agree on m, n, k, or cuBLASLt would read a
    // different extent than the one checked above.
    let (a_rows, a_cols) = if p.trans_a { (p.k, p.m) } else { (p.m, p.k) };
    let (b_rows, b_cols) = if p.trans_b { (p.n, p.k) } else { (p.k, p.n) };
    if (p.a.rows, p.a.cols) != (a_rows, a_cols)
        || (p.b.rows, p.b.cols) != (b_rows, b_cols)
        || (p.c.rows, p.c.cols) != (p.m, p.n)
    {
        return Err(GemmError::Shape("layouts disagree with m, n, k".into()));
    }
    if let Some(bias) = bias {
        if (bias.len() as u64) < p.m {
            return Err(GemmError::Shape(format!(
                "bias has {} elements, needs {}",
                bias.len(),
                p.m
            )));
        }
    }

    stream.context().bind_to_thread()?;
    let la = Layout::new(I::DATA_TYPE, &p.a)?;
    let lb = Layout::new(I::DATA_TYPE, &p.b)?;
    let lc = Layout::new(O::DATA_TYPE, &p.c)?;
    let desc = Desc(lt::create_matmul_desc(
        compute,
        sys::cudaDataType::CUDA_R_32F,
    )?);
    // `cublasOperation_t` (a C enum: CUBLAS_OP_N = 0, CUBLAS_OP_T = 1) is not
    // in cudarc's cuBLASLt bindings; pass its 4-byte integer value.
    let op = |t: bool| -> i32 { i32::from(t) };
    desc.set(
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
        &op(p.trans_a),
    )?;
    desc.set(
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
        &op(p.trans_b),
    )?;
    // Keep the bias read guard alive until the matmul is enqueued.
    let bias_guard = match bias {
        Some(bias) => {
            let (ptr, guard) = bias.device_ptr(stream);
            desc.set(
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                &sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
            )?;
            desc.set(
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &ptr,
            )?;
            desc.set(
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                &O::DATA_TYPE,
            )?;
            Some(guard)
        }
        None => None,
    };

    let pref = Pref(lt::create_matmul_pref()?);
    let ws_bytes = workspace.len();
    // SAFETY: `pref.0` is live; the value is a `usize`, the attribute's type.
    unsafe {
        lt::set_matmul_pref_attribute(
            pref.0,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&ws_bytes as *const usize).cast(),
            size_of::<usize>(),
        )?
    };
    // SAFETY: all handles are live objects created above.
    let heuristic =
        unsafe { lt::get_matmul_algo_heuristic(handle.0, desc.0, la.0, lb.0, lc.0, lc.0, pref.0)? };

    let (pa, _ga) = a.device_ptr(stream);
    let (pb, _gb) = b.device_ptr(stream);
    let (pc, _gc) = c.device_ptr_mut(stream);
    let (pw, _gw) = workspace.device_ptr_mut(stream);
    // SAFETY: every device pointer comes from a live cudarc buffer whose
    // extent was checked against its layout above (the bias against `m`);
    // the read/write guards keep cudarc's stream ordering until the call is
    // enqueued. `D` aliases `C` with the same layout, which cuBLASLt allows.
    // `alpha`/`beta` are f32, matching the f32 scale type. The workspace
    // pointer covers `ws_bytes`, the size the heuristic was asked to fit.
    unsafe {
        lt::matmul(
            handle.0,
            desc.0,
            (&alpha as *const f32).cast(),
            (&beta as *const f32).cast(),
            pa as *const _,
            la.0,
            pb as *const _,
            lb.0,
            pc as *const _,
            lc.0,
            pc as *mut _,
            lc.0,
            &heuristic.algo,
            pw as *mut _,
            ws_bytes,
            stream.cu_stream() as *mut _,
        )?
    };
    drop(bias_guard);
    Ok(())
}

// ------------------------------------------------------------ cuTile interop

/// A cudarc device buffer lent to cuTile with no copy.
///
/// Owns the [`CudaSlice`] for as long as cuTile may use it. Nothing outside
/// this module can reach the memory while a tensor exists: the slice comes
/// back only through [`into_slice`](Self::into_slice), which requires the
/// tensor (the other `Arc` holder) to be gone.
#[derive(Debug)]
pub struct CudarcAllocation<T> {
    slice: Option<CudaSlice<T>>,
    ptr: u64,
    len_bytes: usize,
    ctx: Arc<cuda_core::Context>,
}

impl<T> CudarcAllocation<T> {
    /// Wait for all work on the device (cuTile's streams are not ordered
    /// with cudarc's), so no kernel still touches the memory.
    fn quiesce(&self) {
        if let Some(s) = &self.slice {
            let _ = s.context().synchronize();
        }
    }

    /// The cudarc slice back, once no cuTile tensor holds this allocation.
    pub(crate) fn into_slice(this: Arc<Self>) -> Result<CudaSlice<T>, Arc<Self>> {
        let mut me = Arc::try_unwrap(this)?;
        me.quiesce();
        Ok(me.slice.take().expect("slice present until taken"))
    }

    /// Elements in the buffer.
    pub(crate) fn len(&self) -> usize {
        self.slice.as_ref().map_or(0, |s| s.len())
    }
}

impl<T> Drop for CudarcAllocation<T> {
    fn drop(&mut self) {
        // cudarc frees on its own stream; make sure cuTile work is done.
        self.quiesce();
    }
}

// SAFETY: `ptr` is the device address of `slice`, a live allocation of
// `len_bytes` bytes that this struct owns until it is dropped or handed back
// by `into_slice` (which requires sole ownership, i.e. no cuTile tensor
// left). It lives in the device's primary context, and `ctx` is a retained
// handle to that same primary context (checked in `adopt`), so the pointer is
// valid in `ctx`, which stays alive as long as this struct.
unsafe impl<T: Send + Sync + 'static> DeviceAllocation for CudarcAllocation<T> {
    fn device_ptr(&self) -> cuda_core::sys::CUdeviceptr {
        self.ptr as _
    }
    fn len_bytes(&self) -> usize {
        self.len_bytes
    }
    fn context(&self) -> Arc<cuda_core::Context> {
        self.ctx.clone()
    }
    fn device_id(&self) -> usize {
        self.ctx.ordinal()
    }
}

/// Move `slice` into a new allocation and view it as one contiguous cuTile
/// tensor of `shape`. Returns the tensor and the allocation (to recover the
/// slice later).
///
/// `ctx` must be a retained handle to the primary context of the slice's
/// device; the caller (`interop::TileContext`) checks that.
pub(crate) fn adopt<T: InteropElem>(
    ctx: Arc<cuda_core::Context>,
    slice: CudaSlice<T>,
    shape: &[usize],
) -> Result<(Tensor<T>, Arc<CudarcAllocation<T>>), InteropError> {
    let numel: usize = shape.iter().product();
    if numel != slice.len() {
        return Err(InteropError::Shape(format!(
            "shape {shape:?} has {numel} elements, buffer has {}",
            slice.len()
        )));
    }
    let to_i32 = |x: usize| {
        i32::try_from(x).map_err(|_| InteropError::Shape(format!("{x} does not fit i32")))
    };
    let dims = shape
        .iter()
        .map(|&d| to_i32(d))
        .collect::<Result<Vec<_>, _>>()?;
    let mut strides = vec![1i32; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1]
            .checked_mul(dims[i + 1])
            .ok_or_else(|| InteropError::Shape("strides overflow i32".into()))?;
    }
    // Finish cudarc's pending work on the buffer, then read its address. The
    // guard records the access, and after this cudarc never sees the slice
    // again until `into_slice`.
    slice.stream().synchronize()?;
    let ptr = {
        let (ptr, _guard) = slice.device_ptr(slice.stream());
        ptr
    };
    let len_bytes = slice.num_bytes();
    let alloc = Arc::new(CudarcAllocation {
        slice: Some(slice),
        ptr,
        len_bytes,
        ctx,
    });
    // SAFETY: the temporal obligation of `from_foreign`: exclusive access for
    // the tensor's lifetime. The allocation owns the only handle to this
    // memory (the slice was moved in), and the crate creates exactly one
    // tensor per allocation, here. cudarc's last access (its pending stream
    // work) completed in the synchronize above. Every bit pattern is a valid
    // `T` (`InteropElem` is implemented only for plain numeric types).
    let tensor = unsafe { Tensor::from_foreign(alloc.clone(), dims, strides) };
    Ok((tensor, alloc))
}
