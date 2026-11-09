//! [`Communicator`] over NCCL, via cudarc's safe NCCL API.
//!
//! [`NcclComm`] runs the host-slice collectives of [`Communicator`] by
//! staging through device memory on its stream (copy in, collective, copy
//! out), so [`s2s_train::dist`] (DDP, ZeRO-1, sharded checkpoints) runs
//! unchanged on GPUs. Device-resident training calls the NCCL collectives on
//! device buffers directly: see [`NcclComm::inner`] and
//! [`NcclComm::all_reduce_sum_device`].
//!
//! One `NcclComm` per rank, usually one process (or thread) per GPU. Ranks
//! agree on an [`Id`] out of band: rank 0 creates it with [`Id::new`] and
//! sends its bytes ([`Id::internal`]) to the others, which rebuild it with
//! [`Id::uninit`].

use std::sync::Arc;

use cudarc::driver::{CudaStream, DevicePtrMut, DriverError};
use cudarc::nccl::result::NcclError;
use cudarc::nccl::{Comm, NcclType, ReduceOp};
use s2s_train::comm::{CommError, CommResult, Communicator};

pub use cudarc::nccl::Id;

fn nccl_err(what: &str) -> impl Fn(NcclError) -> CommError + '_ {
    move |e| CommError(format!("NCCL {what}: {e:?}"))
}

fn driver_err(e: DriverError) -> CommError {
    CommError(format!("CUDA driver: {e:?}"))
}

/// One rank of an NCCL communicator.
#[derive(Debug)]
pub struct NcclComm {
    comm: Comm,
    stream: Arc<CudaStream>,
}

impl NcclComm {
    /// Join the group `id` as `rank` of `world`, running collectives on
    /// `stream`. Blocks until all ranks have joined.
    pub fn new(stream: Arc<CudaStream>, rank: usize, world: usize, id: Id) -> CommResult<Self> {
        let comm = Comm::from_rank(stream.clone(), rank, world, id).map_err(nccl_err("init"))?;
        Ok(Self { comm, stream })
    }

    /// A group of one (useful for tests and single-GPU runs).
    pub fn single(stream: Arc<CudaStream>) -> CommResult<Self> {
        let id = Id::new().map_err(nccl_err("unique id"))?;
        Self::new(stream, 0, 1, id)
    }

    /// The cudarc communicator, for collectives on device buffers.
    pub fn inner(&self) -> &Comm {
        &self.comm
    }

    /// The stream collectives run on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// In-place sum over ranks of a device buffer (f32, bf16, ...), enqueued
    /// on [`stream`](Self::stream).
    pub fn all_reduce_sum_device<T: NcclType>(&self, buf: &mut impl DevicePtrMut<T>) -> CommResult {
        self.comm
            .all_reduce_in_place(buf, &ReduceOp::Sum)
            .map_err(nccl_err("all_reduce"))?;
        Ok(())
    }

    fn sum_in_place<T: NcclType + cudarc::driver::DeviceRepr>(&self, buf: &mut [T]) -> CommResult {
        if buf.is_empty() {
            return Ok(());
        }
        let mut d = self.stream.clone_htod(buf).map_err(driver_err)?;
        self.all_reduce_sum_device(&mut d)?;
        self.stream.memcpy_dtoh(&d, buf).map_err(driver_err)?;
        self.stream.synchronize().map_err(driver_err)
    }
}

impl Communicator for NcclComm {
    fn rank(&self) -> usize {
        self.comm.rank()
    }

    fn world_size(&self) -> usize {
        self.comm.world_size()
    }

    fn all_reduce_sum(&self, buf: &mut [f32]) -> CommResult {
        self.sum_in_place(buf)
    }

    fn all_reduce_sum_f64(&self, buf: &mut [f64]) -> CommResult {
        self.sum_in_place(buf)
    }

    fn reduce_scatter(&self, input: &[f32], output: &mut [f32]) -> CommResult {
        let world = self.world_size();
        if input.len() != world * output.len() {
            return Err(CommError(format!(
                "reduce_scatter: input has {} elements, expected world {world} x chunk {}",
                input.len(),
                output.len()
            )));
        }
        if output.is_empty() {
            return Ok(());
        }
        let send = self.stream.clone_htod(input).map_err(driver_err)?;
        let mut recv = self
            .stream
            .alloc_zeros::<f32>(output.len())
            .map_err(driver_err)?;
        self.comm
            .reduce_scatter(&send, &mut recv, &ReduceOp::Sum)
            .map_err(nccl_err("reduce_scatter"))?;
        self.stream.memcpy_dtoh(&recv, output).map_err(driver_err)?;
        self.stream.synchronize().map_err(driver_err)
    }

    fn all_gather(&self, input: &[f32], output: &mut [f32]) -> CommResult {
        let world = self.world_size();
        if output.len() != world * input.len() {
            return Err(CommError(format!(
                "all_gather: output has {} elements, expected world {world} x chunk {}",
                output.len(),
                input.len()
            )));
        }
        if input.is_empty() {
            return Ok(());
        }
        let send = self.stream.clone_htod(input).map_err(driver_err)?;
        let mut recv = self
            .stream
            .alloc_zeros::<f32>(output.len())
            .map_err(driver_err)?;
        self.comm
            .all_gather(&send, &mut recv)
            .map_err(nccl_err("all_gather"))?;
        self.stream.memcpy_dtoh(&recv, output).map_err(driver_err)?;
        self.stream.synchronize().map_err(driver_err)
    }

    fn broadcast(&self, buf: &mut [f32], root: usize) -> CommResult {
        if root >= self.world_size() {
            return Err(CommError(format!("broadcast: root {root} out of range")));
        }
        if buf.is_empty() {
            return Ok(());
        }
        let mut d = self.stream.clone_htod(buf).map_err(driver_err)?;
        self.comm
            .broadcast_in_place(&mut d, root as i32)
            .map_err(nccl_err("broadcast"))?;
        self.stream.memcpy_dtoh(&d, buf).map_err(driver_err)?;
        self.stream.synchronize().map_err(driver_err)
    }

    fn barrier(&self) -> CommResult {
        // NCCL has no barrier; a one-element all-reduce, waited on, is one.
        let mut token = [0.0f32];
        self.sum_in_place(&mut token)
    }
}
