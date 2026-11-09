//! Collective communication for data-parallel training.
//!
//! [`Communicator`] is the small set of collectives the data-parallel code in
//! [`crate::dist`] needs. Every collective is called by **all** ranks of the
//! group, in the same order, with buffers of matching lengths (like NCCL and
//! MPI). Buffers are host slices; a GPU implementation (`s2s-cuda`'s
//! `NcclComm`) stages them through device memory and also offers
//! device-buffer variants.
//!
//! [`ThreadComm`] runs N ranks as threads of one process with shared buffers
//! and a barrier, so the parallel algorithms can be developed and tested on a
//! CPU. Its reductions add contributions **in rank order**
//! (`((r0 + r1) + r2) + ...`), so every rank gets bit-identical results and
//! runs are reproducible.

use std::sync::{Arc, Condvar, Mutex, MutexGuard};

/// A failed collective.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommError(pub String);

impl std::fmt::Display for CommError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "collective failed: {}", self.0)
    }
}

impl std::error::Error for CommError {}

/// Result of a collective.
pub type CommResult<T = ()> = Result<T, CommError>;

/// Collectives over a fixed group of `world_size` ranks.
///
/// All ranks must call the same collectives in the same order. Length
/// contracts (checked by implementations where possible):
///
/// | Collective | Input | Output |
/// |---|---|---|
/// | [`all_reduce_sum`](Self::all_reduce_sum) | `n` (in place) | `n` |
/// | [`reduce_scatter`](Self::reduce_scatter) | `world * chunk` | `chunk` |
/// | [`all_gather`](Self::all_gather) | `chunk` | `world * chunk` |
/// | [`broadcast`](Self::broadcast) | `n` (in place) | `n` |
pub trait Communicator {
    /// This rank, in `0..world_size()`.
    fn rank(&self) -> usize;

    /// Number of ranks in the group.
    fn world_size(&self) -> usize;

    /// Element-wise sum over ranks, in place; every rank gets the result.
    fn all_reduce_sum(&self, buf: &mut [f32]) -> CommResult;

    /// [`all_reduce_sum`](Self::all_reduce_sum) in `f64`, for small
    /// statistics that need the precision (e.g. squared gradient norms).
    fn all_reduce_sum_f64(&self, buf: &mut [f64]) -> CommResult;

    /// Sum over ranks, scattered: `input` is `world_size` equal chunks, and
    /// this rank receives the sum of chunk `rank()` over all ranks.
    fn reduce_scatter(&self, input: &[f32], output: &mut [f32]) -> CommResult;

    /// Concatenate every rank's `input` in rank order into `output`.
    fn all_gather(&self, input: &[f32], output: &mut [f32]) -> CommResult;

    /// Copy `root`'s `buf` to every rank.
    fn broadcast(&self, buf: &mut [f32], root: usize) -> CommResult;

    /// Wait until every rank has reached this point.
    fn barrier(&self) -> CommResult;

    /// Broadcast a byte string of any length from `root` (built on
    /// [`broadcast`](Self::broadcast); used for checkpoint manifests).
    /// Non-root ranks' `bytes` are replaced.
    fn broadcast_bytes(&self, bytes: &mut Vec<u8>, root: usize) -> CommResult {
        // Lengths and bytes travel as exactly representable f32 values:
        // the length as four 16-bit pieces, each byte as itself.
        let mut len = [0f32; 4];
        if self.rank() == root {
            let n = bytes.len() as u64;
            for (i, l) in len.iter_mut().enumerate() {
                *l = ((n >> (16 * i)) & 0xffff) as f32;
            }
        }
        self.broadcast(&mut len, root)?;
        let n = len
            .iter()
            .enumerate()
            .fold(0u64, |n, (i, &l)| n | ((l as u64) << (16 * i))) as usize;
        let mut buf: Vec<f32> = if self.rank() == root {
            bytes.iter().map(|&b| b as f32).collect()
        } else {
            vec![0.0; n]
        };
        self.broadcast(&mut buf, root)?;
        if self.rank() != root {
            *bytes = buf.iter().map(|&b| b as u8).collect();
        }
        Ok(())
    }
}

/// Barrier state: arrivals in the current generation, and a poison flag set
/// when a rank panics so the others fail instead of waiting forever.
#[derive(Debug, Default)]
struct BarrierState {
    arrived: usize,
    generation: u64,
    poisoned: bool,
}

/// Buffers posted by each rank for the collective in progress.
#[derive(Debug, Default)]
struct Slots {
    f32s: Vec<Vec<f32>>,
    f64s: Vec<Vec<f64>>,
}

#[derive(Debug)]
struct Shared {
    world: usize,
    barrier: Mutex<BarrierState>,
    cv: Condvar,
    slots: Mutex<Slots>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panic while holding the lock leaves plain data behind; peers learn
    // about the panic through the barrier's poison flag instead.
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Shared {
    fn wait(&self) -> CommResult {
        let mut st = lock(&self.barrier);
        if st.poisoned {
            return Err(poisoned());
        }
        st.arrived += 1;
        if st.arrived == self.world {
            st.arrived = 0;
            st.generation += 1;
            self.cv.notify_all();
            return Ok(());
        }
        let gen = st.generation;
        while st.generation == gen && !st.poisoned {
            st = self.cv.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        if st.generation == gen {
            Err(poisoned())
        } else {
            Ok(())
        }
    }

    fn poison(&self) {
        lock(&self.barrier).poisoned = true;
        self.cv.notify_all();
    }
}

fn poisoned() -> CommError {
    CommError("a peer rank left the group (finished, failed or panicked)".into())
}

/// One rank of an in-process thread group. Create the group with
/// [`ThreadComm::group`] and move one handle into each thread, or use
/// [`run_ranks`].
///
/// When a handle is dropped (its rank finished, returned early with an error,
/// or panicked), the group is poisoned: peers blocked in, or later entering, a
/// collective get a [`CommError`] instead of deadlocking.
#[derive(Debug)]
pub struct ThreadComm {
    rank: usize,
    shared: Arc<Shared>,
}

impl ThreadComm {
    /// Handles for a group of `world` ranks, indexed by rank.
    pub fn group(world: usize) -> Vec<ThreadComm> {
        assert!(world >= 1, "world size must be at least 1");
        let shared = Arc::new(Shared {
            world,
            barrier: Mutex::new(BarrierState::default()),
            cv: Condvar::new(),
            slots: Mutex::new(Slots {
                f32s: vec![Vec::new(); world],
                f64s: vec![Vec::new(); world],
            }),
        });
        (0..world)
            .map(|rank| ThreadComm {
                rank,
                shared: shared.clone(),
            })
            .collect()
    }

    /// Post `data` to this rank's slot, wait for all ranks, run `read` over
    /// every rank's posted data, then wait again so no rank overwrites its
    /// slot while another is still reading.
    fn exchange<R>(
        &self,
        post: impl FnOnce(&mut Slots),
        read: impl FnOnce(&Slots) -> CommResult<R>,
    ) -> CommResult<R> {
        post(&mut lock(&self.shared.slots));
        self.shared.wait()?;
        let out = read(&lock(&self.shared.slots));
        self.shared.wait()?;
        out
    }
}

impl Drop for ThreadComm {
    fn drop(&mut self) {
        // This rank will never arrive at another barrier: fail any peer that
        // waits for it. (Peers already released from their last barrier are
        // unaffected.)
        self.shared.poison();
    }
}

fn check_lengths<T>(slots: &[Vec<T>], what: &str) -> CommResult<usize> {
    let n = slots[0].len();
    if let Some((r, s)) = slots.iter().enumerate().find(|(_, s)| s.len() != n) {
        return Err(CommError(format!(
            "{what}: rank 0 posted {n} elements but rank {r} posted {}",
            s.len()
        )));
    }
    Ok(n)
}

/// `out[i] = ((s0[i] + s1[i]) + s2[i]) + ...` over `slots[*][range]`.
fn sum_in_rank_order<T: Copy + std::ops::Add<Output = T>>(
    slots: &[Vec<T>],
    range: std::ops::Range<usize>,
    out: &mut [T],
) {
    out.copy_from_slice(&slots[0][range.clone()]);
    for s in &slots[1..] {
        for (o, &x) in out.iter_mut().zip(&s[range.clone()]) {
            *o = *o + x;
        }
    }
}

impl Communicator for ThreadComm {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.shared.world
    }

    fn all_reduce_sum(&self, buf: &mut [f32]) -> CommResult {
        let r = self.rank;
        let data = buf.to_vec();
        self.exchange(
            |s| s.f32s[r] = data,
            |s| {
                let n = check_lengths(&s.f32s, "all_reduce_sum")?;
                sum_in_rank_order(&s.f32s, 0..n, buf);
                Ok(())
            },
        )
    }

    fn all_reduce_sum_f64(&self, buf: &mut [f64]) -> CommResult {
        let r = self.rank;
        let data = buf.to_vec();
        self.exchange(
            |s| s.f64s[r] = data,
            |s| {
                let n = check_lengths(&s.f64s, "all_reduce_sum_f64")?;
                sum_in_rank_order(&s.f64s, 0..n, buf);
                Ok(())
            },
        )
    }

    fn reduce_scatter(&self, input: &[f32], output: &mut [f32]) -> CommResult {
        let (r, world) = (self.rank, self.shared.world);
        let chunk = output.len();
        let data = input.to_vec();
        self.exchange(
            |s| s.f32s[r] = data,
            |s| {
                let n = check_lengths(&s.f32s, "reduce_scatter")?;
                if n != world * chunk {
                    return Err(CommError(format!(
                        "reduce_scatter: input has {n} elements, expected world {world} x chunk {chunk}"
                    )));
                }
                sum_in_rank_order(&s.f32s, r * chunk..(r + 1) * chunk, output);
                Ok(())
            },
        )
    }

    fn all_gather(&self, input: &[f32], output: &mut [f32]) -> CommResult {
        let (r, world) = (self.rank, self.shared.world);
        let data = input.to_vec();
        self.exchange(
            |s| s.f32s[r] = data,
            |s| {
                let n = check_lengths(&s.f32s, "all_gather")?;
                if output.len() != world * n {
                    return Err(CommError(format!(
                        "all_gather: output has {} elements, expected world {world} x chunk {n}",
                        output.len()
                    )));
                }
                for (o, s) in output.chunks_exact_mut(n.max(1)).zip(&s.f32s) {
                    o.copy_from_slice(s);
                }
                Ok(())
            },
        )
    }

    fn broadcast(&self, buf: &mut [f32], root: usize) -> CommResult {
        let (r, world) = (self.rank, self.shared.world);
        if root >= world {
            return Err(CommError(format!(
                "broadcast: root {root} out of range for world {world}"
            )));
        }
        let data = if r == root { buf.to_vec() } else { Vec::new() };
        self.exchange(
            |s| s.f32s[r] = data,
            |s| {
                let src = &s.f32s[root];
                if src.len() != buf.len() {
                    return Err(CommError(format!(
                        "broadcast: root sent {} elements, rank {r} expects {}",
                        src.len(),
                        buf.len()
                    )));
                }
                if r != root {
                    buf.copy_from_slice(src);
                }
                Ok(())
            },
        )
    }

    fn barrier(&self) -> CommResult {
        self.shared.wait()
    }
}

/// Run `f` once per rank of a fresh [`ThreadComm`] group of size `world`, each
/// on its own thread, and return the results in rank order.
///
/// Threads get a 16 MiB stack. A panic on any rank poisons the group (so peers
/// stop instead of deadlocking) and is re-raised here.
pub fn run_ranks<R: Send>(world: usize, f: impl Fn(ThreadComm) -> R + Sync) -> Vec<R> {
    let f = &f;
    std::thread::scope(|scope| {
        let handles: Vec<_> = ThreadComm::group(world)
            .into_iter()
            .map(|comm| {
                std::thread::Builder::new()
                    .name(format!("rank-{}", comm.rank))
                    .stack_size(16 << 20)
                    .spawn_scoped(scope, move || f(comm))
                    .expect("spawn rank thread")
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|p| std::panic::resume_unwind(p)))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rank_data(rank: usize, n: usize) -> Vec<f32> {
        (0..n).map(|i| (rank * 100 + i) as f32 * 0.5).collect()
    }

    #[test]
    fn all_reduce_sums_in_rank_order() {
        for world in [1, 2, 3, 5] {
            let out = run_ranks(world, |c| {
                let mut b = rank_data(c.rank(), 7);
                c.all_reduce_sum(&mut b).unwrap();
                b
            });
            let expect: Vec<f32> = (0..7)
                .map(|i| (1..world).fold(rank_data(0, 7)[i], |acc, r| acc + rank_data(r, 7)[i]))
                .collect();
            for b in out {
                assert_eq!(b, expect);
            }
        }
        // Rank order matters for bitwise results: 1e8 + 1 - 1e8 != 1e8 - 1e8 + 1.
        let out = run_ranks(3, |c| {
            let mut b = [[1e8f32], [1.0], [-1e8]][c.rank()];
            c.all_reduce_sum(&mut b).unwrap();
            b[0]
        });
        assert_eq!(out, vec![(1e8f32 + 1.0) - 1e8; 3]);
    }

    #[test]
    fn all_reduce_f64() {
        let out = run_ranks(4, |c| {
            let mut b = vec![c.rank() as f64 + 0.25, 1e-300];
            c.all_reduce_sum_f64(&mut b).unwrap();
            b
        });
        for b in out {
            assert_eq!(b, vec![7.0, 4e-300]);
        }
    }

    #[test]
    fn reduce_scatter_and_all_gather() {
        let world = 3;
        let chunk = 4;
        let out = run_ranks(world, |c| {
            let input = rank_data(c.rank(), world * chunk);
            let mut mine = vec![0.0; chunk];
            c.reduce_scatter(&input, &mut mine).unwrap();
            let mut all = vec![0.0; world * chunk];
            c.all_gather(&mine, &mut all).unwrap();
            (mine, all)
        });
        let mut full = rank_data(0, world * chunk);
        for r in 1..world {
            for (f, x) in full.iter_mut().zip(rank_data(r, world * chunk)) {
                *f += x;
            }
        }
        for (r, (mine, all)) in out.iter().enumerate() {
            assert_eq!(mine[..], full[r * chunk..(r + 1) * chunk]);
            assert_eq!(all, &full);
        }
    }

    #[test]
    fn broadcast_and_bytes() {
        let out = run_ranks(3, |c| {
            let mut b = rank_data(c.rank(), 5);
            c.broadcast(&mut b, 2).unwrap();
            let mut bytes = if c.rank() == 1 {
                (0..=255u8).cycle().take(70_000).collect()
            } else {
                vec![9; 3]
            };
            c.broadcast_bytes(&mut bytes, 1).unwrap();
            c.barrier().unwrap();
            (b, bytes)
        });
        let expect_bytes: Vec<u8> = (0..=255u8).cycle().take(70_000).collect();
        for (b, bytes) in out {
            assert_eq!(b, rank_data(2, 5));
            assert_eq!(bytes, expect_bytes);
        }
    }

    #[test]
    fn mismatched_lengths_fail_on_every_rank() {
        let out = run_ranks(2, |c| {
            let mut b = vec![1.0; 3 + c.rank()];
            let e = c.all_reduce_sum(&mut b).unwrap_err();
            let mut out = vec![0.0; 2];
            let e2 = c.reduce_scatter(&[1.0; 3], &mut out).unwrap_err();
            (e, e2)
        });
        assert!(out.iter().all(|(e, _)| e.0.contains("rank 1 posted 4")));
        assert!(out.iter().all(|(_, e)| e.0.contains("expected world 2")));
    }

    #[test]
    fn a_panicking_rank_poisons_the_group() {
        let mut comms = ThreadComm::group(2);
        let c1 = comms.pop().unwrap();
        let c0 = comms.pop().unwrap();
        let peer = std::thread::spawn(move || {
            let _c = c1;
            panic!("rank 1 fails");
        });
        assert!(peer.join().is_err());
        let mut b = [1.0];
        assert!(c0.all_reduce_sum(&mut b).is_err());
        assert!(c0.barrier().is_err());
    }
}
