//! Data-parallel training over a [`Communicator`]: DDP, ZeRO-1 and sharded
//! checkpoints (plan §8.2 stages 2 and 3).
//!
//! Every rank holds a full model replica and a full copy of the data
//! sampler. Each optimizer step, every rank draws the same **global** batch
//! (`TrainConfig::batch` sequences) and trains on its contiguous slice of
//! `batch / world` sequences, so the sequence of global batches, and the
//! checkpointed data cursor, do not depend on the world size.
//!
//! - **DDP** ([`Strategy::Ddp`]): gradients are packed into size-capped
//!   buckets in [`Module::visit_params`] order ([`BucketPlan`]), each bucket
//!   is all-reduced and divided by the world size, and every rank runs the
//!   full AdamW update.
//! - **ZeRO-1** ([`Strategy::Zero1`]): parameters are split into contiguous,
//!   size-balanced runs, one per rank ([`ParamPartition`]). Gradients are
//!   reduce-scattered so each rank receives the averaged gradients of the
//!   parameters it owns; it keeps AdamW moments only for those, updates them,
//!   and the updated parameters are all-gathered.
//!
//! **Gradient clipping** uses the canonical norm of
//! [`optim::global_grad_norm`]: each parameter's sum of squares is computed
//! (on its owner, for ZeRO-1), the per-parameter values are all-reduced in
//! `f64` (exact, since each entry has exactly one non-zero contributor), and
//! summed in parameter order. So DDP and ZeRO-1 apply bit-identical clip
//! scales, and with [`ThreadComm`](crate::comm::ThreadComm) (rank-ordered
//! reductions) the two strategies produce **bit-identical** parameters.
//!
//! **Checkpoints** ([`DistTrainer::save`], [`DistTrainer::load`]) use the
//! single-process format of [`crate::checkpoint`] unchanged: on save, moment
//! shards are gathered to rank 0, which writes the file; on load, rank 0
//! reads it and scatters parameters, moments and the manifest. Since the
//! file holds full tensors and the data cursor is global, a run can resume
//! with a different world size or strategy, or in a single process.
//!
//! Gradients are averaged as the mean of per-rank mean losses, which equals
//! the global mean loss when every rank's slice has the same number of
//! scored tokens (true for the character corpus, which ignores no targets).

use std::ops::Range;
use std::path::Path;

use s2s_nn::{Module, Param};

use crate::checkpoint::{self, CheckpointError, Manifest};
use crate::comm::{CommError, CommResult, Communicator};
use crate::optim::{self, AdamW, Moments, StepStats};
use crate::trainer::{StepReport, Trainer};

/// Element counts of every parameter, in [`Module::visit_params`] order.
pub fn param_numels<T: s2s_core::Float>(model: &dyn Module<T>) -> Vec<usize> {
    let mut v = Vec::new();
    model.visit_params(&mut |p: &Param<T>| v.push(p.numel()));
    v
}

fn flat_grads(model: &dyn Module<f32>) -> Vec<f32> {
    let mut v = Vec::new();
    model.visit_params(&mut |p: &Param<f32>| v.extend_from_slice(&p.grad));
    v
}

fn flat_values(model: &dyn Module<f32>) -> Vec<f32> {
    let mut v = Vec::new();
    model.visit_params(&mut |p: &Param<f32>| v.extend_from_slice(&p.value));
    v
}

fn set_flat(model: &mut dyn Module<f32>, flat: &[f32], grads: bool) {
    let mut off = 0;
    model.visit_params_mut(&mut |p: &mut Param<f32>| {
        let n = p.numel();
        let dst = if grads { &mut p.grad } else { &mut p.value };
        dst.copy_from_slice(&flat[off..off + n]);
        off += n;
    });
    assert_eq!(off, flat.len(), "flat buffer does not match the model");
}

/// One DDP gradient bucket: a run of consecutive parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bucket {
    /// Parameter indices (in [`Module::visit_params`] order).
    pub params: Range<usize>,
    /// Element range in the flat concatenation of all gradients.
    pub elems: Range<usize>,
}

/// DDP gradient buckets: consecutive parameters in [`Module::visit_params`]
/// order, packed until adding the next one would exceed the element cap. A
/// parameter larger than the cap gets a bucket of its own; parameters are
/// never split. The plan depends only on the parameter sizes and the cap, so
/// it is identical on every rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BucketPlan {
    /// Buckets in all-reduce order.
    pub buckets: Vec<Bucket>,
}

impl BucketPlan {
    /// Plan buckets of at most `cap_elems` elements (where possible).
    pub fn new(numels: &[usize], cap_elems: usize) -> Self {
        let mut buckets = Vec::new();
        let (mut p0, mut e0, mut e) = (0, 0, 0);
        for (i, &n) in numels.iter().enumerate() {
            if e > e0 && e - e0 + n > cap_elems {
                buckets.push(Bucket {
                    params: p0..i,
                    elems: e0..e,
                });
                (p0, e0) = (i, e);
            }
            e += n;
        }
        if p0 < numels.len() {
            buckets.push(Bucket {
                params: p0..numels.len(),
                elems: e0..e,
            });
        }
        Self { buckets }
    }
}

/// ZeRO-1 ownership: a contiguous run of parameters (in
/// [`Module::visit_params`] order) per rank, balanced by element count.
///
/// The split before rank `r` is placed at the parameter boundary closest to
/// `r * total / world` elements, so the partition is a pure function of the
/// parameter sizes and the world size. Ranks may own no parameters when there
/// are more ranks than parameters.
///
/// Collectives use a padded layout of `world * shard_len` elements in which
/// rank `r`'s chunk holds its parameters back to back, zero-padded to
/// [`shard_len`](Self::shard_len).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamPartition {
    world: usize,
    /// Element offset of each parameter in the flat concatenation (len n+1).
    offsets: Vec<usize>,
    /// Parameter index range per rank.
    ranges: Vec<Range<usize>>,
    shard_len: usize,
}

impl ParamPartition {
    /// Partition parameters with these sizes over `world` ranks.
    pub fn new(numels: &[usize], world: usize) -> Self {
        assert!(world >= 1, "world size must be at least 1");
        let mut offsets = Vec::with_capacity(numels.len() + 1);
        offsets.push(0);
        for &n in numels {
            offsets.push(offsets.last().unwrap() + n);
        }
        let total = *offsets.last().unwrap() as u128;
        let n = numels.len();
        let mut cuts = vec![0usize];
        for r in 1..world {
            // Closest boundary to r/world of the elements, never moving back.
            let target = r as u128 * total;
            let dist = |i: usize| (offsets[i] as u128 * world as u128).abs_diff(target);
            let prev = *cuts.last().unwrap();
            let best = (prev..=n).min_by_key(|&i| dist(i)).unwrap();
            cuts.push(best);
        }
        cuts.push(n);
        let ranges: Vec<Range<usize>> = cuts.windows(2).map(|w| w[0]..w[1]).collect();
        let shard_len = ranges
            .iter()
            .map(|r| offsets[r.end] - offsets[r.start])
            .max()
            .unwrap_or(0);
        Self {
            world,
            offsets,
            ranges,
            shard_len,
        }
    }

    /// Partition `model`'s parameters over `world` ranks.
    pub fn for_model<T: s2s_core::Float>(model: &dyn Module<T>, world: usize) -> Self {
        Self::new(&param_numels(model), world)
    }

    /// Number of ranks.
    pub fn world(&self) -> usize {
        self.world
    }

    /// Parameter indices owned by `rank`.
    pub fn params_of(&self, rank: usize) -> Range<usize> {
        self.ranges[rank].clone()
    }

    /// The rank that owns parameter `index`.
    pub fn owner(&self, index: usize) -> usize {
        self.ranges
            .iter()
            .position(|r| r.contains(&index))
            .expect("parameter index out of range")
    }

    /// Elements in the flat concatenation owned by `rank`.
    pub fn elems_of(&self, rank: usize) -> Range<usize> {
        let r = &self.ranges[rank];
        self.offsets[r.start]..self.offsets[r.end]
    }

    /// Padded chunk length: the largest number of elements any rank owns.
    pub fn shard_len(&self) -> usize {
        self.shard_len
    }

    /// Rank `rank`'s chunk of a flat (concatenated) buffer, zero-padded to
    /// [`shard_len`](Self::shard_len).
    pub fn shard(&self, flat: &[f32], rank: usize) -> Vec<f32> {
        let mut out = vec![0.0; self.shard_len];
        let e = self.elems_of(rank);
        out[..e.len()].copy_from_slice(&flat[e]);
        out
    }

    /// Flat buffer → padded layout (`world * shard_len`).
    pub fn pad(&self, flat: &[f32]) -> Vec<f32> {
        (0..self.world).flat_map(|r| self.shard(flat, r)).collect()
    }

    /// Padded layout → flat buffer.
    pub fn unpad(&self, padded: &[f32]) -> Vec<f32> {
        assert_eq!(padded.len(), self.world * self.shard_len);
        (0..self.world)
            .flat_map(|r| {
                let n = self.elems_of(r).len();
                padded[r * self.shard_len..r * self.shard_len + n]
                    .iter()
                    .copied()
            })
            .collect()
    }
}

/// How gradients and optimizer state are distributed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    /// Replicated optimizer state; bucketed gradient all-reduce.
    Ddp {
        /// Bucket capacity in elements (PyTorch's default is 25 MB, i.e.
        /// about 6.5M f32 elements).
        bucket_elems: usize,
    },
    /// Optimizer state sharded by [`ParamPartition`]; gradient
    /// reduce-scatter and parameter all-gather.
    Zero1,
}

/// All-reduce and average gradients bucket by bucket (DDP). Afterwards every
/// rank holds the mean gradient over ranks.
pub fn ddp_reduce_grads<C: Communicator + ?Sized>(
    model: &mut dyn Module<f32>,
    plan: &BucketPlan,
    comm: &C,
) -> CommResult {
    let world = comm.world_size() as f32;
    let mut flat = flat_grads(model);
    for b in &plan.buckets {
        let buf = &mut flat[b.elems.clone()];
        comm.all_reduce_sum(buf)?;
        buf.iter_mut().for_each(|g| *g /= world);
    }
    set_flat(model, &flat, true);
    Ok(())
}

/// DDP: average gradients across ranks, then a full AdamW step on every rank.
pub fn ddp_step<C: Communicator + ?Sized>(
    model: &mut dyn Module<f32>,
    opt: &mut AdamW<f32>,
    plan: &BucketPlan,
    comm: &C,
    lr: f64,
) -> CommResult<StepStats> {
    ddp_reduce_grads(model, plan, comm)?;
    let norm = optim::global_grad_norm(&optim::param_grad_sq_norms(model));
    Ok(opt.step_with_norm(model, lr, norm, &|_| true))
}

/// ZeRO-1: reduce-scatter gradients, clip by the global norm, update this
/// rank's parameters with its moment shard, and all-gather parameters.
///
/// Afterwards every rank holds identical parameters. Gradients of parameters
/// owned by other ranks are left as this rank's local (un-reduced) values.
pub fn zero1_step<C: Communicator + ?Sized>(
    model: &mut dyn Module<f32>,
    opt: &mut AdamW<f32>,
    part: &ParamPartition,
    comm: &C,
    lr: f64,
) -> CommResult<StepStats> {
    let (rank, world) = (comm.rank(), comm.world_size());
    check_partition(part, world)?;
    let mut flat = flat_grads(model);

    // Averaged gradients of my parameters.
    let mut mine = vec![0.0; part.shard_len()];
    comm.reduce_scatter(&part.pad(&flat), &mut mine)?;
    let elems = part.elems_of(rank);
    let n = elems.len();
    mine[..n].iter_mut().for_each(|g| *g /= world as f32);
    flat[elems].copy_from_slice(&mine[..n]);
    set_flat(model, &flat, true);

    // Global norm from owners' per-parameter sums of squares.
    let owned = part.params_of(rank);
    let mut sq = vec![0.0f64; part.offsets.len() - 1];
    let mut i = 0;
    model.visit_params(&mut |p: &Param<f32>| {
        if owned.contains(&i) {
            sq[i] = optim::grad_sq_norm(&p.grad);
        }
        i += 1;
    });
    comm.all_reduce_sum_f64(&mut sq)?;
    let norm = optim::global_grad_norm(&sq);

    let stats = opt.step_with_norm(model, lr, norm, &|i| owned.contains(&i));

    let mut all = vec![0.0; world * part.shard_len()];
    comm.all_gather(&part.shard(&flat_values(model), rank), &mut all)?;
    set_flat(model, &part.unpad(&all), false);
    Ok(stats)
}

fn check_partition(part: &ParamPartition, world: usize) -> CommResult {
    if part.world() != world {
        return Err(CommError(format!(
            "partition is for world {} but the communicator has {world} ranks",
            part.world()
        )));
    }
    Ok(())
}

/// Which parameters' moments a rank keeps: all (`None`) or its partition
/// shard.
fn owns(part: Option<&ParamPartition>, rank: usize, index: usize) -> bool {
    part.is_none_or(|p| p.params_of(rank).contains(&index))
}

/// Write a checkpoint from a data-parallel group, in the single-process
/// [`checkpoint`] format. Collective: every rank must call it.
///
/// `part` is `Some` when moments are sharded (ZeRO-1) and `None` when every
/// rank holds all of them (DDP). Sharded moments are all-gathered to build
/// the full optimizer state; rank 0 writes the file and broadcasts whether
/// that succeeded, so every rank returns the same outcome.
pub fn save_sharded<C: Communicator + ?Sized>(
    path: &Path,
    model: &dyn Module<f32>,
    opt: &AdamW<f32>,
    part: Option<&ParamPartition>,
    manifest: &Manifest,
    comm: &C,
) -> Result<(), CheckpointError> {
    let rank = comm.rank();
    let mut full = opt.clone();
    if let Some(part) = part {
        check_partition(part, comm.world_size())?;
        full.state = gather_moments(model, opt, part, comm)?;
    }
    let mut status = [0.0f32];
    let result = if rank == 0 {
        let r = checkpoint::save(path, model, &full, manifest);
        status[0] = if r.is_ok() { 1.0 } else { 0.0 };
        r
    } else {
        Ok(())
    };
    comm.broadcast(&mut status, 0)?;
    match result {
        Ok(()) if status[0] == 1.0 => Ok(()),
        Ok(()) => Err(CheckpointError::Comm(CommError(
            "rank 0 failed to write the checkpoint".into(),
        ))),
        Err(e) => Err(e),
    }
}

/// All-gather sharded moments into the full state (on every rank).
fn gather_moments<C: Communicator + ?Sized>(
    model: &dyn Module<f32>,
    opt: &AdamW<f32>,
    part: &ParamPartition,
    comm: &C,
) -> Result<std::collections::BTreeMap<String, Moments<f32>>, CheckpointError> {
    let rank = comm.rank();
    let mut state = std::collections::BTreeMap::new();
    if opt.step == 0 {
        // No update yet, so no rank has moments.
        return Ok(state);
    }
    let owned = part.params_of(rank);
    let (mut m, mut v) = (Vec::new(), Vec::new());
    let mut missing = 0.0f64;
    let mut names = Vec::new();
    let mut i = 0;
    model.visit_params(&mut |p: &Param<f32>| {
        names.push((p.name.clone(), p.numel()));
        if owned.contains(&i) {
            match opt.state.get(&p.name) {
                Some(mom) => {
                    m.extend_from_slice(&mom.m);
                    v.extend_from_slice(&mom.v);
                }
                None => {
                    missing += 1.0;
                    m.extend(std::iter::repeat_n(0.0, p.numel()));
                    v.extend(std::iter::repeat_n(0.0, p.numel()));
                }
            }
        }
        i += 1;
    });
    let mut flag = [missing];
    comm.all_reduce_sum_f64(&mut flag)?;
    if flag[0] > 0.0 {
        return Err(CheckpointError::Mismatch(format!(
            "{} owned parameters have no AdamW moments at step {}",
            flag[0], opt.step
        )));
    }
    let shard_len = part.shard_len();
    let mut all_m = vec![0.0; comm.world_size() * shard_len];
    let mut all_v = vec![0.0; comm.world_size() * shard_len];
    m.resize(shard_len, 0.0);
    v.resize(shard_len, 0.0);
    comm.all_gather(&m, &mut all_m)?;
    comm.all_gather(&v, &mut all_v)?;
    let (all_m, all_v) = (part.unpad(&all_m), part.unpad(&all_v));
    let mut off = 0;
    for (name, n) in names {
        state.insert(
            name,
            Moments {
                m: all_m[off..off + n].to_vec(),
                v: all_v[off..off + n].to_vec(),
            },
        );
        off += n;
    }
    Ok(state)
}

/// Read a checkpoint into a data-parallel group. Collective: every rank must
/// call it.
///
/// Rank 0 reads and validates the file (against its model), then broadcasts
/// the manifest and parameters and scatters the moments: each rank keeps all
/// of them (`part == None`, DDP) or its own shard (ZeRO-1). The file may have
/// been written by any world size or by a single process.
pub fn load_sharded<C: Communicator + ?Sized>(
    path: &Path,
    model: &mut dyn Module<f32>,
    opt: &mut AdamW<f32>,
    part: Option<&ParamPartition>,
    comm: &C,
) -> Result<Manifest, CheckpointError> {
    let rank = comm.rank();
    if let Some(part) = part {
        check_partition(part, comm.world_size())?;
    }
    let mut full = opt.clone();
    full.state.clear();
    let mut status = [0.0f32];
    let mut manifest_bytes = Vec::new();
    let mut root_err = None;
    if rank == 0 {
        match read_full(path, model, &mut full) {
            Ok(m) => {
                status[0] = 1.0;
                manifest_bytes =
                    serde_json::to_vec(&m).map_err(|e| CheckpointError::Format(e.to_string()))?;
            }
            Err(e) => root_err = Some(e),
        }
    }
    comm.broadcast(&mut status, 0)?;
    if let Some(e) = root_err {
        return Err(e);
    }
    if status[0] != 1.0 {
        return Err(CheckpointError::Comm(CommError(
            "rank 0 failed to read the checkpoint".into(),
        )));
    }
    comm.broadcast_bytes(&mut manifest_bytes, 0)?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| CheckpointError::Format(e.to_string()))?;

    let mut values = flat_values(model);
    comm.broadcast(&mut values, 0)?;
    set_flat(model, &values, false);

    let mut has_moments = [if full.state.is_empty() { 0.0 } else { 1.0 }];
    comm.broadcast(&mut has_moments, 0)?;
    opt.state.clear();
    if has_moments[0] == 1.0 {
        let flat_moments = |pick: fn(&Moments<f32>) -> &Vec<f32>| {
            let mut out = Vec::new();
            model.visit_params(&mut |p: &Param<f32>| match full.state.get(&p.name) {
                Some(mom) => out.extend_from_slice(pick(mom)),
                None => out.extend(std::iter::repeat_n(0.0, p.numel())),
            });
            out
        };
        let mut m = flat_moments(|x| &x.m);
        let mut v = flat_moments(|x| &x.v);
        comm.broadcast(&mut m, 0)?;
        comm.broadcast(&mut v, 0)?;
        let (mut i, mut off) = (0, 0);
        model.visit_params(&mut |p: &Param<f32>| {
            let n = p.numel();
            if owns(part, rank, i) {
                opt.state.insert(
                    p.name.clone(),
                    Moments {
                        m: m[off..off + n].to_vec(),
                        v: v[off..off + n].to_vec(),
                    },
                );
            }
            i += 1;
            off += n;
        });
    }
    opt.step = manifest.step;
    Ok(manifest)
}

/// Rank 0's half of [`load_sharded`]: read the file and require moments for
/// all parameters or none.
fn read_full(
    path: &Path,
    model: &mut dyn Module<f32>,
    full: &mut AdamW<f32>,
) -> Result<Manifest, CheckpointError> {
    let manifest = checkpoint::load(path, model, full)?;
    let n = param_numels(model).len();
    if !full.state.is_empty() && full.state.len() != n {
        return Err(CheckpointError::Mismatch(format!(
            "checkpoint has AdamW moments for {} of {n} parameters",
            full.state.len()
        )));
    }
    Ok(manifest)
}

/// A [`Trainer`] replica on one rank of a data-parallel group.
///
/// `trainer.cfg.batch` is the **global** batch (sequences per micro-batch
/// summed over ranks) and must be divisible by the world size. Every rank's
/// `trainer.data` must start at the same cursor (same seed), and its model
/// is overwritten with rank 0's parameters at construction.
#[derive(Debug)]
pub struct DistTrainer<C> {
    /// The local replica (model, optimizer state or shard, data, config).
    pub trainer: Trainer,
    /// The group.
    pub comm: C,
    strategy: Strategy,
    buckets: BucketPlan,
    partition: ParamPartition,
}

impl<C: Communicator> DistTrainer<C> {
    /// Join the group: broadcast rank 0's parameters and, for ZeRO-1, drop
    /// optimizer state this rank does not own. Collective.
    pub fn new(mut trainer: Trainer, comm: C, strategy: Strategy) -> CommResult<Self> {
        let world = comm.world_size();
        if !trainer.cfg.batch.is_multiple_of(world) {
            return Err(CommError(format!(
                "global batch {} is not divisible by world size {world}",
                trainer.cfg.batch
            )));
        }
        let numels = param_numels(&trainer.model);
        let buckets = BucketPlan::new(
            &numels,
            match strategy {
                Strategy::Ddp { bucket_elems } => bucket_elems,
                Strategy::Zero1 => usize::MAX,
            },
        );
        let partition = ParamPartition::new(&numels, world);
        let mut values = flat_values(&trainer.model);
        comm.broadcast(&mut values, 0)?;
        set_flat(&mut trainer.model, &values, false);
        let mut t = Self {
            trainer,
            comm,
            strategy,
            buckets,
            partition,
        };
        if let Some(part) = t.sharding() {
            let names = t.trainer.model.param_names();
            let mine = part.params_of(t.comm.rank());
            let keep: Vec<&String> = names[mine].iter().collect();
            t.trainer.opt.state.retain(|k, _| keep.contains(&k));
        }
        Ok(t)
    }

    /// The strategy.
    pub fn strategy(&self) -> Strategy {
        self.strategy
    }

    /// DDP gradient buckets.
    pub fn buckets(&self) -> &BucketPlan {
        &self.buckets
    }

    /// ZeRO-1 parameter ownership (computed for DDP too, unused there).
    pub fn partition(&self) -> &ParamPartition {
        &self.partition
    }

    fn sharding(&self) -> Option<&ParamPartition> {
        match self.strategy {
            Strategy::Ddp { .. } => None,
            Strategy::Zero1 => Some(&self.partition),
        }
    }

    /// One optimizer step across the group. Collective.
    ///
    /// The reported loss is the mean over ranks of each rank's mean loss.
    pub fn step(&mut self) -> CommResult<StepReport> {
        let (rank, world) = (self.comm.rank(), self.comm.world_size());
        let t = &mut self.trainer;
        let c = t.cfg;
        let local = c.batch / world;
        let rows = rank * local * c.seq..(rank + 1) * local * c.seq;
        t.model.zero_grad();
        let mut loss_sum = 0.0;
        for _ in 0..c.grad_accum {
            let (tokens, targets) = t.data.batch(c.batch, c.seq);
            let (tokens, targets) = (&tokens[rows.clone()], &targets[rows.clone()]);
            let (loss, ctx) = t
                .model
                .forward_loss(tokens, targets, local, c.seq, c.checkpoint);
            t.model.backward(ctx, targets);
            loss_sum += loss;
        }
        if c.grad_accum > 1 {
            let s = 1.0 / c.grad_accum as f32;
            t.model
                .visit_params_mut(&mut |p| p.grad.iter_mut().for_each(|g| *g *= s));
        }
        let lr = c.schedule.lr(t.opt.step);
        let stats = match self.strategy {
            Strategy::Ddp { .. } => {
                ddp_step(&mut t.model, &mut t.opt, &self.buckets, &self.comm, lr)?
            }
            Strategy::Zero1 => {
                zero1_step(&mut t.model, &mut t.opt, &self.partition, &self.comm, lr)?
            }
        };
        let mut loss = [loss_sum / c.grad_accum as f32];
        self.comm.all_reduce_sum(&mut loss)?;
        Ok(StepReport {
            loss: loss[0] / world as f32,
            stats,
        })
    }

    /// The manifest (world-size independent: the batch is global).
    pub fn manifest(&self) -> Manifest {
        self.trainer.manifest()
    }

    /// Save a checkpoint in the single-process format (rank 0 writes).
    /// Collective.
    pub fn save(&self, path: &Path) -> Result<(), CheckpointError> {
        save_sharded(
            path,
            &self.trainer.model,
            &self.trainer.opt,
            self.sharding(),
            &self.manifest(),
            &self.comm,
        )
    }

    /// Resume from a checkpoint written by any world size (or a single
    /// process), including the data cursor. Collective.
    pub fn load(&mut self, path: &Path) -> Result<Manifest, CheckpointError> {
        let part = match self.strategy {
            Strategy::Ddp { .. } => None,
            Strategy::Zero1 => Some(&self.partition),
        };
        let m = load_sharded(
            path,
            &mut self.trainer.model,
            &mut self.trainer.opt,
            part,
            &self.comm,
        )?;
        self.trainer.data.restore(m.data);
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_follow_visit_order_and_cap() {
        let plan = BucketPlan::new(&[4, 4, 10, 1, 1, 1, 3], 8);
        let got: Vec<_> = plan
            .buckets
            .iter()
            .map(|b| (b.params.clone(), b.elems.clone()))
            .collect();
        assert_eq!(
            got,
            vec![(0..2, 0..8), (2..3, 8..18), (3..7, 18..24)],
            "oversized params get their own bucket; others pack up to the cap"
        );
        assert_eq!(BucketPlan::new(&[], 8).buckets, vec![]);
        assert_eq!(BucketPlan::new(&[3, 3], usize::MAX).buckets.len(), 1);
    }

    #[test]
    fn partition_is_contiguous_balanced_and_padded() {
        let numels = [10, 10, 10, 10, 1, 1, 30];
        let p = ParamPartition::new(&numels, 3);
        // Cuts at the boundaries closest to 24 and 48 of 72 elements.
        assert_eq!(p.params_of(0), 0..2);
        assert_eq!(p.params_of(1), 2..6);
        assert_eq!(p.params_of(2), 6..7);
        assert_eq!(p.elems_of(1), 20..42);
        assert_eq!(p.shard_len(), 30);
        assert_eq!(p.owner(4), 1);
        let flat: Vec<f32> = (0..72).map(|x| x as f32).collect();
        let padded = p.pad(&flat);
        assert_eq!(padded.len(), 90);
        assert_eq!(padded[30..52], flat[20..42]);
        assert!(padded[52..60].iter().all(|&x| x == 0.0));
        assert_eq!(p.unpad(&padded), flat);
        // More ranks than parameters: some ranks own nothing.
        let q = ParamPartition::new(&[5, 5], 4);
        let owned: usize = (0..4).map(|r| q.params_of(r).len()).sum();
        assert_eq!(owned, 2);
        assert_eq!(q.unpad(&q.pad(&[1.0; 10])), vec![1.0; 10]);
        // One rank owns everything.
        let one = ParamPartition::new(&numels, 1);
        assert_eq!(one.params_of(0), 0..7);
    }
}
