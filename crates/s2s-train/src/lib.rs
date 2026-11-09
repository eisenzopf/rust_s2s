//! Training loop pieces for rust_s2s.
//!
//! - [`optim::AdamW`]: AdamW over any [`s2s_nn::Module`], with per-parameter
//!   groups (learning-rate multiplier, weight decay on/off) chosen by name,
//!   and global gradient-norm clipping fused into the update.
//! - [`schedule::LrSchedule`]: linear warmup + cosine, and warmup-stable-decay.
//! - [`checkpoint`]: save/load parameters, optimizer state and a JSON manifest
//!   (step, RNG state, data cursor) as safetensors, for **bit-exact resume**.
//! - [`data::CharCorpus`]: a tiny character-level corpus with a resumable,
//!   seeded batch sampler, for trainer bring-up on CPU.
//! - [`trainer`]: a minimal loop tying these together for the transformer LM.
//! - [`comm`]: the [`comm::Communicator`] trait (all-reduce, reduce-scatter,
//!   all-gather, broadcast, barrier) and [`comm::ThreadComm`], which runs N
//!   ranks as threads of one process.
//! - [`dist`]: data parallelism over a communicator: DDP with bucketed
//!   all-reduce, ZeRO-1 optimizer-state sharding, and sharded checkpoints that
//!   resume at any world size.

#![forbid(unsafe_code)]

pub mod checkpoint;
pub mod comm;
pub mod data;
pub mod dist;
pub mod optim;
pub mod schedule;
pub mod trainer;
