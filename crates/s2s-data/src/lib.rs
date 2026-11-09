//! Training-data pipeline for rust_s2s (plan §9.2–9.3).
//!
//! - [`sample`]: one conversation in memory ([`Sample`]): an aligned
//!   frame-major token grid laid out by [`s2s_core::streams::StreamLayout`],
//!   injection segments, a word-timestamp side table and metadata.
//! - [`shard`]: the versioned binary shard format, [`ShardWriter`] and the
//!   memory-mapped, CRC-checked [`ShardReader`].
//! - [`assemble`](mod@assemble): sample → training [`Example`]: delays, text-only injection
//!   positions, targets shifted by one position, per-target loss weights.
//! - [`sampler`] and [`loader`]: deterministic, resumable fixed-window
//!   batching ([`Loader`], [`Cursor`], optional background [`Prefetcher`]).
//! - [`synth`]: synthetic tool-dialogue timeline assembly from a JSON
//!   [`Script`].
//! - [`tokenizer`]: the [`TextTokenizer`] interface and a deterministic
//!   [`ByteTokenizer`].
//! - [`validate`] and [`dump`]: invariant checks and grid dumps.
//!
//! Everything runs on the CPU. The `s2s-data` binary wraps the pieces in a
//! small CLI (`synth`, `inspect`, `validate`).
//!
//! The only `unsafe` code is the read-only memory map in `shard/mmap.rs`.

pub mod assemble;
pub mod dump;
mod error;
pub mod loader;
pub mod sample;
pub mod sampler;
pub mod shard;
pub mod synth;
pub mod tokenizer;
pub mod validate;

pub use assemble::{assemble, AssemblyConfig, Example, LossWeights, IGNORE_INDEX};
pub use error::{Error, Result};
pub use loader::{Batch, Dataset, Loader, LoaderConfig, Prefetcher};
pub use sample::{Injection, InjectionKind, Sample, SampleMeta, SpeakerInfo, WordTiming};
pub use sampler::{Cursor, Sampler, Window, WindowMode};
pub use shard::{ShardInfo, ShardReader, ShardWriter};
pub use synth::{synthesize, Script, SynthConfig, Synthesized};
pub use tokenizer::{ByteTokenizer, SpecialIds, TextTokenizer};
