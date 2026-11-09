//! Binary shard format (plan §9.3): a data file of sample records plus an
//! index file for random access. The data file is memory-mapped on read.
//!
//! A shard is two files: `NAME` (data) and `NAME.idx` (index; see
//! [`index_path`]). All integers are **little-endian**. Version 1.
//!
//! # Data file
//!
//! | Offset | Size | Field |
//! |---|---|---|
//! | 0 | 8 | magic `b"S2SSHARD"` |
//! | 8 | 4 | format version (`1`) |
//! | 12 | 4 | header length (`64`) |
//! | 16 | 8 | number of samples |
//! | 24 | 8 | data file length in bytes |
//! | 32 | 4 | flags (bit 0: finalized; a writer that did not finish leaves it 0) |
//! | 36 | 4 | CRC-32 of the whole index file |
//! | 40 | 20 | reserved (zero) |
//! | 60 | 4 | CRC-32 of bytes `0..60` |
//!
//! Records follow, each starting at an 8-byte-aligned offset (zero padding in
//! between). A record is:
//!
//! | Section | Size (bytes) |
//! |---|---|
//! | fixed header (below) | 64 |
//! | token grid, `u32` frame-major `[frames, num_streams]`, **aligned (undelayed)** | `4 * frames * num_streams` |
//! | injection table: per segment `frame u32, kind u32, len u32` | `12 * num_injections` |
//! | injection tokens, `u32`, segments concatenated in table order | `4 * num_injection_tokens` |
//! | word table: per word `stream u32, start_ms u32, end_ms u32, text_len u32` | `16 * num_words` |
//! | word text, UTF-8, concatenated | `word_text_len` |
//! | metadata, UTF-8 JSON ([`SampleMeta`](crate::SampleMeta)) | `meta_len` |
//!
//! Record header: magic `b"SMPL"`, header length `u32` (64), `frames u32`,
//! `num_streams u32`, `text_slots u16`, `codebooks u16`, `user_text_delay u16`,
//! `acoustic_delay u16`, `text_vocab u32`, `text_pad u32`,
//! `num_injections u32`, `num_injection_tokens u32`, `num_words u32`,
//! `word_text_len u32`, `meta_len u32`, then zero padding to 64 bytes.
//!
//! Tokens are `u32` rather than the plan's `u16` because text vocabularies of
//! current LLMs exceed 65 535 ids. Audio codes (`< 2049`) waste half of that
//! width; a packed audio encoding is a possible version-2 change.
//!
//! # Index file
//!
//! | Offset | Size | Field |
//! |---|---|---|
//! | 0 | 8 | magic `b"S2SINDEX"` |
//! | 8 | 4 | format version (`1`) |
//! | 12 | 4 | entry length (`32`) |
//! | 16 | 8 | number of samples |
//! | 24 | 8 | data file length (must match the data file) |
//! | 32 | `32 * n` | entries: `offset u64, len u64, crc32 u32, frames u32, num_streams u32, reserved u32` |
//! | end | 4 | CRC-32 of everything before it |
//!
//! `len` excludes alignment padding and `crc32` covers exactly the record's
//! `len` bytes.
//!
//! # Integrity
//!
//! Opening checks magics, versions, the finalized flag, both header CRCs, the
//! index CRC recorded in the data header, file lengths, and that every entry
//! lies inside the data file, is aligned, and does not overlap the previous
//! one, and that all padding between and after records is zero (so every byte
//! of both files is covered by a check). Reading a sample checks its CRC-32 and that every section length is
//! consistent. Every failure is an [`Error::Format`](crate::Error::Format);
//! nothing panics on corrupt input.

mod mmap;
mod reader;
mod record;
mod writer;

use std::path::{Path, PathBuf};

pub use reader::ShardReader;
pub use record::{decode_record, encode_record};
pub use writer::{ShardInfo, ShardWriter};

/// Data file magic.
pub const SHARD_MAGIC: [u8; 8] = *b"S2SSHARD";
/// Index file magic.
pub const INDEX_MAGIC: [u8; 8] = *b"S2SINDEX";
/// Format version written and accepted.
pub const FORMAT_VERSION: u32 = 1;
/// Data file header length.
pub const SHARD_HEADER_LEN: usize = 64;
/// Index file header length.
pub const INDEX_HEADER_LEN: usize = 32;
/// Index entry length.
pub const INDEX_ENTRY_LEN: usize = 32;
/// Record alignment within the data file.
pub const RECORD_ALIGN: u64 = 8;
const FLAG_FINALIZED: u32 = 1;

/// Location and checksum of one record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexEntry {
    /// Byte offset of the record in the data file.
    pub offset: u64,
    /// Record length in bytes (without alignment padding).
    pub len: u64,
    /// CRC-32 of the record's bytes.
    pub crc32: u32,
    /// Frames in the sample.
    pub frames: u32,
    /// Streams per frame.
    pub num_streams: u32,
}

/// The index file that belongs to a data file: `<data path>.idx`.
pub fn index_path(data: &Path) -> PathBuf {
    let mut s = data.as_os_str().to_owned();
    s.push(".idx");
    PathBuf::from(s)
}

pub(crate) fn crc32(b: &[u8]) -> u32 {
    crc32fast::hash(b)
}
