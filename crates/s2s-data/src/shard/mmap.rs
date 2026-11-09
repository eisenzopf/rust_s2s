//! The crate's only `unsafe` code: memory-mapping a shard file read-only.
#![allow(unsafe_code)]

use std::fs::File;

use memmap2::Mmap;

/// Map `file` read-only.
///
/// The caller has already checked that the file is non-empty.
pub(super) fn map_readonly(file: &File) -> std::io::Result<Mmap> {
    // SAFETY: `Mmap::map` is unsafe because the mapping aliases the file: if
    // another process truncates or rewrites the file while it is mapped, reads
    // through the returned slice can observe changing bytes or fault (SIGBUS).
    // Shards are write-once: `ShardWriter` produces them completely and nothing
    // in this project modifies a finished shard, so the mapped bytes are
    // treated as immutable for the mapping's lifetime. Content corruption that
    // is present when reading (as opposed to concurrent modification) is
    // caught by the length and CRC-32 checks in `ShardReader`, never by UB.
    unsafe { Mmap::map(file) }
}
