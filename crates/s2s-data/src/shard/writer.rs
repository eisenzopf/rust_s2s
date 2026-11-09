use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use super::record::encode_record;
use super::{
    crc32, index_path, IndexEntry, FLAG_FINALIZED, FORMAT_VERSION, INDEX_ENTRY_LEN,
    INDEX_HEADER_LEN, INDEX_MAGIC, RECORD_ALIGN, SHARD_HEADER_LEN, SHARD_MAGIC,
};
use crate::{validate, Error, Result, Sample};

/// Summary of a finished shard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardInfo {
    /// Data file path.
    pub path: PathBuf,
    /// Index file path.
    pub index_path: PathBuf,
    /// Samples written.
    pub num_samples: usize,
    /// Total frames over all samples.
    pub total_frames: u64,
    /// Data file length in bytes.
    pub data_len: u64,
}

/// Writes a shard: [`create`](Self::create), [`append`](Self::append) samples,
/// then [`finish`](Self::finish). A shard whose writer was dropped without
/// `finish` is rejected by the reader (its finalized flag is unset and it has
/// no index).
#[derive(Debug)]
pub struct ShardWriter {
    path: PathBuf,
    file: BufWriter<File>,
    entries: Vec<IndexEntry>,
    pos: u64,
}

pub(super) fn shard_header(num_samples: u64, data_len: u64, flags: u32, index_crc: u32) -> Vec<u8> {
    let mut h = Vec::with_capacity(SHARD_HEADER_LEN);
    h.extend_from_slice(&SHARD_MAGIC);
    h.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    h.extend_from_slice(&(SHARD_HEADER_LEN as u32).to_le_bytes());
    h.extend_from_slice(&num_samples.to_le_bytes());
    h.extend_from_slice(&data_len.to_le_bytes());
    h.extend_from_slice(&flags.to_le_bytes());
    h.extend_from_slice(&index_crc.to_le_bytes());
    h.resize(SHARD_HEADER_LEN - 4, 0);
    let c = crc32(&h);
    h.extend_from_slice(&c.to_le_bytes());
    h
}

pub(super) fn index_bytes(entries: &[IndexEntry], data_len: u64) -> Vec<u8> {
    let mut b = Vec::with_capacity(INDEX_HEADER_LEN + INDEX_ENTRY_LEN * entries.len() + 4);
    b.extend_from_slice(&INDEX_MAGIC);
    b.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    b.extend_from_slice(&(INDEX_ENTRY_LEN as u32).to_le_bytes());
    b.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    b.extend_from_slice(&data_len.to_le_bytes());
    for e in entries {
        b.extend_from_slice(&e.offset.to_le_bytes());
        b.extend_from_slice(&e.len.to_le_bytes());
        b.extend_from_slice(&e.crc32.to_le_bytes());
        b.extend_from_slice(&e.frames.to_le_bytes());
        b.extend_from_slice(&e.num_streams.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
    }
    let c = crc32(&b);
    b.extend_from_slice(&c.to_le_bytes());
    b
}

impl ShardWriter {
    /// Create (truncate) the data file at `path`. The index goes to
    /// [`index_path`]`(path)` on [`finish`](Self::finish).
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::create(&path).map_err(|e| Error::io(path.display().to_string(), e))?;
        let mut file = BufWriter::new(file);
        file.write_all(&shard_header(0, 0, 0, 0))
            .map_err(|e| Error::io(path.display().to_string(), e))?;
        Ok(Self {
            path,
            file,
            entries: Vec::new(),
            pos: SHARD_HEADER_LEN as u64,
        })
    }

    /// Samples appended so far.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing has been appended yet.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Validate ([`validate::check`]) and append a sample. Returns its index
    /// within the shard.
    pub fn append(&mut self, sample: &Sample) -> Result<usize> {
        validate::check(sample)?;
        let rec = encode_record(sample)?;
        let ctx = || self.path.display().to_string();
        self.file.write_all(&rec).map_err(|e| Error::io(ctx(), e))?;
        let len = rec.len() as u64;
        let pad = (RECORD_ALIGN - len % RECORD_ALIGN) % RECORD_ALIGN;
        self.file
            .write_all(&[0u8; RECORD_ALIGN as usize][..pad as usize])
            .map_err(|e| Error::io(ctx(), e))?;
        self.entries.push(IndexEntry {
            offset: self.pos,
            len,
            crc32: crc32(&rec),
            frames: sample.frames as u32,
            num_streams: sample.num_streams() as u32,
        });
        self.pos += len + pad;
        Ok(self.entries.len() - 1)
    }

    /// Write the index and the final header, and sync both files to disk.
    pub fn finish(mut self) -> Result<ShardInfo> {
        let ctx = self.path.display().to_string();
        let idx_path = index_path(&self.path);
        let data_len = self.pos;
        let index = index_bytes(&self.entries, data_len);

        // Index first (via a temporary + rename), then the header that names
        // its CRC: a crash in between leaves an unfinalized data file.
        let mut tmp = idx_path.as_os_str().to_owned();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);
        {
            let mut f = File::create(&tmp).map_err(|e| Error::io(tmp.display().to_string(), e))?;
            f.write_all(&index)
                .and_then(|_| f.sync_all())
                .map_err(|e| Error::io(tmp.display().to_string(), e))?;
        }
        std::fs::rename(&tmp, &idx_path)
            .map_err(|e| Error::io(idx_path.display().to_string(), e))?;

        let header = shard_header(
            self.entries.len() as u64,
            data_len,
            FLAG_FINALIZED,
            crc32(&index),
        );
        self.file.flush().map_err(|e| Error::io(ctx.clone(), e))?;
        let mut file = self
            .file
            .into_inner()
            .map_err(|e| Error::io(ctx.clone(), e.into_error()))?;
        file.seek(SeekFrom::Start(0))
            .and_then(|_| file.write_all(&header))
            .and_then(|_| file.sync_all())
            .map_err(|e| Error::io(ctx, e))?;
        Ok(ShardInfo {
            path: self.path,
            index_path: idx_path,
            num_samples: self.entries.len(),
            total_frames: self.entries.iter().map(|e| e.frames as u64).sum(),
            data_len,
        })
    }
}
