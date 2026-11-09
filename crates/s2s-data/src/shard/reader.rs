use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use super::record::{decode_record, decode_record_meta, Cur};
use super::{
    crc32, index_path, IndexEntry, FLAG_FINALIZED, FORMAT_VERSION, INDEX_ENTRY_LEN,
    INDEX_HEADER_LEN, INDEX_MAGIC, RECORD_ALIGN, SHARD_HEADER_LEN, SHARD_MAGIC,
};
use crate::{Error, Result, Sample, SampleMeta};

#[derive(Debug)]
enum Backing {
    Mapped(Mmap),
    Owned(Vec<u8>),
}

impl Backing {
    fn bytes(&self) -> &[u8] {
        match self {
            Backing::Mapped(m) => m,
            Backing::Owned(v) => v,
        }
    }
}

/// Random-access reader over one shard. The data file is memory-mapped
/// ([`open`](Self::open)) or held in memory ([`open_in_memory`](Self::open_in_memory),
/// [`from_bytes`](Self::from_bytes)). Cheap to share across threads.
#[derive(Debug)]
pub struct ShardReader {
    path: Option<PathBuf>,
    data: Backing,
    entries: Vec<IndexEntry>,
}

fn fmt_err(m: impl Into<String>) -> Error {
    Error::Format(m.into())
}

fn read_file(p: &Path) -> Result<Vec<u8>> {
    std::fs::read(p).map_err(|e| Error::io(p.display().to_string(), e))
}

impl ShardReader {
    /// Open the shard at `path` (and its index, [`index_path`]), memory-mapping
    /// the data file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let ctx = || path.display().to_string();
        let file = File::open(path).map_err(|e| Error::io(ctx(), e))?;
        let len = file.metadata().map_err(|e| Error::io(ctx(), e))?.len();
        if len < SHARD_HEADER_LEN as u64 {
            return Err(fmt_err(format!(
                "{}: file is {len} bytes, shorter than the {SHARD_HEADER_LEN}-byte header",
                ctx()
            )));
        }
        let map = super::mmap::map_readonly(&file).map_err(|e| Error::io(ctx(), e))?;
        let index = read_file(&index_path(path))?;
        Self::build(Some(path.to_path_buf()), Backing::Mapped(map), &index)
    }

    /// Open the shard at `path`, reading the data file into memory instead of
    /// mapping it.
    pub fn open_in_memory(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let data = read_file(path)?;
        let index = read_file(&index_path(path))?;
        Self::build(Some(path.to_path_buf()), Backing::Owned(data), &index)
    }

    /// Build a reader from the raw bytes of a data file and its index.
    pub fn from_bytes(data: Vec<u8>, index: &[u8]) -> Result<Self> {
        Self::build(None, Backing::Owned(data), index)
    }

    fn build(path: Option<PathBuf>, data: Backing, index: &[u8]) -> Result<Self> {
        let entries = parse(data.bytes(), index)?;
        Ok(Self {
            path,
            data,
            entries,
        })
    }

    /// The data file path, if opened from disk.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the shard holds no samples.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Index entries, in sample order.
    pub fn entries(&self) -> &[IndexEntry] {
        &self.entries
    }

    fn record(&self, i: usize) -> Result<&[u8]> {
        let e = self.entries.get(i).ok_or_else(|| {
            Error::Invalid(format!("sample {i} out of range ({} samples)", self.len()))
        })?;
        // `parse` checked that the entry lies inside the data.
        let rec = &self.data.bytes()[e.offset as usize..(e.offset + e.len) as usize];
        let c = crc32(rec);
        if c != e.crc32 {
            return Err(fmt_err(format!(
                "sample {i}: CRC-32 mismatch (index {:08x}, data {c:08x})",
                e.crc32
            )));
        }
        Ok(rec)
    }

    /// Read and decode sample `i` (CRC-checked).
    pub fn get(&self, i: usize) -> Result<Sample> {
        let s = decode_record(self.record(i)?)?;
        let e = &self.entries[i];
        if s.frames != e.frames as usize || s.num_streams() != e.num_streams as usize {
            return Err(fmt_err(format!(
                "sample {i}: index says {} frames x {} streams, record has {} x {}",
                e.frames,
                e.num_streams,
                s.frames,
                s.num_streams()
            )));
        }
        Ok(s)
    }

    /// Read only the metadata of sample `i` (CRC-checked, token grid not decoded).
    pub fn meta(&self, i: usize) -> Result<SampleMeta> {
        decode_record_meta(self.record(i)?).map(|(_, m)| m)
    }

    /// Check every sample's CRC and decode it. Returns the first error.
    pub fn verify(&self) -> Result<()> {
        (0..self.len()).try_for_each(|i| self.get(i).map(|_| ()))
    }
}

/// Validate the data header and the index; return the entries.
fn parse(data: &[u8], index: &[u8]) -> Result<Vec<IndexEntry>> {
    // Data header.
    if data.len() < SHARD_HEADER_LEN {
        return Err(fmt_err(format!(
            "data file is {} bytes, shorter than the header",
            data.len()
        )));
    }
    let mut c = Cur::new(&data[..SHARD_HEADER_LEN], "shard header");
    if c.bytes(8)? != SHARD_MAGIC {
        return Err(fmt_err("not a shard file (bad magic)"));
    }
    let stored_crc = u32::from_le_bytes(
        data[SHARD_HEADER_LEN - 4..SHARD_HEADER_LEN]
            .try_into()
            .map_err(|_| fmt_err("shard header"))?,
    );
    if crc32(&data[..SHARD_HEADER_LEN - 4]) != stored_crc {
        return Err(fmt_err("shard header CRC mismatch"));
    }
    let version = c.u32()?;
    if version != FORMAT_VERSION {
        return Err(fmt_err(format!("unsupported shard version {version}")));
    }
    let header_len = c.u32()?;
    if header_len as usize != SHARD_HEADER_LEN {
        return Err(fmt_err(format!(
            "unsupported shard header length {header_len}"
        )));
    }
    let n = c.u64()?;
    let data_len = c.u64()?;
    let flags = c.u32()?;
    let index_crc = c.u32()?;
    if flags & FLAG_FINALIZED == 0 {
        return Err(fmt_err("shard was not finalized (writer did not finish)"));
    }
    if data_len != data.len() as u64 {
        return Err(fmt_err(format!(
            "data file is {} bytes but its header says {data_len} (truncated or extended)",
            data.len()
        )));
    }

    // Index.
    if index.len() < INDEX_HEADER_LEN + 4 {
        return Err(fmt_err(format!(
            "index is {} bytes, too short",
            index.len()
        )));
    }
    let body = &index[..index.len() - 4];
    let idx_crc_stored = u32::from_le_bytes(
        index[index.len() - 4..]
            .try_into()
            .map_err(|_| fmt_err("index"))?,
    );
    if crc32(body) != idx_crc_stored {
        return Err(fmt_err("index CRC mismatch"));
    }
    if crc32(index) != index_crc {
        return Err(fmt_err(
            "index does not belong to this data file (CRC differs)",
        ));
    }
    let mut c = Cur::new(body, "index");
    if c.bytes(8)? != INDEX_MAGIC {
        return Err(fmt_err("not a shard index (bad magic)"));
    }
    let version = c.u32()?;
    if version != FORMAT_VERSION {
        return Err(fmt_err(format!("unsupported index version {version}")));
    }
    let entry_len = c.u32()?;
    if entry_len as usize != INDEX_ENTRY_LEN {
        return Err(fmt_err(format!(
            "unsupported index entry length {entry_len}"
        )));
    }
    let idx_n = c.u64()?;
    let idx_data_len = c.u64()?;
    if idx_n != n {
        return Err(fmt_err(format!(
            "index lists {idx_n} samples, data header says {n}"
        )));
    }
    if idx_data_len != data_len {
        return Err(fmt_err("index and data header disagree on the data length"));
    }
    let expect = (n as u128) * INDEX_ENTRY_LEN as u128 + INDEX_HEADER_LEN as u128;
    if body.len() as u128 != expect {
        return Err(fmt_err(format!(
            "index body is {} bytes, expected {expect} for {n} entries",
            body.len()
        )));
    }
    let mut entries = Vec::with_capacity(n as usize);
    let mut prev_end = SHARD_HEADER_LEN as u64;
    for i in 0..n {
        let e = IndexEntry {
            offset: c.u64()?,
            len: c.u64()?,
            crc32: c.u32()?,
            frames: c.u32()?,
            num_streams: c.u32()?,
        };
        c.u32()?; // reserved
        let end = e.offset.checked_add(e.len);
        let ok = e.offset >= prev_end
            && e.offset.is_multiple_of(RECORD_ALIGN)
            && matches!(end, Some(end) if end <= data_len);
        if !ok {
            return Err(fmt_err(format!(
                "index entry {i} (offset {}, len {}) is out of bounds, misaligned or overlapping",
                e.offset, e.len
            )));
        }
        if data[prev_end as usize..e.offset as usize]
            .iter()
            .any(|&b| b != 0)
        {
            return Err(fmt_err(format!("non-zero padding before sample {i}")));
        }
        prev_end = e.offset + e.len;
        entries.push(e);
    }
    if data[prev_end as usize..].iter().any(|&b| b != 0) {
        return Err(fmt_err("non-zero bytes after the last sample"));
    }
    Ok(entries)
}
