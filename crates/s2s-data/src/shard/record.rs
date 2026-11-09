//! Encoding and decoding of one sample record (see the [module docs](super)).

use s2s_core::streams::StreamLayout;

use crate::sample::{Injection, InjectionKind, Sample, SampleMeta, WordTiming};
use crate::{Error, Result};

/// Magic at the start of every record.
pub const RECORD_MAGIC: [u8; 4] = *b"SMPL";
/// Length of the fixed record header.
pub const RECORD_HEADER_LEN: usize = 64;
/// Bytes per injection-table entry.
const INJ_ENTRY: u64 = 12;
/// Bytes per word-table entry.
const WORD_ENTRY: u64 = 16;

fn fmt_err(m: impl Into<String>) -> Error {
    Error::Format(m.into())
}

/// Bounds-checked little-endian reader over a byte slice.
pub(crate) struct Cur<'a> {
    b: &'a [u8],
    pos: usize,
    what: &'static str,
}

impl<'a> Cur<'a> {
    pub(crate) fn new(b: &'a [u8], what: &'static str) -> Self {
        Self { b, pos: 0, what }
    }

    pub(crate) fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.b.len());
        match end {
            Some(end) => {
                let s = &self.b[self.pos..end];
                self.pos = end;
                Ok(s)
            }
            None => Err(fmt_err(format!(
                "{}: truncated (need {n} bytes at offset {}, have {})",
                self.what,
                self.pos,
                self.b.len()
            ))),
        }
    }

    pub(crate) fn u16(&mut self) -> Result<u16> {
        let s = self.bytes(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    pub(crate) fn u32(&mut self) -> Result<u32> {
        let s = self.bytes(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    pub(crate) fn u64(&mut self) -> Result<u64> {
        let s = self.bytes(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        Ok(u64::from_le_bytes(a))
    }

    pub(crate) fn u32_vec(&mut self, n: usize) -> Result<Vec<u32>> {
        let len = n
            .checked_mul(4)
            .ok_or_else(|| fmt_err(format!("{}: length overflow", self.what)))?;
        let s = self.bytes(len)?;
        Ok(s.as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub(crate) fn pos(&self) -> usize {
        self.pos
    }
}

fn to_u32(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::Invalid(format!("{what} = {v} does not fit in u32")))
}

fn to_u16(v: usize, what: &str) -> Result<u16> {
    u16::try_from(v).map_err(|_| Error::Invalid(format!("{what} = {v} does not fit in u16")))
}

/// The fixed header of a record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RecordHeader {
    pub layout: StreamLayout,
    pub frames: u32,
    pub num_streams: u32,
    pub text_vocab: u32,
    pub text_pad: u32,
    pub num_injections: u32,
    pub num_injection_tokens: u32,
    pub num_words: u32,
    pub word_text_len: u32,
    pub meta_len: u32,
}

impl RecordHeader {
    fn parse(b: &[u8]) -> Result<Self> {
        let mut c = Cur::new(b, "record header");
        if c.bytes(4)? != RECORD_MAGIC {
            return Err(fmt_err("bad record magic"));
        }
        let header_len = c.u32()?;
        if header_len as usize != RECORD_HEADER_LEN {
            return Err(fmt_err(format!(
                "unsupported record header length {header_len}"
            )));
        }
        let frames = c.u32()?;
        let num_streams = c.u32()?;
        let layout = StreamLayout {
            text_slots: c.u16()? as usize,
            codebooks: c.u16()? as usize,
            user_text_delay: c.u16()? as usize,
            acoustic_delay: c.u16()? as usize,
        };
        let h = Self {
            layout,
            frames,
            num_streams,
            text_vocab: c.u32()?,
            text_pad: c.u32()?,
            num_injections: c.u32()?,
            num_injection_tokens: c.u32()?,
            num_words: c.u32()?,
            word_text_len: c.u32()?,
            meta_len: c.u32()?,
        };
        if layout.num_streams() != num_streams as usize {
            return Err(fmt_err(format!(
                "record says {num_streams} streams but its layout has {}",
                layout.num_streams()
            )));
        }
        Ok(h)
    }

    /// Byte length of the record body implied by the header (no padding).
    fn total_len(&self) -> Option<u64> {
        let tok = (self.frames as u64)
            .checked_mul(self.num_streams as u64)?
            .checked_mul(4)?;
        let parts = [
            RECORD_HEADER_LEN as u64,
            tok,
            self.num_injections as u64 * INJ_ENTRY,
            self.num_injection_tokens as u64 * 4,
            self.num_words as u64 * WORD_ENTRY,
            self.word_text_len as u64,
            self.meta_len as u64,
        ];
        parts.iter().try_fold(0u64, |a, &p| a.checked_add(p))
    }

    /// Byte offset of the metadata JSON within the record.
    fn meta_offset(&self) -> Option<u64> {
        self.total_len()?.checked_sub(self.meta_len as u64)
    }
}

/// Encode a sample as one record (no trailing alignment padding).
///
/// Checks structural consistency (grid size, counts fitting their on-disk
/// widths) but not token ranges; see [`crate::validate`].
pub fn encode_record(s: &Sample) -> Result<Vec<u8>> {
    let ns = s.layout.num_streams();
    let expect = s
        .frames
        .checked_mul(ns)
        .ok_or_else(|| Error::Invalid("frames * streams overflows".into()))?;
    if s.tokens.len() != expect {
        return Err(Error::Invalid(format!(
            "token grid has {} entries, expected frames {} x streams {ns} = {expect}",
            s.tokens.len(),
            s.frames
        )));
    }
    let meta = serde_json::to_vec(&s.meta)?;
    let word_text_len: usize = s.words.iter().map(|w| w.text.len()).sum();
    let inj_tokens = s.injection_tokens();

    let mut out = Vec::with_capacity(
        RECORD_HEADER_LEN
            + 4 * (s.tokens.len() + inj_tokens)
            + 12 * s.injections.len()
            + 16 * s.words.len()
            + word_text_len
            + meta.len(),
    );
    out.extend_from_slice(&RECORD_MAGIC);
    out.extend_from_slice(&(RECORD_HEADER_LEN as u32).to_le_bytes());
    out.extend_from_slice(&to_u32(s.frames, "frames")?.to_le_bytes());
    out.extend_from_slice(&to_u32(ns, "num_streams")?.to_le_bytes());
    for (v, what) in [
        (s.layout.text_slots, "text_slots"),
        (s.layout.codebooks, "codebooks"),
        (s.layout.user_text_delay, "user_text_delay"),
        (s.layout.acoustic_delay, "acoustic_delay"),
    ] {
        out.extend_from_slice(&to_u16(v, what)?.to_le_bytes());
    }
    for v in [
        s.text_vocab,
        s.text_pad,
        to_u32(s.injections.len(), "injections")?,
        to_u32(inj_tokens, "injection tokens")?,
        to_u32(s.words.len(), "words")?,
        to_u32(word_text_len, "word text bytes")?,
        to_u32(meta.len(), "metadata bytes")?,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.resize(RECORD_HEADER_LEN, 0);

    for &t in &s.tokens {
        out.extend_from_slice(&t.to_le_bytes());
    }
    for inj in &s.injections {
        out.extend_from_slice(&inj.frame.to_le_bytes());
        out.extend_from_slice(&inj.kind.code().to_le_bytes());
        out.extend_from_slice(&to_u32(inj.tokens.len(), "injection length")?.to_le_bytes());
    }
    for inj in &s.injections {
        for &t in &inj.tokens {
            out.extend_from_slice(&t.to_le_bytes());
        }
    }
    for w in &s.words {
        out.extend_from_slice(&w.stream.to_le_bytes());
        out.extend_from_slice(&w.start_ms.to_le_bytes());
        out.extend_from_slice(&w.end_ms.to_le_bytes());
        out.extend_from_slice(&to_u32(w.text.len(), "word length")?.to_le_bytes());
    }
    for w in &s.words {
        out.extend_from_slice(w.text.as_bytes());
    }
    out.extend_from_slice(&meta);
    Ok(out)
}

fn parse_checked_header(b: &[u8]) -> Result<RecordHeader> {
    let h = RecordHeader::parse(b)?;
    let total = h
        .total_len()
        .ok_or_else(|| fmt_err("record section lengths overflow"))?;
    if total != b.len() as u64 {
        return Err(fmt_err(format!(
            "record length {} does not match the {total} bytes its header describes",
            b.len()
        )));
    }
    Ok(h)
}

/// Decode a record produced by [`encode_record`]. Any inconsistency is a
/// [`Error::Format`]; this never panics on malformed input.
pub fn decode_record(b: &[u8]) -> Result<Sample> {
    let h = parse_checked_header(b)?;
    let mut c = Cur::new(b, "record");
    c.bytes(RECORD_HEADER_LEN)?;
    let tokens = c.u32_vec(h.frames as usize * h.num_streams as usize)?;

    let mut inj_heads = Vec::with_capacity(h.num_injections as usize);
    let mut inj_sum: u64 = 0;
    for _ in 0..h.num_injections {
        let frame = c.u32()?;
        let code = c.u32()?;
        let kind = InjectionKind::from_code(code)
            .ok_or_else(|| fmt_err(format!("unknown injection kind {code}")))?;
        let n = c.u32()?;
        inj_sum += n as u64;
        inj_heads.push((frame, kind, n));
    }
    if inj_sum != h.num_injection_tokens as u64 {
        return Err(fmt_err(format!(
            "injection lengths sum to {inj_sum}, header says {}",
            h.num_injection_tokens
        )));
    }
    let mut injections = Vec::with_capacity(inj_heads.len());
    for (frame, kind, n) in inj_heads {
        injections.push(Injection {
            frame,
            kind,
            tokens: c.u32_vec(n as usize)?,
        });
    }

    let mut word_heads = Vec::with_capacity(h.num_words as usize);
    let mut text_sum: u64 = 0;
    for _ in 0..h.num_words {
        let stream = c.u32()?;
        let start_ms = c.u32()?;
        let end_ms = c.u32()?;
        let len = c.u32()?;
        text_sum += len as u64;
        word_heads.push((stream, start_ms, end_ms, len));
    }
    if text_sum != h.word_text_len as u64 {
        return Err(fmt_err(format!(
            "word lengths sum to {text_sum}, header says {}",
            h.word_text_len
        )));
    }
    let mut words = Vec::with_capacity(word_heads.len());
    for (stream, start_ms, end_ms, len) in word_heads {
        let text = std::str::from_utf8(c.bytes(len as usize)?)
            .map_err(|_| fmt_err("word text is not UTF-8"))?
            .to_string();
        words.push(WordTiming {
            stream,
            start_ms,
            end_ms,
            text,
        });
    }
    let meta: SampleMeta = serde_json::from_slice(c.bytes(h.meta_len as usize)?)
        .map_err(|e| fmt_err(format!("metadata JSON: {e}")))?;
    debug_assert_eq!(c.pos(), b.len());
    Ok(Sample {
        layout: h.layout,
        frames: h.frames as usize,
        text_vocab: h.text_vocab,
        text_pad: h.text_pad,
        tokens,
        injections,
        words,
        meta,
    })
}

/// Decode only the layout, frame count and metadata of a record, skipping
/// the token grid.
pub(crate) fn decode_record_meta(b: &[u8]) -> Result<(RecordHeader, SampleMeta)> {
    let h = parse_checked_header(b)?;
    let off = h
        .meta_offset()
        .ok_or_else(|| fmt_err("record section lengths overflow"))? as usize;
    let meta =
        serde_json::from_slice(&b[off..]).map_err(|e| fmt_err(format!("metadata JSON: {e}")))?;
    Ok((h, meta))
}
