//! Weight sources: a lazy safetensors reader and name-checked loading.
//!
//! Kyutai's checkpoints are stored in bf16 (`*-candle-bf16` repos) or f32;
//! everything is converted to `f32` on load. [`SafeTensors::open`] reads only
//! the header up front and then each tensor on demand, so peak memory is the
//! converted model plus one tensor (no unsafe memory mapping).

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use super::error::{Error, Result};

/// Element type of a stored tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoredDtype {
    /// IEEE half.
    F16,
    /// bfloat16.
    BF16,
    /// IEEE single.
    F32,
    /// IEEE double.
    F64,
}

impl StoredDtype {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "F32" => Self::F32,
            "F64" => Self::F64,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

/// Header entry of one tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    /// Stored element type.
    pub dtype: StoredDtype,
    /// Shape.
    pub shape: Vec<usize>,
    /// Byte range within the data section.
    pub offsets: (usize, usize),
}

/// Anything that can hand out named `f32` tensors.
pub trait WeightSource {
    /// Shape of tensor `name`, or `None` if absent.
    fn shape(&self, name: &str) -> Option<Vec<usize>>;
    /// Read tensor `name` as `f32` (row-major).
    fn read(&mut self, name: &str) -> Result<Vec<f32>>;
    /// All tensor names, sorted.
    fn names(&self) -> Vec<String>;
}

enum Backing {
    File { file: File, data_start: u64 },
    Bytes { bytes: Vec<u8>, data_start: usize },
}

/// A safetensors file (or buffer) read lazily, tensor by tensor.
pub struct SafeTensors {
    header: BTreeMap<String, TensorInfo>,
    backing: Backing,
}

impl std::fmt::Debug for SafeTensors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafeTensors")
            .field("tensors", &self.header.len())
            .finish()
    }
}

fn parse_header(json: &[u8], data_len: Option<u64>) -> Result<BTreeMap<String, TensorInfo>> {
    let v: serde_json::Value = serde_json::from_slice(json)
        .map_err(|e| Error::Format(format!("safetensors header: {e}")))?;
    let obj = v
        .as_object()
        .ok_or_else(|| Error::Format("safetensors header is not an object".into()))?;
    let mut out = BTreeMap::new();
    for (name, entry) in obj {
        if name == "__metadata__" {
            continue;
        }
        let bad = |m: &str| Error::Format(format!("safetensors entry `{name}`: {m}"));
        let dtype_s = entry
            .get("dtype")
            .and_then(|d| d.as_str())
            .ok_or_else(|| bad("missing dtype"))?;
        let dtype = StoredDtype::parse(dtype_s)
            .ok_or_else(|| bad(&format!("unsupported dtype {dtype_s}")))?;
        let shape: Vec<usize> = entry
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or_else(|| bad("missing shape"))?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize))
            .collect::<Option<_>>()
            .ok_or_else(|| bad("bad shape"))?;
        let offs = entry
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .filter(|o| o.len() == 2)
            .ok_or_else(|| bad("missing data_offsets"))?;
        let (start, end) = match (offs[0].as_u64(), offs[1].as_u64()) {
            (Some(s), Some(e)) if s <= e => (s as usize, e as usize),
            _ => return Err(bad("bad data_offsets")),
        };
        let numel: usize = shape.iter().product();
        if end - start != numel * dtype.size() {
            return Err(bad("byte length does not match shape and dtype"));
        }
        if let Some(len) = data_len {
            if end as u64 > len {
                return Err(bad("data extends past the end of the file"));
            }
        }
        out.insert(
            name.clone(),
            TensorInfo {
                dtype,
                shape,
                offsets: (start, end),
            },
        );
    }
    Ok(out)
}

/// IEEE half to single precision (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: renormalize.
            let mut e = 113u32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// bfloat16 to single precision (exact).
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn convert(dtype: StoredDtype, raw: &[u8]) -> Vec<f32> {
    match dtype {
        StoredDtype::F32 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        StoredDtype::BF16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        StoredDtype::F16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        StoredDtype::F64 => raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
    }
}

impl SafeTensors {
    /// Open a file and read its header. Tensors are read on demand.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut n = [0u8; 8];
        file.read_exact(&mut n)?;
        let n = u64::from_le_bytes(n);
        if n > 100_000_000 || 8 + n > file_len {
            return Err(Error::Format(format!(
                "{}: bad safetensors header length {n}",
                path.display()
            )));
        }
        let mut json = vec![0u8; n as usize];
        file.read_exact(&mut json)?;
        let header = parse_header(&json, Some(file_len - 8 - n))?;
        Ok(Self {
            header,
            backing: Backing::File {
                file,
                data_start: 8 + n,
            },
        })
    }

    /// Parse an in-memory safetensors buffer.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(Error::Format("safetensors buffer too short".into()));
        }
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap_or([0; 8])) as usize;
        if 8 + n > bytes.len() {
            return Err(Error::Format("bad safetensors header length".into()));
        }
        let header = parse_header(&bytes[8..8 + n], Some((bytes.len() - 8 - n) as u64))?;
        Ok(Self {
            header,
            backing: Backing::Bytes {
                bytes,
                data_start: 8 + n,
            },
        })
    }

    /// Header entry of tensor `name`.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.header.get(name)
    }

    /// All header entries, by name.
    pub fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.header
    }
}

impl SafeTensors {
    /// The stored bytes of tensor `name` (little-endian, in its stored
    /// dtype), with that dtype. For loaders that keep the stored precision,
    /// e.g. bf16 weights uploaded to a GPU as is.
    pub fn read_raw(&mut self, name: &str) -> Result<(StoredDtype, Vec<u8>)> {
        let info = self
            .header
            .get(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))?;
        let (dtype, (s, e)) = (info.dtype, info.offsets);
        Ok((dtype, self.raw_range(s, e)?))
    }

    fn raw_range(&mut self, s: usize, e: usize) -> Result<Vec<u8>> {
        Ok(match &mut self.backing {
            Backing::Bytes { bytes, data_start } => {
                bytes[*data_start + s..*data_start + e].to_vec()
            }
            Backing::File { file, data_start } => {
                file.seek(SeekFrom::Start(*data_start + s as u64))?;
                let mut buf = vec![0u8; e - s];
                file.read_exact(&mut buf)?;
                buf
            }
        })
    }
}

impl WeightSource for SafeTensors {
    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.header.get(name).map(|i| i.shape.clone())
    }

    fn read(&mut self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .header
            .get(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))?;
        let (dtype, (s, e)) = (info.dtype, info.offsets);
        let raw = self.raw_range(s, e)?;
        Ok(convert(dtype, &raw))
    }

    fn names(&self) -> Vec<String> {
        self.header.keys().cloned().collect()
    }
}

/// An in-memory set of named `f32` tensors (handy for tests and for
/// converting from other formats).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TensorMap(pub BTreeMap<String, (Vec<usize>, Vec<f32>)>);

impl WeightSource for TensorMap {
    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.0.get(name).map(|(s, _)| s.clone())
    }

    fn read(&mut self, name: &str) -> Result<Vec<f32>> {
        self.0
            .get(name)
            .map(|(_, d)| d.clone())
            .ok_or_else(|| Error::MissingTensor(name.to_string()))
    }

    fn names(&self) -> Vec<String> {
        self.0.keys().cloned().collect()
    }
}

/// Names and shapes only; reads return empty buffers. Used to check that a
/// checkpoint's tensor names and shapes are exactly what a config needs
/// without materializing (possibly huge) weights.
pub(crate) struct ShapeManifest<'a>(pub &'a BTreeMap<String, Vec<usize>>);

impl WeightSource for ShapeManifest<'_> {
    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.0.get(name).cloned()
    }

    fn read(&mut self, name: &str) -> Result<Vec<f32>> {
        if self.0.contains_key(name) {
            Ok(Vec::new())
        } else {
            Err(Error::MissingTensor(name.to_string()))
        }
    }

    fn names(&self) -> Vec<String> {
        self.0.keys().cloned().collect()
    }
}

/// Name-checked access to a [`WeightSource`] with a VarBuilder-style prefix,
/// recording which tensors were used.
pub(crate) struct Loader<'a> {
    src: &'a mut dyn WeightSource,
    used: BTreeSet<String>,
    /// When false, reads are skipped and empty buffers returned (shape check only).
    pub materialize: bool,
}

impl<'a> Loader<'a> {
    pub fn new(src: &'a mut dyn WeightSource, materialize: bool) -> Self {
        Self {
            src,
            used: BTreeSet::new(),
            materialize,
        }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.src.shape(name).is_some()
    }

    /// Fetch `name`, which must have exactly `shape`.
    pub fn get(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let got = self
            .src
            .shape(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))?;
        if got != shape {
            return Err(Error::ShapeMismatch {
                name: name.to_string(),
                expected: shape.to_vec(),
                got,
            });
        }
        self.used.insert(name.to_string());
        if self.materialize {
            let data = self.src.read(name)?;
            let numel: usize = shape.iter().product();
            if data.len() != numel {
                return Err(Error::Format(format!(
                    "tensor `{name}`: read {} values, expected {numel}",
                    data.len()
                )));
            }
            Ok(data)
        } else {
            Ok(Vec::new())
        }
    }

    /// Names in the source that were never requested.
    pub fn unused(&self) -> Vec<String> {
        self.src
            .names()
            .into_iter()
            .filter(|n| !self.used.contains(n))
            .collect()
    }

    pub fn used_count(&self) -> usize {
        self.used.len()
    }
}

/// `prefix.name`, or `name` when the prefix is empty.
pub(crate) fn join(prefix: &str, name: impl std::fmt::Display) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal safetensors writer for tests.
    pub(crate) fn write_safetensors(tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = data.len();
            data.extend_from_slice(bytes);
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        header.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
        let h = serde_json::to_vec(&header).unwrap();
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&h);
        out.extend_from_slice(&data);
        out
    }

    #[test]
    fn half_conversions() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x0200), 2f32.powi(-15));
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xc040), -3.0);
    }

    #[test]
    fn reads_all_dtypes_from_bytes_and_file() {
        let f32s: Vec<u8> = [1.5f32, -2.0, 0.25, 8.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bf16s: Vec<u8> = [0x3f80u16, 0xc040]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let f16s: Vec<u8> = [0x3c00u16, 0xc000, 0x3800]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let f64s: Vec<u8> = [3.25f64].iter().flat_map(|v| v.to_le_bytes()).collect();
        let bytes = write_safetensors(&[
            ("a", "F32", vec![2, 2], f32s),
            ("b", "BF16", vec![2], bf16s),
            ("c", "F16", vec![3, 1], f16s),
            ("d", "F64", vec![], f64s),
        ]);
        let dir = std::env::temp_dir().join(format!("s2s-model-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.safetensors");
        std::fs::write(&path, &bytes).unwrap();
        let mut sources: Vec<Box<dyn WeightSource>> = vec![
            Box::new(SafeTensors::from_bytes(bytes).unwrap()),
            Box::new(SafeTensors::open(&path).unwrap()),
        ];
        for st in sources.iter_mut() {
            assert_eq!(st.names(), vec!["a", "b", "c", "d"]);
            assert_eq!(st.shape("c"), Some(vec![3, 1]));
            assert_eq!(st.read("a").unwrap(), vec![1.5, -2.0, 0.25, 8.0]);
            assert_eq!(st.read("b").unwrap(), vec![1.0, -3.0]);
            assert_eq!(st.read("c").unwrap(), vec![1.0, -2.0, 0.5]);
            assert_eq!(st.read("d").unwrap(), vec![3.25]);
            assert!(matches!(st.read("zz"), Err(Error::MissingTensor(_))));
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_inconsistent_headers() {
        let bytes = write_safetensors(&[("a", "F32", vec![3], vec![0u8; 8])]);
        assert!(matches!(
            SafeTensors::from_bytes(bytes),
            Err(Error::Format(_))
        ));
        let bytes = write_safetensors(&[("a", "I64", vec![1], vec![0u8; 8])]);
        assert!(SafeTensors::from_bytes(bytes).is_err());
        assert!(SafeTensors::from_bytes(vec![1, 2, 3]).is_err());
    }

    #[test]
    fn loader_checks_names_and_shapes() {
        let mut m = TensorMap::default();
        m.0.insert("x.w".into(), (vec![2, 3], vec![0.0; 6]));
        m.0.insert("x.extra".into(), (vec![1], vec![0.0]));
        let mut l = Loader::new(&mut m, true);
        assert_eq!(l.get("x.w", &[2, 3]).unwrap().len(), 6);
        assert!(matches!(
            l.get("x.w", &[3, 2]),
            Err(Error::ShapeMismatch { .. })
        ));
        assert!(matches!(l.get("x.v", &[1]), Err(Error::MissingTensor(_))));
        assert_eq!(l.unused(), vec!["x.extra".to_string()]);
        assert_eq!(join("", "a"), "a");
        assert_eq!(join("a.b", 3), "a.b.3");
    }
}
