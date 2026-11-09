//! Checkpoint access by Kyutai's tensor names.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use s2s_kernels::reference::conv1d::weight_norm_fold;
use safetensors::tensor::{Dtype, TensorInfo};
use safetensors::SafeTensors;

use crate::{MimiError, Result};

/// Somewhere to read named tensors from, converted to `f32`.
pub trait TensorSource {
    /// Shape of `name`, or `None` if absent.
    fn shape(&self, name: &str) -> Option<Vec<usize>>;
    /// Values of `name` as `f32`, row-major.
    fn load_f32(&self, name: &str) -> Result<Vec<f32>>;
    /// Every tensor name, sorted.
    fn names(&self) -> Vec<String>;
}

/// A safetensors checkpoint held in memory. Tensors are converted to `f32`
/// on access (`F32`, `F16`, `BF16` and `F64` are supported).
#[derive(Debug)]
pub struct SafetensorsCheckpoint {
    bytes: Vec<u8>,
    data_start: usize,
    infos: HashMap<String, TensorInfo>,
}

impl SafetensorsCheckpoint {
    /// Read a `.safetensors` file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_bytes(std::fs::read(path)?)
    }

    /// Parse an in-memory `.safetensors` file.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let (header_len, meta) = SafeTensors::read_metadata(&bytes)
            .map_err(|e| MimiError::Safetensors(e.to_string()))?;
        let infos = meta
            .tensors()
            .into_iter()
            .map(|(k, v)| (k, v.clone()))
            .collect();
        Ok(Self {
            bytes,
            data_start: 8 + header_len,
            infos,
        })
    }
}

impl TensorSource for SafetensorsCheckpoint {
    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.infos.get(name).map(|i| i.shape.clone())
    }

    fn load_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .infos
            .get(name)
            .ok_or_else(|| MimiError::MissingTensor(name.to_string()))?;
        let (a, b) = info.data_offsets;
        let raw = self
            .bytes
            .get(self.data_start + a..self.data_start + b)
            .ok_or_else(|| MimiError::Safetensors(format!("`{name}`: data out of bounds")))?;
        let n: usize = info.shape.iter().product();
        let out: Vec<f32> = match info.dtype {
            Dtype::F32 => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Dtype::F64 => raw
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| {
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect(),
            Dtype::BF16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            Dtype::F16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            other => {
                return Err(MimiError::UnsupportedDtype {
                    name: name.to_string(),
                    dtype: format!("{other:?}"),
                })
            }
        };
        if out.len() != n {
            return Err(MimiError::Safetensors(format!(
                "`{name}`: {} values for shape {:?}",
                out.len(),
                info.shape
            )));
        }
        Ok(out)
    }

    fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.infos.keys().cloned().collect();
        v.sort();
        v
    }
}

/// IEEE half -> single precision (exact).
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = match (exp, mant) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: normalize.
            let (mut e, mut m) = (127 - 15 + 1, mant);
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, _) => sign | 0x7f80_0000 | (mant << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

/// An in-memory tensor source (tests, conversions, synthetic checkpoints).
#[derive(Clone, Debug, Default)]
pub struct TensorMap {
    map: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
}

impl TensorMap {
    /// Empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a tensor.
    ///
    /// # Panics
    /// If `values.len()` does not match `shape`.
    pub fn insert(&mut self, name: impl Into<String>, shape: Vec<usize>, values: Vec<f32>) {
        assert_eq!(
            shape.iter().product::<usize>(),
            values.len(),
            "TensorMap::insert"
        );
        self.map.insert(name.into(), (shape, values));
    }

    /// Remove a tensor, returning it.
    pub fn remove(&mut self, name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
        self.map.remove(name)
    }

    /// Copy every tensor of `src`.
    pub fn from_source(src: &dyn TensorSource) -> Result<Self> {
        let mut m = Self::new();
        for name in src.names() {
            let shape = src.shape(&name).unwrap_or_default();
            m.insert(name.clone(), shape, src.load_f32(&name)?);
        }
        Ok(m)
    }

    /// Replace every `<p>.weight_g` / `<p>.weight_v` pair by the folded
    /// `<p>.weight` (what an exported, weight-norm-free checkpoint holds).
    pub fn fold_weight_norm(&mut self) {
        let prefixes: Vec<String> = self
            .map
            .keys()
            .filter_map(|k| k.strip_suffix(".weight_g").map(str::to_string))
            .collect();
        for p in prefixes {
            let (_, g) = self.map.remove(&format!("{p}.weight_g")).expect("listed");
            let (shape, v) = self
                .map
                .remove(&format!("{p}.weight_v"))
                .expect("weight_g without weight_v");
            let w = weight_norm_fold(&g, &v, shape[0], v.len() / shape[0]);
            self.map.insert(format!("{p}.weight"), (shape, w));
        }
    }
}

impl TensorSource for TensorMap {
    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.map.get(name).map(|(s, _)| s.clone())
    }

    fn load_f32(&self, name: &str) -> Result<Vec<f32>> {
        self.map
            .get(name)
            .map(|(_, v)| v.clone())
            .ok_or_else(|| MimiError::MissingTensor(name.to_string()))
    }

    fn names(&self) -> Vec<String> {
        self.map.keys().cloned().collect()
    }
}

/// Reads tensors by name with shape checks and records which were used.
pub(crate) struct Loader<'a> {
    src: &'a dyn TensorSource,
    used: RefCell<BTreeSet<String>>,
}

impl<'a> Loader<'a> {
    pub(crate) fn new(src: &'a dyn TensorSource) -> Self {
        Self {
            src,
            used: RefCell::new(BTreeSet::new()),
        }
    }

    pub(crate) fn has(&self, name: &str) -> bool {
        self.src.shape(name).is_some()
    }

    /// Load `name`, whose shape must equal `shape`.
    pub(crate) fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let got = self
            .src
            .shape(name)
            .ok_or_else(|| MimiError::MissingTensor(name.to_string()))?;
        if got != shape {
            return Err(MimiError::ShapeMismatch {
                name: name.to_string(),
                expected: shape.to_vec(),
                got,
            });
        }
        self.used.borrow_mut().insert(name.to_string());
        self.src.load_f32(name)
    }

    /// Load `name` whose element count must be `n`, accepting any of `shapes`.
    pub(crate) fn get_any(&self, name: &str, shapes: &[&[usize]]) -> Result<Vec<f32>> {
        let got = self
            .src
            .shape(name)
            .ok_or_else(|| MimiError::MissingTensor(name.to_string()))?;
        match shapes.iter().find(|s| **s == got.as_slice()) {
            Some(s) => self.get(name, s),
            None => Err(MimiError::ShapeMismatch {
                name: name.to_string(),
                expected: shapes[0].to_vec(),
                got,
            }),
        }
    }

    /// A (possibly weight-normalized) conv weight of `shape = [d0, d1, k]`:
    /// `<prefix>.weight`, or `<prefix>.weight_g` `[d0, 1, 1]` and
    /// `<prefix>.weight_v` `[d0, d1, k]` folded like Kyutai's loader.
    pub(crate) fn conv_weight(&self, prefix: &str, shape: [usize; 3]) -> Result<Vec<f32>> {
        let w = format!("{prefix}.weight");
        if self.has(&w) {
            return self.get(&w, &shape);
        }
        let g = self.get(&format!("{prefix}.weight_g"), &[shape[0], 1, 1])?;
        let v = self.get(&format!("{prefix}.weight_v"), &shape)?;
        Ok(weight_norm_fold(&g, &v, shape[0], shape[1] * shape[2]))
    }

    /// Names in the source that were never loaded.
    pub(crate) fn unused(&self) -> Vec<String> {
        let used = self.used.borrow();
        self.src
            .names()
            .into_iter()
            .filter(|n| !used.contains(n))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_precision_conversion() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0x8000).to_bits(), 0x8000_0000);
    }

    #[test]
    fn safetensors_dtypes() {
        use safetensors::tensor::TensorView;
        let f32s: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bf16s: Vec<u8> = [0x3fc0u16, 0xc000]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let f16s: Vec<u8> = [0x3e00u16, 0xc000]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let i32s: Vec<u8> = [1i32, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
        let views = vec![
            ("a", TensorView::new(Dtype::F32, vec![2], &f32s).unwrap()),
            (
                "b",
                TensorView::new(Dtype::BF16, vec![1, 2], &bf16s).unwrap(),
            ),
            ("c", TensorView::new(Dtype::F16, vec![2], &f16s).unwrap()),
            ("d", TensorView::new(Dtype::I32, vec![2], &i32s).unwrap()),
        ];
        let bytes = safetensors::serialize(views, None).unwrap();
        let ck = SafetensorsCheckpoint::from_bytes(bytes).unwrap();
        assert_eq!(ck.names(), ["a", "b", "c", "d"]);
        assert_eq!(ck.shape("b"), Some(vec![1, 2]));
        assert_eq!(ck.load_f32("a").unwrap(), [1.5, -2.0]);
        assert_eq!(ck.load_f32("b").unwrap(), [1.5, -2.0]);
        assert_eq!(ck.load_f32("c").unwrap(), [1.5, -2.0]);
        assert!(matches!(
            ck.load_f32("d"),
            Err(MimiError::UnsupportedDtype { .. })
        ));
        assert!(matches!(ck.load_f32("e"), Err(MimiError::MissingTensor(_))));
    }

    #[test]
    fn loader_shapes_and_weight_norm() {
        let mut m = TensorMap::new();
        m.insert("c.weight_g", vec![2, 1, 1], vec![2.0, 1.0]);
        m.insert("c.weight_v", vec![2, 1, 2], vec![3.0, 4.0, 0.0, -5.0]);
        m.insert("x", vec![3], vec![1.0, 2.0, 3.0]);
        let l = Loader::new(&m);
        assert_eq!(
            l.conv_weight("c", [2, 1, 2]).unwrap(),
            [1.2, 1.6, 0.0, -1.0]
        );
        assert!(matches!(
            l.get("x", &[4]),
            Err(MimiError::ShapeMismatch { .. })
        ));
        assert_eq!(l.unused(), ["x"]);
        assert_eq!(l.get_any("x", &[&[1, 3], &[3]]).unwrap(), [1.0, 2.0, 3.0]);
        assert!(l.unused().is_empty());
        let mut folded = m.clone();
        folded.fold_weight_norm();
        assert_eq!(folded.names(), ["c.weight", "x"]);
        assert_eq!(
            Loader::new(&folded).conv_weight("c", [2, 1, 2]).unwrap(),
            [1.2, 1.6, 0.0, -1.0]
        );
    }
}
