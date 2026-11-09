//! Checkpoints as safetensors: parameters, optimizer moments and a manifest.
//!
//! Tensors are stored in `f32` little-endian under these names:
//! - `param/<name>` — parameter values
//! - `adam_m/<name>`, `adam_v/<name>` — AdamW moments
//!
//! The safetensors `__metadata__` map holds one key, `manifest`, with the
//! JSON-encoded [`Manifest`] (step counts, data cursor, schedule, config).
//! Loading checks names and shapes against the model, so a checkpoint can't
//! be silently applied to the wrong architecture.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};

use s2s_nn::{Module, Param};

use crate::comm::CommError;
use crate::data::DataCursor;
use crate::optim::{AdamW, Moments};
use crate::schedule::LrSchedule;

/// Everything besides tensors that a resume needs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// Format version.
    pub version: u32,
    /// Optimizer steps taken.
    pub step: u64,
    /// Data sampler position.
    pub data: DataCursor,
    /// Learning-rate schedule.
    pub schedule: LrSchedule,
    /// Free-form model/run configuration (for humans and sanity checks).
    pub config: serde_json::Value,
}

/// Checkpoint errors.
#[derive(Debug)]
pub enum CheckpointError {
    /// Filesystem error.
    Io(std::io::Error),
    /// Malformed safetensors data.
    Format(String),
    /// The checkpoint does not match the model.
    Mismatch(String),
    /// A collective failed while saving or loading from a data-parallel group.
    Comm(CommError),
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointError::Io(e) => write!(f, "checkpoint I/O: {e}"),
            CheckpointError::Format(e) => write!(f, "checkpoint format: {e}"),
            CheckpointError::Mismatch(e) => write!(f, "checkpoint mismatch: {e}"),
            CheckpointError::Comm(e) => write!(f, "checkpoint: {e}"),
        }
    }
}

impl std::error::Error for CheckpointError {}

impl From<CommError> for CheckpointError {
    fn from(e: CommError) -> Self {
        CheckpointError::Comm(e)
    }
}

impl From<std::io::Error> for CheckpointError {
    fn from(e: std::io::Error) -> Self {
        CheckpointError::Io(e)
    }
}

fn to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn from_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Serialize model + optimizer + manifest to safetensors bytes.
pub fn to_safetensors(
    model: &dyn Module<f32>,
    opt: &AdamW<f32>,
    manifest: &Manifest,
) -> Result<Vec<u8>, CheckpointError> {
    let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    model.visit_params(&mut |p: &Param<f32>| {
        owned.push((
            format!("param/{}", p.name),
            p.shape.clone(),
            to_bytes(&p.value),
        ));
        if let Some(mom) = opt.state.get(&p.name) {
            owned.push((
                format!("adam_m/{}", p.name),
                p.shape.clone(),
                to_bytes(&mom.m),
            ));
            owned.push((
                format!("adam_v/{}", p.name),
                p.shape.clone(),
                to_bytes(&mom.v),
            ));
        }
    });
    let mut views = BTreeMap::new();
    for (name, shape, bytes) in &owned {
        let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
            .map_err(|e| CheckpointError::Format(e.to_string()))?;
        views.insert(name.clone(), view);
    }
    let meta: HashMap<String, String> = [(
        "manifest".to_string(),
        serde_json::to_string(manifest).map_err(|e| CheckpointError::Format(e.to_string()))?,
    )]
    .into();
    safetensors::serialize(views, Some(meta)).map_err(|e| CheckpointError::Format(e.to_string()))
}

/// Restore model + optimizer from safetensors bytes; returns the manifest.
pub fn from_safetensors(
    bytes: &[u8],
    model: &mut dyn Module<f32>,
    opt: &mut AdamW<f32>,
) -> Result<Manifest, CheckpointError> {
    let (_, meta) =
        SafeTensors::read_metadata(bytes).map_err(|e| CheckpointError::Format(e.to_string()))?;
    let manifest_json = meta
        .metadata()
        .as_ref()
        .and_then(|m| m.get("manifest").cloned())
        .ok_or_else(|| CheckpointError::Format("missing manifest".into()))?;
    let manifest: Manifest =
        serde_json::from_str(&manifest_json).map_err(|e| CheckpointError::Format(e.to_string()))?;
    let st = SafeTensors::deserialize(bytes).map_err(|e| CheckpointError::Format(e.to_string()))?;

    let read = |name: &str, shape: &[usize]| -> Result<Option<Vec<f32>>, CheckpointError> {
        match st.tensor(name) {
            Ok(t) => {
                if t.dtype() != Dtype::F32 || t.shape() != shape {
                    return Err(CheckpointError::Mismatch(format!(
                        "{name}: stored {:?} {:?}, model expects F32 {shape:?}",
                        t.dtype(),
                        t.shape()
                    )));
                }
                Ok(Some(from_bytes(t.data())))
            }
            Err(_) => Ok(None),
        }
    };

    let mut err = None;
    let mut new_state = BTreeMap::new();
    let mut expected = 0usize;
    model.visit_params_mut(&mut |p: &mut Param<f32>| {
        if err.is_some() {
            return;
        }
        expected += 1;
        match read(&format!("param/{}", p.name), &p.shape) {
            Ok(Some(v)) => p.value = v,
            Ok(None) => {
                err = Some(CheckpointError::Mismatch(format!(
                    "missing param/{}",
                    p.name
                )))
            }
            Err(e) => err = Some(e),
        }
        let m = read(&format!("adam_m/{}", p.name), &p.shape);
        let v = read(&format!("adam_v/{}", p.name), &p.shape);
        match (m, v) {
            (Ok(Some(m)), Ok(Some(v))) => {
                new_state.insert(p.name.clone(), Moments { m, v });
            }
            (Ok(None), Ok(None)) => {}
            (Err(e), _) | (_, Err(e)) => err = Some(e),
            _ => {
                err = Some(CheckpointError::Mismatch(format!(
                    "{}: only one Adam moment stored",
                    p.name
                )))
            }
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    let stored_params = st
        .names()
        .iter()
        .filter(|n| n.starts_with("param/"))
        .count();
    if stored_params != expected {
        return Err(CheckpointError::Mismatch(format!(
            "checkpoint has {stored_params} parameters, model has {expected}"
        )));
    }
    opt.state = new_state;
    opt.step = manifest.step;
    Ok(manifest)
}

/// Write a checkpoint file (atomically: temp file then rename).
pub fn save(
    path: &Path,
    model: &dyn Module<f32>,
    opt: &AdamW<f32>,
    manifest: &Manifest,
) -> Result<(), CheckpointError> {
    let bytes = to_safetensors(model, opt, manifest)?;
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Read a checkpoint file into `model` and `opt`.
pub fn load(
    path: &Path,
    model: &mut dyn Module<f32>,
    opt: &mut AdamW<f32>,
) -> Result<Manifest, CheckpointError> {
    let bytes = std::fs::read(path)?;
    from_safetensors(&bytes, model, opt)
}
