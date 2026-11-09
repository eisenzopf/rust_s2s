//! Error type.

use std::fmt;

/// Errors from loading or running Mimi.
#[derive(Debug)]
pub enum MimiError {
    /// Reading a file failed.
    Io(std::io::Error),
    /// The safetensors file is malformed.
    Safetensors(String),
    /// A required tensor is missing from the checkpoint.
    MissingTensor(String),
    /// A tensor has an unexpected shape.
    ShapeMismatch {
        /// Tensor name.
        name: String,
        /// Expected shape.
        expected: Vec<usize>,
        /// Shape in the checkpoint.
        got: Vec<usize>,
    },
    /// A tensor has a dtype the loader cannot convert to `f32`.
    UnsupportedDtype {
        /// Tensor name.
        name: String,
        /// Its dtype, as safetensors names it.
        dtype: String,
    },
    /// The configuration is inconsistent.
    Config(String),
    /// Bad input (wrong length, code out of range, ...).
    Input(String),
}

impl fmt::Display for MimiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MimiError::Io(e) => write!(f, "io error: {e}"),
            MimiError::Safetensors(e) => write!(f, "safetensors error: {e}"),
            MimiError::MissingTensor(n) => write!(f, "missing tensor `{n}`"),
            MimiError::ShapeMismatch {
                name,
                expected,
                got,
            } => write!(
                f,
                "tensor `{name}`: expected shape {expected:?}, got {got:?}"
            ),
            MimiError::UnsupportedDtype { name, dtype } => {
                write!(f, "tensor `{name}`: unsupported dtype {dtype}")
            }
            MimiError::Config(m) => write!(f, "invalid config: {m}"),
            MimiError::Input(m) => write!(f, "invalid input: {m}"),
        }
    }
}

impl std::error::Error for MimiError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MimiError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for MimiError {
    fn from(e: std::io::Error) -> Self {
        MimiError::Io(e)
    }
}

/// Result alias using [`MimiError`].
pub type Result<T> = std::result::Result<T, MimiError>;
