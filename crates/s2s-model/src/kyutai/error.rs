//! Error type for the Kyutai port.

use std::fmt;

/// Errors from loading or running a Kyutai model.
#[derive(Debug)]
pub enum Error {
    /// Filesystem error.
    Io(std::io::Error),
    /// Malformed file (safetensors header, JSON, TOML).
    Format(String),
    /// Invalid or inconsistent configuration.
    Config(String),
    /// A tensor the config requires is not in the checkpoint.
    MissingTensor(String),
    /// A checkpoint tensor has an unexpected shape.
    ShapeMismatch {
        /// Tensor name.
        name: String,
        /// Shape the config implies.
        expected: Vec<usize>,
        /// Shape in the checkpoint.
        got: Vec<usize>,
    },
    /// A feature of Kyutai's code that this port does not implement.
    Unsupported(String),
    /// Bad runtime input (token out of range, wrong number of codebooks, ...).
    Input(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Format(m) => write!(f, "format error: {m}"),
            Error::Config(m) => write!(f, "config error: {m}"),
            Error::MissingTensor(n) => write!(f, "missing tensor `{n}`"),
            Error::ShapeMismatch {
                name,
                expected,
                got,
            } => write!(
                f,
                "tensor `{name}`: expected shape {expected:?}, got {got:?}"
            ),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Input(m) => write!(f, "invalid input: {m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// Result alias for this module.
pub type Result<T> = std::result::Result<T, Error>;
