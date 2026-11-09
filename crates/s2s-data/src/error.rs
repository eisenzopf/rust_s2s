use std::fmt;

/// Errors from the data pipeline.
///
/// Reading untrusted bytes (a shard, an index, a script) never panics: every
/// malformed input is reported as one of these.
#[derive(Debug)]
pub enum Error {
    /// An I/O error, with the path or operation it came from.
    Io {
        /// What was being done (usually a path).
        context: String,
        /// The underlying error.
        source: std::io::Error,
    },
    /// A shard or index file is malformed: bad magic, unsupported version,
    /// truncated, checksum mismatch, inconsistent lengths.
    Format(String),
    /// A sample violates an invariant (token out of range, unsorted
    /// injections, layout mismatch, ...).
    Invalid(String),
    /// A synthetic dialogue script is inconsistent.
    Script(String),
    /// JSON (de)serialization failed.
    Json(serde_json::Error),
}

impl Error {
    pub(crate) fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Error::Io {
            context: context.into(),
            source,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { context, source } => write!(f, "{context}: {source}"),
            Error::Format(m) => write!(f, "malformed shard data: {m}"),
            Error::Invalid(m) => write!(f, "invalid sample: {m}"),
            Error::Script(m) => write!(f, "invalid script: {m}"),
            Error::Json(e) => write!(f, "json: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            Error::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}

/// Result alias using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
