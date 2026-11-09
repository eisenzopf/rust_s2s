use std::fmt;

use crate::Shape;

/// Errors shared across the project.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Two shapes that must agree do not.
    ShapeMismatch {
        /// What was being checked.
        context: &'static str,
        /// The expected shape.
        expected: Shape,
        /// The shape actually supplied.
        got: Shape,
    },
    /// A buffer has the wrong number of elements for its declared shape.
    LengthMismatch {
        /// What was being checked.
        context: &'static str,
        /// Expected element count.
        expected: usize,
        /// Actual element count.
        got: usize,
    },
    /// An argument is outside its valid range.
    InvalidArgument(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ShapeMismatch {
                context,
                expected,
                got,
            } => write!(f, "{context}: expected shape {expected}, got {got}"),
            Error::LengthMismatch {
                context,
                expected,
                got,
            } => write!(f, "{context}: expected {expected} elements, got {got}"),
            Error::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// Result alias using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
