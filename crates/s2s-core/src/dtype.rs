use std::fmt;

/// Element types used anywhere in the project.
///
/// The set is closed on purpose: every kernel is written for a known list of
/// types, and adding one is a deliberate decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE 754 binary32. Master weights, optimizer state, reductions.
    F32,
    /// IEEE 754 binary16.
    F16,
    /// bfloat16. Default compute dtype for weights and activations.
    BF16,
    /// FP8 E4M3 (later optimization, see plan M10).
    F8E4M3,
    /// Signed 32-bit integer (positions, indices).
    I32,
    /// Unsigned 32-bit integer (text token ids).
    U32,
    /// Unsigned 16-bit integer (audio codebook ids; Mimi codebooks have 2048 entries).
    U16,
}

impl DType {
    /// Size of one element in bytes.
    pub const fn size_bytes(self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::BF16 | DType::U16 => 2,
            DType::F8E4M3 => 1,
        }
    }

    /// Whether this is a floating-point type.
    pub const fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::BF16 | DType::F8E4M3)
    }

    /// Name as used in safetensors headers.
    pub const fn safetensors_name(self) -> &'static str {
        match self {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::F8E4M3 => "F8_E4M3",
            DType::I32 => "I32",
            DType::U32 => "U32",
            DType::U16 => "U16",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.safetensors_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F8E4M3.size_bytes(), 1);
        assert!(DType::BF16.is_float());
        assert!(!DType::U16.is_float());
    }
}
