use std::fmt;

/// A row-major tensor shape.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// Create a shape from its dimensions.
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self(dims.into())
    }

    /// The dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements (1 for a scalar shape).
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Size of dimension `i`; negative indices count from the end.
    ///
    /// # Panics
    /// If `i` is out of range.
    pub fn dim(&self, i: isize) -> usize {
        let r = self.rank() as isize;
        let idx = if i < 0 { r + i } else { i };
        assert!((0..r).contains(&idx), "dim {i} out of range for rank {r}");
        self.0[idx as usize]
    }

    /// Contiguous row-major strides, in elements.
    pub fn strides(&self) -> Vec<usize> {
        let mut s = vec![1; self.rank()];
        for i in (0..self.rank().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * self.0[i + 1];
        }
        s
    }

    /// Product of all dimensions except the last ("rows" of a 2-D view).
    pub fn rows(&self) -> usize {
        if self.0.is_empty() {
            1
        } else {
            self.0[..self.0.len() - 1].iter().product()
        }
    }
}

impl From<&[usize]> for Shape {
    fn from(d: &[usize]) -> Self {
        Self(d.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(d: [usize; N]) -> Self {
        Self(d.to_vec())
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        let s = Shape::from([2, 3, 4]);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.strides(), vec![12, 4, 1]);
        assert_eq!(s.dim(-1), 4);
        assert_eq!(s.rows(), 6);
        assert_eq!(s.to_string(), "[2, 3, 4]");
        assert_eq!(Shape::new(vec![]).numel(), 1);
    }
}
