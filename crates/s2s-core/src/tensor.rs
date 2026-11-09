use crate::{Error, Result, Shape};

/// An owned, contiguous, row-major tensor in host memory.
///
/// This is the currency of the CPU reference backend and of test fixtures.
/// Device tensors live in the backend crates.
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor<T> {
    data: Vec<T>,
    shape: Shape,
}

impl<T: Clone + Default> HostTensor<T> {
    /// A tensor filled with `T::default()` (zero for numbers).
    pub fn zeros(shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        Self {
            data: vec![T::default(); shape.numel()],
            shape,
        }
    }
}

impl<T> HostTensor<T> {
    /// Wrap a buffer. Fails if its length does not match the shape.
    pub fn from_vec(data: Vec<T>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        if data.len() != shape.numel() {
            return Err(Error::LengthMismatch {
                context: "HostTensor::from_vec",
                expected: shape.numel(),
                got: data.len(),
            });
        }
        Ok(Self { data, shape })
    }

    /// The shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The elements, row-major.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Mutable elements, row-major.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consume and return the buffer.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Reinterpret with a new shape of the same element count.
    pub fn reshape(self, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        if shape.numel() != self.data.len() {
            return Err(Error::ShapeMismatch {
                context: "HostTensor::reshape",
                expected: self.shape,
                got: shape,
            });
        }
        Ok(Self {
            data: self.data,
            shape,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_and_reshape() {
        let t = HostTensor::from_vec((0..6).map(|x| x as f32).collect(), [2, 3]).unwrap();
        assert_eq!(t.shape().dims(), &[2, 3]);
        let t = t.reshape([3, 2]).unwrap();
        assert_eq!(t.data()[5], 5.0);
        assert!(HostTensor::from_vec(vec![0.0f32; 5], [2, 3]).is_err());
        assert!(t.reshape([4, 2]).is_err());
        let z: HostTensor<f32> = HostTensor::zeros([2, 2]);
        assert_eq!(z.data(), &[0.0; 4]);
    }
}
