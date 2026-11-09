use std::fmt::Debug;
use std::iter::Sum;

/// Float trait the CPU reference kernels are generic over.
///
/// Implemented for `f32` (normal tests, golden comparisons) and `f64`
/// (finite-difference gradient checks, where `f32` rounding noise would swamp
/// the signal).
pub trait Float: num_traits::Float + Sum + Debug + Default + Send + Sync + 'static {}

impl Float for f32 {}
impl Float for f64 {}

/// Convert an `f64` constant to `T`.
///
/// # Panics
/// Never for `f32`/`f64`; the conversion is total for both.
#[inline]
pub fn cast<T: Float>(x: f64) -> T {
    T::from(x).expect("f64 -> Float conversion is total for f32/f64")
}
