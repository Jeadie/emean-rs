//! The element types that e-means can cluster.

/// A floating-point element type for vectors and centroids: [`f32`] or [`f64`].
///
/// The nearest-centroid search runs in this type, because it dominates the
/// cost of training. Everything else (running statistics, centroid updates,
/// objectives and reported distances) is computed in [`f64`], whatever the
/// element type.
///
/// This trait is sealed: it cannot be implemented outside this crate, and it
/// has no methods that code outside this crate can call.
pub trait Scalar: sealed::Sealed {}

impl Scalar for f32 {}
impl Scalar for f64 {}

/// The operations that the crate needs from an element type, without the
/// [`sealed::Token`] argument that keeps them private.
pub(crate) trait Element: Scalar {
    /// The largest finite value, as an `f64`.
    fn max_f64() -> f64;
    /// The unit roundoff `u`: half the distance from 1 to the next value.
    /// Every basic operation of this type has a relative error of at most `u`.
    fn unit_roundoff() -> f64;
    /// The smallest positive value (a subnormal), as an `f64`.
    fn smallest_positive() -> f64;
    /// Converts to `f64` without loss.
    fn to_f64(self) -> f64;
    /// Rounds an `f64` to the nearest value of this type.
    fn from_f64(value: f64) -> Self;
    /// Whether the value is neither NaN nor infinite.
    fn is_finite(self) -> bool;
    /// Computes `c = a · bᵀ`, where `a` is `m × d`, `b` is `n × d` and `c` is
    /// `m × n`, all dense and row-major.
    ///
    /// # Panics
    ///
    /// Panics if a slice is shorter than its shape requires.
    fn gemm_abt(m: usize, n: usize, d: usize, a: &[Self], b: &[Self], c: &mut [Self]);
}

impl<T: Scalar> Element for T {
    #[inline]
    fn max_f64() -> f64 {
        <T as sealed::Sealed>::raw_max_f64(sealed::Token)
    }

    #[inline]
    fn unit_roundoff() -> f64 {
        <T as sealed::Sealed>::raw_unit_roundoff(sealed::Token)
    }

    #[inline]
    fn smallest_positive() -> f64 {
        <T as sealed::Sealed>::raw_smallest_positive(sealed::Token)
    }

    #[inline]
    fn to_f64(self) -> f64 {
        sealed::Sealed::raw_to_f64(self, sealed::Token)
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        sealed::Sealed::raw_from_f64(value, sealed::Token)
    }

    #[inline]
    fn is_finite(self) -> bool {
        sealed::Sealed::raw_is_finite(self, sealed::Token)
    }

    fn gemm_abt(m: usize, n: usize, d: usize, a: &[Self], b: &[Self], c: &mut [Self]) {
        sealed::Sealed::raw_gemm_abt(m, n, d, a, b, c, sealed::Token);
    }
}

pub(crate) mod sealed {
    use core::fmt::Debug;
    use core::ops::Sub;

    /// A value that only this crate can create. Every method of [`Sealed`]
    /// takes one, so code outside the crate cannot call them through a
    /// `Scalar` bound.
    #[derive(Clone, Copy, Debug)]
    pub struct Token;

    /// The operations behind [`super::Element`].
    pub trait Sealed:
        Copy + Send + Sync + PartialOrd + Debug + Sub<Output = Self> + 'static
    {
        fn raw_max_f64(token: Token) -> f64;
        fn raw_unit_roundoff(token: Token) -> f64;
        fn raw_smallest_positive(token: Token) -> f64;
        fn raw_to_f64(self, token: Token) -> f64;
        fn raw_from_f64(value: f64, token: Token) -> Self;
        fn raw_is_finite(self, token: Token) -> bool;
        fn raw_gemm_abt(
            m: usize,
            n: usize,
            d: usize,
            a: &[Self],
            b: &[Self],
            c: &mut [Self],
            token: Token,
        );
    }

    /// Checks the slice lengths and strides that `gemm_abt` gives to
    /// `matrixmultiply`, and returns the strides.
    fn gemm_strides<T>(m: usize, n: usize, d: usize, a: &[T], b: &[T], c: &[T]) -> (isize, isize) {
        let fits = |rows: usize, len: usize| rows.checked_mul(d).is_some_and(|need| need <= len);
        assert!(fits(m, a.len()), "gemm_abt: `a` is shorter than m × d");
        assert!(fits(n, b.len()), "gemm_abt: `b` is shorter than n × d");
        assert!(
            m.checked_mul(n).is_some_and(|need| need <= c.len()),
            "gemm_abt: `c` is shorter than m × n"
        );
        let d = isize::try_from(d).expect("gemm_abt: d exceeds isize::MAX");
        let n = isize::try_from(n).expect("gemm_abt: n exceeds isize::MAX");
        (d, n)
    }

    macro_rules! impl_sealed {
        ($t:ty, $gemm:path) => {
            impl Sealed for $t {
                #[inline]
                fn raw_max_f64(_: Token) -> f64 {
                    <$t>::MAX as f64
                }

                #[inline]
                fn raw_unit_roundoff(_: Token) -> f64 {
                    (<$t>::EPSILON / 2.0) as f64
                }

                #[inline]
                fn raw_smallest_positive(_: Token) -> f64 {
                    <$t>::from_bits(1) as f64
                }

                #[inline]
                fn raw_to_f64(self, _: Token) -> f64 {
                    self as f64
                }

                #[inline]
                fn raw_from_f64(value: f64, _: Token) -> Self {
                    value as $t
                }

                #[inline]
                fn raw_is_finite(self, _: Token) -> bool {
                    <$t>::is_finite(self)
                }

                fn raw_gemm_abt(
                    m: usize,
                    n: usize,
                    d: usize,
                    a: &[Self],
                    b: &[Self],
                    c: &mut [Self],
                    _: Token,
                ) {
                    let (d_stride, n_stride) = gemm_strides(m, n, d, a, b, c);
                    if m == 0 || n == 0 {
                        return;
                    }
                    // SAFETY: `gemm_strides` checked that `a` holds m × d
                    // elements, `b` holds n × d elements and `c` holds m × n
                    // elements. The strides address `a` and `c` as dense
                    // row-major matrices and `b` as the transpose of a dense
                    // row-major n × d matrix, so every access is in bounds.
                    // `c` is a unique borrow, so it does not alias `a` or `b`.
                    // With beta = 0, `c` is written without being read.
                    unsafe {
                        $gemm(
                            m,
                            d,
                            n,
                            1.0,
                            a.as_ptr(),
                            d_stride,
                            1,
                            b.as_ptr(),
                            1,
                            d_stride,
                            0.0,
                            c.as_mut_ptr(),
                            n_stride,
                            1,
                        );
                    }
                }
            }
        };
    }

    impl_sealed!(f32, matrixmultiply::sgemm);
    impl_sealed!(f64, matrixmultiply::dgemm);
}

#[cfg(test)]
mod tests {
    use super::Element;

    #[test]
    fn gemm_abt_matches_naive_product() {
        let (m, n, d) = (3, 4, 5);
        let a: Vec<f64> = (0..m * d).map(|i| i as f64 * 0.5 - 3.0).collect();
        let b: Vec<f64> = (0..n * d).map(|i| (i as f64).sin()).collect();
        let mut c = vec![f64::NAN; m * n];
        f64::gemm_abt(m, n, d, &a, &b, &mut c);
        for i in 0..m {
            for j in 0..n {
                let expected: f64 = (0..d).map(|t| a[i * d + t] * b[j * d + t]).sum();
                assert!((c[i * n + j] - expected).abs() < 1e-12, "({i}, {j})");
            }
        }
    }

    #[test]
    #[should_panic(expected = "`b` is shorter")]
    fn gemm_abt_rejects_short_input() {
        let mut c = [0.0f32; 4];
        f32::gemm_abt(2, 2, 2, &[0.0; 4], &[0.0; 3], &mut c);
    }

    #[test]
    fn unit_roundoff_is_half_an_ulp_of_one() {
        assert_eq!(f32::unit_roundoff(), 2f64.powi(-24));
        assert_eq!(f64::unit_roundoff(), 2f64.powi(-53));
        assert_eq!(f32::smallest_positive(), 2f64.powi(-149));
        assert_eq!(f64::smallest_positive(), f64::from_bits(1));
    }
}
