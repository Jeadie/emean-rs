//! Borrowed row-major matrices, and the checks that training data must pass.

use core::fmt;
use core::num::NonZeroUsize;

use crate::Error;
use crate::config::Metric;
use crate::parallel;
use crate::scalar::{Element, Scalar};

/// The largest distance between 1 and the Euclidean norm of a row that
/// [`Metric::Spherical`] accepts.
///
/// The value accepts vectors that were normalized in `f32` and rounded
/// afterwards (for example, stored as 16-bit floats), and rejects vectors
/// that were not normalized.
pub const UNIT_NORM_TOLERANCE: f64 = 1e-3;

/// The largest squared norm of a row, as a fraction of the largest finite
/// value of the element type.
pub(crate) const MAX_SQUARED_NORM_FRACTION: f64 = 1.0 / 16.0;

/// A borrowed, dense, row-major matrix: one vector of `dim` elements per row.
///
/// `RowMatrix` checks only the shape. Training and assignment check the
/// values (see [`Emeans::fit`](crate::Emeans::fit)).
///
/// ```
/// use emeans::RowMatrix;
///
/// let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let matrix = RowMatrix::new(&data, 3)?;
/// assert_eq!((matrix.rows(), matrix.dim()), (2, 3));
/// assert_eq!(matrix.row(1), Some(&[4.0, 5.0, 6.0][..]));
/// # Ok::<(), emeans::Error>(())
/// ```
pub struct RowMatrix<'a, T> {
    data: &'a [T],
    dim: NonZeroUsize,
}

impl<'a, T> RowMatrix<'a, T> {
    /// Views `data` as rows of `dim` elements.
    ///
    /// An empty slice is a valid matrix with zero rows.
    ///
    /// # Errors
    ///
    /// [`Error::ZeroDimension`] if `dim` is zero, and [`Error::RaggedRows`]
    /// if `data.len()` is not a multiple of `dim`.
    pub fn new(data: &'a [T], dim: usize) -> Result<Self, Error> {
        let dim = NonZeroUsize::new(dim).ok_or(Error::ZeroDimension)?;
        if data.len() % dim != 0 {
            return Err(Error::RaggedRows {
                len: data.len(),
                dim: dim.get(),
            });
        }
        Ok(Self { data, dim })
    }

    /// The number of rows.
    pub fn rows(&self) -> usize {
        self.data.len() / self.dim
    }

    /// The number of elements in each row.
    pub fn dim(&self) -> usize {
        self.dim.get()
    }

    /// Whether the matrix has no rows.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The row at `index`, or `None` if `index >= self.rows()`.
    pub fn row(&self, index: usize) -> Option<&'a [T]> {
        let start = index.checked_mul(self.dim.get())?;
        self.data.get(start..start.checked_add(self.dim.get())?)
    }

    /// An iterator over the rows.
    pub fn iter(&self) -> core::slice::ChunksExact<'a, T> {
        self.data.chunks_exact(self.dim.get())
    }

    /// All elements, row after row.
    pub fn as_slice(&self) -> &'a [T] {
        self.data
    }

    /// The row at `index`. Panics if `index >= self.rows()`.
    pub(crate) fn row_at(&self, index: usize) -> &'a [T] {
        let dim = self.dim.get();
        &self.data[index * dim..(index + 1) * dim]
    }
}

impl<T> Clone for RowMatrix<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for RowMatrix<'_, T> {}

impl<T: fmt::Debug> fmt::Debug for RowMatrix<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RowMatrix")
            .field("rows", &self.rows())
            .field("dim", &self.dim)
            .field("data", &self.data)
            .finish()
    }
}

impl<T: PartialEq> PartialEq for RowMatrix<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim && self.data == other.data
    }
}

impl<'a, T> IntoIterator for RowMatrix<'a, T> {
    type Item = &'a [T];
    type IntoIter = core::slice::ChunksExact<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// What a row must satisfy, beyond being finite and not too long.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowNorm {
    /// Any norm.
    Any,
    /// A Euclidean norm of 1, within [`UNIT_NORM_TOLERANCE`].
    Unit,
    /// A Euclidean norm of 1, within [`UNIT_NORM_TOLERANCE`], or exactly 0:
    /// the centroids of spherical clustering.
    UnitOrZero,
}

impl RowNorm {
    /// The requirement on training and query rows under `metric`.
    pub(crate) fn for_metric(metric: Metric) -> Self {
        match metric {
            Metric::SquaredEuclidean => Self::Any,
            Metric::Spherical => Self::Unit,
        }
    }

    /// The requirement on stored centroids under `metric`.
    pub(crate) fn for_centroids(metric: Metric) -> Self {
        match metric {
            Metric::SquaredEuclidean => Self::Any,
            Metric::Spherical => Self::UnitOrZero,
        }
    }

    /// Whether every row that passes `self` also passes `required`.
    pub(crate) fn satisfies(self, required: Self) -> bool {
        match required {
            Self::Any => true,
            Self::Unit => self == Self::Unit,
            Self::UnitOrZero => matches!(self, Self::Unit | Self::UnitOrZero),
        }
    }
}

/// Checks that every value is finite, that no squared norm can overflow
/// `T`, and that rows have the required norm. Reports the first bad row.
pub(crate) fn check_rows<T: Scalar>(matrix: RowMatrix<'_, T>, norm: RowNorm) -> Result<(), Error> {
    parallel::try_for_each_row(matrix.as_slice(), matrix.dim(), |row, values| {
        check_row(row, values, norm)
    })
}

fn check_row<T: Scalar>(row: usize, values: &[T], norm: RowNorm) -> Result<(), Error> {
    let mut squared_norm = 0.0f64;
    for (column, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::NonFinite { row, column });
        }
        let value = value.to_f64();
        squared_norm += value * value;
    }
    // The limit keeps every distance, inner product and search score between
    // such rows, their means and their medians below MAX (see `kernel`). The
    // values are finite, so the sum is a number (possibly infinite).
    if squared_norm > MAX_SQUARED_NORM_FRACTION * T::max_f64() {
        return Err(Error::MagnitudeTooLarge { row });
    }
    let zero_allowed = norm == RowNorm::UnitOrZero && squared_norm == 0.0;
    if norm != RowNorm::Any && !zero_allowed {
        let length = squared_norm.sqrt();
        if (length - 1.0).abs() > UNIT_NORM_TOLERANCE {
            return Err(Error::NotUnitNorm { row, norm: length });
        }
    }
    Ok(())
}

/// Scales each row of `data` to a Euclidean norm of 1, in place.
///
/// Use it to prepare data for [`Metric::Spherical`]. The norm is computed in
/// `f64` after scaling the row by its largest absolute value, so the result
/// is accurate for rows of any finite magnitude. Rows that are all zero stay
/// zero (and spherical training rejects them).
///
/// ```
/// let mut data = [3.0f32, 4.0, 0.0, 0.0];
/// emeans::normalize_rows(&mut data, 2)?;
/// assert_eq!(data, [0.6, 0.8, 0.0, 0.0]);
/// # Ok::<(), emeans::Error>(())
/// ```
///
/// # Errors
///
/// [`Error::ZeroDimension`] and [`Error::RaggedRows`] as for
/// [`RowMatrix::new`], and [`Error::NonFinite`] if a value is NaN or
/// infinite. On error, no row is changed.
pub fn normalize_rows<T: Scalar>(data: &mut [T], dim: usize) -> Result<(), Error> {
    let matrix = RowMatrix::new(&*data, dim)?;
    parallel::try_for_each_row(matrix.as_slice(), dim, |row, values| {
        match values.iter().position(|value| !value.is_finite()) {
            Some(column) => Err(Error::NonFinite { row, column }),
            None => Ok(()),
        }
    })?;
    let mut scratch = vec![0.0f64; dim];
    for row in data.chunks_exact_mut(dim) {
        for (out, value) in scratch.iter_mut().zip(row.iter()) {
            *out = value.to_f64();
        }
        crate::numeric::normalize_in_place(&mut scratch);
        for (value, &unit) in row.iter_mut().zip(scratch.iter()) {
            *value = T::from_f64(unit);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_checks_the_shape() {
        assert_eq!(RowMatrix::<f32>::new(&[], 0), Err(Error::ZeroDimension));
        assert_eq!(
            RowMatrix::new(&[1.0f32; 5], 2),
            Err(Error::RaggedRows { len: 5, dim: 2 })
        );
        let empty = RowMatrix::<f64>::new(&[], 4).unwrap();
        assert_eq!((empty.rows(), empty.dim(), empty.is_empty()), (0, 4, true));
    }

    #[test]
    fn row_access_is_bounds_checked() {
        let data = [0.0f64, 1.0, 2.0, 3.0];
        let matrix = RowMatrix::new(&data, 2).unwrap();
        assert_eq!(matrix.row(1), Some(&[2.0, 3.0][..]));
        assert_eq!(matrix.row(2), None);
        assert_eq!(matrix.row(usize::MAX), None);
        assert_eq!(matrix.iter().count(), 2);
    }

    #[test]
    fn check_rows_reports_the_first_bad_value() {
        let data = [0.0f32, 1.0, 2.0, f32::NAN, f32::INFINITY, 0.0];
        let matrix = RowMatrix::new(&data, 2).unwrap();
        assert_eq!(
            check_rows(matrix, RowNorm::Any),
            Err(Error::NonFinite { row: 1, column: 1 })
        );
    }

    #[test]
    fn check_rows_rejects_rows_that_can_overflow() {
        let big = (f32::MAX / 8.0).sqrt();
        let data = [big, 0.0, 1.0, 1.0];
        let matrix = RowMatrix::new(&data, 2).unwrap();
        assert_eq!(
            check_rows(matrix, RowNorm::Any),
            Err(Error::MagnitudeTooLarge { row: 0 })
        );
        let ok = [(f32::MAX / 16.0).sqrt() * 0.99, 0.0];
        assert_eq!(
            check_rows(RowMatrix::new(&ok, 2).unwrap(), RowNorm::Any),
            Ok(())
        );
    }

    #[test]
    fn check_rows_enforces_unit_norm_for_spherical() {
        let data = [0.6f32, 0.8, 0.6, 0.81];
        let matrix = RowMatrix::new(&data, 2).unwrap();
        match check_rows(matrix, RowNorm::Unit) {
            Err(Error::NotUnitNorm { row: 1, norm }) => assert!((norm - 1.00802).abs() < 1e-4),
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(check_rows(matrix, RowNorm::Any), Ok(()));
    }

    #[test]
    fn centroid_requirement_also_accepts_zero_rows() {
        let data = [0.0f32, 0.0, 0.6, 0.8];
        let matrix = RowMatrix::new(&data, 2).unwrap();
        assert!(matches!(
            check_rows(matrix, RowNorm::Unit),
            Err(Error::NotUnitNorm { row: 0, .. })
        ));
        assert_eq!(check_rows(matrix, RowNorm::UnitOrZero), Ok(()));
        assert!(RowNorm::Unit.satisfies(RowNorm::UnitOrZero));
        assert!(!RowNorm::UnitOrZero.satisfies(RowNorm::Unit));
        assert!(!RowNorm::Any.satisfies(RowNorm::Unit));
    }

    #[test]
    fn normalize_rows_handles_extreme_magnitudes() {
        let mut data = [f64::MAX, f64::MAX, 1e-300, 0.0, 0.0, 0.0];
        normalize_rows(&mut data, 2).unwrap();
        let unit = 1.0 / 2f64.sqrt();
        assert_eq!(data, [unit, unit, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn normalize_rows_changes_nothing_on_error() {
        let mut data = [3.0f32, 4.0, 1.0, f32::NAN];
        assert_eq!(
            normalize_rows(&mut data, 2),
            Err(Error::NonFinite { row: 1, column: 1 })
        );
        assert_eq!(data[..2], [3.0, 4.0]);
    }
}
