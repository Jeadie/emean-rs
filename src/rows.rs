//! Views of rows that passed the input checks: what the nearest-centroid
//! search and the batch statistics read.
//!
//! The search and the statistics trust their input: they assume finite
//! values, rows of the stated dimension and, for spherical clustering, unit
//! norms. A [`RowSource`] carries these guarantees. Its only implementations
//! are [`CheckedRows`], which can be made only by checking a matrix, and the
//! views built from it, so unchecked rows cannot reach the search.

use crate::Error;
use crate::matrix::{self, RowMatrix, RowNorm};
use crate::scalar::Scalar;

/// Rows that passed [`matrix::check_rows`]: every row has `dim()` finite
/// values and satisfies `norm()`.
pub(crate) trait RowSource<T>: Sync {
    /// The number of rows.
    fn len(&self) -> usize;
    /// The number of values in each row.
    fn dim(&self) -> usize;
    /// The norm requirement that every row passed.
    fn norm(&self) -> RowNorm;
    /// The row at `index < self.len()`, of exactly `self.dim()` values.
    fn row(&self, index: usize) -> &[T];
}

/// A matrix whose rows passed [`matrix::check_rows`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct CheckedRows<'a, T> {
    matrix: RowMatrix<'a, T>,
    norm: RowNorm,
}

impl<'a, T: Scalar> CheckedRows<'a, T> {
    /// Checks every row of `matrix` against `norm`.
    ///
    /// # Errors
    ///
    /// The error of the first row that fails (see [`matrix::check_rows`]).
    pub(crate) fn new(matrix: RowMatrix<'a, T>, norm: RowNorm) -> Result<Self, Error> {
        matrix::check_rows(matrix, norm)?;
        Ok(Self { matrix, norm })
    }

    /// The rows at `indices`, in that order.
    ///
    /// # Panics
    ///
    /// Panics when a row of the view is read if its index is out of range.
    pub(crate) fn gather(self, indices: &'a [usize]) -> Gathered<'a, T> {
        Gathered {
            rows: self,
            indices,
        }
    }

    /// The checked matrix.
    pub(crate) fn matrix(&self) -> RowMatrix<'a, T> {
        self.matrix
    }
}

impl<T: Scalar> RowSource<T> for CheckedRows<'_, T> {
    fn len(&self) -> usize {
        self.matrix.rows()
    }

    fn dim(&self) -> usize {
        self.matrix.dim()
    }

    fn norm(&self) -> RowNorm {
        self.norm
    }

    fn row(&self, index: usize) -> &[T] {
        self.matrix.row_at(index)
    }
}

/// The rows of a checked matrix at the given indices, in that order.
pub(crate) struct Gathered<'a, T> {
    rows: CheckedRows<'a, T>,
    indices: &'a [usize],
}

impl<T: Scalar> RowSource<T> for Gathered<'_, T> {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn dim(&self) -> usize {
        self.rows.dim()
    }

    fn norm(&self) -> RowNorm {
        self.rows.norm()
    }

    fn row(&self, index: usize) -> &[T] {
        self.rows.row(self.indices[index])
    }
}

/// Rows `start..start + len` of another source.
pub(crate) struct Window<'a, S> {
    source: &'a S,
    start: usize,
    len: usize,
}

impl<'a, S> Window<'a, S> {
    /// The rows `start..start + len` of `source`.
    ///
    /// # Panics
    ///
    /// Panics if the range is not inside `source`.
    pub(crate) fn new<T>(source: &'a S, start: usize, len: usize) -> Self
    where
        S: RowSource<T>,
    {
        assert!(
            start
                .checked_add(len)
                .is_some_and(|end| end <= source.len()),
            "window {start}..{start}+{len} is outside a source of {} rows",
            source.len()
        );
        Self { source, start, len }
    }
}

impl<T, S: RowSource<T>> RowSource<T> for Window<'_, S> {
    fn len(&self) -> usize {
        self.len
    }

    fn dim(&self) -> usize {
        self.source.dim()
    }

    fn norm(&self) -> RowNorm {
        self.source.norm()
    }

    fn row(&self, index: usize) -> &[T] {
        assert!(
            index < self.len,
            "row {index} of a window of {} rows",
            self.len
        );
        self.source.row(self.start + index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_keep_the_dimension_and_the_norm_requirement() {
        let data = [0.6f32, 0.8, 1.0, 0.0, 0.0, 1.0];
        let rows = CheckedRows::new(RowMatrix::new(&data, 2).unwrap(), RowNorm::Unit).unwrap();
        let indices = [2, 0];
        let gathered = rows.gather(&indices);
        let window = Window::new(&gathered, 1, 1);
        assert_eq!(
            (window.len(), window.dim(), window.norm()),
            (1, 2, RowNorm::Unit)
        );
        assert_eq!(window.row(0), &[0.6, 0.8]);
    }

    #[test]
    fn checking_rejects_bad_rows() {
        let data = [3.0f64, 4.0];
        let matrix = RowMatrix::new(&data, 2).unwrap();
        assert!(matches!(
            CheckedRows::new(matrix, RowNorm::Unit),
            Err(Error::NotUnitNorm { row: 0, .. })
        ));
        assert!(CheckedRows::new(matrix, RowNorm::Any).is_ok());
    }

    #[test]
    #[should_panic(expected = "outside a source")]
    fn windows_must_fit_their_source() {
        let data = [1.0f32; 4];
        let rows = CheckedRows::new(RowMatrix::new(&data, 2).unwrap(), RowNorm::Any).unwrap();
        let _ = Window::new(&rows, 1, 2);
    }
}
