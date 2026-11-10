//! Centroid positions: `k × d` row-major `f64` coordinates, the form in
//! which training keeps centroids and from which the search is built.
//!
//! There are two ways to turn rows of the element type into positions, and
//! they must not be confused:
//!
//! - [`from_vectors`] takes vectors as centroids, the way training does: for
//!   spherical clustering, it scales each to norm 1.
//! - [`exact`] keeps the values: it restores centroids that a model already
//!   produced. Scaling those again could move a coordinate by one unit in the
//!   last place.

use crate::config::Metric;
use crate::numeric;
use crate::scalar::{Element, Scalar};

/// Widens `rows` to `f64` without changing them.
pub(crate) fn exact<'a, T: Scalar>(rows: impl Iterator<Item = &'a [T]>) -> Vec<f64> {
    rows.flat_map(|row| row.iter().map(|value| value.to_f64()))
        .collect()
}

/// Widens `rows` (each of `dim` values) to `f64` and, for
/// [`Metric::Spherical`], scales each to norm 1 (a zero row stays zero).
pub(crate) fn from_vectors<'a, T: Scalar>(
    rows: impl Iterator<Item = &'a [T]>,
    dim: usize,
    metric: Metric,
) -> Vec<f64> {
    let mut positions = exact(rows);
    if metric == Metric::Spherical {
        normalize_each(&mut positions, dim);
    }
    positions
}

/// Scales each row of `dim` values to norm 1, in place. Zero rows stay zero.
pub(crate) fn normalize_each(positions: &mut [f64], dim: usize) {
    for row in positions.chunks_exact_mut(dim) {
        numeric::normalize_in_place(row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_spherical_vectors_are_scaled() {
        let rows = [[3.0f32, 4.0], [0.0, 0.0]];
        let rows = || rows.iter().map(|row| &row[..]);
        assert_eq!(exact(rows()), [3.0, 4.0, 0.0, 0.0]);
        assert_eq!(
            from_vectors(rows(), 2, Metric::SquaredEuclidean),
            [3.0, 4.0, 0.0, 0.0]
        );
        assert_eq!(
            from_vectors(rows(), 2, Metric::Spherical),
            [0.6, 0.8, 0.0, 0.0]
        );
    }
}
