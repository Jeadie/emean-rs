//! Trained centroids and what training reports about them.

use core::num::{NonZeroU32, NonZeroUsize};

use crate::Error;
use crate::config::Metric;
use crate::kernel::SearchCentroids;
use crate::matrix::{RowMatrix, RowNorm};
use crate::positions;
use crate::recipe::Recipe;
use crate::rows::CheckedRows;
use crate::scalar::Scalar;
use crate::stats::EpochStats;

/// A set of `k` centroids that can assign vectors to their nearest centroid.
///
/// Training returns them inside a [`Model`]. There are two ways to make them
/// from vectors, and they differ for [`Metric::Spherical`]:
///
/// - [`Centroids::restore`] restores centroids that a model produced (for
///   example, stored from [`Centroids::as_slice`]). It checks the values and
///   keeps them exactly, so a stored model assigns exactly as before.
/// - [`Centroids::from_vectors`] takes any vectors as centroids, the way
///   training takes its initial centroids: it scales spherical ones to norm 1.
#[derive(Clone, Debug)]
pub struct Centroids<T: Scalar> {
    search: SearchCentroids<T>,
    clusters: NonZeroU32,
    dim: NonZeroUsize,
    metric: Metric,
}

impl<T: Scalar> Centroids<T> {
    /// Restores centroids that a model produced, one per row, without
    /// changing any value.
    ///
    /// ```
    /// # use std::num::NonZeroU32;
    /// # use emeans::{Centroids, Emeans, RowMatrix};
    /// # let values: Vec<f32> = (0..400).map(|i| (i % 37) as f32).collect();
    /// # let data = RowMatrix::new(&values, 4)?;
    /// let model = Emeans::new(NonZeroU32::new(8).unwrap()).fit(data)?.model;
    /// let stored: Vec<f32> = model.centroids().as_slice().to_vec();
    ///
    /// let restored = Centroids::restore(RowMatrix::new(&stored, 4)?, model.centroids().metric())?;
    /// assert_eq!(restored.as_slice(), stored.as_slice());
    /// assert_eq!(restored.assign(data)?, model.assign(data)?);
    /// # Ok::<(), emeans::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::NoCentroids`] or [`Error::TooManyCentroids`] unless
    /// `1 <= k <= u32::MAX`; [`Error::NonFinite`] or
    /// [`Error::MagnitudeTooLarge`] for a bad row; and, for
    /// [`Metric::Spherical`], [`Error::NotUnitNorm`] for a row that has
    /// neither norm 1 (within [`UNIT_NORM_TOLERANCE`](crate::UNIT_NORM_TOLERANCE))
    /// nor norm 0, which no trained spherical centroid has.
    pub fn restore(matrix: RowMatrix<'_, T>, metric: Metric) -> Result<Self, Error> {
        let rows = Self::check(matrix, RowNorm::for_centroids(metric))?;
        let positions = positions::exact(rows.matrix().iter());
        Ok(Self::from_positions(&positions, matrix.dim(), metric))
    }

    /// Takes vectors as centroids, one per row. For [`Metric::Spherical`],
    /// each row is scaled to norm 1 (a zero row stays zero), as training
    /// does with its initial centroids.
    ///
    /// # Errors
    ///
    /// [`Error::NoCentroids`] or [`Error::TooManyCentroids`] unless
    /// `1 <= k <= u32::MAX`, and [`Error::NonFinite`] or
    /// [`Error::MagnitudeTooLarge`] for a bad row.
    pub fn from_vectors(matrix: RowMatrix<'_, T>, metric: Metric) -> Result<Self, Error> {
        let rows = Self::check(matrix, RowNorm::Any)?;
        let positions = positions::from_vectors(rows.matrix().iter(), matrix.dim(), metric);
        Ok(Self::from_positions(&positions, matrix.dim(), metric))
    }

    /// Checks the number of centroids and their values.
    fn check(matrix: RowMatrix<'_, T>, norm: RowNorm) -> Result<CheckedRows<'_, T>, Error> {
        if matrix.is_empty() {
            return Err(Error::NoCentroids);
        }
        if u32::try_from(matrix.rows()).is_err() {
            return Err(Error::TooManyCentroids {
                rows: matrix.rows(),
            });
        }
        CheckedRows::new(matrix, norm)
    }

    /// Creates centroids from checked `f64` positions, rounded to `T`.
    pub(crate) fn from_positions(positions: &[f64], dim: usize, metric: Metric) -> Self {
        let rows = positions.len() / dim;
        Self {
            search: SearchCentroids::new(positions, dim, metric),
            clusters: u32::try_from(rows)
                .ok()
                .and_then(NonZeroU32::new)
                .expect("1 <= k <= u32::MAX"),
            dim: NonZeroUsize::new(dim).expect("dim >= 1"),
            metric,
        }
    }

    /// The number of centroids `k`.
    pub fn clusters(&self) -> NonZeroU32 {
        self.clusters
    }

    /// The dimension `d` of the centroids.
    pub fn dim(&self) -> usize {
        self.dim.get()
    }

    /// The geometry of the clustering.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// The centroids, one per row.
    pub fn as_matrix(&self) -> RowMatrix<'_, T> {
        RowMatrix::new(self.search.centroids(), self.dim.get()).expect("k × d centroids")
    }

    /// The centroids, `k × d`, row-major.
    pub fn as_slice(&self) -> &[T] {
        self.search.centroids()
    }

    /// The centroid of cluster `cluster`, or `None` if `cluster >= k`.
    pub fn centroid(&self, cluster: u32) -> Option<&[T]> {
        self.as_matrix().row(cluster as usize)
    }

    /// The centroids, `k × d`, row-major.
    pub fn into_vec(self) -> Vec<T> {
        self.search.into_centroids()
    }

    /// Finds the nearest centroid of each row of `data`.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`], and the value errors of
    /// [`Emeans::fit`](crate::Emeans::fit).
    pub fn assign(&self, data: RowMatrix<'_, T>) -> Result<Assignment, Error> {
        let mut labels = vec![0; data.rows()];
        let mut distances = vec![0.0; data.rows()];
        self.assign_into(data, &mut labels, &mut distances)?;
        Ok(Assignment { labels, distances })
    }

    /// Finds the nearest centroid of each row of `data`, into buffers that
    /// the caller owns.
    ///
    /// # Errors
    ///
    /// The errors of [`Centroids::assign`], and [`Error::BufferLength`] if a
    /// buffer does not have one element per row.
    pub fn assign_into(
        &self,
        data: RowMatrix<'_, T>,
        labels: &mut [u32],
        distances: &mut [f64],
    ) -> Result<(), Error> {
        if data.dim() != self.dim() {
            return Err(Error::DimensionMismatch {
                expected: self.dim(),
                actual: data.dim(),
            });
        }
        for len in [labels.len(), distances.len()] {
            if len != data.rows() {
                return Err(Error::BufferLength {
                    expected: data.rows(),
                    actual: len,
                });
            }
        }
        let rows = CheckedRows::new(data, RowNorm::for_metric(self.metric))?;
        self.search.assign(&rows, labels, distances);
        Ok(())
    }
}

/// The nearest centroid of each row, and the distance to it.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Assignment {
    /// The index of the nearest centroid of each row. Ties go to the lower
    /// index.
    pub labels: Vec<u32>,
    /// The distance from each row to its centroid, computed in `f64`: the
    /// squared Euclidean distance, or `2 − 2⟨x, c⟩` for
    /// [`Metric::Spherical`].
    pub distances: Vec<f64>,
}

impl Assignment {
    /// The mean distance over the rows that were assigned, summed with
    /// compensation. For [`Metric::SquaredEuclidean`] this is the k-means
    /// objective, the mean squared error. NaN when there are no rows.
    pub fn mean_distance(&self) -> f64 {
        let mut sum = crate::numeric::CompensatedSum::default();
        for &distance in &self.distances {
            sum.add(distance);
        }
        sum.total() / self.distances.len() as f64
    }
}

/// A trained e-means model: the centroids, and how training went.
#[derive(Clone, Debug)]
pub struct Model<T: Scalar> {
    centroids: Centroids<T>,
    recipe: Recipe,
    history: Vec<EpochStats>,
    revivals: u64,
}

impl<T: Scalar> Model<T> {
    pub(crate) fn new(
        centroids: Centroids<T>,
        recipe: Recipe,
        history: Vec<EpochStats>,
        revivals: u64,
    ) -> Self {
        Self {
            centroids,
            recipe,
            history,
            revivals,
        }
    }

    /// The trained centroids.
    pub fn centroids(&self) -> &Centroids<T> {
        &self.centroids
    }

    /// The trained centroids, without the training report.
    pub fn into_centroids(self) -> Centroids<T> {
        self.centroids
    }

    /// The resolved hyperparameters of the run.
    pub fn recipe(&self) -> &Recipe {
        &self.recipe
    }

    /// The statistics of each epoch, in order.
    pub fn history(&self) -> &[EpochStats] {
        &self.history
    }

    /// The number of dead clusters that the run revived.
    pub fn revivals(&self) -> u64 {
        self.revivals
    }

    /// Finds the nearest centroid of each row of `data`. See
    /// [`Centroids::assign`].
    ///
    /// # Errors
    ///
    /// The errors of [`Centroids::assign`].
    pub fn assign(&self, data: RowMatrix<'_, T>) -> Result<Assignment, Error> {
        self.centroids.assign(data)
    }
}

/// The result of [`Emeans::fit`](crate::Emeans::fit).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Fit<T: Scalar> {
    /// The trained model.
    pub model: Model<T>,
    /// The cluster that the last epoch assigned each training row to, or
    /// `None` when the last epoch does not visit every row (see
    /// [`Recipe::last_epoch_is_full`](crate::Recipe::last_epoch_is_full)).
    ///
    /// The labels cost no extra search (only 4 bytes per row), but they come
    /// from centroids that were still moving during the last epoch. So a few
    /// labels can differ from the nearest final centroid; [`Model::assign`]
    /// gives the nearest final centroids for the cost of one more pass.
    /// Each final centroid is the mean of the rows with its label (scaled to
    /// norm 1 for [`Metric::Spherical`](crate::Metric::Spherical)), or its
    /// last moving position if no row has its label.
    pub labels: Option<Vec<u32>>,
}
