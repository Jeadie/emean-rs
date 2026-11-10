//! Statistics that a run records for each epoch, the sums of a batch, and
//! the sums of the closing means.

use core::num::NonZeroUsize;

use crate::kernel;
use crate::rows::RowSource;
use crate::scalar::Scalar;

/// What one epoch of training measured.
///
/// The objective is measured while the centroids move: each row's distance
/// is to the centroids of the moment it was assigned, not to the final
/// centroids. Use [`Model::assign`](crate::Model::assign) to measure the
/// final centroids.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct EpochStats {
    /// The index of the epoch, from 0.
    pub epoch: u64,
    /// The number of rows of the epoch.
    pub rows: usize,
    /// The number of updates of the epoch.
    pub updates: u64,
    /// The sum over the epoch's rows of the distance to the centroid each
    /// row was assigned to (see [`Metric`](crate::Metric)).
    pub objective: f64,
    /// How uneven the cluster sizes of the epoch are:
    /// `k · Σ_j size_j² / rows²`. It is 1 when every cluster has the same
    /// size, and `k` when one cluster has every row.
    pub imbalance: f64,
    /// The number of dead clusters revived during the epoch.
    pub revivals: u64,
}

impl EpochStats {
    /// The mean distance per row: [`EpochStats::objective`] divided by
    /// [`EpochStats::rows`].
    pub fn mean_objective(&self) -> f64 {
        self.objective / self.rows as f64
    }
}

/// The statistics of a run's completed epochs, in order, and their total
/// revivals.
#[derive(Clone, Debug, Default)]
pub(crate) struct History {
    epochs: Vec<EpochStats>,
    revivals: u64,
}

impl History {
    /// Adds the next completed epoch.
    pub(crate) fn push(&mut self, epoch: EpochStats) {
        self.revivals += epoch.revivals;
        self.epochs.push(epoch);
    }

    /// The completed epochs, in order.
    pub(crate) fn epochs(&self) -> &[EpochStats] {
        &self.epochs
    }

    /// The clusters revived in the completed epochs.
    pub(crate) fn revivals(&self) -> u64 {
        self.revivals
    }

    /// The completed epochs, in order.
    pub(crate) fn into_epochs(self) -> Vec<EpochStats> {
        self.epochs
    }
}

/// The per-cluster sums `s_j` (`k × d`, row-major) and counts `n_j` of the
/// rows fed into the current batch (Algorithm 1, line 8).
///
/// A batch is open for `capacity` rows: the recipe's batch size, or fewer
/// for the last batch of an epoch. It is full when it has them all.
#[derive(Clone, Debug)]
pub(crate) struct BatchStats {
    dim: usize,
    sums: Vec<f64>,
    counts: Vec<u64>,
    rows: usize,
    capacity: NonZeroUsize,
}

impl BatchStats {
    /// An empty batch of `capacity` rows, for `clusters` clusters of
    /// dimension `dim`.
    pub(crate) fn new(clusters: usize, dim: usize, capacity: NonZeroUsize) -> Self {
        Self {
            dim,
            sums: vec![0.0; clusters * dim],
            counts: vec![0; clusters],
            rows: 0,
            capacity,
        }
    }

    /// Adds each row of `source` to the cluster in `labels`.
    ///
    /// # Panics
    ///
    /// Panics if `source` has more rows than [`BatchStats::remaining`].
    pub(crate) fn add<T: Scalar, S: RowSource<T>>(&mut self, source: &S, labels: &[u32]) {
        assert!(
            source.len() <= self.remaining(),
            "{} rows for a batch with room for {}",
            source.len(),
            self.remaining()
        );
        kernel::accumulate(source, labels, self.dim, &mut self.sums, &mut self.counts);
        self.rows += source.len();
    }

    /// The rows that complete the batch.
    pub(crate) fn remaining(&self) -> usize {
        self.capacity.get() - self.rows
    }

    /// Whether the batch has all its rows.
    pub(crate) fn is_full(&self) -> bool {
        self.rows == self.capacity.get()
    }

    /// The sums `s_j`, `k × d`, row-major.
    pub(crate) fn sums(&self) -> &[f64] {
        &self.sums
    }

    /// The counts `n_j`.
    pub(crate) fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// Empties the batch and opens it for `capacity` rows.
    pub(crate) fn reopen(&mut self, capacity: NonZeroUsize) {
        self.sums.fill(0.0);
        self.counts.fill(0);
        self.rows = 0;
        self.capacity = capacity;
    }
}

/// The per-cluster counts and sums of the last epoch, from which the final
/// centroids come (Algorithm 1, line 12).
#[derive(Clone, Debug, Default)]
pub(crate) struct ClosingSums {
    counts: Vec<u64>,
    sums: Vec<f64>,
}

impl ClosingSums {
    /// The closing sums of an epoch: its rows per cluster and their sums
    /// (`k × d`, row-major).
    pub(crate) fn new(counts: Vec<u64>, sums: Vec<f64>) -> Self {
        debug_assert_eq!(sums.len() % counts.len().max(1), 0);
        Self { counts, sums }
    }

    /// Replaces the position of each cluster that the last epoch assigned
    /// rows to with the mean of those rows. Other clusters keep `positions`.
    pub(crate) fn apply_means(&self, positions: &mut [f64], dim: usize) {
        let rows = positions
            .chunks_exact_mut(dim)
            .zip(self.sums.chunks_exact(dim));
        for ((position, sums), &count) in rows.zip(&self.counts) {
            if count > 0 {
                for (c, &s) in position.iter_mut().zip(sums) {
                    *c = s / count as f64;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::{RowMatrix, RowNorm};
    use crate::rows::CheckedRows;

    #[test]
    fn a_batch_fills_to_its_capacity_and_reopens_empty() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rows = CheckedRows::new(RowMatrix::new(&data, 2).unwrap(), RowNorm::Any).unwrap();
        let mut batch = BatchStats::new(2, 2, NonZeroUsize::new(5).unwrap());
        batch.add(&rows, &[1, 1, 0]);
        assert_eq!((batch.remaining(), batch.is_full()), (2, false));
        assert_eq!(
            (batch.counts(), batch.sums()),
            (&[1, 2][..], &[5.0, 6.0, 4.0, 6.0][..])
        );
        batch.reopen(NonZeroUsize::new(3).unwrap());
        assert_eq!(
            (batch.remaining(), batch.counts(), batch.sums()),
            (3, &[0, 0][..], &[0.0; 4][..])
        );
        batch.add(&rows, &[0, 0, 0]);
        assert!(batch.is_full());
    }

    #[test]
    #[should_panic(expected = "room for 2")]
    fn a_batch_takes_no_more_than_its_capacity() {
        let data = [1.0f64; 3];
        let rows = CheckedRows::new(RowMatrix::new(&data, 1).unwrap(), RowNorm::Any).unwrap();
        let mut batch = BatchStats::new(1, 1, NonZeroUsize::new(2).unwrap());
        batch.add(&rows, &[0, 0, 0]);
    }

    #[test]
    fn the_history_keeps_the_total_revivals() {
        let epoch = |epoch, revivals| EpochStats {
            epoch,
            rows: 10,
            updates: 2,
            objective: 1.0,
            imbalance: 1.0,
            revivals,
        };
        let mut history = History::default();
        history.push(epoch(0, 3));
        history.push(epoch(1, 4));
        assert_eq!(history.revivals(), 7);
        assert_eq!(history.epochs(), [epoch(0, 3), epoch(1, 4)]);
    }

    #[test]
    fn closing_means_replace_only_clusters_with_rows() {
        let closing = ClosingSums::new(vec![2, 0], vec![6.0, 9.0]);
        let mut positions = [7.0, 8.0];
        closing.apply_means(&mut positions, 1);
        assert_eq!(positions, [3.0, 8.0]);
    }
}
