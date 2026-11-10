//! Statistics that a run records for each epoch, and the private
//! accumulators of a batch and an epoch.

use crate::kernel;
use crate::numeric::CompensatedSum;
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

/// The per-cluster sums `s_j` (`k × d`, row-major) and counts `n_j` of the
/// rows fed into the current batch (Algorithm 1, line 8).
#[derive(Clone, Debug)]
pub(crate) struct BatchStats {
    dim: usize,
    sums: Vec<f64>,
    counts: Vec<u64>,
    rows: usize,
}

impl BatchStats {
    /// An empty batch for `clusters` clusters of dimension `dim`.
    pub(crate) fn new(clusters: usize, dim: usize) -> Self {
        Self {
            dim,
            sums: vec![0.0; clusters * dim],
            counts: vec![0; clusters],
            rows: 0,
        }
    }

    /// Adds each row of `source` to the cluster in `labels`.
    pub(crate) fn add<T: Scalar, S: RowSource<T>>(&mut self, source: &S, labels: &[u32]) {
        kernel::accumulate(source, labels, self.dim, &mut self.sums, &mut self.counts);
        self.rows += source.len();
    }

    /// The number of rows in the batch.
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// The sums `s_j`, `k × d`, row-major.
    pub(crate) fn sums(&self) -> &[f64] {
        &self.sums
    }

    /// The counts `n_j`.
    pub(crate) fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// Empties the batch.
    pub(crate) fn clear(&mut self) {
        self.sums.fill(0.0);
        self.counts.fill(0);
        self.rows = 0;
    }
}

/// What the current epoch has measured so far: its rows, objective, cluster
/// sizes, updates and revivals, and, in the last epoch only, the per-cluster
/// sums for the closing means.
#[derive(Clone, Debug)]
pub(crate) struct EpochAccumulator {
    epoch: u64,
    /// Rows that the recipe plans for this epoch.
    planned: usize,
    /// Rows assigned so far.
    assigned: usize,
    objective: CompensatedSum,
    counts: Vec<u64>,
    updates: u64,
    revivals: u64,
    /// `Some` in the last epoch only.
    closing_sums: Option<Vec<f64>>,
}

impl EpochAccumulator {
    /// Starts epoch `epoch` of `planned` rows. The last epoch also keeps the
    /// per-cluster sums for the closing means.
    pub(crate) fn new(
        epoch: u64,
        planned: usize,
        clusters: usize,
        dim: usize,
        is_last: bool,
    ) -> Self {
        Self {
            epoch,
            planned,
            assigned: 0,
            objective: CompensatedSum::default(),
            counts: vec![0; clusters],
            updates: 0,
            revivals: 0,
            closing_sums: is_last.then(|| vec![0.0; clusters * dim]),
        }
    }

    /// The index of the epoch.
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The rows that the recipe plans for this epoch.
    pub(crate) fn planned(&self) -> usize {
        self.planned
    }

    /// The rows assigned so far.
    pub(crate) fn assigned(&self) -> usize {
        self.assigned
    }

    /// Whether every planned row has been assigned.
    pub(crate) fn is_complete(&self) -> bool {
        self.assigned == self.planned
    }

    /// Records the distances of rows that were just assigned, in row order.
    pub(crate) fn add_assigned(&mut self, distances: &[f64]) {
        debug_assert!(self.assigned + distances.len() <= self.planned);
        for &distance in distances {
            self.objective.add(distance);
        }
        self.assigned += distances.len();
    }

    /// Records a complete batch: its cluster sizes, and its sums in the last
    /// epoch.
    pub(crate) fn add_batch(&mut self, batch: &BatchStats) {
        for (total, &count) in self.counts.iter_mut().zip(batch.counts()) {
            *total += count;
        }
        if let Some(sums) = &mut self.closing_sums {
            for (total, &sum) in sums.iter_mut().zip(batch.sums()) {
                *total += sum;
            }
        }
    }

    /// Records one update and the clusters it revived.
    pub(crate) fn add_update(&mut self, revivals: u64) {
        self.updates += 1;
        self.revivals += revivals;
    }

    /// Ends the epoch. Returns its statistics, and the closing sums if it is
    /// the last epoch.
    pub(crate) fn finish(self) -> (EpochStats, Option<ClosingSums>) {
        debug_assert!(self.is_complete());
        let clusters = self.counts.len() as f64;
        let squares: u128 = self
            .counts
            .iter()
            .map(|&count| u128::from(count) * u128::from(count))
            .sum();
        let rows = self.planned as f64;
        let stats = EpochStats {
            epoch: self.epoch,
            rows: self.planned,
            updates: self.updates,
            objective: self.objective.total(),
            imbalance: clusters * squares as f64 / (rows * rows),
            revivals: self.revivals,
        };
        let closing = self.closing_sums.map(|sums| ClosingSums {
            counts: self.counts,
            sums,
        });
        (stats, closing)
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
    fn a_batch_sums_its_rows_and_clears() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rows = CheckedRows::new(RowMatrix::new(&data, 2).unwrap(), RowNorm::Any).unwrap();
        let mut batch = BatchStats::new(2, 2);
        batch.add(&rows, &[1, 1, 0]);
        assert_eq!(
            (batch.rows(), batch.counts(), batch.sums()),
            (3, &[1, 2][..], &[5.0, 6.0, 4.0, 6.0][..])
        );
        batch.clear();
        assert_eq!(
            (batch.rows(), batch.counts(), batch.sums()),
            (0, &[0, 0][..], &[0.0; 4][..])
        );
    }

    #[test]
    fn only_the_last_epoch_yields_closing_means() {
        let data = [1.0f64, 3.0, 5.0];
        let rows = CheckedRows::new(RowMatrix::new(&data, 1).unwrap(), RowNorm::Any).unwrap();
        let mut batch = BatchStats::new(2, 1);
        batch.add(&rows, &[0, 0, 0]);
        for is_last in [false, true] {
            let mut epoch = EpochAccumulator::new(0, 3, 2, 1, is_last);
            epoch.add_assigned(&[1.0, 2.0, 3.0]);
            epoch.add_batch(&batch);
            epoch.add_update(1);
            assert!(epoch.is_complete());
            let (stats, closing) = epoch.finish();
            assert_eq!(
                (stats.rows, stats.updates, stats.revivals, stats.objective),
                (3, 1, 1, 6.0)
            );
            // All rows in one of two clusters: the imbalance is k = 2.
            assert_eq!(stats.imbalance, 2.0);
            assert_eq!(closing.is_some(), is_last);
            if let Some(closing) = closing {
                let mut positions = [7.0, 8.0];
                closing.apply_means(&mut positions, 1);
                assert_eq!(positions, [3.0, 8.0]);
            }
        }
    }
}
