//! The progress of a training run through its plan: the open epoch, its
//! current batch, and what the epoch has measured so far.
//!
//! ```text
//!                 add rows
//!              ┌───────────┐
//!              ▼           │ (batch not full)
//!  open ──► batch open ────┘
//!              │ add rows (batch full)
//!              ▼
//!           FullBatch ── the caller applies the update ── close
//!              │                                            │
//!              │                    rows left in the epoch ─┤─ none left
//!              ▼                                            ▼
//!           batch open (the next batch)        BatchEnd::EpochComplete
//!                                                           │ finish
//!                                       ┌───────────────────┴─────────────┐
//!                                       ▼                                 ▼
//!                       Phase::Epoch (the next epoch)   Phase::Complete (closing sums)
//! ```
//!
//! The run's totals (rows, updates) come from the epoch's place in the plan
//! and its own counts, so no other counter has to agree with them.

use core::num::NonZeroUsize;

use crate::numeric::CompensatedSum;
use crate::recipe::{PlannedEpoch, Recipe};
use crate::rows::RowSource;
use crate::scalar::Scalar;
use crate::stats::{BatchStats, ClosingSums, EpochStats};

/// Where a run is in its plan.
#[derive(Debug)]
pub(crate) enum Phase {
    /// An epoch is open.
    Epoch(Epoch),
    /// Every epoch is complete; the last one left these sums for the
    /// closing means.
    Complete(ClosingSums),
}

/// An open epoch.
#[derive(Clone, Debug)]
pub(crate) struct Epoch {
    plan: PlannedEpoch,
    batch_size: NonZeroUsize,
    batch: BatchStats,
    /// Rows assigned so far: those of the closed batches and the open one.
    assigned: usize,
    objective: CompensatedSum,
    /// Rows per cluster, over the closed batches.
    counts: Vec<u64>,
    /// Updates so far: one per closed batch.
    updates: u64,
    revivals: u64,
    role: Role,
}

/// What an epoch collects besides its statistics, by its place in the plan.
#[derive(Clone, Debug)]
enum Role {
    /// Another epoch follows.
    Intermediate,
    /// The last epoch also sums its rows per cluster (`k × d`, row-major)
    /// for the closing means.
    Last { sums: Vec<f64> },
}

/// What [`FullBatch::close`] leaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub(crate) enum BatchEnd {
    /// The next batch of the epoch is open.
    EpochOpen,
    /// The epoch has all its rows; [`Epoch::finish`] ends it.
    EpochComplete,
}

impl Epoch {
    /// Opens epoch `plan` of `recipe`.
    pub(crate) fn open(recipe: &Recipe, plan: PlannedEpoch) -> Self {
        let clusters = recipe.clusters().get() as usize;
        let batch_size = recipe.batch_size();
        let batch = BatchStats::new(clusters, recipe.dim(), batch_size.min(plan.rows()));
        Self::start(plan, batch_size, batch, vec![0; clusters], recipe.dim())
    }

    /// Opens epoch `plan` in the buffers of a finished epoch.
    fn start(
        plan: PlannedEpoch,
        batch_size: NonZeroUsize,
        mut batch: BatchStats,
        mut counts: Vec<u64>,
        dim: usize,
    ) -> Self {
        batch.reopen(batch_size.min(plan.rows()));
        counts.fill(0);
        let role = if plan.is_last() {
            Role::Last {
                sums: vec![0.0; counts.len() * dim],
            }
        } else {
            Role::Intermediate
        };
        Self {
            plan,
            batch_size,
            batch,
            assigned: 0,
            objective: CompensatedSum::default(),
            counts,
            updates: 0,
            revivals: 0,
            role,
        }
    }

    /// The epoch's entry in the plan.
    pub(crate) fn plan(&self) -> PlannedEpoch {
        self.plan
    }

    /// The rows of the run so far.
    pub(crate) fn rows_done(&self) -> u64 {
        self.plan.first_row() + self.assigned as u64
    }

    /// The updates of the run so far, which is also the index of the next
    /// update.
    pub(crate) fn updates_done(&self) -> u64 {
        self.plan.first_update() + self.updates
    }

    /// The clusters revived in this epoch so far.
    pub(crate) fn revivals(&self) -> u64 {
        self.revivals
    }

    /// The rows that complete the open batch.
    pub(crate) fn rows_to_next_update(&self) -> usize {
        self.batch.remaining()
    }

    /// Records rows that the search assigned: `labels` and `distances` are
    /// those of the rows of `source`. Returns the batch if these rows
    /// complete it.
    ///
    /// # Panics
    ///
    /// Panics if `source` has more rows than
    /// [`Epoch::rows_to_next_update`], if the lengths differ, or if a full
    /// batch was not closed.
    #[must_use = "a full batch must be closed after its update"]
    pub(crate) fn add<T: Scalar, S: RowSource<T>>(
        &mut self,
        source: &S,
        labels: &[u32],
        distances: &[f64],
    ) -> Option<FullBatch<'_>> {
        assert!(
            !self.batch.is_full(),
            "a full batch must be closed before the epoch takes more rows"
        );
        assert_eq!(distances.len(), source.len());
        self.batch.add(source, labels);
        for &distance in distances {
            self.objective.add(distance);
        }
        self.assigned += source.len();
        if self.batch.is_full() {
            Some(FullBatch { epoch: self })
        } else {
            None
        }
    }

    /// Ends the epoch. Returns its statistics, and the next phase of the
    /// run: the next epoch of `recipe`, or the completion of the run after
    /// the last epoch.
    ///
    /// # Panics
    ///
    /// Panics if the epoch does not have all its rows.
    pub(crate) fn finish(self, recipe: &Recipe) -> (EpochStats, Phase) {
        let rows = self.plan.rows().get();
        assert_eq!(
            self.assigned, rows,
            "an epoch ends when it has all its rows"
        );
        debug_assert_eq!(
            Some(self.updates),
            recipe.updates_in_epoch(self.plan.index())
        );
        let clusters = self.counts.len() as f64;
        let squares: u128 = self
            .counts
            .iter()
            .map(|&count| u128::from(count) * u128::from(count))
            .sum();
        let stats = EpochStats {
            epoch: self.plan.index(),
            rows,
            updates: self.updates,
            objective: self.objective.total(),
            imbalance: clusters * squares as f64 / (rows as f64 * rows as f64),
            revivals: self.revivals,
        };
        let next = match self.role {
            Role::Intermediate => {
                let plan = recipe
                    .planned_epoch(self.plan.index() + 1)
                    .expect("an epoch before the last one has a next epoch");
                let dim = recipe.dim();
                Phase::Epoch(Self::start(
                    plan,
                    self.batch_size,
                    self.batch,
                    self.counts,
                    dim,
                ))
            }
            Role::Last { sums } => Phase::Complete(ClosingSums::new(self.counts, sums)),
        };
        (stats, next)
    }
}

/// The full batch of an epoch, from the time it is full to its
/// [`FullBatch::close`]. While it exists, the epoch takes no rows.
#[must_use = "a full batch must be closed after its update"]
pub(crate) struct FullBatch<'a> {
    epoch: &'a mut Epoch,
}

impl FullBatch<'_> {
    /// The index of this batch's update in the run, from 0.
    pub(crate) fn update_index(&self) -> u64 {
        self.epoch.updates_done()
    }

    /// The batch's sums `s_j`, `k × d`, row-major.
    pub(crate) fn sums(&self) -> &[f64] {
        self.epoch.batch.sums()
    }

    /// The batch's counts `n_j`.
    pub(crate) fn counts(&self) -> &[u64] {
        self.epoch.batch.counts()
    }

    /// Records the batch and its update, which revived `revivals` clusters,
    /// and opens the next batch if the epoch has more rows.
    pub(crate) fn close(self, revivals: u64) -> BatchEnd {
        let epoch = self.epoch;
        for (total, &count) in epoch.counts.iter_mut().zip(epoch.batch.counts()) {
            *total += count;
        }
        if let Role::Last { sums } = &mut epoch.role {
            for (total, &sum) in sums.iter_mut().zip(epoch.batch.sums()) {
                *total += sum;
            }
        }
        epoch.updates += 1;
        epoch.revivals += revivals;
        match NonZeroUsize::new(epoch.plan.rows().get() - epoch.assigned) {
            Some(left) => {
                epoch.batch.reopen(left.min(epoch.batch_size));
                BatchEnd::EpochOpen
            }
            None => BatchEnd::EpochComplete,
        }
    }
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU32;

    use super::*;
    use crate::matrix::{RowMatrix, RowNorm};
    use crate::rows::{CheckedRows, Window};
    use crate::{Batch, Emeans, Epochs};

    /// A recipe of `epochs` epochs over 5 rows of dimension 1, in batches of
    /// 2 rows, into 2 clusters.
    fn recipe(epochs: f64) -> Recipe {
        Emeans::new(NonZeroU32::new(2).unwrap())
            .with_epochs(Epochs::new(epochs).unwrap())
            .with_batch(Batch::Fixed(NonZeroUsize::new(2).unwrap()))
            .recipe(5, 1)
            .unwrap()
    }

    /// Feeds the 5 rows to `epoch` one at a time, all into cluster 0, with
    /// distance 1. Returns the batch sizes and how each batch ended.
    fn feed(epoch: &mut Epoch, rows: &CheckedRows<'_, f64>) -> Vec<(usize, BatchEnd)> {
        let mut ends = Vec::new();
        let mut in_batch = 0;
        for index in 0..5 {
            let row = Window::new(rows, index, 1);
            in_batch += 1;
            if let Some(batch) = epoch.add(&row, &[0], &[1.0]) {
                assert_eq!(batch.counts(), [in_batch as u64, 0]);
                ends.push((in_batch, batch.close(1)));
                in_batch = 0;
            }
        }
        ends
    }

    #[test]
    fn an_epoch_cuts_its_rows_into_batches_and_moves_to_the_next() {
        let recipe = recipe(2.0);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let rows = CheckedRows::new(RowMatrix::new(&data, 1).unwrap(), RowNorm::Any).unwrap();
        let mut epoch = Epoch::open(&recipe, recipe.planned_epoch(0).unwrap());
        assert_eq!(epoch.rows_to_next_update(), 2);
        let ends = feed(&mut epoch, &rows);
        use BatchEnd::{EpochComplete, EpochOpen};
        assert_eq!(ends, [(2, EpochOpen), (2, EpochOpen), (1, EpochComplete)]);
        assert_eq!((epoch.rows_done(), epoch.updates_done()), (5, 3));

        let (stats, next) = epoch.finish(&recipe);
        assert_eq!(
            (stats.rows, stats.updates, stats.revivals, stats.objective),
            (5, 3, 3, 5.0)
        );
        // All rows in one of two clusters: the imbalance is k = 2.
        assert_eq!(stats.imbalance, 2.0);
        let Phase::Epoch(mut epoch) = next else {
            panic!("a second epoch follows the first");
        };
        assert_eq!((epoch.rows_done(), epoch.updates_done()), (5, 3));
        assert_eq!(epoch.rows_to_next_update(), 2);
        let _ = feed(&mut epoch, &rows);
        let (_, next) = epoch.finish(&recipe);
        let Phase::Complete(closing) = next else {
            panic!("the run is complete after the last epoch");
        };
        let mut positions = [7.0, 8.0];
        closing.apply_means(&mut positions, 1);
        assert_eq!(positions, [3.0, 8.0]);
    }

    #[test]
    #[should_panic(expected = "must be closed")]
    fn a_full_batch_is_not_handed_out_twice() {
        let recipe = recipe(1.0);
        let data = [1.0, 2.0, 3.0];
        let rows = CheckedRows::new(RowMatrix::new(&data, 1).unwrap(), RowNorm::Any).unwrap();
        let mut epoch = Epoch::open(&recipe, recipe.planned_epoch(0).unwrap());
        // The full batch goes out of scope without a close.
        assert!(
            epoch
                .add(&Window::new(&rows, 0, 2), &[0, 0], &[1.0, 1.0])
                .is_some()
        );
        // Without the check, the same batch would be applied twice.
        let _again = epoch.add(&Window::new(&rows, 2, 0), &[], &[]);
    }

    #[test]
    #[should_panic(expected = "all its rows")]
    fn an_epoch_does_not_end_early() {
        let recipe = recipe(1.0);
        let _ = Epoch::open(&recipe, recipe.planned_epoch(0).unwrap()).finish(&recipe);
    }
}
