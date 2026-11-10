//! Caller-driven training, for data that is streamed from storage.

use core::{fmt, mem};

use crate::Error;
use crate::config::{Emeans, Metric};
use crate::epoch::{BatchEnd, Epoch, Phase};
use crate::matrix::{RowMatrix, RowNorm};
use crate::model::{Centroids, Model};
use crate::positions;
use crate::recipe::Recipe;
use crate::rng::{Rng, Stream};
use crate::rows::{CheckedRows, RowSource, Window};
use crate::running::RunningCentroids;
use crate::scalar::Scalar;
use crate::stats::{ClosingSums, EpochStats, History};

/// A training run that the caller feeds with rows.
///
/// [`Emeans::fit`] is a loop over a `Session`. Use a session directly when
/// the training data does not fit in memory: read the rows from storage in
/// any chunk size, and the session applies an update each time a batch of
/// [`Recipe::batch_size`] rows is complete.
///
/// The caller's duties are the ones that `fit` does internally:
///
/// - Feed [`Recipe::rows_in_epoch`] rows per epoch, in a fresh random
///   order each epoch, each training vector at most once per epoch. The
///   session counts the rows; it cannot check which rows they are.
/// - Give initial centroids, usually rows sampled uniformly from the data.
///
/// The labels from [`Session::feed_labeled`] in the last epoch are the
/// labels that [`Fit::labels`](crate::Fit::labels) holds.
///
/// ```
/// use std::num::NonZeroU32;
/// use emeans::{Emeans, Epochs, RowMatrix};
///
/// let dim = 2;
/// let data: Vec<f64> = (0..2000).map(|i| ((i * 7919) % 1000) as f64 / 100.0).collect();
/// let rows = data.len() / dim;
/// let config = Emeans::new(NonZeroU32::new(4).unwrap()).with_epochs(Epochs::new(2.0)?);
///
/// let init = RowMatrix::new(&data[..4 * dim], dim)?;
/// let mut session = config.begin(rows, init)?;
/// for epoch in 0..session.recipe().epoch_count() {
///     let epoch_rows = session.recipe().rows_in_epoch(epoch).unwrap();
///     // A real caller reads the rows in a fresh random order from storage.
///     let order: Vec<usize> = (0..rows).map(|i| (i * 37 + epoch as usize * 11) % rows).collect();
///     for chunk in order[..epoch_rows].chunks(100) {
///         let chunk: Vec<f64> = chunk.iter().flat_map(|&i| data[i * dim..(i + 1) * dim].iter().copied()).collect();
///         session.feed(RowMatrix::new(&chunk, dim)?)?;
///     }
/// }
/// let model = session.finish()?;
/// assert_eq!(model.centroids().clusters().get(), 4);
/// # Ok::<(), emeans::Error>(())
/// ```
pub struct Session<T: Scalar> {
    recipe: Recipe,
    /// The running centroids, and the search over them.
    centroids: RunningCentroids<T>,
    /// The open epoch, or the closing sums once every epoch is complete.
    phase: Phase,
    history: History,
    /// Search output buffers, at most one batch long.
    labels: Vec<u32>,
    distances: Vec<f64>,
}

impl<T: Scalar> Session<T> {
    pub(crate) fn begin(
        config: &Emeans,
        rows: usize,
        init: RowMatrix<'_, T>,
    ) -> Result<Self, Error> {
        let recipe = config.recipe(rows, init.dim())?;
        let positions = initial_positions(config, init)?;
        Ok(Self::start(recipe, positions))
    }

    /// Starts a run from checked initial centroids (`k × d`, `f64`, already
    /// of norm 1 for [`Metric::Spherical`]).
    pub(crate) fn start(recipe: Recipe, positions: Vec<f64>) -> Self {
        debug_assert_eq!(
            positions.len(),
            recipe.clusters().get() as usize * recipe.dim()
        );
        let first = recipe
            .planned_epoch(0)
            .expect("a plan has at least one epoch");
        Self {
            centroids: RunningCentroids::new(
                positions,
                recipe.dim(),
                recipe.metric(),
                Rng::new(recipe.seed(), Stream::Revival),
            ),
            phase: Phase::Epoch(Epoch::open(&recipe, first)),
            history: History::default(),
            labels: Vec::new(),
            distances: Vec::new(),
            recipe,
        }
    }

    /// The resolved plan of this run.
    pub fn recipe(&self) -> &Recipe {
        &self.recipe
    }

    /// The index of the current epoch (0-based). After the last epoch, the
    /// number of epochs.
    pub fn epoch(&self) -> u64 {
        match &self.phase {
            Phase::Epoch(epoch) => epoch.plan().index(),
            Phase::Complete(_) => self.recipe.epoch_count(),
        }
    }

    /// The number of updates applied so far.
    pub fn updates(&self) -> u64 {
        match &self.phase {
            Phase::Epoch(epoch) => epoch.updates_done(),
            Phase::Complete(_) => self.recipe.total_updates(),
        }
    }

    /// The number of rows that the recipe still expects.
    pub fn remaining_rows(&self) -> u64 {
        match &self.phase {
            Phase::Epoch(epoch) => self.recipe.total_rows() - epoch.rows_done(),
            Phase::Complete(_) => 0,
        }
    }

    /// The number of rows that complete the current batch, or 0 when
    /// training is complete.
    pub fn rows_to_next_update(&self) -> usize {
        match &self.phase {
            Phase::Epoch(epoch) => epoch.rows_to_next_update(),
            Phase::Complete(_) => 0,
        }
    }

    /// Whether every row of the recipe has been fed, so that
    /// [`Session::finish`] can run.
    pub fn is_complete(&self) -> bool {
        matches!(self.phase, Phase::Complete(_))
    }

    /// The statistics of each completed epoch.
    pub fn history(&self) -> &[EpochStats] {
        self.history.epochs()
    }

    /// The number of dead clusters revived so far.
    pub fn revivals(&self) -> u64 {
        let open = match &self.phase {
            Phase::Epoch(epoch) => epoch.revivals(),
            Phase::Complete(_) => 0,
        };
        self.history.revivals() + open
    }

    /// The current centroids, `k × d`: the centroids that the next rows are
    /// assigned to. They are not the final centroids, which
    /// [`Session::finish`] computes.
    pub fn centroids(&self) -> RowMatrix<'_, T> {
        let centroids = self.centroids.search().centroids();
        RowMatrix::new(centroids, self.recipe.dim()).expect("k × d centroids")
    }

    /// Feeds rows to the run.
    ///
    /// The rows can end in the middle of a batch or an epoch, and can span
    /// several batches and epochs.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`], [`Error::BudgetExceeded`] if the recipe
    /// expects fewer rows, and the value errors of
    /// [`Emeans::fit`](crate::Emeans::fit) (their row indices count from the
    /// start of `rows`). On error, no row is used.
    pub fn feed(&mut self, rows: RowMatrix<'_, T>) -> Result<(), Error> {
        let rows = self.check(rows)?;
        self.process(&rows, |_, _| {});
        Ok(())
    }

    /// Feeds rows to the run, like [`Session::feed`], and writes the cluster
    /// that each row is assigned to into `labels`.
    ///
    /// # Errors
    ///
    /// The errors of [`Session::feed`], and [`Error::BufferLength`] if
    /// `labels` does not have one element per row.
    pub fn feed_labeled(
        &mut self,
        rows: RowMatrix<'_, T>,
        labels: &mut [u32],
    ) -> Result<(), Error> {
        if labels.len() != rows.rows() {
            return Err(Error::BufferLength {
                expected: rows.rows(),
                actual: labels.len(),
            });
        }
        let rows = self.check(rows)?;
        self.process(&rows, |offset, segment| {
            labels[offset..offset + segment.len()].copy_from_slice(segment);
        });
        Ok(())
    }

    /// Ends the run and returns the trained model.
    ///
    /// The final centroid of each cluster is the mean of the rows that the
    /// last epoch assigned to it (Algorithm 1, line 12). A cluster that the
    /// last epoch assigned no row to keeps its current centroid.
    ///
    /// # Errors
    ///
    /// [`Unfinished`] if the recipe expects more rows. It gives the session
    /// back unchanged, and converts to [`Error::Incomplete`] with `?`.
    pub fn finish(self) -> Result<Model<T>, Unfinished<T>> {
        let Self {
            recipe,
            centroids,
            phase: Phase::Complete(closing),
            history,
            ..
        } = self
        else {
            return Err(Unfinished {
                session: Box::new(self),
            });
        };
        let dim = recipe.dim();
        let metric = recipe.metric();
        let mut positions = centroids.into_positions();
        closing.apply_means(&mut positions, dim);
        if metric == Metric::Spherical {
            positions::normalize_each(&mut positions, dim);
        }
        let centroids = Centroids::from_positions(&positions, dim, metric);
        let revivals = history.revivals();
        Ok(Model::new(
            centroids,
            recipe,
            history.into_epochs(),
            revivals,
        ))
    }

    /// Checks rows before they are fed.
    fn check<'a>(&self, rows: RowMatrix<'a, T>) -> Result<CheckedRows<'a, T>, Error> {
        if rows.dim() != self.recipe.dim() {
            return Err(Error::DimensionMismatch {
                expected: self.recipe.dim(),
                actual: rows.dim(),
            });
        }
        let remaining = self.remaining_rows();
        if rows.rows() as u64 > remaining {
            return Err(Error::BudgetExceeded {
                remaining,
                offered: rows.rows(),
            });
        }
        CheckedRows::new(rows, RowNorm::for_metric(self.recipe.metric()))
    }

    /// Feeds checked rows: assigns them, adds them to the batch and applies
    /// each update that completes. `sink(offset, labels)` receives the labels
    /// of the rows from `offset` on, one piece at a time.
    ///
    /// # Panics
    ///
    /// Panics if the rows have another dimension than the recipe, were not
    /// checked for the recipe's metric, or are more than the recipe expects.
    pub(crate) fn process<S: RowSource<T>>(
        &mut self,
        source: &S,
        mut sink: impl FnMut(usize, &[u32]),
    ) {
        assert_eq!(
            source.dim(),
            self.recipe.dim(),
            "rows of the wrong dimension"
        );
        assert!(
            source
                .norm()
                .satisfies(RowNorm::for_metric(self.recipe.metric())),
            "rows that were not checked for the metric"
        );
        assert!(
            source.len() as u64 <= self.remaining_rows(),
            "more rows than the recipe expects"
        );
        let mut offset = 0;
        while offset < source.len() {
            let Phase::Epoch(epoch) = &mut self.phase else {
                unreachable!("the row budget ends with the last epoch");
            };
            let take = epoch.rows_to_next_update().min(source.len() - offset);
            let piece = Window::new(source, offset, take);
            self.labels.resize(take, 0);
            self.distances.resize(take, 0.0);
            self.centroids
                .search()
                .assign(&piece, &mut self.labels, &mut self.distances);
            let full = epoch.add(&piece, &self.labels, &self.distances);
            sink(offset, &self.labels);
            offset += take;
            let Some(batch) = full else {
                continue;
            };
            let update = self.recipe.update(batch.update_index());
            let revived = self.centroids.apply(update, batch.sums(), batch.counts());
            if batch.close(revived) == BatchEnd::EpochComplete {
                self.finish_epoch();
            }
        }
    }

    /// Records the statistics of the complete epoch and opens the next one,
    /// or completes the run after the last epoch.
    fn finish_epoch(&mut self) {
        // Take the epoch out to finish it by value. The placeholder does not
        // allocate, and the next phase replaces it.
        let placeholder = Phase::Complete(ClosingSums::default());
        let Phase::Epoch(epoch) = mem::replace(&mut self.phase, placeholder) else {
            unreachable!("only an open epoch finishes");
        };
        let (stats, next) = epoch.finish(&self.recipe);
        self.history.push(stats);
        self.phase = next;
    }
}

impl<T: Scalar> fmt::Debug for Session<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session")
            .field("recipe", &self.recipe)
            .field("epoch", &self.epoch())
            .field("updates", &self.updates())
            .field("remaining_rows", &self.remaining_rows())
            .field("revivals", &self.revivals())
            .finish_non_exhaustive()
    }
}

/// The error of [`Session::finish`] when the recipe expects more rows. It
/// holds the session, unchanged.
///
/// ```
/// # use std::num::NonZeroU32;
/// # use emeans::{Emeans, RowMatrix};
/// # let data = [0.0f32, 1.0, 2.0, 3.0];
/// let mut session = Emeans::new(NonZeroU32::new(2).unwrap()).begin(4, RowMatrix::new(&data[..2], 1)?)?;
/// let unfinished = session.finish().unwrap_err();
/// assert_eq!(unfinished.remaining_rows(), 20); // 5 epochs of 4 rows
/// let session = unfinished.into_session();     // nothing is lost
/// # Ok::<(), emeans::Error>(())
/// ```
pub struct Unfinished<T: Scalar> {
    session: Box<Session<T>>,
}

impl<T: Scalar> Unfinished<T> {
    /// The number of rows that the recipe still expects.
    pub fn remaining_rows(&self) -> u64 {
        self.session.remaining_rows()
    }

    /// The session, to feed it the remaining rows.
    pub fn into_session(self) -> Session<T> {
        *self.session
    }
}

impl<T: Scalar> fmt::Debug for Unfinished<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Unfinished")
            .field("remaining_rows", &self.remaining_rows())
            .finish_non_exhaustive()
    }
}

impl<T: Scalar> fmt::Display for Unfinished<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(
            &Error::Incomplete {
                remaining: self.remaining_rows(),
            },
            f,
        )
    }
}

impl<T: Scalar> std::error::Error for Unfinished<T> {}

impl<T: Scalar> From<Unfinished<T>> for Error {
    fn from(unfinished: Unfinished<T>) -> Self {
        Self::Incomplete {
            remaining: unfinished.remaining_rows(),
        }
    }
}

/// Checks user-given initial centroids and converts them to `f64`, scaled
/// to norm 1 for [`Metric::Spherical`].
pub(crate) fn initial_positions<T: Scalar>(
    config: &Emeans,
    init: RowMatrix<'_, T>,
) -> Result<Vec<f64>, Error> {
    let clusters = config.clusters().get();
    if init.rows() != clusters as usize {
        return Err(Error::CentroidCountMismatch {
            expected: clusters,
            actual: init.rows(),
        });
    }
    let init = CheckedRows::new(init, RowNorm::Any)?;
    Ok(positions::from_vectors(
        init.matrix().iter(),
        init.matrix().dim(),
        config.metric(),
    ))
}
