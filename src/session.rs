//! Caller-driven training, for data that is streamed from storage.

use core::fmt;

use crate::Error;
use crate::config::{Emeans, Metric};
use crate::kernel::SearchCentroids;
use crate::matrix::{RowMatrix, RowNorm};
use crate::model::{Centroids, Model};
use crate::positions;
use crate::recipe::{REVIVAL_INTERVAL, Recipe};
use crate::rng::{Rng, Stream};
use crate::rows::{CheckedRows, RowSource, Window};
use crate::scalar::Scalar;
use crate::state::EmaState;
use crate::stats::{BatchStats, ClosingSums, EpochAccumulator, EpochStats};

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
    state: EmaState,
    search: SearchCentroids<T>,
    revival_rng: Rng,
    /// Unit-norm copies of the centroids, for [`Metric::Spherical`].
    unit_positions: Vec<f64>,
    phase: Phase,
    /// The rows of the current batch.
    batch: BatchStats,
    /// Rows fed since the start.
    fed: u64,
    /// Updates applied since the start.
    updates: u64,
    history: Vec<EpochStats>,
    revivals: u64,
    /// Search output buffers, at most one batch long.
    labels: Vec<u32>,
    distances: Vec<f64>,
}

/// Where a session is in its recipe.
#[derive(Debug)]
enum Phase {
    /// An epoch is open.
    Epoch(EpochAccumulator),
    /// Every epoch is complete; the last one left these sums for the
    /// closing means.
    Complete(ClosingSums),
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
        let dim = recipe.dim();
        let clusters = recipe.clusters().get() as usize;
        debug_assert_eq!(positions.len(), clusters * dim);
        let metric = recipe.metric();
        let unit_positions = match metric {
            Metric::SquaredEuclidean => Vec::new(),
            Metric::Spherical => positions.clone(),
        };
        Self {
            search: SearchCentroids::new(&positions, dim, metric),
            state: EmaState::new(positions, dim),
            revival_rng: Rng::new(recipe.seed(), Stream::Revival),
            unit_positions,
            phase: Phase::Epoch(open_epoch(&recipe, 0)),
            batch: BatchStats::new(clusters, dim),
            fed: 0,
            updates: 0,
            history: Vec::new(),
            revivals: 0,
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
            Phase::Epoch(epoch) => epoch.epoch(),
            Phase::Complete(_) => self.recipe.epoch_count(),
        }
    }

    /// The number of updates applied so far.
    pub fn updates(&self) -> u64 {
        self.updates
    }

    /// The number of rows that the recipe still expects.
    pub fn remaining_rows(&self) -> u64 {
        self.recipe.total_rows() - self.fed
    }

    /// The number of rows that complete the current batch, or 0 when
    /// training is complete.
    pub fn rows_to_next_update(&self) -> usize {
        match &self.phase {
            Phase::Epoch(epoch) => batch_len(&self.recipe, epoch, &self.batch) - self.batch.rows(),
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
        &self.history
    }

    /// The number of dead clusters revived so far.
    pub fn revivals(&self) -> u64 {
        self.revivals
    }

    /// The current centroids, `k × d`: the centroids that the next rows are
    /// assigned to. They are not the final centroids, which
    /// [`Session::finish`] computes.
    pub fn centroids(&self) -> RowMatrix<'_, T> {
        RowMatrix::new(self.search.centroids(), self.recipe.dim()).expect("k × d centroids")
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
        let Phase::Complete(closing) = &self.phase else {
            return Err(Unfinished {
                session: Box::new(self),
            });
        };
        let dim = self.recipe.dim();
        let metric = self.recipe.metric();
        let mut positions = self.state.positions().to_vec();
        closing.apply_means(&mut positions, dim);
        if metric == Metric::Spherical {
            positions::normalize_each(&mut positions, dim);
        }
        let centroids = Centroids::from_positions(&positions, dim, metric);
        Ok(Model::new(
            centroids,
            self.recipe,
            self.history,
            self.revivals,
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
            let take = (batch_len(&self.recipe, epoch, &self.batch) - self.batch.rows())
                .min(source.len() - offset);
            let piece = Window::new(source, offset, take);
            self.labels.resize(take, 0);
            self.distances.resize(take, 0.0);
            self.search
                .assign(&piece, &mut self.labels, &mut self.distances);
            self.batch.add(&piece, &self.labels);
            epoch.add_assigned(&self.distances);
            sink(offset, &self.labels);
            offset += take;
            self.fed += take as u64;
            if self.batch.rows() == batch_len(&self.recipe, epoch, &self.batch) {
                self.finish_batch();
            }
        }
    }

    /// Applies the update of the complete batch, revives dead clusters every
    /// [`REVIVAL_INTERVAL`] updates, and closes the epoch when it is complete.
    fn finish_batch(&mut self) {
        let Phase::Epoch(epoch) = &mut self.phase else {
            unreachable!("a batch belongs to an open epoch");
        };
        let alpha = self.recipe.alpha(self.updates);
        epoch.add_batch(&self.batch);
        self.state
            .apply(alpha, self.batch.sums(), self.batch.counts());
        self.updates += 1;
        let revived = match self.recipe.revival() {
            Some(threshold) if self.updates % REVIVAL_INTERVAL == 0 => {
                self.state
                    .revive::<T>(threshold.get(), self.recipe.metric(), &mut self.revival_rng)
                    as u64
            }
            _ => 0,
        };
        epoch.add_update(revived);
        self.revivals += revived;
        self.batch.clear();
        let epoch_complete = epoch.is_complete();
        self.refresh_search();
        if epoch_complete {
            self.finish_epoch();
        }
    }

    /// Records the statistics of the complete epoch and opens the next one,
    /// or completes the run after the last epoch.
    fn finish_epoch(&mut self) {
        // Take the open epoch out; the empty placeholder (no allocation) is
        // replaced below.
        let placeholder = Phase::Complete(ClosingSums::default());
        let Phase::Epoch(epoch) = core::mem::replace(&mut self.phase, placeholder) else {
            unreachable!("only an open epoch can finish");
        };
        let next = epoch.epoch() + 1;
        let (stats, closing) = epoch.finish();
        self.history.push(stats);
        self.phase = match closing {
            Some(closing) => Phase::Complete(closing),
            None => Phase::Epoch(open_epoch(&self.recipe, next)),
        };
    }

    /// Loads the current centroids into the search.
    fn refresh_search(&mut self) {
        match self.recipe.metric() {
            Metric::SquaredEuclidean => self.search.load(self.state.positions()),
            Metric::Spherical => {
                self.unit_positions.copy_from_slice(self.state.positions());
                positions::normalize_each(&mut self.unit_positions, self.recipe.dim());
                self.search.load(&self.unit_positions);
            }
        }
    }
}

impl<T: Scalar> fmt::Debug for Session<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session")
            .field("recipe", &self.recipe)
            .field("epoch", &self.epoch())
            .field("updates", &self.updates)
            .field("remaining_rows", &self.remaining_rows())
            .field("revivals", &self.revivals)
            .finish_non_exhaustive()
    }
}

/// The accumulator of epoch `epoch` of `recipe`.
fn open_epoch(recipe: &Recipe, epoch: u64) -> EpochAccumulator {
    let planned = recipe.rows_in_epoch(epoch).expect("an epoch of the recipe");
    let is_last = epoch + 1 == recipe.epoch_count();
    EpochAccumulator::new(
        epoch,
        planned,
        recipe.clusters().get() as usize,
        recipe.dim(),
        is_last,
    )
}

/// The number of rows of the current batch of `epoch`.
fn batch_len(recipe: &Recipe, epoch: &EpochAccumulator, batch: &BatchStats) -> usize {
    let batch_start = epoch.assigned() - batch.rows();
    recipe.batch_size().get().min(epoch.planned() - batch_start)
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
