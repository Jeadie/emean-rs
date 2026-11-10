//! The configuration of an e-means run.

use core::num::{NonZeroU32, NonZeroUsize};

use crate::recipe::Recipe;
use crate::scalar::Scalar;
use crate::{Error, Fit, RowMatrix, Session};

/// The seed of [`Emeans::new`], the same as the reference implementation's.
pub const DEFAULT_SEED: u64 = 1234;

/// An epoch budget: the number `T` of passes over the training data.
///
/// The budget can be fractional. With `T = 2.5`, a run makes one full pass,
/// then a pass over half of the rows, then a final full pass (see
/// [`Recipe::rows_in_epoch`]). Five epochs usually reach a lower loss than
/// 25 iterations of Lloyd's algorithm.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Epochs(f64);

impl Epochs {
    /// The default budget: 5 epochs.
    pub const DEFAULT: Self = Self(5.0);

    /// The largest budget that [`Epochs::new`] accepts: `2^32 - 1` epochs.
    pub const MAX: Self = Self(u32::MAX as f64);

    /// Creates a budget of `value` epochs.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidEpochs`] unless `0 < value <= Epochs::MAX`.
    pub fn new(value: f64) -> Result<Self, Error> {
        if value > 0.0 && value <= Self::MAX.0 {
            Ok(Self(value))
        } else {
            Err(Error::InvalidEpochs { value })
        }
    }

    /// Creates a budget of a whole number of epochs.
    pub const fn whole(epochs: NonZeroU32) -> Self {
        Self(epochs.get() as f64)
    }

    /// The budget, in epochs.
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl Default for Epochs {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl TryFrom<f64> for Epochs {
    type Error = Error;

    fn try_from(value: f64) -> Result<Self, Error> {
        Self::new(value)
    }
}

/// A step size `α` of the exponential moving average, in the interval (0, 1].
///
/// Each update moves a cluster's running count and sum a fraction `α` of the
/// way towards the batch's count and sum. With `α = 1`, an update replaces
/// them, as Lloyd's algorithm does.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Alpha(f64);

impl Alpha {
    /// Creates a step size.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidStep`] unless `0 < value <= 1`.
    pub fn new(value: f64) -> Result<Self, Error> {
        if value > 0.0 && value <= 1.0 {
            Ok(Self(value))
        } else {
            Err(Error::InvalidStep { value })
        }
    }

    /// The step size.
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for Alpha {
    type Error = Error;

    fn try_from(value: f64) -> Result<Self, Error> {
        Self::new(value)
    }
}

/// A dead-cluster revival threshold `τ`.
///
/// Every [`REVIVAL_INTERVAL`](crate::REVIVAL_INTERVAL) updates, clusters
/// whose running count is below `τ` times the mean running count are revived:
/// each takes half of a live cluster, drawn in proportion to its running
/// count. When there are fewer live clusters than dead ones, only that many
/// are revived.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Threshold(f64);

impl Threshold {
    /// The default threshold, `τ = 0.1`.
    pub const DEFAULT: Self = Self(0.1);

    /// Creates a threshold.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidThreshold`] unless `value` is positive and finite.
    /// Use [`Revival::Off`] to turn revival off.
    pub fn new(value: f64) -> Result<Self, Error> {
        if value > 0.0 && value.is_finite() {
            Ok(Self(value))
        } else {
            Err(Error::InvalidThreshold { value })
        }
    }

    /// The threshold.
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl Default for Threshold {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl TryFrom<f64> for Threshold {
    type Error = Error;

    fn try_from(value: f64) -> Result<Self, Error> {
        Self::new(value)
    }
}

/// A fitted scaling law that sets the batch size from the problem size.
///
/// See [`BatchLaw::fit`] for the formula and the coefficients.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BatchLaw {
    /// Larger batches, for higher throughput. Equation (1) of the reference
    /// implementation's documentation.
    Throughput,
    /// Batches about 8 times smaller than [`BatchLaw::Throughput`]. They give
    /// a lower loss, at a lower throughput. Equation (2).
    Quality,
}

/// How a run chooses its batch size `B`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Batch {
    /// [`BatchLaw::Quality`] for vectors of at most
    /// [`AUTO_QUALITY_MAX_DIM`](crate::AUTO_QUALITY_MAX_DIM) dimensions, and
    /// [`BatchLaw::Throughput`] for longer vectors. Small batches help more
    /// in low dimensions.
    #[default]
    Auto,
    /// The given scaling law.
    Law(BatchLaw),
    /// A fixed batch size. A size larger than the number of rows uses all
    /// rows. Unless the step is also fixed, the step law adapts the step
    /// size to this batch size.
    Fixed(NonZeroUsize),
}

/// How a run chooses its initial step size `α₀`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Step {
    /// The fitted step law, evaluated at the resolved batch size (see
    /// [`STEP_LAW`](crate::STEP_LAW)).
    #[default]
    Law,
    /// A fixed initial step size.
    Fixed(Alpha),
}

/// Whether and when a run revives dead clusters.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Revival {
    /// Off for vectors of at most
    /// [`AUTO_NO_REVIVAL_MAX_DIM`](crate::AUTO_NO_REVIVAL_MAX_DIM) dimensions,
    /// and [`Threshold::DEFAULT`] otherwise. The reference implementation
    /// measured that revival costs held-out error on 2-D data.
    #[default]
    Auto,
    /// Never revive clusters.
    Off,
    /// Revive clusters below the given threshold.
    Threshold(Threshold),
}

/// The geometry of the clustering.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Metric {
    /// Minimize the squared Euclidean distance `‖x − c‖²` between each vector
    /// and its centroid: the k-means problem.
    #[default]
    SquaredEuclidean,
    /// Spherical k-means: maximize the cosine similarity between each vector
    /// and its centroid.
    ///
    /// Rows must have a Euclidean norm of 1, within
    /// [`UNIT_NORM_TOLERANCE`](crate::UNIT_NORM_TOLERANCE) (see
    /// [`normalize_rows`](crate::normalize_rows)). Centroids are kept at
    /// norm 1, and each vector goes to the centroid with the largest inner
    /// product. Distances are `2 − 2⟨x, c⟩`, which is `‖x − c‖²` for unit
    /// vectors. For a row whose norm is slightly above 1, the distance to a
    /// centroid in the same direction can be slightly negative (down to about
    /// `−2 · UNIT_NORM_TOLERANCE`).
    Spherical,
}

/// The configuration of an e-means run: the entry point of this crate.
///
/// Only the number of clusters is required. Every other setting has a
/// default that the scaling laws of the paper make a good choice for any
/// problem size.
///
/// ```
/// use std::num::NonZeroU32;
/// use emeans::{Emeans, Epochs, RowMatrix};
///
/// // 1,000 points in 2 dimensions, around 4 corners.
/// let data: Vec<f32> = (0..1000)
///     .flat_map(|i| {
///         let corner = (i % 4) as f32;
///         let jitter = (i as f32 * 0.618).fract() * 0.1;
///         [(corner % 2.0) * 10.0 + jitter, (corner / 2.0).floor() * 10.0 - jitter]
///     })
///     .collect();
/// let data = RowMatrix::new(&data, 2)?;
///
/// let fit = Emeans::new(NonZeroU32::new(4).unwrap())
///     .with_epochs(Epochs::new(3.0)?)
///     .fit(data)?;
///
/// assert_eq!(fit.model.centroids().clusters().get(), 4);
/// let labels = fit.labels.expect("the last epoch visits every row");
/// assert_eq!(labels.len(), 1000);
/// # Ok::<(), emeans::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emeans {
    clusters: NonZeroU32,
    epochs: Epochs,
    batch: Batch,
    step: Step,
    revival: Revival,
    metric: Metric,
    seed: u64,
}

impl Emeans {
    /// A configuration for `clusters` clusters, with default settings:
    /// [`Epochs::DEFAULT`], [`Batch::Auto`], [`Step::Law`], [`Revival::Auto`],
    /// [`Metric::SquaredEuclidean`] and [`DEFAULT_SEED`].
    pub const fn new(clusters: NonZeroU32) -> Self {
        Self {
            clusters,
            epochs: Epochs::DEFAULT,
            batch: Batch::Auto,
            step: Step::Law,
            revival: Revival::Auto,
            metric: Metric::SquaredEuclidean,
            seed: DEFAULT_SEED,
        }
    }

    /// Sets the epoch budget.
    #[must_use]
    pub const fn with_epochs(mut self, epochs: Epochs) -> Self {
        self.epochs = epochs;
        self
    }

    /// Sets how the batch size is chosen.
    #[must_use]
    pub const fn with_batch(mut self, batch: Batch) -> Self {
        self.batch = batch;
        self
    }

    /// Sets how the initial step size is chosen.
    #[must_use]
    pub const fn with_step(mut self, step: Step) -> Self {
        self.step = step;
        self
    }

    /// Sets the dead-cluster revival policy.
    #[must_use]
    pub const fn with_revival(mut self, revival: Revival) -> Self {
        self.revival = revival;
        self
    }

    /// Sets the geometry of the clustering.
    #[must_use]
    pub const fn with_metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    /// Sets the seed of every random choice: the initial centroids, the row
    /// order of each epoch and the revivals.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The number of clusters `k`.
    pub const fn clusters(&self) -> NonZeroU32 {
        self.clusters
    }

    /// The epoch budget `T`.
    pub const fn epochs(&self) -> Epochs {
        self.epochs
    }

    /// How the batch size is chosen.
    pub const fn batch(&self) -> Batch {
        self.batch
    }

    /// How the initial step size is chosen.
    pub const fn step(&self) -> Step {
        self.step
    }

    /// The dead-cluster revival policy.
    pub const fn revival(&self) -> Revival {
        self.revival
    }

    /// The geometry of the clustering.
    pub const fn metric(&self) -> Metric {
        self.metric
    }

    /// The seed of every random choice.
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Resolves the hyperparameters for `rows` training vectors of dimension
    /// `dim`, without training.
    ///
    /// # Errors
    ///
    /// [`Error::ZeroDimension`] if `dim` is zero, [`Error::TooFewRows`] if
    /// `rows` is less than the number of clusters, [`Error::EmptyBudget`] if
    /// a budget below one epoch rounds to zero rows, and
    /// [`Error::BudgetTooLarge`] if the number of updates overflows a `u64`.
    pub fn recipe(&self, rows: usize, dim: usize) -> Result<Recipe, Error> {
        let dim = NonZeroUsize::new(dim).ok_or(Error::ZeroDimension)?;
        Recipe::resolve(self, rows, dim)
    }

    /// Trains on `data`, starting from centroids sampled uniformly without
    /// replacement from `data`.
    ///
    /// Returns the trained [`Model`](crate::Model) and the label of each row
    /// from the last epoch. See [`Fit`] for what the labels mean.
    ///
    /// # Errors
    ///
    /// The errors of [`Emeans::recipe`], and [`Error::NonFinite`],
    /// [`Error::MagnitudeTooLarge`] or (for [`Metric::Spherical`])
    /// [`Error::NotUnitNorm`] for the first row of `data` that fails these
    /// checks.
    pub fn fit<T: Scalar>(&self, data: RowMatrix<'_, T>) -> Result<Fit<T>, Error> {
        crate::fit::fit(self, data, None)
    }

    /// Trains on `data`, starting from the given initial centroids, one per
    /// row of `init`.
    ///
    /// For [`Metric::Spherical`], the initial centroids are scaled to norm 1.
    ///
    /// # Errors
    ///
    /// The errors of [`Emeans::fit`], and [`Error::DimensionMismatch`],
    /// [`Error::CentroidCountMismatch`], [`Error::NonFinite`] or
    /// [`Error::MagnitudeTooLarge`] for a bad `init`.
    pub fn fit_from<T: Scalar>(
        &self,
        data: RowMatrix<'_, T>,
        init: RowMatrix<'_, T>,
    ) -> Result<Fit<T>, Error> {
        crate::fit::fit(self, data, Some(init))
    }

    /// Starts a training [`Session`] that the caller feeds with rows, for
    /// data that does not fit in memory.
    ///
    /// `rows` is the number of training vectors `n`. Each epoch, the caller
    /// feeds [`Recipe::rows_in_epoch`] rows, each training vector at most
    /// once, in a fresh random order. See [`Session`] for an example.
    ///
    /// # Errors
    ///
    /// The errors of [`Emeans::recipe`], and those of [`Emeans::fit_from`]
    /// for a bad `init`.
    pub fn begin<T: Scalar>(
        &self,
        rows: usize,
        init: RowMatrix<'_, T>,
    ) -> Result<Session<T>, Error> {
        Session::begin(self, rows, init)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newtypes_reject_values_outside_their_domain() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, Epochs::MAX.get() * 2.0] {
            assert!(
                matches!(Epochs::new(bad), Err(Error::InvalidEpochs { .. })),
                "{bad}"
            );
        }
        assert_eq!(Epochs::new(Epochs::MAX.get()), Ok(Epochs::MAX));
        assert_eq!(Epochs::new(0.25).map(Epochs::get), Ok(0.25));
        assert!(Alpha::new(0.0).is_err());
        assert!(Alpha::new(1.0 + f64::EPSILON).is_err());
        assert!(Alpha::new(f64::NAN).is_err());
        assert_eq!(Alpha::new(1.0).map(Alpha::get), Ok(1.0));
        assert!(Threshold::new(0.0).is_err());
        assert!(Threshold::new(f64::INFINITY).is_err());
        assert_eq!(Threshold::new(2.0).map(Threshold::get), Ok(2.0));
    }

    #[test]
    fn builder_sets_each_field() {
        let k = NonZeroU32::new(7).unwrap();
        let config = Emeans::new(k)
            .with_epochs(Epochs::whole(NonZeroU32::new(2).unwrap()))
            .with_batch(Batch::Law(BatchLaw::Quality))
            .with_step(Step::Fixed(Alpha::new(0.5).unwrap()))
            .with_revival(Revival::Off)
            .with_metric(Metric::Spherical)
            .with_seed(9);
        assert_eq!(config.clusters(), k);
        assert_eq!(config.epochs().get(), 2.0);
        assert_eq!(config.batch(), Batch::Law(BatchLaw::Quality));
        assert_eq!(config.step(), Step::Fixed(Alpha::new(0.5).unwrap()));
        assert_eq!(config.revival(), Revival::Off);
        assert_eq!(config.metric(), Metric::Spherical);
        assert_eq!(config.seed(), 9);
        assert_eq!(Emeans::new(k).seed(), DEFAULT_SEED);
    }
}
