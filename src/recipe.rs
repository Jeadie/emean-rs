//! The resolved hyperparameters of a run: batch size, step schedule and
//! epoch plan, with the fitted scaling laws that choose them.
//!
//! The constants follow the reference implementation in faiss
//! (`contrib/e_means/base.py`, commit `fc1095a9`). Its comments note that
//! the shipped scaling-law coefficients differ slightly from Table 2 of the
//! paper: they come from a later fit.

use core::fmt;
use core::num::{NonZeroU32, NonZeroUsize};

use crate::Error;
use crate::config::{Batch, BatchLaw, Emeans, Epochs, Metric, Revival, Step, Threshold};

/// The floor of the cosine step schedule, as a fraction of `α₀`: `ε` in
/// Equation 5 of the paper.
pub const ALPHA_MIN_FACTOR: f64 = 0.1;

/// The smallest initial step size `α₀` that the step law returns.
pub const MIN_ALPHA0: f64 = 1e-4;

/// The number of updates between two revival checks.
pub const REVIVAL_INTERVAL: u64 = 5;

/// The running count `N_j` of every cluster before the first update.
///
/// A small value makes the first batch that reaches a cluster almost
/// replace its initial position. The paper uses `1e-10`; the reference
/// implementation, and this crate, use `2^-16`.
pub const INITIAL_MASS: f64 = 1.0 / 65536.0;

/// The relative perturbation `δ` that splits a donor centroid in two
/// (Equation 8 of the paper).
pub const SPLIT_EPSILON: f64 = 1e-4;

/// [`Batch::Auto`] uses [`BatchLaw::Quality`] up to this dimension, and
/// [`BatchLaw::Throughput`] above it.
pub const AUTO_QUALITY_MAX_DIM: usize = 16;

/// [`Revival::Auto`] turns revival off up to this dimension.
pub const AUTO_NO_REVIVAL_MAX_DIM: usize = 2;

/// The coefficients of a batch-size law
/// `B = coefficient · k^clusters_exponent · n^rows_exponent · T^epochs_exponent`,
/// for `k` clusters, `n` rows and `T` epochs.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchLawFit {
    /// The constant factor `c_B`.
    pub coefficient: f64,
    /// The exponent `b_k` of the number of clusters.
    pub clusters_exponent: f64,
    /// The exponent `b_n` of the number of rows.
    pub rows_exponent: f64,
    /// The exponent `b_T` of the epoch budget.
    pub epochs_exponent: f64,
}

impl BatchLawFit {
    /// The unrounded batch size for `clusters` clusters, `rows` rows and
    /// `epochs` epochs.
    pub fn evaluate(&self, clusters: f64, rows: f64, epochs: f64) -> f64 {
        self.coefficient
            * clusters.powf(self.clusters_exponent)
            * rows.powf(self.rows_exponent)
            * epochs.powf(self.epochs_exponent)
    }
}

/// The coefficients of the step law
/// `α₀ = coefficient · k^clusters_exponent · B^batch_exponent · n^rows_exponent · T^epochs_exponent`,
/// for `k` clusters, batch size `B`, `n` rows and `T` epochs.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepLawFit {
    /// The constant factor `c_α`.
    pub coefficient: f64,
    /// The exponent `a_k` of the number of clusters.
    pub clusters_exponent: f64,
    /// The exponent `a_B` of the batch size.
    pub batch_exponent: f64,
    /// The exponent `a_n` of the number of rows.
    pub rows_exponent: f64,
    /// The exponent `a_T` of the epoch budget.
    pub epochs_exponent: f64,
}

impl StepLawFit {
    /// The unclamped initial step size. A [`Recipe`] clamps it to
    /// `[MIN_ALPHA0, 1]`.
    pub fn evaluate(&self, clusters: f64, batch: f64, rows: f64, epochs: f64) -> f64 {
        self.coefficient
            * clusters.powf(self.clusters_exponent)
            * batch.powf(self.batch_exponent)
            * rows.powf(self.rows_exponent)
            * epochs.powf(self.epochs_exponent)
    }
}

/// The throughput batch law: `B = 0.035134 · k^0.134749 · n^0.876473 · T^0.813344`.
pub const THROUGHPUT_LAW: BatchLawFit = BatchLawFit {
    coefficient: 0.035134,
    clusters_exponent: 0.134749,
    rows_exponent: 0.876473,
    epochs_exponent: 0.813344,
};

/// The quality batch law: the throughput law with a coefficient 8 times
/// smaller, `B = 0.00439175 · k^0.134749 · n^0.876473 · T^0.813344`.
pub const QUALITY_LAW: BatchLawFit = BatchLawFit {
    coefficient: 0.00439175,
    ..THROUGHPUT_LAW
};

/// The step law:
/// `α₀ = 0.8244 · k^-0.4617 · B^0.9003 · n^-0.4498 · T^-0.1057`.
pub const STEP_LAW: StepLawFit = StepLawFit {
    coefficient: 0.8244,
    clusters_exponent: -0.4617,
    batch_exponent: 0.9003,
    rows_exponent: -0.4498,
    epochs_exponent: -0.1057,
};

impl BatchLaw {
    /// The fitted coefficients of this law: [`THROUGHPUT_LAW`] or
    /// [`QUALITY_LAW`].
    pub const fn fit(self) -> BatchLawFit {
        match self {
            Self::Throughput => THROUGHPUT_LAW,
            Self::Quality => QUALITY_LAW,
        }
    }
}

/// Where a recipe's batch size comes from.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BatchSource {
    /// The law that [`Batch::Law`] named.
    Law(BatchLaw),
    /// The law that [`Batch::Auto`] chose from the dimension.
    AutoLaw(BatchLaw),
    /// [`Batch::Fixed`]. The batch size is the requested size, capped at the
    /// number of rows.
    Fixed {
        /// The requested batch size.
        requested: NonZeroUsize,
    },
}

/// Where a recipe's initial step size comes from.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StepSource {
    /// The step law, at the resolved batch size.
    Law,
    /// [`Step::Fixed`].
    Fixed,
}

/// The resolved plan of a run.
///
/// A recipe fixes everything that the configuration leaves to the scaling
/// laws: the batch size `B`, the initial step size `α₀`, the rows of each
/// epoch and the step size of each update. Get one from
/// [`Emeans::recipe`] to inspect a run before training, or from a trained
/// [`Model`](crate::Model).
///
/// # Epoch plan
///
/// A budget of `T` epochs over `n` rows makes `⌊T⌋` full passes and one
/// partial pass of `round(frac(T) · n)` rows (rounded half to even). The
/// partial pass comes just before the last full pass, because the last pass
/// must visit every row to compute the closing means. Below one epoch, the
/// only pass is the partial one.
///
/// ```text
/// T = 2.5, n = 1000:   [ 1000 rows ][ 500 rows ][ 1000 rows ]
/// T = 0.6, n = 1000:   [ 600 rows ]
/// ```
///
/// Each epoch is cut into batches of `B` rows; the last batch of an epoch
/// keeps the remaining rows, so it can be smaller.
#[derive(Clone, Debug, PartialEq)]
pub struct Recipe {
    rows: usize,
    dim: NonZeroUsize,
    clusters: NonZeroU32,
    epochs: Epochs,
    metric: Metric,
    batch_size: NonZeroUsize,
    batch_source: BatchSource,
    alpha0: f64,
    step_source: StepSource,
    revival: Option<Threshold>,
    /// Number of full passes, `⌊T⌋`.
    full_passes: u64,
    /// Rows of the partial pass, or zero if there is none.
    partial_rows: usize,
    total_rows: u64,
    total_updates: u64,
    seed: u64,
}

impl Recipe {
    pub(crate) fn resolve(config: &Emeans, rows: usize, dim: NonZeroUsize) -> Result<Self, Error> {
        let clusters = config.clusters();
        if rows < clusters.get() as usize {
            return Err(Error::TooFewRows {
                rows,
                clusters: clusters.get(),
            });
        }
        let epochs = config.epochs();
        let k = f64::from(clusters.get());
        let n = rows as f64;
        let t = epochs.get();

        let (batch_size, batch_source) = match config.batch() {
            Batch::Fixed(requested) => {
                // `rows >= clusters >= 1`, so the minimum is not zero.
                let size = NonZeroUsize::new(requested.get().min(rows)).expect("rows >= 1");
                (size, BatchSource::Fixed { requested })
            }
            Batch::Law(law) => (law_batch_size(law, k, rows, t), BatchSource::Law(law)),
            Batch::Auto => {
                let law = if dim.get() <= AUTO_QUALITY_MAX_DIM {
                    BatchLaw::Quality
                } else {
                    BatchLaw::Throughput
                };
                (law_batch_size(law, k, rows, t), BatchSource::AutoLaw(law))
            }
        };

        let (alpha0, step_source) = match config.step() {
            Step::Fixed(alpha) => (alpha.get(), StepSource::Fixed),
            Step::Law => {
                let raw = STEP_LAW.evaluate(k, batch_size.get() as f64, n, t);
                (raw.clamp(MIN_ALPHA0, 1.0), StepSource::Law)
            }
        };

        let revival = match config.revival() {
            Revival::Off => None,
            Revival::Threshold(threshold) => Some(threshold),
            Revival::Auto if dim.get() <= AUTO_NO_REVIVAL_MAX_DIM => None,
            Revival::Auto => Some(Threshold::DEFAULT),
        };

        let whole = t.floor();
        // `T <= Epochs::MAX = u32::MAX`, so the cast is exact.
        let full_passes = whole as u64;
        // `0 <= frac(T) < 1`, so the product is at most `n` and the cast is exact.
        let partial_rows = ((t - whole) * n).round_ties_even() as usize;
        if full_passes == 0 && partial_rows == 0 {
            return Err(Error::EmptyBudget { epochs: t, rows });
        }

        let mut recipe = Self {
            rows,
            dim,
            clusters,
            epochs,
            metric: config.metric(),
            batch_size,
            batch_source,
            alpha0,
            step_source,
            revival,
            full_passes,
            partial_rows,
            total_rows: 0,
            total_updates: 0,
            seed: config.seed(),
        };
        recipe.total_rows = full_passes
            .checked_mul(rows as u64)
            .and_then(|total| total.checked_add(partial_rows as u64))
            .ok_or(Error::BudgetTooLarge)?;
        recipe.total_updates = recipe
            .updates_per_pass()
            .checked_mul(full_passes)
            .and_then(|updates| updates.checked_add(recipe.batches_in(partial_rows)))
            .ok_or(Error::BudgetTooLarge)?;
        Ok(recipe)
    }

    /// The number of training rows `n`.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The dimension `d` of the vectors.
    pub fn dim(&self) -> usize {
        self.dim.get()
    }

    /// The number of clusters `k`.
    pub fn clusters(&self) -> NonZeroU32 {
        self.clusters
    }

    /// The epoch budget `T`.
    pub fn epochs(&self) -> Epochs {
        self.epochs
    }

    /// The geometry of the clustering.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// The batch size `B`: the number of rows per update (the last batch of
    /// an epoch can be smaller).
    pub fn batch_size(&self) -> NonZeroUsize {
        self.batch_size
    }

    /// Where the batch size comes from.
    pub fn batch_source(&self) -> BatchSource {
        self.batch_source
    }

    /// The initial step size `α₀` of the cosine schedule.
    pub fn alpha0(&self) -> f64 {
        self.alpha0
    }

    /// Where the initial step size comes from.
    pub fn step_source(&self) -> StepSource {
        self.step_source
    }

    /// The revival threshold `τ`, or `None` if revival is off.
    pub fn revival(&self) -> Option<Threshold> {
        self.revival
    }

    /// The number of epochs (passes over the data, full or partial).
    pub fn epoch_count(&self) -> u64 {
        if self.full_passes == 0 {
            1
        } else {
            self.full_passes + u64::from(self.partial_rows > 0)
        }
    }

    /// The number of rows of epoch `epoch` (0-based), or `None` if the run
    /// has fewer epochs.
    pub fn rows_in_epoch(&self, epoch: u64) -> Option<usize> {
        self.planned_epoch(epoch).map(|epoch| epoch.rows().get())
    }

    /// The number of rows of each epoch, in order.
    pub fn epoch_sizes(&self) -> impl Iterator<Item = usize> + '_ {
        self.plan().map(|epoch| epoch.rows().get())
    }

    /// Epoch `index` (0-based) of the plan, or `None` if the run has fewer
    /// epochs.
    pub(crate) fn planned_epoch(&self, index: u64) -> Option<PlannedEpoch> {
        let count = self.epoch_count();
        if index >= count {
            return None;
        }
        let has_partial = self.partial_rows > 0;
        let is_partial = self.full_passes == 0 || (has_partial && index == count - 2);
        let rows = if is_partial {
            self.partial_rows
        } else {
            self.rows
        };
        // The epochs before `index` are full passes, and the partial pass
        // when `index` is the last full pass that follows it.
        let partial_before = has_partial && self.full_passes > 0 && index == count - 1;
        let full_before = index - u64::from(partial_before);
        let (partial_rows, partial_updates) = if partial_before {
            (self.partial_rows as u64, self.batches_in(self.partial_rows))
        } else {
            (0, 0)
        };
        // Both sums are at most the run's totals, which fit in `u64`.
        Some(PlannedEpoch {
            index,
            rows: NonZeroUsize::new(rows).expect("every epoch of a plan has rows"),
            first_row: full_before * self.rows as u64 + partial_rows,
            first_update: full_before * self.updates_per_pass() + partial_updates,
            is_last: index + 1 == count,
        })
    }

    /// The epochs of the plan, in order.
    pub(crate) fn plan(&self) -> impl Iterator<Item = PlannedEpoch> + '_ {
        (0..self.epoch_count()).map(|index| self.planned_epoch(index).expect("index < epoch_count"))
    }

    /// Update `index` (0-based) of the run: its step size, and the revival
    /// check that follows it, if any.
    pub(crate) fn update(&self, index: u64) -> Update {
        let revival_due = (index + 1) % REVIVAL_INTERVAL == 0;
        Update {
            alpha: self.alpha(index),
            revival: self.revival.filter(|_| revival_due),
        }
    }

    /// The number of updates of epoch `epoch`, or `None` if the run has fewer
    /// epochs.
    pub fn updates_in_epoch(&self, epoch: u64) -> Option<u64> {
        self.rows_in_epoch(epoch).map(|rows| self.batches_in(rows))
    }

    /// The number of updates of a full pass over the data, `⌈n / B⌉`.
    pub fn updates_per_pass(&self) -> u64 {
        self.batches_in(self.rows)
    }

    /// The number of updates of the whole run, `M`.
    pub fn total_updates(&self) -> u64 {
        self.total_updates
    }

    /// The number of rows of the whole run.
    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    /// The seed of every random choice of the run.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Whether the last epoch visits every row, so every row gets a label
    /// from it. False only for a budget below one epoch whose rows round to
    /// fewer than `n`.
    pub fn last_epoch_is_full(&self) -> bool {
        self.full_passes > 0 || self.partial_rows == self.rows
    }

    /// The step size `α(t)` of update `t` (0-based), from the cosine schedule
    /// of Equation 5 of the paper:
    ///
    /// ```text
    /// α(t) = α₀ · (ε + (1 − ε) · ½ · (1 + cos(π · t / M))),   ε = ALPHA_MIN_FACTOR
    /// ```
    ///
    /// The step anneals from `α₀` at the first update towards `ε · α₀` at
    /// update `M`, the total number of updates. Updates past `M` keep the
    /// floor.
    pub fn alpha(&self, update: u64) -> f64 {
        let fraction = update.min(self.total_updates) as f64 / self.total_updates.max(1) as f64;
        let cosine = 0.5 * (1.0 + (core::f64::consts::PI * fraction).cos());
        self.alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)
    }

    /// The number of batches that cover `rows` rows.
    fn batches_in(&self, rows: usize) -> u64 {
        rows.div_ceil(self.batch_size.get()) as u64
    }
}

/// One epoch of a [`Recipe`]'s plan, with its place in the run.
///
/// The epoch is cut into batches of [`Recipe::batch_size`] rows; its last
/// batch keeps the remaining rows, so it has
/// [`Recipe::updates_in_epoch`] updates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlannedEpoch {
    index: u64,
    rows: NonZeroUsize,
    /// The rows of the run before this epoch.
    first_row: u64,
    /// The updates of the run before this epoch.
    first_update: u64,
    is_last: bool,
}

impl PlannedEpoch {
    /// The index of the epoch, from 0.
    pub(crate) fn index(&self) -> u64 {
        self.index
    }

    /// The rows of the epoch.
    pub(crate) fn rows(&self) -> NonZeroUsize {
        self.rows
    }

    /// The rows of the run before this epoch.
    pub(crate) fn first_row(&self) -> u64 {
        self.first_row
    }

    /// The updates of the run before this epoch.
    pub(crate) fn first_update(&self) -> u64 {
        self.first_update
    }

    /// Whether this is the last epoch, whose rows give the closing means.
    pub(crate) fn is_last(&self) -> bool {
        self.is_last
    }
}

/// What one update of a run does: Algorithm 1, line 9, with step size
/// `alpha`, then a revival of dead clusters if `revival` is set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Update {
    /// The step size `α(t)`.
    pub(crate) alpha: f64,
    /// The revival threshold, on every [`REVIVAL_INTERVAL`]th update when
    /// revival is on.
    pub(crate) revival: Option<Threshold>,
}

impl fmt::Display for Recipe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} rows of dimension {} into {} clusters, {} epochs, {} updates (batch {}",
            self.rows,
            self.dim,
            self.clusters,
            self.epochs.get(),
            self.total_updates,
            self.batch_size,
        )?;
        match self.batch_source {
            BatchSource::Law(law) => write!(f, " from the {law:?} law")?,
            BatchSource::AutoLaw(law) => write!(f, " from the {law:?} law, chosen by dimension")?,
            BatchSource::Fixed { .. } => write!(f, ", fixed")?,
        }
        write!(f, "; alpha0 {:.6}", self.alpha0)?;
        if self.step_source == StepSource::Fixed {
            write!(f, ", fixed")?;
        }
        match self.revival {
            Some(threshold) => write!(f, "; revival below {} of the mean count", threshold.get())?,
            None => write!(f, "; revival off")?,
        }
        if self.metric == Metric::Spherical {
            write!(f, "; spherical")?;
        }
        write!(f, "; seed {})", self.seed)
    }
}

/// The batch size that `law` gives, rounded to a power of two and capped
/// as the reference implementation does.
fn law_batch_size(law: BatchLaw, clusters: f64, rows: usize, epochs: f64) -> NonZeroUsize {
    let raw = law.fit().evaluate(clusters, rows as f64, epochs);
    let rounded = round_to_power_of_two(raw);
    // The reference implementation caps the two laws differently; this
    // reproduces both of its fitted recipes.
    let capped = match law {
        BatchLaw::Throughput => rounded.min(round_to_power_of_two(rows as f64 / 2.0)),
        BatchLaw::Quality => rounded.min((rows / 2) as f64),
    };
    // `capped` is a whole number; the cast saturates at `usize::MAX`.
    let size = (capped as usize).clamp(1, rows.max(1));
    NonZeroUsize::new(size).expect("size >= 1")
}

/// Rounds a positive value to the nearest power of two, comparing ratios
/// (so the midpoint between `2^m` and `2^(m+1)` is `2^m · √2`). Values of
/// at most 1 give 1; a value at the midpoint rounds up.
fn round_to_power_of_two(value: f64) -> f64 {
    if value.is_nan() || value <= 1.0 {
        return 1.0;
    }
    let low = value.log2().floor().exp2();
    let high = 2.0 * low;
    if value / low < high / value {
        low
    } else {
        high
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(clusters: u32) -> Emeans {
        Emeans::new(NonZeroU32::new(clusters).unwrap())
    }

    fn resolve(config: Emeans, rows: usize, dim: usize) -> Recipe {
        config.recipe(rows, dim).unwrap()
    }

    #[test]
    fn rounds_to_the_geometrically_nearest_power_of_two() {
        assert_eq!(round_to_power_of_two(f64::NAN), 1.0);
        assert_eq!(round_to_power_of_two(-3.0), 1.0);
        assert_eq!(round_to_power_of_two(1.0), 1.0);
        assert_eq!(round_to_power_of_two(1.4), 1.0);
        assert_eq!(round_to_power_of_two(1.5), 2.0);
        assert_eq!(round_to_power_of_two(1024.0), 1024.0);
        assert_eq!(round_to_power_of_two(1448.0), 1024.0);
        assert_eq!(round_to_power_of_two(1449.0), 2048.0);
    }

    #[test]
    fn plans_fractional_epochs_before_the_last_full_pass() {
        let recipe = resolve(config(4).with_epochs(Epochs::new(2.5).unwrap()), 1000, 8);
        assert_eq!(recipe.epoch_sizes().collect::<Vec<_>>(), [1000, 500, 1000]);
        assert!(recipe.last_epoch_is_full());
        assert_eq!(recipe.total_rows(), 2500);
        assert_eq!(recipe.rows_in_epoch(3), None);

        let recipe = resolve(config(4).with_epochs(Epochs::new(0.6).unwrap()), 1000, 8);
        assert_eq!(recipe.epoch_sizes().collect::<Vec<_>>(), [600]);
        assert!(!recipe.last_epoch_is_full());

        let recipe = resolve(config(4).with_epochs(Epochs::new(3.0).unwrap()), 1000, 8);
        assert_eq!(recipe.epoch_sizes().collect::<Vec<_>>(), [1000, 1000, 1000]);
    }

    #[test]
    fn rounds_partial_epochs_half_to_even() {
        // 0.5 · 5 = 2.5 rounds to 2; 0.5 · 7 = 3.5 rounds to 4.
        let half = Epochs::new(1.5).unwrap();
        assert_eq!(
            resolve(config(1).with_epochs(half), 5, 3).rows_in_epoch(0),
            Some(2)
        );
        assert_eq!(
            resolve(config(1).with_epochs(half), 7, 3).rows_in_epoch(0),
            Some(4)
        );
    }

    #[test]
    fn rejects_budgets_that_round_to_no_rows() {
        let tiny = config(1).with_epochs(Epochs::new(0.001).unwrap());
        assert_eq!(
            tiny.recipe(100, 2),
            Err(Error::EmptyBudget {
                epochs: 0.001,
                rows: 100
            })
        );
    }

    #[test]
    fn rejects_runs_whose_row_count_overflows() {
        // 2^24 passes of 2^40 rows: the updates fit in a u64, the rows do not.
        let config = config(1).with_epochs(Epochs::new(f64::from(1u32 << 24)).unwrap());
        assert_eq!(config.recipe(1 << 40, 17), Err(Error::BudgetTooLarge));
    }

    #[test]
    fn rejects_fewer_rows_than_clusters() {
        assert_eq!(
            config(5).recipe(4, 2),
            Err(Error::TooFewRows {
                rows: 4,
                clusters: 5
            })
        );
        assert_eq!(config(5).recipe(4, 0), Err(Error::ZeroDimension));
    }

    #[test]
    fn counts_updates_with_a_smaller_last_batch() {
        let size = NonZeroUsize::new(300).unwrap();
        let recipe = resolve(
            config(4)
                .with_batch(Batch::Fixed(size))
                .with_epochs(Epochs::new(1.5).unwrap()),
            1000,
            8,
        );
        assert_eq!(recipe.updates_per_pass(), 4);
        assert_eq!(recipe.updates_in_epoch(0), Some(2));
        assert_eq!(recipe.updates_in_epoch(1), Some(4));
        assert_eq!(recipe.total_updates(), 6);
    }

    #[test]
    fn each_planned_epoch_starts_where_the_previous_one_ends() {
        let size = NonZeroUsize::new(300).unwrap();
        for epochs in [0.6, 1.0, 1.5, 2.5, 3.0, 4.25] {
            let recipe = resolve(
                config(4)
                    .with_batch(Batch::Fixed(size))
                    .with_epochs(Epochs::new(epochs).unwrap()),
                1000,
                8,
            );
            let plan: Vec<PlannedEpoch> = recipe.plan().collect();
            let (mut rows, mut updates) = (0, 0);
            for (index, epoch) in plan.iter().enumerate() {
                assert_eq!(epoch.index(), index as u64, "T = {epochs}");
                assert_eq!(
                    (epoch.first_row(), epoch.first_update()),
                    (rows, updates),
                    "T = {epochs}, epoch {index}"
                );
                assert_eq!(epoch.is_last(), index + 1 == plan.len(), "T = {epochs}");
                rows += epoch.rows().get() as u64;
                updates += recipe.updates_in_epoch(epoch.index()).unwrap();
            }
            assert_eq!(
                (rows, updates),
                (recipe.total_rows(), recipe.total_updates()),
                "T = {epochs}"
            );
            assert_eq!(recipe.planned_epoch(plan.len() as u64), None);
        }
    }

    #[test]
    fn every_fifth_update_checks_for_revival() {
        let revived = |recipe: &Recipe| -> Vec<u64> {
            (0..12)
                .filter(|&t| recipe.update(t).revival.is_some())
                .collect()
        };
        let threshold = Threshold::new(0.25).unwrap();
        let on = resolve(
            config(4).with_revival(Revival::Threshold(threshold)),
            1000,
            8,
        );
        assert_eq!(revived(&on), [4, 9]);
        assert_eq!(on.update(4).revival, Some(threshold));
        assert_eq!(on.update(7).alpha, on.alpha(7));
        let off = resolve(config(4).with_revival(Revival::Off), 1000, 8);
        assert_eq!(revived(&off), []);
    }

    #[test]
    fn fixed_batch_is_capped_at_the_rows_and_keeps_the_step_law() {
        let requested = NonZeroUsize::new(5000).unwrap();
        let recipe = resolve(config(10).with_batch(Batch::Fixed(requested)), 1000, 8);
        assert_eq!(recipe.batch_size().get(), 1000);
        assert_eq!(recipe.batch_source(), BatchSource::Fixed { requested });
        assert_eq!(recipe.step_source(), StepSource::Law);
    }

    #[test]
    fn auto_choices_depend_on_the_dimension() {
        let low = resolve(config(16), 100_000, 2);
        assert_eq!(low.batch_source(), BatchSource::AutoLaw(BatchLaw::Quality));
        assert_eq!(low.revival(), None);
        let mid = resolve(config(16), 100_000, 16);
        assert_eq!(mid.batch_source(), BatchSource::AutoLaw(BatchLaw::Quality));
        assert_eq!(mid.revival(), Some(Threshold::DEFAULT));
        let high = resolve(config(16), 100_000, 17);
        assert_eq!(
            high.batch_source(),
            BatchSource::AutoLaw(BatchLaw::Throughput)
        );
        assert!(high.batch_size() > mid.batch_size());
    }

    #[test]
    fn step_schedule_anneals_from_alpha0_to_a_tenth() {
        let alpha = Alpha::new(0.5).unwrap();
        let recipe = resolve(config(4).with_step(Step::Fixed(alpha)), 10_000, 8);
        let last = recipe.total_updates();
        assert_eq!(recipe.alpha(0), 0.5);
        assert!((recipe.alpha(last / 2) - 0.5 * 0.55).abs() < 0.5 * 0.9 * 0.5 / last as f64 * 4.0);
        assert!((recipe.alpha(last) - 0.05).abs() < 1e-15);
        assert_eq!(recipe.alpha(last + 10), recipe.alpha(last));
        let mut previous = f64::INFINITY;
        for update in 0..last {
            let value = recipe.alpha(update);
            assert!(value < previous && value > 0.05);
            previous = value;
        }
    }

    use crate::config::Alpha;
}
