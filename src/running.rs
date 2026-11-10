//! The running centroids of a training run, and the search that assigns
//! rows to them.
//!
//! The search holds its own copy of the centroids, rounded to the element
//! type and, for spherical clustering, scaled to norm 1. [`RunningCentroids`]
//! owns both, and every change to the centroids reloads the search before it
//! returns, so the search never assigns rows to old centroids.

use crate::config::Metric;
use crate::kernel::SearchCentroids;
use crate::positions;
use crate::recipe::Update;
use crate::rng::Rng;
use crate::scalar::Scalar;
use crate::state::EmaState;

/// The EMA state of a run and the search built from it.
pub(crate) struct RunningCentroids<T: Scalar> {
    state: EmaState,
    search: SearchCentroids<T>,
    view: SearchView,
    /// The random choices of revival.
    revival_rng: Rng,
}

/// How the search sees the running centroids.
enum SearchView {
    /// As they are ([`Metric::SquaredEuclidean`]).
    Direct,
    /// Scaled to norm 1 ([`Metric::Spherical`]), in this `k × d` buffer.
    Unit(Vec<f64>),
}

impl<T: Scalar> RunningCentroids<T> {
    /// Starts from `positions` (`k × d`, row-major; of norm 1 for
    /// [`Metric::Spherical`]).
    pub(crate) fn new(positions: Vec<f64>, dim: usize, metric: Metric, revival_rng: Rng) -> Self {
        let view = match metric {
            Metric::SquaredEuclidean => SearchView::Direct,
            Metric::Spherical => SearchView::Unit(positions.clone()),
        };
        Self {
            search: SearchCentroids::new(&positions, dim, metric),
            state: EmaState::new(positions, dim),
            view,
            revival_rng,
        }
    }

    /// The search over the current centroids.
    pub(crate) fn search(&self) -> &SearchCentroids<T> {
        &self.search
    }

    /// Applies `update` with a batch's sums `s_j` (`k × d`) and counts
    /// `n_j`, revives dead clusters if the update says so, and reloads the
    /// search. Returns the number of clusters revived.
    pub(crate) fn apply(&mut self, update: Update, sums: &[f64], counts: &[u64]) -> u64 {
        self.state.apply(update.alpha, sums, counts);
        let revived = match update.revival {
            Some(threshold) => {
                let metric = self.search.metric();
                self.state
                    .revive::<T>(threshold.get(), metric, &mut self.revival_rng)
                    as u64
            }
            None => 0,
        };
        self.reload_search();
        revived
    }

    /// The running centroids `c_j`, `k × d`, row-major.
    pub(crate) fn into_positions(self) -> Vec<f64> {
        self.state.into_positions()
    }

    fn reload_search(&mut self) {
        match &mut self.view {
            SearchView::Direct => self.search.load(self.state.positions()),
            SearchView::Unit(unit) => {
                unit.copy_from_slice(self.state.positions());
                positions::normalize_each(unit, self.search.dim());
                self.search.load(unit);
            }
        }
    }
}
