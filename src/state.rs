//! The running state of the clusters: Algorithm 1 of the paper, lines 3, 9
//! and 11, and the dead-cluster revival of its Appendix H.

use crate::config::Metric;
use crate::recipe::{INITIAL_MASS, SPLIT_EPSILON};
use crate::rng::Rng;
use crate::scalar::{Element, Scalar};

/// The running count `N_j` and centroid `c_j = S_j / N_j` of every cluster.
///
/// The paper keeps the running sum `S_j`; this keeps the centroid instead,
/// which is the same state in a form that never divides by a small count.
/// All values are `f64`, whatever the element type of the data.
#[derive(Clone, Debug)]
pub(crate) struct EmaState {
    dim: usize,
    /// `k × d` row-major centroids `c_j`.
    positions: Vec<f64>,
    /// Running counts `N_j`.
    mass: Vec<f64>,
}

impl EmaState {
    /// Starts every cluster at its initial centroid with the running count
    /// [`INITIAL_MASS`] (Algorithm 1, line 3).
    pub(crate) fn new(positions: Vec<f64>, dim: usize) -> Self {
        debug_assert!(dim > 0 && positions.len() % dim == 0);
        let clusters = positions.len() / dim;
        Self {
            dim,
            positions,
            mass: vec![INITIAL_MASS; clusters],
        }
    }

    /// `k × d` row-major centroids.
    pub(crate) fn positions(&self) -> &[f64] {
        &self.positions
    }

    /// `k × d` row-major centroids.
    pub(crate) fn into_positions(self) -> Vec<f64> {
        self.positions
    }

    /// Running counts `N_j`.
    #[cfg(test)]
    pub(crate) fn mass(&self) -> &[f64] {
        &self.mass
    }

    /// Applies one batch with step size `alpha` (Algorithm 1, line 9):
    ///
    /// ```text
    /// N_j ← (1 − α) · N_j + α · n_j
    /// S_j ← (1 − α) · S_j + α · s_j
    /// ```
    ///
    /// `sums` holds `s_j` (`k × d`) and `counts` holds `n_j`. A cluster with
    /// `n_j > 0` moves to `c_j + w · (s_j / n_j − c_j)` with
    /// `w = α · n_j / N_j'`, which equals `S_j' / N_j'`. A cluster with
    /// `n_j = 0` keeps its centroid; only its running count decays.
    ///
    /// With `α = 1` the batch replaces the state of the clusters it reaches,
    /// and the other clusters keep their running counts, so that no count
    /// drops to zero. This is the reference implementation's rule.
    pub(crate) fn apply(&mut self, alpha: f64, sums: &[f64], counts: &[u64]) {
        debug_assert_eq!(counts.len(), self.mass.len());
        let replace = alpha == 1.0;
        if !replace {
            let decay = 1.0 - alpha;
            for mass in &mut self.mass {
                *mass *= decay;
            }
        }
        let dim = self.dim;
        for (cluster, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let count = count as f64;
            let retained = if replace { 0.0 } else { self.mass[cluster] };
            let mass = retained + alpha * count;
            let weight = alpha * count / mass;
            let sum = &sums[cluster * dim..(cluster + 1) * dim];
            let position = &mut self.positions[cluster * dim..(cluster + 1) * dim];
            for (c, &s) in position.iter_mut().zip(sum) {
                let mean = s / count;
                *c = if weight == 1.0 {
                    mean
                } else {
                    *c + weight * (mean - *c)
                };
            }
            self.mass[cluster] = mass;
        }
    }

    /// Revives dead clusters (Appendix H of the paper). Returns the number of
    /// clusters revived.
    ///
    /// A cluster is dead when its running count is below `threshold` times
    /// the mean running count. Each dead cluster, in index order, takes the
    /// place of one half of a donor: a live cluster with a positive count,
    /// drawn without replacement with probability proportional to its count.
    /// There are at most as many revivals as donors. The donor's centroid is
    /// perturbed in two opposite directions (Equation 8): the dead cluster
    /// takes `c ⊙ (1 + δ, 1 − δ, 1 + δ, …)`, the donor keeps
    /// `c ⊙ (1 − δ, 1 + δ, 1 − δ, …)`, and each takes half of the donor's
    /// running count.
    ///
    /// The two halves must differ as the nearest-centroid search sees them
    /// (rounded to `T` and, for [`Metric::Spherical`], scaled to norm 1), or
    /// the dead cluster would lose every tie and stay dead. When Equation 8
    /// does not separate them, [`EmaState::split`] adds a second perturbation.
    pub(crate) fn revive<T: Scalar>(
        &mut self,
        threshold: f64,
        metric: Metric,
        rng: &mut Rng,
    ) -> usize {
        let clusters = self.mass.len();
        let total: f64 = self.mass.iter().sum();
        let limit = total * threshold / clusters as f64;
        let dead: Vec<usize> = (0..clusters).filter(|&j| self.mass[j] < limit).collect();
        let live = |mass: f64| mass >= limit && mass > 0.0;
        let donor_count = self.mass.iter().filter(|&&mass| live(mass)).count();
        let splits = dead.len().min(donor_count);
        if splits == 0 {
            return 0;
        }

        // Weighted sampling without replacement with the Gumbel-top-k trick:
        // the `splits` largest keys `ln(N_j) − ln(−ln(u_j))` are a sample with
        // probabilities proportional to `N_j`. One `u_j` is drawn per cluster,
        // in index order, so the draw does not depend on which clusters are
        // donors.
        let mut keys: Vec<(f64, usize)> = Vec::with_capacity(donor_count);
        for (cluster, &mass) in self.mass.iter().enumerate() {
            let uniform = rng.open_unit();
            if live(mass) {
                keys.push((mass.ln() - (-uniform.ln()).ln(), cluster));
            }
        }
        // Largest key first; ties (of probability zero) go to the lower index.
        keys.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

        for (&(_, donor), &target) in keys.iter().zip(&dead) {
            self.split::<T>(donor, target, metric);
        }
        splits
    }

    /// Splits cluster `donor` in two, moving one half to cluster `target`.
    ///
    /// If Equation 8 leaves the halves equal in the search, a second
    /// perturbation of `±δ · scale` goes on one coordinate:
    ///
    /// - Squared Euclidean: this happens only for a donor at (or within a
    ///   few subnormals of) the origin. The coordinate is the first one, and
    ///   `scale` is the root mean square of the centroid norms (1 if they
    ///   are all zero), a length typical of the data.
    /// - Spherical: this happens when every nonzero coordinate of the donor
    ///   has the same sign in Equation 8 (for example, a donor with one
    ///   nonzero coordinate), so the halves keep the donor's direction. The
    ///   coordinate is the one of smallest magnitude, and `scale` is the
    ///   donor's norm (1 if it is zero). This turns the halves by about `δ`.
    ///
    /// With `d = 1` in spherical clustering, every direction is ±1 and no
    /// perturbation separates the halves.
    fn split<T: Scalar>(&mut self, donor: usize, target: usize, metric: Metric) {
        let dim = self.dim;
        let half = self.mass[donor] / 2.0;
        self.mass[donor] = half;
        self.mass[target] = half;

        let source = self.positions[donor * dim..(donor + 1) * dim].to_vec();
        let mut kept: Vec<f64> = Vec::with_capacity(dim);
        let mut moved: Vec<f64> = Vec::with_capacity(dim);
        for (t, &c) in source.iter().enumerate() {
            let offset = if t % 2 == 0 {
                SPLIT_EPSILON
            } else {
                -SPLIT_EPSILON
            };
            kept.push(c * (1.0 - offset));
            moved.push(c * (1.0 + offset));
        }
        if !separated_in_search::<T>(&kept, &moved, metric) {
            let (coordinate, scale) = match metric {
                Metric::SquaredEuclidean => {
                    let squares: f64 = self.positions.iter().map(|v| v * v).sum();
                    let rms = (squares / self.mass.len() as f64).sqrt();
                    (0, if rms > 0.0 { rms } else { 1.0 })
                }
                Metric::Spherical => {
                    let smallest = (0..dim)
                        .min_by(|&a, &b| source[a].abs().total_cmp(&source[b].abs()))
                        .unwrap_or(0);
                    let norm = source.iter().map(|v| v * v).sum::<f64>().sqrt();
                    (smallest, if norm > 0.0 { norm } else { 1.0 })
                }
            };
            kept[coordinate] -= SPLIT_EPSILON * scale;
            moved[coordinate] += SPLIT_EPSILON * scale;
        }
        self.positions[donor * dim..(donor + 1) * dim].copy_from_slice(&kept);
        self.positions[target * dim..(target + 1) * dim].copy_from_slice(&moved);
    }
}

/// Whether two centroids differ as the nearest-centroid search sees them:
/// rounded to `T` and, for [`Metric::Spherical`], scaled to norm 1 first.
fn separated_in_search<T: Scalar>(a: &[f64], b: &[f64], metric: Metric) -> bool {
    let view = |values: &[f64]| -> Vec<T> {
        let mut values = values.to_vec();
        if metric == Metric::Spherical {
            crate::numeric::normalize_in_place(&mut values);
        }
        values.into_iter().map(T::from_f64).collect()
    };
    view(a) != view(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Stream;

    /// The sum-and-count form of the update, written as in the paper.
    fn reference_update(
        sum: &mut [f64],
        mass: &mut f64,
        alpha: f64,
        batch_sum: &[f64],
        batch_count: f64,
    ) {
        *mass = (1.0 - alpha) * *mass + alpha * batch_count;
        for (s, &b) in sum.iter_mut().zip(batch_sum) {
            *s = (1.0 - alpha) * *s + alpha * b;
        }
    }

    #[test]
    fn apply_matches_the_sum_and_count_form() {
        let mut state = EmaState::new(vec![1.0, 2.0, -3.0, 0.5], 2);
        let mut sums = [
            1.0 * INITIAL_MASS,
            2.0 * INITIAL_MASS,
            -3.0 * INITIAL_MASS,
            0.5 * INITIAL_MASS,
        ];
        let mut masses = [INITIAL_MASS; 2];
        let batches: [(f64, [f64; 4], [u64; 2]); 3] = [
            (0.7, [4.0, 6.0, 0.0, 0.0], [2, 0]),
            (0.3, [1.0, 1.0, -5.0, 2.0], [1, 3]),
            (0.05, [10.0, -2.0, 3.0, 3.0], [4, 2]),
        ];
        for (alpha, batch_sums, counts) in batches {
            state.apply(alpha, &batch_sums, &counts);
            for j in 0..2 {
                reference_update(
                    &mut sums[2 * j..2 * j + 2],
                    &mut masses[j],
                    alpha,
                    &batch_sums[2 * j..2 * j + 2],
                    counts[j] as f64,
                );
            }
        }
        for j in 0..2 {
            assert!((state.mass()[j] - masses[j]).abs() <= 1e-15 * masses[j]);
            for t in 0..2 {
                let expected = sums[2 * j + t] / masses[j];
                let actual = state.positions()[2 * j + t];
                assert!(
                    (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                    "{actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn untouched_clusters_keep_their_centroid() {
        let mut state = EmaState::new(vec![1.0, 2.0, 3.0, 4.0], 2);
        state.apply(0.5, &[2.0, 2.0, 0.0, 0.0], &[1, 0]);
        assert_eq!(&state.positions()[2..], &[3.0, 4.0]);
        assert_eq!(state.mass()[1], INITIAL_MASS * 0.5);
    }

    #[test]
    fn alpha_one_replaces_reached_clusters_and_keeps_the_others() {
        let mut state = EmaState::new(vec![0.0, 0.0, 9.0, 9.0], 2);
        state.apply(1.0, &[3.0, 6.0, 0.0, 0.0], &[3, 0]);
        assert_eq!(state.positions(), &[1.0, 2.0, 9.0, 9.0]);
        assert_eq!(state.mass(), &[3.0, INITIAL_MASS]);
    }

    #[test]
    fn revive_splits_donors_into_dead_clusters() {
        // Mean count 101.1 / 4, so the limit is 2.5275: clusters 1 and 3 are dead
        // and clusters 0 and 2 are the donors.
        let initial = [2.0, 4.0, 0.0, 0.0, -1.0, 8.0, 5.0, 5.0];
        let mut state = EmaState::new(initial.to_vec(), 2);
        state.mass = vec![50.0, 0.1, 51.0, 0.0];
        let mut rng = Rng::new(0, Stream::Revival);
        assert_eq!(
            state.revive::<f32>(0.1, Metric::SquaredEuclidean, &mut rng),
            2
        );

        // Dead clusters are filled in index order, from donors in sampled order.
        let first_donor = if state.mass()[1] == 25.0 { 0 } else { 2 };
        let second_donor = 2 - first_donor;
        for (donor, target) in [(first_donor, 1), (second_donor, 3)] {
            let half = [50.0, 0.1, 51.0, 0.0][donor] / 2.0;
            assert_eq!((state.mass()[donor], state.mass()[target]), (half, half));
            let c = &initial[2 * donor..2 * donor + 2];
            let kept = [c[0] * (1.0 - SPLIT_EPSILON), c[1] * (1.0 + SPLIT_EPSILON)];
            let moved = [c[0] * (1.0 + SPLIT_EPSILON), c[1] * (1.0 - SPLIT_EPSILON)];
            assert_eq!(&state.positions()[2 * donor..2 * donor + 2], &kept);
            assert_eq!(&state.positions()[2 * target..2 * target + 2], &moved);
        }
    }

    #[test]
    fn revive_does_nothing_without_dead_clusters_or_donors() {
        let mut state = EmaState::new(vec![1.0, 2.0, 3.0], 1);
        state.mass = vec![1.0, 1.0, 1.0];
        let mut rng = Rng::new(0, Stream::Revival);
        assert_eq!(
            state.revive::<f64>(0.1, Metric::SquaredEuclidean, &mut rng),
            0
        );
        state.mass = vec![0.0, 0.0, 0.0];
        assert_eq!(
            state.revive::<f64>(0.1, Metric::SquaredEuclidean, &mut rng),
            0
        );
        assert_eq!(state.positions(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn revive_draws_donors_in_proportion_to_their_counts() {
        let trials = 20_000;
        let mut rng = Rng::new(5, Stream::Revival);
        let mut heavy = 0;
        for _ in 0..trials {
            let mut state = EmaState::new(vec![0.0, 10.0, 20.0], 1);
            state.mass = vec![0.0, 1.0, 3.0];
            assert_eq!(
                state.revive::<f64>(0.1, Metric::SquaredEuclidean, &mut rng),
                1
            );
            heavy += usize::from(state.mass()[2] == 1.5);
        }
        // Expect 3/4 of the trials; the standard deviation is about 0.003.
        let share = heavy as f64 / trials as f64;
        assert!((share - 0.75).abs() < 0.02, "{share}");
    }

    #[test]
    fn revive_separates_a_donor_at_the_origin() {
        // The root mean square of the centroid norms is √((0 + 98) / 2) = 7.
        let mut state = EmaState::new(vec![0.0, 0.0, 7.0, 7.0], 2);
        state.mass = vec![10.0, 0.0];
        let mut rng = Rng::new(0, Stream::Revival);
        assert_eq!(
            state.revive::<f32>(0.1, Metric::SquaredEuclidean, &mut rng),
            1
        );
        let step = SPLIT_EPSILON * 7.0;
        assert_eq!(&state.positions()[..2], &[-step, 0.0]);
        assert_eq!(&state.positions()[2..], &[step, 0.0]);
    }

    #[test]
    fn spherical_revive_turns_halves_that_keep_the_donor_direction() {
        // Both donors have their nonzero coordinates at even positions, so
        // Equation 8 only scales them.
        for donor in [vec![0.0, 0.0, 1.0, 0.0], vec![0.6, 0.0, 0.8, 0.0]] {
            let mut positions = donor.clone();
            positions.extend([0.0, 1.0, 0.0, 0.0]);
            let mut state = EmaState::new(positions, 4);
            state.mass = vec![10.0, 0.0];
            let mut rng = Rng::new(0, Stream::Revival);
            assert_eq!(state.revive::<f32>(0.1, Metric::Spherical, &mut rng), 1);
            let (kept, moved) = state.positions().split_at(4);
            assert!(
                separated_in_search::<f32>(kept, moved, Metric::Spherical),
                "{donor:?}"
            );
            // The halves turn by about δ, in opposite directions.
            let unit = |v: &[f64]| {
                let mut v = v.to_vec();
                crate::numeric::normalize_in_place(&mut v);
                v
            };
            let cosine: f64 = unit(kept).iter().zip(unit(moved)).map(|(a, b)| a * b).sum();
            let angle = cosine.min(1.0).acos();
            assert!(
                (angle - 2.0 * SPLIT_EPSILON).abs() < 0.1 * SPLIT_EPSILON,
                "{donor:?}: {angle}"
            );
        }
    }

    #[test]
    fn equation_8_separates_ordinary_spherical_donors() {
        let mut state = EmaState::new(vec![0.6, 0.8, 0.0, 1.0], 2);
        state.mass = vec![10.0, 0.0];
        let mut rng = Rng::new(0, Stream::Revival);
        assert_eq!(state.revive::<f32>(0.1, Metric::Spherical, &mut rng), 1);
        let kept = [0.6 * (1.0 - SPLIT_EPSILON), 0.8 * (1.0 + SPLIT_EPSILON)];
        assert_eq!(&state.positions()[..2], &kept);
    }
}
