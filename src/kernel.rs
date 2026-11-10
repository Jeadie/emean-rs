//! Nearest-centroid search, and the per-cluster sums and counts of a batch
//! (Algorithm 1 of the paper, lines 7 and 8).
//!
//! # What the search returns
//!
//! For each row `x`, the search returns the centroid `ĉ_j` that minimizes
//! the distance `t_j`, computed directly in `f64`:
//!
//! ```text
//! t_j = Σ_t (x_t − ĉ_jt)²        (squared Euclidean)
//! t_j = 2 − 2 · Σ_t x_t · ĉ_jt   (spherical)
//! ```
//!
//! Here `ĉ_j` is the centroid rounded to the element type `T`: the centroid
//! that the caller sees. Ties go to the lowest index. The result is exact:
//! it is the same as a search that computes every `t_j`. It depends only on
//! `x` and the `ĉ_j`, not on the matrix-product code, the shift below or the
//! number of threads.
//!
//! # How the search is fast and exact
//!
//! Computing every `t_j` costs `k · d` operations in `f64` per row. The
//! search instead computes, for a tile of rows and a block of centroids,
//!
//! ```text
//! S_j = h_j − ⟨x', c'_j⟩      x' = x − s,  c'_j = ĉ_j − s,  h_j = ½ ‖c'_j‖²  (squared Euclidean)
//! S_j =     − ⟨x,  ĉ_j⟩                                                    (spherical)
//! ```
//!
//! in `T`, with one matrix product (GEMM). Without rounding,
//! `t_j = ‖x'‖² + 2 · S_j` (squared Euclidean) or `t_j = 2 + 2 · S_j`
//! (spherical), so the order of the `S_j` is the order of the `t_j`.
//!
//! With rounding, each `S_j` has an error. With `r = ‖x'‖`, `n_j = ‖c'_j‖`
//! and the unit roundoff `u` of `T`, the standard error bound of a dot
//! product of length `d` (Higham, *Accuracy and Stability of Numerical
//! Algorithms*, 2002, §3.1) gives, for every summation order and with or
//! without fused multiply-add:
//!
//! ```text
//! |base + 2 · S_j − t_j| ≤ κ · (r + n_j)² + η,    κ = 2 · (2γ_d + 5u + 5γ'_{d+2} + 6u')
//! ```
//!
//! where `γ_d = d·u / (1 − d·u)`, `u'` and `γ'` are the same quantities for
//! `f64`, and `η` covers underflow. The bound also includes the rounding of
//! `x'`, `c'_j` and `h_j` to `T`, and the rounding of `t_j` itself. The
//! leading factor 2 is a safety margin.
//!
//! So `t_j` lies in `[lo_j, hi_j] = 2·S_j ± (κ·(r + n_j)² + η)`, shifted by
//! the same `base` for every `j`. A centroid can be the nearest only if
//! `lo_j ≤ min_i hi_i`. The search keeps these candidates, usually one or
//! two, and computes `t_j` directly for each of them.
//!
//! The shift `s` is the coordinate-wise median of the centroids when the
//! search is built. It keeps `r` and `n_j` small for data far from the
//! origin, so that the intervals stay narrow and the candidates few. It
//! changes only the speed, never the result.
//!
//! # Determinism
//!
//! Each row's label and distance depend only on that row and the centroids.
//! Each cluster's sum adds its rows in row order. So results do not depend
//! on the number of threads, nor on how a batch is cut into pieces.

use crate::config::Metric;
use crate::parallel;
use crate::rows::RowSource;
use crate::scalar::{Element, Scalar};

/// Rows per tile of the search: each tile is one unit of parallel work.
const ROW_TILE: usize = 128;

/// Centroids per GEMM call. The inner-product buffer of one tile holds
/// `ROW_TILE × CENTROID_BLOCK` values.
const CENTROID_BLOCK: usize = 256;

/// A row's candidate list is pruned when it grows past this length.
const CANDIDATE_PRUNE_LEN: usize = 32;

/// Below this many rows, a batch's sums are added on one thread, because
/// sorting rows by cluster would cost more than it saves.
const PARALLEL_SUMS_MIN_ROWS: usize = 2048;

/// The bound `κ` on the error of `2 · S_j` relative to `(r + n_j)²` (see the
/// module documentation), or `None` when `d` is so large that the bound does
/// not apply.
fn error_factor<T: Scalar>(dim: usize) -> Option<f64> {
    let u = T::unit_roundoff();
    let u64 = f64::unit_roundoff();
    let gamma = |n: usize, u: f64| {
        let nu = n as f64 * u;
        (nu < 0.5).then(|| nu / (1.0 - nu))
    };
    let gamma_d = gamma(dim, u)?;
    let gamma_f64 = gamma(dim + 2, u64)?;
    Some(2.0 * (2.0 * gamma_d + 5.0 * u + 5.0 * gamma_f64 + 6.0 * u64))
}

/// Centroids prepared for the search.
#[derive(Clone, Debug)]
pub(crate) struct SearchCentroids<T> {
    clusters: usize,
    dim: usize,
    metric: Metric,
    /// `k × d` centroids `ĉ_j`, rounded to `T`: the centroids that distances
    /// are measured to.
    centroids: Vec<T>,
    /// `k × d` rows `c'_j = ĉ_j − s`, rounded to `T` (squared Euclidean only).
    shifted: Vec<T>,
    /// `h_j = ½ ‖c'_j‖²`, rounded to `T` (squared Euclidean only).
    half_norms: Vec<T>,
    /// `‖c'_j‖` (`‖ĉ_j‖` for spherical).
    norms: Vec<f64>,
    /// The shift `s` (squared Euclidean only).
    shift: Vec<f64>,
    /// `κ`, or `None` to compute every distance directly.
    error_factor: Option<f64>,
    /// `η`: an absolute bound that covers underflow in the search.
    underflow: f64,
}

impl<T: Scalar> SearchCentroids<T> {
    /// Prepares `positions` (`k × d`, row-major) for the search, with the
    /// coordinate-wise median of `positions` as the shift.
    ///
    /// For [`Metric::Spherical`], `positions` must already have norm 1 (or
    /// 0).
    pub(crate) fn new(positions: &[f64], dim: usize, metric: Metric) -> Self {
        let clusters = positions.len() / dim;
        let euclidean = metric == Metric::SquaredEuclidean;
        let zero = T::from_f64(0.0);
        let mut prepared = Self {
            clusters,
            dim,
            metric,
            centroids: vec![zero; clusters * dim],
            shifted: if euclidean {
                vec![zero; clusters * dim]
            } else {
                Vec::new()
            },
            half_norms: if euclidean {
                vec![zero; clusters]
            } else {
                Vec::new()
            },
            norms: vec![0.0; clusters],
            shift: if euclidean {
                crate::numeric::column_medians(positions, dim)
            } else {
                Vec::new()
            },
            error_factor: error_factor::<T>(dim),
            // Each score takes about 2d + 4 operations, and an operation
            // that underflows has an absolute error of at most half the
            // smallest positive value of T; `2 · S_j` doubles the error.
            underflow: 8.0 * (dim + 4) as f64 * T::smallest_positive(),
        };
        prepared.load(positions);
        prepared
    }

    /// Replaces the centroids with `positions`, keeping the shift.
    pub(crate) fn load(&mut self, positions: &[f64]) {
        debug_assert_eq!(positions.len(), self.clusters * self.dim);
        for (rounded, &value) in self.centroids.iter_mut().zip(positions) {
            *rounded = T::from_f64(value);
        }
        let dim = self.dim;
        for cluster in 0..self.clusters {
            let range = cluster * dim..(cluster + 1) * dim;
            let mut squared = 0.0f64;
            match self.metric {
                Metric::SquaredEuclidean => {
                    let rows = self.shifted[range.clone()]
                        .iter_mut()
                        .zip(&self.centroids[range]);
                    for ((out, &rounded), &shift) in rows.zip(&self.shift) {
                        *out = T::from_f64(rounded.to_f64() - shift);
                        let value = out.to_f64();
                        squared += value * value;
                    }
                    self.half_norms[cluster] = T::from_f64(0.5 * squared);
                }
                Metric::Spherical => {
                    for &value in &self.centroids[range] {
                        squared += value.to_f64() * value.to_f64();
                    }
                }
            }
            self.norms[cluster] = squared.sqrt();
        }
    }

    /// The metric that distances are measured in.
    pub(crate) fn metric(&self) -> Metric {
        self.metric
    }

    /// The dimension `d` of the centroids.
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    /// The `k × d` centroids `ĉ_j`, rounded to `T`.
    pub(crate) fn centroids(&self) -> &[T] {
        &self.centroids
    }

    /// The `k × d` centroids `ĉ_j`, rounded to `T`.
    pub(crate) fn into_centroids(self) -> Vec<T> {
        self.centroids
    }

    /// Writes the nearest centroid of each row of `source` to `labels`, and
    /// the distance `t_j` to it to `distances`.
    pub(crate) fn assign<S: RowSource<T>>(
        &self,
        source: &S,
        labels: &mut [u32],
        distances: &mut [f64],
    ) {
        let rows = source.len();
        assert_eq!(
            source.dim(),
            self.dim,
            "rows and centroids differ in dimension"
        );
        assert!(labels.len() == rows && distances.len() == rows);
        let dim = self.dim;
        parallel::for_each_chunk_pair(
            labels,
            distances,
            ROW_TILE,
            || TileScratch::new(dim),
            |scratch, tile, labels, distances| {
                let start = tile * ROW_TILE;
                match self.error_factor {
                    Some(factor) => {
                        self.assign_tile(source, start, factor, scratch, labels, distances)
                    }
                    None => self.assign_exhaustively(source, start, labels, distances),
                }
            },
        );
    }

    /// The search of the module documentation, for the rows of one tile.
    fn assign_tile<S: RowSource<T>>(
        &self,
        source: &S,
        start: usize,
        factor: f64,
        scratch: &mut TileScratch<T>,
        labels: &mut [u32],
        distances: &mut [f64],
    ) {
        let rows = labels.len();
        let dim = self.dim;
        let euclidean = self.metric == Metric::SquaredEuclidean;

        // Pack the tile's rows x' (shifted, for squared Euclidean) and their
        // norms r.
        let tile = &mut scratch.tile[..rows * dim];
        for (i, packed) in tile.chunks_exact_mut(dim).enumerate() {
            let row = source.row(start + i);
            if euclidean {
                for ((out, &value), &shift) in packed.iter_mut().zip(row).zip(&self.shift) {
                    *out = T::from_f64(value.to_f64() - shift);
                }
            } else {
                packed.copy_from_slice(row);
            }
            scratch.row_norms[i] = packed
                .iter()
                .map(|v| v.to_f64() * v.to_f64())
                .sum::<f64>()
                .sqrt();
        }

        // Collect, for each row, the centroids whose interval [lo_j, hi_j]
        // reaches below the lowest hi_i so far.
        let best_high = &mut scratch.best_high[..rows];
        best_high.fill(f64::INFINITY);
        for candidates in &mut scratch.candidates[..rows] {
            candidates.clear();
        }
        let search = if euclidean {
            &self.shifted
        } else {
            &self.centroids
        };
        for block_start in (0..self.clusters).step_by(CENTROID_BLOCK) {
            let block = CENTROID_BLOCK.min(self.clusters - block_start);
            let products = &mut scratch.products[..rows * block];
            let block_rows = &search[block_start * dim..(block_start + block) * dim];
            T::gemm_abt(rows, block, dim, tile, block_rows, products);
            for (i, row_products) in products.chunks_exact(block).enumerate() {
                let r = scratch.row_norms[i];
                let best = &mut best_high[i];
                let candidates = &mut scratch.candidates[i];
                for (j, &product) in row_products.iter().enumerate() {
                    let cluster = block_start + j;
                    let score = if euclidean {
                        self.half_norms[cluster] - product
                    } else {
                        T::from_f64(0.0) - product
                    };
                    let twice = 2.0 * score.to_f64();
                    let reach = r + self.norms[cluster];
                    let slack = factor * reach * reach + self.underflow;
                    let high = twice + slack;
                    let low = twice - slack;
                    if high < *best {
                        *best = high;
                    }
                    if may_be_nearest(low, *best) {
                        candidates.push((cluster as u32, low));
                    }
                }
                if candidates.len() > CANDIDATE_PRUNE_LEN {
                    let best = *best;
                    candidates.retain(|&(_, low)| may_be_nearest(low, best));
                }
            }
        }

        // Compute t_j directly for the remaining candidates, in index order.
        for (i, (label, distance)) in labels.iter_mut().zip(distances.iter_mut()).enumerate() {
            let row = source.row(start + i);
            let best = best_high[i];
            let mut nearest: Option<(u32, f64)> = None;
            for &(cluster, low) in &scratch.candidates[i] {
                if may_be_nearest(low, best) {
                    let t = self.distance(row, cluster as usize);
                    if nearest.is_none_or(|(_, nearest)| t < nearest) {
                        nearest = Some((cluster, t));
                    }
                }
            }
            (*label, *distance) = nearest.expect("the nearest centroid is always a candidate");
        }
    }

    /// Computes every distance directly. Used when `d` is too large for the
    /// error bound.
    fn assign_exhaustively<S: RowSource<T>>(
        &self,
        source: &S,
        start: usize,
        labels: &mut [u32],
        distances: &mut [f64],
    ) {
        for (i, (label, distance)) in labels.iter_mut().zip(distances.iter_mut()).enumerate() {
            let row = source.row(start + i);
            let mut nearest: Option<(u32, f64)> = None;
            for cluster in 0..self.clusters {
                let t = self.distance(row, cluster);
                if nearest.is_none_or(|(_, nearest)| t < nearest) {
                    nearest = Some((cluster as u32, t));
                }
            }
            (*label, *distance) = nearest.expect("there is at least one centroid");
        }
    }

    /// The distance `t_j` from `row` to centroid `cluster`, computed
    /// directly in `f64`, adding the terms in order.
    fn distance(&self, row: &[T], cluster: usize) -> f64 {
        let centroid = &self.centroids[cluster * self.dim..(cluster + 1) * self.dim];
        match self.metric {
            Metric::SquaredEuclidean => row
                .iter()
                .zip(centroid)
                .map(|(&x, &c)| {
                    let difference = x.to_f64() - c.to_f64();
                    difference * difference
                })
                .sum(),
            Metric::Spherical => {
                let product: f64 = row
                    .iter()
                    .zip(centroid)
                    .map(|(&x, &c)| x.to_f64() * c.to_f64())
                    .sum();
                2.0 - 2.0 * product
            }
        }
    }
}

/// Whether a centroid whose distance is at least `low` can be nearer than
/// one whose distance is at most `best`. A NaN bound keeps the centroid: the
/// magnitude check on inputs rules NaN out, and the direct computation would
/// still give the right answer.
fn may_be_nearest(low: f64, best: f64) -> bool {
    low <= best || low.is_nan()
}

/// Per-worker buffers of the search.
struct TileScratch<T> {
    tile: Vec<T>,
    products: Vec<T>,
    row_norms: Vec<f64>,
    best_high: Vec<f64>,
    candidates: Vec<Vec<(u32, f64)>>,
}

impl<T: Scalar> TileScratch<T> {
    fn new(dim: usize) -> Self {
        Self {
            tile: vec![T::from_f64(0.0); ROW_TILE * dim],
            products: vec![T::from_f64(0.0); ROW_TILE * CENTROID_BLOCK],
            row_norms: vec![0.0; ROW_TILE],
            best_high: vec![f64::INFINITY; ROW_TILE],
            candidates: vec![Vec::new(); ROW_TILE],
        }
    }
}

/// Adds each row of `source` to the sum and count of its cluster.
///
/// `sums` is `k × d`, row-major. Each cluster adds its rows in row order,
/// whichever path runs, so the result is bit-identical on every path.
pub(crate) fn accumulate<T: Scalar, S: RowSource<T>>(
    source: &S,
    labels: &[u32],
    dim: usize,
    sums: &mut [f64],
    counts: &mut [u64],
) {
    assert_eq!(source.dim(), dim, "rows and sums differ in dimension");
    assert_eq!(labels.len(), source.len());
    if !parallel::enabled() || labels.len() < PARALLEL_SUMS_MIN_ROWS {
        for (index, &label) in labels.iter().enumerate() {
            let cluster = label as usize;
            counts[cluster] += 1;
            let sum = &mut sums[cluster * dim..(cluster + 1) * dim];
            for (s, &value) in sum.iter_mut().zip(source.row(index)) {
                *s += value.to_f64();
            }
        }
        return;
    }

    // Group the rows by cluster with a stable counting sort, then let each
    // cluster add its own rows.
    let clusters = counts.len();
    let mut starts = vec![0usize; clusters + 1];
    for &label in labels {
        starts[label as usize + 1] += 1;
    }
    for cluster in 0..clusters {
        starts[cluster + 1] += starts[cluster];
    }
    let mut members = vec![0usize; labels.len()];
    let mut cursor = starts[..clusters].to_vec();
    for (index, &label) in labels.iter().enumerate() {
        let slot = &mut cursor[label as usize];
        members[*slot] = index;
        *slot += 1;
    }
    for (cluster, count) in counts.iter_mut().enumerate() {
        *count += (starts[cluster + 1] - starts[cluster]) as u64;
    }
    parallel::for_each_row_mut(sums, dim, |cluster, sum| {
        for &index in &members[starts[cluster]..starts[cluster + 1]] {
            for (s, &value) in sum.iter_mut().zip(source.row(index)) {
                *s += value.to_f64();
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::{RowMatrix, RowNorm};
    use crate::rng::{Rng, Stream};
    use crate::rows::{CheckedRows, Window};

    fn uniform(rows: usize, dim: usize, low: f64, high: f64, seed: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed, Stream::Epoch(0));
        (0..rows * dim)
            .map(|_| low + (high - low) * rng.open_unit())
            .collect()
    }

    fn rounded<T: Scalar>(values: &[f64]) -> Vec<T> {
        values.iter().map(|&v| T::from_f64(v)).collect()
    }

    /// The exhaustive search that the module documentation defines: every
    /// distance in `f64`, ties to the lowest index.
    fn exhaustive<T: Scalar>(row: &[T], centroids: &[T], dim: usize, metric: Metric) -> (u32, f64) {
        let mut best = (0, f64::INFINITY);
        for (j, c) in centroids.chunks_exact(dim).enumerate() {
            let t = match metric {
                Metric::SquaredEuclidean => row
                    .iter()
                    .zip(c)
                    // Not `powi`: its precision is unspecified and can
                    // differ between platforms.
                    .map(|(&x, &y)| {
                        let difference = x.to_f64() - y.to_f64();
                        difference * difference
                    })
                    .sum(),
                Metric::Spherical => {
                    2.0 - 2.0
                        * row
                            .iter()
                            .zip(c)
                            .map(|(&x, &y)| x.to_f64() * y.to_f64())
                            .sum::<f64>()
                }
            };
            if t < best.1 {
                best = (j as u32, t);
            }
        }
        best
    }

    /// Checks that the search returns exactly what the exhaustive search does.
    fn check_exact<T: Scalar>(
        data: &[f64],
        positions: &[f64],
        dim: usize,
        metric: Metric,
    ) -> SearchCentroids<T> {
        let rows: Vec<T> = rounded(data);
        let search = SearchCentroids::<T>::new(positions, dim, metric);
        let matrix = RowMatrix::new(&rows, dim).unwrap();
        let checked = CheckedRows::new(matrix, RowNorm::for_metric(metric)).unwrap();
        let mut labels = vec![u32::MAX; matrix.rows()];
        let mut distances = vec![f64::NAN; matrix.rows()];
        search.assign(&checked, &mut labels, &mut distances);
        for (i, row) in matrix.iter().enumerate() {
            let expected = exhaustive(row, search.centroids(), dim, metric);
            assert_eq!(
                (labels[i], distances[i].to_bits()),
                (expected.0, expected.1.to_bits()),
                "row {i}"
            );
        }
        search
    }

    #[test]
    fn search_is_exact_across_tiles_and_blocks() {
        let (dim, clusters) = (7, CENTROID_BLOCK + 41);
        let data = uniform(ROW_TILE * 2 + 17, dim, -2.0, 2.0, 1);
        let centroids = uniform(clusters, dim, -2.0, 2.0, 2);
        check_exact::<f32>(&data, &centroids, dim, Metric::SquaredEuclidean);
        check_exact::<f64>(&data, &centroids, dim, Metric::SquaredEuclidean);
    }

    #[test]
    fn search_is_exact_far_from_the_origin() {
        let (dim, clusters) = (5, 40);
        let data = uniform(300, dim, 1e6 - 2.0, 1e6 + 2.0, 3);
        let centroids = uniform(clusters, dim, 1e6 - 2.0, 1e6 + 2.0, 4);
        check_exact::<f32>(&data, &centroids, dim, Metric::SquaredEuclidean);
        check_exact::<f64>(&data, &centroids, dim, Metric::SquaredEuclidean);
    }

    #[test]
    fn search_is_exact_with_a_far_outlier_centroid() {
        // Two close centroids inside a tiny cloud, and one far away that no
        // row is nearest to. A mean-based shift would sit near the far
        // centroid and leave f32 no precision to tell the two apart.
        let dim = 4;
        let data = uniform(5000, dim, 1e-3, 2e-3, 5);
        for far in [10.0, 1000.0] {
            let mut centroids = vec![1.5e-3 * (1.0 - 1e-4); dim];
            centroids.extend(vec![1.5e-3 * (1.0 + 1e-4); dim]);
            centroids.extend(vec![far; dim]);
            check_exact::<f32>(&data, &centroids, dim, Metric::SquaredEuclidean);
        }
    }

    #[test]
    fn search_is_exact_with_many_near_ties() {
        // 600 centroids on a line, 1e-6 apart: every row has many candidates,
        // which also exercises the pruning of candidate lists.
        let dim = 3;
        let centroids: Vec<f64> = (0..600)
            .flat_map(|j| [1.0 + j as f64 * 1e-6, 2.0, 3.0])
            .collect();
        let data = uniform(500, dim, 0.999, 1.001, 6);
        let data: Vec<f64> = data
            .chunks_exact(dim)
            .flat_map(|r| [r[0], 2.0, 3.0])
            .collect();
        check_exact::<f32>(&data, &centroids, dim, Metric::SquaredEuclidean);
        check_exact::<f64>(&data, &centroids, dim, Metric::SquaredEuclidean);
    }

    #[test]
    fn spherical_search_is_exact() {
        let dim = 9;
        let mut data = uniform(300, dim, -1.0, 1.0, 7);
        let mut centroids = uniform(300, dim, -1.0, 1.0, 8);
        for row in data
            .chunks_exact_mut(dim)
            .chain(centroids.chunks_exact_mut(dim))
        {
            crate::numeric::normalize_in_place(row);
        }
        // A zero centroid, as a revived or empty spherical cluster can be.
        centroids[..dim].fill(0.0);
        check_exact::<f32>(&data, &centroids, dim, Metric::Spherical);
        check_exact::<f64>(&data, &centroids, dim, Metric::Spherical);
    }

    #[test]
    fn search_is_exact_for_tiny_magnitudes() {
        // Products of these values underflow in f32.
        let dim = 6;
        let data = uniform(200, dim, 0.0, 1e-20, 9);
        let centroids = uniform(20, dim, 0.0, 1e-20, 10);
        check_exact::<f32>(&data, &centroids, dim, Metric::SquaredEuclidean);
    }

    #[test]
    fn the_result_does_not_depend_on_the_shift() {
        let dim = 8;
        let data = uniform(400, dim, -1.0, 1.0, 11);
        let centroids = uniform(64, dim, -1.0, 1.0, 12);
        let mut search = check_exact::<f32>(&data, &centroids, dim, Metric::SquaredEuclidean);
        for shift in [0.0, 50.0, -3e4] {
            search.shift = vec![shift; dim];
            search.load(&centroids);
            let rows: Vec<f32> = rounded(&data);
            let matrix = RowMatrix::new(&rows, dim).unwrap();
            let checked = CheckedRows::new(matrix, RowNorm::Any).unwrap();
            let mut labels = vec![0; 400];
            let mut distances = vec![0.0; 400];
            search.assign(&checked, &mut labels, &mut distances);
            for (i, row) in matrix.iter().enumerate() {
                assert_eq!(
                    labels[i],
                    exhaustive(row, search.centroids(), dim, Metric::SquaredEuclidean).0,
                    "shift {shift}"
                );
            }
        }
    }

    #[test]
    fn ties_go_to_the_lowest_index() {
        let search = SearchCentroids::<f64>::new(&[1.0, -1.0, 1.0], 1, Metric::SquaredEuclidean);
        let mut labels = vec![9; 1];
        let mut distances = vec![0.0; 1];
        let rows = CheckedRows::new(RowMatrix::new(&[0.0f64], 1).unwrap(), RowNorm::Any).unwrap();
        search.assign(&rows, &mut labels, &mut distances);
        assert_eq!((labels[0], distances[0]), (0, 1.0));
    }

    #[test]
    fn accumulate_paths_agree_bit_for_bit() {
        let (rows, dim, clusters) = (PARALLEL_SUMS_MIN_ROWS * 2 + 3, 5, 13);
        let data = uniform(rows, dim, -1.7, 2.3, 4);
        let matrix = CheckedRows::new(RowMatrix::new(&data, dim).unwrap(), RowNorm::Any).unwrap();
        let mut rng = Rng::new(8, Stream::Revival);
        let labels: Vec<u32> = (0..rows)
            .map(|_| rng.below(clusters as u64) as u32)
            .collect();

        let mut sums = vec![0.0; clusters * dim];
        let mut counts = vec![0; clusters];
        accumulate(&matrix, &labels, dim, &mut sums, &mut counts);

        // Small pieces take the sequential path; the large batch takes the
        // parallel one when the `parallel` feature is on.
        let mut piece_sums = vec![0.0; clusters * dim];
        let mut piece_counts = vec![0; clusters];
        for start in (0..rows).step_by(100) {
            let len = 100.min(rows - start);
            let window = Window::new(&matrix, start, len);
            accumulate(
                &window,
                &labels[start..start + len],
                dim,
                &mut piece_sums,
                &mut piece_counts,
            );
        }
        assert_eq!(counts, piece_counts);
        let bits = |v: &[f64]| v.iter().map(|s| s.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&sums), bits(&piece_sums));
        assert_eq!(counts.iter().sum::<u64>(), rows as u64);
    }
}
