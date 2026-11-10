//! # e-means
//!
//! A Rust implementation of **e-means**, the mini-batch k-means algorithm of
//! Bessa et al., ["e-means: outperforming k-means (in one-tenth the
//! epochs)"](https://arxiv.org/abs/2611.09501) (Meta FAIR, 2026).
//!
//! e-means solves the k-means problem: given `n` vectors `x_i` and a number
//! of clusters `k`, find centroids `c_1 … c_k` that minimize the mean squared
//! distance from each vector to its nearest centroid,
//!
//! ```text
//! MSE(X, C) = (1/n) · Σ_i ‖x_i − c_a(i)‖²,    a(i) = argmin_j ‖x_i − c_j‖²
//! ```
//!
//! Lloyd's algorithm reads all the data before each update. e-means updates
//! after each mini-batch, with an exponential moving average. The paper
//! reports that it reaches a lower loss than 25 iterations of Lloyd's
//! algorithm in 1.2 to 3 epochs, with no hyperparameter to tune: scaling laws
//! fitted on many datasets set the batch size and the step size.
//!
//! ## Quick start
//!
//! ```
//! use std::num::NonZeroU32;
//! use emeans::{Emeans, RowMatrix};
//!
//! // 10,000 vectors of dimension 8, one vector per row.
//! let dim = 8;
//! let values: Vec<f32> = (0..10_000 * dim).map(|i| ((i * 7919) % 1013) as f32 / 1013.0).collect();
//! let data = RowMatrix::new(&values, dim)?;
//!
//! // 64 clusters; 5 epochs by default.
//! let fit = Emeans::new(NonZeroU32::new(64).unwrap()).fit(data)?;
//!
//! let centroids = fit.model.centroids();            // 64 × 8
//! let labels = fit.labels.expect("5 epochs >= 1");  // one per training row
//! let new = centroids.assign(data)?;                // labels and distances of any rows
//! println!("{}", fit.model.recipe());               // the hyperparameters that were used
//! println!("MSE = {}", new.mean_distance());
//! # assert_eq!((centroids.clusters().get(), labels.len()), (64, 10_000));
//! # Ok::<(), emeans::Error>(())
//! ```
//!
//! ## The algorithm
//!
//! Each cluster `j` keeps a running count `N_j` and a running sum `S_j`; its
//! centroid is `c_j = S_j / N_j`. For each batch, every vector goes to its
//! nearest centroid. Let `n_j` and `s_j` be the count and the sum of the
//! batch's vectors that go to cluster `j`. Then, with a step size `α`:
//!
//! ```text
//! N_j ← (1 − α) · N_j + α · n_j
//! S_j ← (1 − α) · S_j + α · s_j
//! ```
//!
//! A run with `T` epochs of `n` rows and batches of `B` rows:
//!
//! ```text
//!  init: k random rows ──► c_1 … c_k,  N_j = 2⁻¹⁶
//!   │
//!   ▼
//!  ┌─ for each epoch: a fresh random order of the rows ──────────────┐
//!  │   ┌─ for each batch of B rows ────────────────────────────────┐ │
//!  │   │  assign each row to its nearest centroid                  │ │
//!  │   │  EMA update of N_j, S_j with step α(t) (cosine schedule)  │ │
//!  │   │  every 5 updates: revive dead clusters by splitting       │ │
//!  │   └───────────────────────────────────────────────────────────┘ │
//!  └─────────────────────────────────────────────────────────────────┘
//!   │
//!   ▼
//!  final c_j = mean of the rows that the last epoch assigned to j
//! ```
//!
//! The parts, and where this crate configures them:
//!
//! | Part | Rule | Configuration |
//! |---|---|---|
//! | Batch size `B` | A fitted power law of `k`, `n` and `T`, rounded to a power of two | [`Batch`], [`BatchLaw`], [`THROUGHPUT_LAW`], [`QUALITY_LAW`] |
//! | Initial step `α₀` | A fitted power law of `k`, `B`, `n` and `T` | [`Step`], [`STEP_LAW`] |
//! | Step schedule | Cosine from `α₀` down to `α₀ / 10` | [`Recipe::alpha`], [`ALPHA_MIN_FACTOR`] |
//! | Dead-cluster revival | Every 5 updates, a cluster below `τ` times the mean count takes half of a donor drawn in proportion to its count | [`Revival`], [`Threshold`], [`REVIVAL_INTERVAL`] |
//! | Finalization | Each centroid becomes the exact mean of its rows in the last epoch | always on |
//!
//! With `B = n` and a constant `α = 1`, one update is one iteration of
//! Lloyd's algorithm. The paper shows that this corner of the
//! hyperparameter space is far from the best one.
//!
//! [`Emeans::recipe`] shows the resolved hyperparameters of a problem
//! without training.
//!
//! ## Data that does not fit in memory
//!
//! [`Emeans::begin`] returns a [`Session`] that the caller feeds with rows,
//! in chunks of any size. [`Emeans::fit`] is a loop over a session, so both
//! paths run the same code.
//!
//! ## Cosine similarity
//!
//! [`Metric::Spherical`] clusters vectors of norm 1 by direction. Prepare
//! the data with [`normalize_rows`].
//!
//! ## Numerical behavior
//!
//! - Running counts, centroids, batch sums, objectives and distances are
//!   all `f64`, for both element types `T` ([`f32`] and [`f64`]).
//! - The nearest-centroid search is exact: each label is the centroid with
//!   the smallest distance computed directly in `f64` (ties go to the lower
//!   index), and each distance is that `f64` value. The centroids are the
//!   ones the caller sees, rounded to `T`. For speed, the search first
//!   computes all scores with a matrix product (GEMM) in `T`. A rigorous
//!   bound on the rounding error of each score then leaves a few
//!   candidates, usually one or two, and only those get the direct `f64`
//!   computation.
//! - Inputs are checked: values must be finite, and squared norms must stay
//!   below a sixteenth of the largest value of `T`, so no distance or score
//!   can overflow. Errors name the first bad row.
//!
//! ## Reproducibility
//!
//! For a given configuration, seed and input, the results are bit-identical
//! for every number of threads, with or without the `parallel` feature, and
//! for every chunking of a [`Session`]'s input. Because the search is exact,
//! results do not depend on how the GEMM rounds either. The random
//! generators are a fixed algorithm (xoshiro256++ seeded by SplitMix64), not
//! a dependency that can change. Across platforms, results can differ in the
//! last bits only where the platform's math library rounds `pow`, `cos` or
//! `ln` differently (in the scaling laws, the step schedule and the revival
//! draws).
//!
//! ## Feature flags
//!
//! - `parallel` (default): run the search and the batch sums on all cores
//!   with [rayon](https://docs.rs/rayon). Use the global rayon pool, or
//!   install a pool to limit the threads.
//!
//! ## Relation to the reference implementation
//!
//! This crate follows the reference implementation in faiss
//! (`contrib/e_means`, MIT license) for every rule and constant: the
//! scaling-law coefficients, the rounding of the batch size, the epoch plan,
//! the schedule, the revival and the finalization. The differences are:
//!
//! - The state is `f64`, where the reference uses `f32`.
//! - The nearest-centroid search is exact (see above). The reference uses
//!   the rounded GEMM scores directly.
//! - When Equation 8 of the paper leaves the two halves of a revival split
//!   identical in the search (a donor at the origin, or a spherical donor
//!   whose direction the perturbation does not change), a second, small
//!   perturbation separates them. The reference leaves them identical, so
//!   the revived cluster stays dead.
//! - The random generators differ, so the same seed gives different (but
//!   equally distributed) random choices.
//! - Spherical clustering checks that rows have norm 1.
//! - The reference implementation's GPU kernels, low-precision formats and
//!   multi-process training are not implemented.
//!
//! ## Citation
//!
//! If you use e-means, cite the paper:
//!
//! ```bibtex
//! @misc{bessa2026emeans,
//!   title         = {e-means: outperforming k-means (in one-tenth the epochs)},
//!   author        = {Swann Bessa and Pierre-Emmanuel Mazaré and Gergely Szilvasy and
//!                    Hugo Touvron and David Picard and Matthijs Douze and Hervé Jégou},
//!   year          = {2026},
//!   eprint        = {2611.09501},
//!   archivePrefix = {arXiv},
//!   primaryClass  = {cs.LG},
//!   url           = {https://arxiv.org/abs/2611.09501}
//! }
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

mod config;
mod error;
mod fit;
mod kernel;
mod matrix;
mod model;
mod numeric;
mod parallel;
mod positions;
mod recipe;
mod rng;
mod rows;
mod scalar;
mod session;
mod state;
mod stats;

pub use config::{
    Alpha, Batch, BatchLaw, DEFAULT_SEED, Emeans, Epochs, Metric, Revival, Step, Threshold,
};
pub use error::Error;
pub use matrix::{RowMatrix, UNIT_NORM_TOLERANCE, normalize_rows};
pub use model::{Assignment, Centroids, Fit, Model};
pub use recipe::{
    ALPHA_MIN_FACTOR, AUTO_NO_REVIVAL_MAX_DIM, AUTO_QUALITY_MAX_DIM, BatchLawFit, BatchSource,
    INITIAL_MASS, MIN_ALPHA0, QUALITY_LAW, REVIVAL_INTERVAL, Recipe, SPLIT_EPSILON, STEP_LAW,
    StepLawFit, StepSource, THROUGHPUT_LAW,
};
pub use scalar::Scalar;
pub use session::{Session, Unfinished};
pub use stats::EpochStats;
