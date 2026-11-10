//! Compares e-means with Lloyd's algorithm on a synthetic Gaussian mixture.
//!
//! For each seed, both methods start from the same random rows. Lloyd's
//! algorithm runs 25 iterations (25 epochs); e-means runs 1 to 5 epochs. The
//! table shows the mean squared error on the training rows and on held-out
//! test rows, relative to Lloyd's algorithm after 25 iterations (lower is
//! better), as the median and range over the seeds.
//!
//! ```text
//! cargo run --release --example compare_with_lloyd -- [rows] [dim] [clusters] [seeds]
//! ```

use std::num::NonZeroU32;
use std::time::Instant;

use emeans::{Centroids, Emeans, Epochs, Metric, RowMatrix};

/// A small SplitMix64 generator, so the example needs no other crate.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1).
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal, by the Box–Muller transform.
    fn normal(&mut self) -> f64 {
        (-2.0 * self.unit().ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// Rows drawn from a mixture of `components` Gaussians with random means,
/// scales and weights.
fn gaussian_mixture(rows: usize, dim: usize, components: usize, rng: &mut SplitMix64) -> Vec<f32> {
    let means: Vec<f64> = (0..components * dim).map(|_| 4.0 * rng.normal()).collect();
    let scales: Vec<f64> = (0..components).map(|_| 0.3 + rng.unit()).collect();
    let weights: Vec<f64> = (0..components).map(|_| rng.unit() + 0.1).collect();
    let total: f64 = weights.iter().sum();
    let mut data = Vec::with_capacity(rows * dim);
    for _ in 0..rows {
        let mut pick = rng.unit() * total;
        let mut component = 0;
        while component + 1 < components && pick >= weights[component] {
            pick -= weights[component];
            component += 1;
        }
        for t in 0..dim {
            data.push((means[component * dim + t] + scales[component] * rng.normal()) as f32);
        }
    }
    data
}

/// Runs Lloyd's algorithm for `iterations` iterations. A cluster that loses
/// all its rows keeps its centroid.
fn lloyd(
    train: RowMatrix<'_, f32>,
    init: Vec<f32>,
    iterations: usize,
) -> Result<Vec<f32>, emeans::Error> {
    let dim = train.dim();
    let mut centroids = init;
    for _ in 0..iterations {
        let assignment =
            Centroids::from_vectors(RowMatrix::new(&centroids, dim)?, Metric::SquaredEuclidean)?
                .assign(train)?;
        let clusters = centroids.len() / dim;
        let mut sums = vec![0.0f64; clusters * dim];
        let mut counts = vec![0u64; clusters];
        for (row, &label) in train.iter().zip(&assignment.labels) {
            let label = label as usize;
            counts[label] += 1;
            for (sum, &value) in sums[label * dim..(label + 1) * dim].iter_mut().zip(row) {
                *sum += f64::from(value);
            }
        }
        for (cluster, &count) in counts.iter().enumerate() {
            if count > 0 {
                for t in 0..dim {
                    centroids[cluster * dim + t] = (sums[cluster * dim + t] / count as f64) as f32;
                }
            }
        }
    }
    Ok(centroids)
}

fn mse(centroids: &[f32], data: RowMatrix<'_, f32>) -> Result<f64, emeans::Error> {
    Ok(Centroids::from_vectors(
        RowMatrix::new(centroids, data.dim())?,
        Metric::SquaredEuclidean,
    )?
    .assign(data)?
    .mean_distance())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).map(|arg| arg.parse::<usize>());
    let rows = args.next().transpose()?.unwrap_or(200_000);
    let dim = args.next().transpose()?.unwrap_or(32);
    let clusters = args.next().transpose()?.unwrap_or(256);
    let seeds = args.next().transpose()?.unwrap_or(5) as u64;
    let budgets = [1.0, 2.0, 3.0, 5.0];

    let mut rng = SplitMix64(0);
    let values = gaussian_mixture(rows + rows / 5, dim, clusters / 2, &mut rng);
    let (train_values, test_values) = values.split_at(rows * dim);
    let train = RowMatrix::new(train_values, dim)?;
    let test = RowMatrix::new(test_values, dim)?;
    let k = NonZeroU32::new(u32::try_from(clusters)?).ok_or("clusters must be at least 1")?;
    println!(
        "{rows} training rows, {} test rows, dimension {dim}, {clusters} clusters, {seeds} seeds",
        rows / 5
    );

    // Ratios to Lloyd's algorithm with the same initial rows, per budget.
    let mut train_ratios = vec![Vec::new(); budgets.len()];
    let mut test_ratios = vec![Vec::new(); budgets.len()];
    let (mut lloyd_seconds, mut emeans_seconds) = (0.0, vec![0.0; budgets.len()]);
    for seed in 1..=seeds {
        let mut rng = SplitMix64(seed);
        let mut chosen = Vec::with_capacity(clusters);
        while chosen.len() < clusters {
            let row = rng.below(rows);
            if !chosen.contains(&row) {
                chosen.push(row);
            }
        }
        let init: Vec<f32> = chosen
            .iter()
            .flat_map(|&row| train.row(row).unwrap().iter().copied())
            .collect();

        let start = Instant::now();
        let reference = lloyd(train, init.clone(), 25)?;
        lloyd_seconds += start.elapsed().as_secs_f64();
        let (reference_train, reference_test) = (mse(&reference, train)?, mse(&reference, test)?);

        for (i, &epochs) in budgets.iter().enumerate() {
            let config = Emeans::new(k)
                .with_epochs(Epochs::new(epochs)?)
                .with_seed(seed);
            let start = Instant::now();
            let fit = config.fit_from(train, RowMatrix::new(&init, dim)?)?;
            emeans_seconds[i] += start.elapsed().as_secs_f64();
            let centroids = fit.model.centroids().as_slice();
            train_ratios[i].push(mse(centroids, train)? / reference_train);
            test_ratios[i].push(mse(centroids, test)? / reference_test);
        }
    }

    let median = |values: &[f64]| {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted[sorted.len() / 2]
    };
    let range = |values: &[f64]| {
        let low = values.iter().copied().fold(f64::INFINITY, f64::min);
        let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        format!("{low:.3}-{high:.3}")
    };
    println!(
        "{:<18} {:>6} {:>13} {:>13} {:>13} {:>13} {:>9}",
        "method", "epochs", "train median", "train range", "test median", "test range", "seconds"
    );
    println!(
        "{:<18} {:>6} {:>13.4} {:>13} {:>13.4} {:>13} {:>9.2}",
        "Lloyd (reference)",
        25,
        1.0,
        "-",
        1.0,
        "-",
        lloyd_seconds / seeds as f64
    );
    for (i, epochs) in budgets.iter().enumerate() {
        println!(
            "{:<18} {:>6} {:>13.4} {:>13} {:>13.4} {:>13} {:>9.2}",
            "e-means",
            epochs,
            median(&train_ratios[i]),
            range(&train_ratios[i]),
            median(&test_ratios[i]),
            range(&test_ratios[i]),
            emeans_seconds[i] / seeds as f64
        );
    }
    println!("MSE is relative to Lloyd's algorithm from the same initial rows (lower is better).");
    println!("Seconds are the mean wall time of one run, without warm-up: a rough guide only.");
    Ok(())
}
