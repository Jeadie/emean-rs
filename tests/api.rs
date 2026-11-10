//! Tests of the public API: determinism, input checks, sessions and models.

use std::num::{NonZeroU32, NonZeroUsize};

use emeans::{Batch, Centroids, Emeans, Epochs, Error, Metric, Model, Revival, RowMatrix};

/// A small deterministic generator for test data (SplitMix64).
struct TestRng(u64);

impl TestRng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `rows` points: `centers` well-separated centers on a scaled grid, plus
/// uniform noise of half-width `noise`.
fn blobs(rows: usize, dim: usize, centers: usize, noise: f64, seed: u64) -> Vec<f32> {
    let mut rng = TestRng(seed);
    let mut data = Vec::with_capacity(rows * dim);
    for i in 0..rows {
        let center = i % centers;
        for t in 0..dim {
            let middle = if t == center % dim {
                20.0 * (1 + center / dim) as f64
            } else {
                0.0
            };
            data.push((middle + noise * (2.0 * rng.next() - 1.0)) as f32);
        }
    }
    data
}

fn k(clusters: u32) -> NonZeroU32 {
    NonZeroU32::new(clusters).unwrap()
}

fn bits<T: emeans::Scalar + Copy + Into<f64>>(model: &Model<T>) -> Vec<u64> {
    model
        .centroids()
        .as_slice()
        .iter()
        .map(|&v| v.into().to_bits())
        .collect()
}

#[test]
fn results_do_not_depend_on_the_thread_count() {
    let dim = 24;
    let values = blobs(20_000, dim, 40, 3.0, 1);
    let data = RowMatrix::new(&values, dim).unwrap();
    // A large batch also exercises the parallel computation of batch sums.
    let config = Emeans::new(k(50))
        .with_batch(Batch::Fixed(NonZeroUsize::new(4096).unwrap()))
        .with_epochs(Epochs::new(2.0).unwrap());
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| config.fit(data).unwrap())
    };
    let one = run(1);
    for threads in [2, 3, 8] {
        let many = run(threads);
        assert_eq!(bits(&many.model), bits(&one.model), "{threads} threads");
        assert_eq!(many.labels, one.labels, "{threads} threads");
        assert_eq!(
            many.model.history(),
            one.model.history(),
            "{threads} threads"
        );
    }
}

#[test]
fn equal_seeds_reproduce_and_different_seeds_differ() {
    let values = blobs(3000, 6, 12, 4.0, 2);
    let data = RowMatrix::new(&values, 6).unwrap();
    let config = Emeans::new(k(12));
    assert_eq!(
        bits(&config.fit(data).unwrap().model),
        bits(&config.fit(data).unwrap().model)
    );
    assert_ne!(
        bits(&config.fit(data).unwrap().model),
        bits(&config.with_seed(99).fit(data).unwrap().model)
    );
}

#[test]
fn recovers_well_separated_clusters() {
    let dim = 4;
    let values = blobs(4000, dim, 8, 1.0, 3);
    let data = RowMatrix::new(&values, dim).unwrap();
    // Row i belongs to center i % 8, so the first 8 rows start one centroid in
    // each cluster. From a random start, k-means can stop in a local minimum
    // that leaves two clusters under one centroid.
    let init = RowMatrix::new(&values[..8 * dim], dim).unwrap();
    let fit = Emeans::new(k(8)).fit_from(data, init).unwrap();
    // Uniform noise of half-width 1 has a variance of 1/3 per coordinate.
    let mse = fit.model.assign(data).unwrap().mean_distance();
    assert!((mse - 4.0 / 3.0).abs() < 0.1, "MSE {mse}");
    // Each center's rows share one label, and different centers have different labels.
    let labels = fit.labels.unwrap();
    let mut label_of_center = [u32::MAX; 8];
    for (row, &label) in labels.iter().enumerate() {
        let center = row % 8;
        if label_of_center[center] == u32::MAX {
            label_of_center[center] = label;
        }
        assert_eq!(label, label_of_center[center], "row {row}");
    }
    let mut distinct = label_of_center.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), 8);
}

#[test]
fn recipe_is_the_one_that_training_uses() {
    let values = blobs(5000, 20, 10, 2.0, 4);
    let data = RowMatrix::new(&values, 20).unwrap();
    let config = Emeans::new(k(30)).with_epochs(Epochs::new(1.7).unwrap());
    let fit = config.fit(data).unwrap();
    assert_eq!(&config.recipe(5000, 20).unwrap(), fit.model.recipe());
    let history = fit.model.history();
    assert_eq!(history.len() as u64, fit.model.recipe().epoch_count());
    for (epoch, stats) in history.iter().enumerate() {
        assert_eq!(stats.epoch, epoch as u64);
        assert_eq!(
            Some(stats.rows),
            fit.model.recipe().rows_in_epoch(epoch as u64)
        );
        assert_eq!(
            Some(stats.updates),
            fit.model.recipe().updates_in_epoch(epoch as u64)
        );
        assert!(stats.imbalance >= 1.0 && stats.imbalance <= 30.0);
        assert!(stats.objective > 0.0);
    }
}

#[test]
fn every_row_its_own_cluster_when_k_equals_n() {
    let values: Vec<f64> = (0..20).map(|i| (i * i) as f64).collect();
    let data = RowMatrix::new(&values, 2).unwrap();
    let fit = Emeans::new(k(10)).fit(data).unwrap();
    assert_eq!(fit.model.centroids().as_slice(), values.as_slice());
    assert_eq!(fit.labels.unwrap(), (0..10).collect::<Vec<u32>>());
    assert_eq!(fit.model.assign(data).unwrap().mean_distance(), 0.0);
}

#[test]
fn budgets_below_one_epoch_give_no_labels() {
    let values = blobs(1000, 3, 4, 1.0, 5);
    let data = RowMatrix::new(&values, 3).unwrap();
    let fit = Emeans::new(k(4))
        .with_epochs(Epochs::new(0.5).unwrap())
        .fit(data)
        .unwrap();
    assert_eq!(fit.labels, None);
    assert_eq!(fit.model.history()[0].rows, 500);
}

#[test]
fn fit_checks_its_inputs() {
    let values = blobs(100, 3, 4, 1.0, 6);
    let data = RowMatrix::new(&values, 3).unwrap();
    assert_eq!(
        Emeans::new(k(101)).fit(data).unwrap_err(),
        Error::TooFewRows {
            rows: 100,
            clusters: 101
        }
    );

    let mut bad = values.clone();
    bad[3 * 17 + 2] = f32::NAN;
    let bad = RowMatrix::new(&bad, 3).unwrap();
    assert_eq!(
        Emeans::new(k(4)).fit(bad).unwrap_err(),
        Error::NonFinite { row: 17, column: 2 }
    );

    let spherical = Emeans::new(k(4)).with_metric(Metric::Spherical);
    assert!(matches!(
        spherical.fit(data),
        Err(Error::NotUnitNorm { row: 0, .. })
    ));

    let init = RowMatrix::new(&values[..9], 3).unwrap();
    assert_eq!(
        Emeans::new(k(4)).fit_from(data, init).unwrap_err(),
        Error::CentroidCountMismatch {
            expected: 4,
            actual: 3
        }
    );
    let init = RowMatrix::new(&values[..8], 2).unwrap();
    assert_eq!(
        Emeans::new(k(4)).fit_from(data, init).unwrap_err(),
        Error::DimensionMismatch {
            expected: 3,
            actual: 2
        }
    );
}

#[test]
fn errors_leave_a_session_unchanged() {
    let dim = 5;
    let values = blobs(2000, dim, 10, 2.0, 7);
    let config = Emeans::new(k(10))
        .with_epochs(Epochs::new(2.0).unwrap())
        .with_revival(Revival::Auto);
    let init = RowMatrix::new(&values[..10 * dim], dim).unwrap();

    let run = |inject_errors: bool| {
        let mut session = config.begin(2000, init).unwrap();
        for _ in 0..2 {
            for (index, chunk) in values.chunks(dim * 300).enumerate() {
                if inject_errors && index == 2 {
                    let mut poisoned = chunk.to_vec();
                    poisoned[dim * 5 + 1] = f32::INFINITY;
                    let poisoned = RowMatrix::new(&poisoned, dim).unwrap();
                    assert_eq!(
                        session.feed(poisoned),
                        Err(Error::NonFinite { row: 5, column: 1 })
                    );
                    let narrow = RowMatrix::new(&chunk[..dim * 4], dim - 1).unwrap();
                    assert!(matches!(
                        session.feed(narrow),
                        Err(Error::DimensionMismatch { .. })
                    ));
                    let mut short = vec![0; 3];
                    let rows = RowMatrix::new(chunk, dim).unwrap();
                    assert!(matches!(
                        session.feed_labeled(rows, &mut short),
                        Err(Error::BufferLength { .. })
                    ));
                }
                session.feed(RowMatrix::new(chunk, dim).unwrap()).unwrap();
            }
        }
        session.finish().unwrap()
    };
    let clean = run(false);
    let after_errors = run(true);
    assert_eq!(bits(&after_errors), bits(&clean));
    assert_eq!(after_errors.history(), clean.history());
}

#[test]
fn a_session_enforces_its_row_budget() {
    let dim = 2;
    let values = blobs(100, dim, 4, 1.0, 8);
    let config = Emeans::new(k(4)).with_epochs(Epochs::new(1.5).unwrap());
    let init = RowMatrix::new(&values[..4 * dim], dim).unwrap();
    let mut session = config.begin(100, init).unwrap();
    assert_eq!(session.remaining_rows(), 150);
    let rows = RowMatrix::new(&values, dim).unwrap();
    session.feed(rows).unwrap();
    assert_eq!(session.epoch(), 1, "the first epoch has 50 rows");
    assert!(!session.is_complete());
    assert_eq!(
        session.feed(rows).unwrap_err(),
        Error::BudgetExceeded {
            remaining: 50,
            offered: 100
        }
    );
    let half = RowMatrix::new(&values[..50 * dim], dim).unwrap();
    session.feed(half).unwrap();
    assert!(session.is_complete());
    assert_eq!(session.rows_to_next_update(), 0);
    assert_eq!(session.history().len(), 2);
    assert!(session.finish().is_ok());

    let session = config.begin(100, init).unwrap();
    let unfinished = session.finish().unwrap_err();
    assert_eq!(unfinished.remaining_rows(), 150);
    let mut session = unfinished.into_session();
    session.feed(rows).unwrap();
    assert_eq!(
        Error::from(session.finish().unwrap_err()),
        Error::Incomplete { remaining: 50 }
    );
}

#[test]
fn centroids_can_be_rebuilt_from_stored_values() {
    let dim = 7;
    let values = blobs(3000, dim, 9, 2.0, 9);
    let data = RowMatrix::new(&values, dim).unwrap();
    let model = Emeans::new(k(9)).fit(data).unwrap().model;
    let stored = model.centroids().as_slice().to_vec();
    let rebuilt = Centroids::restore(
        RowMatrix::new(&stored, dim).unwrap(),
        Metric::SquaredEuclidean,
    )
    .unwrap();
    assert_eq!(rebuilt.assign(data).unwrap(), model.assign(data).unwrap());
    assert_eq!(rebuilt.into_vec(), stored);

    let empty = RowMatrix::<f32>::new(&[], 3).unwrap();
    assert_eq!(
        Centroids::restore(empty, Metric::SquaredEuclidean).unwrap_err(),
        Error::NoCentroids
    );
    assert_eq!(
        Centroids::from_vectors(empty, Metric::SquaredEuclidean).unwrap_err(),
        Error::NoCentroids
    );
    let mut labels = vec![0; 10];
    let mut distances = vec![0.0; 9];
    assert_eq!(
        model.centroids().assign_into(
            RowMatrix::new(&values[..10 * dim], dim).unwrap(),
            &mut labels,
            &mut distances
        ),
        Err(Error::BufferLength {
            expected: 10,
            actual: 9
        })
    );
}

#[test]
fn f32_and_f64_data_give_close_results() {
    let dim = 6;
    let values = blobs(4000, dim, 10, 2.0, 10);
    let wide: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
    let config = Emeans::new(k(10));
    let narrow_fit = config.fit(RowMatrix::new(&values, dim).unwrap()).unwrap();
    let wide_fit = config.fit(RowMatrix::new(&wide, dim).unwrap()).unwrap();
    // The same random choices and well-separated data give the same labels.
    assert_eq!(narrow_fit.labels, wide_fit.labels);
    for (a, b) in narrow_fit
        .model
        .centroids()
        .as_slice()
        .iter()
        .zip(wide_fit.model.centroids().as_slice())
    {
        assert!((f64::from(*a) - b).abs() < 1e-5 * b.abs().max(1.0));
    }
}

#[test]
fn stored_centroids_restore_bit_for_bit() {
    // Spherical centroids are stored after they were scaled to norm 1 and
    // rounded to f32. Scaling such a vector again can move a coordinate by
    // one unit in the last place (about 1 vector in 6,000), so restoring must
    // not scale.
    let dim = 16;
    let mut stored = blobs(50_000, dim, 50, 3.0, 12);
    emeans::normalize_rows(&mut stored, dim).unwrap();
    let restored =
        Centroids::restore(RowMatrix::new(&stored, dim).unwrap(), Metric::Spherical).unwrap();
    let changed = stored
        .iter()
        .zip(restored.as_slice())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(changed, 0, "{changed} coordinates changed");

    for metric in [Metric::SquaredEuclidean, Metric::Spherical] {
        let mut values = blobs(20_000, dim, 30, 3.0, 11);
        if metric == Metric::Spherical {
            emeans::normalize_rows(&mut values, dim).unwrap();
        }
        let data = RowMatrix::new(&values, dim).unwrap();
        let model = Emeans::new(k(200))
            .with_metric(metric)
            .fit(data)
            .unwrap()
            .model;
        let stored = model.centroids().as_slice().to_vec();
        let restored = Centroids::restore(RowMatrix::new(&stored, dim).unwrap(), metric).unwrap();
        assert_eq!(restored.as_slice(), stored.as_slice(), "{metric:?}");
        assert_eq!(
            restored.assign(data).unwrap(),
            model.assign(data).unwrap(),
            "{metric:?}"
        );
    }
}

#[test]
fn restoring_checks_spherical_centroids_and_taking_vectors_scales_them() {
    // A unit row, a zero row (as an empty spherical cluster can keep) and a
    // row of norm 5.
    let values = [0.6f32, 0.8, 0.0, 0.0, 3.0, 4.0];
    let matrix = RowMatrix::new(&values, 2).unwrap();
    assert!(matches!(
        Centroids::restore(matrix, Metric::Spherical),
        Err(Error::NotUnitNorm { row: 2, .. })
    ));
    let restored =
        Centroids::restore(RowMatrix::new(&values[..4], 2).unwrap(), Metric::Spherical).unwrap();
    assert_eq!(restored.as_slice(), &values[..4]);

    let taken = Centroids::from_vectors(matrix, Metric::Spherical).unwrap();
    assert_eq!(taken.as_slice(), &[0.6, 0.8, 0.0, 0.0, 0.6, 0.8]);
    // Squared Euclidean centroids are kept as they are either way.
    let euclidean = Centroids::from_vectors(matrix, Metric::SquaredEuclidean).unwrap();
    assert_eq!(euclidean.as_slice(), &values);
}
