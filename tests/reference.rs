//! Differential tests against the reference implementation (faiss
//! `contrib/e_means`). `tools/reference/README.md` explains how the fixtures
//! were made and how to make them again.

mod common;

use std::num::{NonZeroU32, NonZeroUsize};

use common::synthetic;
use emeans::{
    Alpha, Batch, BatchLaw, BatchSource, Emeans, Epochs, Error, Metric, Model, Recipe, Revival,
    RowMatrix, Step, Threshold,
};
use serde::Deserialize;

const RECIPES: &str = include_str!("fixtures/reference_recipes.json");
const RUNS: &str = include_str!("fixtures/reference_runs.json");

/// The relative tolerance for values that the reference computes with the
/// platform's `pow` and `cos`, which may differ in the last bit.
const LIBM_TOLERANCE: f64 = 1e-12;

fn relative_difference(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE)
}

fn batch(law: &str, size: Option<u64>) -> Batch {
    match (size, law) {
        (Some(size), _) => {
            Batch::Fixed(NonZeroUsize::new(size as usize).expect("fixtures give positive batches"))
        }
        (None, "auto") => Batch::Auto,
        (None, "throughput") => Batch::Law(BatchLaw::Throughput),
        (None, "quality") => Batch::Law(BatchLaw::Quality),
        (None, other) => panic!("unknown batch law {other}"),
    }
}

fn law_name(source: BatchSource) -> Option<&'static str> {
    let name = |law| match law {
        BatchLaw::Throughput => "throughput",
        BatchLaw::Quality => "quality",
        _ => panic!("a batch law that the reference does not have"),
    };
    match source {
        BatchSource::Law(law) | BatchSource::AutoLaw(law) => Some(name(law)),
        _ => None,
    }
}

fn revival_threshold(recipe: &Recipe) -> f64 {
    recipe.revival().map_or(0.0, Threshold::get)
}

// ---------------------------------------------------------------- recipes

#[derive(Deserialize)]
struct RecipeFixtures {
    cases: Vec<RecipeCase>,
    raising_cases: Vec<RaisingCase>,
}

#[derive(Deserialize, Debug)]
struct RecipeInputs {
    n: u64,
    k: u64,
    d: usize,
    nepochs: f64,
    batch_law: String,
    batch_size: Option<i64>,
    alpha0: Option<f64>,
    unused_threshold: Option<f64>,
}

#[derive(Deserialize)]
struct RecipeExpected {
    batch_law: Option<String>,
    batch_size: u64,
    alpha0: f64,
    epoch_rows: Vec<u64>,
    epoch_steps: Vec<u64>,
    total_steps: u64,
    steps_per_pass: u64,
    unused_threshold: f64,
    alpha_steps: Vec<u64>,
    alphas: Vec<f64>,
}

#[derive(Deserialize)]
struct RecipeCase {
    group: String,
    inputs: RecipeInputs,
    expected: RecipeExpected,
}

#[derive(Deserialize)]
struct RaisingCase {
    inputs: RecipeInputs,
    error: String,
}

fn recipe_config(inputs: &RecipeInputs) -> Emeans {
    let clusters = NonZeroU32::new(u32::try_from(inputs.k).unwrap()).unwrap();
    let mut config = Emeans::new(clusters)
        .with_epochs(Epochs::new(inputs.nepochs).unwrap())
        .with_batch(batch(
            &inputs.batch_law,
            inputs.batch_size.map(|b| b as u64),
        ));
    if let Some(alpha0) = inputs.alpha0 {
        config = config.with_step(Step::Fixed(Alpha::new(alpha0).unwrap()));
    }
    config
}

#[test]
fn recipes_match_the_reference() {
    let fixtures: RecipeFixtures = serde_json::from_str(RECIPES).unwrap();
    assert!(fixtures.cases.len() > 200);
    let mut checked = 0;
    for case in &fixtures.cases {
        let inputs = &case.inputs;
        let expected = &case.expected;
        let label = format!("{} {:?}", case.group, inputs);
        let rows = usize::try_from(inputs.n).unwrap();
        let resolved = recipe_config(inputs).recipe(rows, inputs.d);
        if expected.epoch_rows == [0] {
            // The reference plans an empty run (and then divides by zero when
            // it ends the epoch); this crate rejects the budget up front.
            assert_eq!(
                resolved,
                Err(Error::EmptyBudget {
                    epochs: inputs.nepochs,
                    rows
                }),
                "{label}"
            );
            continue;
        }
        let recipe = resolved.unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_eq!(
            recipe.batch_size().get() as u64,
            expected.batch_size,
            "{label}"
        );
        assert_eq!(
            law_name(recipe.batch_source()),
            expected.batch_law.as_deref(),
            "{label}"
        );
        assert!(
            relative_difference(recipe.alpha0(), expected.alpha0) <= LIBM_TOLERANCE,
            "{label}"
        );
        let epoch_rows: Vec<u64> = recipe.epoch_sizes().map(|rows| rows as u64).collect();
        assert_eq!(epoch_rows, expected.epoch_rows, "{label}");
        let epoch_steps: Vec<u64> = (0..recipe.epoch_count())
            .map(|e| recipe.updates_in_epoch(e).unwrap())
            .collect();
        assert_eq!(epoch_steps, expected.epoch_steps, "{label}");
        assert_eq!(recipe.total_updates(), expected.total_steps, "{label}");
        assert_eq!(
            recipe.updates_per_pass(),
            expected.steps_per_pass,
            "{label}"
        );
        assert_eq!(
            revival_threshold(&recipe),
            expected.unused_threshold,
            "{label}"
        );
        assert_eq!(
            recipe.last_epoch_is_full(),
            inputs.nepochs >= 1.0,
            "{label}"
        );
        for (&step, &alpha) in expected.alpha_steps.iter().zip(&expected.alphas) {
            assert!(
                relative_difference(recipe.alpha(step), alpha) <= LIBM_TOLERANCE,
                "{label}, step {step}"
            );
        }
        checked += 1;
    }
    assert!(checked >= fixtures.cases.len() - 2, "{checked}");
}

#[test]
fn inputs_that_break_the_reference_cannot_be_expressed() {
    let fixtures: RecipeFixtures = serde_json::from_str(RECIPES).unwrap();
    assert!(!fixtures.raising_cases.is_empty());
    for case in &fixtures.raising_cases {
        let inputs = &case.inputs;
        let rejected = !matches!(inputs.batch_law.as_str(), "auto" | "throughput" | "quality")
            || inputs.batch_size.is_some_and(|size| {
                usize::try_from(size)
                    .ok()
                    .and_then(NonZeroUsize::new)
                    .is_none()
            })
            || Epochs::new(inputs.nepochs).is_err()
            || u32::try_from(inputs.k)
                .ok()
                .and_then(NonZeroU32::new)
                .is_none()
            || inputs
                .unused_threshold
                .is_some_and(|value| Threshold::new(value).is_err())
            || (inputs.k <= u32::MAX.into()
                && inputs.k > 0
                && Emeans::new(NonZeroU32::new(inputs.k as u32).unwrap())
                    .recipe(inputs.n as usize, inputs.d)
                    .is_err());
        assert!(
            rejected,
            "the reference raises {} for {:?}, but the input is accepted",
            case.error, inputs
        );
    }
}

// ------------------------------------------------------------------- runs

#[derive(Deserialize)]
struct RunFixtures {
    scenarios: Vec<RunResult>,
    revival_statistics: Vec<RevivalResult>,
}

#[derive(Deserialize, Debug, Clone)]
struct RunInputs {
    name: String,
    seed: u64,
    n: usize,
    d: usize,
    k: usize,
    centers: usize,
    spread: f64,
    nepochs: f64,
    normalize: bool,
    spherical: bool,
    batch_size: Option<usize>,
    alpha0: Option<f64>,
    batch_law: String,
    unused_threshold: Option<f64>,
}

/// What the generator recorded about a scenario's inputs, to show that
/// `synthetic` rebuilds them bit for bit.
#[derive(Deserialize)]
struct DataCheck {
    /// The first 8 values of the data.
    head: Vec<f64>,
    /// [`synthetic::fnv1a64_f32`] of all the data, as 16 hex digits.
    fnv1a64: String,
    init_indices: Vec<usize>,
    /// The first 8 rows of each epoch's order.
    epoch_order_heads: Vec<Vec<usize>>,
}

#[derive(Deserialize)]
struct RunRecipe {
    batch_size: usize,
    alpha0: f64,
    epoch_rows: Vec<usize>,
    total_steps: u64,
    unused_threshold: f64,
}

#[derive(Deserialize)]
struct EpochStat {
    obj: f64,
    imbalance_factor: f64,
    nsplit: u64,
}

#[derive(Deserialize)]
struct RunResult {
    inputs: RunInputs,
    data_check: DataCheck,
    recipe: RunRecipe,
    epochs: Vec<EpochStat>,
    n_splits: u64,
    centroids: Vec<Vec<f64>>,
    labels: Vec<i64>,
    train_mse: f64,
}

#[derive(Deserialize)]
struct RevivalResult {
    inputs: RunInputs,
    n_splits: u64,
    train_mse: f64,
}

fn run_config(inputs: &RunInputs) -> Emeans {
    let revival = match inputs.unused_threshold {
        None => Revival::Auto,
        Some(0.0) => Revival::Off,
        Some(threshold) => Revival::Threshold(Threshold::new(threshold).unwrap()),
    };
    let step = inputs
        .alpha0
        .map_or(Step::Law, |alpha0| Step::Fixed(Alpha::new(alpha0).unwrap()));
    Emeans::new(NonZeroU32::new(inputs.k as u32).unwrap())
        .with_epochs(Epochs::new(inputs.nepochs).unwrap())
        .with_batch(batch(
            &inputs.batch_law,
            inputs.batch_size.map(|b| b as u64),
        ))
        .with_step(step)
        .with_revival(revival)
        .with_metric(if inputs.spherical {
            Metric::Spherical
        } else {
            Metric::SquaredEuclidean
        })
}

fn shim_metric(inputs: &RunInputs) -> synthetic::Metric {
    if inputs.spherical {
        synthetic::Metric::InnerProduct
    } else {
        synthetic::Metric::L2
    }
}

/// A run of this crate on the fixture's data, initial centroids and row
/// orders. Returns the model, the last epoch's labels (-1 for rows that it
/// did not visit) and the training MSE measured as the fixture measures it.
struct Run {
    model: Model<f32>,
    labels: Vec<i64>,
    train_mse: f64,
}

fn run(inputs: &RunInputs) -> Run {
    let (n, d) = (inputs.n, inputs.d);
    let x = synthetic::blobs(
        inputs.seed,
        n,
        d,
        inputs.centers,
        inputs.spread,
        inputs.normalize,
    );
    let init = synthetic::gather_rows(&x, d, &synthetic::init_indices(inputs.seed, n, inputs.k));
    let mut session = run_config(inputs)
        .begin(n, RowMatrix::new(&init, d).unwrap())
        .unwrap();
    let recipe = session.recipe().clone();
    let mut labels = vec![-1i64; n];
    for epoch in 0..recipe.epoch_count() {
        let rows = recipe.rows_in_epoch(epoch).unwrap();
        let order = synthetic::epoch_order(inputs.seed, n, epoch as usize, rows);
        let last = epoch + 1 == recipe.epoch_count();
        for batch in order.chunks(recipe.batch_size().get()) {
            let values = synthetic::gather_rows(&x, d, batch);
            let mut batch_labels = vec![0u32; batch.len()];
            session
                .feed_labeled(RowMatrix::new(&values, d).unwrap(), &mut batch_labels)
                .unwrap();
            if last {
                for (&row, &label) in batch.iter().zip(&batch_labels) {
                    labels[row] = i64::from(label);
                }
            }
        }
    }
    let model = session.finish().unwrap();
    let train_mse = synthetic::train_mse(&x, model.centroids().as_slice(), d, shim_metric(inputs));
    Run {
        model,
        labels,
        train_mse,
    }
}

#[test]
fn inputs_are_the_reference_inputs_bit_for_bit() {
    synthetic::self_check();
    let fixtures: RunFixtures = serde_json::from_str(RUNS).unwrap();
    for scenario in &fixtures.scenarios {
        let (inputs, check) = (&scenario.inputs, &scenario.data_check);
        let name = &inputs.name;
        let x = synthetic::blobs(
            inputs.seed,
            inputs.n,
            inputs.d,
            inputs.centers,
            inputs.spread,
            inputs.normalize,
        );
        let head: Vec<f64> = x.iter().take(8).map(|&value| f64::from(value)).collect();
        assert_eq!(head, check.head, "{name}");
        assert_eq!(
            format!("{:016x}", synthetic::fnv1a64_f32(&x)),
            check.fnv1a64,
            "{name}"
        );
        assert_eq!(
            synthetic::init_indices(inputs.seed, inputs.n, inputs.k),
            check.init_indices,
            "{name}"
        );
        let recipe = run_config(inputs).recipe(inputs.n, inputs.d).unwrap();
        let heads: Vec<Vec<usize>> = (0..recipe.epoch_count())
            .map(|epoch| {
                let rows = recipe.rows_in_epoch(epoch).unwrap();
                let order = synthetic::epoch_order(inputs.seed, inputs.n, epoch as usize, rows);
                order[..order.len().min(8)].to_vec()
            })
            .collect();
        assert_eq!(heads, check.epoch_order_heads, "{name}");
    }
}

#[test]
fn runs_match_the_reference() {
    let fixtures: RunFixtures = serde_json::from_str(RUNS).unwrap();
    assert_eq!(fixtures.scenarios.len(), 6);
    for scenario in &fixtures.scenarios {
        let name = &scenario.inputs.name;
        let ours = run(&scenario.inputs);
        let recipe = ours.model.recipe();

        // The plan is the same.
        assert_eq!(
            recipe.batch_size().get(),
            scenario.recipe.batch_size,
            "{name}"
        );
        assert!(
            relative_difference(recipe.alpha0(), scenario.recipe.alpha0) <= LIBM_TOLERANCE,
            "{name}"
        );
        assert_eq!(
            recipe.epoch_sizes().collect::<Vec<_>>(),
            scenario.recipe.epoch_rows,
            "{name}"
        );
        assert_eq!(
            recipe.total_updates(),
            scenario.recipe.total_steps,
            "{name}"
        );
        assert_eq!(
            revival_threshold(recipe),
            scenario.recipe.unused_threshold,
            "{name}"
        );
        assert_eq!(ours.model.revivals(), scenario.n_splits, "{name}");

        // Every row of the last epoch has the same label.
        assert_eq!(ours.labels, scenario.labels, "{name}");

        // Per-epoch statistics agree. The reference keeps its state in f32
        // and this crate in f64, so values agree to f32 precision, not bits.
        let history = ours.model.history();
        assert_eq!(history.len(), scenario.epochs.len(), "{name}");
        for (stats, expected) in history.iter().zip(&scenario.epochs) {
            assert!(
                relative_difference(stats.objective, expected.obj) < 1e-5,
                "{name}: {} vs {}",
                stats.objective,
                expected.obj
            );
            assert!(
                relative_difference(stats.imbalance, expected.imbalance_factor) < 1e-12,
                "{name}"
            );
            assert_eq!(stats.revivals, expected.nsplit, "{name}");
        }

        // The final centroids agree to f32 precision.
        let centroids = ours.model.centroids();
        let mut worst = 0.0f64;
        for (cluster, expected) in scenario.centroids.iter().enumerate() {
            for (&actual, &expected) in centroids
                .centroid(cluster as u32)
                .unwrap()
                .iter()
                .zip(expected)
            {
                worst = worst.max((f64::from(actual) - expected).abs() / expected.abs().max(1.0));
            }
        }
        assert!(
            worst < 1e-5,
            "{name}: largest centroid difference {worst:e}"
        );
        assert!(
            relative_difference(ours.train_mse, scenario.train_mse) < 1e-6,
            "{name}: {} vs {}",
            ours.train_mse,
            scenario.train_mse
        );
    }
}

#[test]
fn revival_matches_the_reference_statistically() {
    let fixtures: RunFixtures = serde_json::from_str(RUNS).unwrap();
    let mut ours_mse = Vec::new();
    let mut reference_mse = Vec::new();
    let mut ours_splits = Vec::new();
    let mut reference_splits = Vec::new();
    let mut off_runs_that_track = 0;
    for result in &fixtures.revival_statistics {
        let ours = run(&result.inputs);
        if result.inputs.unused_threshold == Some(0.0) {
            // Without revival there is no random choice, so the runs follow
            // the same path. With 64 centroids on 8 blobs, rows often sit
            // almost exactly between two centroids; the reference's f32 state
            // can then send a row to the other centroid than this crate's f64
            // state does, and the paths separate a little.
            assert_eq!(ours.model.revivals(), 0);
            let difference = relative_difference(ours.train_mse, result.train_mse);
            assert!(difference < 1e-3, "{:?}: {difference:e}", result.inputs);
            off_runs_that_track += usize::from(difference < 1e-7);
        } else {
            ours_mse.push(ours.train_mse);
            reference_mse.push(result.train_mse);
            ours_splits.push(ours.model.revivals() as f64);
            reference_splits.push(result.n_splits as f64);
        }
    }
    assert!(
        off_runs_that_track >= 8,
        "only {off_runs_that_track} of 10 runs without revival track the reference"
    );
    assert_eq!(ours_mse.len(), 10);
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    // The revival draws differ (different random generators), so only the
    // distributions can agree. Over these 10 seeds the reference's MSE has a
    // spread of about 1%, and its revival count ranges from 13 to 23.
    let (ours, reference) = (mean(&ours_mse), mean(&reference_mse));
    eprintln!("mean train MSE: ours {ours:.5}, reference {reference:.5}");
    eprintln!(
        "mean revivals: ours {:.1}, reference {:.1}",
        mean(&ours_splits),
        mean(&reference_splits)
    );
    assert!(
        relative_difference(ours, reference) < 0.01,
        "mean train MSE: ours {ours}, reference {reference}"
    );
    let (ours, reference) = (mean(&ours_splits), mean(&reference_splits));
    assert!(
        (ours - reference).abs() < 0.3 * reference,
        "mean revivals: ours {ours}, reference {reference}"
    );
}
