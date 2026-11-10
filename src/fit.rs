//! Training on data that is in memory.

use crate::Error;
use crate::config::Emeans;
use crate::matrix::{RowMatrix, RowNorm};
use crate::model::Fit;
use crate::positions;
use crate::rng::{Rng, Stream};
use crate::rows::CheckedRows;
use crate::scalar::Scalar;
use crate::session::{self, Session};

/// Runs Algorithm 1 of the paper on `data`: random initial centroids (or
/// `init`), then a fresh random order of the rows each epoch, cut into
/// batches.
pub(crate) fn fit<T: Scalar>(
    config: &Emeans,
    data: RowMatrix<'_, T>,
    init: Option<RowMatrix<'_, T>>,
) -> Result<Fit<T>, Error> {
    let recipe = config.recipe(data.rows(), data.dim())?;
    if let Some(init) = init {
        if init.dim() != data.dim() {
            return Err(Error::DimensionMismatch {
                expected: data.dim(),
                actual: init.dim(),
            });
        }
    }
    // Every batch below is a view of these checked rows.
    let data = CheckedRows::new(data, RowNorm::for_metric(config.metric()))?;

    let rows = data.matrix().rows();
    let clusters = config.clusters().get() as usize;
    let seed = config.seed();
    let mut order: Vec<usize> = (0..rows).collect();
    let positions = match init {
        Some(init) => session::initial_positions(config, init)?,
        None => {
            // Uniform sample without replacement: the first k rows of a
            // random permutation. With k = n, every row, in order.
            if rows != clusters {
                Rng::new(seed, Stream::Initialization).shuffle(&mut order);
            }
            let sample = order[..clusters]
                .iter()
                .map(|&row| data.matrix().row_at(row));
            positions::from_vectors(sample, data.matrix().dim(), config.metric())
        }
    };

    let mut session = Session::start(recipe, positions);
    let recipe = session.recipe().clone();
    let mut labels = recipe.last_epoch_is_full().then(|| vec![0u32; rows]);
    for epoch in recipe.plan() {
        for (position, row) in order.iter_mut().enumerate() {
            *row = position;
        }
        Rng::new(seed, Stream::Epoch(epoch.index())).shuffle(&mut order);
        for batch in order[..epoch.rows().get()].chunks(recipe.batch_size().get()) {
            let source = data.gather(batch);
            match labels.as_mut() {
                Some(out) if epoch.is_last() => session.process(&source, |offset, segment| {
                    for (&row, &label) in batch[offset..].iter().zip(segment) {
                        out[row] = label;
                    }
                }),
                _ => session.process(&source, |_, _| {}),
            }
        }
    }
    let model = session.finish()?;
    Ok(Fit { model, labels })
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU32;

    use super::*;
    use crate::{Epochs, Metric, Model, Revival, Threshold};

    /// Points around `centers` random centers in `[-5, 5]^dim`.
    fn blobs(rows: usize, dim: usize, centers: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed, Stream::Revival);
        let middles: Vec<f64> = (0..centers * dim)
            .map(|_| 10.0 * rng.open_unit() - 5.0)
            .collect();
        let mut data = Vec::with_capacity(rows * dim);
        for _ in 0..rows {
            let center = rng.below(centers as u64) as usize;
            for t in 0..dim {
                data.push((middles[center * dim + t] + rng.open_unit() - 0.5) as f32);
            }
        }
        data
    }

    fn bits(model: &Model<f32>) -> Vec<u32> {
        model
            .centroids()
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    /// Replays `fit` through the public session API, feeding `chunk` rows at
    /// a time, and returns the model and the last epoch's labels.
    fn replay(config: &Emeans, data: RowMatrix<'_, f32>, chunk: usize) -> (Model<f32>, Vec<u32>) {
        let rows = data.rows();
        let clusters = config.clusters().get() as usize;
        let mut order: Vec<usize> = (0..rows).collect();
        Rng::new(config.seed(), Stream::Initialization).shuffle(&mut order);
        let init: Vec<f32> = order[..clusters]
            .iter()
            .flat_map(|&row| data.row_at(row).iter().copied())
            .collect();
        let mut session = config
            .begin(rows, RowMatrix::new(&init, data.dim()).unwrap())
            .unwrap();
        let mut labels = vec![u32::MAX; rows];
        let epochs = session.recipe().epoch_count();
        for epoch in 0..epochs {
            order = (0..rows).collect();
            Rng::new(config.seed(), Stream::Epoch(epoch)).shuffle(&mut order);
            let epoch_rows = session.recipe().rows_in_epoch(epoch).unwrap();
            for indices in order[..epoch_rows].chunks(chunk) {
                let values: Vec<f32> = indices
                    .iter()
                    .flat_map(|&row| data.row_at(row).iter().copied())
                    .collect();
                let piece = RowMatrix::new(&values, data.dim()).unwrap();
                let mut piece_labels = vec![0; indices.len()];
                session.feed_labeled(piece, &mut piece_labels).unwrap();
                for (&row, &label) in indices.iter().zip(&piece_labels) {
                    labels[row] = label;
                }
            }
        }
        (session.finish().unwrap(), labels)
    }

    #[test]
    fn fit_equals_a_session_fed_in_any_chunk_size() {
        let dim = 5;
        let values = blobs(1500, dim, 9, 3);
        let data = RowMatrix::new(&values, dim).unwrap();
        let config = Emeans::new(NonZeroU32::new(12).unwrap())
            .with_epochs(Epochs::new(2.5).unwrap())
            .with_revival(Revival::Threshold(Threshold::new(0.5).unwrap()))
            .with_seed(77);
        let fit = config.fit(data).unwrap();
        assert!(fit.model.revivals() > 0, "the test should exercise revival");
        assert!(fit.model.recipe().batch_size().get() > 1);

        for chunk in [1, 37, 333, 4096] {
            let (model, labels) = replay(&config, data, chunk);
            assert_eq!(bits(&model), bits(&fit.model), "chunk {chunk}");
            assert_eq!(model.history(), fit.model.history(), "chunk {chunk}");
            assert_eq!(Some(labels), fit.labels, "chunk {chunk}");
        }
    }

    #[test]
    fn final_centroids_are_the_means_of_their_last_epoch_rows() {
        let dim = 3;
        let values = blobs(2000, dim, 6, 5);
        let data = RowMatrix::new(&values, dim).unwrap();
        let fit = Emeans::new(NonZeroU32::new(10).unwrap()).fit(data).unwrap();
        let labels = fit.labels.unwrap();
        let centroids = fit.model.centroids();
        for cluster in 0..10u32 {
            let members: Vec<&[f32]> = data
                .iter()
                .zip(&labels)
                .filter(|&(_, &l)| l == cluster)
                .map(|(row, _)| row)
                .collect();
            let centroid = centroids.centroid(cluster).unwrap();
            if members.is_empty() {
                continue;
            }
            for t in 0..dim {
                let mean =
                    members.iter().map(|row| f64::from(row[t])).sum::<f64>() / members.len() as f64;
                assert_eq!(centroid[t], mean as f32, "cluster {cluster}, column {t}");
            }
        }
    }

    #[test]
    fn spherical_fit_keeps_unit_centroids() {
        let dim = 4;
        let mut values = blobs(1000, dim, 5, 8);
        crate::normalize_rows(&mut values, dim).unwrap();
        let data = RowMatrix::new(&values, dim).unwrap();
        let fit = Emeans::new(NonZeroU32::new(7).unwrap())
            .with_metric(Metric::Spherical)
            .fit(data)
            .unwrap();
        for centroid in fit.model.centroids().as_matrix() {
            let norm = centroid
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-6, "{norm}");
        }
        let assignment = fit.model.assign(data).unwrap();
        assert!(
            assignment
                .distances
                .iter()
                .all(|&d| (-1e-6..=4.0).contains(&d))
        );
    }
}
