# emeans

A Rust implementation of **e-means**, the mini-batch k-means algorithm from
Meta FAIR.

e-means replaces the full-batch update of Lloyd's algorithm with mini-batch
updates of an exponential moving average. Fitted scaling laws set the batch
size and the step size, so there is no hyperparameter to tune. The paper
reports that e-means reaches a lower loss than 25 iterations of Lloyd's
algorithm in 1.2 to 3 epochs.

## Usage

```toml
[dependencies]
emeans = "0.1"
```

```rust
use std::num::NonZeroU32;
use emeans::{Emeans, Epochs, RowMatrix};

fn cluster(values: &[f32], dim: usize) -> Result<(), emeans::Error> {
    let data = RowMatrix::new(values, dim)?;           // one vector per row
    let fit = Emeans::new(NonZeroU32::new(1024).unwrap())
        .with_epochs(Epochs::new(3.0)?)                   // 5 by default
        .fit(data)?;

    println!("{}", fit.model.recipe());                   // batch size, step size, epochs
    let centroids = fit.model.centroids();                // 1024 × dim
    let labels = fit.labels;                              // last-epoch label of each row
    let assignment = centroids.assign(data)?;             // exact nearest centroids and distances
    println!("MSE = {}", assignment.mean_distance());
    Ok(())
}
```

- **Data that does not fit in memory:** `Emeans::begin` returns a `Session`.
  Feed it rows from storage in chunks of any size.
- **Cosine similarity:** `Metric::Spherical`, on rows that `normalize_rows`
  scaled to norm 1.
- **Inspect before training:** `Emeans::recipe(rows, dim)` returns the batch
  size, the step schedule and the epoch plan.
- **Store and reload:** save `centroids.as_slice()`, then
  `Centroids::restore` brings the centroids back bit for bit, so they assign
  exactly as before.

The [crate documentation](https://docs.rs/emeans) describes the algorithm,
every setting and the numerical behavior.

## The algorithm in brief

```text
 init: k random rows ──► c_1 … c_k,  N_j = 2⁻¹⁶
  │
  ▼
 ┌─ for each epoch: a fresh random order of the rows ──────────────┐
 │   ┌─ for each batch of B rows ────────────────────────────────┐ │
 │   │  assign each row to its nearest centroid                  │ │
 │   │  N_j ← (1−α)·N_j + α·n_j,   S_j ← (1−α)·S_j + α·s_j       │ │
 │   │  every 5 updates: revive dead clusters by splitting       │ │
 │   └───────────────────────────────────────────────────────────┘ │
 └─────────────────────────────────────────────────────────────────┘
  │
  ▼
 final c_j = mean of the rows that the last epoch assigned to j
```

| Setting | Rule (from the reference implementation) |
|---|---|
| Batch size `B` | `0.035134 · k^0.134749 · n^0.876473 · T^0.813344` (throughput law; the quality law is 8× smaller), rounded to a power of two |
| Initial step `α₀` | `0.8244 · k^-0.4617 · B^0.9003 · n^-0.4498 · T^-0.1057`, clamped to `[1e-4, 1]` |
| Step schedule | cosine from `α₀` to `α₀ / 10` over the run |
| Revival | every 5 updates, clusters below 0.1 × the mean running count take half of a donor drawn in proportion to its count |
| Finalization | each centroid becomes the exact mean of its rows in the last epoch |

## Precision and reproducibility

- Running statistics, centroid updates, objectives and distances are `f64`
  for both `f32` and `f64` data.
- The nearest-centroid search is exact: each label is the centroid with the
  smallest distance computed in `f64`. A matrix product in the element type
  gives the speed, and a rigorous error bound decides which few centroids
  need the exact computation.
- For a given seed, results are bit-identical for any number of threads and
  for any chunking of a session's input.

## Status

This crate implements the CPU algorithm of the reference implementation
([faiss `contrib/e_means`](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means)).
The reference's GPU kernels, low-precision formats and multi-process
training are not implemented. Tests compare this crate with the reference
implementation; see
[`tools/reference/`](https://github.com/Jeadie/emean-rs/tree/main/tools/reference).

## Citation

If you use e-means, cite the paper:

```bibtex
@misc{bessa2026emeans,
  title         = {e-means: outperforming k-means (in one-tenth the epochs)},
  author        = {Swann Bessa and Pierre-Emmanuel Mazaré and Gergely Szilvasy and
                   Hugo Touvron and David Picard and Matthijs Douze and Hervé Jégou},
  year          = {2026},
  eprint        = {2611.09501},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2611.09501}
}
```

## License

Apache-2.0. The algorithm, its constants and its scaling-law coefficients
come from the paper and from the reference implementation in faiss
(MIT license, Copyright (c) Meta Platforms, Inc. and affiliates). This crate
is an independent implementation in Rust.
