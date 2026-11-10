# Reference fixtures for e-means

The tools in this directory make the test fixtures in `tests/fixtures/`.
The fixtures contain the output of the faiss e-means NumPy reference.
The tests in `tests/reference.rs` compare the `emeans` crate with these values.

The fixtures are not in the repository (`.gitignore` excludes `tests/fixtures/`).
Thus `cargo test` ignores the tests that read them.
The CI job `reference` makes the fixtures and runs these tests on each pull request.

| File | Function |
| --- | --- |
| `generate_fixtures.py` | Checks the reference files, runs the reference and writes the two fixture files. |
| `faiss_shim.py` | Replaces `faiss` with an exact NumPy search. |
| `../../tests/common/synthetic.rs` | Makes the same synthetic data in Rust, bit for bit. |

## Pinned reference

- Repository: `facebookresearch/faiss`, directory `contrib/e_means`.
- Commit: `fc1095a9314f892938bf83f59e7ac966b5095def`.
- License: MIT.
- Paper: "e-means: outperforming k-means (in one-tenth the epochs)", arXiv:2611.09501.

`REFERENCE_FILES` in `generate_fixtures.py` gives the SHA-256 of `__init__.py`, `base.py`,
`e_means.py` and `e_means_spherical.py` at this commit.
The generator stops before it imports the reference if a file is different.
It also records the values in `source.reference_sha256` of each fixture file.

## Requirements

- Python 3.10 or later, because the code uses `int | None`. We tested only Python 3.14.6.
- NumPy. We used NumPy 2.5.3.
- Do not install faiss. The generator does not use it.

## Make the fixtures and run the tests

1. Get `contrib/e_means` at the pinned commit. For example:

   ```sh
   git clone --filter=blob:none --sparse https://github.com/facebookresearch/faiss.git /tmp/faiss
   git -C /tmp/faiss sparse-checkout set contrib/e_means
   git -C /tmp/faiss checkout fc1095a9314f892938bf83f59e7ac966b5095def
   ```

2. Go to the repository root.
3. Run the generator:

   ```sh
   python3 -I tools/reference/generate_fixtures.py \
       --reference-dir /tmp/faiss/contrib/e_means --out-dir tests/fixtures
   ```

4. Run the reference tests, including the ignored ones:

   ```sh
   cargo test --test reference -- --include-ignored
   ```

The generator takes approximately 2 seconds.
The output contains no time stamps and no local paths.
Thus two runs give the same bytes.

### Safety

Treat the reference directory as data from an external source.
The generator imports the reference package from `--reference-dir` with `importlib`.
Do not run other code from that directory.
Use `python3 -I`. This flag ignores environment variables, the user site directory and the current directory.
The flag `-I` does not stop bytecode writes.
Thus the generator sets `sys.dont_write_bytecode`, and it writes no `__pycache__` into the reference directory.

## The faiss shim

The reference `e_means.py` does `import faiss`. It uses only two calls:

- `faiss.knn(x, y, 1)` for Euclidean assignment.
- `faiss.knn(x, y, 1, metric=faiss.METRIC_INNER_PRODUCT)` for spherical assignment.

Real faiss selects a direct loop or a BLAS expansion `||x||^2 - 2xy + ||y||^2` from the matrix sizes.
Thus its distances and its choice between almost equal centroids change with the build and the batch shape.
The shim replaces this with one exact rule:

- The shim converts the float32 inputs to float64.
- For each query row and each centroid, the shim adds the terms in coordinate order, `t = 0 .. d-1`.
- L2 uses `s += (x[t] - y[t]) * (x[t] - y[t])`. Inner product uses `s += x[t] * y[t]`.
- Each step is one IEEE-754 float64 operation. There is no FMA and no pairwise sum.
- If two centroids have the same score, the lowest index wins.
- The shim returns `(D, I)` like faiss. `D` is float32 with shape (n, 1). `I` is int64 with shape (n, 1).
- `D` is the float64 score, rounded to float32.

The generator registers the shim as `sys.modules["faiss"]` before it imports the reference.
`synthetic::score` and `synthetic::nearest` in Rust use the same rule.

## Shared data generator

Python (`generate_fixtures.py`) and Rust (`tests/common/synthetic.rs`) must give the same bits.
Change both files together.
The generator uses only IEEE-754 basic operations and `sqrt`.
Thus the results are the same on all platforms.
Python uses plain loops for these values. It does not use NumPy sums.

### SplitMix64

All arithmetic is modulo 2^64.

```text
next():    state = state + 0x9E3779B97F4A7C15
           z = state
           z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
           z = (z ^ (z >> 27)) * 0x94D049BB133111EB
           return z ^ (z >> 31)
uniform(): (next() >> 11) as f64 * 2^-53          -> f64 in [0, 1)
below(m):  next() % m                             -> integer in [0, m), modulo bias is acceptable
```

### Blobs

`blobs(seed, n, d, centers, spread, normalize)` returns row-major `n x d` float32 values.

```text
rng = SplitMix64(seed)
for j in 0..centers: for t in 0..d:
    center[j][t] = (rng.uniform() * 2.0 - 1.0) * 10.0                    (f64)
for i in 0..n:
    j = rng.below(centers)
    for t in 0..d: v[t] = center[j][t] + (rng.uniform() - 0.5) * (2.0 * spread)   (f64)
    if normalize:
        s = 0.0; for t in 0..d (in order): s += v[t] * v[t]
        norm = sqrt(s)
        if norm != 0.0: for t in 0..d: v[t] = v[t] / norm
    x[i][t] = v[t] rounded to f32 (round to nearest, ties to even)
```

A zero row has a probability of almost zero. If it occurs, the row stays zero on both sides.

### Permutation

```text
permutation(m, rng): a = [0 .. m)
                     for i in (1 .. m).rev(): j = rng.below(i + 1); swap(a[i], a[j])
```

### Seeds

A scenario has a data seed `S`.

- Data: `blobs(S, ...)`.
- Initial centroid rows: `permutation(n, SplitMix64(S + 1))[..k]`.
- Row order of epoch `e` (0-based): `permutation(n, SplitMix64(S + 100 + e))[..epoch_rows[e]]`.

### Other helpers

- `fnv1a64_f32`: FNV-1a 64 over the little-endian bytes of each float32 value.
  The offset basis is `0xcbf29ce484222325`. The prime is `0x100000001b3`.
- `train_mse`: for each row, take the best shim score against the final centroids.
  For Euclidean runs, use the minimum squared distance.
  For spherical runs, use `2 - 2 * max inner product`.
  Add the row values with a correctly rounded sum (`math.fsum`, `synthetic::fsum`).
  Divide by n.

## Run procedure

For each scenario, the generator does these steps:

1. Make `x` with `blobs`.
2. Set the initial centroids to the rows at the initial indices.
3. Make `NumpyEmeans(d, k, nepochs=T, batch_size, alpha0, batch_law, seed=1234, capture_assignments=True, spherical, unused_threshold)`.
4. Call `begin(n, init)`.
5. For each epoch `e`, cut the row order into batches `order[b : b + B]`, with `B = recipe["batch_size"]`.
6. For each batch, call `stats = accumulate(x[batch])`, then `update(stats)`.
7. In the last epoch, copy `stats["labels"]` to the rows of the batch.
8. At the end of each epoch, call `end_epoch()`.
9. Call `finalize()`.

The six main scenarios use `unused_threshold=0.0`.
This setting stops revival, because the Rust crate cannot copy the reference revival RNG (`np.random.RandomState`).
The `revival_statistics` runs use the default threshold (0.1 for d = 8) and also 0.0.
Use them only for statistical comparisons.

| Scenario | Property |
| --- | --- |
| `euclid_auto_quality` | d = 8, `auto` selects the quality law. |
| `euclid_fractional_throughput` | d = 24, `auto` selects the throughput law. T = 2.5 gives a half epoch. |
| `spherical` | Unit rows, spherical assignment and closing. |
| `alpha_one_full_batch` | B = n, alpha0 = 1. The first update uses the `alpha == 1` branch. |
| `manual_batch_step_law` | B = 100 is given. The step law sets alpha0. |
| `epochs_below_one` | T = 0.6. One epoch of 480 rows. 320 rows get label -1. The law gives B = 1. |

## Fixture files

Both files have `schema_version` (1) and `source` (`faiss_commit`, `generator`, `reference_sha256`).

`reference_recipes.json` contains these items:

- `constants`: the fitted laws and the constants from `base.py`.
- `cases`: inputs (`n`, `k`, `d`, `nepochs`, `batch_law`, `batch_size`, `alpha0`, `unused_threshold`) and the expected recipe.
- `raising_cases`: inputs that the reference rejects, with the Python exception name.
- `round_power_of_two`: `input`, `input_bits` (f64 bits as 16 lowercase hex digits) and `output`.
- `fnv1a64` in `reference_runs.json` also uses 16 lowercase hex digits, with no `0x` prefix.

The generator calls only the `base.py` functions for `cases`.
It also builds the recipe through `NumpyEmeans._resolve_run_recipe` and makes sure that the two results are equal.

`reference_runs.json` contains these items:

- `scenarios`: inputs, `data_check`, recipe, per-epoch statistics, `n_splits`, final centroids, last-epoch labels and `train_mse`.
- `revival_statistics`: 20 runs (seeds 1000 to 1009, with and without revival), with `n_splits`, per-epoch statistics and `train_mse`.

Use `data_check` to make sure that the Rust generator is correct before you compare results.

### Selection of recipe cases

The full grid has too many cases. The generator keeps these groups:

- `grid`: each (n, k) pair with k <= n. When n <= 7, the group has each T. For larger n, it has 3 values of T in rotation. The law and d change in a cycle.
- `auto_boundary`: `auto` at each d (2, 3, 16, 17, 128).
- `k_equals_n`: k = n.
- `epoch_rounding`: fractional rows at an exact half, or near a half in float64.
- `explicit_batch`: given `batch_size`, also larger than n, with and without `alpha0`.
- `explicit_alpha`: a law batch with a given `alpha0`.

## Inputs that the reference rejects

The generator does not put these inputs in `cases`. It records them in `raising_cases`.

| Input | Python error | Location |
| --- | --- | --- |
| Unknown `batch_law` and no `batch_size` | `KeyError` | `_resolve_recipe` |
| `batch_size = 0` (with or without `alpha0`) | `ZeroDivisionError` | `_resolve_run_recipe` (step counts) |
| `batch_size < 0` and no `alpha0` | `TypeError` | `_step_law` (complex power) |
| `nepochs = 0` and no `alpha0` | `ZeroDivisionError` | `_step_law` (`0 ** negative`) |
| `nepochs < 0` | `TypeError` | `_round_power_of_two` (complex power) |
| `n = 0` | `ZeroDivisionError` | `_step_law` |
| `k = 0` | `ZeroDivisionError` | `_step_law` |
| `unused_threshold < 0` | `ValueError` | `_resolve_unused_threshold` |

## Reference behaviour to know

- Epoch rows use Python `round`, which rounds ties to even. For example, n = 5 and T = 2.5 gives `[5, 2, 5]`. Rust `f64::round` rounds ties away from zero. Use `round_ties_even`.
- The fractional part is `(T - floor(T)) * n` in float64. For n = 5, T = 0.1 gives exactly 0.5, thus 0 rows. T = 1.1 gives 0.5000000000000004, thus 1 row.
- n = 1 and T = 0.5 gives `epoch_rows = [0]` and `total_steps = 0`. The recipe resolves, but a run raises `ZeroDivisionError` in `end_epoch` (imbalance divides by `rows * rows`).
- The reference does not check `batch_law` when `batch_size` is given. `batch_law="bogus"` with `batch_size=64` gives a valid recipe.
- A given `alpha0` is not clamped. For example, `alpha0 = 2.0` stays 2.0. Only the step law clamps to [1e-4, 1].
- A negative `batch_size` with a given `alpha0` does not raise. It gives negative step counts.
- `nepochs = 0` with a given `alpha0` does not raise. It gives `epoch_rows = [0]`.
- The `alpha == 1` branch uses an exact comparison. It occurs at step 0 when alpha0 is 1.0, given or clamped (`0.1 + 0.9 * 1.0 == 1.0`).
- The recipe `batch_law` is `null` when `batch_size` is given. The runs fixture also records `batch_law_resolved`.
- In `revival_statistics`, all splits occur in the first epoch.

## Comparison guidance

- Parse JSON floats with full precision. With `serde_json`, enable the `float_roundtrip` feature.
- Centroids are float32 values written as float64. Parse as f64, then cast to f32. This cast is exact.
- `alpha0` and `alphas` use `pow` and `cos` from the platform math library. Different platforms can give 1 ULP of difference. Use a small relative tolerance for these values.
- Integer recipe fields must be equal.
- Run results depend on float32 sums in the reference. Compare centroids, objectives and `train_mse` with a tolerance. Compare labels by agreement rate.
