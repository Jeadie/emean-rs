//! Deterministic synthetic data for differential tests against the faiss
//! e-means reference.
//!
//! This module mirrors `tools/reference/generate_fixtures.py` bit for bit:
//! the same `SplitMix64` stream, blob data, permutations, seeds and scoring.
//! The fixtures in `tests/fixtures/` were produced from that Python code, so
//! a test can rebuild every input here instead of storing it in JSON.
//! Change both sides together. `tools/reference/README.md` has the spec.
//!
//! Only IEEE-754 basic operations and `sqrt` are used, in a fixed order,
//! with no fused multiply-add, so results are identical on every platform.

/// Seed offset for the initial-centroid permutation: `seed + 1`.
pub const INIT_SEED_OFFSET: u64 = 1;

/// Seed offset for epoch row orders: epoch `e` uses `seed + 100 + e`.
pub const EPOCH_SEED_OFFSET: u64 = 100;

/// `2^-53`, the spacing of `SplitMix64::uniform` outputs.
const TWO_POW_NEG_53: f64 = 1.0 / 9_007_199_254_740_992.0;

/// `SplitMix64` pseudo-random generator (Steele, Lea and Flood; Vigna's
/// constants). All arithmetic wraps modulo 2^64.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a generator whose internal state starts at `seed`.
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The current internal state (the seed before the first draw).
    pub const fn state(&self) -> u64 {
        self.state
    }

    /// Advance the state by the golden-ratio increment and return the mixed
    /// 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A float in `[0, 1)` built from the top 53 bits: `(next >> 11) * 2^-53`.
    #[allow(clippy::cast_precision_loss)] // the shifted value is < 2^53: exact
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * TWO_POW_NEG_53
    }

    /// An integer in `[0, m)` computed as `next % m` (modulo bias accepted).
    ///
    /// # Panics
    /// Panics if `m == 0`, as the Python side raises.
    pub fn below(&mut self, m: u64) -> u64 {
        assert!(m > 0, "SplitMix64::below needs m >= 1");
        self.next_u64() % m
    }

    /// `below` for `usize` bounds and results.
    pub fn below_usize(&mut self, m: usize) -> usize {
        let m = u64::try_from(m).expect("usize bound fits in u64");
        usize::try_from(self.below(m)).expect("a value below a usize bound fits in usize")
    }
}

/// Row-major `n x d` blob data as `f32`.
///
/// Draw `centers` centers with coordinates `(u * 2 - 1) * 10`, then for each
/// row pick a center with `below(centers)` and add `(u - 0.5) * (2 * spread)`
/// per coordinate, all in `f64`. With `normalize`, divide the row by its
/// Euclidean norm (sum of squares accumulated in coordinate order; a zero row
/// stays zero). Finally round each value to the nearest `f32` (ties to even).
///
/// # Panics
/// Panics if `n > 0` and `centers == 0`.
#[allow(clippy::cast_possible_truncation)] // f64 -> f32 rounding is the spec
#[allow(clippy::many_single_char_names)] // names follow the written spec
pub fn blobs(
    seed: u64,
    n: usize,
    d: usize,
    centers: usize,
    spread: f64,
    normalize: bool,
) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut center = Vec::with_capacity(centers * d);
    for _ in 0..centers * d {
        center.push((rng.uniform() * 2.0 - 1.0) * 10.0);
    }
    let mut out = Vec::with_capacity(n * d);
    let mut v = vec![0.0f64; d];
    for _ in 0..n {
        let j = rng.below_usize(centers);
        for (t, value) in v.iter_mut().enumerate() {
            *value = center[j * d + t] + (rng.uniform() - 0.5) * (2.0 * spread);
        }
        if normalize {
            let mut s = 0.0f64;
            for &value in &v {
                s += value * value;
            }
            let norm = s.sqrt();
            if norm != 0.0 {
                for value in &mut v {
                    *value /= norm;
                }
            }
        }
        out.extend(v.iter().map(|&value| value as f32));
    }
    out
}

/// Fisher-Yates permutation of `0..m`: for `i` from `m - 1` down to 1, swap
/// `a[i]` with `a[rng.below(i + 1)]`.
pub fn permutation(m: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut a: Vec<usize> = (0..m).collect();
    for i in (1..m).rev() {
        let j = rng.below_usize(i + 1);
        a.swap(i, j);
    }
    a
}

/// Rows used as initial centroids for data seed `seed`:
/// `permutation(n, SplitMix64::new(seed + 1))[..k]`.
///
/// # Panics
/// Panics if `k > n` or `seed + 1` overflows.
pub fn init_indices(seed: u64, n: usize, k: usize) -> Vec<usize> {
    let seed = seed
        .checked_add(INIT_SEED_OFFSET)
        .expect("init seed overflows u64");
    let mut order = permutation(n, &mut SplitMix64::new(seed));
    assert!(k <= n, "init_indices needs k <= n");
    order.truncate(k);
    order
}

/// Row ids visited in 0-based `epoch`, in visit order:
/// `permutation(n, SplitMix64::new(seed + 100 + epoch))[..rows]`.
///
/// # Panics
/// Panics if `rows > n` or the epoch seed overflows.
pub fn epoch_order(seed: u64, n: usize, epoch: usize, rows: usize) -> Vec<usize> {
    let seed = seed
        .checked_add(EPOCH_SEED_OFFSET)
        .and_then(|s| s.checked_add(epoch as u64))
        .expect("epoch seed overflows u64");
    let mut order = permutation(n, &mut SplitMix64::new(seed));
    assert!(rows <= n, "epoch_order needs rows <= n");
    order.truncate(rows);
    order
}

/// Copy the listed rows of a row-major matrix with `d` columns, in order.
///
/// # Panics
/// Panics if a row id is out of range.
pub fn gather_rows(x: &[f32], d: usize, rows: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len() * d);
    for &row in rows {
        out.extend_from_slice(&x[row * d..(row + 1) * d]);
    }
    out
}

/// FNV-1a 64 hash over the little-endian IEEE-754 bytes of `values`.
///
/// Fixtures store it as 16 lowercase hex digits (`format!("{:016x}", h)`).
pub fn fnv1a64_f32(values: &[f32]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}

/// Correctly rounded sum of finite `f64` values (a port of `math.fsum` from
/// `CPython`: Shewchuk's exact partials with a half-even fix-up).
///
/// # Panics
/// Panics on a non-finite input or an overflowing partial sum.
#[allow(clippy::float_cmp, clippy::many_single_char_names)] // exact-arithmetic algorithm
pub fn fsum<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for value in values {
        assert!(value.is_finite(), "fsum needs finite inputs");
        let mut x = value;
        let mut kept = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[kept] = lo;
                kept += 1;
            }
            x = hi;
        }
        partials.truncate(kept);
        if x != 0.0 {
            assert!(x.is_finite(), "fsum intermediate overflow");
            partials.push(x);
        }
    }
    let Some(mut n) = partials.len().checked_sub(1) else {
        return 0.0;
    };
    let mut hi = partials[n];
    let mut lo = 0.0;
    while n > 0 {
        let x = hi;
        n -= 1;
        let y = partials[n];
        hi = x + y;
        lo = y - (hi - x);
        if lo != 0.0 {
            break;
        }
    }
    if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
        let y = lo * 2.0;
        let x = hi + y;
        if y == x - hi {
            hi = x;
        }
    }
    hi
}

/// How a row is scored against a centroid, as in `tools/reference/faiss_shim.py`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Squared Euclidean distance; smaller is better.
    L2,
    /// Inner product; larger is better (spherical clustering).
    InnerProduct,
}

/// Score `a` against `b` in `f64`, accumulating over coordinates in order:
/// `s += (a - b) * (a - b)` for L2 or `s += a * b` for inner product.
///
/// # Panics
/// Panics if the slices have different lengths.
pub fn score(a: &[f32], b: &[f32], metric: Metric) -> f64 {
    assert_eq!(a.len(), b.len(), "score needs equal lengths");
    let mut total = 0.0f64;
    for (&left, &right) in a.iter().zip(b) {
        let (left, right) = (f64::from(left), f64::from(right));
        match metric {
            Metric::L2 => {
                let diff = left - right;
                total += diff * diff;
            }
            Metric::InnerProduct => total += left * right,
        }
    }
    total
}

/// Best centroid for `row` and its `f64` score, ties to the lowest index.
///
/// This is the exact assignment the reference fixtures used (the faiss shim).
///
/// # Panics
/// Panics if `centroids` is empty or not a multiple of `row.len()`.
pub fn nearest(row: &[f32], centroids: &[f32], metric: Metric) -> (usize, f64) {
    let d = row.len();
    assert!(
        d > 0 && !centroids.is_empty() && centroids.len() % d == 0,
        "bad centroid table"
    );
    let mut best = (0, score(row, &centroids[..d], metric));
    for (j, centroid) in centroids.chunks_exact(d).enumerate().skip(1) {
        let s = score(row, centroid, metric);
        let better = match metric {
            Metric::L2 => s < best.1,
            Metric::InnerProduct => s > best.1,
        };
        if better {
            best = (j, s);
        }
    }
    best
}

/// Final training error, computed like `train_mse` in the fixtures: for each
/// row of `x` the best `nearest` score (L2: squared distance; inner product:
/// `2 - 2 * similarity`), summed with `fsum` and divided by the row count.
///
/// # Panics
/// Panics if `x` is empty, `d == 0`, or the shapes do not match `d`.
#[allow(clippy::cast_precision_loss)] // row counts are far below 2^53
pub fn train_mse(x: &[f32], centroids: &[f32], d: usize, metric: Metric) -> f64 {
    assert!(d > 0 && !x.is_empty() && x.len() % d == 0, "bad data shape");
    let n = x.len() / d;
    let per_row = x.chunks_exact(d).map(|row| {
        let (_, best) = nearest(row, centroids, metric);
        match metric {
            Metric::L2 => best,
            Metric::InnerProduct => 2.0 - 2.0 * best,
        }
    });
    fsum(per_row) / n as f64
}

/// Check this module against values from outside it: the published
/// `SplitMix64` outputs for seed 0, `fsum` cases with known exact sums, and
/// generator outputs printed by `tools/reference/generate_fixtures.py`.
///
/// # Panics
/// Panics on the first mismatch.
#[allow(clippy::float_cmp)] // the expected sums are exact
pub fn self_check() {
    // SplitMix64 seeded with 0 (Vigna's reference implementation).
    let mut rng = SplitMix64::new(0);
    let expected = [
        0xE220_A839_7B1D_CDAF,
        0x6E78_9E6A_A1B9_65F4,
        0x06C4_5D18_8009_454F,
        0xF88B_B8A8_724C_81EC,
        0x1B39_896A_51A8_749B,
    ];
    for want in expected {
        assert_eq!(rng.next_u64(), want, "SplitMix64(0) stream");
    }

    // Exact sums: plain summation gets each of these wrong.
    assert_eq!(
        fsum([1e100, 1.0, -1e100, 1e-100, 1e50, -1.0, -1e50]),
        1e-100
    );
    assert_eq!(fsum([0.1; 10]), 1.0);
    assert_eq!(fsum([1e-16, 1.0, 1e16]), 10_000_000_000_000_002.0);
    assert_eq!(fsum(std::iter::empty()), 0.0);

    // Printed by the Python generator (generate_fixtures.py).
    let raw: [u32; 8] = [
        0xC02E_06F6,
        0xC12C_2EF0,
        0x40F8_25FE,
        0x4042_7254,
        0x3E9B_1545,
        0xC079_6E95,
        0xC003_BF2A,
        0xC108_6803,
    ];
    let unit: [u32; 8] = [
        0xBE4D_A455,
        0xBF4B_768F,
        0x3F12_9D3C,
        0x3F1D_1908,
        0x3D7A_971A,
        0xBF49_858D,
        0xBE36_B96A,
        0xBF3D_2FD0,
    ];
    let bits = |v: Vec<f32>| v.iter().take(8).map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(blobs(7, 5, 3, 2, 1.5, false)), raw, "blobs raw");
    assert_eq!(bits(blobs(7, 5, 3, 2, 1.5, true)), unit, "blobs normalized");
    assert_eq!(
        permutation(10, &mut SplitMix64::new(3)),
        [2, 8, 7, 4, 5, 6, 0, 1, 9, 3],
        "permutation(10, seed 3)"
    );
    assert_eq!(
        init_indices(10, 2000, 5),
        [622, 1127, 1266, 527, 555],
        "init_indices"
    );
    assert_eq!(
        epoch_order(20, 3000, 1, 8),
        [1298, 504, 522, 2221, 1341, 2786, 1182, 1154],
        "epoch_order"
    );
    assert_eq!(
        fnv1a64_f32(&blobs(30, 1500, 16, 12, 3.0, true)),
        0xD311_7FCE_AAA3_8EC1,
        "fnv1a64 of the spherical scenario data"
    );
}
