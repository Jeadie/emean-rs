"""Generate differential-test fixtures from the faiss e-means NumPy reference.

Usage:

    python3 -I tools/reference/generate_fixtures.py \
        --reference-dir <path to faiss contrib/e_means> --out-dir tests/fixtures

The reference package is imported from ``--reference-dir`` with an exact NumPy
stand-in for ``faiss`` (``faiss_shim.py``, next to this file). The output is a
pure function of the reference sources and this file: running the generator
twice gives byte-identical JSON. See README.md in this directory.
"""

import argparse
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The faiss commit the reference sources come from (contrib/e_means).
FAISS_COMMIT = "fc1095a9314f892938bf83f59e7ac966b5095def"


@dataclass(frozen=True)
class PinnedFile:
    """A reference file and its SHA-256 at FAISS_COMMIT."""

    name: str
    sha256: str


# The reference files at FAISS_COMMIT. The generator refuses other contents,
# and records the digests in the fixtures.
REFERENCE_FILES = (
    PinnedFile("__init__.py", "ad196786de8eb8ab7a229908b86ad8548a72459ade95e9e9c3275943dd50bb78"),
    PinnedFile("base.py", "bcffd7829ebfc25e1354308770b49e6f99d8566ba5e474a41a22e9ec414b17cd"),
    PinnedFile("e_means.py", "fc7750ba735b502d7831a27ccebb8c22b91c0405d74717a12c90df22eb85e06b"),
    PinnedFile(
        "e_means_spherical.py",
        "d4e4177593f70e284bd2f6ad8812170e2a830ea62c697c514edc6298a00ea7a6",
    ),
)

# Package name the reference is imported under (it uses relative imports).
REFERENCE_PACKAGE = "emeans_reference"

SCHEMA_VERSION = 1

# NumpyEmeans seed. It only drives the reference revival RNG
# (np.random.RandomState), which the Rust crate does not reproduce.
REFERENCE_SEED = 1234

MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# Shared synthetic data generator. tests/common/synthetic.rs mirrors this
# bit for bit; keep both in sync. Pure-Python loops keep the float operation
# order explicit.
# ---------------------------------------------------------------------------


@dataclass
class SplitMix64:
    """SplitMix64 generator; all arithmetic is modulo 2**64."""

    state: int

    def __post_init__(self):
        if not 0 <= self.state <= MASK64:
            raise ValueError("seed must be in [0, 2**64)")

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def uniform(self):
        """Float in [0, 1) from the top 53 bits."""
        return float(self.next_u64() >> 11) * 2.0**-53

    def below(self, m):
        """Integer in [0, m) by modulo (the bias is acceptable here)."""
        if m <= 0:
            raise ValueError("below() needs m >= 1")
        return self.next_u64() % m


def blobs(seed, n, d, centers, spread, normalize):
    """Gaussian-free blob data: an (n, d) float32 array, built in float64."""
    rng = SplitMix64(seed)
    center = [
        [(rng.uniform() * 2.0 - 1.0) * 10.0 for _ in range(d)]
        for _ in range(centers)
    ]
    values = []
    for _ in range(n):
        j = rng.below(centers)
        v = [center[j][t] + (rng.uniform() - 0.5) * (2.0 * spread) for t in range(d)]
        if normalize:
            s = 0.0
            for t in range(d):
                s += v[t] * v[t]
            norm = math.sqrt(s)
            # A zero row has probability ~0; it stays zero on both sides.
            if norm != 0.0:
                v = [vt / norm for vt in v]
        values.extend(v)
    return np.array(values, dtype=np.float64).astype(np.float32).reshape(n, d)


def permutation(m, rng):
    """Fisher-Yates shuffle of range(m), drawing j = rng.below(i + 1)."""
    a = list(range(m))
    for i in range(m - 1, 0, -1):
        j = rng.below(i + 1)
        a[i], a[j] = a[j], a[i]
    return a


def init_indices(seed, n, k):
    """Rows used as initial centroids for a scenario with data seed ``seed``."""
    return permutation(n, SplitMix64(seed + 1))[:k]


def epoch_order(seed, n, epoch, rows):
    """Row ids visited in 0-based ``epoch``, in order."""
    return permutation(n, SplitMix64(seed + 100 + epoch))[:rows]


def fnv1a64_f32(values):
    """FNV-1a 64 over the little-endian IEEE bytes of float32 values."""
    h = 0xCBF29CE484222325
    for byte in np.ascontiguousarray(values, dtype="<f4").tobytes():
        h ^= byte
        h = (h * 0x100000001B3) & MASK64
    return h


def train_mse(shim, x, centroids, spherical):
    """Mean over rows of the best score against ``centroids``, in float64.

    Euclidean: min squared L2 distance. Spherical: 2 - 2 * max inner product.
    Per-row scores accumulate sequentially over coordinates; the row sum is
    correctly rounded (math.fsum) and then divided by n.
    """
    if spherical:
        scores = shim.pairwise_scores(x, centroids, shim.METRIC_INNER_PRODUCT)
        per_row = 2.0 - 2.0 * scores.max(axis=1)
    else:
        scores = shim.pairwise_scores(x, centroids, shim.METRIC_L2)
        per_row = scores.min(axis=1)
    return math.fsum(per_row.tolist()) / len(x)


# ---------------------------------------------------------------------------
# Loading the reference.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reference:
    """Handles to the imported reference modules and the faiss shim."""

    base: object
    e_means: object
    shim: object
    files_sha256: dict


def load_module(name, path, search_locations=None):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=search_locations
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load %s from %s" % (name, path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_reference(reference_dir):
    # Keep the reference directory and this directory free of __pycache__
    # (python -I does not stop bytecode writes).
    sys.dont_write_bytecode = True
    reference_dir = Path(reference_dir).resolve()
    # Check the files before importing them.
    digests = {
        pinned.name: hashlib.sha256((reference_dir / pinned.name).read_bytes()).hexdigest()
        for pinned in REFERENCE_FILES
    }
    changed = [pinned.name for pinned in REFERENCE_FILES if digests[pinned.name] != pinned.sha256]
    if changed:
        raise SystemExit(
            f"{reference_dir}: {', '.join(changed)} differ from faiss commit {FAISS_COMMIT}"
        )
    shim = load_module("faiss", Path(__file__).resolve().with_name("faiss_shim.py"))
    if sys.modules.get("faiss") is not shim:
        raise RuntimeError("the faiss shim is not registered as 'faiss'")
    load_module(
        REFERENCE_PACKAGE,
        reference_dir / "__init__.py",
        search_locations=[str(reference_dir)],
    )
    base = importlib.import_module(REFERENCE_PACKAGE + ".base")
    e_means = importlib.import_module(REFERENCE_PACKAGE + ".e_means")
    if e_means.faiss is not shim:
        raise RuntimeError("the reference did not bind the faiss shim")
    return Reference(base=base, e_means=e_means, shim=shim, files_sha256=digests)


def normalized_nepochs(nepochs):
    """The epoch budget exactly as the NumpyEmeans constructor stores it."""
    nepochs = float(nepochs)
    return int(nepochs) if nepochs.is_integer() else nepochs


# ---------------------------------------------------------------------------
# reference_recipes.json
# ---------------------------------------------------------------------------

RECIPE_N = (1, 2, 3, 7, 100, 1000, 12345, 100000, 1000000, 10000000, 1000000000)
RECIPE_K = (1, 2, 10, 256, 16384, 1000000)
RECIPE_D = (2, 3, 16, 17, 128)
RECIPE_T = (0.5, 1.0, 1.4, 2.5, 3.0, 5.0, 25.0)
RECIPE_LAWS = ("auto", "throughput", "quality")
# Grid rows with n at or below this get every T in RECIPE_T.
TINY_N_MAX = 7


@dataclass(frozen=True)
class RecipeInputs:
    """Constructor options plus the dataset size n."""

    n: int
    k: int
    d: int
    nepochs: float
    batch_law: str
    batch_size: int | None = None
    alpha0: float | None = None
    unused_threshold: float | None = None


@dataclass
class RecipeExpected:
    """What the reference resolves for one RecipeInputs."""

    recipe_name: str
    batch_law: str | None
    batch_size: int
    alpha0: float
    epoch_rows: list
    epoch_steps: list
    total_steps: int
    steps_per_pass: int
    unused_threshold: float
    alpha_steps: list
    alphas: list


@dataclass
class RecipeCase:
    group: str
    inputs: RecipeInputs
    expected: RecipeExpected


@dataclass
class RaisingCase:
    group: str
    inputs: RecipeInputs
    error: str


@dataclass
class PowerOfTwoCase:
    input: float
    input_bits: str
    output: int


def resolve_recipe_case(ref, inputs):
    """Resolve a recipe with base.py functions; cross-check against begin()'s path."""
    base = ref.base
    nepochs = normalized_nepochs(inputs.nepochs)
    unused_threshold = base._resolve_unused_threshold(inputs.unused_threshold, inputs.d)
    law, _reason = base._resolve_batch_law(inputs.batch_law, inputs.d)
    batch_size, alpha0 = base._resolve_recipe(
        inputs.n, inputs.k, nepochs, law, inputs.batch_size, inputs.alpha0
    )
    epoch_rows = base._epoch_rows(inputs.n, nepochs)
    epoch_steps = [(rows + batch_size - 1) // batch_size for rows in epoch_rows]
    steps_per_pass = (inputs.n + batch_size - 1) // batch_size
    total_steps = sum(epoch_steps)
    batch_given = inputs.batch_size is not None
    recipe_name = (
        ("manual" if inputs.alpha0 is not None else "manual_batch")
        if batch_given
        else "canonical_v1"
    )
    steps = sorted(
        {t for t in (0, 1, total_steps // 2, total_steps - 1, total_steps) if t >= 0}
    )
    alphas = [base._scheduled_alpha(alpha0, t, total_steps) for t in steps]
    expected = RecipeExpected(
        recipe_name=recipe_name,
        batch_law=None if batch_given else law,
        batch_size=batch_size,
        alpha0=alpha0,
        epoch_rows=list(epoch_rows),
        epoch_steps=epoch_steps,
        total_steps=total_steps,
        steps_per_pass=steps_per_pass,
        unused_threshold=unused_threshold,
        alpha_steps=steps,
        alphas=alphas,
    )

    # The recipe begin() builds must agree with the values above.
    recipe = begin_recipe(ref, inputs)
    observed = (
        recipe["name"], recipe["batch_law"], recipe["batch_size"], recipe["alpha0"],
        recipe["epoch_rows"], recipe["epoch_steps"], recipe["total_steps"],
        recipe["steps_per_pass"], recipe["unused_threshold"],
    )
    wanted = (
        expected.recipe_name, expected.batch_law, expected.batch_size,
        expected.alpha0, expected.epoch_rows, expected.epoch_steps,
        expected.total_steps, expected.steps_per_pass, expected.unused_threshold,
    )
    if observed != wanted:
        raise AssertionError("recipe mismatch for %r: %r != %r" % (inputs, observed, wanted))
    return expected


def begin_recipe(ref, inputs):
    """The recipe dict NumpyEmeans.begin() would build (no data needed)."""
    probe = ref.e_means.NumpyEmeans(
        inputs.d,
        inputs.k,
        nepochs=inputs.nepochs,
        batch_size=inputs.batch_size,
        alpha0=inputs.alpha0,
        batch_law=inputs.batch_law,
        unused_threshold=inputs.unused_threshold,
    )
    probe._resolve_run_recipe(inputs.n)
    return probe.recipe


def raised_by(action):
    """Name of the exception type ``action()`` raises, or None."""
    try:
        action()
    except AssertionError:
        raise
    except Exception as error:  # noqa: BLE001 - recording the kind is the point
        return type(error).__name__
    return None


def recipe_inputs_grid():
    """(group, RecipeInputs) pairs, before de-duplication.

    The full n x k x d x T x law product is too large, so:
    * "grid": every (n, k <= n) pair. Tiny n (where epoch rows round to 0 or
      to an even half) gets every T; larger n gets 3 of the 7 T values, in
      rotation, so each n still meets several T. The law cycles through
      auto/throughput/quality and d cycles through RECIPE_D, so each law
      meets each d.
    * "auto_boundary": batch_law="auto" at every d, around the d = 2/3 and
      16/17 boundaries, on a few (n, k, T).
    * "k_equals_n": k == n beyond the n = 1 and n = 2 grid cases.
    * "epoch_rounding": fractional rows (T - floor(T)) * n at an exact half
      (ties to even: 2.5 -> 2, 5.5 -> 6) or a float64 near-half
      (5 * 0.1 == 0.5 -> 0, but 5 * (1.1 - 1) > 0.5 -> 1).
    * "explicit_batch": given batch_size (1, small, n, n + 1, larger), with
      and without alpha0.
    * "explicit_alpha": a law-chosen batch with a given alpha0 (given values
      are not clamped).
    """
    out = []
    pairs = [(n, k) for n in RECIPE_N for k in RECIPE_K if k <= n]
    index = 0
    for pair, (n, k) in enumerate(pairs):
        if n <= TINY_N_MAX:
            budgets = RECIPE_T
        else:
            budgets = [RECIPE_T[(pair + 2 * j) % len(RECIPE_T)] for j in range(3)]
        for nepochs in budgets:
            law = RECIPE_LAWS[index % len(RECIPE_LAWS)]
            d = RECIPE_D[(index // len(RECIPE_LAWS)) % len(RECIPE_D)]
            out.append(("grid", RecipeInputs(n=n, k=k, d=d, nepochs=nepochs, batch_law=law)))
            index += 1

    for n, k, nepochs in ((1000, 10, 1.0), (10000000, 16384, 1.4)):
        for d in RECIPE_D:
            out.append(
                ("auto_boundary", RecipeInputs(n=n, k=k, d=d, nepochs=nepochs, batch_law="auto"))
            )

    for n, nepochs, law in ((3, 1.4, "quality"), (7, 0.5, "throughput"),
                            (100, 2.5, "auto"), (1000, 3.0, "quality"),
                            (12345, 1.0, "throughput")):
        out.append(("k_equals_n", RecipeInputs(n=n, k=n, d=16, nepochs=nepochs, batch_law=law)))

    for i, (n, nepochs) in enumerate(((5, 2.5), (9, 2.5), (11, 2.5), (13, 0.5),
                                      (5, 1.1), (5, 0.1), (25, 0.1), (5, 3.3))):
        out.append(("epoch_rounding", RecipeInputs(
            n=n, k=1 + i % 2, d=16, nepochs=nepochs,
            batch_law=RECIPE_LAWS[i % len(RECIPE_LAWS)])))

    for n, k, d, nepochs in ((1, 1, 3, 1.0), (7, 2, 17, 2.5), (1000, 10, 16, 3.0),
                             (1000000000, 1000000, 2, 0.5)):
        for i, batch_size in enumerate((1, 3, 64, 1000, n, n + 1, 2 * n + 5)):
            out.append(("explicit_batch", RecipeInputs(
                n=n, k=k, d=d, nepochs=nepochs, batch_law="auto",
                batch_size=batch_size, alpha0=None if i % 2 else 0.05)))

    for n, k, d, nepochs, law in ((2, 2, 2, 25.0, "quality"),
                                  (1000, 10, 16, 3.0, "auto"),
                                  (1000000, 256, 128, 1.4, "throughput")):
        for alpha0 in (1.0, 0.5, 1e-06):
            out.append(("explicit_alpha", RecipeInputs(
                n=n, k=k, d=d, nepochs=nepochs, batch_law=law, alpha0=alpha0)))
    return out


def raising_inputs():
    """Inputs the reference rejects with an exception."""
    return [
        ("unknown_batch_law", RecipeInputs(n=1000, k=10, d=16, nepochs=3.0, batch_law="bogus")),
        ("zero_batch_size", RecipeInputs(n=1000, k=10, d=16, nepochs=3.0, batch_law="auto", batch_size=0)),
        ("zero_batch_size_given_alpha", RecipeInputs(
            n=1000, k=10, d=16, nepochs=3.0, batch_law="auto", batch_size=0, alpha0=0.1)),
        ("negative_batch_size", RecipeInputs(
            n=1000, k=10, d=16, nepochs=3.0, batch_law="auto", batch_size=-1)),
        ("zero_nepochs", RecipeInputs(n=1000, k=10, d=16, nepochs=0.0, batch_law="auto")),
        ("negative_nepochs", RecipeInputs(n=1000, k=10, d=16, nepochs=-1.0, batch_law="auto")),
        ("zero_n", RecipeInputs(n=0, k=1, d=16, nepochs=3.0, batch_law="auto")),
        ("zero_k", RecipeInputs(n=1000, k=0, d=16, nepochs=3.0, batch_law="auto")),
        ("negative_unused_threshold", RecipeInputs(
            n=1000, k=10, d=16, nepochs=3.0, batch_law="auto", unused_threshold=-0.5)),
    ]


def round_power_of_two_inputs():
    values = [-1.0, 0.0, 0.25, 0.5, 1.0, math.nextafter(1.0, math.inf), 1.4, 1.5,
              2.0, 2.5, 3.0, 5.0, 6.0, 7.0, 12345.0, 1e6, 1e9, 1e12, 1e15, 1e18]
    for m in (0, 1, 2, 3, 4, 7, 10, 16, 20, 31, 32, 40, 52, 53, 60):
        mid = 2.0**m * math.sqrt(2.0)
        values += [math.nextafter(mid, -math.inf), mid, math.nextafter(mid, math.inf)]
    for m in (1, 10, 53, 60):
        values += [math.nextafter(2.0**m, 0.0), 2.0**m, math.nextafter(2.0**m, math.inf)]
    return sorted(set(values))


def build_recipes(ref):
    cases = []
    seen = set()
    for group, inputs in recipe_inputs_grid():
        if inputs in seen:
            continue
        seen.add(inputs)
        try:
            expected = resolve_recipe_case(ref, inputs)
        except AssertionError:
            raise
        except Exception as error:  # noqa: BLE001 - the grid must not raise
            raise RuntimeError("grid case raised %r: %r" % (inputs, error)) from error
        cases.append(RecipeCase(group=group, inputs=inputs, expected=expected))

    raising = []
    for group, inputs in raising_inputs():
        error = raised_by(lambda: resolve_recipe_case(ref, inputs))
        begin_error = raised_by(lambda: begin_recipe(ref, inputs))
        if error is None or error != begin_error:
            raise RuntimeError("expected %s to raise the same error on both paths, got "
                               "%r and %r" % (inputs, error, begin_error))
        raising.append(RaisingCase(group=group, inputs=inputs, error=error))

    powers = [
        PowerOfTwoCase(
            input=value,
            input_bits="%016x" % struct.unpack("<Q", struct.pack("<d", value))[0],
            output=ref.base._round_power_of_two(value),
        )
        for value in round_power_of_two_inputs()
    ]

    base = ref.base
    constants = {
        "alpha_law": list(base._ALPHA_LAW),
        "alpha_min_factor": base._ALPHA_MIN_FACTOR,
        "auto_quality_max_d": base._AUTO_QUALITY_MAX_D,
        "batch_laws": {name: list(law) for name, law in base._BATCH_LAWS.items()},
        "check_unused_every": base._CHECK_UNUSED_EVERY,
        "init_usage": base._INIT_USAGE,
        "no_revival_max_d": base._NO_REVIVAL_MAX_D,
        "split_epsilon": base._SPLIT_EPSILON,
        "unused_threshold": base._UNUSED_THRESHOLD,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source_block(ref),
        "constants": constants,
        "cases": [dataclasses.asdict(case) for case in cases],
        "raising_cases": [dataclasses.asdict(case) for case in raising],
        "round_power_of_two": [dataclasses.asdict(case) for case in powers],
    }


# ---------------------------------------------------------------------------
# reference_runs.json
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunInputs:
    """One end-to-end run: blobs data, deterministic init and row orders."""

    name: str
    seed: int
    n: int
    d: int
    k: int
    centers: int
    spread: float
    nepochs: float
    normalize: bool = False
    spherical: bool = False
    batch_size: int | None = None
    alpha0: float | None = None
    batch_law: str = "auto"
    unused_threshold: float | None = 0.0
    reference_seed: int = REFERENCE_SEED


@dataclass
class RunRecipe:
    recipe_name: str
    batch_law: str | None
    batch_law_resolved: str
    batch_size: int
    alpha0: float
    epoch_rows: list
    epoch_steps: list
    total_steps: int
    steps_per_pass: int
    unused_threshold: float


@dataclass
class EpochStat:
    obj: float
    imbalance_factor: float
    nsplit: int


@dataclass
class DataCheck:
    """Values that let a port check its generator before comparing results."""

    head: list
    fnv1a64: str
    init_indices: list
    epoch_order_heads: list


@dataclass
class RunResult:
    inputs: RunInputs
    data_check: DataCheck
    recipe: RunRecipe
    epochs: list
    n_splits: int
    centroids: list
    labels: list
    train_mse: float


@dataclass
class RevivalResult:
    inputs: RunInputs
    unused_threshold_resolved: float
    n_splits: int
    epochs: list
    train_mse: float


RUN_SCENARIOS = (
    RunInputs(name="euclid_auto_quality", seed=10, n=2000, d=8, k=16, centers=12,
              spread=1.5, nepochs=3.0),
    RunInputs(name="euclid_fractional_throughput", seed=20, n=3000, d=24, k=10,
              centers=10, spread=2.0, nepochs=2.5),
    RunInputs(name="spherical", seed=30, n=1500, d=16, k=12, centers=12, spread=3.0,
              nepochs=2.0, normalize=True, spherical=True),
    RunInputs(name="alpha_one_full_batch", seed=40, n=600, d=4, k=6, centers=6,
              spread=1.0, nepochs=2.0, batch_size=600, alpha0=1.0),
    RunInputs(name="manual_batch_step_law", seed=50, n=1200, d=32, k=20, centers=8,
              spread=2.5, nepochs=1.5, batch_size=100),
    RunInputs(name="epochs_below_one", seed=60, n=800, d=6, k=8, centers=8,
              spread=1.0, nepochs=0.6),
)

REVIVAL_SEEDS = tuple(range(1000, 1010))


def revival_inputs():
    out = []
    for name, threshold in (("revival_default_threshold", None), ("revival_off", 0.0)):
        for seed in REVIVAL_SEEDS:
            out.append(RunInputs(name=name, seed=seed, n=4000, d=8, k=64, centers=8,
                                 spread=1.0, nepochs=3.0, unused_threshold=threshold))
    return out


@dataclass
class Trace:
    """Everything one reference run produced."""

    x: object
    init: list
    orders: list
    model: object
    labels: list
    centroids: object


def run_reference(ref, inputs):
    """Drive NumpyEmeans with begin/accumulate/update/end_epoch/finalize."""
    x = blobs(inputs.seed, inputs.n, inputs.d, inputs.centers, inputs.spread,
              inputs.normalize)
    init = init_indices(inputs.seed, inputs.n, inputs.k)
    model = ref.e_means.NumpyEmeans(
        inputs.d,
        inputs.k,
        nepochs=inputs.nepochs,
        batch_size=inputs.batch_size,
        alpha0=inputs.alpha0,
        batch_law=inputs.batch_law,
        seed=inputs.reference_seed,
        capture_assignments=True,
        spherical=inputs.spherical,
        unused_threshold=inputs.unused_threshold,
    )
    model.begin(inputs.n, x[init])
    epoch_rows = list(model.recipe["epoch_rows"])
    batch = model.recipe["batch_size"]
    last = len(epoch_rows) - 1
    labels = [-1] * inputs.n
    orders = []
    for epoch, rows in enumerate(epoch_rows):
        order = epoch_order(inputs.seed, inputs.n, epoch, rows)
        orders.append(order)
        for begin in range(0, rows, batch):
            ids = order[begin:begin + batch]
            stats = model.accumulate(x[ids])
            if ("labels" in stats) != (epoch == last):
                raise AssertionError("labels must appear in the last epoch only")
            if epoch == last:
                for row, label in zip(ids, stats["labels"].tolist()):
                    labels[row] = label
            model.update(stats)
        model.end_epoch()
    centroids = model.finalize()
    return Trace(x=x, init=init, orders=orders, model=model, labels=labels,
                 centroids=centroids)


def run_recipe_of(model):
    recipe = model.recipe
    return RunRecipe(
        recipe_name=recipe["name"],
        batch_law=recipe["batch_law"],
        batch_law_resolved=model._batch_law_resolved,
        batch_size=recipe["batch_size"],
        alpha0=float(recipe["alpha0"]),
        epoch_rows=list(recipe["epoch_rows"]),
        epoch_steps=list(recipe["epoch_steps"]),
        total_steps=recipe["total_steps"],
        steps_per_pass=recipe["steps_per_pass"],
        unused_threshold=float(recipe["unused_threshold"]),
    )


def epoch_stats_of(model):
    return [
        EpochStat(obj=float(stat["obj"]),
                  imbalance_factor=float(stat["imbalance_factor"]),
                  nsplit=int(stat["nsplit"]))
        for stat in model.iteration_stats
    ]


def build_runs(ref):
    scenarios = []
    for inputs in RUN_SCENARIOS:
        trace = run_reference(ref, inputs)
        flat = trace.x.ravel()
        scenarios.append(RunResult(
            inputs=inputs,
            data_check=DataCheck(
                head=[float(v) for v in flat[:8]],
                fnv1a64="%016x" % fnv1a64_f32(flat),
                init_indices=list(trace.init),
                epoch_order_heads=[order[:8] for order in trace.orders],
            ),
            recipe=run_recipe_of(trace.model),
            epochs=epoch_stats_of(trace.model),
            n_splits=int(trace.model.n_splits),
            centroids=[[float(v) for v in row] for row in trace.centroids],
            labels=trace.labels,
            train_mse=train_mse(ref.shim, trace.x, trace.centroids, inputs.spherical),
        ))

    revival = []
    for inputs in revival_inputs():
        trace = run_reference(ref, inputs)
        revival.append(RevivalResult(
            inputs=inputs,
            unused_threshold_resolved=float(trace.model.recipe["unused_threshold"]),
            n_splits=int(trace.model.n_splits),
            epochs=epoch_stats_of(trace.model),
            train_mse=train_mse(ref.shim, trace.x, trace.centroids, inputs.spherical),
        ))

    return {
        "schema_version": SCHEMA_VERSION,
        "source": source_block(ref),
        "scenarios": [dataclasses.asdict(result) for result in scenarios],
        "revival_statistics": [dataclasses.asdict(result) for result in revival],
    }


# ---------------------------------------------------------------------------
# Output.
# ---------------------------------------------------------------------------


def source_block(ref):
    return {
        "faiss_commit": FAISS_COMMIT,
        "generator": "tools/reference/generate_fixtures.py",
        "reference_sha256": dict(ref.files_sha256),
    }


def write_json(path, payload):
    text = json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference-dir", required=True,
                        help="directory of faiss contrib/e_means")
    parser.add_argument("--out-dir", required=True, help="fixture output directory")
    args = parser.parse_args(argv)

    ref = load_reference(args.reference_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "reference_recipes.json", build_recipes(ref))
    write_json(out_dir / "reference_runs.json", build_runs(ref))


if __name__ == "__main__":
    main()
