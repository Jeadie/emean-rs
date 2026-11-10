"""Exact NumPy stand-in for the ``faiss`` calls made by the e-means reference.

The faiss ``contrib/e_means`` NumPy reference (``e_means.py``) only calls
``faiss.knn(x, y, 1)`` and ``faiss.knn(x, y, 1, metric=METRIC_INNER_PRODUCT)``.
Real faiss picks between a direct loop and a BLAS ``||x||^2 - 2xy + ||y||^2``
expansion depending on the sizes, so its float32 distances and its choice
between near-tied centroids depend on the build and the batch shape. This
module replaces it with a definition that is exact and easy to reproduce:

* Scores are computed in float64 from the float32 inputs. For each query row
  and each database row the score is accumulated sequentially over the
  coordinates ``t = 0 .. d-1``:
  L2:  ``s += (x[t] - y[t]) * (x[t] - y[t])``
  IP:  ``s += x[t] * y[t]``
  Every step is one IEEE-754 float64 operation (no FMA, no pairwise sum).
* Ties go to the lowest database index (first minimum for L2, first maximum
  for inner product).
* Like faiss, the result is ``(D, I)`` with ``D`` float32 of shape (n, k) and
  ``I`` int64 of shape (n, k). ``D`` is the float64 score rounded to float32.

Register it with ``sys.modules["faiss"] = <this module>`` before importing
the reference package.
"""

import numpy as np

# Same numeric values as faiss.MetricType.
METRIC_INNER_PRODUCT = 0
METRIC_L2 = 1

# Query rows per score block; bounds scratch memory to rows * len(database).
_QUERY_BLOCK = 4096


def pairwise_scores(queries, database, metric=METRIC_L2):
    """Return float64 scores of shape (len(queries), len(database)).

    L2 scores are squared Euclidean distances, IP scores are dot products.
    Both are accumulated in coordinate order, one float64 operation at a time.
    """
    x = np.asarray(queries, dtype=np.float64)
    y = np.asarray(database, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(
            "queries and database must be 2-D with the same dimension, got "
            "%r and %r" % (x.shape, y.shape)
        )
    scores = np.zeros((x.shape[0], y.shape[0]), dtype=np.float64)
    for t in range(x.shape[1]):
        column = x[:, t, None]
        row = y[None, :, t]
        if metric == METRIC_L2:
            diff = column - row
            scores += diff * diff
        elif metric == METRIC_INNER_PRODUCT:
            scores += column * row
        else:
            raise ValueError("unsupported metric %r" % (metric,))
    return scores


def knn(xq, xb, k, metric=METRIC_L2):
    """Exact brute-force k nearest neighbours of each row of ``xq`` in ``xb``.

    Return ``(D, I)``: float32 distances (L2) or similarities (IP), and int64
    database indices, both of shape (len(xq), k). Ties resolve to the lowest
    database index.
    """
    k = int(k)
    xq = np.ascontiguousarray(xq, dtype=np.float32)
    xb = np.ascontiguousarray(xb, dtype=np.float32)
    if not 1 <= k <= len(xb):
        raise ValueError("k must be in [1, len(xb)], got %d" % k)
    n = len(xq)
    distances = np.empty((n, k), dtype=np.float32)
    labels = np.empty((n, k), dtype=np.int64)
    for begin in range(0, n, _QUERY_BLOCK):
        end = min(begin + _QUERY_BLOCK, n)
        scores = pairwise_scores(xq[begin:end], xb, metric)
        # Negation is exact, so a stable ascending sort of -IP keeps the
        # lowest index first among equal similarities.
        keys = scores if metric == METRIC_L2 else -scores
        if k == 1:
            order = np.argmin(keys, axis=1)[:, None]
        else:
            order = np.argsort(keys, axis=1, kind="stable")[:, :k]
        distances[begin:end] = np.take_along_axis(scores, order, axis=1)
        labels[begin:end] = order
    return distances, labels
