"""DTW Barycenter Averaging (DBA) utilities for counterfactual explainers.

Functions
---------
dtw_pair_average
    DTW-based pairwise averaging of two sequences.
weighted_dba_pair
    Weighted DTW barycenter averaging for two sequences.
weighted_dba_multich
    Weighted DTW barycenter averaging per channel (multivariate).
dba_barycenter_multich
    DTW barycenter averaging for multiple sequences.

CAM Replacement Helpers
-----------------------
replace_topk_univariate
    Replace top-k important points in univariate series.
replace_topk_multivariate
    Replace top-k important entries in multivariate series.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

# Optional: tslearn for DTW and DTW-barycenter averaging
try:
    from tslearn.metrics import dtw_path

    TSLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    dtw_path = None  # type: ignore
    TSLEARN_AVAILABLE = False


def dtw_pair_average(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """DTW pairwise average for two 1-D sequences.

    Parameters
    ----------
    a, b : np.ndarray
        1-D sequences with the same nominal length (T,). If ``tslearn`` is
        unavailable a simple arithmetic mean is returned.

    Returns
    -------
    np.ndarray
        Averaged sequence of shape (T,) computed by aligning ``b`` to ``a``
        via the DTW path and averaging corresponding points.
    """
    if not TSLEARN_AVAILABLE:
        result: np.ndarray = 0.5 * (a + b)
        return result
    dtw_path_fn = cast("Callable[[np.ndarray, np.ndarray], Any]", dtw_path)
    path, _ = dtw_path_fn(a, b)
    T = len(a)
    aligned: list[list[float]] = [[] for _ in range(T)]
    for i, j in path:
        if 0 <= i < T:
            aligned[i].append(b[j])
    out = np.empty_like(a, dtype=float)
    for i in range(T):
        out[i] = 0.5 * (a[i] + (float(np.mean(aligned[i])) if aligned[i] else float(a[i])))
    return out


def weighted_dba_pair(a: np.ndarray, b: np.ndarray, weight_b: float) -> np.ndarray:
    """Weighted DTW barycenter averaging for two 1-D sequences.

    Parameters
    ----------
    a : np.ndarray
        First sequence of shape (T,), typically the query.
    b : np.ndarray
        Second sequence of shape (T,), typically the native guide.
    weight_b : float
        Weight for sequence b in [0, 1]. The weight for a is (1 - weight_b).

    Returns
    -------
    np.ndarray
        Weighted averaged sequence of shape (T,).
    """
    weight_a = 1.0 - weight_b
    if not TSLEARN_AVAILABLE:
        return weight_a * a + weight_b * b
    dtw_path_fn = cast("Callable[[np.ndarray, np.ndarray], Any]", dtw_path)
    path, _ = dtw_path_fn(a, b)
    T = len(a)
    aligned: list[list[float]] = [[] for _ in range(T)]
    for i, j in path:
        if 0 <= i < T:
            aligned[i].append(b[j])
    out = np.empty_like(a, dtype=float)
    for i in range(T):
        b_avg = float(np.mean(aligned[i])) if aligned[i] else float(b[min(i, len(b) - 1)])
        out[i] = weight_a * a[i] + weight_b * b_avg
    return out


def weighted_dba_multich(query: np.ndarray, guide: np.ndarray, weight_guide: float) -> np.ndarray:
    """Weighted DTW barycenter averaging per channel.

    This implements the original NativeGuide paper's weighted blending approach.

    Parameters
    ----------
    query : np.ndarray
        Query series of shape (T,) for univariate or (C, T) for multivariate.
    guide : np.ndarray
        Native guide series of same shape as query.
    weight_guide : float
        Weight for the guide in [0, 1]. The weight for query is (1 - weight_guide).

    Returns
    -------
    np.ndarray
        Weighted averaged series of same shape as input.
    """
    if query.ndim == 1:  # (T,)
        return weighted_dba_pair(query, guide, weight_guide)

    if query.ndim == 2:  # (C, T)
        C, _T = query.shape
        out = np.empty_like(query, dtype=float)
        for c in range(C):
            out[c] = weighted_dba_pair(query[c], guide[c], weight_guide)
        return out

    raise ValueError(f"Unsupported query shape: {query.shape}")


def dba_barycenter_multich(neighbors: np.ndarray) -> np.ndarray:
    """DTW barycenter averaging per channel.

    Parameters
    ----------
    neighbors : np.ndarray
        Neighbor series with shape (K, T) for univariate or (K, C, T) for
        multivariate.

    Returns
    -------
    np.ndarray
        Averaged series. Shape is (T,) for univariate input and (C, T)
        for multivariate input.

    Raises
    ------
    ValueError
        If ``neighbors`` has unsupported number of dimensions.
    """
    if neighbors.ndim == 2:  # (K, T)
        if not TSLEARN_AVAILABLE:
            result: np.ndarray = neighbors.mean(axis=0)
            return result
        g = neighbors[0].copy()
        for i in range(1, neighbors.shape[0]):
            g = dtw_pair_average(g, neighbors[i])
        g_result: np.ndarray = g
        return g_result

    if neighbors.ndim == 3:  # (K, C, T)
        K, C, T = neighbors.shape
        if not TSLEARN_AVAILABLE:
            mean_result: np.ndarray = neighbors.mean(axis=0)  # (C, T)
            return mean_result
        guide = np.empty((C, T), dtype=float)
        for c in range(C):
            gc = neighbors[0, c].copy()
            for i in range(1, K):
                gc = dtw_pair_average(gc, neighbors[i, c])
            guide[c] = gc
        return guide

    raise ValueError(f"neighbors shape not supported for DBA: {neighbors.shape}")


def replace_topk_univariate(
    x: np.ndarray, neigh: np.ndarray, imp: np.ndarray, p: float = 0.2
) -> np.ndarray:
    """Replace the top-p important points in a univariate series with neighbor values.

    Parameters
    ----------
    x : np.ndarray
        Univariate series of shape (T,).
    neigh : np.ndarray
        Neighbor series of shape (T,) used to replace important points.
    imp : np.ndarray
        Importance scores of shape (T,) where larger is more important.
    p : float, optional
        Fraction of points to replace (0 < p <= 1), by default 0.2.

    Returns
    -------
    np.ndarray
        Modified series with the top-k important points replaced by ``neigh``.
    """
    T = x.shape[0]
    k = max(1, int(np.round(p * T)))
    idx = np.argsort(-imp)[:k]
    g = x.copy()
    g[idx] = neigh[idx]
    return g


def replace_topk_multivariate(
    x: np.ndarray, neigh: np.ndarray, imp: np.ndarray, p: float = 0.2
) -> np.ndarray:
    """Replace the top-p important entries in a multivariate series with
    neighbor values.

    Parameters
    ----------
    x : np.ndarray
        Multivariate series of shape (C, T).
    neigh : np.ndarray
        Neighbor series of shape (C, T) used to replace important points.
    imp : np.ndarray
        Importance matrix of shape (C, T) or flattened importance scores.
    p : float, optional
        Fraction of entries to replace, by default 0.2.

    Returns
    -------
    np.ndarray
        Modified multivariate series with the top-k entries replaced by ``neigh``.
    """
    # x, neigh, imp: (C, T)
    C, T = x.shape
    k = max(1, int(np.round(p * C * T)))
    flat_idx = np.atleast_1d(np.argsort(-imp.ravel())[:k])
    rr_cc = cast("tuple[np.ndarray, np.ndarray]", np.unravel_index(flat_idx, (C, T)))
    rr, cc = rr_cc
    g = x.copy()
    g[rr, cc] = neigh[rr, cc]
    return g
