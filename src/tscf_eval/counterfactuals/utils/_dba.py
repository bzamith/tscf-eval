"""DTW Barycenter Averaging (DBA) utilities for counterfactual explainers.

Functions
---------
weighted_dba_multich
    Weighted DTW barycenter averaging per channel (multivariate).
dba_barycenter_multich
    DTW barycenter averaging for multiple sequences.
"""

from __future__ import annotations

import warnings

import numpy as np

# Optional: tslearn for DTW-barycenter averaging
try:
    from tslearn.barycenters import dtw_barycenter_averaging

    TSLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    dtw_barycenter_averaging = None  # type: ignore
    TSLEARN_AVAILABLE = False

# Track whether the tslearn fallback warning has been issued to avoid flooding.
_tslearn_warned = False


def _warn_tslearn_fallback(func_name: str) -> None:
    """Emit an at-most-once warning about tslearn being unavailable.

    Parameters
    ----------
    func_name : str
        Name of the calling function, included in the warning message.
    """
    global _tslearn_warned  # noqa: PLW0603
    if not _tslearn_warned:
        warnings.warn(
            f"tslearn is not installed. {func_name}() is using a simple fallback "
            f"instead of DTW-based computation. Results may differ from the "
            f"intended algorithm. Install tslearn for proper DTW support: "
            f"pip install tslearn",
            UserWarning,
            stacklevel=3,
        )
        _tslearn_warned = True


def weighted_dba_multich(query: np.ndarray, guide: np.ndarray, weight_guide: float) -> np.ndarray:
    """Weighted DTW barycenter averaging using tslearn.

    Matches the original NativeGuide paper: passes exactly two sequences
    (query and guide) with weights ``[(1 - weight_guide), weight_guide]`` to
    ``tslearn.barycenters.dtw_barycenter_averaging``, which runs an iterative
    EM algorithm (default ``max_iter=30``).

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
    if not TSLEARN_AVAILABLE:
        _warn_tslearn_fallback("weighted_dba_multich")
        weight_a = 1.0 - weight_guide
        return weight_a * query + weight_guide * guide

    weights = np.array([1.0 - weight_guide, weight_guide])

    if query.ndim == 1:  # (T,)
        # tslearn expects (n_ts, sz, d) = (2, T, 1)
        stacked = np.stack([query[:, np.newaxis], guide[:, np.newaxis]])
        result = dtw_barycenter_averaging(stacked, weights=weights)  # (T, 1)
        ravel_result: np.ndarray = result.ravel()
        return ravel_result

    if query.ndim == 2:  # (C, T) -> tslearn needs (2, T, C)
        stacked = np.stack([query.T, guide.T])  # (2, T, C)
        result = dtw_barycenter_averaging(stacked, weights=weights)  # (T, C)
        out: np.ndarray = result.T  # (C, T)
        return out

    raise ValueError(f"Unsupported query shape: {query.shape}")


def dba_barycenter_multich(neighbors: np.ndarray) -> np.ndarray:
    """DTW barycenter averaging using tslearn.

    Delegates to ``tslearn.barycenters.dtw_barycenter_averaging`` which
    computes the barycenter via an iterative EM algorithm.

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
            _warn_tslearn_fallback("dba_barycenter_multich")
            result: np.ndarray = neighbors.mean(axis=0)
            return result
        # tslearn expects (K, T, 1)
        stacked = neighbors[:, :, np.newaxis]
        bary = dtw_barycenter_averaging(stacked)  # (T, 1)
        bary_result: np.ndarray = bary.ravel()
        return bary_result

    if neighbors.ndim == 3:  # (K, C, T)
        if not TSLEARN_AVAILABLE:
            _warn_tslearn_fallback("dba_barycenter_multich")
            mean_result: np.ndarray = neighbors.mean(axis=0)  # (C, T)
            return mean_result
        # (K, C, T) -> (K, T, C) for tslearn
        stacked = np.transpose(neighbors, (0, 2, 1))
        bary = dtw_barycenter_averaging(stacked)  # (T, C)
        out: np.ndarray = bary.T  # (C, T)
        return out

    raise ValueError(f"neighbors shape not supported for DBA: {neighbors.shape}")
