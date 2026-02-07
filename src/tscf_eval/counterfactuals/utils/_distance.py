"""Distance functions for counterfactual explainers.

Functions
---------
euclidean_cdist_flat
    Computes pairwise Euclidean distances after flattening.
dtw_distance_vec_multich
    Computes DTW distance averaged across channels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
import warnings

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

from ._shape import ensure_batch_shape

# Optional: tslearn for DTW
try:
    from tslearn.metrics import dtw as tslearn_dtw

    TSLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    tslearn_dtw = None  # type: ignore
    TSLEARN_AVAILABLE = False

# Optional: SciPy for fast euclidean pairwise distance
try:
    from scipy.spatial.distance import cdist

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    cdist = None  # type: ignore
    SCIPY_AVAILABLE = False


def euclidean_cdist_flat(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Euclidean distances between A and B after flattening non-batch dims.

    Parameters
    ----------
    A : np.ndarray
        Array of shape (Na, T) or (Na, C, T).
    B : np.ndarray
        Array of shape (Nb, T) or (Nb, C, T).

    Returns
    -------
    np.ndarray
        Distance matrix of shape (Na, Nb) with Euclidean distances after
        flattening each sample to a vector.
    """
    A2 = A.reshape(A.shape[0], -1)
    B2 = B.reshape(B.shape[0], -1)
    if SCIPY_AVAILABLE:
        cdist_fn = cast("Callable[..., np.ndarray]", cdist)
        return cdist_fn(A2, B2, metric="euclidean")
    warnings.warn(
        "scipy is not installed. euclidean_cdist_flat() is using a manual NumPy "
        "fallback which may be slower for large arrays. "
        "Install scipy for optimized computation: pip install scipy",
        UserWarning,
        stacklevel=2,
    )
    diff = A2[:, None, :] - B2[None, :, :]
    result: np.ndarray = np.sqrt(np.sum(diff * diff, axis=2))
    return result


def dtw_distance_vec_multich(x: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-channel DTW distance averaged across channels.

    Parameters
    ----------
    x : np.ndarray
        Query series, either (T,) for univariate or (C, T) for multivariate.
    B : np.ndarray
        Candidate series array, either (N, T) or (N, C, T).

    Returns
    -------
    np.ndarray
        1-D array of length N with the average per-channel DTW distance
        between ``x`` and each series in ``B``.

    Notes
    -----
    If ``tslearn`` is unavailable this falls back to Euclidean distances
    computed after flattening the channels/time dimensions.
    """
    if not TSLEARN_AVAILABLE:
        warnings.warn(
            "tslearn is not installed. dtw_distance_vec_multich() is falling back "
            "to Euclidean distance, which ignores temporal alignment. "
            "Install tslearn for proper DTW distances: pip install tslearn",
            UserWarning,
            stacklevel=2,
        )
        xb, _ = ensure_batch_shape(x)
        return euclidean_cdist_flat(xb, B).ravel()

    # Normalize shapes to (C, T) for x and (N, C, T) for B
    if x.ndim == 1:  # (T,)
        x_c = x[None, :]
    elif x.ndim == 2:  # (C, T)
        x_c = x
    else:
        raise ValueError(f"x shape not supported: {x.shape}")

    if B.ndim == 2:  # (N, T) -> (N, 1, T)
        B_c = B[:, None, :]
    elif B.ndim == 3:  # (N, C, T)
        B_c = B
    else:
        raise ValueError(f"B shape not supported: {B.shape}")

    Cx, _ = x_c.shape
    NB = B_c.shape[0]
    dists = np.zeros(NB, dtype=float)

    for i in range(NB):
        bi = B_c[i]
        Cb = bi.shape[0]
        Cuse = min(Cx, Cb)
        acc = 0.0
        tslearn_dtw_fn = cast("Callable[[np.ndarray, np.ndarray], float]", tslearn_dtw)
        for c in range(Cuse):
            acc += tslearn_dtw_fn(x_c[c], bi[c])
        dists[i] = acc / Cuse
    return dists
