"""Shape manipulation utilities for counterfactual explainers.

Functions
---------
ensure_batch_shape
    Ensures array has a leading batch dimension.
strip_batch
    Removes batch dimension added by ``ensure_batch_shape``.
"""

from __future__ import annotations

import numpy as np


def ensure_batch_shape(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return an array with a leading batch axis.

    This helper is intended for callers that may receive a single time-series
    instance (either univariate ``(T,)`` or multivariate ``(C, T)``) and want
    to ensure a batch axis for model calls. It will convert:

    - ``(T,)`` -> ``(1, T)``
    - ``(C, T)`` -> ``(1, C, T)``
    - ``(N, C, T)`` -> unchanged (already batched)

    Note: callers that already have a batch of univariate series ``(N, T)``
    should pass a 3-D array ``(N, 1, T)`` or avoid calling this helper. The
    function prefers the interpretation that a 2-D array is a single series
    (possibly multivariate) to keep the wrapper safe for the common single-
    instance code paths used in explainers.

    Parameters
    ----------
    x : np.ndarray
        Time series with shape ``(T,)``, ``(C, T)``, or ``(N, C, T)``.

    Returns
    -------
    xb : np.ndarray
        Array with a leading batch axis (shape ``(1, ...)`` or unchanged).
    added : bool
        True if a batch axis was added by this function.

    Raises
    ------
    ValueError
        If the input has unsupported number of dimensions.
    """
    arr = np.asarray(x)
    if arr.ndim == 1:  # (T,) -> (1, T)
        return arr[None, :], True
    if arr.ndim == 2:
        # Interpret 2-D inputs as a single multivariate series (C, T)
        # and add a batch axis to produce (1, C, T).
        return arr[None, ...], True
    if arr.ndim == 3:  # already (N, C, T)
        return arr, False
    raise ValueError(f"Unsupported shape: {arr.shape}")


def strip_batch(xb: np.ndarray, added: bool) -> np.ndarray:
    """Remove a previously added batch axis.

    Parameters
    ----------
    xb : np.ndarray
        Array with a leading batch axis.
    added : bool
        Whether the batch axis was artificially added by
        :func:`ensure_batch_shape`.

    Returns
    -------
    np.ndarray
        The array with the leading batch axis removed when ``added`` is
        True, otherwise ``xb`` unchanged.
    """
    return xb[0] if added else xb
