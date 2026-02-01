"""Shared utilities for metric implementations."""

from __future__ import annotations

import numpy as np


def ensure_array(X: np.ndarray) -> np.ndarray:
    """Convert input to numpy array if not already.

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    np.ndarray
        Input as numpy array.
    """
    return np.asarray(X)
