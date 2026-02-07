"""Nearest Unlike Neighbor (NUN) search utilities.

This module provides a shared implementation for finding the nearest
unlike neighbor(s) in a reference dataset, used by multiple counterfactual
explainers (CELS, SETS).

Functions
---------
find_nearest_unlike_neighbor
    Find k nearest instances of a target class in a reference dataset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import warnings

import numpy as np

from ._distance import euclidean_cdist_flat

if TYPE_CHECKING:
    from collections.abc import Callable


def find_nearest_unlike_neighbor(
    x: np.ndarray,
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    target_class: Any,
    *,
    distance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    k: int = 1,
    fallback_labels: np.ndarray | None = None,
    fallback_all: bool = False,
) -> tuple[list[np.ndarray], list[int]]:
    """Find k nearest neighbors of a given target class in a reference set.

    Searches ``X_ref`` for instances whose label in ``y_ref`` matches
    ``target_class``, then returns the ``k`` closest to ``x`` according
    to ``distance_fn``.

    Parameters
    ----------
    x : np.ndarray
        Query instance of shape ``(T,)`` or ``(C, T)``.
    X_ref : np.ndarray
        Reference dataset of shape ``(N, T)`` or ``(N, C, T)``.
    y_ref : np.ndarray
        Labels for each instance in ``X_ref``, shape ``(N,)``.
    target_class : Any
        Target class label to filter by.
    distance_fn : callable, optional
        Function ``(x_batch, X_candidates) -> distances`` returning a 1-D
        array of distances. If ``None``, uses ``euclidean_cdist_flat``.
    k : int, optional
        Number of nearest neighbors to return. Default is 1.
    fallback_labels : np.ndarray, optional
        Alternative label array to try if no instances of ``target_class``
        exist in ``y_ref``. Useful when ``y_ref`` contains model predictions
        and ``fallback_labels`` contains ground-truth labels.
    fallback_all : bool, optional
        If ``True`` and no instances match even after trying
        ``fallback_labels``, search all instances regardless of class.
        Default is ``False``.

    Returns
    -------
    nuns : list[np.ndarray]
        List of up to ``k`` nearest unlike neighbors, each with the
        same shape as individual instances in ``X_ref``.
    indices : list[int]
        Indices of the returned neighbors in ``X_ref``.

    Warns
    -----
    UserWarning
        If falling back to ``fallback_labels`` or to all instances.
    """
    if distance_fn is None:

        def distance_fn(q: np.ndarray, candidates: np.ndarray) -> np.ndarray:
            """Compute Euclidean distance from query to candidates.

            Parameters
            ----------
            q : np.ndarray
                Query instance of shape ``(1, D)`` after internal reshaping.
            candidates : np.ndarray
                Candidate array of shape ``(M, D)`` after internal reshaping.

            Returns
            -------
            np.ndarray
                1-D distance array of length ``M``.
            """
            return euclidean_cdist_flat(
                q.reshape(1, -1), candidates.reshape(candidates.shape[0], -1)
            ).ravel()

    mask = y_ref == target_class
    source = "primary"

    if not np.any(mask) and fallback_labels is not None:
        warnings.warn(
            f"No instances of target class {target_class!r} found using primary "
            f"labels. Falling back to alternative labels.",
            UserWarning,
            stacklevel=2,
        )
        mask = fallback_labels == target_class
        source = "fallback"

    if not np.any(mask) and fallback_all:
        warnings.warn(
            f"No instances of target class {target_class!r} found in any label "
            f"set ({source}). Falling back to all instances regardless of class.",
            UserWarning,
            stacklevel=2,
        )
        mask = np.ones(len(y_ref), dtype=bool)

    if not np.any(mask):
        return [], []

    candidate_indices = np.where(mask)[0]
    candidates = X_ref[candidate_indices]

    dists = distance_fn(x, candidates)
    n_return = min(k, len(candidate_indices))
    nearest_local = np.argsort(dists)[:n_return]

    nuns = [X_ref[int(candidate_indices[idx])].copy() for idx in nearest_local]
    global_indices = [int(candidate_indices[idx]) for idx in nearest_local]

    return nuns, global_indices
