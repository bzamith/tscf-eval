"""Stability-based counterfactual evaluation metrics.

This module provides the Robustness metric that estimates local
Lipschitz-like stability of counterfactual generation.

The concept of Lipschitz continuity is used to assess how sensitive
a counterfactual generation method is to small perturbations in the
input space. Lower Lipschitz constants indicate more stable methods.

Classes
-------
Robustness
    Estimates sensitivity of counterfactuals using k-nearest neighbor analysis.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.neighbors import NearestNeighbors

from tscf_eval.counterfactuals.utils._distance import dtw_distance_vec_multich

from ..base import Metric
from ._utils import ensure_array


class Robustness(Metric):
    """Local Lipschitz-like robustness estimate using k-nearest neighbors.

    This metric estimates how sensitive counterfactuals are relative to
    the original inputs by scanning local neighbor pairs. For each instance
    i and its k nearest neighbors j (excluding i) it computes the ratio
    ``d(x_cf[i], x_cf[j]) / d(x[i], x[j])``. Larger values indicate that
    small perturbations in the original space can produce larger changes in
    the counterfactual space, i.e., lower local robustness.

    Parameters
    ----------
    k : int, optional
        Number of neighbors to consider. Default is 5. If the dataset has
        fewer instances than ``k + 1``, the number of neighbors is reduced
        accordingly.
    distance : {"euclidean", "dtw"}, default "dtw"
        Distance function to use.

        - ``"euclidean"``: Euclidean distance on flattened vectors.
        - ``"dtw"``: Per-channel DTW distance (averaged across channels).
          Requires ``tslearn``; falls back to Euclidean if unavailable.

    See Ates et al. (2021) for details.
    """

    direction = "minimize"  # Lower Lipschitz ratio means more stable/robust

    def __init__(self, k: int = 5, distance: Literal["euclidean", "dtw"] = "dtw"):
        """Initialize the Robustness metric.

        Parameters
        ----------
        k : int, default 5
            Number of nearest neighbors to consider.
        distance : {"euclidean", "dtw"}, default "dtw"
            Distance function to use.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        if distance not in ("euclidean", "dtw"):
            raise ValueError("distance must be one of {'euclidean', 'dtw'}")
        self.k = int(k)
        self.distance = distance

    def name(self) -> str:
        """Return the metric name.

        Returns
        -------
        str
            ``'robustness_lipschitz'`` for Euclidean distance or
            ``'robustness_lipschitz_dtw'`` for DTW distance.
        """
        if self.distance == "dtw":
            return "robustness_lipschitz_dtw"
        return "robustness_lipschitz"

    def compute(
        self,
        X: np.ndarray,
        X_cf: np.ndarray,
        X_train: np.ndarray | None = None,
        **kwargs,
    ) -> float:
        """Compute robustness score.

        Parameters
        ----------
        X : np.ndarray
            Original instances of shape ``(M, ...)``.
        X_cf : np.ndarray
            Counterfactual instances of shape ``(M, ...)``.
        X_train : np.ndarray, optional
            Training data (unused, present for API compatibility).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            95th-percentile neighbor ratio (>= 0). Returns 0.0 when there
            are not enough instances to form neighbor pairs.
        """
        X = ensure_array(X)
        X_cf = ensure_array(X_cf)
        M = X.shape[0]
        if M <= 1:
            return 0.0

        if self.distance == "dtw":
            return self._compute_dtw(X, X_cf, M)
        if self.distance == "euclidean":
            return self._compute_euclidean(X, X_cf, M)
        raise ValueError(f"Unknown distance: {self.distance!r}. Expected 'euclidean' or 'dtw'.")

    def _compute_euclidean(self, X: np.ndarray, X_cf: np.ndarray, M: int) -> float:
        """Compute robustness using Euclidean distance.

        Parameters
        ----------
        X : np.ndarray
            Original instances, shape ``(M, ...)``.
        X_cf : np.ndarray
            Counterfactual instances, same shape as ``X``.
        M : int
            Number of instances.

        Returns
        -------
        float
            95th-percentile neighbor ratio (>= 0).
        """
        n_neighbors = min(self.k + 1, M)
        X_flat = X.reshape(M, -1)
        X_cf_flat = X_cf.reshape(M, -1)

        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_flat)
        _, indices = nbrs.kneighbors(X_flat)

        eps = 1e-12
        neigh_idx = indices[:, 1:]  # (M, k)

        X_neigh = X_flat[neigh_idx]  # (M, k, D)
        X_cf_neigh = X_cf_flat[neigh_idx]  # (M, k, D)

        diff_orig = X_flat[:, None, :] - X_neigh
        diff_cf = X_cf_flat[:, None, :] - X_cf_neigh

        norms_orig = np.sqrt(np.sum(diff_orig**2, axis=2))
        norms_cf = np.sqrt(np.sum(diff_cf**2, axis=2))

        norms_orig_safe = np.maximum(norms_orig, eps)
        ratios = norms_cf / norms_orig_safe

        if ratios.size == 0:
            return 0.0
        return float(np.percentile(ratios, 95))

    def _compute_dtw(self, X: np.ndarray, X_cf: np.ndarray, M: int) -> float:
        """Compute robustness using DTW distance.

        Builds precomputed pairwise DTW distance matrices for both X and
        X_cf, finds k-nearest neighbors in DTW space, and computes the
        Lipschitz-like ratio using DTW distances.

        Parameters
        ----------
        X : np.ndarray
            Original instances, shape ``(M, ...)``.
        X_cf : np.ndarray
            Counterfactual instances, same shape as ``X``.
        M : int
            Number of instances.

        Returns
        -------
        float
            95th-percentile neighbor ratio (>= 0).
        """
        # Build pairwise DTW distance matrix for X
        D_X = np.zeros((M, M), dtype=float)
        for i in range(M):
            D_X[i] = dtw_distance_vec_multich(X[i], X)
        D_X = 0.5 * (D_X + D_X.T)

        # Build pairwise DTW distance matrix for X_cf
        D_cf = np.zeros((M, M), dtype=float)
        for i in range(M):
            D_cf[i] = dtw_distance_vec_multich(X_cf[i], X_cf)
        D_cf = 0.5 * (D_cf + D_cf.T)

        # Find k-nearest neighbors using precomputed DTW distances on X
        n_neighbors = min(self.k + 1, M)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed").fit(D_X)
        _, indices = nbrs.kneighbors(D_X)

        eps = 1e-12
        neigh_idx = indices[:, 1:]  # (M, k)

        # Gather pairwise distances for each instance and its neighbors
        row_idx = np.arange(M)[:, None]  # (M, 1)
        norms_orig = D_X[row_idx, neigh_idx]  # (M, k)
        norms_cf = D_cf[row_idx, neigh_idx]  # (M, k)

        norms_orig_safe = np.maximum(norms_orig, eps)
        ratios = norms_cf / norms_orig_safe

        if ratios.size == 0:
            return 0.0
        return float(np.percentile(ratios, 95))
