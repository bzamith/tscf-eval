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

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..base import Metric
from ._utils import ensure_array


class Robustness(Metric):
    """Local Lipschitz-like robustness estimate using k-nearest neighbors.

    This metric estimates how sensitive counterfactuals are relative to
    the original inputs by scanning local neighbor pairs. For each instance
    i and its k nearest neighbors j (excluding i) it computes the ratio
    ``||x_cf[i] - x_cf[j]|| / ||x[i] - x[j]||`` (Euclidean norm after
    flattening the per-instance arrays). Larger values indicate that small
    perturbations in the original space can produce larger changes in the
    counterfactual space, i.e., lower local robustness.

    Parameters
    ----------
    k : int, optional
        Number of neighbors to consider. Default is 5. If the dataset has
        fewer instances than ``k + 1``, the number of neighbors is reduced
        accordingly.

    See Ates et al. (2021) for details.
    """

    direction = "minimize"  # Lower Lipschitz ratio means more stable/robust

    def __init__(self, k: int = 5):
        self.k = int(k)

    def name(self) -> str:
        """Return the metric name.

        Returns
        -------
        str
            ``'robustness_lipschitz'``.
        """
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

        n_neighbors = min(self.k + 1, M)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
        _, indices = nbrs.kneighbors(X)

        # Vectorized computation of neighbor ratios
        eps = 1e-12

        # Flatten instances for norm computation
        X_flat = X.reshape(M, -1)  # (M, D)
        X_cf_flat = X_cf.reshape(M, -1)  # (M, D)

        # neigh_idx has shape (M, k) where k = n_neighbors - 1 (excluding self)
        neigh_idx = indices[:, 1:]  # (M, k)

        # Gather neighbor arrays using fancy indexing
        # X_flat[neigh_idx] has shape (M, k, D)
        X_neigh = X_flat[neigh_idx]  # (M, k, D)
        X_cf_neigh = X_cf_flat[neigh_idx]  # (M, k, D)

        # Compute differences: each instance vs its neighbors
        # X_flat[:, None, :] has shape (M, 1, D), broadcasting to (M, k_actual, D)
        diff_orig = X_flat[:, None, :] - X_neigh  # (M, k_actual, D)
        diff_cf = X_cf_flat[:, None, :] - X_cf_neigh  # (M, k_actual, D)

        # Compute norms along the feature dimension
        norms_orig = np.sqrt(np.sum(diff_orig**2, axis=2))  # (M, k_actual)
        norms_cf = np.sqrt(np.sum(diff_cf**2, axis=2))  # (M, k_actual)

        # Compute ratios with safe denominator
        norms_orig_safe = np.maximum(norms_orig, eps)
        ratios = norms_cf / norms_orig_safe  # (M, k_actual)

        if ratios.size == 0:
            return 0.0
        # Use 95th percentile instead of max to reduce sensitivity to
        # single outlier neighbor pairs.
        return float(np.percentile(ratios, 95))
