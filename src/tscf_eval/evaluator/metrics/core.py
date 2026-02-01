"""Core counterfactual evaluation metrics.

This module provides the fundamental metrics for evaluating counterfactual
quality: Validity, Proximity, and Sparsity.

These metrics are foundational to counterfactual evaluation and are widely
used in the literature.

Classes
-------
Validity
    Fraction of counterfactuals that change the model prediction.
Proximity
    Proximity score (1/(1+distance)) between original and counterfactual.
Sparsity
    Fraction of features changed.
"""

from __future__ import annotations

import numpy as np

from ..base import Metric
from ._utils import ensure_array


class Validity(Metric):
    """Fraction of counterfactuals that change the model prediction.

    Accepts either a fitted ``model`` (with ``predict``) or two label arrays
    ``y`` and ``y_cf``. When ``model`` is provided, compares model predictions
    on ``X`` and ``X_cf``; otherwise compares the provided label arrays.

    See Li et al. (2023) for details.
    """

    direction = "maximize"

    def name(self) -> str:
        """Return the metric name.

        Returns
        -------
        str
            ``'validity'``.
        """
        return "validity"

    def compute(
        self,
        X: np.ndarray,
        X_cf: np.ndarray,
        model=None,
        y=None,
        y_cf=None,
        **kwargs,
    ) -> float:
        """Compute validity score.

        Parameters
        ----------
        X : np.ndarray
            Original instances.
        X_cf : np.ndarray
            Counterfactual instances.
        model : object, optional
            Classifier with ``predict`` method.
        y : array-like, optional
            Original labels (used if ``model`` is ``None``).
        y_cf : array-like, optional
            Counterfactual labels (used if ``model`` is ``None``).
        **kwargs
            Ignored.

        Returns
        -------
        float
            Fraction of instances where the label changed, in ``[0, 1]``.

        Raises
        ------
        ValueError
            If neither ``model`` nor ``(y, y_cf)`` are provided.
        """
        X = ensure_array(X)
        X_cf = ensure_array(X_cf)
        if model is not None:
            # Use cached predictions if available (from Evaluator pre-computation)
            y_pred = kwargs.get("_cached_y_pred")
            y_cf_pred = kwargs.get("_cached_y_cf_pred")
            if y_pred is None:
                y_pred = model.predict(X)
            if y_cf_pred is None:
                y_cf_pred = model.predict(X_cf)
            return float(np.mean(np.asarray(y_pred) != np.asarray(y_cf_pred)))
        if y is not None and y_cf is not None:
            return float(np.mean(np.asarray(y) != np.asarray(y_cf)))
        raise ValueError("Validity requires either a model or (y, y_cf) arrays.")


class Proximity(Metric):
    """Proximity score between original and counterfactual instances.

    Computed as ``1 / (1 + d)`` where ``d`` is the per-instance L-p distance.
    Values are in ``[0, 1]`` where 1 means identical and higher is better.

    Parameters
    ----------
    p : int or float, default 2
        Norm order (1 for L1, 2 for L2, ``np.inf`` or ``float('inf')`` for Linf).

    See Delaney et al. (2021) and Bahri et al. (2022) for details.
    """

    direction = "maximize"

    def __init__(self, p: int | float = 2):
        self.p = p

    def name(self) -> str:
        """Return the metric name.

        Returns
        -------
        str
            ``'proximity_l{p}'`` where ``p`` is the norm order.
        """
        return f"proximity_l{self.p}"

    def compute(self, X: np.ndarray, X_cf: np.ndarray, **kwargs) -> float:
        """Compute mean proximity score across instances.

        The score is ``1 / (1 + d)`` where ``d`` is the L-p distance,
        averaged over all instances.

        Parameters
        ----------
        X : np.ndarray
            Original instances.
        X_cf : np.ndarray
            Counterfactual instances.
        **kwargs
            Ignored.

        Returns
        -------
        float
            Mean proximity score in ``[0, 1]``. Higher values indicate
            counterfactuals closer to the originals.

        Raises
        ------
        ValueError
            If ``X`` and ``X_cf`` have different numbers of instances.
        """
        X = ensure_array(X)
        X_cf = ensure_array(X_cf)
        if X.shape[0] != X_cf.shape[0]:
            raise ValueError("X and X_cf must have the same number of instances.")
        diff = (X - X_cf).reshape(X.shape[0], -1)
        if self.p == 1:
            per_inst = np.sum(np.abs(diff), axis=1)
        elif self.p == 2:
            per_inst = np.sqrt(np.sum(diff**2, axis=1))
        elif self.p == float("inf") or self.p == np.inf:
            per_inst = np.max(np.abs(diff), axis=1)
        else:
            per_inst = np.linalg.norm(diff, ord=self.p, axis=1)
        return float(np.mean(1.0 / (1.0 + per_inst)))


class Sparsity(Metric):
    """Fraction of features/time-points changed between original and counterfactual.

    Flattens per-instance arrays and reports the mean fraction of entries
    that differ between ``X`` and ``X_cf``. Lower values indicate sparser
    (more targeted) edits.

    Parameters
    ----------
    atol : float, default 1e-8
        Absolute tolerance for considering a feature unchanged.
    rtol : float, default 1e-5
        Relative tolerance for considering a feature unchanged.

    Notes
    -----
    A feature is considered unchanged if ``|X[i] - X_cf[i]| <= atol + rtol * |X[i]|``.
    This avoids false positives from floating-point precision issues.

    See Mothilal et al. (2020) for details.
    """

    direction = "minimize"

    def __init__(self, atol: float = 1e-8, rtol: float = 1e-5):
        self.atol = atol
        self.rtol = rtol

    def name(self) -> str:
        """Return the metric name.

        Returns
        -------
        str
            ``'sparsity'``.
        """
        return "sparsity"

    def compute(self, X: np.ndarray, X_cf: np.ndarray, **kwargs) -> float:
        """Compute mean sparsity across instances.

        Parameters
        ----------
        X : np.ndarray
            Original instances.
        X_cf : np.ndarray
            Counterfactual instances.
        **kwargs
            Ignored.

        Returns
        -------
        float
            Mean fraction of changed features in ``[0, 1]``. Lower values
            indicate sparser (more targeted) edits.

        Raises
        ------
        ValueError
            If ``X`` and ``X_cf`` have different numbers of instances.
        """
        X = ensure_array(X)
        X_cf = ensure_array(X_cf)
        if X.shape[0] != X_cf.shape[0]:
            raise ValueError("X and X_cf must have the same number of instances.")
        flat_X = X.reshape(X.shape[0], -1)
        flat_Xcf = X_cf.reshape(X_cf.shape[0], -1)
        # Use tolerance-based comparison instead of exact equality
        changed = ~np.isclose(flat_X, flat_Xcf, atol=self.atol, rtol=self.rtol)
        changed_count = np.sum(changed, axis=1).astype(float)
        denom = float(flat_X.shape[1]) if flat_X.shape[1] > 0 else 1.0
        sparsity_per_inst = changed_count / denom
        return float(np.mean(sparsity_per_inst))
