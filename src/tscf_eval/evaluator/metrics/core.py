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

from typing import Literal

import numpy as np

from tscf_eval.counterfactuals.utils._distance import dtw_distance_vec_multich

from ..base import Metric
from ._utils import ensure_array


class Validity(Metric):
    """Fraction of counterfactuals that change the model prediction.

    Accepts either a fitted ``model`` (with ``predict``) or two label arrays
    ``y`` and ``y_cf``. When ``model`` is provided, compares model predictions
    on ``X`` and ``X_cf``; otherwise compares the provided label arrays.

    Parameters
    ----------
    mode : {"hard", "soft"}, default "hard"
        Evaluation mode.

        - ``"hard"``: Binary indicator — fraction of instances where the
          predicted label changed.
        - ``"soft"``: Mean probability shift toward the target class.
          Computed as ``P(target_class | X_cf) - P(target_class | X)``
          per instance, clipped to ``[0, 1]``. Requires a model with
          ``predict_proba``. Falls back to hard validity when only
          label arrays are provided.

    See Li et al. (2023) for details.
    """

    direction = "maximize"

    def __init__(self, mode: Literal["hard", "soft"] = "soft"):
        """Initialize the Validity metric.

        Parameters
        ----------
        mode : {"hard", "soft"}, default "soft"
            Evaluation mode. ``"hard"`` uses binary label change;
            ``"soft"`` uses probability shift toward the target class.
        """
        if mode not in ("hard", "soft"):
            raise ValueError(f"Unknown validity mode: {mode!r}. Expected 'hard' or 'soft'.")
        self.mode = mode

    def name(self) -> str:
        """Return the metric name.

        Returns
        -------
        str
            ``'validity'`` for hard mode, ``'validity_soft'`` for soft mode.
        """
        if self.mode == "soft":
            return "validity_soft"
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
            Classifier with ``predict`` method (and ``predict_proba``
            for soft mode).
        y : array-like, optional
            Original labels (used if ``model`` is ``None``).
        y_cf : array-like, optional
            Counterfactual labels (used if ``model`` is ``None``).
        **kwargs
            Additional keyword arguments. Recognized internal keys:

            - ``_cached_y_pred``, ``_cached_y_cf_pred``: Pre-computed
              hard predictions from the Evaluator.
            - ``_cached_proba_X``, ``_cached_proba_X_cf``: Pre-computed
              class probabilities from the Evaluator (used in soft mode).

        Returns
        -------
        float
            For hard mode: fraction of instances where the label changed,
            in ``[0, 1]``.
            For soft mode: mean probability shift toward the target class,
            in ``[0, 1]``.

        Raises
        ------
        ValueError
            If neither ``model`` nor ``(y, y_cf)`` are provided, or if
            soft mode is requested but the model lacks ``predict_proba``.
        """
        X = ensure_array(X)
        X_cf = ensure_array(X_cf)

        if self.mode == "soft":
            return self._compute_soft(X, X_cf, model, y, y_cf, **kwargs)
        return self._compute_hard(X, X_cf, model, y, y_cf, **kwargs)

    def _compute_hard(
        self,
        X: np.ndarray,
        X_cf: np.ndarray,
        model=None,
        y=None,
        y_cf=None,
        **kwargs,
    ) -> float:
        """Compute hard validity as the fraction of label changes.

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
            May contain ``_cached_y_pred`` and ``_cached_y_cf_pred``.

        Returns
        -------
        float
            Fraction of instances where the predicted label changed,
            in ``[0, 1]``.
        """
        if model is not None:
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

    def _compute_soft(
        self,
        X: np.ndarray,
        X_cf: np.ndarray,
        model=None,
        y=None,
        y_cf=None,
        **kwargs,
    ) -> float:
        """Compute soft validity as the mean probability shift toward the target class.

        For each instance, the target class is the counterfactual label
        (from ``y_cf`` or ``model.predict(X_cf)``). The score is the
        increase in ``P(target_class)`` from the original to the
        counterfactual, clipped to ``[0, 1]``.

        Parameters
        ----------
        X : np.ndarray
            Original instances.
        X_cf : np.ndarray
            Counterfactual instances.
        model : object, optional
            Classifier with ``predict`` and ``predict_proba`` methods.
        y : array-like, optional
            Original labels (used for hard fallback if ``model`` is ``None``).
        y_cf : array-like, optional
            Counterfactual labels (used for hard fallback if ``model`` is ``None``).
        **kwargs
            May contain ``_cached_proba_X``, ``_cached_proba_X_cf``,
            and ``_cached_y_cf_pred``.

        Returns
        -------
        float
            Mean probability shift toward the target class, in ``[0, 1]``.
        """
        if model is None:
            if y is not None and y_cf is not None:
                # Fall back to hard validity when no model probabilities
                return self._compute_hard(X, X_cf, model, y, y_cf, **kwargs)
            raise ValueError(
                "Soft validity requires a model with predict_proba, "
                "or falls back to hard validity with (y, y_cf) arrays."
            )

        # Get class probabilities
        proba_X = kwargs.get("_cached_proba_X")
        proba_X_cf = kwargs.get("_cached_proba_X_cf")
        if proba_X is None or proba_X_cf is None:
            if not hasattr(model, "predict_proba"):
                raise ValueError("Soft validity requires a model with predict_proba.")
            proba_X = np.asarray(model.predict_proba(X))
            proba_X_cf = np.asarray(model.predict_proba(X_cf))

        # Determine target class per instance: the counterfactual label
        y_cf_pred = kwargs.get("_cached_y_cf_pred")
        if y_cf_pred is None:
            y_cf_pred = np.asarray(model.predict(X_cf))
        target_labels = np.asarray(y_cf_pred)

        # Map class labels to column indices in the probability matrix.
        # model.classes_ maps column index -> label, so we invert it.
        if hasattr(model, "classes_"):
            classes = np.asarray(model.classes_)
            label_to_col = {int(c): i for i, c in enumerate(classes)}
            target_cols = np.array([label_to_col[int(t)] for t in target_labels])
        else:
            # Assume labels are already valid column indices
            target_cols = target_labels.astype(int)

        # Probability of the target class before and after
        n = X.shape[0]
        idx = np.arange(n)
        p_target_orig = proba_X[idx, target_cols]
        p_target_cf = proba_X_cf[idx, target_cols]

        shift = np.clip(p_target_cf - p_target_orig, 0.0, 1.0)
        return float(np.mean(shift))


class Proximity(Metric):
    """Proximity score between original and counterfactual instances.

    Computed as ``1 / (1 + d)`` where ``d`` is the per-instance distance.
    Values are in ``[0, 1]`` where 1 means identical and higher is better.

    Parameters
    ----------
    p : int or float, default 2
        Norm order (1 for L1, 2 for L2, ``np.inf`` or ``float('inf')`` for Linf).
        Only used when ``distance="lp"``.
    distance : {"lp", "dtw"}, default "dtw"
        Distance function to use.

        - ``"lp"``: L-p norm distance (controlled by ``p``).
        - ``"dtw"``: Dynamic Time Warping distance (per-channel, averaged).
          Requires ``tslearn``; falls back to Euclidean if unavailable.

    See Delaney et al. (2021) and Bahri et al. (2022) for details.
    """

    direction = "maximize"

    def __init__(self, p: int | float = 2, distance: Literal["lp", "dtw"] = "dtw"):
        """Initialize the Proximity metric.

        Parameters
        ----------
        p : int or float, default 2
            Norm order for Lp distance. Only used when ``distance="lp"``.
        distance : {"lp", "dtw"}, default "dtw"
            Distance function to use.
        """
        if p <= 0:
            raise ValueError("p must be > 0")
        if distance not in ("lp", "dtw"):
            raise ValueError("distance must be one of {'lp', 'dtw'}")
        self.p = p
        self.distance = distance

    def name(self) -> str:
        """Return the metric name.

        Returns
        -------
        str
            ``'proximity_l{p}'`` for Lp distance or ``'proximity_dtw'``
            for DTW distance.
        """
        if self.distance == "dtw":
            return "proximity_dtw"
        return f"proximity_l{self.p}"

    def compute(self, X: np.ndarray, X_cf: np.ndarray, **kwargs) -> float:
        """Compute mean proximity score across instances.

        The score is ``1 / (1 + d)`` where ``d`` is the distance,
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
            If ``X`` and ``X_cf`` have different numbers of instances,
            or if ``distance`` is not a supported value.
        """
        X = ensure_array(X)
        X_cf = ensure_array(X_cf)
        if X.shape[0] != X_cf.shape[0]:
            raise ValueError("X and X_cf must have the same number of instances.")

        if self.distance == "dtw":
            per_inst = self._dtw_distances(X, X_cf)
        elif self.distance == "lp":
            per_inst = self._lp_distances(X, X_cf)
        else:
            raise ValueError(f"Unknown distance: {self.distance!r}. Expected 'lp' or 'dtw'.")

        return float(np.mean(1.0 / (1.0 + per_inst)))

    def _lp_distances(self, X: np.ndarray, X_cf: np.ndarray) -> np.ndarray:
        """Compute per-instance Lp distances.

        Parameters
        ----------
        X : np.ndarray
            Original instances, shape ``(M, ...)``.
        X_cf : np.ndarray
            Counterfactual instances, same shape as ``X``.

        Returns
        -------
        np.ndarray
            Per-instance distances, shape ``(M,)``.
        """
        diff = (X - X_cf).reshape(X.shape[0], -1)
        if self.p == 1:
            dists: np.ndarray = np.sum(np.abs(diff), axis=1)
        elif self.p == 2:
            dists = np.sqrt(np.sum(diff**2, axis=1))
        elif self.p == float("inf") or self.p == np.inf:
            dists = np.max(np.abs(diff), axis=1)
        else:
            dists = np.linalg.norm(diff, ord=self.p, axis=1)
        return dists

    @staticmethod
    def _dtw_distances(X: np.ndarray, X_cf: np.ndarray) -> np.ndarray:
        """Compute per-instance DTW distances.

        Parameters
        ----------
        X : np.ndarray
            Original instances, shape ``(M, ...)``.
        X_cf : np.ndarray
            Counterfactual instances, same shape as ``X``.

        Returns
        -------
        np.ndarray
            Per-instance DTW distances, shape ``(M,)``.
        """
        M = X.shape[0]
        dists = np.empty(M, dtype=float)
        for i in range(M):
            dists[i] = dtw_distance_vec_multich(X[i], X_cf[i : i + 1])[0]
        return dists


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
        """Initialize the Sparsity metric.

        Parameters
        ----------
        atol : float, default 1e-8
            Absolute tolerance for considering a feature unchanged.
        rtol : float, default 1e-5
            Relative tolerance for considering a feature unchanged.
        """
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
