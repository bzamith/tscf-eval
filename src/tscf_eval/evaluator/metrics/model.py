"""Model-based counterfactual evaluation metrics.

This module provides metrics that require a fitted model to evaluate
counterfactuals: Confidence and Controllability.

Classes
-------
Confidence
    Reports model confidence statistics for instances.
Controllability
    Assesses how easily a counterfactual can be reverted.
"""

from __future__ import annotations

import numpy as np

from ..base import Metric
from ._utils import ensure_array


class Confidence(Metric):
    """Confidence summaries (maximum predicted probability) for instances.

    Reports the mean maximum predicted probability for both original and
    counterfactual instances, as well as the mean difference.

    See Le et al. (2023) for details.
    """

    direction = "maximize"  # Higher confidence in CF prediction is better

    def name(self) -> str:
        """Return the metric name.

        Returns
        -------
        str
            ``'confidence'``.
        """
        return "confidence"

    def compute(
        self,
        X: np.ndarray,
        X_cf: np.ndarray,
        model=None,
        **kwargs,
    ) -> dict[str, float]:
        """Compute confidence statistics.

        Parameters
        ----------
        X : np.ndarray
            Original instances of shape ``(M, ...)``.
        X_cf : np.ndarray
            Counterfactual instances of shape ``(M, ...)``.
        model : object
            Classifier with a ``predict_proba`` method.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        dict
            Dictionary with keys:

            - ``mean_conf_orig``: Mean max probability for original instances.
            - ``mean_conf_cf``: Mean max probability for counterfactuals.
            - ``mean_conf_delta``: Mean difference (cf - orig).

        Raises
        ------
        ValueError
            If ``model`` is ``None``.
        """
        if model is None:
            raise ValueError("Confidence requires a `model` with predict_proba.")
        X = ensure_array(X)
        X_cf = ensure_array(X_cf)
        # Use cached probabilities if available (from Evaluator pre-computation)
        p_orig = kwargs.get("_cached_proba_X")
        p_cf = kwargs.get("_cached_proba_X_cf")
        if p_orig is None:
            p_orig = model.predict_proba(X)
        if p_cf is None:
            p_cf = model.predict_proba(X_cf)
        max_orig = np.max(p_orig, axis=1)
        max_cf = np.max(p_cf, axis=1)
        return {
            "mean_conf_orig": float(np.mean(max_orig)),
            "mean_conf_cf": float(np.mean(max_cf)),
            "mean_conf_delta": float(np.mean(max_cf - max_orig)),
        }


class Controllability(Metric):
    """How easily a counterfactual can be reverted by partial controlled edits.

    For each counterfactual, this metric reverts random subsets of changed
    features at several fraction levels and checks whether the original
    prediction is restored.  The score is the fraction of revert attempts
    that succeed, averaged across fractions, samples, and instances.

    Parameters
    ----------
    revert_fractions : list of float, optional
        Fractions of changed features to revert at each probe level.
        Default is ``[0.1, 0.2, 0.3, 0.4, 0.5]``.
    n_samples : int, optional
        Number of random subsets to draw per fraction per instance.
        Default is ``10``.
    random_state : int or None, optional
        Seed for reproducibility.  Default is ``None``.

    See Verma et al. (2024) for details.
    """

    direction = "maximize"  # Higher controllability is better

    def __init__(
        self,
        revert_fractions: list[float] | None = None,
        n_samples: int = 10,
        random_state: int | None = None,
    ):
        """Initialize the Controllability metric.

        Parameters
        ----------
        revert_fractions : list of float, optional
            Fractions of changed features to revert at each probe level.
            Default is ``[0.1, 0.2, 0.3, 0.4, 0.5]``.
        n_samples : int, default 10
            Number of random subsets to draw per fraction per instance.
        random_state : int or None, default None
            Seed for reproducibility.
        """
        self.revert_fractions = revert_fractions or [0.1, 0.2, 0.3, 0.4, 0.5]
        self.n_samples = n_samples
        self.random_state = random_state

    def name(self) -> str:
        """Return the metric name.

        Returns
        -------
        str
            ``'controllability'``.
        """
        return "controllability"

    def compute(
        self,
        X: np.ndarray,
        X_cf: np.ndarray,
        model=None,
        **kwargs,
    ) -> float:
        """Compute controllability score via random subset reverts.

        For each instance the method identifies which features changed,
        then for every fraction in ``revert_fractions`` it draws
        ``n_samples`` random subsets of that size from the changed
        features, reverts them to their original values, and checks
        whether the model prediction is restored.

        Parameters
        ----------
        X : np.ndarray
            Original instances of shape ``(M, ...)``.
        X_cf : np.ndarray
            Counterfactual instances of shape ``(M, ...)``.
        model : object
            Classifier with a ``predict`` method.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Mean controllability score in ``[0, 1]``. Higher values indicate
            that counterfactuals can be more easily reverted.

        Raises
        ------
        ValueError
            If ``model`` is ``None``.
        """
        if model is None:
            raise ValueError("Controllability requires a `model` to probe reverts.")
        X = ensure_array(X)
        X_cf = ensure_array(X_cf)
        M = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        # Use cached predictions if available, otherwise batch predict once
        all_orig_labels = kwargs.get("_cached_y_pred")
        all_cf_labels = kwargs.get("_cached_y_cf_pred")
        if all_orig_labels is None:
            all_orig_labels = np.asarray(model.predict(X))
        if all_cf_labels is None:
            all_cf_labels = np.asarray(model.predict(X_cf))

        scores = []
        for i in range(M):
            xi = X[i]
            xfi = X_cf[i]
            orig_label = int(all_orig_labels[i])
            cf_label = int(all_cf_labels[i])
            if orig_label == cf_label:
                scores.append(0.0)
                continue
            flat_x = xi.reshape(-1)
            flat_xf = xfi.reshape(-1)
            changed_idx = np.nonzero(~np.isclose(flat_x, flat_xf))[0]
            if changed_idx.size == 0:
                scores.append(0.0)
                continue

            # Build candidates across all fractions and samples
            candidates = []
            for frac in self.revert_fractions:
                n_revert = max(1, round(frac * changed_idx.size))
                for _ in range(self.n_samples):
                    subset = rng.choice(changed_idx, size=n_revert, replace=False)
                    cand = flat_xf.copy()
                    cand[subset] = flat_x[subset]
                    candidates.append(cand.reshape(xi.shape))

            cand_batch = np.stack(candidates)
            cand_preds = np.asarray(model.predict(cand_batch))
            n_reverting = int(np.sum(cand_preds == orig_label))
            scores.append(n_reverting / float(len(candidates)))

        return float(np.mean(scores))
