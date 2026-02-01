"""NativeGuide counterfactual explainer implementation.

This module provides the ``NativeGuide`` class, an implementation of the
Native Guide algorithm for generating counterfactual explanations for
time series classification.

The algorithm was originally developed by Eoin Delaney, Derek Greene, and
Mark T. Keane at University College Dublin's Insight Centre for Data Analytics.

Original implementation: https://github.com/e-delaney/Instance-Based_CFE_TSC

Classes
-------
NativeGuide
    NativeGuide counterfactual generator using nearest-unlike-neighbor guidance.

Algorithm Overview
------------------
NativeGuide generates counterfactuals through instance-based reasoning:

1. Find the nearest unlike neighbor (NUN) - the closest instance in the
   reference set that is predicted as a different class.
2. Generate a counterfactual by blending the query with the NUN using one
   of several methods:

   - **blend**: Weighted DTW barycenter averaging, incrementally increasing
     the NUN's influence until the prediction flips (original paper method).
   - **ng**: Copy a contiguous window from the NUN, growing until flip.
   - **dtw_dba**: Like 'ng' but uses a DTW-DBA barycenter of k unlike neighbors.
   - **cam**: Like 'ng' but uses CAM importance to select the window location.

Examples
--------
>>> from tscf_eval.counterfactuals import NativeGuide
>>> import numpy as np
>>>
>>> # Assume clf is a trained classifier
>>> ng = NativeGuide(
...     model=clf,
...     data=(X_train, y_train),
...     method="blend",  # Original paper method
...     distance="dtw",
... )
>>>
>>> # Generate counterfactual for a test instance
>>> cf, cf_label, meta = ng.explain(x_test)
>>> print(f"Beta (blend weight): {meta['beta']}")
>>> print(f"NUN index: {meta['nun_index_in_X']}")

References
----------
.. [ng1] Delaney, E., Greene, D., & Keane, M. T. (2021).
       Instance-Based Counterfactual Explanations for Time Series Classification.
       In Case-Based Reasoning Research and Development (ICCBR 2021),
       pp. 32-47. Springer International Publishing.
       DOI: 10.1007/978-3-030-86957-1_3

.. [ng2] Hollig, J., Kulbach, C., & Thoma, S. (2023).
       TSInterpret: A Python Package for the Interpretability of Time Series
       Classification. Journal of Open Source Software, 8(85), 5220.
       https://doi.org/10.21105/joss.05220
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
import warnings

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

from .base import Counterfactual
from .utils import (
    dba_barycenter_multich,
    dtw_distance_vec_multich,
    ensure_batch_shape,
    euclidean_cdist_flat,
    soft_predict_proba_fn,
    strip_batch,
    weighted_dba_multich,
)

try:
    from tslearn.metrics import dtw as _tslearn_dtw  # noqa: F401

    TSLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    TSLEARN_AVAILABLE = False


@dataclass
class NativeGuide(Counterfactual):
    """NativeGuide counterfactual generator for time-series.

    Implementation of the NativeGuide algorithm by Delaney et al. (2021) [ng1]_.

    The algorithm retrieves a "native guide" (nearest-unlike neighbor, NUN)
    from a reference set. Depending on the method, it either:

    - **'blend'** (original paper): Blends the query with the NUN using
      weighted DTW barycenter averaging, incrementally increasing the
      guide's influence until prediction flips.
    - **'ng'**: Copies a contiguous window from the NUN into the query,
      growing the window until prediction flips.
    - **'dtw_dba'**: Like 'ng' but uses a DTW-DBA barycenter of k unlike
      neighbors as the guide.
    - **'cam'**: Like 'ng' but uses a CAM importance function to select
      the discriminative window.

    Parameters
    ----------
    model : object
        A classifier-like object that exposes a probability estimator. The
        internal helper ``predict_proba_fn`` adapts common interfaces (e.g.
        scikit-learn, aeon).
    data : tuple
        A tuple ``(X_ref, y_ref)`` containing the reference dataset used to
        select distractors. ``X_ref`` can have shape ``(N, T)`` or ``(N, C, T)``.
    method : {'blend', 'ng', 'dtw_dba', 'cam'}, default 'blend'
        Strategy for counterfactual generation:

        - 'blend': Original paper method. Weighted averaging of query and NUN
          using DTW barycenter, incrementally increasing NUN influence.
        - 'ng': Window replacement using nearest-unlike neighbor.
        - 'dtw_dba': Window replacement using DTW-DBA barycenter of k neighbors.
        - 'cam': Window replacement guided by CAM importance function.
    distance : {'euclidean', 'dtw'}, default 'dtw'
        Distance metric used to rank distractors when searching the reference set.
    k_unlike : int, default 5
        Number of unlike neighbors to consider when computing a DTW-DBA guide.
    random_state : int or None, default 0
        PRNG seed for deterministic behaviour where applicable.
    beta_step : float, default 0.01
        For ``method='blend'``: increment for the blending weight beta at each
        iteration (original paper uses 0.01).
    target_prob : float, default 0.5
        For ``method='blend'``: target probability threshold for the counterfactual
        class (original paper uses 0.5).
    init_window_frac : float, default 0.1
        For window-based methods: initial window size as a fraction of T.
    step_frac : float, default 0.05
        For window-based methods: fractional increment for window growth.
    max_window_frac : float, default 1.0
        For window-based methods: maximum allowed window size as a fraction of T.
    cam_importance_fn : callable or None
        When ``method=='cam'``, a function with signature
        ``(series, y_pred) -> np.ndarray`` that returns an importance map of
        shape ``(T,)`` or ``(C, T)``.

    Notes
    -----
    The public API is ``explain(x, y_pred=None) -> (cf, cf_label, meta)``.
    The returned ``meta`` dictionary contains keys such as
    ``nun_index_in_X``, ``neighbor_indices``, ``neighbor_distance``,
    ``window_start``, ``window_len``, and ``beta`` (for blend method).

    References
    ----------
    .. [ng1] Delaney, E., Greene, D., & Keane, M. T. (2021).
           Instance-Based Counterfactual Explanations for Time Series
           Classification. ICCBR 2021.
           https://github.com/e-delaney/Instance-Based_CFE_TSC
    """

    model: Any
    data: tuple[np.ndarray, np.ndarray]
    method: Literal["blend", "ng", "dtw_dba", "cam"] = "blend"
    distance: Literal["euclidean", "dtw"] = "dtw"
    k_unlike: int = 5
    random_state: int | None = 0

    # Blend method hyperparameters (original paper)
    beta_step: float = 0.01  # increment for blending weight
    target_prob: float = 0.5  # target probability threshold

    # Window-growth hyperparameters (fractions of T)
    init_window_frac: float = 0.1  # start with 10% of the length
    step_frac: float = 0.05  # grow by +5% each attempt
    max_window_frac: float = 1.0  # cap at 100%

    # Only used when method="cam": importance fn(series, y_pred) -> (T,) or (C,T)
    cam_importance_fn: Callable[[np.ndarray, int], np.ndarray] | None = None

    def __post_init__(self):
        X_ref, y_ref = self.data
        self.X_ref = np.asarray(X_ref)
        self.y_ref = np.asarray(y_ref).ravel()
        if self.X_ref.shape[0] != self.y_ref.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        self.predict_proba = soft_predict_proba_fn(self.model)
        self.rng = np.random.default_rng(self.random_state)
        # Pre-compute reference set predictions to avoid redundant calls
        self._ref_probs = self.predict_proba(self.X_ref)
        self._ref_yhat = np.argmax(self._ref_probs, axis=1)

        if self.method not in {"blend", "ng", "dtw_dba", "cam"}:
            raise ValueError("method must be one of {'blend', 'ng', 'dtw_dba', 'cam'}")
        if self.distance not in {"euclidean", "dtw"}:
            raise ValueError("distance must be one of {'euclidean', 'dtw'}")
        if self.method == "dtw_dba" and self.k_unlike < 2:
            self.k_unlike = 2
        if self.method == "cam" and self.cam_importance_fn is None:
            raise ValueError("cam_importance_fn must be provided when method='cam'.")

        # sanity checks for window-based methods
        for frac_name, frac in [
            ("init_window_frac", self.init_window_frac),
            ("step_frac", self.step_frac),
            ("max_window_frac", self.max_window_frac),
        ]:
            if not (0.0 < frac <= 1.0):
                raise ValueError(f"{frac_name} must be in (0, 1].")

        # sanity checks for blend method
        if not (0.0 < self.beta_step <= 1.0):
            raise ValueError("beta_step must be in (0, 1].")
        if not (0.0 < self.target_prob <= 1.0):
            raise ValueError("target_prob must be in (0, 1].")

    def explain(
        self, x: np.ndarray, y_pred: int | None = None
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        """Generate a counterfactual explanation for a time series instance.

        Parameters
        ----------
        x : np.ndarray
            Input time series of shape ``(T,)`` for univariate or ``(C, T)``
            for multivariate data.
        y_pred : int, optional
            Precomputed predicted class for ``x``. If ``None``, computed
            via the model.

        Returns
        -------
        cf : np.ndarray
            Counterfactual time series with the same shape as ``x``.
        cf_label : int
            Predicted class label for the counterfactual.
        meta : dict
            Metadata dictionary containing:

            - ``method``: Algorithm variant used.
            - ``distance``: Distance metric used.
            - ``nun_index_in_X``: Index of nearest unlike neighbor.
            - ``neighbor_indices``: Indices of neighbors (for dtw_dba).
            - ``neighbor_distance``: Distance to nearest unlike neighbor.
            - ``beta``: Blending weight (for blend method, else ``None``).
            - ``window_start``: Start of replacement window (else ``None``).
            - ``window_len``: Length of replacement window (else ``None``).
        """

        xb, added = ensure_batch_shape(x)
        x1 = strip_batch(xb, added)

        base_label = int(np.argmax(self.predict_proba(xb)[0])) if y_pred is None else int(y_pred)

        # 1) Build native guide (NUN or DTW-DBA over unlike neighbors)
        guide, gmeta = self._build_native_guide(x1, base_label)

        # 2) Generate counterfactual based on method
        if self.method == "blend":
            # Original paper: weighted DTW barycenter averaging
            cf, cf_label, beta = self._blend_search(x1, guide, base_label)
            meta = {
                "method": self.method,
                "distance": self.distance,
                "nun_index_in_X": gmeta.get("nun_index_in_X"),
                "neighbor_indices": gmeta.get("neighbor_indices"),
                "neighbor_distance": gmeta.get("neighbor_distance"),
                "beta": beta,
                "window_start": None,
                "window_len": None,
            }
        else:
            # Window-based methods: ng, dtw_dba, cam
            imp = self._get_importance(x1, base_label)  # (T,) or None
            start, L, cf, cf_label = self._window_growth_search(x1, guide, base_label, imp)
            meta = {
                "method": self.method,
                "distance": self.distance,
                "nun_index_in_X": gmeta.get("nun_index_in_X"),
                "neighbor_indices": gmeta.get("neighbor_indices"),
                "neighbor_distance": gmeta.get("neighbor_distance"),
                "beta": None,
                "window_start": start,
                "window_len": L,
            }

        return cf, cf_label, meta

    def _build_native_guide(
        self, x: np.ndarray, base_label: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Retrieve the native guide used as the content source.

        Parameters
        ----------
        x : np.ndarray
            Query time series of shape ``(T,)`` or ``(C, T)``.
        base_label : int
            Original predicted class label to find unlike neighbors for.

        Returns
        -------
        guide : np.ndarray
            Native guide series with the same shape as ``x``.
        metadata : dict
            Dictionary containing ``'nun_index_in_X'``, ``'neighbor_indices'``,
            and ``'neighbor_distance'``.
        """
        # Use pre-computed reference set predictions (computed once in __post_init__)
        yhat = self._ref_yhat
        unlike_mask = yhat != base_label
        if not np.any(unlike_mask):
            # No unlike neighbor available: fallback to global mean
            warnings.warn(
                f"NativeGuide: No unlike neighbors found in reference set. "
                f"The classifier predicts all {len(yhat)} reference samples as class "
                f"{base_label}. Falling back to global mean, which may not produce "
                f"a valid counterfactual. Consider using a different dataset or "
                f"classifier with more diverse predictions.",
                UserWarning,
                stacklevel=3,
            )
            guide = self._global_mean_like(self.X_ref)
            return guide, {
                "nun_index_in_X": None,
                "neighbor_indices": None,
                "neighbor_distance": float(np.nan),
                "failure_reason": "no_unlike_neighbors",
            }

        X_unlike = self.X_ref[unlike_mask]
        dvec = self._distance_vec(x, X_unlike)
        order = np.argsort(dvec)
        nun_in_unlike = int(order[0])

        # Map indices back to the original reference indices and provide metadata
        ref_indices = np.flatnonzero(unlike_mask)
        nun_index_in_ref = int(ref_indices[nun_in_unlike])

        if self.method == "dtw_dba":
            k = min(self.k_unlike, len(order))
            nbr_idx_unlike = order[:k]
            nbrs = X_unlike[nbr_idx_unlike]
            guide = dba_barycenter_multich(nbrs)
            neighbor_indices = ref_indices[nbr_idx_unlike].tolist()
            neighbor_distance = float(dvec[nbr_idx_unlike[0]])
            return guide, {
                "nun_index_in_X": nun_index_in_ref,
                "neighbor_indices": neighbor_indices,
                "neighbor_distance": neighbor_distance,
            }

        # 'blend', 'ng' and 'cam' use the single nearest-unlike neighbor as the guide
        nun = X_unlike[nun_in_unlike]
        return nun, {
            "nun_index_in_X": nun_index_in_ref,
            "neighbor_indices": None,
            "neighbor_distance": float(dvec[nun_in_unlike]),
        }

    def _blend_search(
        self, x: np.ndarray, guide: np.ndarray, base_label: int
    ) -> tuple[np.ndarray, int, float]:
        """Original paper method: weighted DTW barycenter averaging.

        Incrementally increases the guide's influence (beta) until the model's
        predicted class flips or the target probability is reached.

        Parameters
        ----------
        x : np.ndarray
            Query series of shape (T,) or (C, T).
        guide : np.ndarray
            Native guide (NUN) of same shape as x.
        base_label : int
            Original predicted class label.

        Returns
        -------
        cf : np.ndarray
            Counterfactual series with the same shape as ``x``.
        cf_label : int
            Predicted class label for the counterfactual.
        beta : float
            Final blending weight in ``[0, 1]``.
        """

        def _predict(arr: np.ndarray) -> tuple[int, float]:
            """Return (predicted_label, target_class_probability)."""
            cb, _ = ensure_batch_shape(arr)
            probs = self.predict_proba(cb)[0]
            pred_label = int(np.argmax(probs))
            # Target class is any class != base_label; use highest non-base prob
            target_probs = [(i, p) for i, p in enumerate(probs) if i != base_label]
            target_prob = max(p for _, p in target_probs) if target_probs else 0.0
            return pred_label, target_prob

        beta = 0.0
        cf = x.copy()
        cf_label, target_prob = _predict(cf)

        # Iterate until target probability threshold or prediction flip
        while target_prob < self.target_prob and beta < 1.0:
            beta += self.beta_step
            beta = min(beta, 1.0)  # cap at 1.0
            cf = weighted_dba_multich(x, guide, beta)
            cf_label, target_prob = _predict(cf)
            if cf_label != base_label:
                break

        return cf, cf_label, beta

    def _get_importance(self, x: np.ndarray, base_label: int) -> np.ndarray | None:
        """Return a 1-D importance map used to pick the discriminative window.

        If ``method=='cam'`` the provided CAM function is used. Otherwise
        ``None`` is returned and a heuristic ``|guide - x|`` is used
        on-demand inside window search.

        Parameters
        ----------
        x : np.ndarray
            Query time series of shape ``(T,)`` or ``(C, T)``.
        base_label : int
            Original predicted class label.

        Returns
        -------
        np.ndarray or None
            Importance map of shape ``(T,)`` if ``method='cam'``,
            otherwise ``None``.
        """
        if self.method != "cam":
            return None

        # cam_importance_fn is guaranteed to be present when method == 'cam'
        assert self.cam_importance_fn is not None
        imp = np.asarray(self.cam_importance_fn(x, base_label))
        # Normalize / validate shapes and reduce to (T,) when necessary
        if x.ndim == 1:
            if imp.ndim != 1 or imp.shape[0] != x.shape[0]:
                raise ValueError("cam_importance_fn must return shape (T,) for univariate input.")
            return imp

        # multivariate input: prefer (C, T) or (T,)
        if imp.ndim == 1:
            if imp.shape[0] != x.shape[1]:
                raise ValueError("When returning (T,), length must match time length.")
            return imp
        if imp.ndim == 2 and imp.shape == x.shape:
            sum_result: np.ndarray = imp.sum(axis=0)
            return sum_result
        raise ValueError("cam_importance_fn must return (T,) or (C,T) for multivariate input.")

    def _window_growth_search(
        self,
        x: np.ndarray,
        guide: np.ndarray,
        base_label: int,
        imp: np.ndarray | None,
    ) -> tuple[int, int, np.ndarray, int]:
        """Grow a contiguous window and copy content from guide until flip.

        Iteratively grows a contiguous window (positioned by importance) and
        copies content from the guide until the model's prediction flips.

        Parameters
        ----------
        x : np.ndarray
            Query time series of shape ``(T,)`` or ``(C, T)``.
        guide : np.ndarray
            Native guide series with the same shape as ``x``.
        base_label : int
            Original predicted class label.
        imp : np.ndarray or None
            Importance map of shape ``(T,)``. If ``None``, uses
            ``|guide - x|`` as a heuristic.

        Returns
        -------
        start : int
            Start index of the replacement window.
        L : int
            Length of the replacement window.
        cf : np.ndarray
            Counterfactual series with the same shape as ``x``.
        cf_label : int
            Predicted class label for the counterfactual.
        """
        T = x.shape[-1] if x.ndim == 2 else x.shape[0]
        init_L = max(1, round(self.init_window_frac * T))
        step = max(1, round(self.step_frac * T))
        max_L = max(1, round(self.max_window_frac * T))

        # If no CAM: define importance as |guide - x| aggregated over channels
        if imp is None:
            imp = np.abs(guide - x) if x.ndim == 1 else np.abs(guide - x).sum(axis=0)
        elif imp.ndim != 1 or imp.shape[0] != T:
            raise ValueError("Internal: importance must be (T,) at this point.")

        # Precompute cumulative sum for argmax window queries
        assert imp is not None
        csum = np.concatenate([[0.0], np.cumsum(imp)])  # length T+1

        # convenience to evaluate and check flip
        def _predict_label(arr: np.ndarray) -> int:
            cb, _ = ensure_batch_shape(arr)
            return int(np.argmax(self.predict_proba(cb)[0]))

        # linearly grow L until flip or max_L
        L = init_L

        def best_start_for_L(L_: int) -> int:
            """Return the start index of the contiguous window of length L_ with
            maximal importance sum."""

            return int(np.argmax(csum[L_:] - csum[:-L_]))

        while max_L >= L:
            start = best_start_for_L(L)
            cf = self._replace_window(x, guide, start, L)
            cf_label = _predict_label(cf)
            if cf_label != base_label:
                return start, L, cf, cf_label
            L += step

        # Worst case: replace entire series
        cf = self._replace_window(x, guide, 0, T)
        cf_label = _predict_label(cf)
        return 0, T, cf, cf_label

    def _replace_window(self, x: np.ndarray, guide: np.ndarray, start: int, L: int) -> np.ndarray:
        """Copy a contiguous window from guide into x.

        Parameters
        ----------
        x : np.ndarray
            Original time series of shape ``(T,)`` or ``(C, T)``.
        guide : np.ndarray
            Guide series with the same shape as ``x``.
        start : int
            Start index of the window to replace.
        L : int
            Length of the window to replace.

        Returns
        -------
        np.ndarray
            Modified series with window ``[start, start+L)`` replaced
            from ``guide``.
        """
        end = min(start + L, x.shape[-1] if x.ndim == 2 else x.shape[0])
        if x.ndim == 1:
            out = x.copy()
            out[start:end] = guide[start:end]
            return out
        else:
            out = x.copy()
            out[:, start:end] = guide[:, start:end]
            return out

    def _global_mean_like(self, X: np.ndarray) -> np.ndarray:
        """Compute the global mean of the reference set.

        Parameters
        ----------
        X : np.ndarray
            Reference set of shape ``(N, T)`` or ``(N, C, T)``.

        Returns
        -------
        np.ndarray
            Mean series of shape ``(T,)`` or ``(C, T)``.

        Raises
        ------
        ValueError
            If ``X`` has unsupported number of dimensions.
        """
        if X.ndim == 2:  # (N, T)
            mean_2d: np.ndarray = X.mean(axis=0)
            return mean_2d
        if X.ndim == 3:  # (N, C, T)
            mean_3d: np.ndarray = X.mean(axis=0)
            return mean_3d
        raise ValueError(f"Unsupported X shape for mean: {X.shape}")

    def _distance_vec(self, x: np.ndarray, Xc: np.ndarray) -> np.ndarray:
        """Compute distances from query to candidate set.

        Parameters
        ----------
        x : np.ndarray
            Query time series of shape ``(T,)`` or ``(C, T)``.
        Xc : np.ndarray
            Candidate set of shape ``(N, T)`` or ``(N, C, T)``.

        Returns
        -------
        np.ndarray
            1-D array of distances of length ``N``.
        """
        if self.distance == "euclidean" or not TSLEARN_AVAILABLE:
            xb, _ = ensure_batch_shape(x)
            return euclidean_cdist_flat(xb, Xc).ravel()
        return dtw_distance_vec_multich(x, Xc)

    def explain_k(
        self,
        x: np.ndarray,
        k: int = 5,
        y_pred: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Generate k diverse counterfactuals using different unlike neighbors.

        NativeGuide naturally supports diverse counterfactual generation by
        using different unlike neighbors as guides. Each counterfactual is
        generated using a different neighbor, producing structurally diverse
        explanations.

        Parameters
        ----------
        x : np.ndarray
            Input time series of shape ``(T,)`` or ``(C, T)``.
        k : int, default 5
            Number of counterfactuals to generate.
        y_pred : int, optional
            Precomputed predicted label for ``x``.

        Returns
        -------
        cfs : np.ndarray
            Array of k counterfactuals with shape ``(k, ...)``.
        cf_labels : np.ndarray
            Array of k predicted labels.
        metas : list[dict]
            List of k metadata dictionaries.
        """
        xb, added = ensure_batch_shape(x)
        x1 = strip_batch(xb, added)

        base_label = int(np.argmax(self.predict_proba(xb)[0])) if y_pred is None else int(y_pred)

        # Find k unlike neighbors to use as diverse guides
        # Use pre-computed reference set predictions (computed once in __post_init__)
        yhat = self._ref_yhat
        unlike_mask = yhat != base_label

        if not np.any(unlike_mask):
            # No unlike neighbors - fall back to default behavior
            return super().explain_k(x, k=k, y_pred=y_pred)

        X_unlike = self.X_ref[unlike_mask]
        dvec = self._distance_vec(x1, X_unlike)
        order = np.argsort(dvec)
        ref_indices = np.flatnonzero(unlike_mask)

        # Generate CF for each of the k nearest unlike neighbors
        cfs: list[np.ndarray] = []
        cf_labels: list[int] = []
        metas: list[dict[str, Any]] = []

        n_available = min(k, len(order))
        for i in range(n_available):
            nun_in_unlike = int(order[i])
            nun_index_in_ref = int(ref_indices[nun_in_unlike])
            guide = X_unlike[nun_in_unlike]
            neighbor_distance = float(dvec[nun_in_unlike])

            meta: dict[str, Any]
            if self.method == "blend":
                cf, cf_label, beta = self._blend_search(x1, guide, base_label)
                meta = {
                    "method": self.method,
                    "distance": self.distance,
                    "k_index": i,
                    "nun_index_in_X": nun_index_in_ref,
                    "neighbor_indices": None,
                    "neighbor_distance": neighbor_distance,
                    "beta": beta,
                    "window_start": None,
                    "window_len": None,
                }
            else:
                imp = self._get_importance(x1, base_label)
                start, L, cf, cf_label = self._window_growth_search(x1, guide, base_label, imp)
                meta = {
                    "method": self.method,
                    "distance": self.distance,
                    "k_index": i,
                    "nun_index_in_X": nun_index_in_ref,
                    "neighbor_indices": None,
                    "neighbor_distance": neighbor_distance,
                    "beta": None,
                    "window_start": start,
                    "window_len": L,
                }

            cfs.append(cf)
            cf_labels.append(cf_label)
            metas.append(meta)

        # If we have fewer than k unlike neighbors, pad with best result
        while len(cfs) < k:
            best_idx = 0  # Use the nearest neighbor result
            cf = cfs[best_idx].copy()
            cf_label = cf_labels[best_idx]
            new_meta: dict[str, Any] = metas[best_idx].copy()
            new_meta["k_index"] = len(cfs)
            new_meta["note"] = "duplicated from nearest neighbor"
            cfs.append(cf)
            cf_labels.append(cf_label)
            metas.append(new_meta)

        return np.array(cfs), np.array(cf_labels), metas
