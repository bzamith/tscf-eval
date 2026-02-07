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
    from tslearn.neighbors import KNeighborsTimeSeries

    TSLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    TSLEARN_AVAILABLE = False
    KNeighborsTimeSeries = None  # type: ignore


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

        - ``'euclidean'``: Euclidean distance on flattened vectors. Faster but
          ignores temporal alignment.
        - ``'dtw'``: Dynamic Time Warping distance (per-channel, averaged).
          Respects temporal shifts and is recommended for time series.
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

    # Only used when method="cam": importance fn(series, y_pred) -> (T,) or (C,T)
    cam_importance_fn: Callable[[np.ndarray, int], np.ndarray] | None = None

    def __post_init__(self):
        """Initialise probability wrapper, RNG, reference data, and label mapping.

        Validates all hyperparameters, pre-computes reference-set
        predictions, and checks method-specific requirements (e.g.
        ``cam_importance_fn`` when ``method='cam'``).

        Raises
        ------
        ValueError
            If ``X`` and ``y`` have mismatched sample counts, ``method`` or
            ``distance`` is not in the allowed set, ``beta_step`` or
            ``target_prob`` is outside ``(0, 1]``, or ``method='cam'``
            without a ``cam_importance_fn``.
        """
        X_ref, y_ref = self.data
        self.X_ref = np.asarray(X_ref)
        self.y_ref = np.asarray(y_ref).ravel()
        if self.X_ref.shape[0] != self.y_ref.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        self.predict_proba = soft_predict_proba_fn(self.model)
        self.rng = np.random.default_rng(self.random_state)

        self._init_label_mapping(self.model, self.y_ref)

        # Pre-compute reference set predictions to avoid redundant calls
        self._ref_probs = self.predict_proba(self.X_ref)
        # Store as probability column indices (consistent with internal index space)
        self._ref_yhat = np.argmax(self._ref_probs, axis=1)

        if self.method not in {"blend", "ng", "dtw_dba", "cam"}:
            raise ValueError("method must be one of {'blend', 'ng', 'dtw_dba', 'cam'}")
        if self.distance not in {"euclidean", "dtw"}:
            raise ValueError("distance must be one of {'euclidean', 'dtw'}")
        if self.method == "dtw_dba" and self.k_unlike < 2:
            self.k_unlike = 2
        if self.method == "cam" and self.cam_importance_fn is None:
            raise ValueError("cam_importance_fn must be provided when method='cam'.")

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

        if y_pred is None:
            base_idx = int(np.argmax(self.predict_proba(xb)[0]))
        else:
            base_idx = self._label_to_idx(y_pred)

        # Step 1: Retrieve the native guide
        if self.method == "dtw_dba":
            guide, guide_meta = self._build_dba_guide(x1, base_idx)
        else:
            guide, guide_meta = self._find_nearest_unlike_neighbor(x1, base_idx)

        # Step 2: Generate counterfactual using the chosen strategy
        if self.method == "blend":
            cf, cf_idx, beta = self._blend_query_with_guide(x1, guide, base_idx)
            window_start, window_len = None, None
        else:
            importance = self._compute_cam_importance(guide) if self.method == "cam" else None
            window_start, window_len, cf, cf_idx = self._grow_window_until_flip(
                x1, guide, base_idx, importance
            )
            beta = None

        # Step 3: Assemble result
        cf_label = self._idx_to_label(cf_idx)
        meta = self._build_meta(
            guide_meta, beta=beta, window_start=window_start, window_len=window_len
        )
        return cf, cf_label, meta

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

        if y_pred is None:
            base_idx = int(np.argmax(self.predict_proba(xb)[0]))
        else:
            base_idx = self._label_to_idx(y_pred)

        # Step 1: Find k unlike neighbors to use as diverse guides
        result = self._find_unlike_neighbors(x1, base_idx, k=k)
        if result is None:
            return super().explain_k(x, k=k, y_pred=y_pred)

        distances, indices_in_unlike, ref_indices, X_unlike = result

        # Step 2: Generate a counterfactual for each unlike neighbor
        cfs: list[np.ndarray] = []
        cf_labels: list[int] = []
        metas: list[dict[str, Any]] = []

        for i in range(len(indices_in_unlike)):
            nun_in_unlike = int(indices_in_unlike[i])
            guide = X_unlike[nun_in_unlike]
            guide_meta = {
                "nun_index_in_X": int(ref_indices[nun_in_unlike]),
                "neighbor_indices": None,
                "neighbor_distance": float(distances[i]),
            }

            if self.method == "blend":
                cf, cf_idx, beta = self._blend_query_with_guide(x1, guide, base_idx)
                window_start, window_len = None, None
            else:
                importance = self._compute_cam_importance(guide) if self.method == "cam" else None
                window_start, window_len, cf, cf_idx = self._grow_window_until_flip(
                    x1, guide, base_idx, importance
                )
                beta = None

            cf_label = self._idx_to_label(cf_idx)
            meta = self._build_meta(
                guide_meta,
                beta=beta,
                window_start=window_start,
                window_len=window_len,
                k_index=i,
            )
            cfs.append(cf)
            cf_labels.append(cf_label)
            metas.append(meta)

        # Step 3: Pad with nearest neighbor result if fewer than k available
        while len(cfs) < k:
            best_idx = 0
            cf = cfs[best_idx].copy()
            cf_label = cf_labels[best_idx]
            new_meta: dict[str, Any] = metas[best_idx].copy()
            new_meta["k_index"] = len(cfs)
            new_meta["note"] = "duplicated from nearest neighbor"
            cfs.append(cf)
            cf_labels.append(cf_label)
            metas.append(new_meta)

        return np.array(cfs), np.array(cf_labels), metas

    def _find_nearest_unlike_neighbor(
        self, x: np.ndarray, base_idx: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Find the single nearest unlike neighbor (NUN) from the reference set.

        Used by the 'blend', 'ng', and 'cam' methods.

        Parameters
        ----------
        x : np.ndarray
            Query time series of shape ``(T,)`` or ``(C, T)``.
        base_idx : int
            Probability column index of the original predicted class.

        Returns
        -------
        nun : np.ndarray
            Nearest unlike neighbor with the same shape as ``x``.
        metadata : dict
            Dictionary with ``'nun_index_in_X'``, ``'neighbor_indices'``,
            and ``'neighbor_distance'``.
        """
        result = self._find_unlike_neighbors(x, base_idx, k=1)
        if result is None:
            guide = self._fallback_global_mean(self.X_ref)
            return guide, {
                "nun_index_in_X": None,
                "neighbor_indices": None,
                "neighbor_distance": float(np.nan),
                "failure_reason": "no_unlike_neighbors",
            }

        distances, indices_in_unlike, ref_indices, X_unlike = result
        nun_in_unlike = int(indices_in_unlike[0])
        nun = X_unlike[nun_in_unlike]
        return nun, {
            "nun_index_in_X": int(ref_indices[nun_in_unlike]),
            "neighbor_indices": None,
            "neighbor_distance": float(distances[0]),
        }

    def _build_dba_guide(self, x: np.ndarray, base_idx: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Build a DTW-DBA barycenter guide from k unlike neighbors.

        Used by the 'dtw_dba' method.

        Parameters
        ----------
        x : np.ndarray
            Query time series of shape ``(T,)`` or ``(C, T)``.
        base_idx : int
            Probability column index of the original predicted class.

        Returns
        -------
        guide : np.ndarray
            DBA barycenter of k unlike neighbors, same shape as ``x``.
        metadata : dict
            Dictionary with ``'nun_index_in_X'``, ``'neighbor_indices'``,
            and ``'neighbor_distance'``.
        """
        result = self._find_unlike_neighbors(x, base_idx, k=self.k_unlike)
        if result is None:
            guide = self._fallback_global_mean(self.X_ref)
            return guide, {
                "nun_index_in_X": None,
                "neighbor_indices": None,
                "neighbor_distance": float(np.nan),
                "failure_reason": "no_unlike_neighbors",
            }

        distances, indices_in_unlike, ref_indices, X_unlike = result
        nbrs = X_unlike[indices_in_unlike]
        guide = dba_barycenter_multich(nbrs)
        nun_in_unlike = int(indices_in_unlike[0])
        return guide, {
            "nun_index_in_X": int(ref_indices[nun_in_unlike]),
            "neighbor_indices": ref_indices[indices_in_unlike].tolist(),
            "neighbor_distance": float(distances[0]),
        }

    def _find_unlike_neighbors(
        self, x: np.ndarray, base_idx: int, k: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Find k unlike neighbors from the reference set.

        Shared helper for both single-NUN and multi-neighbor retrieval.

        Parameters
        ----------
        x : np.ndarray
            Query time series of shape ``(T,)`` or ``(C, T)``.
        base_idx : int
            Probability column index of the original predicted class.
        k : int
            Number of unlike neighbors to retrieve.

        Returns
        -------
        tuple or None
            ``(distances, indices_in_unlike, ref_indices, X_unlike)`` if
            unlike neighbors exist, otherwise ``None`` (with a warning).
        """
        yhat = self._ref_yhat
        unlike_mask = yhat != base_idx
        if not np.any(unlike_mask):
            warnings.warn(
                f"NativeGuide: No unlike neighbors found in reference set. "
                f"The classifier predicts all {len(yhat)} reference samples as class "
                f"{self._idx_to_label(base_idx)}. Falling back to global mean, which "
                f"may not produce a valid counterfactual. Consider using a different "
                f"dataset or classifier with more diverse predictions.",
                UserWarning,
                stacklevel=3,
            )
            return None

        X_unlike = self.X_ref[unlike_mask]
        ref_indices = np.flatnonzero(unlike_mask)
        k_actual = min(k, len(X_unlike))

        if TSLEARN_AVAILABLE:
            distances, indices_in_unlike = self._find_k_neighbors_tslearn(x, X_unlike, k=k_actual)
        else:
            warnings.warn(
                "tslearn is not installed. NativeGuide is using a manual fallback "
                "for neighbor search instead of KNeighborsTimeSeries. "
                "Install tslearn for the original algorithm: pip install tslearn",
                UserWarning,
                stacklevel=3,
            )
            dvec = self._compute_distances(x, X_unlike)
            order = np.argsort(dvec)[:k_actual]
            indices_in_unlike = order
            distances = dvec[order]

        return distances, indices_in_unlike, ref_indices, X_unlike

    def _blend_query_with_guide(
        self, x: np.ndarray, guide: np.ndarray, base_idx: int
    ) -> tuple[np.ndarray, int, float]:
        """Blend query with guide via weighted DTW barycenter averaging.

        Incrementally increases the guide's influence (beta) until the model's
        predicted class flips or the target probability is reached.

        Parameters
        ----------
        x : np.ndarray
            Query series of shape ``(T,)`` or ``(C, T)``.
        guide : np.ndarray
            Native guide (NUN) of same shape as ``x``.
        base_idx : int
            Probability column index of the original predicted class.

        Returns
        -------
        cf : np.ndarray
            Counterfactual series with the same shape as ``x``.
        cf_idx : int
            Probability column index for the counterfactual prediction.
        beta : float
            Final blending weight in ``[0, 1]``.
        """
        beta = 0.0
        cf = x.copy()
        cf_idx, alt_prob = self._predict_idx_and_max_alt_prob(cf, base_idx)

        while alt_prob < self.target_prob and beta < 1.0:
            beta = min(beta + self.beta_step, 1.0)
            cf = weighted_dba_multich(x, guide, beta)
            cf_idx, alt_prob = self._predict_idx_and_max_alt_prob(cf, base_idx)
            if cf_idx != base_idx:
                break

        return cf, cf_idx, beta

    def _compute_cam_importance(self, guide: np.ndarray) -> np.ndarray:
        """Compute the CAM importance map from the guide's class activation.

        Predicts the guide's class internally and passes it to the user-provided
        ``cam_importance_fn``. This matches the original paper where the NUN's
        CAM weights identify which region to swap.

        Parameters
        ----------
        guide : np.ndarray
            The native guide (NUN) of shape ``(T,)`` or ``(C, T)``.

        Returns
        -------
        np.ndarray
            Importance map of shape ``(T,)``.

        Raises
        ------
        ValueError
            If the shape returned by ``cam_importance_fn`` does not match
            ``(T,)`` for univariate input or ``(T,)`` / ``(C, T)`` for
            multivariate input.
        """
        assert self.cam_importance_fn is not None
        guide_b, _ = ensure_batch_shape(guide)
        guide_pred_idx = int(np.argmax(self.predict_proba(guide_b)[0]))
        guide_label = self._idx_to_label(guide_pred_idx)
        imp = np.asarray(self.cam_importance_fn(guide, guide_label))

        if guide.ndim == 1:
            if imp.ndim != 1 or imp.shape[0] != guide.shape[0]:
                raise ValueError("cam_importance_fn must return shape (T,) for univariate input.")
            return imp

        # multivariate input: accept (T,) or (C, T)
        if imp.ndim == 1:
            if imp.shape[0] != guide.shape[1]:
                raise ValueError("When returning (T,), length must match time length.")
            return imp
        if imp.ndim == 2 and imp.shape == guide.shape:
            sum_result: np.ndarray = imp.sum(axis=0)
            return sum_result
        raise ValueError("cam_importance_fn must return (T,) or (C,T) for multivariate input.")

    def _grow_window_until_flip(
        self,
        x: np.ndarray,
        guide: np.ndarray,
        base_idx: int,
        importance: np.ndarray | None,
    ) -> tuple[int, int, np.ndarray, int]:
        """Grow a window from the guide into the query until the prediction flips.

        Iteratively increases the window length, positioning it at the most
        important region, and copies guide content into the query.

        Parameters
        ----------
        x : np.ndarray
            Query time series of shape ``(T,)`` or ``(C, T)``.
        guide : np.ndarray
            Native guide series with the same shape as ``x``.
        base_idx : int
            Probability column index of the original predicted class.
        importance : np.ndarray or None
            Importance map of shape ``(T,)``. If ``None``, uses
            ``|guide - x|`` as a heuristic.

        Returns
        -------
        start : int
            Start index of the replacement window.
        length : int
            Length of the replacement window.
        cf : np.ndarray
            Counterfactual series with the same shape as ``x``.
        cf_idx : int
            Probability column index for the counterfactual prediction.

        Raises
        ------
        ValueError
            If ``importance`` is not ``None`` and its shape is not ``(T,)``.
        """
        T = x.shape[-1] if x.ndim == 2 else x.shape[0]

        # Compute importance if not provided (heuristic: |guide - x|)
        if importance is None:
            importance = np.abs(guide - x) if x.ndim == 1 else np.abs(guide - x).sum(axis=0)
        elif importance.ndim != 1 or importance.shape[0] != T:
            raise ValueError("Internal: importance must be (T,) at this point.")

        # Precompute cumulative sum for fast window-start queries
        cumsum = np.concatenate([[0.0], np.cumsum(importance)])

        # Grow window from length 1 to T, checking for flip at each step
        length = 1
        while length <= T:
            start = self._best_window_start(cumsum, length)
            cf = self._splice_guide_segment(x, guide, start, length)
            cf_idx = self._predict_class_idx(cf)
            if cf_idx != base_idx:
                return start, length, cf, cf_idx
            length += 1

        # Worst case: replace entire series
        cf = self._splice_guide_segment(x, guide, 0, T)
        cf_idx = self._predict_class_idx(cf)
        return 0, T, cf, cf_idx

    def _predict_class_idx(self, arr: np.ndarray) -> int:
        """Return the predicted probability column index for a series.

        Parameters
        ----------
        arr : np.ndarray
            Time series of shape ``(T,)`` or ``(C, T)``.

        Returns
        -------
        int
            Argmax index of the model's probability vector.
        """
        cb, _ = ensure_batch_shape(arr)
        return int(np.argmax(self.predict_proba(cb)[0]))

    def _predict_idx_and_max_alt_prob(self, arr: np.ndarray, base_idx: int) -> tuple[int, float]:
        """Predict class index and highest non-base probability.

        Parameters
        ----------
        arr : np.ndarray
            Time series of shape ``(T,)`` or ``(C, T)``.
        base_idx : int
            Probability column index of the base (original) class.

        Returns
        -------
        pred_idx : int
            Argmax index of the model's probability vector.
        alt_prob : float
            Highest probability among classes other than ``base_idx``.
        """
        cb, _ = ensure_batch_shape(arr)
        probs = self.predict_proba(cb)[0]
        pred_idx = int(np.argmax(probs))
        alt_prob = max((p for i, p in enumerate(probs) if i != base_idx), default=0.0)
        return pred_idx, alt_prob

    @staticmethod
    def _best_window_start(cumsum: np.ndarray, length: int) -> int:
        """Return the start index maximising importance over a window.

        Uses a precomputed cumulative sum for O(1) window-sum queries.

        Parameters
        ----------
        cumsum : np.ndarray
            Cumulative sum of the importance map, of length ``T + 1``
            (prepended with 0).
        length : int
            Window length to evaluate.

        Returns
        -------
        int
            Start index of the window with the highest total importance.
        """
        return int(np.argmax(cumsum[length:] - cumsum[:-length]))

    def _splice_guide_segment(
        self, x: np.ndarray, guide: np.ndarray, start: int, length: int
    ) -> np.ndarray:
        """Copy a contiguous segment from guide into x.

        Parameters
        ----------
        x : np.ndarray
            Original time series of shape ``(T,)`` or ``(C, T)``.
        guide : np.ndarray
            Guide series with the same shape as ``x``.
        start : int
            Start index of the segment to replace.
        length : int
            Length of the segment to replace.

        Returns
        -------
        np.ndarray
            Modified series with ``[start, start+length)`` replaced from guide.
        """
        end = min(start + length, x.shape[-1] if x.ndim == 2 else x.shape[0])
        out = x.copy()
        if x.ndim == 1:
            out[start:end] = guide[start:end]
        else:
            out[:, start:end] = guide[:, start:end]
        return out

    def _fallback_global_mean(self, X: np.ndarray) -> np.ndarray:
        """Compute the global mean of the reference set (fallback guide).

        Used when no unlike neighbors exist in the reference set.

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
            If ``X`` has fewer than 2 or more than 3 dimensions.
        """
        if X.ndim in (2, 3):
            result: np.ndarray = X.mean(axis=0)
            return result
        raise ValueError(f"Unsupported X shape for mean: {X.shape}")

    def _compute_distances(self, x: np.ndarray, X_candidates: np.ndarray) -> np.ndarray:
        """Compute distances from query to candidate set.

        Parameters
        ----------
        x : np.ndarray
            Query time series of shape ``(T,)`` or ``(C, T)``.
        X_candidates : np.ndarray
            Candidate set of shape ``(N, T)`` or ``(N, C, T)``.

        Returns
        -------
        np.ndarray
            1-D array of distances of length ``N``.
        """
        if self.distance != "euclidean" and not TSLEARN_AVAILABLE:
            warnings.warn(
                f"NativeGuide: distance='{self.distance}' was requested but tslearn "
                f"is not installed. Falling back to Euclidean distance. "
                f"Install tslearn for DTW support: pip install tslearn",
                UserWarning,
                stacklevel=2,
            )
        if self.distance == "euclidean" or not TSLEARN_AVAILABLE:
            xb, _ = ensure_batch_shape(x)
            return euclidean_cdist_flat(xb, X_candidates).ravel()
        return dtw_distance_vec_multich(x, X_candidates)

    def _find_k_neighbors_tslearn(
        self, x: np.ndarray, X_candidates: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors using tslearn's KNeighborsTimeSeries.

        Handles shape conversion between codebase convention ``(N, C, T)``
        and tslearn convention ``(N, T, C)``.

        Parameters
        ----------
        x : np.ndarray
            Query series of shape ``(T,)`` or ``(C, T)``.
        X_candidates : np.ndarray
            Candidate set of shape ``(N, T)`` or ``(N, C, T)``.
        k : int
            Number of neighbors to find.

        Returns
        -------
        distances : np.ndarray
            1-D array of k distances.
        indices : np.ndarray
            1-D array of k indices into ``X_candidates``.
        """
        if X_candidates.ndim == 2:
            X_tl = X_candidates[:, :, np.newaxis]
        else:
            X_tl = np.transpose(X_candidates, (0, 2, 1))

        q_tl = x[np.newaxis, :, np.newaxis] if x.ndim == 1 else x.T[np.newaxis, :, :]

        knn = KNeighborsTimeSeries(n_neighbors=k, metric=self.distance)
        knn.fit(X_tl)
        dists, inds = knn.kneighbors(q_tl, return_distance=True)
        return dists[0], inds[0]

    def _build_meta(
        self,
        guide_meta: dict[str, Any],
        *,
        beta: float | None = None,
        window_start: int | None = None,
        window_len: int | None = None,
        k_index: int | None = None,
    ) -> dict[str, Any]:
        """Build the metadata dictionary for an explanation result.

        Parameters
        ----------
        guide_meta : dict
            Metadata from guide retrieval, containing ``'nun_index_in_X'``,
            ``'neighbor_indices'``, and ``'neighbor_distance'``.
        beta : float or None
            Blending weight (only for ``method='blend'``).
        window_start : int or None
            Start index of the replacement window (window-based methods).
        window_len : int or None
            Length of the replacement window (window-based methods).
        k_index : int or None
            Index of this result within a ``explain_k`` batch. Omitted
            from the dict when ``None``.

        Returns
        -------
        dict
            Metadata dictionary with keys ``method``, ``distance``,
            ``nun_index_in_X``, ``neighbor_indices``, ``neighbor_distance``,
            ``beta``, ``window_start``, and ``window_len`` (plus ``k_index``
            when applicable).
        """
        meta: dict[str, Any] = {
            "method": self.method,
            "distance": self.distance,
            "nun_index_in_X": guide_meta.get("nun_index_in_X"),
            "neighbor_indices": guide_meta.get("neighbor_indices"),
            "neighbor_distance": guide_meta.get("neighbor_distance"),
            "beta": beta,
            "window_start": window_start,
            "window_len": window_len,
        }
        if k_index is not None:
            meta["k_index"] = k_index
        return meta
