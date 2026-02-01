"""Distribution-based counterfactual evaluation metrics.

This module provides metrics that evaluate counterfactuals based on their
relationship to the data distribution: Plausibility and Diversity.

Classes
-------
Plausibility
    Evaluates whether counterfactuals lie within the training data distribution.
Diversity
    Measures diversity among multiple counterfactuals for the same query.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.utils.extmath import randomized_svd

from ..base import Metric


class Plausibility(Metric):
    """Plausibility scored via an outlier detector.

    Evaluates whether counterfactuals lie within the training data
    distribution using outlier detection methods.

    Parameters
    ----------
    method : {'lof', 'if', 'mp_ocsvm'}, default 'lof'
        Detector backend:

        - ``'lof'``: LocalOutlierFactor in novelty mode (Breunig et al., 2000).
        - ``'if'``: IsolationForest (Liu et al., 2008).
        - ``'mp_ocsvm'``: Matrix Profile features (Yeh et al., 2016) with OneClassSVM.
    **kwargs
        Additional arguments passed to the detector.

    Notes
    -----
    When optional packages (e.g., ``stumpy``) are unavailable, the
    implementation falls back to safe alternatives.
    """

    direction = "maximize"

    def __init__(self, method: str = "lof", **kwargs):
        self.method = method
        self.kwargs = kwargs
        # Cache fitted detectors to avoid refitting on the same training data.
        # Cache key is a tuple of (method, id(train_data), train_data.shape).
        self._detector_cache: dict[tuple, Any] = {}
        self._mp_feature_cache: dict[tuple, tuple[np.ndarray, Any]] = {}

    def name(self) -> str:
        """Return the metric name.

        Returns
        -------
        str
            ``'plausibility_{method}'``.
        """
        return f"plausibility_{self.method}"

    def _get_or_fit_detector(self, Y: np.ndarray, train_id: int) -> Any:
        """Get cached detector or fit a new one.

        Parameters
        ----------
        Y : np.ndarray
            Flattened training data for fitting the detector.
        train_id : int
            Identity of the original training data array (from id()).

        Returns
        -------
        detector
            Fitted LOF or IsolationForest detector.
        """
        cache_key = (self.method, train_id, Y.shape)
        if cache_key in self._detector_cache:
            return self._detector_cache[cache_key]

        if self.method == "lof":
            detector = LocalOutlierFactor(novelty=True, **self.kwargs)
        elif self.method == "if":
            detector = IsolationForest(**self.kwargs)
        else:
            raise ValueError(f"Cannot cache detector for method: {self.method}")

        detector.fit(Y)
        self._detector_cache[cache_key] = detector
        return detector

    def clear_cache(self) -> None:
        """Clear cached fitted detectors to free memory."""
        self._detector_cache.clear()
        self._mp_feature_cache.clear()

    def compute(
        self,
        X: np.ndarray,
        X_cf: np.ndarray,
        X_train: np.ndarray | None = None,
        **kwargs,
    ) -> float:
        """Compute plausibility score.

        Parameters
        ----------
        X : np.ndarray
            Original instances.
        X_cf : np.ndarray
            Counterfactual instances.
        X_train : np.ndarray, optional
            Training data for fitting the detector. If ``None``, uses ``X``.
        **kwargs
            Ignored.

        Returns
        -------
        float
            Fraction of counterfactuals classified as inliers, in ``[0, 1]``.
        """
        X_train = np.asarray(X_train) if X_train is not None else None
        X_cf = np.asarray(X_cf)
        train_for_detector = X_train if X_train is not None else X
        Y = np.asarray(train_for_detector).reshape(train_for_detector.shape[0], -1)
        Z = X_cf.reshape(X_cf.shape[0], -1)

        # Use id() of the original training data for cache key
        train_id = id(train_for_detector)

        if self.method in ("lof", "if"):
            detector = self._get_or_fit_detector(Y, train_id)
            pred = detector.predict(Z)
            inlier = pred == 1
            return float(np.mean(inlier))
        elif self.method == "mp_ocsvm":
            return self._compute_mp_ocsvm(train_for_detector, X_cf, Y, Z, train_id)
        else:
            raise ValueError(f"Unknown plausibility method: {self.method}")

    def _compute_mp_ocsvm(
        self,
        train_for_detector: np.ndarray,
        X_cf: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        train_id: int,
    ) -> float:
        """Compute plausibility using Matrix Profile + OneClassSVM."""
        try:
            import stumpy
        except Exception:
            stumpy = None

        if stumpy is None:
            # Fallback: use cached OneClassSVM on flattened features
            cache_key = ("mp_ocsvm_fallback", train_id, Y.shape)
            if cache_key in self._detector_cache:
                oc = self._detector_cache[cache_key]
            else:
                oc = OneClassSVM(**self.kwargs)
                oc.fit(Y)
                self._detector_cache[cache_key] = oc
            pred = oc.predict(Z)
            inlier = pred == 1
            return float(np.mean(inlier))

        def _mp_feature(series: np.ndarray, train_set: np.ndarray) -> float:
            q = np.asarray(series).reshape(-1)
            mins = []
            for t in train_set:
                tflat = np.asarray(t).reshape(-1)
                try:
                    if tflat.size < q.size:
                        prof = stumpy.core.mass(tflat, q)
                    else:
                        prof = stumpy.core.mass(q, tflat)
                    mins.append(float(np.min(prof)))
                except Exception:
                    L = q.size
                    if tflat.size < L:
                        mins.append(float(np.linalg.norm(q - tflat)))
                    else:
                        windows = np.lib.stride_tricks.sliding_window_view(tflat, L)
                        d = np.sqrt(np.sum((windows - q) ** 2, axis=1))
                        mins.append(float(np.min(d)))
            return float(np.mean(mins))

        # Check if we have cached train features and fitted SVM for this training data
        cache_key = ("mp_ocsvm", train_id, train_for_detector.shape)
        if cache_key in self._mp_feature_cache:
            train_feats, oc = self._mp_feature_cache[cache_key]
        else:
            train_feats = np.array(
                [
                    _mp_feature(train_for_detector[i], train_for_detector)
                    for i in range(train_for_detector.shape[0])
                ]
            )
            train_feats = train_feats.reshape(-1, 1)
            oc = OneClassSVM(**self.kwargs)
            oc.fit(train_feats)
            self._mp_feature_cache[cache_key] = (train_feats, oc)

        cf_feats = np.array(
            [_mp_feature(X_cf[j].reshape(-1), train_for_detector) for j in range(X_cf.shape[0])]
        )
        cf_feats = cf_feats.reshape(-1, 1)

        # Use cached fitted SVM (oc was retrieved from cache or just fitted above)
        pred = oc.predict(cf_feats)
        inlier = pred == 1
        return float(np.mean(inlier))


class Diversity(Metric):
    """Diversity of multiple counterfactuals using DPP-inspired log-determinant.

    Measures diversity among multiple counterfactuals for the same query.
    Higher values indicate more diverse counterfactuals.

    Notes
    -----
    Expects ``X_cf`` with shape ``(M, K, ...)`` where ``K`` is the number of
    counterfactuals per query.

    See Mothilal et al. (2020) and Kulesza & Taskar (2012) for details.
    """

    direction = "maximize"

    def name(self) -> str:
        """Return the metric name.

        Returns
        -------
        str
            ``'diversity_dpp'``.
        """
        return "diversity_dpp"

    def compute(
        self,
        X: np.ndarray,
        X_cf: np.ndarray,
        max_components: int = 50,
        **kwargs,
    ) -> float:
        """Compute diversity score.

        Parameters
        ----------
        X : np.ndarray
            Original instances.
        X_cf : np.ndarray
            Counterfactual instances of shape ``(M, K, ...)`` where ``K`` is
            the number of counterfactuals per query.
        max_components : int, default 50
            Maximum number of components for randomized SVD approximation.
        **kwargs
            May contain ``_X_cf_all`` with full counterfactuals when the
            benchmark passes first-CF-only as ``X_cf`` for other metrics.

        Returns
        -------
        float
            Diversity score (higher = more diverse). Returns ``np.nan`` if
            ``X_cf`` has fewer than 3 dimensions (single CF per query).
        """
        # Use _X_cf_all if provided (contains all k counterfactuals per instance)
        X_cfs = np.asarray(kwargs.get("_X_cf_all", X_cf))
        if X_cfs.ndim < 3:
            # Single counterfactual per query - diversity is not applicable
            return float("nan")
        M, K = X_cfs.shape[:2]
        logdets: list[float] = []
        for i in range(M):
            S = X_cfs[i].reshape(K, -1)
            D = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(axis=2))
            Kmat = 1.0 / (1.0 + D + 1e-12)
            Kmat = 0.5 * (Kmat + Kmat.T)
            if max_components >= K:
                try:
                    sign, logdet = np.linalg.slogdet(Kmat)
                    logdets.append(float(logdet) if sign > 0 else float(-np.inf))
                except Exception:
                    vals = np.linalg.eigvalsh(Kmat)
                    vals = vals[vals > 0]
                    logdets.append(float(np.sum(np.log(vals))) if vals.size else float(-np.inf))
            else:
                try:
                    r = min(max_components, K - 1)
                    _, Svals, _ = randomized_svd(Kmat, n_components=r)
                    approx_logdet = float(np.sum(np.log(Svals + 1e-12)))
                    logdets.append(approx_logdet)
                except Exception:
                    vals = np.linalg.eigvalsh(Kmat)
                    vals = vals[vals > 0]
                    logdets.append(float(np.sum(np.log(vals))) if vals.size else float(-np.inf))

        valid = [ld for ld in logdets if np.isfinite(ld) and ld > -1e300]
        if not valid:
            return float("nan")
        mean_logdet = float(np.mean(valid))
        return float(np.exp(mean_logdet))
