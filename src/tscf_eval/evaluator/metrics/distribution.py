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

from typing import Any, Literal
import warnings

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.utils.extmath import randomized_svd

from tscf_eval.counterfactuals.utils._distance import dtw_distance_vec_multich

from ..base import Metric


class Plausibility(Metric):
    """Plausibility scored via an outlier detector.

    Evaluates whether counterfactuals lie within the training data
    distribution using outlier detection methods.

    Parameters
    ----------
    method : {'lof', 'if', 'mp_ocsvm', 'dtw_lof'}, default 'dtw_lof'
        Detector backend:

        - ``'lof'``: LocalOutlierFactor in novelty mode (Breunig et al., 2000).
        - ``'if'``: IsolationForest (Liu et al., 2008).
        - ``'mp_ocsvm'``: Matrix Profile features (Yeh et al., 2016) with OneClassSVM.
        - ``'dtw_lof'``: LOF with DTW distance (precomputed distance matrix).
          Uses ``tslearn`` for DTW; falls back to Euclidean if unavailable.
          More appropriate for time series as it respects temporal alignment.
    **kwargs
        Additional arguments passed to the detector.

    Notes
    -----
    When optional packages (e.g., ``stumpy``) are unavailable, the
    implementation falls back to safe alternatives.
    """

    direction = "maximize"

    def __init__(self, method: Literal["lof", "if", "mp_ocsvm", "dtw_lof"] = "dtw_lof", **kwargs):
        """Initialize the Plausibility metric.

        Parameters
        ----------
        method : {"lof", "if", "mp_ocsvm", "dtw_lof"}, default "dtw_lof"
            Outlier detection backend to use.
        **kwargs
            Additional arguments passed to the underlying detector.
        """
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
        """Clear cached fitted detectors and matrix profile features to free memory."""
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
        elif self.method == "dtw_lof":
            return self._compute_dtw_lof(train_for_detector, X_cf, train_id)
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
        """Compute plausibility using Matrix Profile features with OneClassSVM.

        Parameters
        ----------
        train_for_detector : np.ndarray
            Training data in original shape, used for matrix profile computation.
        X_cf : np.ndarray
            Counterfactual instances in original shape.
        Y : np.ndarray
            Flattened training data, shape ``(n_train, n_features)``.
        Z : np.ndarray
            Flattened counterfactual data, shape ``(n_cf, n_features)``.
        train_id : int
            Identity of the training data array for caching.

        Returns
        -------
        float
            Fraction of counterfactuals classified as inliers, in ``[0, 1]``.
        """
        try:
            import stumpy
        except Exception:
            stumpy = None

        if stumpy is None:
            warnings.warn(
                "stumpy is not installed. Plausibility(method='mp_ocsvm') is falling "
                "back to OneClassSVM on flattened features instead of Matrix Profile "
                "features. Install stumpy for proper matrix profile computation: "
                "pip install stumpy",
                UserWarning,
                stacklevel=2,
            )
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
            """Compute mean matrix profile distance of a series against a training set.

            Parameters
            ----------
            series : np.ndarray
                Query time series.
            train_set : np.ndarray
                Training set to compare against.

            Returns
            -------
            float
                Mean of the minimum matrix profile distances.
            """
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

    def _compute_dtw_lof(
        self,
        X_train: np.ndarray,
        X_cf: np.ndarray,
        train_id: int,
    ) -> float:
        """Compute plausibility using LOF with precomputed DTW distances.

        Builds a full DTW distance matrix between training data and
        counterfactuals, then uses LOF in ``metric="precomputed"`` mode.
        This respects temporal alignment, unlike the flat LOF/IF methods.

        Parameters
        ----------
        X_train : np.ndarray
            Training data in original shape, shape ``(n_train, ...)``.
        X_cf : np.ndarray
            Counterfactual instances, shape ``(n_cf, ...)``.
        train_id : int
            Identity of the training data array for caching.

        Returns
        -------
        float
            Fraction of counterfactuals classified as inliers, in ``[0, 1]``.
        """
        N_train = X_train.shape[0]
        N_cf = X_cf.shape[0]

        # Check cache for pre-fitted LOF and training distance matrix
        cache_key = ("dtw_lof", train_id, X_train.shape)

        if cache_key in self._detector_cache:
            lof, D_train = self._detector_cache[cache_key]
        else:
            # Compute pairwise DTW distance matrix for training data
            D_train = np.zeros((N_train, N_train), dtype=float)
            for i in range(N_train):
                # dtw_distance_vec_multich(x, B) returns distances from x to each row of B
                D_train[i] = dtw_distance_vec_multich(X_train[i], X_train)
            # Ensure symmetry (numerical precision)
            D_train = 0.5 * (D_train + D_train.T)

            # Filter kwargs: only pass LOF-compatible arguments
            lof_kwargs = {
                k: v
                for k, v in self.kwargs.items()
                if k
                in (
                    "n_neighbors",
                    "algorithm",
                    "leaf_size",
                    "contamination",
                    "n_jobs",
                )
            }
            lof = LocalOutlierFactor(novelty=True, metric="precomputed", **lof_kwargs)
            lof.fit(D_train)
            self._detector_cache[cache_key] = (lof, D_train)

        # Compute distance matrix from each CF to each training instance
        D_cf = np.zeros((N_cf, N_train), dtype=float)
        for i in range(N_cf):
            D_cf[i] = dtw_distance_vec_multich(X_cf[i], X_train)

        pred = lof.predict(D_cf)
        inlier = pred == 1
        return float(np.mean(inlier))


class Diversity(Metric):
    """Diversity of multiple counterfactuals using DPP-inspired log-determinant.

    Measures diversity among multiple counterfactuals for the same query.
    Higher values indicate more diverse counterfactuals.

    Parameters
    ----------
    distance : {"euclidean", "dtw"}, default "dtw"
        Distance function used to build the pairwise distance matrix
        between counterfactuals for each query.

        - ``"euclidean"``: Euclidean distance on flattened vectors.
        - ``"dtw"``: Per-channel DTW distance (averaged across channels).
          Requires ``tslearn``; falls back to Euclidean if unavailable.

    Notes
    -----
    Expects ``X_cf`` with shape ``(M, K, ...)`` where ``K`` is the number of
    counterfactuals per query.

    See Mothilal et al. (2020) and Kulesza & Taskar (2012) for details.
    """

    direction = "maximize"

    def __init__(self, distance: Literal["euclidean", "dtw"] = "dtw"):
        """Initialize the Diversity metric.

        Parameters
        ----------
        distance : {"euclidean", "dtw"}, default "dtw"
            Distance function for building pairwise distance matrices.
        """
        self.distance = distance

    def name(self) -> str:
        """Return the metric name.

        Returns
        -------
        str
            ``'diversity_dpp'`` for Euclidean distance or
            ``'diversity_dpp_dtw'`` for DTW distance.
        """
        if self.distance == "dtw":
            return "diversity_dpp_dtw"
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

        Raises
        ------
        ValueError
            If ``distance`` is not a supported value.
        """
        # Use _X_cf_all if provided (contains all k counterfactuals per instance)
        X_cfs = np.asarray(kwargs.get("_X_cf_all", X_cf))
        if X_cfs.ndim < 3:
            # Single counterfactual per query - diversity is not applicable
            return float("nan")
        M, K = X_cfs.shape[:2]
        logdets: list[float] = []
        for i in range(M):
            D = self._pairwise_distances(X_cfs[i], K)
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

    def _pairwise_distances(self, cfs: np.ndarray, K: int) -> np.ndarray:
        """Compute pairwise distance matrix between K counterfactuals.

        Parameters
        ----------
        cfs : np.ndarray
            Counterfactuals for a single query, shape ``(K, ...)``.
        K : int
            Number of counterfactuals.

        Returns
        -------
        np.ndarray
            Pairwise distance matrix of shape ``(K, K)``.
        """
        if self.distance == "euclidean":
            S = cfs.reshape(K, -1)
            D: np.ndarray = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(axis=2))
            return D
        if self.distance == "dtw":
            D = np.zeros((K, K), dtype=float)
            for j in range(K):
                D[j] = dtw_distance_vec_multich(cfs[j], cfs)
            D = 0.5 * (D + D.T)
            return D
        raise ValueError(f"Unknown distance: {self.distance!r}. Expected 'euclidean' or 'dtw'.")
