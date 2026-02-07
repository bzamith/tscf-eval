"""Glacier counterfactual explainer implementation.

This module provides the ``Glacier`` class, an implementation of the Glacier
(Guided Locally Constrained Counterfactual Explanations) algorithm for
generating counterfactual explanations for time series classification.

The algorithm was originally developed by Zhendong Wang, Isak Samsten,
Ioanna Miliou, Rami Mochaourab, and Panagiotis Papapetrou at Stockholm University.

Original implementation: https://github.com/zhendong3wang/learning-time-series-counterfactuals

Classes
-------
Glacier
    Glacier counterfactual generator using gradient-based optimization with
    guided constraints.

Algorithm Overview
------------------
Glacier generates counterfactuals through gradient-based optimization:

1. Optionally encode the input time series into a latent space using an
   autoencoder (CNN or LSTM-based).
2. Compute importance weights using segment-based LIME (local importance),
   uniform weights, or unconstrained weights.
3. Optimize a composite loss function that balances:

   - **Prediction margin loss**: Drives the counterfactual toward the target class
   - **Proximity loss**: Penalizes deviations from the original, weighted by importance

4. Iterate until the classifier predicts the target class with sufficient
   confidence or the maximum iterations are reached.

5. If using an autoencoder, decode the optimized latent representation back
   to the original time series space.

Examples
--------
>>> from tscf_eval.counterfactuals import Glacier
>>> import numpy as np
>>>
>>> # Assume clf is a trained classifier with predict_proba
>>> glacier = Glacier(
...     model=clf,
...     data=(X_train, y_train),
...     pred_margin_weight=0.5,
...     learning_rate=0.01,
...     max_iter=100,
... )
>>>
>>> # Generate counterfactual for a test instance
>>> cf, cf_label, meta = glacier.explain(x_test)
>>> print(f"Converged: {meta['converged']}")
>>> print(f"Iterations: {meta['n_iterations']}")

References
----------
.. [glacier1] Wang, Z., Samsten, I., Miliou, I., Mochaourab, R., & Papapetrou, P. (2024).
       Glacier: Guided Locally Constrained Counterfactual Explanations for
       Time Series Classification. Machine Learning, 113(3).
       DOI: 10.1007/s10994-023-06502-x

.. [glacier2] Wang, Z., Samsten, I., Mochaourab, R., & Papapetrou, P. (2021).
       Learning Time Series Counterfactuals via Latent Space Representations.
       In International Conference on Discovery Science (DS'2021).

Notes
-----
This implementation provides a simplified version of Glacier that works
directly in the original time series space (without autoencoder) for
compatibility with any scikit-learn compatible classifier. The core
gradient-based optimization with weighted proximity constraints is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
import warnings

import numpy as np
from sklearn.linear_model import Ridge

from .base import Counterfactual
from .utils import (
    ensure_batch_shape,
    has_expensive_transform,
    soft_predict_proba_fn,
    strip_batch,
    supports_soft_probabilities,
)
from .utils._adam import AdamState

# Optional: stumpy for matrix-profile-based segmentation
try:
    import stumpy

    STUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    stumpy = None  # type: ignore[assignment]
    STUMPY_AVAILABLE = False

# Optional: scipy for STFT-based background identification
try:
    from scipy import signal as sp_signal

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    sp_signal = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False

WeightType = Literal["uniform", "local", "unconstrained"]


@dataclass
class Glacier(Counterfactual):
    """Glacier counterfactual generator using gradient-based optimization.

    Implementation of the Glacier algorithm by Wang et al. (2024) [glacier1]_.

    Glacier uses gradient-based optimization with guided constraints to generate
    counterfactual explanations. The key innovation is applying importance-based
    weights that allow free modification of less-important time series regions
    while preserving critical features.

    The optimization minimizes a composite loss:

        L = w * L_pred + (1-w) * L_proximity

    where:
    - L_pred: Prediction margin loss (distance to target class probability)
    - L_proximity: Weighted distance from original (importance-weighted)
    - w: pred_margin_weight parameter

    Parameters
    ----------
    model : object
        A classifier with a probability estimator (``predict_proba`` or a
        compatible interface). Must be differentiable or approximable.
    data : tuple (``X_ref``, ``y_ref``)
        Reference dataset used for computing feature importance and
        normalization statistics.
    pred_margin_weight : float, default 0.75
        Weight balancing prediction margin loss vs proximity loss.
        Higher values prioritize changing the prediction over staying close
        to the original. Range: [0, 1]. Values >= 0.75 recommended for
        non-neural-network classifiers where finite-difference gradients
        are weak relative to the proximity gradient.
    learning_rate : float, default 0.01
        Step size for Adam optimizer. Internally scaled by data standard
        deviation so the effective step adapts to input magnitude.
    max_iter : int, default 300
        Maximum number of optimization iterations.
    tau : float, default 0.5
        Decision threshold for target class probability. Optimization stops
        when P(target_class) >= tau.
    tolerance : float, default 1e-4
        Convergence tolerance for prediction margin loss.
    weight_type : {'uniform', 'local', 'unconstrained'}, default 'uniform'
        Type of importance weighting:

        - 'uniform': Equal weights across all timesteps
        - 'local': Segment-based LIME importance following the paper.
          Uses matrix-profile changepoint segmentation, STFT background
          perturbation, and Ridge regression surrogate to compute
          per-segment importance, producing binary timestep weights.
          Requires ``stumpy`` and ``scipy`` for full functionality
          (falls back to uniform segments / mean background otherwise).
        - 'unconstrained': No proximity penalty (pure prediction optimization)
    random_state : int or None, default 0
        PRNG seed for reproducible optimization.
    gradient_subsample : int or None, default 50
        Number of features to randomly sample for gradient computation each
        iteration. Uses stochastic gradient descent when set to a value less
        than the total number of features. Set to None to use all features
        (full gradient). Lower values speed up computation but may require
        more iterations to converge.
    temperature : float or None, default None
        Temperature scaling for soft probability computation. Higher values
        produce smoother gradients by preventing sigmoid saturation when
        decision function values are large. If None, auto-calibrates based
        on model decision function values (recommended for most use cases).
        Increase manually (e.g., 2.0-5.0) if counterfactuals are unchanged
        with ROCKET or other margin-based classifiers.
    n_segments : int, default 10
        Number of changepoints for segment-based local importance
        (``weight_type='local'``). Produces ``n_segments + 1`` segments.
        Ignored when ``weight_type`` is not ``'local'``.
    segment_window : int, default 10
        Window size for the matrix-profile segmentation algorithm.
        Ignored when ``weight_type`` is not ``'local'``.
    n_perturbations : int, default 100
        Number of binary perturbation samples for the LIME surrogate model
        used in segment-based local importance. Ignored when ``weight_type``
        is not ``'local'``.

    Attributes
    ----------
    predict_proba : callable
        Wrapped probability prediction function.
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    X_ref : np.ndarray
        Reference dataset features.
    y_ref : np.ndarray
        Reference dataset labels.
    _mean : np.ndarray
        Mean of reference data (for normalization).
    _std : np.ndarray
        Standard deviation of reference data (for normalization).

    References
    ----------
    .. [glacier1] Wang, Z., Samsten, I., Miliou, I., Mochaourab, R., & Papapetrou, P.
           (2024). Glacier: Guided Locally Constrained Counterfactual
           Explanations for Time Series Classification. Machine Learning, 113(3).
           https://github.com/zhendong3wang/learning-time-series-counterfactuals
    """

    model: Any
    data: tuple[np.ndarray, np.ndarray]
    pred_margin_weight: float = 0.75
    learning_rate: float = 0.01
    max_iter: int = 300
    tau: float = 0.5
    tolerance: float = 1e-4
    weight_type: WeightType = "uniform"
    random_state: int | None = 0
    gradient_subsample: int | None = 50
    temperature: float | None = None
    n_segments: int = 10
    segment_window: int = 10
    n_perturbations: int = 100

    # Internal state
    _mean: np.ndarray = field(default_factory=lambda: np.array([0.0]), init=False, repr=False)
    _std: np.ndarray = field(default_factory=lambda: np.array([1.0]), init=False, repr=False)

    def __post_init__(self):
        """Initialise probability wrapper, RNG, reference data, and label mapping.

        Validates all hyperparameters and computes normalisation statistics
        from the reference dataset. Warns if the model is unlikely to work
        well with gradient-based optimisation.
        """
        # Warn about classifiers that may not work well with gradient optimization
        if not supports_soft_probabilities(self.model):
            warnings.warn(
                f"Glacier is a gradient-based method that may not work well with "
                f"{type(self.model).__name__}. Tree-based classifiers return discrete "
                f"probabilities that don't respond well to gradient optimization. "
                f"Consider using COMTE or NativeGuide instead.",
                UserWarning,
                stacklevel=2,
            )

        self.predict_proba = soft_predict_proba_fn(self.model, temperature=self.temperature)
        self.rng = np.random.default_rng(self.random_state)
        self.X_ref = np.asarray(self.data[0])
        self.y_ref = np.asarray(self.data[1]).ravel()

        self._init_label_mapping(self.model, self.y_ref)

        # Compute normalization statistics from reference data
        self._mean = self.X_ref.mean(axis=0)
        self._std = self.X_ref.std(axis=0) + 1e-8  # Avoid division by zero

        # Validate parameters
        if not (0.0 <= self.pred_margin_weight <= 1.0):
            raise ValueError("pred_margin_weight must be in [0, 1]")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError("tau must be in (0, 1]")
        if self.weight_type not in ("uniform", "local", "unconstrained"):
            raise ValueError("weight_type must be one of {'uniform', 'local', 'unconstrained'}")
        if self.gradient_subsample is not None and self.gradient_subsample < 1:
            raise ValueError("gradient_subsample must be >= 1 or None")
        if self.n_segments < 1:
            raise ValueError("n_segments must be >= 1")
        if self.segment_window < 2:
            raise ValueError("segment_window must be >= 2")
        if self.n_perturbations < 10:
            raise ValueError("n_perturbations must be >= 10")

    def explain(
        self,
        x: np.ndarray,
        y_pred: int | None = None,
        *,
        class_of_interest: int | None = None,
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        """Generate a counterfactual explanation using gradient-based optimization.

        Parameters
        ----------
        x : np.ndarray
            Input time series of shape ``(T,)`` for univariate or ``(C, T)``
            for multivariate data.
        y_pred : int, optional
            Base predicted class for ``x``. If ``None``, computed via the model.
        class_of_interest : int, optional
            Target class for the counterfactual. If ``None``, uses the
            highest-probability alternative to ``y_pred``.

        Returns
        -------
        cf : np.ndarray
            Counterfactual time series with the same shape as ``x``.
        cf_label : int
            Predicted class label for the counterfactual.
        meta : dict
            Metadata dictionary containing:

            - ``method``: Algorithm identifier (``'glacier'``).
            - ``weight_type``: Constraint type used.
            - ``class_of_interest``: Target class.
            - ``pred_margin_weight``: Weight parameter used.
            - ``learning_rate``: Learning rate used.
            - ``n_iterations``: Number of iterations performed.
            - ``converged``: Whether optimization converged.
            - ``final_target_prob``: Final probability of target class.
            - ``final_loss``: Final composite loss value.
        """
        xb, added = ensure_batch_shape(x)
        x1 = strip_batch(xb, added)

        # Determine base prediction and target class.
        # All internal work uses probability column *indices* (0, 1, ...),
        # while y_pred and returned cf_label use actual class *labels*.
        base_probs = self.predict_proba(xb)[0]
        base_idx = int(np.argmax(base_probs)) if y_pred is None else self._label_to_idx(y_pred)

        if class_of_interest is None:
            probs_sorted = np.argsort(-base_probs)
            target_idx = int(next(c for c in probs_sorted if c != base_idx))
        else:
            target_idx = self._label_to_idx(class_of_interest)

        # Compute importance weights
        step_weights = self._compute_weights(x1, base_idx)

        # Run gradient-based optimization
        cf, n_iter, converged, final_prob, final_loss = self._optimize(
            x1, base_idx, target_idx, step_weights
        )

        # Get final prediction using soft probabilities (for consistency with optimization)
        cf_probs = self.predict_proba(cf[None, ...])[0]
        cf_idx = int(np.argmax(cf_probs))

        # Also get model's actual prediction for metadata
        model_cf_pred = self.model.predict(cf[None, ...])[0]

        # Return actual class label (not index)
        cf_label = self._idx_to_label(cf_idx)

        meta: dict[str, Any] = {
            "method": "glacier",
            "weight_type": self.weight_type,
            "class_of_interest": self._idx_to_label(target_idx),
            "pred_margin_weight": float(self.pred_margin_weight),
            "learning_rate": float(self.learning_rate),
            "n_iterations": n_iter,
            "converged": converged,
            "final_target_prob": float(final_prob),
            "final_loss": float(final_loss),
            "validity": cf_idx != base_idx,
            "model_cf_prediction": model_cf_pred,
        }

        return cf, cf_label, meta

    def _compute_weights(self, x: np.ndarray, base_label: int) -> np.ndarray:
        """Compute importance weights for proximity loss.

        Parameters
        ----------
        x : np.ndarray
            Original time series of shape ``(T,)`` or ``(C, T)``.
        base_label : int
            Original predicted class label.

        Returns
        -------
        np.ndarray
            Importance weights with the same shape as ``x``.
        """
        if self.weight_type == "unconstrained":
            # No proximity penalty - all weights zero
            return np.zeros_like(x)

        if self.weight_type == "uniform":
            # Equal weights across all timesteps
            return np.ones_like(x)

        # Local importance: segment-based LIME following the Glacier paper.
        # Uses matrix-profile segmentation + STFT background perturbation
        # + Ridge surrogate to compute per-segment importance, then maps
        # to binary timestep weights.
        return self._compute_local_importance(x, base_label)

    # ------------------------------------------------------------------
    # Segment-based local importance (following the Glacier paper)
    # ------------------------------------------------------------------

    def _compute_local_importance(self, x: np.ndarray, base_label: int) -> np.ndarray:
        """Compute local feature importance via segment-based LIME.

        Follows the Glacier paper [glacier1]_:
        1. Segment the time series using matrix-profile changepoint detection.
        2. Perturb segments by replacing them with an STFT-derived background.
        3. Fit a weighted Ridge regression as a LIME surrogate model.
        4. Threshold segment importance and map to binary timestep weights.

        When ``stumpy`` or ``scipy`` are unavailable, falls back to uniform
        segmentation and mean-value perturbation respectively.

        Parameters
        ----------
        x : np.ndarray
            Original time series of shape ``(T,)`` or ``(C, T)``.
        base_label : int
            Probability column index of the original predicted class.

        Returns
        -------
        np.ndarray
            Binary importance weights with the same shape as ``x``.
            0 = segment may be freely modified, 1 = segment is protected.
        """
        is_multivariate = x.ndim == 2
        # Work on the first channel for segmentation; weights are broadcast
        x_1d = x[0] if is_multivariate else x
        T = len(x_1d)

        # 1. Segment the time series
        seg_bounds = self._segment_time_series(x_1d)
        n_segs = len(seg_bounds) - 1

        # 2. Compute background signal for perturbation
        background = self._compute_background(x_1d)

        # 3. Generate binary perturbation samples and their raw versions
        interpretable, raw_samples = self._generate_perturbation_samples(
            x, x_1d, background, seg_bounds, n_segs, is_multivariate
        )

        # 4. Get predictions for all perturbed samples
        probs = self.predict_proba(raw_samples)[:, base_label]

        # 5. Compute Euclidean distance weights for locality
        # Distance between original interpretable repr (all 1s) and each sample
        all_on = np.ones(n_segs)
        dists = np.linalg.norm(interpretable - all_on, axis=1)
        dists_z = (dists - dists.mean()) / dists.std() if dists.std() > 0 else np.zeros_like(dists)
        sample_weights = np.exp(-np.abs(dists_z))

        # 6. Fit Ridge regression surrogate
        ridge = Ridge(alpha=1.0)
        ridge.fit(interpretable, probs, sample_weight=sample_weights)
        seg_importance = ridge.coef_  # one coefficient per segment

        # 7. Threshold and expand to timestep-level binary weights
        threshold = np.percentile(seg_importance, 25)
        # Segments with LOW importance for the base class -> safe to modify
        # (weight 0 means no proximity penalty -> optimizer can change freely)
        mask_indices = np.where(seg_importance <= threshold)[0]

        weights_1d = np.ones(T)
        for idx in mask_indices:
            start = seg_bounds[idx]
            end = seg_bounds[idx + 1]
            weights_1d[start:end] = 0.0

        # Broadcast to full shape
        if is_multivariate:
            return np.broadcast_to(weights_1d[None, :], x.shape).copy()
        return weights_1d

    def _segment_time_series(self, x_1d: np.ndarray) -> list[int]:
        """Segment a univariate time series via matrix-profile changepoints.

        Uses NNSegment-style changepoint detection: compute the matrix profile,
        find discontinuities in nearest-neighbour pointers, rank by a
        variance-based score, and greedily select non-overlapping changepoints.

        Falls back to uniform segmentation when ``stumpy`` is not installed or
        the series is too short for the requested window size.

        Parameters
        ----------
        x_1d : np.ndarray
            Univariate time series of shape ``(T,)``.

        Returns
        -------
        list[int]
            Sorted segment boundary indices including ``0`` and ``T``.
        """
        T = len(x_1d)
        n_cp = self.n_segments
        window = min(self.segment_window, T // 2)

        if not STUMPY_AVAILABLE or window < 3 or 2 * window > T:
            if not STUMPY_AVAILABLE:
                warnings.warn(
                    "stumpy is not installed. Glacier local importance is using "
                    "uniform segmentation instead of matrix-profile changepoints. "
                    "Install stumpy for proper segmentation: pip install stumpy",
                    UserWarning,
                    stacklevel=3,
                )
            return self._uniform_segments(T, n_cp)

        # Compute the matrix profile
        mp = stumpy.stump(x_1d.astype(np.float64), m=window)
        nn_indices = mp[:, 1].astype(int)

        # Find candidate changepoints: discontinuities in NN pointer
        candidates = []
        for i in range(len(nn_indices) - 1):
            if nn_indices[i + 1] != nn_indices[i] + 1:
                candidates.append(i + 1)  # boundary is at i+1

        if not candidates:
            return self._uniform_segments(T, n_cp)

        # Score candidates by mean/variance shift
        tol = max(window // 2, 1)
        scored: list[tuple[float, int]] = []
        for idx in candidates:
            left_start = max(0, idx - tol)
            right_end = min(T, idx + tol)
            if idx - left_start < 2 or right_end - idx < 2:
                continue
            left = x_1d[left_start:idx]
            right = x_1d[idx:right_end]
            mean_change = abs(float(left.mean() - right.mean()))
            std_change = abs(float(left.std() - right.std()))
            std_mean = float((left.std() + right.std()) / 2)
            score = mean_change * std_change / (std_mean + 1e-10)
            scored.append((score, idx))

        if not scored:
            return self._uniform_segments(T, n_cp)

        # Greedy non-overlapping selection
        scored.sort(reverse=True)
        selected: list[int] = []
        for _, idx in scored:
            if len(selected) >= n_cp:
                break
            if all(abs(idx - s) >= tol for s in selected):
                selected.append(idx)

        if not selected:
            return self._uniform_segments(T, n_cp)

        selected.sort()
        return [0, *selected, T]

    @staticmethod
    def _uniform_segments(T: int, n_cp: int) -> list[int]:
        """Create uniform segment boundaries as fallback.

        Parameters
        ----------
        T : int
            Length of the time series.
        n_cp : int
            Desired number of changepoints.

        Returns
        -------
        list[int]
            Sorted segment boundary indices including ``0`` and ``T``.
        """
        n_segs = min(n_cp + 1, T)
        bounds = np.linspace(0, T, n_segs + 1, dtype=int).tolist()
        # Deduplicate (can happen for very short series)
        return sorted(set(bounds))

    def _compute_background(self, x_1d: np.ndarray) -> np.ndarray:
        """Compute a background signal via STFT (Realistic Background Perturbation).

        Isolates the most stable frequency component (highest mean/std ratio
        in the STFT) and reconstructs a signal from only that component.
        This background signal serves as a realistic replacement when
        "turning off" a segment.

        Falls back to the global mean when ``scipy`` is not installed.

        Parameters
        ----------
        x_1d : np.ndarray
            Univariate time series of shape ``(T,)``.

        Returns
        -------
        np.ndarray
            Background signal of the same length as ``x_1d``.
        """
        if not SCIPY_AVAILABLE:
            warnings.warn(
                "scipy is not installed. Glacier local importance is using the "
                "global mean as background instead of STFT-based background "
                "identification. Install scipy for proper background perturbation: "
                "pip install scipy",
                UserWarning,
                stacklevel=3,
            )
            return np.full_like(x_1d, x_1d.mean())

        T = len(x_1d)
        nperseg = min(40, T)
        _f, _t, Zxx = sp_signal.stft(x_1d.astype(np.float64), fs=1.0, nperseg=nperseg)

        # Find the most stable frequency (highest mean/std ratio of magnitude)
        magnitudes = np.abs(Zxx)
        with np.errstate(divide="ignore", invalid="ignore"):
            stds = magnitudes.std(axis=1)
            means = magnitudes.mean(axis=1)
            stability = np.where(stds > 1e-12, means / stds, 0.0)

        best_freq = int(np.argmax(stability))

        # Reconstruct using only the most stable frequency
        mask = np.zeros_like(Zxx)
        mask[best_freq, :] = 1.0
        _, bg_raw = sp_signal.istft(Zxx * mask, fs=1.0, nperseg=nperseg)
        background: np.ndarray = np.asarray(bg_raw, dtype=np.float64)

        # Match length (STFT/ISTFT may produce slightly different length)
        if len(background) >= T:
            return background[:T]
        return np.pad(background, (0, T - len(background)), mode="edge")

    def _generate_perturbation_samples(
        self,
        x: np.ndarray,
        x_1d: np.ndarray,
        background: np.ndarray,
        seg_bounds: list[int],
        n_segs: int,
        is_multivariate: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate binary perturbation samples and corresponding raw signals.

        Each sample is a binary vector (one bit per segment) indicating
        whether to keep (1) or replace (0) each segment. When a segment is
        "turned off", its timesteps are replaced with the background signal.

        Parameters
        ----------
        x : np.ndarray
            Original time series of shape ``(T,)`` or ``(C, T)``.
        x_1d : np.ndarray
            Univariate view of ``x`` (first channel or ``x`` itself).
        background : np.ndarray
            Background signal of shape ``(T,)``.
        seg_bounds : list[int]
            Segment boundary indices.
        n_segs : int
            Number of segments.
        is_multivariate : bool
            Whether the original input is multivariate.

        Returns
        -------
        interpretable : np.ndarray
            Binary matrix of shape ``(n_perturbations, n_segs)``.
        raw_samples : np.ndarray
            Perturbed time series of shape ``(n_perturbations, *x.shape)``.
        """
        interpretable = self.rng.binomial(1, 0.5, size=(self.n_perturbations, n_segs))
        raw_samples = np.tile(
            x, (self.n_perturbations, 1) if x.ndim == 1 else (self.n_perturbations, 1, 1)
        )

        for i in range(self.n_perturbations):
            for s in range(n_segs):
                if interpretable[i, s] == 0:
                    start = seg_bounds[s]
                    end = seg_bounds[s + 1]
                    if is_multivariate:
                        # Replace all channels with background
                        raw_samples[i, :, start:end] = background[start:end]
                    else:
                        raw_samples[i, start:end] = background[start:end]

        return interpretable, raw_samples

    def _optimize(
        self,
        x_original: np.ndarray,
        base_label: int,
        target_class: int,
        step_weights: np.ndarray,
    ) -> tuple[np.ndarray, int, bool, float, float]:
        """Run gradient-based optimization to find counterfactual.

        Uses Adam optimizer with finite-difference gradients, following the
        original Glacier implementation which uses ``tf.GradientTape`` + Adam.

        Parameters
        ----------
        x_original : np.ndarray
            Original time series of shape ``(T,)`` or ``(C, T)``.
        base_label : int
            Original predicted class label.
        target_class : int
            Target class for counterfactual.
        step_weights : np.ndarray
            Importance weights for proximity loss.

        Returns
        -------
        cf : np.ndarray
            Optimized counterfactual.
        n_iterations : int
            Number of iterations performed.
        converged : bool
            Whether optimization converged.
        final_target_prob : float
            Final probability of target class.
        final_loss : float
            Final composite loss value.
        """
        # Initialize counterfactual as copy of original
        cf = x_original.copy().astype(np.float64)

        w = self.pred_margin_weight
        converged = False
        n_iterations = 0
        final_prob = 0.0
        final_loss = float("inf")

        # Scale learning rate to data magnitude. Adam normalizes gradients so
        # effective step ≈ lr regardless of gradient magnitude. For finite
        # differences through non-differentiable transforms (ROCKET), the step
        # needs to be proportional to data scale to cross the decision boundary.
        data_scale = float(self._std.mean())
        effective_lr = self.learning_rate * data_scale

        adam = AdamState.zeros_like(cf)

        for iteration in range(self.max_iter):
            n_iterations = iteration + 1

            # Compute current prediction
            probs = self.predict_proba(cf[None, ...])[0]
            target_prob = probs[target_class]
            pred_label = int(np.argmax(probs))

            # Prediction margin loss: MSE(tau, target_prob) following original
            pred_margin_loss = (self.tau - target_prob) ** 2

            # Proximity loss: weighted MAE from original (mean, not sum)
            diff = cf - x_original
            if self.weight_type == "unconstrained":
                proximity_loss = 0.0
            else:
                proximity_loss = float(np.mean(step_weights * np.abs(diff)))

            # Composite loss
            total_loss = w * pred_margin_loss + (1 - w) * proximity_loss

            final_prob = target_prob
            final_loss = total_loss

            # Check convergence
            if pred_margin_loss < self.tolerance and target_prob >= self.tau:
                converged = True
                break

            if pred_label == target_class and target_prob >= self.tau:
                converged = True
                break

            # Compute gradient via finite differences
            gradient = self._compute_gradient(cf, x_original, target_class, step_weights, w)

            # Adam update
            cf = cf - adam.step(gradient, effective_lr)

        return cf, n_iterations, converged, final_prob, final_loss

    def _compute_gradient(
        self,
        cf: np.ndarray,
        x_original: np.ndarray,
        target_class: int,
        step_weights: np.ndarray,
        w: float,
    ) -> np.ndarray:
        """Compute gradient of composite loss.

        Uses finite differences for the prediction margin loss (which requires
        model evaluation) and an analytical gradient for the proximity loss
        (which has a simple closed form). This separation avoids the proximity
        gradient dominating due to scale differences.

        Parameters
        ----------
        cf : np.ndarray
            Current counterfactual estimate.
        x_original : np.ndarray
            Original time series.
        target_class : int
            Target class for counterfactual.
        step_weights : np.ndarray
            Importance weights for proximity loss.
        w : float
            Prediction margin weight.

        Returns
        -------
        np.ndarray
            Gradient of composite loss with respect to cf.
        """
        # Scale epsilon to data magnitude so gradients are meaningful
        # through non-differentiable transforms (e.g. ROCKET random kernels).
        epsilon = max(float(self._std.mean()) * 0.01, 1e-4)
        flat_cf = cf.flatten()
        n_features = len(flat_cf)

        # Subsample features for stochastic gradient estimation.
        # For classifiers with expensive transforms (ROCKET, RDST), skip the
        # n_features // 2 floor so the user's gradient_subsample is respected,
        # reducing the number of costly transform calls per iteration.
        if self.gradient_subsample is not None and self.gradient_subsample < n_features:
            if has_expensive_transform(self.model):
                n_sample = self.gradient_subsample
            else:
                n_sample = max(self.gradient_subsample, n_features // 2)
            n_sample = min(n_sample, n_features)
            sampled_idx = self.rng.choice(n_features, size=n_sample, replace=False)
        else:
            n_sample = n_features
            sampled_idx = np.arange(n_features)

        # --- Prediction margin gradient via finite differences ---
        perturbations = np.tile(flat_cf, (2 * n_sample, 1))

        for i, feat_idx in enumerate(sampled_idx):
            perturbations[i, feat_idx] += epsilon  # cf_plus
            perturbations[n_sample + i, feat_idx] -= epsilon  # cf_minus

        perturbations_reshaped = perturbations.reshape(2 * n_sample, *cf.shape)
        probs_batch = self.predict_proba(perturbations_reshaped)
        target_probs = probs_batch[:, target_class]

        # d/dcf [ (tau - p(cf))^2 ] via finite differences
        pred_losses_plus = (self.tau - target_probs[:n_sample]) ** 2
        pred_losses_minus = (self.tau - target_probs[n_sample:]) ** 2
        pred_grad_sampled = (pred_losses_plus - pred_losses_minus) / (2 * epsilon)

        pred_gradient = np.zeros(n_features)
        pred_gradient[sampled_idx] = pred_grad_sampled

        # --- Proximity gradient (analytical) ---
        # L_prox = mean(weights * |cf - x_orig|)
        # d/dcf L_prox = weights * sign(cf - x_orig) / n_features
        if self.weight_type == "unconstrained":
            prox_gradient = np.zeros(n_features)
        else:
            flat_weights = step_weights.flatten()
            flat_original = x_original.flatten()
            diff = flat_cf - flat_original
            prox_gradient = flat_weights * np.sign(diff) / n_features

        # Normalize each component so that `w` controls the balance
        # regardless of raw gradient magnitudes. Without this, the
        # proximity gradient (order ~0.04) dominates the prediction
        # gradient (order ~0.0001) through non-differentiable classifiers.
        # When a component norm is negligible (e.g. prediction gradient is
        # zero for tree-based classifiers), zero it out instead of
        # amplifying floating-point noise to unit norm.
        min_norm = 1e-10
        pred_norm = float(np.linalg.norm(pred_gradient))
        prox_norm = float(np.linalg.norm(prox_gradient))

        pred_component = (
            w * pred_gradient / pred_norm if pred_norm > min_norm else np.zeros_like(pred_gradient)
        )
        prox_component = (
            (1 - w) * prox_gradient / prox_norm
            if prox_norm > min_norm
            else np.zeros_like(prox_gradient)
        )
        full_gradient = pred_component + prox_component

        return full_gradient.reshape(cf.shape)
