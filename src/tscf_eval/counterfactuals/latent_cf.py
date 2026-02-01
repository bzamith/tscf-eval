"""LatentCF++ counterfactual explainer implementation.

This module provides the ``LatentCF`` class, an implementation of the LatentCF++
algorithm for generating counterfactual explanations for time series classification.

The algorithm was originally developed by Zhendong Wang, Isak Samsten,
Rami Mochaourab, and Panagiotis Papapetrou at Stockholm University, with
multivariate extensions by Stella Gerantoni.

Original implementations:
- LatentCF++: https://github.com/zhendong3wang/learning-time-series-counterfactuals
- Multivariate: https://github.com/stellagerantoni/LatentCfMultivariate

Classes
-------
LatentCF
    LatentCF++ counterfactual generator using gradient-based optimization
    in latent space representations.

Algorithm Overview
------------------
LatentCF++ generates counterfactuals through latent space optimization:

1. Optionally encode the input time series into a latent representation using
   an autoencoder (user-provided or None for direct optimization).
2. Compute importance weights for each timestep using:
   - 'uniform': Equal weights across all timesteps
   - 'local': Per-sample importance computed via perturbation-based sensitivity
   - 'global': Dataset-level importance computed across reference samples
3. Optimize a composite loss function:

   - **Prediction margin loss**: Drives the sample toward target class probability
   - **Weighted proximity loss**: Penalizes deviations, weighted by importance

4. Iterate until the target probability is reached or max iterations exhausted.
5. If using an autoencoder, decode the optimized latent representation.

Examples
--------
>>> from tscf_eval.counterfactuals import LatentCF
>>> import numpy as np
>>>
>>> # Assume clf is a trained classifier with predict_proba
>>> latent_cf = LatentCF(
...     model=clf,
...     data=(X_train, y_train),
...     pred_margin_weight=1.0,
...     learning_rate=0.0001,
...     max_iter=100,
... )
>>>
>>> # Generate counterfactual for a test instance
>>> cf, cf_label, meta = latent_cf.explain(x_test)
>>> print(f"Converged: {meta['converged']}")
>>> print(f"Iterations: {meta['n_iterations']}")

References
----------
.. [latentcf1] Wang, Z., Samsten, I., Mochaourab, R., & Papapetrou, P. (2021).
       Learning Time Series Counterfactuals via Latent Space Representations.
       In International Conference on Discovery Science (DS 2021),
       Lecture Notes in Computer Science, vol 12986, pp. 369-384. Springer.
       DOI: 10.1007/978-3-030-88942-5_29

Notes
-----
This implementation provides a NumPy-based version of LatentCF++ that works
directly in the original time series space for compatibility with any
scikit-learn compatible classifier. For TensorFlow/Keras-based models with
autoencoders, the original implementation is recommended.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
import warnings

import numpy as np

from .base import Counterfactual
from .utils import (
    ensure_batch_shape,
    soft_predict_proba_fn,
    strip_batch,
    supports_soft_probabilities,
)

WeightStrategy = Literal["uniform", "local", "global"]


@dataclass
class LatentCF(Counterfactual):
    """LatentCF++ counterfactual generator using gradient-based optimization.

    Implementation of the LatentCF++ algorithm by Wang et al. (2021) [latentcf1]_.

    LatentCF++ generates counterfactuals by optimizing in the latent space
    (or directly in input space when no autoencoder is provided). The algorithm
    balances prediction margin loss (driving toward target class) with weighted
    proximity loss (staying close to original, prioritizing less important regions).

    The optimization minimizes a composite loss:

        L = w * L_pred + (1-w) * L_proximity

    where:
    - L_pred: Mean squared error between desired probability (1.0) and current
    - L_proximity: Weighted mean absolute error from original
    - w: pred_margin_weight parameter

    Parameters
    ----------
    model : object
        A classifier with a probability estimator (``predict_proba`` or a
        compatible interface).
    data : tuple (X_ref, y_ref)
        Reference dataset used for computing feature importance (for 'global'
        weight strategy) and normalization statistics.
    probability : float, default 0.5
        Target probability threshold. Optimization aims for P(target) >= probability.
    tolerance : float, default 1e-6
        Convergence tolerance. Optimization stops when prediction margin loss
        is below tolerance AND target probability is reached.
    max_iter : int, default 300
        Maximum number of optimization iterations.
    learning_rate : float, default 0.01
        Step size for Adam optimizer. Internally scaled by data standard
        deviation so the effective step adapts to input magnitude.
    pred_margin_weight : float, default 0.75
        Weight balancing prediction margin loss vs proximity loss.
        Range: [0, 1]. Higher values prioritize changing the prediction.
        Values >= 0.75 recommended for non-neural-network classifiers.
    step_weights : {'uniform', 'local', 'global'}, default 'uniform'
        Strategy for computing importance weights:

        - 'uniform': Equal weights across all timesteps
        - 'local': Per-sample importance via perturbation-based sensitivity
        - 'global': Dataset-level importance computed across reference samples
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
    _global_weights : np.ndarray or None
        Precomputed global weights (cached after first use).

    References
    ----------
    .. [latentcf1] Wang, Z., Samsten, I., Mochaourab, R., & Papapetrou, P. (2021).
           Learning Time Series Counterfactuals via Latent Space Representations.
           In International Conference on Discovery Science (DS 2021).
           https://github.com/zhendong3wang/learning-time-series-counterfactuals
    """

    model: Any
    data: tuple[np.ndarray, np.ndarray]
    probability: float = 0.5
    tolerance: float = 1e-6
    max_iter: int = 300
    learning_rate: float = 0.01
    pred_margin_weight: float = 0.75
    step_weights: WeightStrategy = "uniform"
    random_state: int | None = 0
    gradient_subsample: int | None = 50
    temperature: float | None = None

    # Internal state
    _global_weights: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Warn about classifiers that may not work well with gradient optimization
        if not supports_soft_probabilities(self.model):
            warnings.warn(
                f"LatentCF is a gradient-based method that may not work well with "
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

        # Validate parameters
        if not (0.0 < self.probability <= 1.0):
            raise ValueError("probability must be in (0, 1]")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be > 0")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not (0.0 <= self.pred_margin_weight <= 1.0):
            raise ValueError("pred_margin_weight must be in [0, 1]")
        if self.step_weights not in ("uniform", "local", "global"):
            raise ValueError("step_weights must be one of {'uniform', 'local', 'global'}")
        if self.gradient_subsample is not None and self.gradient_subsample < 1:
            raise ValueError("gradient_subsample must be >= 1 or None")

    def explain(
        self,
        x: np.ndarray,
        y_pred: int | None = None,
        *,
        class_of_interest: int | None = None,
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        """Generate a counterfactual explanation using LatentCF++ optimization.

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

            - ``method``: Algorithm identifier (``'latent_cf'``).
            - ``step_weights``: Weight strategy used.
            - ``class_of_interest``: Target class.
            - ``pred_margin_weight``: Weight parameter used.
            - ``learning_rate``: Learning rate used.
            - ``n_iterations``: Number of iterations performed.
            - ``converged``: Whether optimization converged.
            - ``final_target_prob``: Final probability of target class.
            - ``final_loss``: Final composite loss value.
            - ``validity``: Whether counterfactual changed prediction.
        """
        xb, added = ensure_batch_shape(x)
        x1 = strip_batch(xb, added)

        # Determine base prediction and target class
        base_probs = self.predict_proba(xb)[0]
        base_label = int(np.argmax(base_probs)) if y_pred is None else int(y_pred)

        if class_of_interest is None:
            probs_sorted = np.argsort(-base_probs)
            class_of_interest = int(next(c for c in probs_sorted if c != base_label))

        # Compute importance weights
        weights = self._compute_weights(x1, base_label)

        # Run gradient-based optimization
        cf, n_iter, converged, final_prob, final_loss = self._optimize(
            x1, base_label, class_of_interest, weights
        )

        # Get final prediction using soft probabilities (for consistency with optimization)
        cf_probs = self.predict_proba(cf[None, ...])[0]
        cf_label_idx = int(np.argmax(cf_probs))

        # Also get model's actual prediction for metadata
        model_cf_pred = self.model.predict(cf[None, ...])[0]

        # Return label as the probability index (consistent with base_label)
        # The caller can map to actual class if needed using model.classes_
        cf_label = cf_label_idx

        meta: dict[str, Any] = {
            "method": "latent_cf",
            "step_weights": self.step_weights,
            "class_of_interest": class_of_interest,
            "pred_margin_weight": float(self.pred_margin_weight),
            "learning_rate": float(self.learning_rate),
            "probability_threshold": float(self.probability),
            "n_iterations": n_iter,
            "converged": converged,
            "final_target_prob": float(final_prob),
            "final_loss": float(final_loss),
            "validity": cf_label_idx != base_label,
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
            Lower weight = more freedom to modify.
        """
        if self.step_weights == "uniform":
            return np.ones_like(x)

        if self.step_weights == "local":
            return self._compute_local_importance(x, base_label)

        # Global weights
        return self._compute_global_importance(x)

    def _compute_local_importance(self, x: np.ndarray, base_label: int) -> np.ndarray:
        """Compute local feature importance via batched perturbation.

        Uses batch prediction to compute all perturbations at once,
        measuring how much the prediction changes when each timestep is perturbed.

        Parameters
        ----------
        x : np.ndarray
            Original time series of shape ``(T,)`` or ``(C, T)``.
        base_label : int
            Original predicted class label.

        Returns
        -------
        np.ndarray
            Importance weights (higher = more important = less modifiable).
            Low-importance regions (below 25th percentile) are set to 0.
        """
        # Compute standard deviation for perturbation scaling
        x_std = np.std(x) if np.std(x) > 0 else 1.0
        epsilon = 0.01 * x_std

        flat_x = x.flatten()
        n_features = len(flat_x)

        # Sample subset for efficiency on long series
        n_samples = min(n_features, 100)
        indices = self.rng.choice(n_features, size=n_samples, replace=False)

        # Create all perturbed versions at once: 2 * n_samples samples
        perturbations = np.tile(flat_x, (2 * n_samples, 1))

        for j, i in enumerate(indices):
            perturbations[j, i] += epsilon  # x_plus
            perturbations[n_samples + j, i] -= epsilon  # x_minus

        # Reshape to match expected input shape
        perturbations_reshaped = perturbations.reshape(2 * n_samples, *x.shape)

        # Single batch prediction for all perturbations
        probs_batch = self.predict_proba(perturbations_reshaped)
        probs_base = probs_batch[:, base_label]

        # Extract probabilities for plus and minus perturbations
        probs_plus = probs_base[:n_samples]
        probs_minus = probs_base[n_samples:]

        # Gradient magnitude as importance for sampled indices
        flat_importance = np.zeros(n_features)
        sampled_importance = np.abs(probs_plus - probs_minus) / (2 * epsilon)

        for j, i in enumerate(indices):
            flat_importance[i] = sampled_importance[j]

        # Interpolate non-sampled indices
        if n_samples < n_features:
            sampled_mask = np.zeros(n_features, dtype=bool)
            sampled_mask[indices] = True
            for i in range(n_features):
                if not sampled_mask[i]:
                    # Find nearest sampled neighbors
                    left = i - 1
                    right = i + 1
                    while left >= 0 and not sampled_mask[left]:
                        left -= 1
                    while right < n_features and not sampled_mask[right]:
                        right += 1
                    if left >= 0 and right < n_features:
                        flat_importance[i] = (flat_importance[left] + flat_importance[right]) / 2
                    elif left >= 0:
                        flat_importance[i] = flat_importance[left]
                    elif right < n_features:
                        flat_importance[i] = flat_importance[right]

        importance = flat_importance.reshape(x.shape)

        # Normalize to [0, 1]
        if importance.max() > importance.min():
            importance = (importance - importance.min()) / (importance.max() - importance.min())

        # Mask low-importance regions (below 25th percentile)
        threshold = np.percentile(importance, 25)
        weights = np.where(importance <= threshold, 0.0, importance)

        return weights

    def _compute_global_importance(self, x: np.ndarray) -> np.ndarray:
        """Compute global feature importance from reference dataset.

        Computes importance across all samples in the reference set using
        perturbation-based sensitivity analysis, then thresholds at the
        75th percentile.

        Parameters
        ----------
        x : np.ndarray
            Original time series (used for shape reference).

        Returns
        -------
        np.ndarray
            Importance weights matching shape of ``x``.
        """
        # Use cached global weights if available and shape matches
        if self._global_weights is not None and self._global_weights.shape == x.shape:
            return self._global_weights.copy()

        # Compute global importance from reference set
        n_ref = min(len(self.X_ref), 20)  # Sample for efficiency
        ref_indices = self.rng.choice(len(self.X_ref), size=n_ref, replace=False)

        all_importance = []
        for idx in ref_indices:
            x_ref = self.X_ref[idx]

            # Reshape if needed to match expected shape
            if x_ref.shape != x.shape:
                if x_ref.ndim == 1 and x.ndim == 1:
                    # Both univariate but different lengths - skip
                    continue
                elif x_ref.ndim == 2 and x.ndim == 2:
                    if x_ref.shape != x.shape:
                        continue
                else:
                    continue

            # Get the probability index for this reference sample
            # (use soft probabilities to get the predicted class index)
            ref_probs = self.predict_proba(x_ref[None, ...])[0]
            ref_label_idx = int(np.argmax(ref_probs))

            importance = self._compute_local_importance(x_ref, ref_label_idx)
            all_importance.append(importance)

        if not all_importance:
            return np.ones_like(x)

        # Average importance across reference samples
        global_importance = np.mean(all_importance, axis=0)

        # Threshold at 75th percentile (high importance = protected)
        threshold = np.percentile(global_importance, 75)
        weights = np.where(global_importance >= threshold, global_importance, 0.0)

        # Cache for reuse
        self._global_weights = weights.copy()

        return weights

    def _optimize(
        self,
        x_original: np.ndarray,
        base_label: int,
        target_class: int,
        step_weights: np.ndarray,
    ) -> tuple[np.ndarray, int, bool, float, float]:
        """Run gradient-based optimization to find counterfactual.

        Uses Adam optimizer with finite-difference gradients, following the
        original LatentCF++ implementation which uses ``tf.GradientTape`` + Adam.

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
        data_scale = float(np.std(self.X_ref))
        effective_lr = self.learning_rate * data_scale

        # Adam optimizer state
        m = np.zeros_like(cf)  # First moment
        v = np.zeros_like(cf)  # Second moment
        beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

        for iteration in range(self.max_iter):
            n_iterations = iteration + 1

            # Compute current prediction
            probs = self.predict_proba(cf[None, ...])[0]
            target_prob = probs[target_class]

            # Prediction margin loss: MSE between desired probability and current
            pred_margin_loss = (self.probability - target_prob) ** 2

            # Proximity loss: weighted MAE from original
            diff = cf - x_original
            proximity_loss = float(np.mean(step_weights * np.abs(diff)))

            # Composite loss
            total_loss = w * pred_margin_loss + (1 - w) * proximity_loss

            final_prob = target_prob
            final_loss = total_loss

            # Check convergence: loss below tolerance AND target prob reached
            if pred_margin_loss < self.tolerance and target_prob >= self.probability:
                converged = True
                break

            # Compute gradient via finite differences
            gradient = self._compute_gradient(cf, x_original, target_class, step_weights, w)

            # Adam update
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient**2
            m_hat = m / (1 - beta1**n_iterations)
            v_hat = v / (1 - beta2**n_iterations)
            cf = cf - effective_lr * m_hat / (np.sqrt(v_hat) + adam_eps)

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
        x_std = float(np.std(self.X_ref))
        epsilon = max(x_std * 0.01, 1e-4)
        flat_cf = cf.flatten()
        n_features = len(flat_cf)

        # Use all features for short series; subsample only for very long ones
        if self.gradient_subsample is not None and self.gradient_subsample < n_features:
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

        # d/dcf [ (probability - p(cf))^2 ] via finite differences
        pred_losses_plus = (self.probability - target_probs[:n_sample]) ** 2
        pred_losses_minus = (self.probability - target_probs[n_sample:]) ** 2
        pred_grad_sampled = (pred_losses_plus - pred_losses_minus) / (2 * epsilon)

        pred_gradient = np.zeros(n_features)
        pred_gradient[sampled_idx] = pred_grad_sampled

        # --- Proximity gradient (analytical) ---
        # L_prox = mean(weights * |cf - x_orig|)
        # d/dcf L_prox = weights * sign(cf - x_orig) / n_features
        flat_weights = step_weights.flatten()
        flat_original = x_original.flatten()
        diff = flat_cf - flat_original
        prox_gradient = flat_weights * np.sign(diff) / n_features

        # Normalize each component so that `w` controls the balance
        # regardless of raw gradient magnitudes. Without this, the
        # proximity gradient (order ~0.04) dominates the prediction
        # gradient (order ~0.0001) through non-differentiable classifiers.
        pred_norm = float(np.linalg.norm(pred_gradient)) + 1e-30
        prox_norm = float(np.linalg.norm(prox_gradient)) + 1e-30
        full_gradient = w * pred_gradient / pred_norm + (1 - w) * prox_gradient / prox_norm

        result: np.ndarray = full_gradient.reshape(cf.shape)
        return result
