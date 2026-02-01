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
2. Compute importance weights for each timestep using feature attribution
   methods (local importance) or use uniform/unconstrained weights.
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

from .base import Counterfactual
from .utils import (
    ensure_batch_shape,
    soft_predict_proba_fn,
    strip_batch,
    supports_soft_probabilities,
)

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
    data : tuple (X_ref, y_ref)
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
        - 'local': Per-sample importance computed via gradient magnitude
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

    # Internal state
    _mean: np.ndarray = field(default_factory=lambda: np.array([0.0]), init=False, repr=False)
    _std: np.ndarray = field(default_factory=lambda: np.array([1.0]), init=False, repr=False)

    def __post_init__(self):
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

        # Determine base prediction and target class
        base_probs = self.predict_proba(xb)[0]
        base_label = int(np.argmax(base_probs)) if y_pred is None else int(y_pred)

        if class_of_interest is None:
            probs_sorted = np.argsort(-base_probs)
            class_of_interest = int(next(c for c in probs_sorted if c != base_label))

        # Compute importance weights
        step_weights = self._compute_weights(x1, base_label)

        # Run gradient-based optimization
        cf, n_iter, converged, final_prob, final_loss = self._optimize(
            x1, base_label, class_of_interest, step_weights
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
            "method": "glacier",
            "weight_type": self.weight_type,
            "class_of_interest": class_of_interest,
            "pred_margin_weight": float(self.pred_margin_weight),
            "learning_rate": float(self.learning_rate),
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
        """
        if self.weight_type == "unconstrained":
            # No proximity penalty - all weights zero
            return np.zeros_like(x)

        if self.weight_type == "uniform":
            # Equal weights across all timesteps
            return np.ones_like(x)

        # Local importance: use gradient magnitude as proxy
        # This is a simplified version of LIME-based importance
        return self._compute_local_importance(x, base_label)

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
        """
        epsilon = 0.01 * (self._std.mean() if self._std.ndim > 0 else self._std)

        flat_x = x.flatten()
        n_features = len(flat_x)

        # Create all perturbed versions at once: 2 * n_features samples
        perturbations = np.tile(flat_x, (2 * n_features, 1))

        for i in range(n_features):
            perturbations[i, i] += epsilon  # x_plus
            perturbations[n_features + i, i] -= epsilon  # x_minus

        # Reshape to match expected input shape
        perturbations_reshaped = perturbations.reshape(2 * n_features, *x.shape)

        # Single batch prediction for all perturbations
        probs_batch = self.predict_proba(perturbations_reshaped)
        probs_base = probs_batch[:, base_label]

        # Extract probabilities for plus and minus perturbations
        probs_plus = probs_base[:n_features]
        probs_minus = probs_base[n_features:]

        # Gradient magnitude as importance
        flat_importance = np.abs(probs_plus - probs_minus) / (2 * epsilon)
        importance = flat_importance.reshape(x.shape)

        # Normalize to [0, 1] and invert: low importance = low weight = more modifiable
        if importance.max() > importance.min():
            importance = (importance - importance.min()) / (importance.max() - importance.min())

        # Mask low-importance regions (below 25th percentile)
        threshold = np.percentile(importance, 25)
        weights = np.where(importance <= threshold, 0.0, importance)

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

        # Adam optimizer state
        m = np.zeros_like(cf)  # First moment
        v = np.zeros_like(cf)  # Second moment
        beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

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
        epsilon = max(float(self._std.mean()) * 0.01, 1e-4)
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
        pred_norm = float(np.linalg.norm(pred_gradient)) + 1e-30
        prox_norm = float(np.linalg.norm(prox_gradient)) + 1e-30
        full_gradient = w * pred_gradient / pred_norm + (1 - w) * prox_gradient / prox_norm

        return full_gradient.reshape(cf.shape)
