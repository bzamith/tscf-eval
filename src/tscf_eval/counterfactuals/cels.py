"""CELS counterfactual explainer implementation.

This module provides the ``CELS`` class, an implementation of the CELS
(Counterfactual Explanations via Learned Saliency maps) algorithm for
generating counterfactual explanations for time series classification.

The algorithm was originally developed by Peiyu Li, Beineng Tang, and
Yue Ning at Stevens Institute of Technology.

Original implementation: https://github.com/Luckilyeee/CELS

Classes
-------
CELS
    CELS counterfactual generator using learned saliency maps.

Algorithm Overview
------------------
CELS generates counterfactuals by learning a saliency map that blends the
original instance with its nearest unlike neighbor (NUN):

1. Find the nearest unlike neighbor (NUN) from the reference dataset -- the
   closest instance classified as the target class.
2. Initialize a saliency map theta in [0, 1]^T with small random values.
3. Optimize theta via Adam to minimize a composite loss:

   - **L_Max**: Prediction loss driving the counterfactual toward the target
     class: ``max_coeff * (1 - P(target | x'))``.
   - **L_Budget**: Sparsity penalty on the saliency map:
     ``budget_coeff * mean(|theta|)``.
   - **L_TV**: Total variation norm promoting temporal smoothness:
     ``tv_coeff * mean(|theta_t - theta_{t+1}|^tv_beta)``.

4. The counterfactual is computed as ``x' = x * (1 - theta) + NUN * theta``.
5. Post-process: normalize theta to [0, 1], threshold at ``k`` to produce a
   binary mask, then recompute the counterfactual with the binary mask.

Examples
--------
>>> from tscf_eval.counterfactuals import CELS
>>> import numpy as np
>>>
>>> # Assume clf is a trained classifier with predict_proba
>>> cels = CELS(
...     model=clf,
...     data=(X_train, y_train),
...     budget_coeff=0.6,
...     max_iter=1000,
... )
>>>
>>> # Generate counterfactual for a test instance
>>> cf, cf_label, meta = cels.explain(x_test)
>>> print(f"Converged: {meta['converged']}")
>>> print(f"Mask density: {meta['mask_density']:.2f}")

References
----------
.. [cels1] Li, P., Tang, B., & Ning, Y. (2023).
       CELS: Counterfactual Explanation of Time-Series via Learned Saliency Maps.
       In Proceedings of the IEEE International Conference on Big Data 2023,
       pp. 1952-1957. IEEE.
       DOI: 10.1109/BigData59044.2023.10386404

See Also
--------
tscf_eval.counterfactuals.Glacier : Glacier gradient-based algorithm.
tscf_eval.counterfactuals.LatentCF : LatentCF++ gradient-based algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings

import numpy as np

from .base import Counterfactual
from .utils import (
    ensure_batch_shape,
    has_expensive_transform,
    soft_predict_proba_fn,
    strip_batch,
    supports_soft_probabilities,
)
from .utils._adam import AdamState
from .utils._nun import find_nearest_unlike_neighbor


@dataclass
class CELS(Counterfactual):
    """CELS counterfactual generator via learned saliency maps.

    Implementation of the CELS algorithm by Li et al. (2023) [cels1]_.

    CELS learns a saliency map theta that blends the original instance with
    its nearest unlike neighbor (NUN) to produce a counterfactual explanation.
    The saliency map is optimized to minimize a composite loss balancing
    prediction change, sparsity, and temporal smoothness.

    The counterfactual is computed as:

        x' = x * (1 - theta) + NUN * theta

    After optimization, theta is binarized to produce contiguous edits.

    Parameters
    ----------
    model : object
        A classifier with a probability estimator (``predict_proba`` or a
        compatible interface). Must be differentiable or approximable.
    data : tuple (``X_ref``, ``y_ref``)
        Reference dataset used for finding nearest unlike neighbors.
    budget_coeff : float, default 0.6
        Weight for the budget (sparsity) loss term ``L_Budget = a * mean(|theta|)``.
        Higher values produce sparser saliency maps.
    tv_coeff : float, default 0.5
        Weight for the total variation loss term
        ``L_TV = b * mean(|theta_t - theta_{t+1}|^tv_beta)``.
        Higher values produce temporally smoother saliency maps.
    max_coeff : float, default 0.7
        Weight for the prediction loss term
        ``L_Max = g * (1 - P(target | x'))``.
        Higher values prioritize changing the prediction.
    tv_beta : float, default 3.0
        Exponent for the total variation norm. Higher values penalize
        large jumps more aggressively while tolerating small ones.
    gradient_subsample : int or None, default 50
        Number of features to randomly sample for gradient computation each
        iteration. Uses stochastic gradient descent when set to a value less
        than the total number of features. Set to None to use all features
        (full gradient). Lower values speed up computation but may require
        more iterations to converge.
    learning_rate : float, default 0.1
        Step size for Adam optimizer. Internally scaled by data standard
        deviation so the effective step adapts to input magnitude.
    max_iter : int, default 5000
        Maximum number of optimization iterations.
    tau : float, default 0.5
        Decision threshold for target class probability. Optimization
        considers convergence when ``P(target_class) >= tau``.
    patience : int, default 30
        Number of consecutive iterations where ``P(target) >= tau`` before
        stopping. Allows the optimizer to further refine the saliency map
        after the prediction flips.
    tolerance : float, default 1e-4
        Convergence tolerance for total loss change between iterations.
    threshold : float, default 0.5
        Binarization threshold for the saliency map during post-processing.
        Values of theta above this become 1 (use NUN), below become 0 (keep
        original).
    random_state : int or None, default 0
        PRNG seed for reproducible optimization.
    temperature : float or None, default None
        Temperature scaling for soft probability computation. Higher values
        produce smoother gradients by preventing sigmoid saturation when
        decision function values are large. If None, auto-calibrates based
        on model decision function values.

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

    References
    ----------
    .. [cels1] Li, P., Tang, B., & Ning, Y. (2023).
           CELS: Counterfactual Explanation of Time-Series via Learned
           Saliency Maps. In Proceedings of the IEEE International Conference
           on Big Data 2023, pp. 1952-1957. IEEE.
           https://github.com/Luckilyeee/CELS
    """

    model: Any
    data: tuple[np.ndarray, np.ndarray]
    # Loss coefficients
    budget_coeff: float = 0.6
    tv_coeff: float = 0.5
    max_coeff: float = 0.7
    tv_beta: float = 3.0
    gradient_subsample: int | None = 50
    # Optimization
    learning_rate: float = 0.1
    max_iter: int = 5000
    tau: float = 0.5
    patience: int = 30
    tolerance: float = 1e-4
    # Post-processing
    threshold: float = 0.5
    # Common
    random_state: int | None = 0
    temperature: float | None = None

    # Internal state
    _mean: np.ndarray = field(default_factory=lambda: np.array([0.0]), init=False, repr=False)
    _std: np.ndarray = field(default_factory=lambda: np.array([1.0]), init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialise probability wrapper, RNG, reference data, and label mapping.

        Validates all hyperparameters and computes normalisation statistics
        from the reference dataset. Warns if the model is unlikely to work
        well with gradient-based optimisation.
        """
        if not supports_soft_probabilities(self.model):
            warnings.warn(
                f"CELS is a gradient-based method that may not work well with "
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
        self._std = self.X_ref.std(axis=0) + 1e-8

        # Validate parameters
        if self.budget_coeff < 0:
            raise ValueError("budget_coeff must be >= 0")
        if self.tv_coeff < 0:
            raise ValueError("tv_coeff must be >= 0")
        if self.max_coeff < 0:
            raise ValueError("max_coeff must be >= 0")
        if self.tv_beta <= 0:
            raise ValueError("tv_beta must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError("tau must be in (0, 1]")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if not (0.0 < self.threshold < 1.0):
            raise ValueError("threshold must be in (0, 1)")
        if self.gradient_subsample is not None and self.gradient_subsample < 1:
            raise ValueError("gradient_subsample must be >= 1 or None")

    def explain(
        self,
        x: np.ndarray,
        y_pred: int | None = None,
        *,
        class_of_interest: int | None = None,
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        """Generate a counterfactual explanation via learned saliency map.

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

            - ``method``: Algorithm identifier (``'cels'``).
            - ``class_of_interest``: Target class.
            - ``nun_index_in_ref``: Index of NUN in reference dataset.
            - ``n_iterations``: Number of iterations performed.
            - ``converged``: Whether optimization converged.
            - ``final_target_prob``: Final probability of target class.
            - ``final_loss``: Final composite loss value.
            - ``mask_density``: Fraction of saliency map above threshold.
            - ``validity``: Whether the counterfactual class differs from base.
        """
        xb, added = ensure_batch_shape(x)
        x1 = strip_batch(xb, added)

        # Determine base prediction and target class.
        # All internal work uses probability column *indices* (0, 1, ...),
        # while y_pred and returned cf_label use actual class *labels*.
        base_probs = self.predict_proba(xb)[0]
        base_idx = int(np.argmax(base_probs)) if y_pred is None else self._label_to_idx(y_pred)

        if class_of_interest is None:
            # Pick the highest-probability alternative class (by index)
            probs_sorted = np.argsort(-base_probs)
            target_idx = int(next(c for c in probs_sorted if c != base_idx))
        else:
            target_idx = self._label_to_idx(class_of_interest)

        # Find nearest unlike neighbor (converts index to label internally)
        nun, nun_idx = self._find_nun(x1, target_idx)

        if nun is None:
            target_label = self._idx_to_label(target_idx)
            warnings.warn(
                f"No instances of target class {target_label} found in "
                f"reference data. Returning original instance.",
                UserWarning,
                stacklevel=2,
            )
            cf_label = self._idx_to_label(base_idx)
            meta: dict[str, Any] = {
                "method": "cels",
                "class_of_interest": self._idx_to_label(target_idx),
                "nun_index_in_ref": None,
                "n_iterations": 0,
                "converged": False,
                "final_target_prob": float(base_probs[target_idx]),
                "final_loss": float("inf"),
                "mask_density": 0.0,
                "validity": False,
            }
            return x1.copy(), cf_label, meta

        # Run optimization (works in index space for predict_proba columns)
        theta, n_iter, converged, final_prob, final_loss = self._optimize(x1, nun, target_idx)

        # Post-process: binarize saliency map
        mask = self._binarize_theta(theta)

        # When the binary mask is all zeros (no entries above threshold) and
        # optimization didn't converge, fall back to using the continuous
        # theta to produce a CF that at least differs from the original.
        # This typically happens with tree-based classifiers whose discrete
        # probabilities yield zero gradients.
        if not np.any(mask > 0) and not converged:
            cf = x1 * (1.0 - theta) + nun * theta
        else:
            cf = x1 * (1.0 - mask) + nun * mask

        # Get final prediction — convert index back to actual label
        cf_probs = self.predict_proba(cf[None, ...])[0]
        cf_idx = int(np.argmax(cf_probs))
        cf_label = self._idx_to_label(cf_idx)

        mask_density = float(np.mean(mask > 0) if np.any(mask > 0) else np.mean(theta))

        meta = {
            "method": "cels",
            "class_of_interest": self._idx_to_label(target_idx),
            "nun_index_in_ref": nun_idx,
            "n_iterations": n_iter,
            "converged": converged,
            "final_target_prob": float(final_prob),
            "final_loss": float(final_loss),
            "mask_density": mask_density,
            "validity": cf_idx != base_idx,
        }

        return cf, cf_label, meta

    def explain_k(
        self,
        x: np.ndarray,
        k: int = 5,
        y_pred: int | None = None,
        *,
        class_of_interest: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Generate k diverse counterfactuals using different NUNs.

        CELS supports diverse counterfactual generation by using different
        nearest unlike neighbors as the blending source. Each counterfactual
        is generated with a different NUN, producing structurally diverse
        explanations while keeping the learned saliency map approach.

        Parameters
        ----------
        x : np.ndarray
            Input time series of shape ``(T,)`` or ``(C, T)``.
        k : int, default 5
            Number of counterfactuals to generate.
        y_pred : int, optional
            Precomputed predicted label for ``x``.
        class_of_interest : int, optional
            Target class for counterfactuals.

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

        # Determine base prediction and target class (index space)
        base_probs = self.predict_proba(xb)[0]
        base_idx = int(np.argmax(base_probs)) if y_pred is None else self._label_to_idx(y_pred)

        if class_of_interest is None:
            probs_sorted = np.argsort(-base_probs)
            target_idx = int(next(c for c in probs_sorted if c != base_idx))
        else:
            target_idx = self._label_to_idx(class_of_interest)

        # Get k nearest unlike neighbors (converts index to label internally)
        nuns, nun_indices = self._find_k_nuns(x1, target_idx, k)

        cfs_list: list[np.ndarray] = []
        cf_labels_list: list[Any] = []
        metas_list: list[dict[str, Any]] = []

        for i, (nun, nun_idx) in enumerate(zip(nuns, nun_indices, strict=True)):
            theta, n_iter, converged, final_prob, final_loss = self._optimize(x1, nun, target_idx)

            mask = self._binarize_theta(theta)

            if not np.any(mask > 0) and not converged:
                cf = x1 * (1.0 - theta) + nun * theta
            else:
                cf = x1 * (1.0 - mask) + nun * mask

            cf_probs = self.predict_proba(cf[None, ...])[0]
            cf_idx = int(np.argmax(cf_probs))
            cf_label = self._idx_to_label(cf_idx)

            mask_density = float(np.mean(mask > 0) if np.any(mask > 0) else np.mean(theta))

            meta: dict[str, Any] = {
                "method": "cels",
                "class_of_interest": self._idx_to_label(target_idx),
                "nun_index_in_ref": nun_idx,
                "n_iterations": n_iter,
                "converged": converged,
                "final_target_prob": float(final_prob),
                "final_loss": float(final_loss),
                "mask_density": mask_density,
                "validity": cf_idx != base_idx,
                "k_index": i,
            }

            cfs_list.append(cf)
            cf_labels_list.append(cf_label)
            metas_list.append(meta)

        return np.array(cfs_list), np.array(cf_labels_list), metas_list

    def _find_nun(self, x: np.ndarray, target_class: int) -> tuple[np.ndarray | None, int | None]:
        """Find the nearest unlike neighbor from the target class.

        Delegates to :func:`~tscf_eval.counterfactuals.utils._nun.find_nearest_unlike_neighbor`.

        Parameters
        ----------
        x : np.ndarray
            Original time series of shape ``(T,)`` or ``(C, T)``.
        target_class : int
            Target class as a probability column index.

        Returns
        -------
        nun : np.ndarray or None
            Nearest unlike neighbor, or None if no target-class instances exist.
        nun_idx : int or None
            Index of the NUN in the reference dataset.
        """
        target_label = self._idx_to_label(target_class)
        nuns, indices = find_nearest_unlike_neighbor(
            x,
            self.X_ref,
            self.y_ref,
            target_label,
            k=1,
        )
        if not nuns:
            return None, None
        return nuns[0], indices[0]

    def _find_k_nuns(
        self, x: np.ndarray, target_class: int, k: int
    ) -> tuple[list[np.ndarray], list[int]]:
        """Find the k nearest unlike neighbors from the target class.

        Delegates to :func:`~tscf_eval.counterfactuals.utils._nun.find_nearest_unlike_neighbor`.

        Parameters
        ----------
        x : np.ndarray
            Original time series of shape ``(T,)`` or ``(C, T)``.
        target_class : int
            Target class as a probability column index.
        k : int
            Number of neighbors to return.

        Returns
        -------
        nuns : list[np.ndarray]
            List of up to k nearest unlike neighbors.
        nun_indices : list[int]
            Indices of the NUNs in the reference dataset.
        """
        target_label = self._idx_to_label(target_class)
        return find_nearest_unlike_neighbor(
            x,
            self.X_ref,
            self.y_ref,
            target_label,
            k=k,
        )

    def _binarize_theta(self, theta: np.ndarray) -> np.ndarray:
        """Normalize and binarize the saliency map.

        Parameters
        ----------
        theta : np.ndarray
            Raw saliency map from optimization.

        Returns
        -------
        np.ndarray
            Binary mask (0 or 1) with same shape as theta.
        """
        # Min-max normalize to [0, 1]
        t_min = theta.min()
        t_max = theta.max()
        if t_max - t_min > 1e-12:
            theta_norm = (theta - t_min) / (t_max - t_min)
        else:
            theta_norm = np.zeros_like(theta)

        # Threshold to binary mask
        mask: np.ndarray = (theta_norm >= self.threshold).astype(np.float64)
        return mask

    def _optimize(
        self,
        x: np.ndarray,
        nun: np.ndarray,
        target_class: int,
    ) -> tuple[np.ndarray, int, bool, float, float]:
        """Run gradient-based optimization of the saliency map.

        Uses Adam optimizer with finite-difference gradients for the
        prediction loss and analytical gradients for budget and TV losses.

        Parameters
        ----------
        x : np.ndarray
            Original time series of shape ``(T,)`` or ``(C, T)``.
        nun : np.ndarray
            Nearest unlike neighbor with the same shape as ``x``.
        target_class : int
            Target class for counterfactual.

        Returns
        -------
        theta : np.ndarray
            Optimized saliency map.
        n_iterations : int
            Number of iterations performed.
        converged : bool
            Whether optimization converged.
        final_target_prob : float
            Final probability of target class.
        final_loss : float
            Final composite loss value.
        """
        # Initialize saliency map with small random values
        theta = self.rng.uniform(0.0, 0.01, size=x.shape).astype(np.float64)

        # Scale learning rate to data magnitude (same as Glacier)
        data_scale = float(self._std.mean())
        effective_lr = self.learning_rate * data_scale

        adam = AdamState.zeros_like(theta)

        converged = False
        n_iterations = 0
        final_prob = 0.0
        final_loss = float("inf")
        patience_counter = 0
        prev_loss = float("inf")

        for iteration in range(self.max_iter):
            n_iterations = iteration + 1

            # Clamp theta to [0, 1]
            theta = np.clip(theta, 0.0, 1.0)

            # Compute counterfactual
            cf = x * (1.0 - theta) + nun * theta

            # Compute current prediction
            probs = self.predict_proba(cf[None, ...])[0]
            target_prob = float(probs[target_class])

            # Compute loss components
            l_max = self.max_coeff * (1.0 - target_prob)
            l_budget = self.budget_coeff * float(np.mean(np.abs(theta)))
            l_tv = self._compute_tv_loss(theta)
            total_loss = l_max + l_budget + l_tv

            final_prob = target_prob
            final_loss = total_loss

            # Check convergence via patience
            if target_prob >= self.tau:
                patience_counter += 1
                if patience_counter >= self.patience:
                    converged = True
                    break
            else:
                patience_counter = 0

            # Check loss convergence
            if abs(prev_loss - total_loss) < self.tolerance and target_prob >= self.tau:
                converged = True
                break

            prev_loss = total_loss

            # Compute gradient
            gradient = self._compute_gradient(theta, x, nun, target_class)

            # Adam update
            theta = theta - adam.step(gradient, effective_lr)

        # Final clamp
        theta = np.clip(theta, 0.0, 1.0)

        return theta, n_iterations, converged, final_prob, final_loss

    def _compute_tv_loss(self, theta: np.ndarray) -> float:
        """Compute total variation loss along the time axis.

        Parameters
        ----------
        theta : np.ndarray
            Saliency map of shape ``(T,)`` or ``(C, T)``.

        Returns
        -------
        float
            Total variation loss value.
        """
        # Differences along time axis (last axis)
        diff = np.diff(theta, axis=-1)
        tv: float = self.tv_coeff * float(np.mean(np.abs(diff) ** self.tv_beta))
        return tv

    def _compute_gradient(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        nun: np.ndarray,
        target_class: int,
    ) -> np.ndarray:
        """Compute gradient of the composite loss w.r.t. theta.

        Uses finite differences for the prediction loss (L_Max) through the
        classifier and analytical gradients for the budget and TV losses.

        For L_Max, we exploit the chain rule: since ``x' = x*(1-theta) + nun*theta``,
        the gradient ``dL_Max/dtheta = dL_Max/dx' * (nun - x)`` where
        ``dL_Max/dx'`` is computed via finite differences in the data space.

        Parameters
        ----------
        theta : np.ndarray
            Current saliency map.
        x : np.ndarray
            Original time series.
        nun : np.ndarray
            Nearest unlike neighbor.
        target_class : int
            Target class for counterfactual.

        Returns
        -------
        np.ndarray
            Gradient of composite loss with respect to theta.
        """
        # Current counterfactual
        cf = x * (1.0 - theta) + nun * theta

        # --- L_Max gradient via finite differences + chain rule ---
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

        # Perturbations in the data space (same as Glacier)
        perturbations = np.tile(flat_cf, (2 * n_sample, 1))
        for i, feat_idx in enumerate(sampled_idx):
            perturbations[i, feat_idx] += epsilon
            perturbations[n_sample + i, feat_idx] -= epsilon

        perturbations_reshaped = perturbations.reshape(2 * n_sample, *cf.shape)
        probs_batch = self.predict_proba(perturbations_reshaped)
        target_probs = probs_batch[:, target_class]

        # dL_Max/dx' = d/dx'[g * (1 - p(target|x'))] = -g * dp/dx'
        # Using central differences for dp/dx'
        dp_dx_sampled = (target_probs[:n_sample] - target_probs[n_sample:]) / (2 * epsilon)

        dp_dx = np.zeros(n_features)
        dp_dx[sampled_idx] = dp_dx_sampled
        dl_max_dx = -self.max_coeff * dp_dx  # gradient of g*(1-p) w.r.t. x'

        # Chain rule: dL_Max/dtheta = dL_Max/dx' * (nun - x)
        nun_minus_x = (nun - x).flatten()
        pred_gradient = dl_max_dx * nun_minus_x

        # --- L_Budget gradient (analytical) ---
        # L_Budget = a * mean(|theta|)
        # dL_Budget/dtheta = a * sign(theta) / N
        flat_theta = theta.flatten()
        n_total = len(flat_theta)
        budget_gradient = self.budget_coeff * np.sign(flat_theta) / n_total

        # --- L_TV gradient (analytical) ---
        tv_gradient = self._compute_tv_gradient(theta).flatten()

        # Normalize each component to prevent scale dominance.
        # When a component norm is negligible (e.g. prediction gradient is
        # zero for tree-based classifiers), zero it out instead of amplifying
        # floating-point noise to unit norm.
        min_norm = 1e-10
        pred_norm = float(np.linalg.norm(pred_gradient))
        budget_norm = float(np.linalg.norm(budget_gradient))
        tv_norm = float(np.linalg.norm(tv_gradient))

        if pred_norm > min_norm:
            full_gradient = (
                pred_gradient / pred_norm
                + (budget_gradient / budget_norm if budget_norm > min_norm else 0.0)
                + (tv_gradient / tv_norm if tv_norm > min_norm else 0.0)
            )
        else:
            # No prediction gradient signal (tree-based classifiers return
            # discrete probabilities that don't change under small
            # perturbations).  Fall back to moving theta toward the NUN by
            # using the negative direction (gradient descent with negative
            # gradient increases theta), balanced by TV smoothness only.
            # Budget regularization is suppressed because it would push
            # theta back to zero with no prediction signal to counteract it.
            fallback = -np.ones_like(pred_gradient)
            fb_norm = float(np.linalg.norm(fallback))
            full_gradient = fallback / fb_norm + (
                tv_gradient / tv_norm if tv_norm > min_norm else 0.0
            )

        result: np.ndarray = full_gradient.reshape(theta.shape)
        return result

    def _compute_tv_gradient(self, theta: np.ndarray) -> np.ndarray:
        """Compute analytical gradient of TV loss w.r.t. theta.

        For ``L_TV = b * mean(|theta_t - theta_{t+1}|^tv_beta)``, the gradient
        at position t is::

            b * tv_beta / N * [|theta_t - theta_{t+1}|^(tv_beta-1)
                               * sign(theta_t - theta_{t+1})
                               - |theta_{t-1} - theta_t|^(tv_beta-1)
                               * sign(theta_{t-1} - theta_t)]

        Boundary terms (t=0, t=T-1) have only one neighbor.

        Parameters
        ----------
        theta : np.ndarray
            Saliency map of shape ``(T,)`` or ``(C, T)``.

        Returns
        -------
        np.ndarray
            TV gradient with the same shape as theta.
        """
        grad = np.zeros_like(theta)
        T = theta.shape[-1]
        beta = self.tv_beta

        if T < 2:
            return grad

        # Forward differences: θ_t - θ_{t+1} for t = 0..T-2
        fwd_diff = np.diff(theta, axis=-1)  # shape (..., T-1)
        abs_fwd = np.abs(fwd_diff)
        # Avoid 0^(beta-1) when beta < 1 (though default is 3.0)
        powered = np.where(abs_fwd > 1e-12, abs_fwd ** (beta - 1), 0.0)
        signed = powered * np.sign(fwd_diff)

        n_total = theta.size
        scale = self.tv_coeff * beta / n_total

        # Contribution from forward difference (θ_t - θ_{t+1}): add to position t
        grad[..., :-1] += scale * signed
        # Contribution from backward difference (θ_{t-1} - θ_t): subtract from position t
        grad[..., 1:] -= scale * signed

        return grad
