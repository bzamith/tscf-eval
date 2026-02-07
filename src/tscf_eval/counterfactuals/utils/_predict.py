"""Prediction utilities for counterfactual explainers.

Functions
---------
predict_proba_fn
    Wraps a classifier's ``predict_proba`` method for consistent interface.
soft_predict_proba_fn
    Wraps a classifier to provide soft probabilities for gradient-based methods.
supports_soft_probabilities
    Check if a classifier supports smooth probability estimates for gradients.
has_expensive_transform
    Check if a classifier has an expensive internal transform pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import warnings

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


def predict_proba_fn(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a classifier's ``predict_proba`` to accept array-like inputs.

    Parameters
    ----------
    model : object
        Classifier exposing a ``predict_proba(X)`` method.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        A function that casts input to ``np.asarray`` and calls
        ``model.predict_proba``.
    """

    def _predict(X: np.ndarray) -> np.ndarray:
        """Return class probabilities for *X* via the wrapped model.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape ``(N, ...)`` accepted by the model.

        Returns
        -------
        np.ndarray
            Probability matrix of shape ``(N, n_classes)``.
        """
        result: np.ndarray = model.predict_proba(np.asarray(X))
        return result

    return _predict


def soft_predict_proba_fn(
    model: Any, temperature: float | None = None
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a classifier to provide soft (smooth) probability estimates.

    This function is designed for gradient-based counterfactual methods that
    require smooth probability estimates. It handles classifiers like ROCKET
    that use internal RidgeClassifier and return hard 0/1 probabilities.

    For classifiers with a decision function, it applies sigmoid transformation
    with temperature scaling to produce smooth probabilities suitable for
    gradient computation.

    Parameters
    ----------
    model : object
        Classifier with either:
        - ``predict_proba(X)`` method returning soft probabilities, OR
        - ``decision_function(X)`` method (direct or via internal estimator)
    temperature : float or None, default None
        Temperature scaling for sigmoid/softmax. Higher values produce smoother
        gradients (less saturation). If None, auto-calibrates based on decision
        function magnitude to keep probabilities in [0.1, 0.9] range.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        A function that returns soft probability estimates.

    Notes
    -----
    For ROCKET-style classifiers (aeon), this function uses the internal
    ``pipeline_`` (which includes all intermediate steps such as StandardScaler)
    to transform inputs, then applies the estimator's ``decision_function``
    to compute soft probabilities.

    The temperature parameter prevents sigmoid saturation. If decision values
    are large (e.g., |d| > 5), sigmoid(d) ≈ 0 or 1, producing near-zero
    gradients. Temperature scaling (sigmoid(d/T)) keeps outputs in a range
    where gradients are meaningful.

    Examples
    --------
    >>> from aeon.classification.convolution_based import RocketClassifier
    >>> model = RocketClassifier(n_kernels=500)
    >>> model.fit(X_train, y_train)
    >>> soft_proba = soft_predict_proba_fn(model)
    >>> probs = soft_proba(X_test)  # Returns smooth probabilities
    """
    # Check if model is a ROCKET-style classifier with internal transformer
    has_transformer = hasattr(model, "_transformer") and hasattr(model, "_estimator")
    has_decision_fn = has_transformer and hasattr(model._estimator, "decision_function")

    if has_decision_fn:
        return _make_rocket_soft_proba(model, temperature=temperature)

    # Check if model has direct decision_function
    if hasattr(model, "decision_function"):
        return _make_decision_fn_soft_proba(model, temperature=temperature)

    # Fall back to regular predict_proba (may not be smooth!)
    warnings.warn(
        f"Model {type(model).__name__} does not have a decision_function. "
        f"soft_predict_proba_fn() is falling back to raw predict_proba, which may "
        f"return hard 0/1 probabilities unsuitable for gradient-based optimization.",
        UserWarning,
        stacklevel=2,
    )
    return predict_proba_fn(model)


def _calibrate_from_decision_values(decision: np.ndarray, target_range: float = 2.2) -> float:
    """Calibrate temperature from actual decision function values.

    Uses the median absolute decision value to set a temperature that keeps
    sigmoid(d/T) in a range with meaningful gradients.

    Parameters
    ----------
    decision : np.ndarray
        Decision function values (1D or 2D).
    target_range : float, default 2.2
        Target range for scaled decision values. sigmoid(±2.2) ≈ 0.1/0.9.

    Returns
    -------
    float
        Calibrated temperature value (minimum 1.0).
    """
    abs_vals = np.abs(decision.ravel())
    if abs_vals.size == 0:
        return 1.0
    # Use median absolute value as a robust estimate of typical magnitude
    typical_magnitude = float(np.median(abs_vals))
    if typical_magnitude > target_range:
        return typical_magnitude / target_range
    return 1.0


def supports_soft_probabilities(model: Any) -> bool:
    """Check if a classifier supports smooth probability estimates.

    Gradient-based counterfactual methods (Glacier, LatentCF) require classifiers
    that produce smooth probability surfaces. Tree-based models like RandomForest,
    TimeSeriesForest, and Catch22 (which wraps RandomForest) return discrete
    probabilities that don't respond well to small perturbations.

    Parameters
    ----------
    model : object
        A fitted classifier.

    Returns
    -------
    bool
        True if the classifier supports soft probabilities suitable for
        gradient-based optimization, False otherwise.

    Examples
    --------
    >>> from aeon.classification.convolution_based import RocketClassifier
    >>> from aeon.classification.interval_based import TimeSeriesForestClassifier
    >>> rocket = RocketClassifier().fit(X_train, y_train)
    >>> tsf = TimeSeriesForestClassifier().fit(X_train, y_train)
    >>> supports_soft_probabilities(rocket)  # True (has decision_function)
    True
    >>> supports_soft_probabilities(tsf)  # False (tree-based)
    False
    """
    # ROCKET-style classifiers with internal RidgeClassifier
    has_transformer = hasattr(model, "_transformer") and hasattr(model, "_estimator")
    if has_transformer and hasattr(model._estimator, "decision_function"):
        return True

    # Direct decision_function (SVM, logistic regression, etc.)
    if hasattr(model, "decision_function"):
        return True

    # Check for known non-gradient-friendly classifiers by class name
    class_name = type(model).__name__
    non_gradient_classifiers = {
        "TimeSeriesForestClassifier",
        "Catch22Classifier",
        "RandomForestClassifier",
        "DecisionTreeClassifier",
        "ExtraTreesClassifier",
        "GradientBoostingClassifier",
        "CanonicalIntervalForestClassifier",
        "DrCIFClassifier",
        "Arsenal",
        "HIVECOTEV2",
    }

    if class_name in non_gradient_classifiers:
        return False

    # Check if internal estimator is tree-based (for pipeline-style classifiers)
    if hasattr(model, "_estimator"):
        estimator_name = type(model._estimator).__name__
        if "Forest" in estimator_name or "Tree" in estimator_name:
            return False

    # Known classifiers with native smooth predict_proba (no decision_function
    # needed because their probabilities are already smooth/distance-based)
    smooth_proba_classifiers = {
        "KNeighborsClassifier",
        "KNeighborsTimeSeriesClassifier",
        "LogisticRegression",
        "GaussianNB",
        "MLPClassifier",
    }
    if class_name in smooth_proba_classifiers:
        return True

    # Check if model has predict_proba (heuristic: if it does, assume smooth)
    if hasattr(model, "predict_proba"):
        return True

    # Default: assume soft probabilities are available
    # (may still fail at runtime, but we can't know for sure)
    warnings.warn(
        f"Cannot determine if {type(model).__name__} supports soft probabilities. "
        f"Assuming True; gradient-based methods may fail at runtime if the model "
        f"returns discrete probabilities.",
        UserWarning,
        stacklevel=2,
    )
    return True


def has_expensive_transform(model: Any) -> bool:
    """Check if a classifier has an expensive internal transform pipeline.

    Gradient-based counterfactual methods (CELS, Glacier, LatentCF) compute
    finite-difference gradients by calling ``predict_proba`` many times per
    iteration. For classifiers with expensive transform pipelines (e.g. ROCKET
    convolves with thousands of random kernels, RDST computes shapelet
    distances), each call triggers the full transform, making the gradient
    loop very slow.

    When this returns True, the gradient subsample floor (``n_features // 2``)
    is skipped so the user-specified ``gradient_subsample`` is respected,
    reducing the number of expensive transform calls per iteration.

    Parameters
    ----------
    model : object
        A fitted classifier.

    Returns
    -------
    bool
        True if the classifier has an expensive transform pipeline.
    """
    class_name = type(model).__name__

    # Known classifiers with expensive internal transforms
    expensive_classifiers = {
        "RocketClassifier",
        "MiniRocketClassifier",
        "MultiRocketClassifier",
        "RDSTClassifier",
        "Arsenal",
        "TSFreshClassifier",
        "TimeSeriesForestClassifier",
    }

    if class_name in expensive_classifiers:
        return True

    # Check for pipeline-style classifiers with expensive transformers
    if hasattr(model, "_transformer"):
        transformer_name = type(model._transformer).__name__
        expensive_transformers = {
            "Rocket",
            "MiniRocket",
            "MultiRocket",
            "RandomDilatedShapeletTransform",
        }
        if transformer_name in expensive_transformers:
            return True

    return False


def _make_rocket_soft_proba(
    model: Any, temperature: float | None = None
) -> Callable[[np.ndarray], np.ndarray]:
    """Create soft proba function for ROCKET-style classifiers.

    Parameters
    ----------
    model : object
        ROCKET-style classifier with ``pipeline_`` and ``_estimator`` attributes.
        The ``pipeline_`` is an sklearn Pipeline whose steps may include
        transformers (e.g. Rocket, StandardScaler) followed by the estimator.
    temperature : float or None, default None
        Temperature scaling for the sigmoid/softmax. If None, auto-calibrates
        on the first batch of inputs using actual decision function values.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Function that takes input array and returns soft probabilities.
    """
    # Build the full transform chain: all pipeline steps except the final estimator.
    # ROCKET's pipeline_ is typically: Rocket → StandardScaler → RidgeClassifierCV.
    # Using _transformer alone skips intermediate steps like StandardScaler.
    pipeline_steps = model.pipeline_[:-1] if hasattr(model, "pipeline_") else None

    estimator = model._estimator

    # Mutable container for lazy temperature calibration
    state: dict[str, float | None] = {"temperature": temperature}

    def _soft_predict(X: np.ndarray) -> np.ndarray:
        """Return soft probabilities via the ROCKET pipeline and sigmoid/softmax.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape ``(N, ...)`` accepted by the pipeline.

        Returns
        -------
        np.ndarray
            Soft probability matrix of shape ``(N, n_classes)``.
        """
        X = np.asarray(X)
        # Transform through the full pipeline (Rocket + StandardScaler + ...)
        if pipeline_steps is not None:
            X_transformed = pipeline_steps.transform(X)
        else:
            # Fallback: use _transformer directly (no intermediate steps)
            X_transformed = model._transformer.transform(X)

        # Get decision function values from the estimator
        decision = estimator.decision_function(X_transformed)

        # Lazy calibration: use actual decision values from first call
        if state["temperature"] is None:
            state["temperature"] = _calibrate_from_decision_values(decision)

        temp = state["temperature"]

        # Handle binary vs multiclass
        if decision.ndim == 1:
            # Binary classification: decision is 1D
            # Convert to 2D probabilities using sigmoid with temperature scaling
            prob_pos = _sigmoid(decision / temp)
            probs = np.column_stack([1 - prob_pos, prob_pos])
        else:
            # Multiclass: apply softmax with temperature
            probs = _softmax(decision / temp)

        return probs

    return _soft_predict


def _make_decision_fn_soft_proba(
    model: Any, temperature: float | None = None
) -> Callable[[np.ndarray], np.ndarray]:
    """Create soft proba function using model's decision_function.

    Parameters
    ----------
    model : object
        Classifier with decision_function method.
    temperature : float or None, default None
        Temperature scaling for the sigmoid/softmax. If None, auto-calibrates
        on the first batch of inputs using actual decision function values.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Function that takes input array and returns soft probabilities.
    """
    state: dict[str, float | None] = {"temperature": temperature}

    def _soft_predict(X: np.ndarray) -> np.ndarray:
        """Return soft probabilities via ``decision_function`` and sigmoid/softmax.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape ``(N, ...)`` accepted by the model.

        Returns
        -------
        np.ndarray
            Soft probability matrix of shape ``(N, n_classes)``.
        """
        X = np.asarray(X)
        decision = model.decision_function(X)

        if state["temperature"] is None:
            state["temperature"] = _calibrate_from_decision_values(decision)

        temp = state["temperature"]

        if decision.ndim == 1:
            prob_pos = _sigmoid(decision / temp)
            probs = np.column_stack([1 - prob_pos, prob_pos])
        else:
            probs = _softmax(decision / temp)

        return probs

    return _soft_predict


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply the numerically stable sigmoid function element-wise.

    Parameters
    ----------
    x : np.ndarray
        Input array of any shape.

    Returns
    -------
    np.ndarray
        Element-wise sigmoid values in ``(0, 1)``, same shape as *x*.
    """
    # Use np.clip to avoid overflow warnings
    x = np.clip(x, -500, 500)
    result: np.ndarray = 1.0 / (1.0 + np.exp(-x))
    return result


def _softmax(x: np.ndarray) -> np.ndarray:
    """Apply the numerically stable softmax function row-wise.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape ``(N, C)`` where *C* is the number of classes.

    Returns
    -------
    np.ndarray
        Row-wise softmax probabilities of shape ``(N, C)``.
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    result: np.ndarray = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return result
