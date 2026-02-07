"""Instance selection strategies for benchmark execution.

This module provides functions for selecting test instances to use
in benchmark runs. Strategies range from simple random sampling to
confidence-based stratified sampling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
import warnings

import numpy as np

from tscf_eval.counterfactuals.utils import soft_predict_proba_fn

if TYPE_CHECKING:
    from .config import DatasetConfig, ModelConfig

__all__ = [
    "N_CONFIDENCE_BINS",
    "SelectionStrategy",
    "compute_confidence_bins",
    "select_instances",
]

SelectionStrategy = Literal["random", "stratified_confidence"]
"""Supported instance selection strategies."""

N_CONFIDENCE_BINS = 4
"""Number of quantile-based confidence bins for stratified selection."""


def select_instances(
    dataset: DatasetConfig,
    model: ModelConfig,
    n_instances: int | None,
    strategy: SelectionStrategy,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Select test instances according to the given strategy.

    Parameters
    ----------
    dataset : DatasetConfig
        Dataset containing test instances.
    model : ModelConfig
        Fitted model (used for confidence-based strategies).
    n_instances : int or None
        Number of instances to select. None means use all.
    strategy : {"random", "stratified_confidence"}
        Instance selection strategy.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    X_test : np.ndarray
        Selected test instances.
    y_test : np.ndarray or None
        Corresponding labels, or None if not available.
    bin_indices : np.ndarray or None
        Confidence bin assignment for each selected instance (computed
        over the full test set), or ``None`` when stratified binning
        was not performed (e.g. random strategy, no ``predict_proba``,
        or no subsampling).
    """
    X_test = dataset.X_test
    y_test = dataset.y_test

    # No subsampling needed
    if n_instances is None or n_instances >= len(X_test):
        return X_test, y_test, None

    if strategy == "random":
        indices = _select_random(len(X_test), n_instances, random_state)
        bin_indices = None
    elif strategy == "stratified_confidence":
        indices, bin_indices = _select_stratified_confidence(
            X_test,
            model,
            n_instances,
            random_state,
        )
    else:
        raise ValueError(
            f"Unknown selection strategy: {strategy!r}. "
            f"Expected 'random' or 'stratified_confidence'."
        )

    X_test = X_test[indices]
    if y_test is not None:
        y_test = y_test[indices]

    return X_test, y_test, bin_indices


def _select_random(
    n_total: int,
    n_instances: int,
    random_state: int | None,
) -> np.ndarray:
    """Select instances uniformly at random without replacement.

    Parameters
    ----------
    n_total : int
        Total number of available instances.
    n_instances : int
        Number of instances to select.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of selected indices.
    """
    rng = np.random.default_rng(random_state)
    return rng.choice(n_total, size=n_instances, replace=False)


def _select_stratified_confidence(
    X_test: np.ndarray,
    model: ModelConfig,
    n_instances: int,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Select instances stratified by model confidence.

    Computes model confidence (max predicted probability) for each
    test instance, divides instances into quantile-based bins
    (25th, 50th, 75th percentiles), and samples from each bin.
    This ensures the benchmark covers instances the model is both
    very confident and very uncertain about.

    Falls back to random selection if the model does not support
    ``predict_proba`` or if ``n_instances`` is too small.

    Parameters
    ----------
    X_test : np.ndarray
        Test instances.
    model : ModelConfig
        Fitted model with optional ``predict_proba`` support.
    n_instances : int
        Total number of instances to select.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    indices : np.ndarray
        Array of selected indices.
    bin_indices : np.ndarray or None
        Confidence bin assignment for each selected instance (computed
        over the full test set), or ``None`` when falling back to
        random selection.
    """
    # Guard: need at least one instance per bin for stratification
    if n_instances < N_CONFIDENCE_BINS:
        warnings.warn(
            f"n_instances={n_instances} is less than the number of "
            f"confidence bins ({N_CONFIDENCE_BINS}). "
            f"Falling back to random selection.",
            UserWarning,
            stacklevel=3,
        )
        return _select_random(len(X_test), n_instances, random_state), None

    # Get soft probabilities for confidence estimation.
    # Some classifiers (ROCKET, RDST) use RidgeClassifierCV internally and
    # return hard 0/1 from predict_proba, making all confidences identical.
    # soft_predict_proba_fn converts decision_function outputs to smooth
    # probabilities via sigmoid/softmax, giving meaningful confidence spread.
    # For classifiers with native smooth predict_proba (LR, MLP, deep
    # learning) it falls through to predict_proba unchanged.
    confidence = _get_soft_confidence(X_test, model)
    if confidence is None:
        warnings.warn(
            f"Model '{model.name}' does not support predict_proba. "
            f"Falling back to random instance selection.",
            UserWarning,
            stacklevel=3,
        )
        return _select_random(len(X_test), n_instances, random_state), None

    # Create quantile-based bin edges (4 bins: 0-25%, 25-50%, 50-75%, 75-100%)
    quantiles = np.linspace(0, 1, N_CONFIDENCE_BINS + 1)
    bin_edges = np.quantile(confidence, quantiles)

    # Assign each instance to a bin (over full test set)
    # np.digitize with bin_edges[1:-1] maps to bins 0..N_CONFIDENCE_BINS-1
    bin_indices_full = np.digitize(confidence, bin_edges[1:-1], right=True)

    # Collect indices per bin
    bins: list[np.ndarray] = [np.where(bin_indices_full == b)[0] for b in range(N_CONFIDENCE_BINS)]

    # Compute per-bin allocation
    base_per_bin = n_instances // N_CONFIDENCE_BINS
    remainder = n_instances % N_CONFIDENCE_BINS
    allocations = [base_per_bin] * N_CONFIDENCE_BINS
    for i in range(remainder):
        allocations[i] += 1

    # Sample from each bin; redistribute deficit if a bin is too small
    rng = np.random.default_rng(random_state)
    selected: list[np.ndarray] = []
    deficit = 0

    for b in range(N_CONFIDENCE_BINS):
        available = len(bins[b])
        desired = allocations[b] + deficit
        deficit = 0

        if available == 0:
            deficit += desired
            continue

        take = min(desired, available)
        chosen = rng.choice(bins[b], size=take, replace=False)
        selected.append(chosen)

        if take < desired:
            deficit += desired - take

    # If deficit remains, fill from any remaining instances
    if deficit > 0:
        already_selected = np.concatenate(selected) if selected else np.array([], dtype=int)
        remaining = np.setdiff1d(np.arange(len(X_test)), already_selected)
        if len(remaining) > 0:
            extra = rng.choice(remaining, size=min(deficit, len(remaining)), replace=False)
            selected.append(extra)

    indices = np.concatenate(selected)
    return indices, bin_indices_full[indices]


def compute_confidence_bins(
    X_test: np.ndarray,
    model: ModelConfig,
) -> np.ndarray | None:
    """Compute confidence quartile bin assignments for given instances.

    Uses the same quantile-based binning as
    :func:`_select_stratified_confidence`: confidence is defined as
    ``max(predict_proba)`` per instance, then split into
    :data:`N_CONFIDENCE_BINS` equal-frequency bins.

    Parameters
    ----------
    X_test : np.ndarray
        Test instances (already selected).
    model : ModelConfig
        Fitted model with optional ``predict_proba`` support.

    Returns
    -------
    np.ndarray or None
        Integer array of shape ``(len(X_test),)`` with values in
        ``{0, 1, ..., N_CONFIDENCE_BINS - 1}`` where 0 is the
        lowest-confidence bin. Returns ``None`` if the model does not
        support ``predict_proba``.
    """
    confidence = _get_soft_confidence(X_test, model)
    if confidence is None:
        return None

    quantiles = np.linspace(0, 1, N_CONFIDENCE_BINS + 1)
    bin_edges = np.quantile(confidence, quantiles)
    bin_indices: np.ndarray = np.digitize(confidence, bin_edges[1:-1], right=True)

    return bin_indices


def _get_soft_confidence(
    X_test: np.ndarray,
    model: ModelConfig,
) -> np.ndarray | None:
    """Compute per-instance confidence using soft probabilities.

    Uses :func:`soft_predict_proba_fn` so that classifiers with hard 0/1
    ``predict_proba`` (e.g. ROCKET, RDST with ``RidgeClassifierCV``) are
    converted to smooth probabilities via their ``decision_function``.
    Classifiers with native smooth ``predict_proba`` (logistic regression,
    MLP, deep learning) pass through unchanged.

    Parameters
    ----------
    X_test : np.ndarray
        Test instances.
    model : ModelConfig
        Fitted model wrapper.

    Returns
    -------
    np.ndarray or None
        Max predicted probability per instance, or ``None`` if the model
        supports neither ``predict_proba`` nor ``decision_function``.
    """
    # Try soft_predict_proba_fn first (handles ROCKET/RDST decision_function)
    try:
        soft_proba = soft_predict_proba_fn(model.model)
        proba_arr: np.ndarray = np.asarray(soft_proba(X_test))
        confidence: np.ndarray = np.max(proba_arr, axis=1)
        return confidence
    except (TypeError, AttributeError, ValueError) as exc:
        warnings.warn(
            f"soft_predict_proba_fn failed for model '{model.name}': {exc}. "
            f"Falling back to raw predict_proba for confidence estimation.",
            UserWarning,
            stacklevel=2,
        )

    # Fallback to raw predict_proba
    raw_proba = model.predict_proba(X_test)
    if raw_proba is None:
        return None
    fallback_confidence: np.ndarray = np.max(raw_proba, axis=1)
    return fallback_confidence
