"""Evaluator package for counterfactual explanation metrics.

This module provides an evaluation framework for assessing the quality
of counterfactual explanations in time series classification.

Classes
-------
Evaluator
    Orchestration class that runs multiple metrics over counterfactual pairs.
Metric
    Abstract base class for defining custom evaluation metrics.

Available Metrics
-----------------
Validity
    Measures the fraction of counterfactuals that successfully change
    the model's prediction.
Proximity
    Computes the average L-p distance between original and counterfactual
    instances.
Sparsity
    Calculates the fraction of features that remain unchanged between
    the original and counterfactual.
Plausibility
    Evaluates whether counterfactuals lie within the training data
    distribution using outlier detection methods (LOF, IsolationForest).
Diversity
    Measures the diversity among multiple counterfactuals for the same
    query using a DPP-inspired log-determinant statistic.
Controllability
    Assesses how easily a counterfactual can be reverted to the original
    prediction through minimal edits.
Confidence
    Reports model confidence statistics (probabilities) for both original
    and counterfactual instances.
Composition
    Provides segment-based statistics about the structure of edits
    (number of segments, average segment length).
Contiguity
    Measures how contiguous the edits are (fewer separate edit regions
    yields higher contiguity).
Robustness
    Estimates local Lipschitz-like stability of counterfactual generation
    using k-nearest neighbor analysis.
Efficiency
    Reports timing metrics for counterfactual generation.

Example
-------
>>> from tscf_eval.evaluator import Evaluator, Validity, Proximity, Sparsity
>>> import numpy as np
>>>
>>> # Create evaluator with desired metrics
>>> evaluator = Evaluator([Validity(), Proximity(p=2), Sparsity()])
>>>
>>> # Evaluate counterfactuals
>>> X = np.random.randn(10, 100)
>>> X_cf = X + np.random.randn(10, 100) * 0.1
>>> results = evaluator.evaluate(X, X_cf, y=np.zeros(10), y_cf=np.ones(10))
>>> print(results)

See Also
--------
tscf_eval.counterfactuals : Counterfactual generation algorithms.
"""

from .base import Metric
from .evaluator import Evaluator
from .metrics import (
    Composition,
    Confidence,
    Contiguity,
    Controllability,
    Diversity,
    Efficiency,
    Plausibility,
    Proximity,
    Robustness,
    Sparsity,
    Validity,
)

__all__ = [
    "Composition",
    "Confidence",
    "Contiguity",
    "Controllability",
    "Diversity",
    "Efficiency",
    "Evaluator",
    "Metric",
    "Plausibility",
    "Proximity",
    "Robustness",
    "Sparsity",
    "Validity",
]
