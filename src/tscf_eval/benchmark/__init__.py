"""Benchmark framework for evaluating counterfactual explainers.

This module provides tools for systematically benchmarking counterfactual
explanation methods across multiple classifiers and datasets.

Classes
-------
DatasetConfig
    Configuration for a dataset.
ModelConfig
    Configuration for a classifier model.
ExplainerConfig
    Configuration for a counterfactual explainer.
BenchmarkRunner
    Orchestrates benchmark execution with parallel support.
BenchmarkResults
    Container for benchmark results with analysis methods.
ParetoAnalyzer
    Multi-objective Pareto dominance analysis.

Examples
--------
>>> from tscf_eval.benchmark import (
...     DatasetConfig, ModelConfig, ExplainerConfig,
...     BenchmarkRunner, ParetoAnalyzer
... )
>>> from tscf_eval import COMTE, NativeGuide
>>>
>>> # Configure datasets
>>> datasets = [
...     DatasetConfig("Dataset1", X1_train, y1_train, X1_test, y1_test),
...     DatasetConfig("Dataset2", X2_train, y2_train, X2_test, y2_test),
... ]
>>>
>>> # Configure models (must be fitted)
>>> models = [
...     ModelConfig("knn", knn_clf),
...     ModelConfig("rocket", rocket_clf),
... ]
>>>
>>> # Configure explainers
>>> explainers = [
...     ExplainerConfig("comte", COMTE, {"distance": "dtw"}),
...     ExplainerConfig("ng_blend", NativeGuide, {"method": "blend"}),
... ]
>>>
>>> # Run benchmark
>>> runner = BenchmarkRunner(datasets, models, explainers, n_jobs=-1)
>>> results = runner.run()
>>>
>>> # Analyze with Pareto dominance
>>> analyzer = ParetoAnalyzer(["validity", "proximity_l2"])
>>> ranking = analyzer.dominance_ranking(results)
"""

from .config import DatasetConfig, ExplainerConfig, ModelConfig
from .multi_criteria import (
    FriedmanResult,
    ParetoAnalyzer,
    WeightedScalarizer,
    _is_maximize as is_maximize,
    format_latex_table,
    friedman_test,
)
from .results import BenchmarkResults, ExplainerResult
from .runner import BenchmarkRunner
from .selection import N_CONFIDENCE_BINS, SelectionStrategy, compute_confidence_bins

__all__ = [
    "N_CONFIDENCE_BINS",
    "BenchmarkResults",
    "BenchmarkRunner",
    "DatasetConfig",
    "ExplainerConfig",
    "ExplainerResult",
    "FriedmanResult",
    "ModelConfig",
    "ParetoAnalyzer",
    "SelectionStrategy",
    "WeightedScalarizer",
    "compute_confidence_bins",
    "format_latex_table",
    "friedman_test",
    "is_maximize",
]
