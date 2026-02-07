"""TSCFEval - Time Series Counterfactual Evaluation.

A model-agnostic framework for systematic evaluation of counterfactual
explanations in time series classification.

This package provides:

- **Counterfactual Explainers** (:mod:`tscf_eval.counterfactuals`): Implementations
  of seven counterfactual explanation methods for time series: CELS, CoMTE,
  NativeGuide, SETS, TSEvo, Glacier, and LatentCF.

- **Evaluation Metrics** (:mod:`tscf_eval.evaluator`): 11 metrics organized
  into six quality dimensions for assessing counterfactual quality (validity,
  proximity, sparsity, plausibility, diversity, contiguity, composition,
  controllability, confidence, robustness, efficiency).

- **Data Loaders** (:mod:`tscf_eval.data_loader`): Utilities for loading time
  series classification datasets, including the UCR archive.

- **Benchmarking** (:mod:`tscf_eval.benchmark`): Framework for systematic
  evaluation across datasets, models, and explainers, with Pareto analysis,
  weighted scalarization, and Friedman tests for multi-criteria comparison.

Quick Start
-----------
>>> from tscf_eval.evaluator import Evaluator, Validity, Proximity, Sparsity
>>> from tscf_eval.counterfactuals import COMTE, NativeGuide
>>> from tscf_eval.data_loader import UCRLoader
>>>
>>> # Load data
>>> loader = UCRLoader("ItalyPowerDemand")
>>> train_data = loader.load("train")
>>>
>>> # Evaluate counterfactuals
>>> evaluator = Evaluator([Validity(), Proximity(), Sparsity()])
>>> results = evaluator.evaluate(X, X_cf, model=clf)

References
----------
The counterfactual methods in this package are based on the following papers:

- **CELS**: Li, P., Tang, B., & Ning, Y. (2023).
  CELS: Counterfactual Explanation of Time-Series via Learned Saliency Maps.
  IEEE International Conference on Big Data 2023, pp. 1952-1957.
  Original implementation: https://github.com/Luckilyeee/CELS

- **CoMTE**: Ates, E., Aksar, B., Leung, V. J., & Coskun, A. K. (2021).
  Counterfactual Explanations for Multivariate Time Series. ICAPAI 2021.
  Original implementation: https://github.com/peaclab/CoMTE

- **NativeGuide**: Delaney, E., Greene, D., & Keane, M. T. (2021).
  Instance-based Counterfactual Explanations for Time Series Classification.
  ICCBR 2021. Original implementation: https://github.com/e-delaney/Instance-Based_CFE_TSC

- **SETS**: Bahri, O., Filali Boubrahimi, S., & Hamdi, S. M. (2022).
  Shapelet-Based Counterfactual Explanations for Multivariate Time Series.
  KDD-MiLeTS 2022. Original implementation: https://github.com/omarbahri/SETS

- **TSEvo**: Höllig, J., Kulbach, C., & Thoma, S. (2022).
  TSEvo: Evolutionary Counterfactual Explanations for Time Series Classification.
  ICMLA 2022. Original implementation: https://github.com/JHoelli/TSEvo

- **Glacier**: Wang, Z., Samsten, I., Miliou, I., Mochaourab, R., & Papapetrou, P. (2024).
  Glacier: Guided Locally Constrained Counterfactual Explanations for Time Series
  Classification. Machine Learning, 113(3).
  Original implementation: https://github.com/zhendong3wang/learning-time-series-counterfactuals

- **LatentCF++**: Wang, Z., Samsten, I., Mochaourab, R., & Papapetrou, P. (2021).
  Learning Time Series Counterfactuals via Latent Space Representations.
  International Conference on Discovery Science (DS 2021), pp. 369-384.
  Original implementation: https://github.com/stellagerantoni/LatentCfMultivariate
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Bruna Zamith Santos"
__email__ = "bruna.zamith@hotmail.com"

# Re-export main components for convenient access
from tscf_eval.benchmark import (
    BenchmarkResults,
    BenchmarkRunner,
    DatasetConfig,
    ExplainerConfig,
    ExplainerResult,
    FriedmanResult,
    ModelConfig,
    ParetoAnalyzer,
    WeightedScalarizer,
    format_latex_table,
    friedman_test,
)
from tscf_eval.counterfactuals import (
    CELS,
    COMTE,
    SETS,
    Counterfactual,
    Glacier,
    LatentCF,
    NativeGuide,
    TSEvo,
)
from tscf_eval.data_loader import DataLoader, FileLoader, TSCData, UCRLoader
from tscf_eval.evaluator import (
    Composition,
    Confidence,
    Contiguity,
    Controllability,
    Diversity,
    Efficiency,
    Evaluator,
    Metric,
    Plausibility,
    Proximity,
    Robustness,
    Sparsity,
    Validity,
)

__all__ = [
    "CELS",
    "COMTE",
    "SETS",
    "BenchmarkResults",
    "BenchmarkRunner",
    "Composition",
    "Confidence",
    "Contiguity",
    "Controllability",
    "Counterfactual",
    "DataLoader",
    "DatasetConfig",
    "Diversity",
    "Efficiency",
    "Evaluator",
    "ExplainerConfig",
    "ExplainerResult",
    "FileLoader",
    "FriedmanResult",
    "Glacier",
    "LatentCF",
    "Metric",
    "ModelConfig",
    "NativeGuide",
    "ParetoAnalyzer",
    "Plausibility",
    "Proximity",
    "Robustness",
    "Sparsity",
    "TSCData",
    "TSEvo",
    "UCRLoader",
    "Validity",
    "WeightedScalarizer",
    "__author__",
    "__email__",
    "__version__",
    "format_latex_table",
    "friedman_test",
]
