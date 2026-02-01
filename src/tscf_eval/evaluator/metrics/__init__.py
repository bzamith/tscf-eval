"""Concrete metric implementations for counterfactual evaluation.

This module provides a set of ``Metric`` subclasses that implement common
counterfactual-quality measures organized by category:

- **Core metrics**: Validity, Proximity, Sparsity
- **Distribution metrics**: Plausibility, Diversity
- **Structure metrics**: Composition, Contiguity
- **Model metrics**: Confidence, Controllability
- **Stability metrics**: Robustness
- **Performance metrics**: Efficiency

Shape expectations
------------------
Most metrics expect ``X`` and ``X_cf`` to be NumPy arrays with matching
first-axis length (number of instances). Each individual instance may be
univariate (``(T,)``) or multivariate (``(C, T)``) and metrics typically
flatten per-instance data when computing distances or norms.

References
----------
- ATES, E.; AKSAR, B.; LEUNG, V. J.; COSKUN, A. K. Counterfactual explanations
  for multivariate time series. ICAPAI 2021.
- DELANEY, E.; GREENE, D.; KEANE, M. T. Instance-based counterfactual
  explanations for time series classification. ICCBR 2021.
- BOUBRAHIMI, S. F.; HAMDI, S. M. On the mining of time series data
  counterfactual explanations using barycenters. CIKM 2022.
- BAHRI, O.; BOUBRAHIMI, S. F.; HAMDI, S. M. Shapelet-based counterfactual
  explanations for multivariate time series. KDD-MiLeTS 2022.
- LI, P.; BAHRI, O.; BOUBRAHIMI, S. F.; HAMDI, S. M. Attention-based
  counterfactual explanation for multivariate time series. DaWaK 2023.
- LI, P.; BAHRI, O.; BOUBRAHIMI, S. F.; HAMDI, S. M. CELS: Counterfactual
  explanations for time series data via learned saliency maps. BigData 2023.
- LI, P.; BOUBRAHIMI, S. F.; HAMDI, S. M. Motif-guided time series
  counterfactual explanations. ICPR 2022.
- REFOYO, M.; LUENGO, D. Sub-space: Subsequence-based sparse counterfactual
  explanations for time series classification problems. XAI 2026.
"""

from .core import Proximity, Sparsity, Validity
from .distribution import Diversity, Plausibility
from .model import Confidence, Controllability
from .performance import Efficiency
from .stability import Robustness
from .structure import Composition, Contiguity

__all__ = [
    "Composition",
    "Confidence",
    "Contiguity",
    "Controllability",
    "Diversity",
    "Efficiency",
    "Plausibility",
    "Proximity",
    "Robustness",
    "Sparsity",
    "Validity",
]
