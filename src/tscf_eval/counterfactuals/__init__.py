"""Counterfactual explainers for time-series classification.

This package provides implementations of counterfactual explanation algorithms
for time-series classifiers. Counterfactuals answer the question: "What minimal
change to this time series would cause the classifier to predict a different class?"

Available Explainers
--------------------
COMTE
    Counterfactual Multivariate Time-series Explanations. Uses greedy channel
    substitution from distractor series (Ates et al., 2021).
NativeGuide
    Instance-based counterfactual explanations using nearest-unlike-neighbor
    guidance with DTW barycenter averaging (Delaney et al., 2021).
TSEvo
    Evolutionary counterfactual explanations using multi-objective optimization
    (NSGA-II) with three mutation strategies (Höllig et al., 2022).
Glacier
    Gradient-based counterfactual explanations with guided locally constrained
    optimization (Wang et al., 2024).
LatentCF
    Latent space counterfactual explanations using gradient-based optimization
    with importance-weighted proximity constraints (Wang et al., 2021).

Base Classes
------------
Counterfactual
    Abstract base class defining the explainer interface. All explainers
    implement ``explain(x, y_pred) -> (cf, cf_label, meta)``.

Examples
--------
>>> from tscf_eval.counterfactuals import COMTE, NativeGuide, TSEvo, Glacier, LatentCF
>>> import numpy as np
>>>
>>> # Assume clf is a trained classifier and (X_train, y_train) is training data
>>> comte = COMTE(model=clf, data=(X_train, y_train), distance="dtw")
>>> cf, cf_label, meta = comte.explain(x_test)
>>>
>>> ng = NativeGuide(model=clf, data=(X_train, y_train), method="blend")
>>> cf, cf_label, meta = ng.explain(x_test)
>>>
>>> tsevo = TSEvo(model=clf, data=(X_train, y_train), transformer="authentic")
>>> cf, cf_label, meta = tsevo.explain(x_test)
>>>
>>> glacier = Glacier(model=clf, data=(X_train, y_train), weight_type="uniform")
>>> cf, cf_label, meta = glacier.explain(x_test)
>>>
>>> latent_cf = LatentCF(model=clf, data=(X_train, y_train), step_weights="uniform")
>>> cf, cf_label, meta = latent_cf.explain(x_test)

See Also
--------
tscf_eval.evaluator : Metrics for evaluating counterfactual quality.
"""

from __future__ import annotations

from . import utils
from .base import Counterfactual
from .comte import COMTE
from .glacier import Glacier
from .latent_cf import LatentCF
from .native_guide import NativeGuide
from .tsevo import TSEvo

# Public API
__all__: list[str] = [
    "COMTE",
    "Counterfactual",
    "Glacier",
    "LatentCF",
    "NativeGuide",
    "TSEvo",
    "utils",
]
