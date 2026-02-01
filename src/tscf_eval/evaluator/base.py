"""Abstract base class for evaluation metrics.

This module defines the :class:`Metric` abstract base class that all
evaluation metrics must implement.

Classes
-------
Metric
    Abstract base class for evaluation metrics. Subclasses must implement
    ``name()`` and ``compute(X, X_cf, **kwargs)``.

Creating Custom Metrics
-----------------------
To create a custom metric, subclass :class:`Metric` and implement two methods:

1. ``name()`` - Return a short, stable string identifier for the metric
2. ``compute(X, X_cf, **kwargs)`` - Compute and return the metric value

Example custom metric:

>>> from tscf_eval.evaluator.base import Metric
>>> import numpy as np
>>>
>>> class MeanAbsoluteChange(Metric):
...     def name(self) -> str:
...         return "mean_absolute_change"
...
...     def compute(self, X, X_cf, **kwargs) -> float:
...         return float(np.mean(np.abs(X - X_cf)))

See Also
--------
tscf_eval.evaluator.metrics : Built-in metric implementations.
tscf_eval.evaluator.evaluator : Evaluator orchestration class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np


class Metric(ABC):
    """Abstract base for a single evaluation metric.

    Subclasses must implement ``name`` and ``compute``.

    The ``compute`` method receives two arrays, ``X`` and ``X_cf`` (original
    instances and counterfactuals) and may accept additional keyword
    arguments such as ``model`` or ``X_train`` depending on the metric's
    needs.
    """

    direction: Literal["minimize", "maximize"] = "minimize"  # Default; subclasses override

    @abstractmethod
    def name(self) -> str:
        """Return a short, stable metric name used as the result key.

        Returns
        -------
        str
            Metric identifier used as the key in evaluation results.
        """
        raise NotImplementedError()

    @abstractmethod
    def compute(self, X: np.ndarray, X_cf: np.ndarray, **kwargs) -> Any:
        """Compute the metric.

        Parameters
        ----------
        X : np.ndarray
            Original instances, shape ``(M, ...)``.
        X_cf : np.ndarray
            Counterfactual instances, shape matching ``X``.
        **kwargs
            Optional metric-specific keyword arguments (e.g., ``model``,
            ``X_train``, ``y``, ``y_cf``).

        Returns
        -------
        Any
            Metric result (scalar, array, or mapping).
        """
