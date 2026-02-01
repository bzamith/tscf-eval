"""Base interface for counterfactual explainers.

This module defines the abstract :class:`Counterfactual` class which other
explainers should subclass. The single required method is ``explain`` which
produces a counterfactual for a single query instance.

Classes
-------
Counterfactual
    Abstract base class for counterfactual explainers. Defines the standard
    interface: ``explain(x, y_pred) -> (counterfactual, label, metadata)``.

Notes
-----
The ``explain`` method operates on single instances, not batches. This design
allows explainers to maintain instance-specific state and metadata during
generation. For batch processing, wrap calls in a loop or use parallel
execution.

The ``explain_k`` method generates multiple diverse counterfactuals for a
single query. The default implementation calls ``explain`` multiple times
with random restarts, but subclasses can override for more sophisticated
diversity mechanisms.

Examples
--------
Creating a custom explainer:

>>> from tscf_eval.counterfactuals.base import Counterfactual
>>> import numpy as np
>>>
>>> class SimpleExplainer(Counterfactual):
...     def __init__(self, model):
...         self.model = model
...
...     def explain(self, x, y_pred=None):
...         # Simple example: perturb the series slightly
...         cf = x + np.random.randn(*x.shape) * 0.1
...         cf_label = int(self.model.predict(cf[None, ...])[0])
...         meta = {"method": "random_perturbation"}
...         return cf, cf_label, meta

See Also
--------
tscf_eval.counterfactuals.COMTE : CoMTE algorithm implementation.
tscf_eval.counterfactuals.NativeGuide : NativeGuide algorithm implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Counterfactual(ABC):
    """Minimal base interface for counterfactual explainers.

    Subclasses must implement ``explain``. The method operates on a single
    instance (not batches) and returns the generated counterfactual, its
    predicted label, and an optional metadata dictionary describing how the
    counterfactual was produced.
    """

    @abstractmethod
    def explain(
        self, x: np.ndarray, y_pred: int | None = None
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        """Return a counterfactual for a single instance `x`.

        Parameters
        ----------
        x
            A single time-series instance. Supported shapes include ``(T,)``,
            ``(1, T)`` or ``(1, 1, T)`` for compatibility with callers that may
            add a leading batch or channel dimension.
        y_pred
            Optional precomputed predicted label for ``x``. If ``None``, the
            explainer implementation may compute the prediction from its
            internally-held model.

        Returns
        -------
        cf_x
            Counterfactual series (shape ``(T,)`` or matching input format).
        cf_label
            Predicted label for the counterfactual.
        meta
            Metadata dictionary with information about the generation
            process (e.g., neighbor indices, distances, edits, timings).
        """
        raise NotImplementedError()

    def explain_k(
        self,
        x: np.ndarray,
        k: int = 5,
        y_pred: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Generate k diverse counterfactuals for a single instance.

        The default implementation calls ``explain`` k times. Subclasses may
        override this method to implement more sophisticated diversity
        mechanisms (e.g., different random seeds, target classes, or
        optimization restarts).

        Parameters
        ----------
        x : np.ndarray
            A single time-series instance.
        k : int, default 5
            Number of counterfactuals to generate.
        y_pred : int, optional
            Optional precomputed predicted label for ``x``.

        Returns
        -------
        cfs : np.ndarray
            Array of counterfactuals with shape ``(k, ...)``, where ``...``
            matches the shape of the input ``x``.
        cf_labels : np.ndarray
            Array of predicted labels for each counterfactual, shape ``(k,)``.
        metas : list[dict]
            List of k metadata dictionaries.

        Examples
        --------
        >>> cfs, labels, metas = explainer.explain_k(x, k=5)
        >>> cfs.shape  # (5, T) for univariate or (5, C, T) for multivariate
        """
        cfs = []
        cf_labels = []
        metas = []

        for i in range(k):
            cf, label, meta = self.explain(x, y_pred=y_pred)
            meta["k_index"] = i
            cfs.append(cf)
            cf_labels.append(label)
            metas.append(meta)

        return np.array(cfs), np.array(cf_labels), metas
