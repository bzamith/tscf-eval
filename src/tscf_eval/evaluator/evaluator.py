"""Evaluator orchestration class.

This module provides the :class:`Evaluator` class that orchestrates the
computation of multiple metrics over pairs of original instances and their
counterfactuals.

Classes
-------
Evaluator
    Orchestration class that runs a collection of metrics and returns
    results as a dictionary mapping metric names to computed values.

Example
-------
>>> from tscf_eval.evaluator import Evaluator, Validity, Proximity, Sparsity
>>> import numpy as np
>>>
>>> # Create evaluator with multiple metrics
>>> evaluator = Evaluator([Validity(), Proximity(p=2), Sparsity()])
>>>
>>> # Evaluate counterfactuals
>>> X = np.random.randn(100, 50)
>>> X_cf = X + np.random.randn(100, 50) * 0.1
>>> results = evaluator.evaluate(X, X_cf, y=np.zeros(100), y_cf=np.ones(100))

See Also
--------
tscf_eval.evaluator.base : Metric abstract base class.
tscf_eval.evaluator.metrics : Built-in metric implementations.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .base import Metric


class Evaluator:
    """Run a collection of :class:`Metric` instances over example pairs.

    The Evaluator orchestrates the computation of multiple metrics over
    pairs of original instances and their counterfactuals. It handles
    progress reporting, error handling, and result aggregation.

    Parameters
    ----------
    metrics : iterable of Metric
        Collection of metric instances to compute during evaluation.

    Attributes
    ----------
    metrics : list of Metric
        The configured metric instances.

    Examples
    --------
    >>> from tscf_eval.evaluator import Evaluator, Validity, Proximity, Sparsity
    >>> import numpy as np
    >>>
    >>> # Create evaluator with multiple metrics
    >>> evaluator = Evaluator([Validity(), Proximity(p=2), Sparsity()])
    >>>
    >>> # Evaluate counterfactuals
    >>> X = np.random.randn(100, 50)  # 100 instances, 50 time points
    >>> X_cf = X + np.random.randn(100, 50) * 0.1
    >>> results = evaluator.evaluate(X, X_cf, y=np.zeros(100), y_cf=np.ones(100))
    >>>
    >>> print(results['validity'], results['proximity_l2'], results['sparsity'])
    """

    def __init__(self, metrics: Iterable[Metric]):
        """Initialize the Evaluator with a collection of metrics.

        Parameters
        ----------
        metrics : iterable of Metric
            Metric instances to compute. Each metric must implement
            ``name()`` and ``compute(X, X_cf, **kwargs)``.
        """
        self.metrics: list[Metric] = list(metrics)
        # Cached names for quick validation
        self._metric_names = [m.name() for m in self.metrics]

    def evaluate(self, X: np.ndarray, X_cf: np.ndarray, **kwargs) -> dict[str, Any]:
        """Compute all configured metrics and return a mapping name -> result.

        The evaluator forwards all provided ``kwargs`` to each metric's
        ``compute`` method. To avoid silent behavior, if the caller provides
        ``time_per_instance`` then an ``Efficiency``-style metric must be
        present (i.e., a metric whose ``name()`` returns ``"efficiency_time_s"``)
        that will consume that argument and report a canonical value. This
        avoids the evaluator guessing at how to aggregate timings.

        Parameters
        ----------
        X : np.ndarray
            Original instances, shape ``(M, ...)``.
        X_cf : np.ndarray
            Counterfactual instances, shape matching ``X``.
        **kwargs
            Forwarded to each metric. Common kwargs include:

            - ``model``: Classifier for metrics like Validity, Controllability.
            - ``X_train``: Training data for Plausibility, Robustness.
            - ``y``, ``y_cf``: Labels for Validity when model not provided.
            - ``time_per_instance``: Timings for Efficiency metric.

        Returns
        -------
        dict
            Mapping from metric name to computed result. Also includes
            ``'_evaluator_time_s'`` with total evaluation time.

        Raises
        ------
        ValueError
            If ``X`` and ``X_cf`` have different numbers of instances, or
            if ``time_per_instance`` is provided without an Efficiency metric.
        TypeError
            If a metric raises TypeError due to unexpected kwargs.
        """
        # Minimal validation of inputs
        X = np.asarray(X)
        X_cf = np.asarray(X_cf)
        if X.shape[0] != X_cf.shape[0]:
            raise ValueError(
                "X and X_cf must have the same number of instances; "
                f"got {X.shape[0]} vs {X_cf.shape[0]}"
            )

        # Enforce explicit Efficiency metric when timings are provided
        if "time_per_instance" in kwargs and "efficiency_time_s" not in self._metric_names:
            raise ValueError(
                "Caller provided 'time_per_instance' but evaluator does not contain "
                "an Efficiency metric. Please include an Efficiency() metric to report "
                "timings."
            )

        # Pre-compute model predictions to avoid redundant calls across metrics
        # These cached predictions are passed as kwargs and consumed by metrics
        # that recognize them (Validity, Confidence, Controllability)
        model = kwargs.get("model")
        if model is not None:
            if hasattr(model, "predict"):
                kwargs["_cached_y_pred"] = np.asarray(model.predict(X))
                kwargs["_cached_y_cf_pred"] = np.asarray(model.predict(X_cf))
            if hasattr(model, "predict_proba"):
                # Use soft probabilities when available (e.g., ROCKET classifiers
                # return hard 0/1 from predict_proba, making Confidence meaningless).
                # soft_predict_proba_fn returns smooth sigmoid/softmax probabilities
                # for classifiers with decision_function, otherwise falls back to
                # the model's own predict_proba.
                from tscf_eval.counterfactuals.utils import soft_predict_proba_fn

                soft_proba = soft_predict_proba_fn(model)
                kwargs["_cached_proba_X"] = np.asarray(soft_proba(X))
                kwargs["_cached_proba_X_cf"] = np.asarray(soft_proba(X_cf))

        results: dict[str, Any] = {}

        # Try to import a notebook-friendly tqdm; otherwise use a tiny fallback
        try:
            from tqdm.auto import tqdm  # type: ignore
        except Exception:

            class _DummyTqdm:
                def __init__(
                    self, total: int = 0, desc: str = "", unit: str = "", leave: bool = True
                ):
                    self.total = total
                    self.desc = desc
                    self.n = 0
                    self.leave = leave

                def set_description(self, desc: str):
                    # No-op fallback; kept for API compatibility
                    self.desc = desc

                def update(self, n: int = 1):
                    self.n += n

                def close(self):
                    return None

            def tqdm(
                total: int = 0, desc: str = "", unit: str = "", leave: bool = True
            ) -> _DummyTqdm:
                return _DummyTqdm(total=total, desc=desc, unit=unit, leave=leave)

        total_metrics = len(self.metrics)
        pbar = tqdm(total=total_metrics, desc="Evaluating metrics", unit="metric", leave=False)
        start_time = time.time()
        try:
            for metric in self.metrics:
                # Call metric.compute; if it fails due to unexpected kwargs,
                # surface a clear error (fail-fast) so the user can adjust.
                try:
                    result = metric.compute(X, X_cf, **kwargs)
                except TypeError as exc:
                    # Close progress before raising to keep UI tidy
                    with contextlib.suppress(Exception):
                        pbar.close()
                    # Fail fast with a concise, single-line message.
                    raise TypeError(
                        f"Metric '{metric.name()}' raised TypeError when called "
                        f"with evaluator kwargs: {exc}"
                    ) from exc

                # Store result under metric's canonical name
                # If the metric returns a dict, flatten it with prefixed keys
                if isinstance(result, dict):
                    for key, value in result.items():
                        results[key] = value
                else:
                    results[metric.name()] = result

                # Update progress
                with contextlib.suppress(Exception):
                    pbar.update(1)
        finally:
            with contextlib.suppress(Exception):
                pbar.close()
        elapsed = time.time() - start_time
        results["_evaluator_time_s"] = float(elapsed)
        return results
