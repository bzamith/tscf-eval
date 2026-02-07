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
import warnings

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
        X, X_cf = self._validate_inputs(X, X_cf, **kwargs)
        self._cache_model_predictions(X, X_cf, kwargs)

        results: dict[str, Any] = {}
        tqdm = self._get_progress_bar()

        total_metrics = len(self.metrics)
        pbar = tqdm(total=total_metrics, desc="Evaluating metrics", unit="metric", leave=False)
        start_time = time.time()
        try:
            for metric in self.metrics:
                try:
                    result = metric.compute(X, X_cf, **kwargs)
                except TypeError as exc:
                    with contextlib.suppress(Exception):
                        pbar.close()
                    raise TypeError(
                        f"Metric '{metric.name()}' raised TypeError when called "
                        f"with evaluator kwargs: {exc}"
                    ) from exc

                if isinstance(result, dict):
                    for key, value in result.items():
                        results[key] = value
                else:
                    results[metric.name()] = result

                with contextlib.suppress(Exception):
                    pbar.update(1)
        finally:
            with contextlib.suppress(Exception):
                pbar.close()
        elapsed = time.time() - start_time
        results["_evaluator_time_s"] = float(elapsed)
        return results

    def _validate_inputs(
        self, X: np.ndarray, X_cf: np.ndarray, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and coerce evaluation inputs.

        Parameters
        ----------
        X : np.ndarray
            Original instances.
        X_cf : np.ndarray
            Counterfactual instances.
        **kwargs
            Evaluator kwargs (checked for ``time_per_instance``).

        Returns
        -------
        tuple of np.ndarray
            Validated ``(X, X_cf)`` as numpy arrays.

        Raises
        ------
        ValueError
            If shapes mismatch or ``time_per_instance`` is provided
            without an ``Efficiency`` metric.
        """
        X = np.asarray(X)
        X_cf = np.asarray(X_cf)
        if X.shape[0] != X_cf.shape[0]:
            raise ValueError(
                "X and X_cf must have the same number of instances; "
                f"got {X.shape[0]} vs {X_cf.shape[0]}"
            )
        if "time_per_instance" in kwargs and "efficiency_time_s" not in self._metric_names:
            raise ValueError(
                "Caller provided 'time_per_instance' but evaluator does not contain "
                "an Efficiency metric. Please include an Efficiency() metric to report "
                "timings."
            )
        return X, X_cf

    @staticmethod
    def _cache_model_predictions(X: np.ndarray, X_cf: np.ndarray, kwargs: dict[str, Any]) -> None:
        """Pre-compute and cache model predictions in *kwargs* (mutated in-place).

        Caches ``_cached_y_pred``, ``_cached_y_cf_pred``, ``_cached_proba_X``,
        and ``_cached_proba_X_cf`` so that multiple metrics can reuse them
        without redundant inference calls.

        Parameters
        ----------
        X : np.ndarray
            Original instances.
        X_cf : np.ndarray
            Counterfactual instances.
        kwargs : dict
            Evaluator kwargs dict (modified in-place with cached predictions).
        """
        model = kwargs.get("model")
        if model is None:
            return
        if hasattr(model, "predict"):
            kwargs["_cached_y_pred"] = np.asarray(model.predict(X))
            kwargs["_cached_y_cf_pred"] = np.asarray(model.predict(X_cf))
        if hasattr(model, "predict_proba"):
            from tscf_eval.counterfactuals.utils import soft_predict_proba_fn

            soft_proba = soft_predict_proba_fn(model)
            kwargs["_cached_proba_X"] = np.asarray(soft_proba(X))
            kwargs["_cached_proba_X_cf"] = np.asarray(soft_proba(X_cf))

    @staticmethod
    def _get_progress_bar():
        """Return tqdm progress bar constructor, or a no-op fallback.

        Returns
        -------
        callable
            A tqdm-compatible constructor.
        """
        try:
            from tqdm.auto import tqdm  # type: ignore

            return tqdm
        except Exception:
            warnings.warn(
                "tqdm is not installed. Progress bars are disabled. "
                "Install tqdm for progress reporting: pip install tqdm",
                UserWarning,
                stacklevel=2,
            )

            class _DummyTqdm:
                """No-op progress bar used when tqdm is not installed."""

                def __init__(
                    self, total: int = 0, desc: str = "", unit: str = "", leave: bool = True
                ):
                    """Initialize the dummy progress bar.

                    Parameters
                    ----------
                    total : int, default 0
                        Total number of expected iterations.
                    desc : str, default ""
                        Description prefix for the progress bar.
                    unit : str, default ""
                        Unit name for each iteration.
                    leave : bool, default True
                        Whether to leave the bar on screen after completion.
                    """
                    self.total = total
                    self.desc = desc
                    self.n = 0
                    self.leave = leave

                def set_description(self, desc: str):
                    """Update the progress bar description (no-op).

                    Parameters
                    ----------
                    desc : str
                        New description string.
                    """
                    self.desc = desc

                def update(self, n: int = 1):
                    """Advance the progress counter (no-op).

                    Parameters
                    ----------
                    n : int, default 1
                        Number of iterations to advance.
                    """
                    self.n += n

                def close(self):
                    """Close the progress bar (no-op)."""
                    return None

            def _tqdm_fallback(
                total: int = 0, desc: str = "", unit: str = "", leave: bool = True
            ) -> _DummyTqdm:
                """Create a dummy progress bar as a tqdm replacement.

                Parameters
                ----------
                total : int, default 0
                    Total number of expected iterations.
                desc : str, default ""
                    Description prefix for the progress bar.
                unit : str, default ""
                    Unit name for each iteration.
                leave : bool, default True
                    Whether to leave the bar on screen after completion.

                Returns
                -------
                _DummyTqdm
                    A no-op progress bar instance.
                """
                return _DummyTqdm(total=total, desc=desc, unit=unit, leave=leave)

            return _tqdm_fallback
