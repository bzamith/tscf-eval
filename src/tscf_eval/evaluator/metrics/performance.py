"""Performance-based counterfactual evaluation metrics.

This module provides the Efficiency metric that reports timing statistics
for counterfactual generation.

Computational efficiency is an important consideration when deploying
counterfactual explanation methods in practice, especially for real-time
applications or large-scale evaluations.

Classes
-------
Efficiency
    Reports mean per-instance timing summary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..base import Metric

if TYPE_CHECKING:
    from collections.abc import Iterable


class Efficiency(Metric):
    """Mean per-instance timing summary.

    This metric reports the average elapsed time per instance (in seconds)
    as provided by the caller via the ``time_per_instance`` iterable. The
    metric itself does not inspect ``X`` or ``X_cf`` but follows the
    ``Metric.compute`` signature for compatibility with ``Evaluator``.

    See Li et al. (2023) for details.
    """

    direction = "minimize"

    def name(self) -> str:
        """Return the metric name.

        Returns
        -------
        str
            ``'efficiency_time_s'``.
        """
        return "efficiency_time_s"

    def compute(
        self,
        X: np.ndarray,
        X_cf: np.ndarray,
        time_per_instance: Iterable[float] | None = None,
        **kwargs,
    ) -> float:
        """Compute mean elapsed time per instance.

        Parameters
        ----------
        X : np.ndarray
            Original instances. Present for API compatibility but not used.
        X_cf : np.ndarray
            Counterfactual instances. Present for API compatibility but
            not used.
        time_per_instance : iterable of float, optional
            Iterable of elapsed times (seconds) for each produced
            counterfactual instance. Can be a list, generator, or any
            iterable. If omitted, the metric returns 0.0.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Mean elapsed time per instance in seconds, or 0.0 when no
            timings are provided.
        """
        if time_per_instance is not None:
            times = np.asarray(list(time_per_instance), dtype=float)
            return float(np.mean(times)) if times.size else 0.0
        return 0.0
