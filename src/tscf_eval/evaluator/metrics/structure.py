"""Structure-based counterfactual evaluation metrics.

This module provides metrics that analyze the structure of edits made to
create counterfactuals: Composition and Contiguity.

These metrics are particularly relevant for time series counterfactual
explanations, where the temporal structure of modifications matters for
interpretability. Contiguous edits are generally easier to understand
and more actionable than scattered changes.

Classes
-------
Composition
    Segment-based statistics measuring runs of edits.
Contiguity
    Measure how contiguous edits are.
"""

from __future__ import annotations

import numpy as np

from ..base import Metric
from ._utils import ensure_array


def _count_edit_runs(xi: np.ndarray, xfi: np.ndarray) -> tuple[int, list[int], int]:
    """Count contiguous runs of edits between an instance and its counterfactual.

    Parameters
    ----------
    xi : np.ndarray
        Original instance, shape ``(T,)`` or ``(C, T)``.
    xfi : np.ndarray
        Counterfactual instance, same shape as *xi*.

    Returns
    -------
    n_runs : int
        Number of contiguous edit segments.
    run_lengths : list of int
        Length of each contiguous edit segment.
    total_positions : int
        Total number of time positions in the (flattened) diff vector.
    """
    diff = (~np.isclose(xi, xfi)).astype(int)
    if diff.ndim == 2:
        diff = diff.sum(axis=0) > 0

    flat = np.asarray(diff).ravel()
    runs: list[int] = []
    in_run = False
    run_len = 0
    for v in flat:
        if v:
            if not in_run:
                in_run = True
                run_len = 1
            else:
                run_len += 1
        elif in_run:
            runs.append(run_len)
            in_run = False
    if in_run:
        runs.append(run_len)

    return len(runs), runs, flat.size


class Composition(Metric):
    """Segment-based composition statistics measuring runs of edits.

    Analyzes the structure of edits by counting contiguous segments (runs)
    of changed values and their lengths. In addition to the mean number of
    segments and the mean segment length, it reports a composition score:

    ``composition = mean_segment_length / mean_n_segments``.

    This favors counterfactuals with fewer, longer contiguous edits over
    those with many short, scattered modifications.

    See Delaney et al. (2021) and Ates et al. (2021) for details.
    """

    direction = "maximize"

    def name(self) -> str:
        """Return the metric name.

        Returns
        -------
        str
            ``'composition'``.
        """
        return "composition"

    def compute(self, X: np.ndarray, X_cf: np.ndarray, **kwargs) -> dict[str, float]:
        """Compute composition statistics.

        Parameters
        ----------
        X : np.ndarray
            Original instances of shape ``(M, ...)``.
        X_cf : np.ndarray
            Counterfactual instances of shape ``(M, ...)``.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        dict
            Dictionary with keys:

            - ``mean_n_segments``: Mean number of contiguous edit segments.
            - ``mean_avg_segment_len``: Mean average length of segments.
            - ``composition``: Ratio ``mean_avg_segment_len / mean_n_segments``.
        """
        X = ensure_array(X)
        X_cf = ensure_array(X_cf)
        M = X.shape[0]
        n_segments = []
        avg_seg_len = []
        for i in range(M):
            n_runs, run_lengths, _ = _count_edit_runs(X[i], X_cf[i])
            n_segments.append(n_runs)
            avg_seg_len.append(float(np.mean(run_lengths)) if run_lengths else 0.0)
        mean_n_segments = float(np.mean(n_segments) if n_segments else 0.0)
        mean_avg_segment_len = float(np.mean(avg_seg_len) if avg_seg_len else 0.0)
        composition = mean_avg_segment_len / mean_n_segments if mean_n_segments > 0.0 else 0.0
        return {
            "mean_n_segments": mean_n_segments,
            "mean_avg_segment_len": mean_avg_segment_len,
            "composition": composition,
        }


class Contiguity(Metric):
    """Measure how contiguous edits are (fewer runs = higher contiguity).

    Produces a scalar in ``[0, 1]`` where 1 indicates fully contiguous edits
    (all changes occur in a single uninterrupted segment).

    See Delaney et al. (2021) and Ates et al. (2021) for details.
    """

    direction = "maximize"  # Higher contiguity is better

    def name(self) -> str:
        """Return the metric name.

        Returns
        -------
        str
            ``'contiguity'``.
        """
        return "contiguity"

    def compute(self, X: np.ndarray, X_cf: np.ndarray, **kwargs) -> float:
        """Compute contiguity score.

        Parameters
        ----------
        X : np.ndarray
            Original instances of shape ``(M, ...)``.
        X_cf : np.ndarray
            Counterfactual instances of shape ``(M, ...)``.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        float
            Mean contiguity score in ``[0, 1]``. Higher values indicate
            more contiguous edits.
        """
        X = ensure_array(X)
        X_cf = ensure_array(X_cf)
        M = X.shape[0]
        scores = []
        for i in range(M):
            n_runs, _, total_positions = _count_edit_runs(X[i], X_cf[i])
            max_runs = total_positions if total_positions > 0 else 1
            scores.append(1.0 - n_runs / float(max_runs))
        return float(np.mean(scores) if scores else 0.0)
