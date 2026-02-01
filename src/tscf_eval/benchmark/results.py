"""Result containers for benchmark outputs.

This module provides dataclasses for storing and analyzing benchmark results
across multiple datasets, models, and explainers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

import numpy as np
import pandas as pd

__all__ = [
    "BenchmarkResults",
    "ExplainerResult",
]


@dataclass
class ExplainerResult:
    """Results for a single explainer on a single dataset/model combination.

    Parameters
    ----------
    explainer_name : str
        Name of the explainer configuration.
    dataset_name : str
        Name of the dataset.
    model_name : str
        Name of the model.
    X_cf : np.ndarray
        Generated counterfactuals, shape ``(n_instances, ...)`` or
        ``(n_instances, k, ...)`` if k > 1.
    y_cf : np.ndarray
        Predicted labels for counterfactuals.
    success_mask : np.ndarray
        Boolean mask indicating successful generations.
    metrics : dict[str, Any]
        Evaluation metrics computed by the Evaluator.
    generation_times : list[float]
        Per-instance generation times in seconds.
    metadata : list[dict]
        Per-instance metadata from the explainer.
    """

    explainer_name: str
    dataset_name: str
    model_name: str
    X_cf: np.ndarray
    y_cf: np.ndarray
    success_mask: np.ndarray
    metrics: dict[str, Any]
    generation_times: list[float]
    metadata: list[dict[str, Any]]

    @property
    def n_instances(self) -> int:
        """Number of test instances."""
        return len(self.X_cf)

    @property
    def n_successful(self) -> int:
        """Number of successfully generated counterfactuals."""
        return int(np.sum(self.success_mask))

    @property
    def success_rate(self) -> float:
        """Fraction of successful generations."""
        return self.n_successful / self.n_instances if self.n_instances > 0 else 0.0

    @property
    def mean_time(self) -> float:
        """Mean generation time per instance in seconds."""
        if not self.generation_times:
            return 0.0
        return float(np.mean(self.generation_times))

    @property
    def total_time(self) -> float:
        """Total generation time in seconds."""
        return float(np.sum(self.generation_times))

    def get_metric(self, name: str, default: Any = None) -> Any:
        """Get a metric value by name, with optional default."""
        return self.metrics.get(name, default)


@dataclass
class BenchmarkResults:
    """Container for all benchmark results with analysis methods.

    Stores results indexed by (dataset, model, explainer) combinations.
    Provides methods for querying, aggregating, and exporting results.

    Examples
    --------
    >>> results = runner.run()
    >>>
    >>> # Get specific result
    >>> result = results.get("GunPoint", "knn", "comte")
    >>>
    >>> # Get comparison DataFrame
    >>> df = results.to_dataframe()
    >>>
    >>> # Iterate over results
    >>> for result in results:
    ...     print(f"{result.dataset_name}/{result.model_name}: {result.metrics}")
    """

    _results: dict[tuple[str, str, str], ExplainerResult] = field(default_factory=dict)

    def add(self, result: ExplainerResult) -> None:
        """Add a result to the collection."""
        key = (result.dataset_name, result.model_name, result.explainer_name)
        self._results[key] = result

    def get(
        self,
        dataset: str,
        model: str,
        explainer: str,
    ) -> ExplainerResult | None:
        """Get result for a specific combination."""
        return self._results.get((dataset, model, explainer))

    def __iter__(self) -> Iterator[ExplainerResult]:
        """Iterate over all results."""
        return iter(self._results.values())

    def __len__(self) -> int:
        """Number of results."""
        return len(self._results)

    @property
    def datasets(self) -> list[str]:
        """List of unique dataset names."""
        return sorted({k[0] for k in self._results})

    @property
    def models(self) -> list[str]:
        """List of unique model names."""
        return sorted({k[1] for k in self._results})

    @property
    def explainers(self) -> list[str]:
        """List of unique explainer names."""
        return sorted({k[2] for k in self._results})

    def filter(
        self,
        datasets: list[str] | None = None,
        models: list[str] | None = None,
        explainers: list[str] | None = None,
    ) -> BenchmarkResults:
        """Create a filtered copy of results.

        Parameters
        ----------
        datasets : list[str], optional
            Filter to these datasets. None means all.
        models : list[str], optional
            Filter to these models. None means all.
        explainers : list[str], optional
            Filter to these explainers. None means all.

        Returns
        -------
        BenchmarkResults
            New results containing only matching entries.
        """
        filtered = BenchmarkResults()
        for (ds, mdl, exp), result in self._results.items():
            if datasets is not None and ds not in datasets:
                continue
            if models is not None and mdl not in models:
                continue
            if explainers is not None and exp not in explainers:
                continue
            filtered.add(result)
        return filtered

    def to_dataframe(
        self,
        metrics: list[str] | None = None,
        include_timing: bool = True,
    ) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.

        Parameters
        ----------
        metrics : list[str], optional
            Specific metrics to include. None means all available.
        include_timing : bool, default True
            Include timing columns (mean_time, total_time).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for dataset, model, explainer, and metrics.
        """
        rows = []
        for result in self:
            row = {
                "dataset": result.dataset_name,
                "model": result.model_name,
                "explainer": result.explainer_name,
                "n_instances": result.n_instances,
                "n_successful": result.n_successful,
                "success_rate": result.success_rate,
            }

            if include_timing:
                row["mean_time_s"] = result.mean_time
                row["total_time_s"] = result.total_time

            # Add metrics
            # Note: The evaluator already flattens dict results from metrics like
            # Confidence, so we don't need to re-flatten here. We keep the dict
            # check only for backwards compatibility with older saved results.
            for name, value in result.metrics.items():
                if name.startswith("_"):
                    continue
                if metrics is not None and name not in metrics:
                    continue
                # Skip dict values that have already been flattened by the evaluator
                # (detected by checking if individual keys already exist)
                if isinstance(value, dict):
                    # Check if this dict was already flattened
                    first_key = next(iter(value.keys()), None)
                    if first_key is not None and first_key in result.metrics:
                        # Already flattened, skip this nested entry
                        continue
                    # Legacy: flatten for backwards compatibility
                    for sub_name, sub_value in value.items():
                        row[f"{name}_{sub_name}"] = sub_value
                else:
                    row[name] = value

            rows.append(row)

        return pd.DataFrame(rows)

    def aggregate(
        self,
        by: str = "explainer",
        metrics: list[str] | None = None,
        aggfunc: str | list[str] = "mean",
    ) -> pd.DataFrame:
        """Aggregate metrics across a dimension.

        Parameters
        ----------
        by : str, default "explainer"
            Dimension to group by: "explainer", "model", or "dataset".
        metrics : list[str], optional
            Metrics to aggregate. None means all numeric.
        aggfunc : str or list[str], default "mean"
            Aggregation function(s): "mean", "median", "std", "min", "max".
            When a list is provided (e.g. ``["mean", "std"]``), the returned
            DataFrame has a ``MultiIndex`` on columns with levels
            ``(metric, aggfunc)``.

        Returns
        -------
        pd.DataFrame
            Aggregated results.
        """
        df = self.to_dataframe(metrics=metrics)

        if df.empty:
            return df

        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["n_instances"]]

        return df.groupby(by)[numeric_cols].agg(aggfunc).reset_index()

    def summary(self) -> pd.DataFrame:
        """Get summary statistics aggregated by explainer.

        Returns
        -------
        pd.DataFrame
            Summary with mean metrics per explainer across all datasets/models.
        """
        return self.aggregate(by="explainer", aggfunc="mean")

    def to_dict(self) -> dict[str, Any]:
        """Convert to nested dictionary for serialization."""
        return {
            "datasets": self.datasets,
            "models": self.models,
            "explainers": self.explainers,
            "results": [
                {
                    "dataset": r.dataset_name,
                    "model": r.model_name,
                    "explainer": r.explainer_name,
                    "n_instances": r.n_instances,
                    "n_successful": r.n_successful,
                    "metrics": r.metrics,
                    "mean_time_s": r.mean_time,
                }
                for r in self
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResults:
        """Reconstruct from a dictionary produced by :meth:`to_dict`.

        Only metrics and metadata are restored; counterfactual arrays
        (``X_cf``, ``y_cf``) are not stored by ``to_dict`` and will be
        set to empty arrays.

        Parameters
        ----------
        data : dict
            Dictionary as returned by ``to_dict`` or loaded from JSON.

        Returns
        -------
        BenchmarkResults
        """
        results = cls()
        for entry in data.get("results", []):
            n = entry.get("n_instances", 0)
            n_ok = entry.get("n_successful", n)
            mean_time = entry.get("mean_time_s", 0.0)

            mask = np.zeros(n, dtype=bool)
            mask[:n_ok] = True

            results.add(
                ExplainerResult(
                    explainer_name=entry["explainer"],
                    dataset_name=entry["dataset"],
                    model_name=entry["model"],
                    X_cf=np.empty((n, 0)),
                    y_cf=np.empty(n),
                    success_mask=mask,
                    metrics=entry.get("metrics", {}),
                    generation_times=[mean_time] * n if n > 0 else [],
                    metadata=[{}] * n,
                )
            )
        return results
