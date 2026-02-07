"""Multi-criteria evaluation for counterfactual benchmarks.

This module provides tools for multi-objective analysis of benchmark
results, including Pareto dominance analysis, weighted scalarization,
statistical testing, and LaTeX table generation.

Classes
-------
ParetoAnalyzer
    Multi-objective Pareto dominance analysis with plotting support.
WeightedScalarizer
    Min-max normalized weighted composite scoring.

Functions
---------
friedman_test
    Non-parametric Friedman test across datasets.
format_latex_table
    Format a DataFrame as a LaTeX table with best-value highlighting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NamedTuple
import warnings

import numpy as np
import pandas as pd

from tscf_eval.evaluator.metrics import (
    Composition,
    Confidence,
    Contiguity,
    Controllability,
    Diversity,
    Efficiency,
    Plausibility,
    Proximity,
    Robustness,
    Sparsity,
    Validity,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from tscf_eval.evaluator.base import Metric

    from .results import BenchmarkResults

__all__ = [
    "FriedmanResult",
    "ParetoAnalyzer",
    "WeightedScalarizer",
    "format_latex_table",
    "friedman_test",
]


def _build_direction_registry() -> dict[str, bool]:
    """Build a metric-name → is_maximize mapping from Metric classes.

    Instantiates each built-in metric with default parameters and records
    its ``name()`` and ``direction``.  Also registers common variants of
    parameterized metrics (Proximity with p=1/2/inf, Plausibility with
    lof/if) and non-metric columns produced by the benchmark runner.
    """
    registry: dict[str, bool] = {}

    def _register(metric_instance: Metric) -> None:
        """Register a metric instance in the direction registry.

        Parameters
        ----------
        metric_instance : Metric
            Metric whose name and direction to record.
        """
        is_max = metric_instance.direction == "maximize"
        registry[metric_instance.name()] = is_max

    # Default instances
    for cls in (Validity, Sparsity, Diversity, Contiguity, Controllability, Robustness, Efficiency):
        _register(cls())

    # Parameterized: Validity (both modes)
    _register(Validity(mode="hard"))
    _register(Validity(mode="soft"))

    # Parameterized: Proximity (common norms + DTW)
    for p in (1, 2, float("inf")):
        _register(Proximity(p=p, distance="lp"))
    _register(Proximity(distance="dtw"))

    # Parameterized: Plausibility (common methods)
    for method in ("lof", "if", "dtw_lof"):
        _register(Plausibility(method=method))

    # Dict-returning metrics: register their flattened sub-keys
    _register(Composition())
    _register(Confidence())
    # Composition flattens to composition_mean_n_segments, etc.
    registry["composition_mean_n_segments"] = False
    registry["composition_mean_avg_segment_len"] = False
    # Confidence flattens to mean_conf_orig, mean_conf_cf, mean_conf_delta
    registry["mean_conf_orig"] = True
    registry["mean_conf_cf"] = True
    registry["mean_conf_delta"] = True

    # Non-metric benchmark columns
    registry["mean_time_s"] = False
    registry["total_time_s"] = False
    registry["success_rate"] = True

    return registry


_DIRECTION_REGISTRY: dict[str, bool] = _build_direction_registry()


def _is_maximize(metric: str) -> bool:
    """Check whether higher values are better for a given metric.

    Parameters
    ----------
    metric : str
        Metric name to look up.

    Returns
    -------
    bool
        ``True`` if the metric should be maximized, ``False`` otherwise.
    """
    if metric in _DIRECTION_REGISTRY:
        return _DIRECTION_REGISTRY[metric]

    # Prefix heuristic for unknown parameterized variants
    for prefix in ("proximity_", "plausibility_"):
        if metric.startswith(prefix):
            base_name = next(
                (k for k in _DIRECTION_REGISTRY if k.startswith(prefix)),
                None,
            )
            if base_name is not None:
                return _DIRECTION_REGISTRY[base_name]

    # Conservative default
    return False


class FriedmanResult(NamedTuple):
    """Result of a Friedman statistical test.

    Attributes
    ----------
    statistic : float
        Friedman chi-squared statistic.
    p_value : float
        p-value of the test.
    rankings : pd.DataFrame
        Mean ranks per explainer for each metric.
    """

    statistic: float
    p_value: float
    rankings: pd.DataFrame


def friedman_test(
    results: BenchmarkResults,
    metric: str,
    aggregate_by: str = "explainer",
    group_by: str = "dataset",
) -> FriedmanResult:
    """Run a Friedman test comparing explainers across groups.

    The Friedman test is a non-parametric test for detecting differences
    in treatments across multiple groups (e.g., explainers across datasets).

    Parameters
    ----------
    results : BenchmarkResults
        Benchmark results to analyze.
    metric : str
        Metric name to test.
    aggregate_by : str, default "explainer"
        Treatments to compare (columns of the rank matrix).
    group_by : str, default "dataset"
        Blocking factor (rows of the rank matrix).

    Returns
    -------
    FriedmanResult
        Named tuple with ``statistic``, ``p_value``, and ``rankings``.

    Raises
    ------
    ImportError
        If scipy is not installed.
    ValueError
        If there are fewer than 3 treatments or fewer than 2 groups.
    """
    try:
        from scipy.stats import friedmanchisquare
    except ImportError:
        raise ImportError(
            "scipy is required for friedman_test. Install it with: pip install tscf-eval[full]"
        ) from None

    df = results.to_dataframe()
    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' not found. "
            f"Available: {sorted(df.select_dtypes(include=[np.number]).columns)}"
        )

    # Pivot: rows = groups, columns = treatments
    pivot = df.pivot_table(
        values=metric,
        index=group_by,
        columns=aggregate_by,
        aggfunc="mean",
    ).dropna()

    treatments = list(pivot.columns)
    if len(treatments) < 3:
        raise ValueError(f"Friedman test requires at least 3 treatments, got {len(treatments)}.")
    if len(pivot) < 2:
        raise ValueError(f"Friedman test requires at least 2 groups, got {len(pivot)}.")

    samples = [pivot[t].values for t in treatments]
    stat, p_val = friedmanchisquare(*samples)

    maximize = _is_maximize(metric)
    ranks = pivot.rank(axis=1, ascending=not maximize)
    mean_ranks = ranks.mean(axis=0).to_frame(name="mean_rank")
    mean_ranks = mean_ranks.sort_values("mean_rank")

    return FriedmanResult(
        statistic=float(stat),
        p_value=float(p_val),
        rankings=mean_ranks,
    )


def format_latex_table(
    df: pd.DataFrame,
    directions: dict[str, bool] | None = None,
    bold_best: bool = True,
    arrows: bool = True,
    precision: int = 3,
    midrule_every: int = 0,
    escape_underscores: bool = True,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """Format a DataFrame as a LaTeX table with best-value highlighting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to format. First column is typically the entity name
        (explainer/model), remaining columns are metrics.
    directions : dict[str, bool], optional
        Mapping of column name to ``True`` if higher is better.
        If ``None``, uses the built-in metric direction registry.
    bold_best : bool, default True
        Bold the best value in each numeric column.
    arrows : bool, default True
        Append directional arrows to column headers.
    precision : int, default 3
        Number of decimal places for floats.
    midrule_every : int, default 0
        Insert ``\\midrule`` every N data rows.  0 means no midrules.
    escape_underscores : bool, default True
        Replace ``_`` with ``\\_`` in column headers and string cells.
    caption : str, optional
        LaTeX table caption.
    label : str, optional
        LaTeX table label for cross-referencing.

    Returns
    -------
    str
        LaTeX table source code.
    """
    if directions is None:
        directions = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    best_idx: dict[str, int] = {}
    if bold_best:
        for col in numeric_cols:
            is_max = directions.get(col, _is_maximize(col))
            if is_max:
                best_idx[col] = int(df[col].idxmax())
            else:
                best_idx[col] = int(df[col].idxmin())

    def _header(col: str) -> str:
        """Format a column name as a LaTeX header with optional arrow.

        Parameters
        ----------
        col : str
            Column name to format.

        Returns
        -------
        str
            Formatted header string.
        """
        h = col
        if escape_underscores:
            h = h.replace("_", "\\_")
        if arrows and col in numeric_cols:
            is_max = directions.get(col, _is_maximize(col))
            h += " $\\uparrow$" if is_max else " $\\downarrow$"
        return h

    headers = [_header(c) for c in df.columns]
    alignment = "l" + "r" * (len(df.columns) - 1)

    lines: list[str] = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    if caption:
        cap = caption.replace("_", "\\_") if escape_underscores else caption
        lines.append(f"\\caption{{{cap}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{alignment}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")

    for row_num, (idx, row) in enumerate(df.iterrows()):
        cells: list[str] = []
        for col in df.columns:
            val = row[col]
            if col in numeric_cols:
                cell = f"{val:.{precision}f}"
                if bold_best and best_idx.get(col) == idx:
                    cell = f"\\textbf{{{cell}}}"
            else:
                cell = str(val)
                if escape_underscores:
                    cell = cell.replace("_", "\\_")
            cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")
        if midrule_every > 0 and (row_num + 1) % midrule_every == 0 and row_num + 1 < len(df):
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


@dataclass
class ParetoAnalyzer:
    """Multi-objective Pareto dominance analysis.

    Analyzes benchmark results to identify Pareto-optimal solutions
    and compute dominance rankings.  Includes plotting utilities for
    Pareto fronts and cross-dataset consistency analysis.

    Parameters
    ----------
    metrics : list[str]
        Metric names to use for Pareto analysis.
    directions : dict[str, Literal["min", "max"]], optional
        Override metric directions.  Keys are metric names,
        values are ``"min"`` or ``"max"``.

    Examples
    --------
    >>> from tscf_eval.benchmark import ParetoAnalyzer
    >>>
    >>> analyzer = ParetoAnalyzer(["validity", "proximity_l2", "mean_time_s"])
    >>> ranking = analyzer.dominance_ranking(results)
    >>> pareto_optimal = analyzer.pareto_front(results)
    """

    metrics: list[str]
    directions: dict[str, Literal["min", "max"]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that at least one metric is provided."""
        if not self.metrics:
            raise ValueError("At least one metric is required.")

    def _get_direction(self, metric: str) -> bool:
        """Get the optimization direction for a metric.

        Parameters
        ----------
        metric : str
            Metric name to look up.

        Returns
        -------
        bool
            ``True`` if the metric should be maximized.
        """
        if metric in self.directions:
            return self.directions[metric] == "max"
        return _is_maximize(metric)

    def _extract_values(
        self,
        results: BenchmarkResults,
        aggregate_by: str = "explainer",
    ) -> tuple[list[str], np.ndarray]:
        """Extract metric values as array oriented for minimization.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results to analyze.
        aggregate_by : str
            Dimension to aggregate by: ``"explainer"``, ``"model"``, or
            ``"dataset"``.

        Returns
        -------
        names : list[str]
            Names of the aggregated entities.
        values : np.ndarray
            Shape ``(n_entities, n_metrics)``, oriented for minimization.
        """
        df = results.to_dataframe()

        if df.empty:
            return [], np.zeros((0, len(self.metrics)))

        if aggregate_by not in df.columns:
            raise ValueError(f"Cannot aggregate by '{aggregate_by}'")

        available = [m for m in self.metrics if m in df.columns]
        if not available:
            raise ValueError(
                f"None of the specified metrics found. "
                f"Available: {list(df.select_dtypes(include=[np.number]).columns)}"
            )

        grouped = df.groupby(aggregate_by)[available].mean()
        names = list(grouped.index)
        values = grouped.values.copy()

        for i, metric in enumerate(available):
            if self._get_direction(metric):
                values[:, i] = -values[:, i]

        return names, values

    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Check if solution *a* Pareto-dominates solution *b* (lower is better).

        Parameters
        ----------
        a : np.ndarray
            Objective values for solution a, shape ``(n_metrics,)``.
        b : np.ndarray
            Objective values for solution b, shape ``(n_metrics,)``.

        Returns
        -------
        bool
            ``True`` if ``a`` is at least as good in all objectives and
            strictly better in at least one.
        """
        at_least_as_good = np.all(a <= b)
        strictly_better = np.any(a < b)
        return bool(at_least_as_good and strictly_better)

    def pareto_front(
        self,
        results: BenchmarkResults,
        aggregate_by: str = "explainer",
    ) -> list[str]:
        """Find Pareto-optimal (non-dominated) solutions.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results to analyze.
        aggregate_by : str, default "explainer"
            Dimension to aggregate by.

        Returns
        -------
        list[str]
            Names of Pareto-optimal solutions.
        """
        names, values = self._extract_values(results, aggregate_by)
        n = len(names)
        if n == 0:
            return []

        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and self._dominates(values[j], values[i]):
                    is_dominated[i] = True
                    break

        return [names[i] for i in range(n) if not is_dominated[i]]

    def dominance_count(
        self,
        results: BenchmarkResults,
        aggregate_by: str = "explainer",
    ) -> dict[str, int]:
        """Count how many solutions each solution dominates.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results to analyze.
        aggregate_by : str, default "explainer"
            Dimension to aggregate by.

        Returns
        -------
        dict[str, int]
            Mapping from name to number of dominated solutions.
        """
        names, values = self._extract_values(results, aggregate_by)
        n = len(names)
        return {
            name: sum(1 for j in range(n) if i != j and self._dominates(values[i], values[j]))
            for i, name in enumerate(names)
        }

    def dominated_by_count(
        self,
        results: BenchmarkResults,
        aggregate_by: str = "explainer",
    ) -> dict[str, int]:
        """Count how many solutions dominate each solution.

        Lower is better (0 means Pareto-optimal).

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results to analyze.
        aggregate_by : str, default "explainer"
            Dimension to aggregate by.

        Returns
        -------
        dict[str, int]
            Mapping from name to number of dominating solutions.
        """
        names, values = self._extract_values(results, aggregate_by)
        n = len(names)
        return {
            name: sum(1 for j in range(n) if i != j and self._dominates(values[j], values[i]))
            for i, name in enumerate(names)
        }

    def dominance_ranking(
        self,
        results: BenchmarkResults,
        aggregate_by: str = "explainer",
    ) -> pd.DataFrame:
        """Compute dominance ranking table.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results to analyze.
        aggregate_by : str, default "explainer"
            Dimension to aggregate by.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``name``, ``dominated_by``,
            ``dominates``, ``pareto``, plus one column per metric.
        """
        names, values = self._extract_values(results, aggregate_by)

        if len(names) == 0:
            return pd.DataFrame(
                columns=["name", "dominated_by", "dominates", "pareto", *self.metrics]
            )

        dominated_by = self.dominated_by_count(results, aggregate_by)
        dominates = self.dominance_count(results, aggregate_by)
        pareto_set = set(self.pareto_front(results, aggregate_by))

        display_values = values.copy()
        available_metrics = [m for m in self.metrics if m in results.to_dataframe().columns]
        for i, metric in enumerate(available_metrics):
            if self._get_direction(metric):
                display_values[:, i] = -display_values[:, i]

        rows = []
        for idx, name in enumerate(names):
            row: dict = {
                "name": name,
                "dominated_by": dominated_by.get(name, 0),
                "dominates": dominates.get(name, 0),
                "pareto": name in pareto_set,
            }
            for j, metric in enumerate(available_metrics):
                row[metric] = display_values[idx, j]
            rows.append(row)

        df = pd.DataFrame(rows)

        if available_metrics:
            first_metric = available_metrics[0]
            ascending_first = not self._get_direction(first_metric)
            df = df.sort_values(
                by=["dominated_by", first_metric],
                ascending=[True, ascending_first],
            ).reset_index(drop=True)
        else:
            df = df.sort_values(by="dominated_by").reset_index(drop=True)

        return df

    def to_dataframe(
        self,
        results: BenchmarkResults,
        aggregate_by: str = "explainer",
    ) -> pd.DataFrame:
        """Get DataFrame with metric values and Pareto status.

        Alias for :meth:`dominance_ranking`.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results to analyze.
        aggregate_by : str, default "explainer"
            Dimension to aggregate by.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``name``, ``dominated_by``,
            ``dominates``, ``pareto``, plus one column per metric.
        """
        return self.dominance_ranking(results, aggregate_by)

    def plot_front(
        self,
        results: BenchmarkResults,
        x_metric: str,
        y_metric: str,
        aggregate_by: str = "explainer",
        ax: Axes | None = None,
        annotate: bool = True,
        pareto_color: str = "tab:blue",
        other_color: str = "grey",
        pareto_marker: str = "o",
        other_marker: str = "x",
        title: str | None = None,
    ) -> Axes:
        """Plot a 2-D Pareto front scatter.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results to analyze.
        x_metric, y_metric : str
            Metrics for the x and y axes.
        aggregate_by : str, default "explainer"
            Dimension to aggregate by.
        ax : matplotlib Axes, optional
            Axes to plot on.  Created if ``None``.
        annotate : bool, default True
            Label each point with the entity name.
        pareto_color : str, default "tab:blue"
            Color for Pareto-optimal points.
        other_color : str, default "grey"
            Color for dominated points.
        pareto_marker : str, default "o"
            Marker for Pareto-optimal points.
        other_marker : str, default "x"
            Marker for dominated points.
        title : str, optional
            Plot title.  Defaults to ``"Pareto Front"``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        df = results.to_dataframe()
        if aggregate_by not in df.columns:
            raise ValueError(f"Cannot aggregate by '{aggregate_by}'")
        for m in (x_metric, y_metric):
            if m not in df.columns:
                raise ValueError(f"Metric '{m}' not found in results.")

        grouped = df.groupby(aggregate_by)[[x_metric, y_metric]].mean()
        pareto_set = set(self.pareto_front(results, aggregate_by))

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        pareto_names: list[str] = []
        pareto_xs: list[float] = []
        pareto_ys: list[float] = []
        other_names: list[str] = []
        other_xs: list[float] = []
        other_ys: list[float] = []

        for name in grouped.index:
            xv = grouped.loc[name, x_metric]
            yv = grouped.loc[name, y_metric]
            if name in pareto_set:
                pareto_names.append(name)
                pareto_xs.append(xv)
                pareto_ys.append(yv)
            else:
                other_names.append(name)
                other_xs.append(xv)
                other_ys.append(yv)

        if other_xs:
            ax.scatter(
                other_xs,
                other_ys,
                color=other_color,
                marker=other_marker,
                alpha=0.6,
                label="Dominated",
                zorder=2,
            )

        if pareto_xs:
            ax.scatter(
                pareto_xs,
                pareto_ys,
                color=pareto_color,
                marker=pareto_marker,
                s=80,
                label="Pareto-optimal",
                zorder=3,
            )
            if len(pareto_xs) > 1:
                order = np.argsort(pareto_xs)
                ax.plot(
                    [pareto_xs[i] for i in order],
                    [pareto_ys[i] for i in order],
                    color=pareto_color,
                    linestyle="--",
                    alpha=0.5,
                    zorder=1,
                )

        if annotate:
            all_names = pareto_names + other_names
            all_xs = pareto_xs + other_xs
            all_ys = pareto_ys + other_ys
            try:
                from adjustText import adjust_text

                texts = [
                    ax.text(xv, yv, name, fontsize=8)
                    for name, xv, yv in zip(all_names, all_xs, all_ys, strict=True)
                ]
                adjust_text(texts, ax=ax)
            except ImportError:
                warnings.warn(
                    "adjustText is not installed. Label placement in the Pareto "
                    "plot may overlap. Install adjustText for improved annotation: "
                    "pip install adjustText",
                    UserWarning,
                    stacklevel=2,
                )
                for name, xv, yv in zip(all_names, all_xs, all_ys, strict=True):
                    ax.annotate(
                        name,
                        (xv, yv),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=8,
                    )

        x_arrow = "\u2191" if self._get_direction(x_metric) else "\u2193"
        y_arrow = "\u2191" if self._get_direction(y_metric) else "\u2193"
        ax.set_xlabel(f"{x_metric} ({x_arrow})")
        ax.set_ylabel(f"{y_metric} ({y_arrow})")
        ax.set_title(title or "Pareto Front")
        ax.legend()

        return ax

    def consistency(
        self,
        results_dict: dict[str, BenchmarkResults],
        aggregate_by: str = "explainer",
    ) -> pd.DataFrame:
        """Compute cross-dataset Pareto consistency matrix.

        For each dataset identifies the Pareto-optimal solutions and
        returns a boolean matrix (entity x dataset) with a ``count``
        column.

        Parameters
        ----------
        results_dict : dict[str, BenchmarkResults]
            Mapping from dataset/group name to its benchmark results.
        aggregate_by : str, default "explainer"
            Dimension to aggregate by.

        Returns
        -------
        pd.DataFrame
            Boolean DataFrame with entities as rows, datasets as columns,
            plus a ``count`` column.  Sorted by count descending.
        """
        if not results_dict:
            return pd.DataFrame()

        all_entities: set[str] = set()
        per_dataset: dict[str, set[str]] = {}

        for ds_name, ds_results in results_dict.items():
            front = self.pareto_front(ds_results, aggregate_by)
            per_dataset[ds_name] = set(front)
            df = ds_results.to_dataframe()
            if aggregate_by in df.columns:
                all_entities.update(df[aggregate_by].unique())

        entities = sorted(all_entities)
        datasets = list(results_dict.keys())

        data: dict[str, list[bool]] = {
            ds: [e in per_dataset.get(ds, set()) for e in entities] for ds in datasets
        }

        result_df = pd.DataFrame(data, index=entities)
        result_df["count"] = result_df.sum(axis=1)
        return result_df.sort_values("count", ascending=False)

    def plot_consistency_heatmap(
        self,
        consistency_df: pd.DataFrame,
        ax: Axes | None = None,
        cmap: str = "YlGn",
        title: str | None = None,
    ) -> Axes:
        """Plot Pareto consistency as a heatmap.

        Parameters
        ----------
        consistency_df : pd.DataFrame
            Output of :meth:`consistency`.
        ax : matplotlib Axes, optional
            Axes to plot on.  Created if ``None``.
        cmap : str, default "YlGn"
            Matplotlib colormap name.
        title : str, optional
            Plot title.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        plot_df = consistency_df.drop(columns=["count"], errors="ignore")

        if ax is None:
            fig_w = max(8, len(plot_df.columns) * 1.2)
            fig_h = max(4, len(plot_df) * 0.5)
            _, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(
            plot_df.values.astype(float),
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )

        ax.set_xticks(range(len(plot_df.columns)))
        ax.set_xticklabels(plot_df.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(plot_df.index)))
        ax.set_yticklabels(plot_df.index)

        for i in range(len(plot_df.index)):
            for j in range(len(plot_df.columns)):
                val = plot_df.iloc[i, j]
                ax.text(
                    j,
                    i,
                    "\u2713" if val else "",
                    ha="center",
                    va="center",
                    fontsize=10,
                )

        ax.set_title(title or "Pareto Consistency Across Datasets")
        ax.figure.colorbar(im, ax=ax, label="Pareto-optimal", shrink=0.8)
        return ax

    def to_latex(
        self,
        results: BenchmarkResults,
        aggregate_by: str = "explainer",
        precision: int = 3,
        caption: str | None = None,
        label: str | None = None,
    ) -> str:
        """Generate a LaTeX table of the dominance ranking.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results to analyze.
        aggregate_by : str, default "explainer"
            Dimension to aggregate by.
        precision : int, default 3
            Number of decimal places.
        caption, label : str, optional
            LaTeX caption and label.

        Returns
        -------
        str
            LaTeX table source code.
        """
        ranking = self.dominance_ranking(results, aggregate_by)
        available = [m for m in self.metrics if m in ranking.columns]
        dirs = {m: self._get_direction(m) for m in available}
        return format_latex_table(
            ranking,
            directions=dirs,
            precision=precision,
            caption=caption,
            label=label,
        )


@dataclass
class WeightedScalarizer:
    """Min-max normalized weighted composite scoring.

    Normalizes each metric to ``[0, 1]`` via min-max scaling (respecting
    metric directions so that higher normalized values are always better),
    then computes a weighted sum.

    Parameters
    ----------
    metrics : list[str]
        Metric names to include in the composite score.
    weights : dict[str, float], optional
        Per-metric weights.  Automatically normalized to sum to 1.
        If ``None``, all metrics are weighted equally.
    directions : dict[str, Literal["min", "max"]], optional
        Override metric directions.

    Examples
    --------
    >>> scalarizer = WeightedScalarizer(
    ...     ["validity", "proximity_l2", "sparsity"],
    ...     weights={"validity": 2.0, "proximity_l2": 1.0, "sparsity": 1.0},
    ... )
    >>> scores = scalarizer.score(results)
    """

    metrics: list[str]
    weights: dict[str, float] = field(default_factory=dict)
    directions: dict[str, Literal["min", "max"]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate metrics and normalize weights to sum to 1."""
        if not self.metrics:
            raise ValueError("At least one metric is required.")
        if not self.weights:
            w = 1.0 / len(self.metrics)
            self.weights = dict.fromkeys(self.metrics, w)
        else:
            total = sum(self.weights.get(m, 0.0) for m in self.metrics)
            if total <= 0:
                raise ValueError("Sum of weights must be positive.")
            self.weights = {m: self.weights.get(m, 0.0) / total for m in self.metrics}

    def _get_direction(self, metric: str) -> bool:
        """Get the optimization direction for a metric.

        Parameters
        ----------
        metric : str
            Metric name to look up.

        Returns
        -------
        bool
            ``True`` if the metric should be maximized.
        """
        if metric in self.directions:
            return self.directions[metric] == "max"
        return _is_maximize(metric)

    def score(
        self,
        results: BenchmarkResults,
        aggregate_by: str = "explainer",
    ) -> pd.DataFrame:
        """Compute weighted composite scores.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results to analyze.
        aggregate_by : str, default "explainer"
            Dimension to aggregate by.

        Returns
        -------
        pd.DataFrame
            DataFrame with normalized metric columns plus ``composite``.
            Sorted by composite descending.
        """
        df = results.to_dataframe()
        if df.empty:
            return pd.DataFrame(columns=[aggregate_by, *self.metrics, "composite"])

        available = [m for m in self.metrics if m in df.columns]
        if not available:
            raise ValueError(
                f"None of the specified metrics found. "
                f"Available: {list(df.select_dtypes(include=[np.number]).columns)}"
            )

        grouped = df.groupby(aggregate_by)[available].mean()

        normalized = pd.DataFrame(index=grouped.index)
        for col in available:
            col_min = grouped[col].min()
            col_max = grouped[col].max()
            rng = col_max - col_min
            if rng == 0:
                normalized[col] = 1.0
            elif self._get_direction(col):
                normalized[col] = (grouped[col] - col_min) / rng
            else:
                normalized[col] = (col_max - grouped[col]) / rng

        composite = np.zeros(len(normalized))
        for col in available:
            composite += self.weights.get(col, 0.0) * normalized[col].values

        available_weight_sum = sum(self.weights.get(m, 0.0) for m in available)
        if available_weight_sum > 0 and available_weight_sum != 1.0:
            composite /= available_weight_sum

        normalized["composite"] = composite
        normalized = normalized.sort_values("composite", ascending=False)
        return normalized.reset_index()

    def sensitivity(
        self,
        results: BenchmarkResults,
        vary_metric: str,
        n_steps: int = 11,
        aggregate_by: str = "explainer",
    ) -> pd.DataFrame:
        """Sensitivity analysis by sweeping one metric's weight.

        Varies the weight of *vary_metric* from 0 to 1, redistributing
        the remaining weight proportionally among the other metrics.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results to analyze.
        vary_metric : str
            The metric whose weight to sweep.
        n_steps : int, default 11
            Number of weight values (0 to 1 inclusive).
        aggregate_by : str, default "explainer"
            Dimension to aggregate by.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns
            ``weight``, ``<aggregate_by>``, ``composite``.
        """
        if vary_metric not in self.metrics:
            raise ValueError(f"'{vary_metric}' is not in the metrics list.")

        other_metrics = [m for m in self.metrics if m != vary_metric]
        other_total = sum(self.weights.get(m, 0.0) for m in other_metrics)

        rows = []
        for w_vary in np.linspace(0, 1, n_steps):
            w_rest = 1.0 - w_vary
            new_weights: dict[str, float] = {vary_metric: float(w_vary)}
            for m in other_metrics:
                if other_total > 0:
                    new_weights[m] = w_rest * (self.weights.get(m, 0.0) / other_total)
                else:
                    new_weights[m] = w_rest / len(other_metrics) if other_metrics else 0.0

            temp = WeightedScalarizer(
                metrics=self.metrics,
                weights=new_weights,
                directions=self.directions,
            )
            scored = temp.score(results, aggregate_by)
            for _, row in scored.iterrows():
                rows.append(
                    {
                        "weight": float(w_vary),
                        aggregate_by: row[aggregate_by],
                        "composite": row["composite"],
                    }
                )

        return pd.DataFrame(rows)

    def plot_sensitivity(
        self,
        sensitivity_df: pd.DataFrame,
        aggregate_by: str = "explainer",
        ax: Axes | None = None,
        title: str | None = None,
    ) -> Axes:
        """Plot sensitivity analysis results.

        Parameters
        ----------
        sensitivity_df : pd.DataFrame
            Output of :meth:`sensitivity`.
        aggregate_by : str, default "explainer"
            Column name for the entity dimension.
        ax : matplotlib Axes, optional
            Axes to plot on.  Created if ``None``.
        title : str, optional
            Plot title.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        for entity in sensitivity_df[aggregate_by].unique():
            subset = sensitivity_df[sensitivity_df[aggregate_by] == entity].sort_values("weight")
            ax.plot(
                subset["weight"],
                subset["composite"],
                marker="o",
                label=entity,
                markersize=4,
            )

        ax.set_xlabel("Weight")
        ax.set_ylabel("Composite Score")
        ax.set_title(title or "Sensitivity Analysis")
        ax.legend()
        ax.set_xlim(0, 1)
        return ax

    def to_latex(
        self,
        results: BenchmarkResults,
        aggregate_by: str = "explainer",
        precision: int = 3,
        caption: str | None = None,
        label: str | None = None,
    ) -> str:
        """Generate a LaTeX table of weighted scores.

        Parameters
        ----------
        results : BenchmarkResults
            Benchmark results to analyze.
        aggregate_by : str, default "explainer"
            Dimension to aggregate by.
        precision : int, default 3
            Number of decimal places.
        caption, label : str, optional
            LaTeX caption and label.

        Returns
        -------
        str
            LaTeX table source code.
        """
        scored = self.score(results, aggregate_by)
        available = [m for m in self.metrics if m in scored.columns]
        dirs: dict[str, bool] = dict.fromkeys(available, True)
        dirs["composite"] = True
        return format_latex_table(
            scored,
            directions=dirs,
            precision=precision,
            caption=caption,
            label=label,
        )
