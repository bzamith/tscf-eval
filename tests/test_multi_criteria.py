"""Tests for the multi_criteria module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tscf_eval.benchmark import (
    BenchmarkResults,
    ExplainerResult,
    FriedmanResult,
    ParetoAnalyzer,
    WeightedScalarizer,
    format_latex_table,
    friedman_test,
)

# =============================================================================
# Fixtures
# =============================================================================


def _make_result(
    ds: str,
    model: str,
    exp: str,
    metrics: dict[str, float],
) -> ExplainerResult:
    """Helper to build a minimal ExplainerResult."""
    return ExplainerResult(
        explainer_name=exp,
        dataset_name=ds,
        model_name=model,
        X_cf=np.random.randn(5, 20),
        y_cf=np.random.randint(0, 2, 5),
        success_mask=np.ones(5, dtype=bool),
        metrics=metrics,
        generation_times=[0.1] * 5,
        metadata=[{}] * 5,
    )


@pytest.fixture
def sample_results() -> BenchmarkResults:
    """Results with 3 explainers, 2 datasets, 1 model."""
    results = BenchmarkResults()
    np.random.seed(0)

    # Explainer A: high validity, low proximity
    # Explainer B: medium validity, medium proximity
    # Explainer C: low validity, high proximity
    metric_sets = {
        "A": {"validity": 0.95, "proximity_l2": 0.4, "sparsity": 0.6},
        "B": {"validity": 0.80, "proximity_l2": 0.7, "sparsity": 0.4},
        "C": {"validity": 0.60, "proximity_l2": 0.9, "sparsity": 0.2},
    }

    for ds in ["DS1", "DS2"]:
        for exp, metrics in metric_sets.items():
            # Add small per-dataset variation
            varied = {k: v + np.random.uniform(-0.05, 0.05) for k, v in metrics.items()}
            results.add(_make_result(ds, "knn", exp, varied))

    return results


@pytest.fixture
def results_dict(sample_results: BenchmarkResults) -> dict[str, BenchmarkResults]:
    """Per-dataset results dictionary."""
    return {
        "DS1": sample_results.filter(datasets=["DS1"]),
        "DS2": sample_results.filter(datasets=["DS2"]),
    }


# =============================================================================
# ParetoAnalyzer — preserved behavior
# =============================================================================


class TestParetoAnalyzerPreserved:
    """Tests for ParetoAnalyzer methods carried over from pareto.py."""

    def test_creation(self) -> None:
        analyzer = ParetoAnalyzer(["validity", "proximity_l2"])
        assert len(analyzer.metrics) == 2

    def test_empty_metrics_raises(self) -> None:
        with pytest.raises(ValueError, match="metric"):
            ParetoAnalyzer([])

    def test_pareto_front(self, sample_results: BenchmarkResults) -> None:
        analyzer = ParetoAnalyzer(["validity", "proximity_l2"])
        front = analyzer.pareto_front(sample_results)
        assert isinstance(front, list)
        assert len(front) >= 1

    def test_dominance_count(self, sample_results: BenchmarkResults) -> None:
        analyzer = ParetoAnalyzer(["validity", "proximity_l2"])
        counts = analyzer.dominance_count(sample_results)
        assert isinstance(counts, dict)
        assert all(isinstance(v, int) for v in counts.values())

    def test_dominated_by_count(self, sample_results: BenchmarkResults) -> None:
        analyzer = ParetoAnalyzer(["validity", "proximity_l2"])
        counts = analyzer.dominated_by_count(sample_results)
        # Pareto-optimal solutions have 0 dominators
        front = set(analyzer.pareto_front(sample_results))
        for name in front:
            assert counts[name] == 0

    def test_dominance_ranking(self, sample_results: BenchmarkResults) -> None:
        analyzer = ParetoAnalyzer(["validity", "proximity_l2"])
        ranking = analyzer.dominance_ranking(sample_results)
        assert "name" in ranking.columns
        assert "dominated_by" in ranking.columns
        assert "pareto" in ranking.columns

    def test_to_dataframe(self, sample_results: BenchmarkResults) -> None:
        analyzer = ParetoAnalyzer(["validity", "proximity_l2"])
        df = analyzer.to_dataframe(sample_results)
        assert "name" in df.columns
        assert "pareto" in df.columns

    def test_custom_directions(self, sample_results: BenchmarkResults) -> None:
        analyzer = ParetoAnalyzer(
            ["validity", "proximity_l2"],
            directions={"proximity_l2": "min"},
        )
        ranking = analyzer.dominance_ranking(sample_results)
        assert len(ranking) > 0

    def test_dominates_basic(self) -> None:
        analyzer = ParetoAnalyzer(["a", "b"])
        assert analyzer._dominates(np.array([1.0, 1.0]), np.array([2.0, 2.0]))
        assert not analyzer._dominates(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
        assert not analyzer._dominates(np.array([1.0, 3.0]), np.array([2.0, 2.0]))

    def test_empty_results(self) -> None:
        analyzer = ParetoAnalyzer(["validity"])
        empty = BenchmarkResults()
        assert analyzer.pareto_front(empty) == []
        ranking = analyzer.dominance_ranking(empty)
        assert len(ranking) == 0

    def test_single_explainer(self) -> None:
        results = BenchmarkResults()
        results.add(_make_result("DS1", "knn", "only", {"validity": 0.9}))
        analyzer = ParetoAnalyzer(["validity"])
        front = analyzer.pareto_front(results)
        assert front == ["only"]


# =============================================================================
# ParetoAnalyzer — new methods
# =============================================================================


class TestParetoAnalyzerPlotFront:
    """Tests for ParetoAnalyzer.plot_front."""

    def test_returns_axes(self, sample_results: BenchmarkResults) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")

        analyzer = ParetoAnalyzer(["validity", "proximity_l2"])
        ax = analyzer.plot_front(
            sample_results,
            "validity",
            "proximity_l2",
            annotate=False,
        )

        from matplotlib.axes import Axes

        assert isinstance(ax, Axes)

    def test_custom_ax(self, sample_results: BenchmarkResults) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
        fig, ax = plt.subplots()

        analyzer = ParetoAnalyzer(["validity", "proximity_l2"])
        returned_ax = analyzer.plot_front(
            sample_results,
            "validity",
            "proximity_l2",
            ax=ax,
            annotate=False,
        )
        assert returned_ax is ax
        plt.close(fig)

    def test_missing_metric_raises(self, sample_results: BenchmarkResults) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")

        analyzer = ParetoAnalyzer(["validity"])
        with pytest.raises(ValueError, match="not found"):
            analyzer.plot_front(sample_results, "validity", "nonexistent")


class TestParetoAnalyzerConsistency:
    """Tests for ParetoAnalyzer.consistency."""

    def test_shape(
        self,
        results_dict: dict[str, BenchmarkResults],
    ) -> None:
        analyzer = ParetoAnalyzer(["validity", "proximity_l2"])
        df = analyzer.consistency(results_dict)

        # Should have 3 explainers as rows, 2 datasets + count as columns
        assert len(df) == 3
        assert "count" in df.columns
        assert "DS1" in df.columns
        assert "DS2" in df.columns

    def test_count_column(
        self,
        results_dict: dict[str, BenchmarkResults],
    ) -> None:
        analyzer = ParetoAnalyzer(["validity", "proximity_l2"])
        df = analyzer.consistency(results_dict)

        # count should equal row sum (excluding count itself)
        for idx in df.index:
            expected = sum(df.loc[idx, c] for c in df.columns if c != "count")
            assert df.loc[idx, "count"] == expected

    def test_sorted_by_count(
        self,
        results_dict: dict[str, BenchmarkResults],
    ) -> None:
        analyzer = ParetoAnalyzer(["validity", "proximity_l2"])
        df = analyzer.consistency(results_dict)
        counts = df["count"].tolist()
        assert counts == sorted(counts, reverse=True)

    def test_empty_dict(self) -> None:
        analyzer = ParetoAnalyzer(["validity"])
        df = analyzer.consistency({})
        assert df.empty


class TestParetoAnalyzerConsistencyHeatmap:
    """Tests for ParetoAnalyzer.plot_consistency_heatmap."""

    def test_returns_axes(
        self,
        results_dict: dict[str, BenchmarkResults],
    ) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")

        analyzer = ParetoAnalyzer(["validity", "proximity_l2"])
        consistency_df = analyzer.consistency(results_dict)
        ax = analyzer.plot_consistency_heatmap(consistency_df)

        from matplotlib.axes import Axes

        assert isinstance(ax, Axes)


class TestParetoAnalyzerToLatex:
    """Tests for ParetoAnalyzer.to_latex."""

    def test_produces_string(self, sample_results: BenchmarkResults) -> None:
        analyzer = ParetoAnalyzer(["validity", "proximity_l2"])
        latex = analyzer.to_latex(sample_results)
        assert isinstance(latex, str)
        assert "\\begin{table}" in latex
        assert "\\end{table}" in latex

    def test_with_caption(self, sample_results: BenchmarkResults) -> None:
        analyzer = ParetoAnalyzer(["validity"])
        latex = analyzer.to_latex(sample_results, caption="My Table", label="tab:test")
        assert "\\caption{My Table}" in latex
        assert "\\label{tab:test}" in latex


# =============================================================================
# WeightedScalarizer
# =============================================================================


class TestWeightedScalarizer:
    """Tests for WeightedScalarizer."""

    def test_creation(self) -> None:
        ws = WeightedScalarizer(["validity", "proximity_l2"])
        assert len(ws.metrics) == 2
        # Equal weights by default
        assert ws.weights["validity"] == pytest.approx(0.5)
        assert ws.weights["proximity_l2"] == pytest.approx(0.5)

    def test_empty_metrics_raises(self) -> None:
        with pytest.raises(ValueError, match="metric"):
            WeightedScalarizer([])

    def test_weight_normalization(self) -> None:
        ws = WeightedScalarizer(
            ["validity", "proximity_l2"],
            weights={"validity": 3.0, "proximity_l2": 1.0},
        )
        assert ws.weights["validity"] == pytest.approx(0.75)
        assert ws.weights["proximity_l2"] == pytest.approx(0.25)

    def test_score(self, sample_results: BenchmarkResults) -> None:
        ws = WeightedScalarizer(["validity", "proximity_l2"])
        scored = ws.score(sample_results)

        assert "composite" in scored.columns
        assert "explainer" in scored.columns
        assert len(scored) == 3  # A, B, C

        # Sorted by composite descending
        composites = scored["composite"].tolist()
        assert composites == sorted(composites, reverse=True)

    def test_score_values_in_range(self, sample_results: BenchmarkResults) -> None:
        ws = WeightedScalarizer(["validity", "proximity_l2"])
        scored = ws.score(sample_results)

        for col in ["validity", "proximity_l2", "composite"]:
            assert scored[col].min() >= 0.0 - 1e-9
            assert scored[col].max() <= 1.0 + 1e-9

    def test_score_with_custom_weights(self, sample_results: BenchmarkResults) -> None:
        ws_heavy_validity = WeightedScalarizer(
            ["validity", "proximity_l2"],
            weights={"validity": 10.0, "proximity_l2": 0.1},
        )
        scored = ws_heavy_validity.score(sample_results)
        # With heavy validity weight, A (highest validity) should rank first
        assert scored.iloc[0]["explainer"] == "A"

    def test_score_empty_results(self) -> None:
        ws = WeightedScalarizer(["validity"])
        scored = ws.score(BenchmarkResults())
        assert len(scored) == 0

    def test_missing_metrics_raises(self) -> None:
        results = BenchmarkResults()
        results.add(_make_result("DS1", "knn", "A", {"validity": 0.9}))
        ws = WeightedScalarizer(["nonexistent_metric"])
        with pytest.raises(ValueError, match="None of the specified metrics"):
            ws.score(results)


class TestWeightedScalarizerSensitivity:
    """Tests for WeightedScalarizer.sensitivity."""

    def test_shape(self, sample_results: BenchmarkResults) -> None:
        ws = WeightedScalarizer(["validity", "proximity_l2"])
        sens = ws.sensitivity(sample_results, "validity", n_steps=5)

        # 5 weight steps x 3 explainers = 15 rows
        assert len(sens) == 15
        assert "weight" in sens.columns
        assert "composite" in sens.columns
        assert "explainer" in sens.columns

    def test_weight_range(self, sample_results: BenchmarkResults) -> None:
        ws = WeightedScalarizer(["validity", "proximity_l2"])
        sens = ws.sensitivity(sample_results, "validity", n_steps=11)

        weights = sorted(sens["weight"].unique())
        assert weights[0] == pytest.approx(0.0)
        assert weights[-1] == pytest.approx(1.0)

    def test_invalid_metric_raises(self, sample_results: BenchmarkResults) -> None:
        ws = WeightedScalarizer(["validity", "proximity_l2"])
        with pytest.raises(ValueError, match="not in the metrics list"):
            ws.sensitivity(sample_results, "nonexistent")


class TestWeightedScalarizerPlotSensitivity:
    """Tests for WeightedScalarizer.plot_sensitivity."""

    def test_returns_axes(self, sample_results: BenchmarkResults) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")

        ws = WeightedScalarizer(["validity", "proximity_l2"])
        sens = ws.sensitivity(sample_results, "validity", n_steps=5)
        ax = ws.plot_sensitivity(sens)

        from matplotlib.axes import Axes

        assert isinstance(ax, Axes)


class TestWeightedScalarizerToLatex:
    """Tests for WeightedScalarizer.to_latex."""

    def test_produces_string(self, sample_results: BenchmarkResults) -> None:
        ws = WeightedScalarizer(["validity", "proximity_l2"])
        latex = ws.to_latex(sample_results)
        assert isinstance(latex, str)
        assert "\\begin{table}" in latex


# =============================================================================
# friedman_test
# =============================================================================


class TestFriedmanTest:
    """Tests for the friedman_test function."""

    def test_basic(self) -> None:
        pytest.importorskip("scipy.stats")

        # Need >= 3 explainers x >= 2 datasets
        results = BenchmarkResults()
        for ds in ["DS1", "DS2", "DS3"]:
            results.add(_make_result(ds, "knn", "A", {"validity": 0.9}))
            results.add(_make_result(ds, "knn", "B", {"validity": 0.7}))
            results.add(_make_result(ds, "knn", "C", {"validity": 0.5}))

        fr = friedman_test(results, "validity")

        assert isinstance(fr, FriedmanResult)
        assert isinstance(fr.statistic, float)
        assert isinstance(fr.p_value, float)
        assert isinstance(fr.rankings, pd.DataFrame)
        assert "mean_rank" in fr.rankings.columns

    def test_too_few_treatments_raises(self) -> None:
        pytest.importorskip("scipy.stats")

        results = BenchmarkResults()
        results.add(_make_result("DS1", "knn", "A", {"validity": 0.9}))
        results.add(_make_result("DS1", "knn", "B", {"validity": 0.7}))
        results.add(_make_result("DS2", "knn", "A", {"validity": 0.8}))
        results.add(_make_result("DS2", "knn", "B", {"validity": 0.6}))

        with pytest.raises(ValueError, match="at least 3 treatments"):
            friedman_test(results, "validity")

    def test_missing_metric_raises(self) -> None:
        pytest.importorskip("scipy.stats")

        results = BenchmarkResults()
        results.add(_make_result("DS1", "knn", "A", {"validity": 0.9}))

        with pytest.raises(ValueError, match="not found"):
            friedman_test(results, "nonexistent")


# =============================================================================
# format_latex_table
# =============================================================================


class TestFormatLatexTable:
    """Tests for the format_latex_table function."""

    def test_basic_output(self) -> None:
        df = pd.DataFrame(
            {
                "name": ["A", "B", "C"],
                "validity": [0.9, 0.8, 0.7],
                "sparsity": [0.3, 0.4, 0.5],
            }
        )
        latex = format_latex_table(df)
        assert "\\begin{table}" in latex
        assert "\\end{table}" in latex
        assert "\\toprule" in latex
        assert "\\bottomrule" in latex

    def test_bold_best(self) -> None:
        df = pd.DataFrame(
            {
                "name": ["A", "B"],
                "validity": [0.9, 0.7],
            }
        )
        latex = format_latex_table(df, directions={"validity": True})
        assert "\\textbf{0.900}" in latex

    def test_arrows(self) -> None:
        df = pd.DataFrame(
            {
                "name": ["A"],
                "validity": [0.9],
                "sparsity": [0.3],
            }
        )
        latex = format_latex_table(df, arrows=True)
        assert "$\\uparrow$" in latex  # validity is maximize
        assert "$\\downarrow$" in latex  # sparsity is minimize

    def test_no_arrows(self) -> None:
        df = pd.DataFrame({"name": ["A"], "validity": [0.9]})
        latex = format_latex_table(df, arrows=False)
        assert "$\\uparrow$" not in latex

    def test_midrule_every(self) -> None:
        df = pd.DataFrame(
            {
                "name": ["A", "B", "C", "D"],
                "val": [1.0, 2.0, 3.0, 4.0],
            }
        )
        latex = format_latex_table(df, midrule_every=2)
        # Should have header midrule + one mid-data midrule
        assert latex.count("\\midrule") == 2

    def test_escape_underscores(self) -> None:
        df = pd.DataFrame({"name_col": ["val_a"], "metric_1": [0.5]})
        latex = format_latex_table(df, escape_underscores=True)
        assert "name\\_col" in latex
        assert "val\\_a" in latex

    def test_caption_and_label(self) -> None:
        df = pd.DataFrame({"name": ["A"], "val": [1.0]})
        latex = format_latex_table(df, caption="Test Cap", label="tab:test")
        assert "\\caption{Test Cap}" in latex
        assert "\\label{tab:test}" in latex

    def test_precision(self) -> None:
        df = pd.DataFrame({"name": ["A"], "val": [1.23456789]})
        latex = format_latex_table(df, precision=2)
        assert "1.23" in latex
        assert "1.235" not in latex

    def test_no_numeric_cols(self) -> None:
        df = pd.DataFrame({"name": ["A", "B"], "category": ["x", "y"]})
        latex = format_latex_table(df)
        assert "\\begin{table}" in latex
