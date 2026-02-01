"""Tests for the benchmark module."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from tscf_eval.benchmark import (
    BenchmarkResults,
    BenchmarkRunner,
    DatasetConfig,
    ExplainerConfig,
    ExplainerResult,
    ModelConfig,
    ParetoAnalyzer,
)
from tscf_eval.counterfactuals import COMTE, NativeGuide
from tscf_eval.counterfactuals.base import Counterfactual

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def dataset_config(univariate_data: tuple[np.ndarray, np.ndarray]) -> DatasetConfig:
    """Create a DatasetConfig for testing."""
    X, y = univariate_data
    # Split into train/test
    X_train, X_test = X[:40], X[40:]
    y_train, y_test = y[:40], y[40:]
    return DatasetConfig(
        name="test_dataset",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@pytest.fixture
def model_config(dataset_config: DatasetConfig) -> ModelConfig:
    """Create a ModelConfig with fitted classifier."""
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(dataset_config.X_train, dataset_config.y_train)
    return ModelConfig("knn", clf)


@pytest.fixture
def explainer_configs() -> list[ExplainerConfig]:
    """Create explainer configurations for testing."""
    return [
        ExplainerConfig("comte", COMTE, {"distance": "euclidean"}),
        ExplainerConfig("ng_blend", NativeGuide, {"method": "blend"}),
    ]


@pytest.fixture
def sample_results() -> BenchmarkResults:
    """Create sample benchmark results for testing."""
    results = BenchmarkResults()

    # Add some mock results
    for ds in ["DS1", "DS2"]:
        for model in ["knn", "svm"]:
            for exp in ["comte", "ng"]:
                result = ExplainerResult(
                    explainer_name=exp,
                    dataset_name=ds,
                    model_name=model,
                    X_cf=np.random.randn(10, 50),
                    y_cf=np.random.randint(0, 2, 10),
                    success_mask=np.ones(10, dtype=bool),
                    metrics={
                        "validity": np.random.uniform(0.7, 1.0),
                        "proximity_l2": np.random.uniform(0.5, 2.0),
                        "sparsity": np.random.uniform(0.3, 0.8),
                    },
                    generation_times=[0.1] * 10,
                    metadata=[{}] * 10,
                )
                results.add(result)

    return results


# =============================================================================
# DatasetConfig Tests
# =============================================================================


class TestDatasetConfig:
    """Tests for DatasetConfig."""

    def test_creation(self, univariate_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test basic creation."""
        X, y = univariate_data
        config = DatasetConfig("test", X[:30], y[:30], X[30:], y[30:])

        assert config.name == "test"
        assert config.n_train == 30
        assert config.n_test == 20

    def test_frozen(self, dataset_config: DatasetConfig) -> None:
        """Test that config is immutable."""
        with pytest.raises(AttributeError):
            dataset_config.name = "new_name"  # type: ignore[misc]

    def test_array_conversion(self) -> None:
        """Test that lists are converted to arrays."""
        X_train = [[1, 2, 3], [4, 5, 6]]
        y_train = [0, 1]
        X_test = [[7, 8, 9]]
        y_test = [1]

        config = DatasetConfig("test", X_train, y_train, X_test, y_test)

        assert isinstance(config.X_train, np.ndarray)
        assert isinstance(config.y_train, np.ndarray)
        assert config.y_train.ndim == 1

    def test_series_length(self) -> None:
        """Test series_length property."""
        X_train = np.random.randn(10, 100)
        config = DatasetConfig("test", X_train, np.zeros(10), X_train[:5])

        assert config.series_length == 100


# =============================================================================
# ModelConfig Tests
# =============================================================================


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_creation(self, simple_classifier: KNeighborsClassifier) -> None:
        """Test basic creation."""
        config = ModelConfig("knn", simple_classifier)
        assert config.name == "knn"

    def test_predict(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
    ) -> None:
        """Test predict method."""
        predictions = model_config.predict(dataset_config.X_test)
        assert len(predictions) == dataset_config.n_test

    def test_predict_proba(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
    ) -> None:
        """Test predict_proba method."""
        proba = model_config.predict_proba(dataset_config.X_test)
        assert proba is not None
        assert proba.shape[0] == dataset_config.n_test


# =============================================================================
# ExplainerConfig Tests
# =============================================================================


class TestExplainerConfig:
    """Tests for ExplainerConfig."""

    def test_creation(self) -> None:
        """Test basic creation."""
        config = ExplainerConfig("comte", COMTE, {"distance": "dtw"})
        assert config.name == "comte"
        assert config.explainer_class == COMTE
        assert config.params == {"distance": "dtw"}

    def test_default_params(self) -> None:
        """Test default empty params."""
        config = ExplainerConfig("comte", COMTE)
        assert config.params == {}
        assert config.n_counterfactuals == 1

    def test_create_explainer(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
    ) -> None:
        """Test explainer instantiation."""
        config = ExplainerConfig("comte", COMTE, {"distance": "euclidean"})
        explainer = config.create_explainer(
            model=model_config.model,
            data=(dataset_config.X_train, dataset_config.y_train),
        )
        assert explainer is not None


# =============================================================================
# ExplainerResult Tests
# =============================================================================


class TestExplainerResult:
    """Tests for ExplainerResult."""

    def test_properties(self) -> None:
        """Test result properties."""
        result = ExplainerResult(
            explainer_name="comte",
            dataset_name="test",
            model_name="knn",
            X_cf=np.random.randn(10, 50),
            y_cf=np.random.randint(0, 2, 10),
            success_mask=np.array([True] * 8 + [False] * 2),
            metrics={"validity": 0.9},
            generation_times=[0.1] * 10,
            metadata=[{}] * 10,
        )

        assert result.n_instances == 10
        assert result.n_successful == 8
        assert result.success_rate == 0.8
        assert result.mean_time == pytest.approx(0.1)
        assert result.total_time == pytest.approx(1.0)

    def test_get_metric(self) -> None:
        """Test get_metric method."""
        result = ExplainerResult(
            explainer_name="comte",
            dataset_name="test",
            model_name="knn",
            X_cf=np.zeros((5, 10)),
            y_cf=np.zeros(5, dtype=int),
            success_mask=np.ones(5, dtype=bool),
            metrics={"validity": 0.9, "proximity_l2": 1.5},
            generation_times=[0.1] * 5,
            metadata=[{}] * 5,
        )

        assert result.get_metric("validity") == 0.9
        assert result.get_metric("missing", default=0.0) == 0.0


# =============================================================================
# BenchmarkResults Tests
# =============================================================================


class TestBenchmarkResults:
    """Tests for BenchmarkResults."""

    def test_add_and_get(self) -> None:
        """Test adding and retrieving results."""
        results = BenchmarkResults()
        result = ExplainerResult(
            explainer_name="comte",
            dataset_name="DS1",
            model_name="knn",
            X_cf=np.zeros((5, 10)),
            y_cf=np.zeros(5, dtype=int),
            success_mask=np.ones(5, dtype=bool),
            metrics={"validity": 0.9},
            generation_times=[0.1] * 5,
            metadata=[{}] * 5,
        )
        results.add(result)

        retrieved = results.get("DS1", "knn", "comte")
        assert retrieved is not None
        assert retrieved.explainer_name == "comte"

    def test_iteration(self, sample_results: BenchmarkResults) -> None:
        """Test iterating over results."""
        count = sum(1 for _ in sample_results)
        assert count == len(sample_results)

    def test_datasets_models_explainers(self, sample_results: BenchmarkResults) -> None:
        """Test property accessors."""
        assert set(sample_results.datasets) == {"DS1", "DS2"}
        assert set(sample_results.models) == {"knn", "svm"}
        assert set(sample_results.explainers) == {"comte", "ng"}

    def test_filter(self, sample_results: BenchmarkResults) -> None:
        """Test filtering results."""
        filtered = sample_results.filter(datasets=["DS1"])
        assert len(filtered) == 4  # 2 models x 2 explainers

        filtered = sample_results.filter(explainers=["comte"])
        assert len(filtered) == 4  # 2 datasets x 2 models

    def test_to_dataframe(self, sample_results: BenchmarkResults) -> None:
        """Test DataFrame conversion."""
        df = sample_results.to_dataframe()

        assert "dataset" in df.columns
        assert "model" in df.columns
        assert "explainer" in df.columns
        assert "validity" in df.columns
        assert len(df) == len(sample_results)

    def test_aggregate(self, sample_results: BenchmarkResults) -> None:
        """Test aggregation."""
        agg = sample_results.aggregate(by="explainer")

        assert len(agg) == 2  # comte, ng
        assert "validity" in agg.columns

    def test_summary(self, sample_results: BenchmarkResults) -> None:
        """Test summary method."""
        summary = sample_results.summary()
        assert len(summary) == 2  # 2 explainers

    def test_to_dict(self, sample_results: BenchmarkResults) -> None:
        """Test to_dict method."""
        d = sample_results.to_dict()

        assert "datasets" in d
        assert "models" in d
        assert "explainers" in d
        assert "results" in d
        assert len(d["results"]) == len(sample_results)


class TestExplainerResultEdgeCases:
    """Edge case tests for ExplainerResult."""

    def test_empty_generation_times(self) -> None:
        """Test with empty generation_times."""
        result = ExplainerResult(
            explainer_name="comte",
            dataset_name="test",
            model_name="knn",
            X_cf=np.zeros((5, 10)),
            y_cf=np.zeros(5, dtype=int),
            success_mask=np.ones(5, dtype=bool),
            metrics={"validity": 0.9},
            generation_times=[],
            metadata=[{}] * 5,
        )

        assert result.mean_time == 0.0
        assert result.total_time == 0.0


# =============================================================================
# Silent Failure Detection Tests
# =============================================================================


class TestSilentFailureDetection:
    """Tests for silent failure detection in BenchmarkRunner."""

    def test_silent_failure_detected(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
    ) -> None:
        """Test that explainers returning x unchanged are marked as failed."""

        class IdentityExplainer(Counterfactual):
            """Explainer that always returns the input unchanged."""

            def __init__(self, model, data, **kwargs):
                self.model = model

            def explain(self, x, y_pred=None):
                return x.copy(), 0, {"method": "identity"}

        runner = BenchmarkRunner(
            datasets=[dataset_config],
            models=[model_config],
            explainers=[ExplainerConfig("identity", IdentityExplainer)],
            n_instances=3,
            verbose=False,
        )

        with pytest.warns(UserWarning, match="silent failure"):
            results = runner.run()

        result = results.get(dataset_config.name, model_config.name, "identity")
        assert result is not None
        # All instances should be marked as unsuccessful
        assert result.n_successful == 0
        # X_cf should be NaN for failed instances
        assert np.all(np.isnan(result.X_cf))

    def test_silent_failure_k_detected(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
    ) -> None:
        """Test silent failure detection for k > 1 counterfactuals."""

        class IdentityExplainerK(Counterfactual):
            """Explainer that always returns k copies of the input."""

            def __init__(self, model, data, **kwargs):
                self.model = model

            def explain(self, x, y_pred=None):
                return x.copy(), 0, {"method": "identity_k"}

            def explain_k(self, x, k=5, y_pred=None):
                cfs = np.array([x.copy() for _ in range(k)])
                labels = np.zeros(k, dtype=int)
                metas = [{"method": "identity_k", "k_index": i} for i in range(k)]
                return cfs, labels, metas

        runner = BenchmarkRunner(
            datasets=[dataset_config],
            models=[model_config],
            explainers=[ExplainerConfig("identity_k", IdentityExplainerK, n_counterfactuals=3)],
            n_instances=3,
            verbose=False,
        )

        with pytest.warns(UserWarning, match="silent failure"):
            results = runner.run()

        result = results.get(dataset_config.name, model_config.name, "identity_k")
        assert result is not None
        assert result.n_successful == 0

    def test_real_counterfactuals_not_flagged(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
    ) -> None:
        """Test that genuine CFs are not flagged as silent failures."""
        runner = BenchmarkRunner(
            datasets=[dataset_config],
            models=[model_config],
            explainers=[ExplainerConfig("comte", COMTE, {"distance": "euclidean"})],
            n_instances=3,
            verbose=False,
        )

        results = runner.run()
        result = results.get(dataset_config.name, model_config.name, "comte")
        assert result is not None
        # COMTE with KNN should produce actual counterfactuals
        assert result.n_successful > 0


# =============================================================================
# BenchmarkRunner Tests
# =============================================================================


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    def test_creation(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        explainer_configs: list[ExplainerConfig],
    ) -> None:
        """Test runner creation."""
        runner = BenchmarkRunner(
            datasets=[dataset_config],
            models=[model_config],
            explainers=explainer_configs,
        )

        assert len(runner.datasets) == 1
        assert len(runner.models) == 1
        assert len(runner.explainers) == 2

    def test_validation_empty_datasets(self) -> None:
        """Test validation rejects empty datasets."""
        with pytest.raises(ValueError, match="dataset"):
            BenchmarkRunner(
                datasets=[],
                models=[ModelConfig("test", None)],
                explainers=[ExplainerConfig("test", COMTE)],
            )

    def test_validation_duplicate_names(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
    ) -> None:
        """Test validation rejects duplicate names."""
        with pytest.raises(ValueError, match="Duplicate"):
            BenchmarkRunner(
                datasets=[dataset_config, dataset_config],
                models=[model_config],
                explainers=[ExplainerConfig("test", COMTE)],
            )

    def test_run_small(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
    ) -> None:
        """Test running a small benchmark."""
        runner = BenchmarkRunner(
            datasets=[dataset_config],
            models=[model_config],
            explainers=[ExplainerConfig("comte", COMTE, {"distance": "euclidean"})],
            n_instances=3,  # Small for speed
            verbose=False,
        )

        results = runner.run()

        assert len(results) == 1
        result = results.get(dataset_config.name, model_config.name, "comte")
        assert result is not None
        assert result.n_instances == 3

    def test_run_k_counterfactuals(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
    ) -> None:
        """Test running with k > 1 counterfactuals per instance."""
        k = 3
        runner = BenchmarkRunner(
            datasets=[dataset_config],
            models=[model_config],
            explainers=[
                ExplainerConfig("comte_k", COMTE, {"distance": "euclidean"}, n_counterfactuals=k)
            ],
            n_instances=2,  # Small for speed
            verbose=False,
        )

        results = runner.run()

        assert len(results) == 1
        result = results.get(dataset_config.name, model_config.name, "comte_k")
        assert result is not None
        assert result.n_instances == 2
        # Check that X_cf has the right shape for k counterfactuals
        assert result.X_cf.ndim == 3  # (n_instances, k, series_length)
        assert result.X_cf.shape[1] == k
        # Check that y_cf has the right shape
        assert result.y_cf.shape == (2, k)


# =============================================================================
# ParetoAnalyzer Tests
# =============================================================================


class TestParetoAnalyzer:
    """Tests for ParetoAnalyzer."""

    def test_creation(self) -> None:
        """Test analyzer creation."""
        analyzer = ParetoAnalyzer(["validity", "proximity_l2"])
        assert len(analyzer.metrics) == 2

    def test_validation_empty_metrics(self) -> None:
        """Test validation rejects empty metrics."""
        with pytest.raises(ValueError, match="metric"):
            ParetoAnalyzer([])

    def test_pareto_front(self, sample_results: BenchmarkResults) -> None:
        """Test finding Pareto front."""
        analyzer = ParetoAnalyzer(["validity", "proximity_l2"])
        pareto = analyzer.pareto_front(sample_results)

        assert isinstance(pareto, list)
        assert len(pareto) > 0

    def test_dominance_ranking(self, sample_results: BenchmarkResults) -> None:
        """Test dominance ranking."""
        analyzer = ParetoAnalyzer(["validity", "proximity_l2"])
        ranking = analyzer.dominance_ranking(sample_results)

        assert "name" in ranking.columns
        assert "dominated_by" in ranking.columns
        assert "pareto" in ranking.columns

    def test_dominance_count(self, sample_results: BenchmarkResults) -> None:
        """Test dominance count."""
        analyzer = ParetoAnalyzer(["validity", "proximity_l2"])
        counts = analyzer.dominance_count(sample_results)

        assert isinstance(counts, dict)
        assert all(isinstance(v, int) for v in counts.values())

    def test_custom_directions(self, sample_results: BenchmarkResults) -> None:
        """Test custom metric directions."""
        analyzer = ParetoAnalyzer(
            ["validity", "proximity_l2"],
            directions={"proximity_l2": "max"},  # Override to maximize
        )
        # Should not raise
        ranking = analyzer.dominance_ranking(sample_results)
        assert len(ranking) > 0

    def test_dominates_basic(self) -> None:
        """Test dominance logic."""
        analyzer = ParetoAnalyzer(["a", "b"])

        # a dominates b (both lower = better in minimization)
        assert analyzer._dominates(np.array([1.0, 1.0]), np.array([2.0, 2.0]))

        # Equal doesn't dominate
        assert not analyzer._dominates(np.array([1.0, 1.0]), np.array([1.0, 1.0]))

        # Mixed doesn't dominate
        assert not analyzer._dominates(np.array([1.0, 3.0]), np.array([2.0, 2.0]))


# =============================================================================
# Instance Selection Integration Tests
# =============================================================================


class TestInstanceSelectionIntegration:
    """Integration tests for instance selection through BenchmarkRunner."""

    def test_default_is_random(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
    ) -> None:
        """Test that the default selection strategy is random."""
        runner = BenchmarkRunner(
            datasets=[dataset_config],
            models=[model_config],
            explainers=[ExplainerConfig("comte", COMTE, {"distance": "euclidean"})],
            n_instances=3,
            verbose=False,
        )
        assert runner.instance_selection == "random"

        results = runner.run()
        result = results.get(dataset_config.name, model_config.name, "comte")
        assert result is not None
        assert result.n_instances == 3

    def test_stratified_confidence_selection(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
    ) -> None:
        """Test stratified confidence selection produces expected count."""
        runner = BenchmarkRunner(
            datasets=[dataset_config],
            models=[model_config],
            explainers=[ExplainerConfig("comte", COMTE, {"distance": "euclidean"})],
            n_instances=8,
            instance_selection="stratified_confidence",
            verbose=False,
        )

        results = runner.run()
        result = results.get(dataset_config.name, model_config.name, "comte")
        assert result is not None
        assert result.n_instances == 8

    def test_stratified_confidence_reproducible(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
    ) -> None:
        """Test that stratified confidence selection is reproducible."""
        kwargs: dict = {
            "datasets": [dataset_config],
            "models": [model_config],
            "explainers": [ExplainerConfig("comte", COMTE, {"distance": "euclidean"})],
            "n_instances": 8,
            "instance_selection": "stratified_confidence",
            "random_state": 42,
            "verbose": False,
        }

        r1 = BenchmarkRunner(**kwargs).run().get(dataset_config.name, model_config.name, "comte")
        r2 = BenchmarkRunner(**kwargs).run().get(dataset_config.name, model_config.name, "comte")
        assert r1 is not None and r2 is not None
        np.testing.assert_array_equal(r1.X_cf, r2.X_cf)

    def test_n_instances_none_uses_all(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
    ) -> None:
        """Test that n_instances=None uses all instances for both strategies."""
        for strategy in ["random", "stratified_confidence"]:
            runner = BenchmarkRunner(
                datasets=[dataset_config],
                models=[model_config],
                explainers=[ExplainerConfig("comte", COMTE, {"distance": "euclidean"})],
                n_instances=None,
                instance_selection=strategy,
                verbose=False,
            )

            results = runner.run()
            result = results.get(dataset_config.name, model_config.name, "comte")
            assert result is not None
            assert result.n_instances == dataset_config.n_test
