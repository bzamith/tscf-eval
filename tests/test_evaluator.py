"""Tests for the evaluator module."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from tscf_eval.evaluator import (
    Composition,
    Confidence,
    Contiguity,
    Controllability,
    Diversity,
    Efficiency,
    Evaluator,
    Metric,
    Plausibility,
    Proximity,
    Robustness,
    Sparsity,
    Validity,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_cf_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Create simple counterfactual data for metric testing."""
    n = 20
    series_length = 50

    # Original instances
    X = rng.normal(0, 1, (n, series_length))

    # Counterfactuals with small perturbations
    X_cf = X + rng.normal(0, 0.1, X.shape)

    return X, X_cf


@pytest.fixture
def labeled_cf_data(
    simple_cf_data: tuple[np.ndarray, np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create labeled counterfactual data."""
    X, X_cf = simple_cf_data
    n = X.shape[0]

    y = rng.integers(0, 2, n)
    y_cf = 1 - y  # Flip labels for counterfactuals

    return X, X_cf, y, y_cf


@pytest.fixture
def classifier_and_data(
    labeled_cf_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[KNeighborsClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a classifier and data for metric testing."""
    X, X_cf, y, y_cf = labeled_cf_data

    # Create training data for classifier
    X_train = np.vstack([X, X_cf])
    y_train = np.concatenate([y, y_cf])

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    return clf, X, X_cf, y, y_cf


# =============================================================================
# Metric Base Tests
# =============================================================================


class TestMetricBase:
    """Tests for Metric abstract base class."""

    def test_is_abstract(self) -> None:
        """Test that Metric is abstract."""
        with pytest.raises(TypeError):
            Metric()  # type: ignore[abstract]

    def test_subclass_must_implement_name_and_compute(self) -> None:
        """Test that subclasses must implement name and compute."""

        class IncompleteMetric(Metric):
            pass

        with pytest.raises(TypeError):
            IncompleteMetric()  # type: ignore[abstract]


# =============================================================================
# Core Metrics Tests
# =============================================================================


class TestValidity:
    """Tests for Validity metric."""

    def test_name(self) -> None:
        """Test metric name."""
        metric = Validity()
        assert metric.name() == "validity_soft"

    def test_direction(self) -> None:
        """Test optimization direction."""
        metric = Validity()
        assert metric.direction == "maximize"

    def test_compute_with_labels(
        self,
        simple_cf_data: tuple[np.ndarray, np.ndarray],
        rng: np.random.Generator,
    ) -> None:
        """Test computing validity with labels."""
        X, X_cf = simple_cf_data
        n = X.shape[0]

        y = rng.integers(0, 2, n)
        y_cf = 1 - y  # All counterfactuals have different labels

        metric = Validity()
        result = metric.compute(X, X_cf, y=y, y_cf=y_cf)

        assert 0.0 <= result <= 1.0
        assert result == 1.0  # All labels are different

    def test_compute_with_model(
        self,
        classifier_and_data: tuple[
            KNeighborsClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ],
    ) -> None:
        """Test computing validity with a model."""
        clf, X, X_cf, y, _y_cf = classifier_and_data

        metric = Validity()
        result = metric.compute(X, X_cf, model=clf, y=y)

        assert 0.0 <= result <= 1.0

    def test_name_soft(self) -> None:
        """Test soft validity metric name."""
        metric = Validity(mode="soft")
        assert metric.name() == "validity_soft"

    def test_name_soft_default(self) -> None:
        """Test that default mode is soft and name is 'validity_soft'."""
        metric = Validity()
        assert metric.mode == "soft"
        assert metric.name() == "validity_soft"

    def test_invalid_mode_raises(self) -> None:
        """Test that an invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown validity mode"):
            Validity(mode="invalid")

    def test_soft_with_model(
        self,
        classifier_and_data: tuple[
            KNeighborsClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ],
    ) -> None:
        """Test soft validity with a model."""
        clf, X, X_cf, _y, _y_cf = classifier_and_data

        metric = Validity(mode="soft")
        result = metric.compute(X, X_cf, model=clf)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_soft_perfect_flip(self) -> None:
        """Test soft validity when model is very confident about the flip."""
        X = np.array([[0.0, 0.0], [0.0, 0.0]])
        X_cf = np.array([[10.0, 10.0], [10.0, 10.0]])

        # Mock model: predict class 0 for X, class 1 for X_cf
        class MockModel:
            def predict(self, data: np.ndarray) -> np.ndarray:
                return (data[:, 0] > 5).astype(int)

            def predict_proba(self, data: np.ndarray) -> np.ndarray:
                p = 1.0 / (1.0 + np.exp(-data[:, 0] + 5))
                return np.column_stack([1 - p, p])

        metric = Validity(mode="soft")
        result = metric.compute(X, X_cf, model=MockModel())

        # Target class is 1 (cf prediction), P(1|X_cf) should be high, P(1|X) low
        assert result > 0.5

    def test_soft_no_change(self) -> None:
        """Test soft validity when X and X_cf are identical (no probability shift)."""
        X = np.random.randn(10, 50)
        X_cf = X.copy()

        class MockModel:
            def predict(self, data: np.ndarray) -> np.ndarray:
                return np.zeros(len(data), dtype=int)

            def predict_proba(self, data: np.ndarray) -> np.ndarray:
                n = len(data)
                return np.column_stack([np.full(n, 0.8), np.full(n, 0.2)])

        metric = Validity(mode="soft")
        result = metric.compute(X, X_cf, model=MockModel())

        assert result == pytest.approx(0.0)

    def test_soft_fallback_to_hard_with_labels(self) -> None:
        """Test that soft mode falls back to hard when only labels are given."""
        X = np.random.randn(10, 50)
        X_cf = X + 0.1
        y = np.zeros(10, dtype=int)
        y_cf = np.ones(10, dtype=int)

        hard = Validity(mode="hard").compute(X, X_cf, y=y, y_cf=y_cf)
        soft = Validity(mode="soft").compute(X, X_cf, y=y, y_cf=y_cf)

        assert soft == hard  # Falls back to hard validity

    def test_soft_no_model_no_labels_raises(self) -> None:
        """Test that soft mode raises when neither model nor labels given."""
        X = np.random.randn(5, 20)
        metric = Validity(mode="soft")
        with pytest.raises(ValueError, match="predict_proba"):
            metric.compute(X, X.copy())

    def test_soft_model_without_predict_proba_raises(self) -> None:
        """Test that soft mode raises when model lacks predict_proba."""
        X = np.random.randn(5, 20)

        class NoProbaCls:
            def predict(self, data: np.ndarray) -> np.ndarray:
                return np.zeros(len(data), dtype=int)

        metric = Validity(mode="soft")
        with pytest.raises(ValueError, match="predict_proba"):
            metric.compute(X, X.copy(), model=NoProbaCls())

    def test_soft_noncontiguous_class_labels(self) -> None:
        """Test soft validity with non-contiguous class labels (e.g., {0, 2})."""
        X = np.random.randn(6, 20)
        X_cf = X + np.random.randn(6, 20) * 0.5

        class NonContigModel:
            classes_ = np.array([0, 2])

            def predict(self, data: np.ndarray) -> np.ndarray:
                # First 3 predict class 0, last 3 predict class 2
                return np.array([0, 0, 0, 2, 2, 2])

            def predict_proba(self, data: np.ndarray) -> np.ndarray:
                n = len(data)
                proba = np.zeros((n, 2))
                proba[:3, 0] = 0.9
                proba[:3, 1] = 0.1
                proba[3:, 0] = 0.1
                proba[3:, 1] = 0.9
                return proba

        metric = Validity(mode="soft")
        result = metric.compute(X, X_cf, model=NonContigModel())
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestProximity:
    """Tests for Proximity metric."""

    def test_name_l1(self) -> None:
        """Test L1 proximity name."""
        metric = Proximity(p=1, distance="lp")
        assert metric.name() == "proximity_l1"

    def test_name_l2(self) -> None:
        """Test L2 proximity name."""
        metric = Proximity(p=2, distance="lp")
        assert metric.name() == "proximity_l2"

    def test_name_linf(self) -> None:
        """Test L-inf proximity name."""
        metric = Proximity(p=float("inf"), distance="lp")
        assert metric.name() == "proximity_linf"

    def test_direction(self) -> None:
        """Test optimization direction."""
        metric = Proximity()
        assert metric.direction == "maximize"

    def test_compute(self, simple_cf_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test computing proximity."""
        X, X_cf = simple_cf_data

        metric = Proximity(p=2)
        result = metric.compute(X, X_cf)

        assert 0.0 < result <= 1.0
        assert isinstance(result, float)

    def test_identical_instances_one(self) -> None:
        """Test that identical instances have proximity 1.0."""
        X = np.random.randn(10, 50)
        X_cf = X.copy()

        metric = Proximity(p=2)
        result = metric.compute(X, X_cf)

        assert result == pytest.approx(1.0)

    def test_name_dtw(self) -> None:
        """Test DTW proximity name."""
        metric = Proximity(distance="dtw")
        assert metric.name() == "proximity_dtw"

    def test_compute_dtw(self, simple_cf_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test computing DTW proximity."""
        X, X_cf = simple_cf_data

        metric = Proximity(distance="dtw")
        result = metric.compute(X, X_cf)

        assert 0.0 < result <= 1.0
        assert isinstance(result, float)

    def test_identical_dtw(self) -> None:
        """Test that identical instances have DTW proximity 1.0."""
        X = np.random.randn(10, 50)
        X_cf = X.copy()

        metric = Proximity(distance="dtw")
        result = metric.compute(X, X_cf)

        assert result == pytest.approx(1.0)

    def test_invalid_distance_raises(self) -> None:
        """Test that an invalid distance raises ValueError."""
        with pytest.raises(ValueError, match="distance must be one of"):
            Proximity(distance="bad")


class TestSparsity:
    """Tests for Sparsity metric."""

    def test_name(self) -> None:
        """Test metric name."""
        metric = Sparsity()
        assert metric.name() == "sparsity"

    def test_direction(self) -> None:
        """Test optimization direction."""
        metric = Sparsity()
        assert metric.direction == "minimize"

    def test_compute(self, simple_cf_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test computing sparsity."""
        X, X_cf = simple_cf_data

        metric = Sparsity()
        result = metric.compute(X, X_cf)

        assert 0.0 <= result <= 1.0

    def test_identical_instances_zero_sparsity(self) -> None:
        """Test that identical instances have zero sparsity (no changes)."""
        X = np.random.randn(10, 50)
        X_cf = X.copy()

        metric = Sparsity()
        result = metric.compute(X, X_cf)

        assert result == pytest.approx(0.0)


# =============================================================================
# Distribution Metrics Tests
# =============================================================================


class TestPlausibility:
    """Tests for Plausibility metric."""

    def test_name_lof(self) -> None:
        """Test LOF plausibility name."""
        metric = Plausibility(method="lof")
        assert metric.name() == "plausibility_lof"

    def test_name_if(self) -> None:
        """Test IF plausibility name."""
        metric = Plausibility(method="if")
        assert metric.name() == "plausibility_if"

    def test_direction(self) -> None:
        """Test optimization direction."""
        metric = Plausibility()
        assert metric.direction == "maximize"

    def test_compute_lof(self, simple_cf_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test computing LOF plausibility."""
        X, X_cf = simple_cf_data

        metric = Plausibility(method="lof")
        result = metric.compute(X, X_cf, X_train=X)

        assert 0.0 <= result <= 1.0

    def test_compute_if(self, simple_cf_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test computing IF plausibility."""
        X, X_cf = simple_cf_data

        metric = Plausibility(method="if")
        result = metric.compute(X, X_cf, X_train=X)

        assert 0.0 <= result <= 1.0

    def test_name_dtw_lof(self) -> None:
        """Test DTW-LOF plausibility name."""
        metric = Plausibility(method="dtw_lof")
        assert metric.name() == "plausibility_dtw_lof"

    def test_compute_dtw_lof(self, simple_cf_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test computing DTW-LOF plausibility."""
        X, X_cf = simple_cf_data

        metric = Plausibility(method="dtw_lof")
        result = metric.compute(X, X_cf, X_train=X)

        assert 0.0 <= result <= 1.0
        assert isinstance(result, float)

    def test_dtw_lof_with_n_neighbors(self, simple_cf_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test DTW-LOF with custom n_neighbors kwarg."""
        X, X_cf = simple_cf_data

        metric = Plausibility(method="dtw_lof", n_neighbors=5)
        result = metric.compute(X, X_cf, X_train=X)

        assert 0.0 <= result <= 1.0

    def test_dtw_lof_cache_reuse(self, simple_cf_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that DTW-LOF caches the fitted detector."""
        X, X_cf = simple_cf_data

        metric = Plausibility(method="dtw_lof")
        r1 = metric.compute(X, X_cf, X_train=X)
        r2 = metric.compute(X, X_cf, X_train=X)

        assert r1 == r2
        # Cache should have one entry
        assert len(metric._detector_cache) == 1

    def test_clear_cache(self) -> None:
        """Test cache clearing."""
        metric = Plausibility()
        metric.clear_cache()
        # Should not raise


class TestDiversity:
    """Tests for Diversity metric."""

    def test_name(self) -> None:
        """Test metric name."""
        metric = Diversity()
        assert metric.name() == "diversity_dpp_dtw"

    def test_direction(self) -> None:
        """Test optimization direction."""
        metric = Diversity()
        assert metric.direction == "maximize"

    def test_compute_with_k_counterfactuals(self, rng: np.random.Generator) -> None:
        """Test computing diversity with k counterfactuals."""
        n = 10
        k = 5
        series_length = 50

        X = rng.normal(0, 1, (n, series_length))
        X_cf = rng.normal(0, 1, (n, k, series_length))  # k CFs per instance

        metric = Diversity()
        result = metric.compute(X, X_cf[:, 0], _X_cf_all=X_cf)

        assert isinstance(result, float)

    def test_compute_single_cf_returns_nan(
        self, simple_cf_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that single CF returns NaN."""
        X, X_cf = simple_cf_data

        metric = Diversity()
        result = metric.compute(X, X_cf)

        assert np.isnan(result)

    def test_name_dtw(self) -> None:
        """Test DTW diversity name."""
        metric = Diversity(distance="dtw")
        assert metric.name() == "diversity_dpp_dtw"

    def test_name_dtw_default(self) -> None:
        """Test that default distance is dtw."""
        metric = Diversity()
        assert metric.distance == "dtw"
        assert metric.name() == "diversity_dpp_dtw"

    def test_compute_dtw(self, rng: np.random.Generator) -> None:
        """Test computing DTW diversity with k counterfactuals."""
        n = 10
        k = 5
        series_length = 50

        X = rng.normal(0, 1, (n, series_length))
        X_cf = rng.normal(0, 1, (n, k, series_length))

        metric = Diversity(distance="dtw")
        result = metric.compute(X, X_cf[:, 0], _X_cf_all=X_cf)

        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_dtw_single_cf_returns_nan(self, simple_cf_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that DTW single CF also returns NaN."""
        X, X_cf = simple_cf_data

        metric = Diversity(distance="dtw")
        result = metric.compute(X, X_cf)

        assert np.isnan(result)

    def test_invalid_distance_raises(self, rng: np.random.Generator) -> None:
        """Test that invalid distance raises ValueError."""
        n = 5
        k = 3
        series_length = 20

        X = rng.normal(0, 1, (n, series_length))
        X_cf = rng.normal(0, 1, (n, k, series_length))

        metric = Diversity(distance="bad")
        with pytest.raises(ValueError, match="Unknown distance"):
            metric.compute(X, X_cf[:, 0], _X_cf_all=X_cf)


# =============================================================================
# Model Metrics Tests
# =============================================================================


class TestControllability:
    """Tests for Controllability metric."""

    def test_name(self) -> None:
        """Test metric name."""
        metric = Controllability()
        assert metric.name() == "controllability"

    def test_compute(
        self,
        classifier_and_data: tuple[
            KNeighborsClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ],
    ) -> None:
        """Test computing controllability."""
        clf, X, X_cf, y, y_cf = classifier_and_data

        metric = Controllability()
        result = metric.compute(X, X_cf, model=clf, y=y, y_cf=y_cf)

        assert 0.0 <= result <= 1.0

    def test_custom_fractions_and_samples(
        self,
        classifier_and_data: tuple[
            KNeighborsClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ],
    ) -> None:
        """Test with custom revert fractions and sample count."""
        clf, X, X_cf, y, y_cf = classifier_and_data

        metric = Controllability(revert_fractions=[0.25, 0.5], n_samples=5)
        result = metric.compute(X, X_cf, model=clf, y=y, y_cf=y_cf)

        assert 0.0 <= result <= 1.0

    def test_reproducibility(
        self,
        classifier_and_data: tuple[
            KNeighborsClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ],
    ) -> None:
        """Test that random_state produces deterministic results."""
        clf, X, X_cf, y, y_cf = classifier_and_data

        metric_a = Controllability(random_state=42)
        metric_b = Controllability(random_state=42)

        result_a = metric_a.compute(X, X_cf, model=clf, y=y, y_cf=y_cf)
        result_b = metric_b.compute(X, X_cf, model=clf, y=y, y_cf=y_cf)

        assert result_a == result_b


class TestConfidence:
    """Tests for Confidence metric."""

    def test_name(self) -> None:
        """Test metric name."""
        metric = Confidence()
        assert metric.name() == "confidence"

    def test_compute_returns_dict(
        self,
        classifier_and_data: tuple[
            KNeighborsClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ],
    ) -> None:
        """Test computing confidence returns a dictionary."""
        clf, X, X_cf, _y, _y_cf = classifier_and_data

        metric = Confidence()
        result = metric.compute(X, X_cf, model=clf)

        # Confidence returns a dict with multiple values
        assert isinstance(result, dict)
        assert "mean_conf_orig" in result
        assert "mean_conf_cf" in result
        assert "mean_conf_delta" in result
        assert 0.0 <= result["mean_conf_orig"] <= 1.0
        assert 0.0 <= result["mean_conf_cf"] <= 1.0


class TestComposition:
    """Tests for Composition metric."""

    def test_name(self) -> None:
        """Test metric name."""
        metric = Composition()
        assert metric.name() == "composition"


# =============================================================================
# Structure Metrics Tests
# =============================================================================


class TestContiguity:
    """Tests for Contiguity metric."""

    def test_name(self) -> None:
        """Test metric name."""
        metric = Contiguity()
        assert metric.name() == "contiguity"

    def test_direction(self) -> None:
        """Test optimization direction."""
        metric = Contiguity()
        assert metric.direction == "maximize"

    def test_compute(self, simple_cf_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test computing contiguity."""
        X, X_cf = simple_cf_data

        metric = Contiguity()
        result = metric.compute(X, X_cf)

        assert 0.0 <= result <= 1.0


# =============================================================================
# Stability Metrics Tests
# =============================================================================


class TestRobustness:
    """Tests for Robustness metric."""

    def test_name(self) -> None:
        """Test metric name."""
        metric = Robustness()
        assert metric.name() == "robustness_lipschitz_dtw"

    def test_direction(self) -> None:
        """Test optimization direction."""
        metric = Robustness()
        assert metric.direction == "minimize"

    def test_compute(self, simple_cf_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test computing robustness."""
        X, X_cf = simple_cf_data

        metric = Robustness()
        result = metric.compute(X, X_cf, X_train=X)

        assert isinstance(result, float)

    def test_name_dtw(self) -> None:
        """Test metric name with DTW distance."""
        metric = Robustness(distance="dtw")
        assert metric.name() == "robustness_lipschitz_dtw"

    def test_compute_dtw(self, simple_cf_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test computing robustness with DTW distance."""
        X, X_cf = simple_cf_data

        metric = Robustness(distance="dtw")
        result = metric.compute(X, X_cf, X_train=X)

        assert isinstance(result, float)

    def test_invalid_distance(self) -> None:
        """Test that invalid distance raises ValueError."""
        with pytest.raises(ValueError, match="distance must be one of"):
            Robustness(distance="invalid")


# =============================================================================
# Performance Metrics Tests
# =============================================================================


class TestEfficiency:
    """Tests for Efficiency metric."""

    def test_name(self) -> None:
        """Test metric name."""
        metric = Efficiency()
        assert metric.name() == "efficiency_time_s"

    def test_direction(self) -> None:
        """Test optimization direction."""
        metric = Efficiency()
        assert metric.direction == "minimize"

    def test_compute(self, simple_cf_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test computing efficiency."""
        X, X_cf = simple_cf_data
        times = [0.1] * len(X)

        metric = Efficiency()
        result = metric.compute(X, X_cf, time_per_instance=times)

        assert result == pytest.approx(0.1)


# =============================================================================
# Evaluator Tests
# =============================================================================


class TestEvaluator:
    """Tests for Evaluator class."""

    def test_creation(self) -> None:
        """Test Evaluator creation."""
        metrics = [Validity(), Proximity(p=2), Sparsity()]
        evaluator = Evaluator(metrics)

        assert len(evaluator.metrics) == 3

    def test_evaluate(
        self,
        classifier_and_data: tuple[
            KNeighborsClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ],
    ) -> None:
        """Test evaluate method."""
        clf, X, X_cf, y, y_cf = classifier_and_data

        evaluator = Evaluator([Validity(), Proximity(p=2, distance="lp"), Sparsity()])
        results = evaluator.evaluate(X, X_cf, model=clf, y=y, y_cf=y_cf)

        assert "validity_soft" in results
        assert "proximity_l2" in results
        assert "sparsity" in results
        assert "_evaluator_time_s" in results

    def test_evaluate_shape_mismatch_raises(self) -> None:
        """Test that mismatched shapes raise error."""
        evaluator = Evaluator([Validity()])

        X = np.random.randn(10, 50)
        X_cf = np.random.randn(15, 50)  # Different n_instances

        with pytest.raises(ValueError, match="same number"):
            evaluator.evaluate(X, X_cf)

    def test_evaluate_with_timing_requires_efficiency_metric(self) -> None:
        """Test that timing requires Efficiency metric."""
        evaluator = Evaluator([Validity()])

        X = np.random.randn(10, 50)
        X_cf = np.random.randn(10, 50)
        times = [0.1] * 10

        with pytest.raises(ValueError, match="Efficiency"):
            evaluator.evaluate(X, X_cf, time_per_instance=times)

    def test_evaluate_with_all_metrics(
        self,
        classifier_and_data: tuple[
            KNeighborsClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ],
    ) -> None:
        """Test with all available metrics."""
        clf, X, X_cf, y, y_cf = classifier_and_data
        times = [0.1] * len(X)

        evaluator = Evaluator(
            [
                Validity(),
                Proximity(p=1),
                Proximity(p=2),
                Sparsity(),
                Plausibility(method="lof"),
                Controllability(),
                Confidence(),
                Contiguity(),
                Robustness(),
                Efficiency(),
            ]
        )

        results = evaluator.evaluate(
            X, X_cf, model=clf, X_train=X, y=y, y_cf=y_cf, time_per_instance=times
        )

        assert len(results) > 5
