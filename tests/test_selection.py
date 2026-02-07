"""Tests for benchmark instance selection strategies."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from tscf_eval.benchmark.config import DatasetConfig, ModelConfig
from tscf_eval.benchmark.selection import (
    N_CONFIDENCE_BINS,
    _select_random,
    _select_stratified_confidence,
    compute_confidence_bins,
    select_instances,
)


@pytest.fixture
def selection_setup(
    univariate_data: tuple[np.ndarray, np.ndarray],
) -> tuple[DatasetConfig, ModelConfig]:
    """Create dataset and fitted model for selection tests."""
    X, y = univariate_data
    X_train, X_test = X[:40], X[40:]
    y_train, y_test = y[:40], y[40:]

    dataset = DatasetConfig("test", X_train, y_train, X_test, y_test)

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    model = ModelConfig("knn", clf)

    return dataset, model


# =============================================================================
# _select_random
# =============================================================================


class TestSelectRandom:
    """Tests for random selection."""

    def test_correct_count(self) -> None:
        """Test that the correct number of indices is returned."""
        indices = _select_random(100, 10, random_state=42)
        assert len(indices) == 10
        assert len(np.unique(indices)) == 10

    def test_reproducible(self) -> None:
        """Test that results are reproducible with the same seed."""
        a = _select_random(100, 10, random_state=42)
        b = _select_random(100, 10, random_state=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds(self) -> None:
        """Test that different seeds produce different results."""
        a = _select_random(100, 10, random_state=42)
        b = _select_random(100, 10, random_state=99)
        assert not np.array_equal(a, b)


# =============================================================================
# _select_stratified_confidence
# =============================================================================


class TestSelectStratifiedConfidence:
    """Tests for stratified confidence selection."""

    def test_correct_count(self, selection_setup: tuple[DatasetConfig, ModelConfig]) -> None:
        """Test that the correct number of indices is returned."""
        dataset, model = selection_setup
        indices, bins = _select_stratified_confidence(
            dataset.X_test, model, n_instances=8, random_state=42
        )
        assert len(indices) == 8
        assert len(np.unique(indices)) == 8
        assert bins is not None
        assert len(bins) == 8

    def test_reproducible(self, selection_setup: tuple[DatasetConfig, ModelConfig]) -> None:
        """Test that results are reproducible with the same seed."""
        dataset, model = selection_setup
        a, a_bins = _select_stratified_confidence(
            dataset.X_test, model, n_instances=8, random_state=42
        )
        b, b_bins = _select_stratified_confidence(
            dataset.X_test, model, n_instances=8, random_state=42
        )
        np.testing.assert_array_equal(a, b)
        assert a_bins is not None and b_bins is not None
        np.testing.assert_array_equal(a_bins, b_bins)

    def test_fallback_no_predict_proba(
        self, selection_setup: tuple[DatasetConfig, ModelConfig]
    ) -> None:
        """Test fallback to random when model lacks predict_proba."""
        dataset, _ = selection_setup

        class NoProbaCls:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.zeros(len(X), dtype=int)

        model = ModelConfig("no_proba", NoProbaCls())

        with pytest.warns(UserWarning, match="does not support predict_proba"):
            indices, bins = _select_stratified_confidence(
                dataset.X_test, model, n_instances=8, random_state=42
            )
        assert len(indices) == 8
        assert bins is None

    def test_fallback_few_instances(
        self, selection_setup: tuple[DatasetConfig, ModelConfig]
    ) -> None:
        """Test fallback to random when n_instances < number of bins."""
        dataset, model = selection_setup

        with pytest.warns(UserWarning, match="less than the number of confidence bins"):
            indices, bins = _select_stratified_confidence(
                dataset.X_test, model, n_instances=2, random_state=42
            )
        assert len(indices) == 2
        assert bins is None

    def test_uneven_allocation(self, selection_setup: tuple[DatasetConfig, ModelConfig]) -> None:
        """Test that remainder instances are distributed correctly."""
        dataset, model = selection_setup
        # 9 instances across 4 bins: 3+2+2+2 or similar
        indices, bins = _select_stratified_confidence(
            dataset.X_test, model, n_instances=9, random_state=42
        )
        assert len(indices) == 9
        assert len(np.unique(indices)) == 9
        assert bins is not None
        assert len(bins) == 9

    def test_bins_match_full_test_set(
        self, selection_setup: tuple[DatasetConfig, ModelConfig]
    ) -> None:
        """Test that returned bins reflect the full test set binning."""
        dataset, model = selection_setup
        indices, bins = _select_stratified_confidence(
            dataset.X_test, model, n_instances=8, random_state=42
        )
        assert bins is not None

        # Compute bins over the full test set for comparison
        full_bins = compute_confidence_bins(dataset.X_test, model)
        assert full_bins is not None

        # The returned bins must match the full-test-set assignments
        np.testing.assert_array_equal(bins, full_bins[indices])


# =============================================================================
# compute_confidence_bins
# =============================================================================


class TestComputeConfidenceBins:
    """Tests for compute_confidence_bins."""

    def test_returns_correct_length(
        self, selection_setup: tuple[DatasetConfig, ModelConfig]
    ) -> None:
        """Test that bin array matches the number of test instances."""
        dataset, model = selection_setup
        bins = compute_confidence_bins(dataset.X_test, model)
        assert bins is not None
        assert len(bins) == len(dataset.X_test)

    def test_values_in_range(self, selection_setup: tuple[DatasetConfig, ModelConfig]) -> None:
        """Test that all bin indices are in {0, ..., N_CONFIDENCE_BINS - 1}."""
        dataset, model = selection_setup
        bins = compute_confidence_bins(dataset.X_test, model)
        assert bins is not None
        assert np.all(bins >= 0)
        assert np.all(bins < N_CONFIDENCE_BINS)

    def test_returns_none_without_predict_proba(
        self, selection_setup: tuple[DatasetConfig, ModelConfig]
    ) -> None:
        """Test that None is returned when model lacks predict_proba."""
        dataset, _ = selection_setup

        class NoProbaCls:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.zeros(len(X), dtype=int)

        model = ModelConfig("no_proba", NoProbaCls())
        assert compute_confidence_bins(dataset.X_test, model) is None

    def test_deterministic(self, selection_setup: tuple[DatasetConfig, ModelConfig]) -> None:
        """Test that the same inputs produce the same bins."""
        dataset, model = selection_setup
        a = compute_confidence_bins(dataset.X_test, model)
        b = compute_confidence_bins(dataset.X_test, model)
        assert a is not None and b is not None
        np.testing.assert_array_equal(a, b)


class TestSelectInstances:
    """Tests for the top-level select_instances dispatcher."""

    def test_none_returns_all(self, selection_setup: tuple[DatasetConfig, ModelConfig]) -> None:
        """Test that n_instances=None returns all instances."""
        dataset, model = selection_setup
        X, y, bins = select_instances(dataset, model, None, "random", 42)
        assert len(X) == dataset.n_test
        assert y is not None
        assert len(y) == dataset.n_test
        assert bins is None

    def test_n_instances_ge_total(self, selection_setup: tuple[DatasetConfig, ModelConfig]) -> None:
        """Test that n_instances >= n_test returns all instances."""
        dataset, model = selection_setup
        X, _y, bins = select_instances(dataset, model, 999, "random", 42)
        assert len(X) == dataset.n_test
        assert bins is None

    def test_random_strategy(self, selection_setup: tuple[DatasetConfig, ModelConfig]) -> None:
        """Test random strategy returns correct count."""
        dataset, model = selection_setup
        X, y, bins = select_instances(dataset, model, 5, "random", 42)
        assert len(X) == 5
        assert y is not None
        assert len(y) == 5
        assert bins is None

    def test_stratified_confidence_strategy(
        self, selection_setup: tuple[DatasetConfig, ModelConfig]
    ) -> None:
        """Test stratified_confidence strategy returns correct count."""
        dataset, model = selection_setup
        X, y, bins = select_instances(dataset, model, 8, "stratified_confidence", 42)
        assert len(X) == 8
        assert y is not None
        assert len(y) == 8
        assert bins is not None
        assert len(bins) == 8

    def test_invalid_strategy_raises(
        self, selection_setup: tuple[DatasetConfig, ModelConfig]
    ) -> None:
        """Test that an invalid strategy raises ValueError."""
        dataset, model = selection_setup
        with pytest.raises(ValueError, match="Unknown selection strategy"):
            select_instances(dataset, model, 5, "invalid_strategy", 42)  # type: ignore[arg-type]

    def test_no_y_test(self) -> None:
        """Test that y_test=None is handled correctly."""
        X_train = np.random.randn(20, 50)
        y_train = np.zeros(20, dtype=int)
        X_test = np.random.randn(10, 50)

        dataset = DatasetConfig("no_y", X_train, y_train, X_test)
        model = ModelConfig("dummy", KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train))

        X, y, _bins = select_instances(dataset, model, 5, "random", 42)
        assert len(X) == 5
        assert y is None
