"""Tests for counterfactual utils module."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from tscf_eval.counterfactuals.utils import (
    dba_barycenter_multich,
    dtw_distance_vec_multich,
    ensure_batch_shape,
    euclidean_cdist_flat,
    predict_proba_fn,
    soft_predict_proba_fn,
    strip_batch,
    supports_soft_probabilities,
    weighted_dba_multich,
)

# =============================================================================
# Shape Utilities Tests
# =============================================================================


class TestEnsureBatchShape:
    """Tests for ensure_batch_shape function."""

    def test_1d_input(self) -> None:
        """Test 1D input gets batch dimension added."""
        x = np.array([1, 2, 3, 4, 5])
        xb, added = ensure_batch_shape(x)

        assert xb.shape == (1, 5)
        assert added is True

    def test_2d_input(self) -> None:
        """Test 2D input gets batch dimension added (treated as single multivariate)."""
        x = np.array([[1, 2, 3], [4, 5, 6]])  # (C, T) = (2, 3)
        xb, added = ensure_batch_shape(x)

        assert xb.shape == (1, 2, 3)
        assert added is True

    def test_3d_input(self) -> None:
        """Test 3D input remains unchanged."""
        x = np.random.randn(10, 3, 50)  # (N, C, T)
        xb, added = ensure_batch_shape(x)

        assert xb.shape == (10, 3, 50)
        assert added is False

    def test_4d_input_raises(self) -> None:
        """Test 4D input raises ValueError."""
        x = np.random.randn(5, 3, 10, 2)

        with pytest.raises(ValueError, match="Unsupported shape"):
            ensure_batch_shape(x)

    def test_preserves_data(self) -> None:
        """Test that data is preserved."""
        x = np.random.randn(100)
        xb, _ = ensure_batch_shape(x)

        np.testing.assert_array_almost_equal(xb.flatten(), x.flatten())


class TestStripBatch:
    """Tests for strip_batch function."""

    def test_strip_added_batch(self) -> None:
        """Test stripping batch dimension when it was added."""
        x = np.array([1, 2, 3, 4, 5])
        xb, added = ensure_batch_shape(x)
        stripped = strip_batch(xb, added)

        np.testing.assert_array_equal(stripped, x)
        assert stripped.shape == (5,)

    def test_no_strip_when_not_added(self) -> None:
        """Test no stripping when batch was not added."""
        x = np.random.randn(10, 3, 50)
        xb, added = ensure_batch_shape(x)
        result = strip_batch(xb, added)

        np.testing.assert_array_equal(result, x)
        assert result.shape == (10, 3, 50)

    def test_strip_2d_multivariate(self) -> None:
        """Test stripping multivariate 2D input."""
        x = np.array([[1, 2, 3], [4, 5, 6]])  # (C, T)
        xb, added = ensure_batch_shape(x)
        stripped = strip_batch(xb, added)

        np.testing.assert_array_equal(stripped, x)
        assert stripped.shape == (2, 3)


# =============================================================================
# Distance Functions Tests
# =============================================================================


class TestEuclideanCdistFlat:
    """Tests for euclidean_cdist_flat function."""

    def test_basic_distance(self) -> None:
        """Test basic Euclidean distance computation."""
        A = np.array([[0, 0], [1, 1]])  # (2, 2)
        B = np.array([[0, 0], [2, 2]])  # (2, 2)

        dist = euclidean_cdist_flat(A, B)

        assert dist.shape == (2, 2)
        # A[0] to B[0]: distance = 0
        assert dist[0, 0] == pytest.approx(0.0)
        # A[0] to B[1]: distance = sqrt(8)
        assert dist[0, 1] == pytest.approx(np.sqrt(8))

    def test_3d_input_flattened(self) -> None:
        """Test that 3D input is flattened correctly."""
        A = np.random.randn(5, 2, 10)  # (N, C, T)
        B = np.random.randn(3, 2, 10)

        dist = euclidean_cdist_flat(A, B)

        assert dist.shape == (5, 3)

    def test_self_distance(self) -> None:
        """Test distance to self is zero."""
        A = np.random.randn(10, 50)

        dist = euclidean_cdist_flat(A, A)

        # Diagonal should be zeros
        np.testing.assert_array_almost_equal(np.diag(dist), np.zeros(10))

    def test_multiple_queries(self) -> None:
        """Test distances from multiple queries."""
        queries = np.random.randn(10, 100)
        reference = np.random.randn(50, 100)

        distances = euclidean_cdist_flat(queries, reference)

        assert distances.shape == (10, 50)


class TestDTWDistanceVecMultich:
    """Tests for dtw_distance_vec_multich function."""

    def test_univariate_distance(self) -> None:
        """Test DTW distance for univariate series."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # (T,)
        B = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],  # Same as x
                [5.0, 4.0, 3.0, 2.0, 1.0],  # Reversed
            ]
        )  # (N, T)

        dist = dtw_distance_vec_multich(x, B)

        assert dist.shape == (2,)
        # First should be smaller (same series)
        assert dist[0] < dist[1]

    def test_multivariate_distance(self) -> None:
        """Test DTW distance for multivariate series."""
        x = np.random.randn(3, 20)  # (C, T)
        B = np.random.randn(5, 3, 20)  # (N, C, T)

        dist = dtw_distance_vec_multich(x, B)

        assert dist.shape == (5,)
        assert np.all(dist >= 0)

    def test_self_distance_near_zero(self) -> None:
        """Test that self-distance is near zero."""
        x = np.random.randn(50)  # (T,)
        B = x[None, :]  # (1, T) - same as x

        dist = dtw_distance_vec_multich(x, B)

        # Distance to self should be zero
        assert dist[0] == pytest.approx(0.0, abs=1e-6)


# =============================================================================
# Prediction Utilities Tests
# =============================================================================


class TestPredictProbaFn:
    """Tests for predict_proba_fn function."""

    def test_sklearn_classifier(self, univariate_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test with sklearn classifier."""
        X, y = univariate_data
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X, y)

        proba_fn = predict_proba_fn(clf)
        probs = proba_fn(X[:5])

        assert probs.shape == (5, 2)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        np.testing.assert_array_almost_equal(probs.sum(axis=1), np.ones(5))

    def test_returns_callable(self, simple_classifier: KNeighborsClassifier) -> None:
        """Test that function returns a callable."""
        proba_fn = predict_proba_fn(simple_classifier)

        assert callable(proba_fn)

    def test_handles_array_conversion(self, univariate_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that input is converted to array."""
        X, y = univariate_data
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X, y)

        proba_fn = predict_proba_fn(clf)

        # Pass as list
        result = proba_fn(X[:2].tolist())

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_normalizes_output(self, univariate_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that output probabilities sum to 1."""
        X, y = univariate_data
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X, y)

        predict_proba = predict_proba_fn(clf)
        proba = predict_proba(X[:10])

        np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(10))


class TestSoftPredictProbaFn:
    """Tests for soft_predict_proba_fn function."""

    def test_with_logistic_regression(self, univariate_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test with model having decision_function."""
        X, y = univariate_data
        clf = LogisticRegression(max_iter=200)
        clf.fit(X, y)

        soft_proba = soft_predict_proba_fn(clf)
        probs = soft_proba(X[:5])

        assert probs.shape == (5, 2)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_with_knn_fallback(self, univariate_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that KNN falls back to regular predict_proba."""
        X, y = univariate_data
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X, y)

        soft_proba = soft_predict_proba_fn(clf)
        probs = soft_proba(X[:5])

        assert probs.shape == (5, 2)


class TestSupportsSoftProbabilities:
    """Tests for supports_soft_probabilities function."""

    def test_logistic_regression_supports(
        self, univariate_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that LogisticRegression supports soft probabilities."""
        X, y = univariate_data
        clf = LogisticRegression(max_iter=200)
        clf.fit(X, y)

        assert supports_soft_probabilities(clf) is True

    def test_knn_supports(self, univariate_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that KNN supports soft probabilities (default True)."""
        X, y = univariate_data
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X, y)

        # KNN doesn't have decision_function, so falls back to True
        assert supports_soft_probabilities(clf) is True


# =============================================================================
# DBA Utilities Tests
# =============================================================================


class TestWeightedDbaMultich:
    """Tests for weighted_dba_multich function."""

    def test_univariate_weighting(self) -> None:
        """Test weighted DBA for univariate series."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        # weight_guide = 0 should return exactly a (query)
        result_0 = weighted_dba_multich(a, b, weight_guide=0.0)
        np.testing.assert_array_almost_equal(result_0, a)

        # weight_guide = 0.5 should produce a result between a and b
        result_mid = weighted_dba_multich(a, b, weight_guide=0.5)
        assert result_mid.shape == a.shape
        # Result should be somewhere between a and b
        assert np.mean(result_mid) > np.mean(a)
        assert np.mean(result_mid) < np.mean(b)

    def test_multivariate_weighting(self) -> None:
        """Test weighted DBA for multivariate series."""
        a = np.random.randn(3, 20)  # (C, T)
        b = np.random.randn(3, 20)

        result = weighted_dba_multich(a, b, weight_guide=0.5)

        assert result.shape == a.shape


class TestDbaBarycenterMultich:
    """Tests for dba_barycenter_multich function."""

    def test_univariate_barycenter(self) -> None:
        """Test DBA barycenter for univariate series."""
        X = np.random.randn(5, 20)  # (N, T)

        result = dba_barycenter_multich(X)

        assert result.shape == (20,)

    def test_multivariate_barycenter(self) -> None:
        """Test DBA barycenter for multivariate series."""
        X = np.random.randn(5, 3, 20)  # (N, C, T)

        result = dba_barycenter_multich(X)

        assert result.shape == (3, 20)
