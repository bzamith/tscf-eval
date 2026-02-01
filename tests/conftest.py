"""Pytest configuration and shared fixtures for tscf-eval tests."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from tscf_eval.data_loader import TSCData


@pytest.fixture
def random_seed() -> int:
    """Provide a fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def rng(random_seed: int) -> np.random.Generator:
    """Provide a numpy random generator with fixed seed."""
    return np.random.default_rng(random_seed)


@pytest.fixture
def univariate_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate univariate time series data for testing.

    Returns
    -------
    X : np.ndarray
        Shape (50, 100) - 50 instances, 100 time points.
    y : np.ndarray
        Shape (50,) - binary labels.
    """
    n_instances = 50
    series_length = 100

    # Create two distinct patterns for binary classification
    X_class0 = rng.normal(0, 1, (n_instances // 2, series_length))
    X_class1 = rng.normal(2, 1, (n_instances // 2, series_length))

    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * (n_instances // 2) + [1] * (n_instances // 2))

    # Shuffle
    idx = rng.permutation(n_instances)
    return X[idx], y[idx]


@pytest.fixture
def multivariate_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate multivariate time series data for testing.

    Returns
    -------
    X : np.ndarray
        Shape (30, 3, 50) - 30 instances, 3 channels, 50 time points.
    y : np.ndarray
        Shape (30,) - binary labels.
    """
    n_instances = 30
    n_channels = 3
    series_length = 50

    X_class0 = rng.normal(0, 1, (n_instances // 2, n_channels, series_length))
    X_class1 = rng.normal(1.5, 1, (n_instances // 2, n_channels, series_length))

    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * (n_instances // 2) + [1] * (n_instances // 2))

    idx = rng.permutation(n_instances)
    return X[idx], y[idx]


@pytest.fixture
def simple_classifier(univariate_data: tuple[np.ndarray, np.ndarray]) -> KNeighborsClassifier:
    """Provide a simple fitted classifier for testing.

    Returns a KNN classifier trained on univariate data.
    """
    X, y = univariate_data
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X, y)
    return clf


@pytest.fixture
def tsc_data_univariate(univariate_data: tuple[np.ndarray, np.ndarray]) -> TSCData:
    """Create a TSCData instance from univariate test data."""
    X, y = univariate_data
    return TSCData.from_arrays("test_univariate", "train", X, y)


@pytest.fixture
def tsc_data_multivariate(multivariate_data: tuple[np.ndarray, np.ndarray]) -> TSCData:
    """Create a TSCData instance from multivariate test data."""
    X, y = multivariate_data
    return TSCData.from_arrays("test_multivariate", "train", X, y, squeeze_univariate=False)


@pytest.fixture
def counterfactual_pair(
    univariate_data: tuple[np.ndarray, np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate original instances and counterfactuals for testing.

    Returns
    -------
    X : np.ndarray
        Original instances (20, 100).
    X_cf : np.ndarray
        Counterfactual instances (20, 100).
    y : np.ndarray
        Original labels.
    y_cf : np.ndarray
        Counterfactual labels (flipped).
    """
    X, y = univariate_data
    X = X[:20]
    y = y[:20]

    # Create counterfactuals by adding noise and flipping labels
    X_cf = X + rng.normal(0, 0.5, X.shape)
    y_cf = 1 - y  # Flip labels

    return X, X_cf, y, y_cf
