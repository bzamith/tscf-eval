"""Tests for the counterfactuals module."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from tscf_eval.counterfactuals import COMTE, Glacier, LatentCF, NativeGuide, TSEvo
from tscf_eval.counterfactuals.base import Counterfactual

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def train_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Create training data for counterfactual testing."""
    n_train = 40
    series_length = 50

    # Create two distinct patterns
    X_class0 = rng.normal(0, 1, (n_train // 2, series_length))
    X_class1 = rng.normal(2, 1, (n_train // 2, series_length))

    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * (n_train // 2) + [1] * (n_train // 2))

    idx = rng.permutation(n_train)
    return X[idx], y[idx]


@pytest.fixture
def test_instances(rng: np.random.Generator) -> np.ndarray:
    """Create test instances for counterfactual generation."""
    return rng.normal(0, 1, (5, 50))


@pytest.fixture
def fitted_clf(train_data: tuple[np.ndarray, np.ndarray]) -> KNeighborsClassifier:
    """Create a fitted classifier."""
    X, y = train_data
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X, y)
    return clf


# =============================================================================
# Base Counterfactual Tests
# =============================================================================


class TestCounterfactualBase:
    """Tests for Counterfactual base class."""

    def test_is_abstract(self) -> None:
        """Test that Counterfactual is abstract."""
        with pytest.raises(TypeError):
            Counterfactual()  # type: ignore[abstract]

    def test_subclass_must_implement_explain(self) -> None:
        """Test that subclasses must implement explain."""

        class IncompleteExplainer(Counterfactual):
            pass

        with pytest.raises(TypeError):
            IncompleteExplainer()  # type: ignore[abstract]


# =============================================================================
# COMTE Tests
# =============================================================================


class TestCOMTE:
    """Tests for COMTE counterfactual generator."""

    def test_creation(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test COMTE instantiation."""
        comte = COMTE(model=fitted_clf, data=train_data, distance="euclidean")
        assert comte.model is fitted_clf
        assert comte.distance == "euclidean"
        assert comte.n_distractors == 10
        assert comte.tau == 0.95

    def test_explain_returns_valid_tuple(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test that explain returns the expected tuple format."""
        comte = COMTE(model=fitted_clf, data=train_data, distance="euclidean")

        x = test_instances[0]
        cf, cf_label, meta = comte.explain(x)

        # Check return types
        assert isinstance(cf, np.ndarray)
        assert isinstance(cf_label, (int, np.integer))
        assert isinstance(meta, dict)

        # Check shapes
        assert cf.shape == x.shape

    def test_explain_with_y_pred(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test explain with precomputed y_pred."""
        comte = COMTE(model=fitted_clf, data=train_data, distance="euclidean")

        x = test_instances[0]
        y_pred = int(fitted_clf.predict(x.reshape(1, -1))[0])

        cf, _cf_label, _meta = comte.explain(x, y_pred=y_pred)
        assert cf.shape == x.shape

    def test_explain_k(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test generating k counterfactuals."""
        comte = COMTE(model=fitted_clf, data=train_data, distance="euclidean")

        x = test_instances[0]
        k = 3
        cfs, cf_labels, metas = comte.explain_k(x, k=k)

        assert cfs.shape == (k, x.shape[0])
        assert cf_labels.shape == (k,)
        assert len(metas) == k

    def test_dtw_distance(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test COMTE with DTW distance."""
        comte = COMTE(model=fitted_clf, data=train_data, distance="dtw")

        x = test_instances[0]
        cf, _cf_label, _meta = comte.explain(x)
        assert cf.shape == x.shape

    def test_metadata_contains_expected_keys(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test that metadata contains expected information."""
        comte = COMTE(model=fitted_clf, data=train_data, distance="euclidean")

        x = test_instances[0]
        _, _, meta = comte.explain(x)

        # COMTE should include edit information
        assert "edits_variables" in meta or "n_edits" in meta or "target_prob" in meta


# =============================================================================
# NativeGuide Tests
# =============================================================================


class TestNativeGuide:
    """Tests for NativeGuide counterfactual generator."""

    def test_creation(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test NativeGuide instantiation."""
        ng = NativeGuide(model=fitted_clf, data=train_data, method="blend")
        assert ng.model is fitted_clf
        assert ng.method == "blend"

    def test_blend_method(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test NativeGuide with blend method."""
        ng = NativeGuide(model=fitted_clf, data=train_data, method="blend")

        x = test_instances[0]
        cf, cf_label, _meta = ng.explain(x)

        assert isinstance(cf, np.ndarray)
        assert cf.shape == x.shape
        assert isinstance(cf_label, (int, np.integer))

    def test_ng_method(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test NativeGuide with ng method."""
        ng = NativeGuide(model=fitted_clf, data=train_data, method="ng")

        x = test_instances[0]
        cf, _cf_label, _meta = ng.explain(x)

        assert cf.shape == x.shape

    def test_dtw_dba_method(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test NativeGuide with dtw_dba method."""
        ng = NativeGuide(model=fitted_clf, data=train_data, method="dtw_dba", k_unlike=3)

        x = test_instances[0]
        cf, _cf_label, _meta = ng.explain(x)

        assert cf.shape == x.shape

    def test_explain_k(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test generating k counterfactuals with NativeGuide."""
        ng = NativeGuide(model=fitted_clf, data=train_data, method="blend")

        x = test_instances[0]
        k = 3
        cfs, cf_labels, metas = ng.explain_k(x, k=k)

        assert cfs.shape[0] == k
        assert cf_labels.shape == (k,)
        assert len(metas) == k

    def test_explain_k_generates_diverse_cfs(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test that explain_k generates diverse counterfactuals using different neighbors."""
        ng = NativeGuide(model=fitted_clf, data=train_data, method="blend")

        x = test_instances[0]
        k = 3
        _cfs, _cf_labels, metas = ng.explain_k(x, k=k)

        # Check that different neighbors are used (if available)
        nun_indices = [m.get("nun_index_in_X") for m in metas]
        # Filter out None values (in case there aren't enough unlike neighbors)
        valid_indices = [i for i in nun_indices if i is not None]
        if len(valid_indices) > 1:
            # Should use different neighbors for diversity
            assert len(set(valid_indices)) > 1, "Should use different unlike neighbors"

        # Check k_index is set correctly
        for i, meta in enumerate(metas):
            assert meta.get("k_index") == i

    def test_invalid_method_raises(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="method must be"):
            NativeGuide(model=fitted_clf, data=train_data, method="invalid")


# =============================================================================
# TSEvo Tests
# =============================================================================


class TestTSEvo:
    """Tests for TSEvo counterfactual generator."""

    def test_creation(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test TSEvo instantiation."""
        tsevo = TSEvo(model=fitted_clf, data=train_data)
        assert tsevo.model is fitted_clf

    def test_explain(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test TSEvo explain method."""
        # Use small parameters for fast testing
        tsevo = TSEvo(
            model=fitted_clf,
            data=train_data,
            n_generations=5,
            population_size=10,
        )

        x = test_instances[0]
        cf, cf_label, _meta = tsevo.explain(x)

        assert isinstance(cf, np.ndarray)
        assert cf.shape == x.shape
        assert isinstance(cf_label, (int, np.integer))


# =============================================================================
# Glacier Tests
# =============================================================================


class TestGlacier:
    """Tests for Glacier counterfactual generator."""

    def test_creation(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test Glacier instantiation."""
        glacier = Glacier(model=fitted_clf, data=train_data)
        assert glacier.model is fitted_clf

    def test_explain(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test Glacier explain method."""
        glacier = Glacier(
            model=fitted_clf,
            data=train_data,
            max_iter=10,
            learning_rate=0.1,
        )

        x = test_instances[0]
        cf, _cf_label, _meta = glacier.explain(x)

        assert isinstance(cf, np.ndarray)
        assert cf.shape == x.shape

    def test_weight_type_uniform(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test Glacier with uniform weight type."""
        glacier = Glacier(
            model=fitted_clf,
            data=train_data,
            max_iter=5,
            weight_type="uniform",
        )

        x = test_instances[0]
        cf, _cf_label, _meta = glacier.explain(x)
        assert cf.shape == x.shape


# =============================================================================
# LatentCF Tests
# =============================================================================


class TestLatentCF:
    """Tests for LatentCF counterfactual generator."""

    def test_creation(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test LatentCF instantiation."""
        latent = LatentCF(model=fitted_clf, data=train_data)
        assert latent.model is fitted_clf

    def test_explain(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test LatentCF explain method."""
        latent = LatentCF(
            model=fitted_clf,
            data=train_data,
            max_iter=10,
            learning_rate=0.01,
        )

        x = test_instances[0]
        cf, _cf_label, _meta = latent.explain(x)

        assert isinstance(cf, np.ndarray)
        assert cf.shape == x.shape

    def test_step_weights_uniform(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test LatentCF with uniform step weights."""
        latent = LatentCF(
            model=fitted_clf,
            data=train_data,
            max_iter=5,
            step_weights="uniform",
        )

        x = test_instances[0]
        cf, _cf_label, _meta = latent.explain(x)
        assert cf.shape == x.shape
