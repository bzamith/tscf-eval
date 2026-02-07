"""Tests for the counterfactuals module."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from tscf_eval.counterfactuals import CELS, COMTE, SETS, Glacier, LatentCF, NativeGuide, TSEvo
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

    def test_cam_importance_uses_nun(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test that CAM importance is computed on the NUN, not the query."""
        received_series: list[np.ndarray] = []

        def mock_cam_fn(series: np.ndarray, y_pred: int) -> np.ndarray:
            received_series.append(series.copy())
            T = series.shape[-1] if series.ndim > 1 else series.shape[0]
            return np.ones(T)

        ng = NativeGuide(
            model=fitted_clf,
            data=train_data,
            method="cam",
            cam_importance_fn=mock_cam_fn,
        )

        x = test_instances[0]
        cf, _cf_label, _meta = ng.explain(x)

        assert cf.shape == x.shape
        # The series passed to cam_fn should be the NUN, not the query
        assert len(received_series) == 1
        assert not np.array_equal(received_series[0], x), (
            "CAM importance should be computed on the NUN, not the query"
        )


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

    def test_weight_type_local(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test Glacier with local (segment-based LIME) weight type."""
        glacier = Glacier(
            model=fitted_clf,
            data=train_data,
            max_iter=5,
            weight_type="local",
            n_segments=3,
            n_perturbations=20,
        )

        x = test_instances[0]
        cf, cf_label, meta = glacier.explain(x)

        assert isinstance(cf, np.ndarray)
        assert cf.shape == x.shape
        assert isinstance(cf_label, (int, np.integer))
        assert meta["weight_type"] == "local"

    def test_weight_type_unconstrained(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test Glacier with unconstrained weight type."""
        glacier = Glacier(
            model=fitted_clf,
            data=train_data,
            max_iter=5,
            weight_type="unconstrained",
        )

        x = test_instances[0]
        cf, _cf_label, meta = glacier.explain(x)
        assert cf.shape == x.shape
        assert meta["weight_type"] == "unconstrained"

    def test_compute_weights_unconstrained(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test that unconstrained weights are all zeros."""
        glacier = Glacier(model=fitted_clf, data=train_data, weight_type="unconstrained")
        x = test_instances[0]
        weights = glacier._compute_weights(x, base_label=0)
        assert np.all(weights == 0.0)

    def test_compute_weights_uniform(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test that uniform weights are all ones."""
        glacier = Glacier(model=fitted_clf, data=train_data, weight_type="uniform")
        x = test_instances[0]
        weights = glacier._compute_weights(x, base_label=0)
        assert np.all(weights == 1.0)

    def test_compute_local_importance_binary_weights(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test that local importance produces binary weights (0 or 1)."""
        glacier = Glacier(
            model=fitted_clf,
            data=train_data,
            weight_type="local",
            n_segments=3,
            n_perturbations=20,
        )
        x = test_instances[0]
        weights = glacier._compute_local_importance(x, base_label=0)
        assert weights.shape == x.shape
        assert np.all((weights == 0.0) | (weights == 1.0))

    def test_uniform_segments(self) -> None:
        """Test uniform segment boundary creation."""
        bounds = Glacier._uniform_segments(100, 4)
        assert bounds[0] == 0
        assert bounds[-1] == 100
        assert len(bounds) == 6  # 5 segments = 4 changepoints + 2 endpoints

    def test_uniform_segments_short_series(self) -> None:
        """Test uniform segments with very short series."""
        bounds = Glacier._uniform_segments(3, 10)
        assert bounds[0] == 0
        assert bounds[-1] == 3
        # Should not have more segments than timesteps
        assert len(bounds) <= 4

    def test_segment_time_series_fallback(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that segmentation falls back for very short series."""
        glacier = Glacier(model=fitted_clf, data=train_data, segment_window=10)
        # Series too short for window=10
        x_short = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bounds = glacier._segment_time_series(x_short)
        assert bounds[0] == 0
        assert bounds[-1] == 5

    def test_compute_background_with_scipy(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test STFT-based background computation."""
        glacier = Glacier(model=fitted_clf, data=train_data)
        x = test_instances[0]
        background = glacier._compute_background(x)
        assert background.shape == x.shape

    def test_generate_perturbation_samples(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test binary perturbation sample generation."""
        glacier = Glacier(model=fitted_clf, data=train_data, n_perturbations=15)
        x = test_instances[0]
        background = np.zeros_like(x)
        seg_bounds = [0, 10, 25, 50]
        n_segs = 3

        interp, raw = glacier._generate_perturbation_samples(
            x, x, background, seg_bounds, n_segs, is_multivariate=False
        )
        assert interp.shape == (15, 3)
        assert raw.shape == (15, *x.shape)
        assert np.all((interp == 0) | (interp == 1))

    def test_metadata_keys(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test that metadata contains expected keys."""
        glacier = Glacier(model=fitted_clf, data=train_data, max_iter=5)
        x = test_instances[0]
        _, _, meta = glacier.explain(x)

        assert meta["method"] == "glacier"
        assert "weight_type" in meta
        assert "class_of_interest" in meta
        assert "n_iterations" in meta
        assert "converged" in meta
        assert "final_target_prob" in meta
        assert "final_loss" in meta
        assert "validity" in meta

    def test_explain_with_class_of_interest(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test explain with explicit class_of_interest."""
        glacier = Glacier(model=fitted_clf, data=train_data, max_iter=5)
        x = test_instances[0]
        cf, _cf_label, meta = glacier.explain(x, class_of_interest=1)
        assert cf.shape == x.shape
        assert meta["class_of_interest"] == 1

    def test_explain_with_y_pred(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test explain with precomputed y_pred."""
        glacier = Glacier(model=fitted_clf, data=train_data, max_iter=5)
        x = test_instances[0]
        y_pred = int(fitted_clf.predict(x.reshape(1, -1))[0])
        cf, _cf_label, _meta = glacier.explain(x, y_pred=y_pred)
        assert cf.shape == x.shape

    def test_invalid_pred_margin_weight(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that invalid pred_margin_weight raises error."""
        with pytest.raises(ValueError, match="pred_margin_weight"):
            Glacier(model=fitted_clf, data=train_data, pred_margin_weight=1.5)

    def test_invalid_learning_rate(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that invalid learning_rate raises error."""
        with pytest.raises(ValueError, match="learning_rate"):
            Glacier(model=fitted_clf, data=train_data, learning_rate=0.0)

    def test_invalid_max_iter(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that invalid max_iter raises error."""
        with pytest.raises(ValueError, match="max_iter"):
            Glacier(model=fitted_clf, data=train_data, max_iter=0)

    def test_invalid_tau(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that invalid tau raises error."""
        with pytest.raises(ValueError, match="tau"):
            Glacier(model=fitted_clf, data=train_data, tau=0.0)

    def test_invalid_weight_type(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that invalid weight_type raises error."""
        with pytest.raises(ValueError, match="weight_type"):
            Glacier(model=fitted_clf, data=train_data, weight_type="invalid")

    def test_invalid_gradient_subsample(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that invalid gradient_subsample raises error."""
        with pytest.raises(ValueError, match="gradient_subsample"):
            Glacier(model=fitted_clf, data=train_data, gradient_subsample=0)

    def test_invalid_n_segments(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that invalid n_segments raises error."""
        with pytest.raises(ValueError, match="n_segments"):
            Glacier(model=fitted_clf, data=train_data, n_segments=0)

    def test_invalid_segment_window(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that invalid segment_window raises error."""
        with pytest.raises(ValueError, match="segment_window"):
            Glacier(model=fitted_clf, data=train_data, segment_window=1)

    def test_invalid_n_perturbations(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that invalid n_perturbations raises error."""
        with pytest.raises(ValueError, match="n_perturbations"):
            Glacier(model=fitted_clf, data=train_data, n_perturbations=5)


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


# =============================================================================
# CELS Tests
# =============================================================================


class TestCELS:
    """Tests for CELS counterfactual generator."""

    def test_creation(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test CELS instantiation."""
        cels = CELS(model=fitted_clf, data=train_data)
        assert cels.model is fitted_clf
        assert cels.budget_coeff == 0.6
        assert cels.tv_coeff == 0.5
        assert cels.max_coeff == 0.7

    def test_explain_returns_valid_tuple(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test that explain returns the expected tuple format."""
        cels = CELS(model=fitted_clf, data=train_data, max_iter=20)

        x = test_instances[0]
        cf, cf_label, meta = cels.explain(x)

        assert isinstance(cf, np.ndarray)
        assert isinstance(cf_label, (int, np.integer))
        assert isinstance(meta, dict)
        assert cf.shape == x.shape

    def test_metadata_keys(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test that metadata contains expected keys."""
        cels = CELS(model=fitted_clf, data=train_data, max_iter=20)

        x = test_instances[0]
        _, _, meta = cels.explain(x)

        assert meta["method"] == "cels"
        assert "class_of_interest" in meta
        assert "nun_index_in_ref" in meta
        assert "n_iterations" in meta
        assert "converged" in meta
        assert "final_target_prob" in meta
        assert "final_loss" in meta
        assert "mask_density" in meta
        assert "validity" in meta

    def test_explain_with_y_pred(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test explain with precomputed y_pred."""
        cels = CELS(model=fitted_clf, data=train_data, max_iter=20)

        x = test_instances[0]
        y_pred = int(fitted_clf.predict(x.reshape(1, -1))[0])

        cf, _cf_label, _meta = cels.explain(x, y_pred=y_pred)
        assert cf.shape == x.shape

    def test_explain_with_class_of_interest(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test explain with explicit class_of_interest."""
        cels = CELS(model=fitted_clf, data=train_data, max_iter=20)

        x = test_instances[0]
        cf, _cf_label, meta = cels.explain(x, class_of_interest=1)
        assert cf.shape == x.shape
        assert meta["class_of_interest"] == 1

    def test_explain_k(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test generating k counterfactuals."""
        cels = CELS(model=fitted_clf, data=train_data, max_iter=20)

        x = test_instances[0]
        k = 3
        cfs, cf_labels, metas = cels.explain_k(x, k=k)

        assert cfs.shape[0] == k
        assert cf_labels.shape == (k,)
        assert len(metas) == k

        # Check k_index is set
        for i, meta in enumerate(metas):
            assert meta.get("k_index") == i

    def test_binarize_theta(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that binarize_theta produces a proper binary mask."""
        cels = CELS(model=fitted_clf, data=train_data)
        theta = np.array([0.1, 0.3, 0.7, 0.9, 0.2])
        mask = cels._binarize_theta(theta)
        assert np.all((mask == 0.0) | (mask == 1.0))

    def test_invalid_budget_coeff_raises(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that negative budget_coeff raises error."""
        with pytest.raises(ValueError, match="budget_coeff"):
            CELS(model=fitted_clf, data=train_data, budget_coeff=-1.0)

    def test_invalid_tau_raises(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that invalid tau raises error."""
        with pytest.raises(ValueError, match="tau"):
            CELS(model=fitted_clf, data=train_data, tau=0.0)

    def test_invalid_learning_rate_raises(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that invalid learning_rate raises error."""
        with pytest.raises(ValueError, match="learning_rate"):
            CELS(model=fitted_clf, data=train_data, learning_rate=0.0)

    def test_invalid_threshold_raises(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that invalid threshold raises error."""
        with pytest.raises(ValueError, match="threshold"):
            CELS(model=fitted_clf, data=train_data, threshold=1.0)

    def test_invalid_gradient_subsample_raises(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that invalid gradient_subsample raises error."""
        with pytest.raises(ValueError, match="gradient_subsample"):
            CELS(model=fitted_clf, data=train_data, gradient_subsample=0)

    def test_gradient_subsample_none(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test that gradient_subsample=None uses full gradient."""
        cels = CELS(model=fitted_clf, data=train_data, max_iter=20, gradient_subsample=None)
        x = test_instances[0]
        cf, _cf_label, _meta = cels.explain(x)
        assert cf.shape == x.shape

    def test_gradient_subsample_value(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test that gradient_subsample with explicit value works."""
        cels = CELS(model=fitted_clf, data=train_data, max_iter=20, gradient_subsample=10)
        x = test_instances[0]
        cf, _cf_label, _meta = cels.explain(x)
        assert cf.shape == x.shape


# =============================================================================
# SETS Tests
# =============================================================================


class TestSETS:
    """Tests for SETS counterfactual generator."""

    def test_creation(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test SETS instantiation."""
        sets = SETS(
            model=fitted_clf,
            data=train_data,
            n_shapelet_samples=200,
            max_shapelets=10,
        )
        assert sets.model is fitted_clf
        assert sets.n_shapelet_samples == 200

    def test_explain_returns_valid_tuple(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test that explain returns the expected tuple format."""
        sets = SETS(
            model=fitted_clf,
            data=train_data,
            n_shapelet_samples=200,
            max_shapelets=10,
        )

        x = test_instances[0]
        cf, cf_label, meta = sets.explain(x)

        assert isinstance(cf, np.ndarray)
        assert isinstance(cf_label, (int, np.integer))
        assert isinstance(meta, dict)
        assert cf.shape == x.shape

    def test_metadata_keys(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test that metadata contains expected keys."""
        sets = SETS(
            model=fitted_clf,
            data=train_data,
            n_shapelet_samples=200,
            max_shapelets=10,
        )

        x = test_instances[0]
        _, _, meta = sets.explain(x)

        assert meta["method"] == "sets"
        assert "class_of_interest" in meta
        assert "nun_index_in_ref" in meta
        assert "dimensions_modified" in meta
        assert "phase_a_edits" in meta
        assert "phase_b_edits" in meta
        assert "validity" in meta

    def test_explain_k(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
        test_instances: np.ndarray,
    ) -> None:
        """Test generating k counterfactuals with SETS."""
        sets = SETS(
            model=fitted_clf,
            data=train_data,
            n_shapelet_samples=200,
            max_shapelets=10,
        )

        x = test_instances[0]
        k = 3
        cfs, cf_labels, metas = sets.explain_k(x, k=k)

        assert cfs.shape[0] == k
        assert cf_labels.shape == (k,)
        assert len(metas) == k

    def test_invalid_threshold_percentile_raises(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that invalid threshold_percentile raises error."""
        with pytest.raises(ValueError, match="threshold_percentile"):
            SETS(
                model=fitted_clf,
                data=train_data,
                threshold_percentile=0.0,
            )

    def test_invalid_max_combination_dims_raises(
        self,
        fitted_clf: KNeighborsClassifier,
        train_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test that invalid max_combination_dims raises error."""
        with pytest.raises(ValueError, match="max_combination_dims"):
            SETS(
                model=fitted_clf,
                data=train_data,
                max_combination_dims=0,
            )
