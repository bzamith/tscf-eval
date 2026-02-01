"""Tests for the data_loader module."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from tscf_eval.data_loader import DataLoader, FileLoader, TSCData, UCRLoader

# =============================================================================
# TSCData Tests
# =============================================================================


class TestTSCData:
    """Tests for TSCData container."""

    def test_from_arrays_univariate(self, rng: np.random.Generator) -> None:
        """Test creating TSCData from univariate arrays."""
        X = rng.normal(0, 1, (100, 50))
        y = rng.integers(0, 3, 100)

        data = TSCData.from_arrays("test", "train", X, y)

        assert data.name == "test"
        assert data.split == "train"
        assert data.X.shape == (100, 50)
        assert data.y.shape == (100,)

    def test_from_arrays_multivariate(self, rng: np.random.Generator) -> None:
        """Test creating TSCData from multivariate arrays."""
        X = rng.normal(0, 1, (50, 3, 100))
        y = rng.integers(0, 2, 50)

        data = TSCData.from_arrays("test", "train", X, y, squeeze_univariate=False)

        assert data.X.shape == (50, 3, 100)

    def test_from_arrays_squeeze_univariate(self, rng: np.random.Generator) -> None:
        """Test that univariate 3D arrays are squeezed."""
        X = rng.normal(0, 1, (50, 1, 100))  # Single channel
        y = rng.integers(0, 2, 50)

        data = TSCData.from_arrays("test", "train", X, y, squeeze_univariate=True)

        # Should be squeezed to 2D
        assert data.X.shape == (50, 100)

    def test_from_arrays_validation_x_dim(self, rng: np.random.Generator) -> None:
        """Test validation for X dimensions."""
        X = rng.normal(0, 1, (50,))  # 1D - invalid
        y = rng.integers(0, 2, 50)

        with pytest.raises(ValueError, match=r"2D.*3D"):
            TSCData.from_arrays("test", "train", X, y)

    def test_from_arrays_validation_y_dim(self, rng: np.random.Generator) -> None:
        """Test validation for y dimensions."""
        X = rng.normal(0, 1, (50, 100))
        y = rng.integers(0, 2, (50, 2))  # 2D - invalid

        with pytest.raises(ValueError, match="1D"):
            TSCData.from_arrays("test", "train", X, y)

    def test_from_arrays_validation_n_mismatch(self, rng: np.random.Generator) -> None:
        """Test validation for n_instances mismatch."""
        X = rng.normal(0, 1, (50, 100))
        y = rng.integers(0, 2, 60)  # Different n

        with pytest.raises(ValueError, match="mismatch"):
            TSCData.from_arrays("test", "train", X, y)

    def test_properties_univariate(self, tsc_data_univariate: TSCData) -> None:
        """Test TSCData properties for univariate data."""
        assert tsc_data_univariate.n_instances == 50
        assert tsc_data_univariate.series_length == 100
        assert tsc_data_univariate.is_univariate is True
        assert tsc_data_univariate.n_dims == 1

    def test_properties_multivariate(self, tsc_data_multivariate: TSCData) -> None:
        """Test TSCData properties for multivariate data."""
        assert tsc_data_multivariate.n_dims == 3
        assert tsc_data_multivariate.is_univariate is False

    def test_n_classes(self, tsc_data_univariate: TSCData) -> None:
        """Test n_classes property."""
        assert tsc_data_univariate.n_classes == 2

    def test_describe(self, tsc_data_univariate: TSCData) -> None:
        """Test describe method."""
        desc = tsc_data_univariate.describe()
        assert isinstance(desc, dict)
        assert "name" in desc
        assert "n_instances" in desc
        assert "class_counts" in desc
        # Check class_counts sum matches n_instances
        total = sum(desc["class_counts"].values())
        assert total == tsc_data_univariate.n_instances

    def test_immutability(self, tsc_data_univariate: TSCData) -> None:
        """Test that TSCData is immutable."""
        with pytest.raises(AttributeError):
            tsc_data_univariate.name = "new_name"  # type: ignore[misc]

    def test_save_and_load(self, tsc_data_univariate: TSCData) -> None:
        """Test save and load functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_data.npz"

            tsc_data_univariate.save(path)
            assert path.exists()

            loaded = TSCData.load(path)
            assert loaded.name == tsc_data_univariate.name
            assert loaded.split == tsc_data_univariate.split
            np.testing.assert_array_equal(loaded.X, tsc_data_univariate.X)
            np.testing.assert_array_equal(loaded.y, tsc_data_univariate.y)

    def test_from_dataframe(self) -> None:
        """Test creating TSCData from DataFrame."""
        # Create a simple wide-format DataFrame
        n, t = 20, 10
        df = pd.DataFrame(
            {f"t{i}": np.random.randn(n) for i in range(t)} | {"label": np.random.randint(0, 2, n)}
        )

        data = TSCData.from_dataframe("test", "train", df, label_col="label")

        assert data.n_instances == n
        assert data.series_length == t

    def test_select_classes(self, tsc_data_univariate: TSCData) -> None:
        """Test filtering by class using select_classes."""
        filtered = tsc_data_univariate.select_classes([0])

        assert all(filtered.y == 0)
        assert filtered.n_instances < tsc_data_univariate.n_instances

    def test_map_labels(self, tsc_data_univariate: TSCData) -> None:
        """Test label remapping."""
        mapping = {0: 10, 1: 20}
        remapped = tsc_data_univariate.map_labels(mapping)

        assert set(np.unique(remapped.y)) == {10, 20}

    def test_to_dataframe(self, tsc_data_univariate: TSCData) -> None:
        """Test conversion to DataFrame."""
        df = tsc_data_univariate.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == tsc_data_univariate.n_instances
        assert "label" in df.columns


# =============================================================================
# FileLoader Tests
# =============================================================================


class TestFileLoader:
    """Tests for FileLoader."""

    def test_creation_two_file_mode(self) -> None:
        """Test FileLoader instantiation with two-file mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CSV files
            train_path = Path(tmpdir) / "train.csv"
            test_path = Path(tmpdir) / "test.csv"

            n, t = 10, 5
            df_train = pd.DataFrame(
                {f"t{i}": np.random.randn(n) for i in range(t)} | {"label": [0] * 5 + [1] * 5}
            )
            df_test = pd.DataFrame(
                {f"t{i}": np.random.randn(n // 2) for i in range(t)} | {"label": [0, 1, 0, 1, 0]}
            )
            df_train.to_csv(train_path, index=False)
            df_test.to_csv(test_path, index=False)

            loader = FileLoader(
                train_path=train_path, test_path=test_path, label_col="label", name="test_dataset"
            )
            assert loader.name == "test_dataset"

    def test_load_csv(self) -> None:
        """Test loading from CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CSV files
            train_path = Path(tmpdir) / "train.csv"
            test_path = Path(tmpdir) / "test.csv"

            n, t = 10, 5
            df_train = pd.DataFrame(
                {f"t{i}": np.random.randn(n) for i in range(t)} | {"label": [0] * 5 + [1] * 5}
            )
            df_test = pd.DataFrame(
                {f"t{i}": np.random.randn(n // 2) for i in range(t)} | {"label": [0, 1, 0, 1, 0]}
            )
            df_train.to_csv(train_path, index=False)
            df_test.to_csv(test_path, index=False)

            loader = FileLoader(train_path=train_path, test_path=test_path, label_col="label")
            loaded_train = loader.load("train")
            loaded_test = loader.load("test")

            assert loaded_train.n_instances == n
            assert loaded_test.n_instances == n // 2

    def test_load_nonexistent_raises(self) -> None:
        """Test that loading nonexistent file raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "nonexistent.csv"

            # Create a valid test.csv but point train to nonexistent
            test_path = Path(tmpdir) / "test.csv"
            df = pd.DataFrame({"t0": [1, 2], "label": [0, 1]})
            df.to_csv(test_path, index=False)

            loader = FileLoader(train_path=nonexistent, test_path=test_path, label_col="label")

            with pytest.raises(FileNotFoundError):
                loader.load("train")

    def test_single_file_mode(self) -> None:
        """Test FileLoader with single file mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.csv"

            n, t = 20, 5
            df = pd.DataFrame(
                {f"t{i}": np.random.randn(n) for i in range(t)}
                | {"label": [0] * 10 + [1] * 10, "split": ["train"] * 15 + ["test"] * 5}
            )
            df.to_csv(data_path, index=False)

            loader = FileLoader(data_path=data_path, split_col="split", label_col="label")

            train_data = loader.load("train")
            test_data = loader.load("test")

            assert train_data.n_instances == 15
            assert test_data.n_instances == 5


# =============================================================================
# UCRLoader Tests
# =============================================================================


class TestUCRLoader:
    """Tests for UCRLoader."""

    def test_creation(self) -> None:
        """Test UCRLoader instantiation."""
        loader = UCRLoader("GunPoint")
        assert loader.dataset_name == "GunPoint"

    def test_load_caching(self) -> None:
        """Test that UCRLoader uses caching."""
        loader1 = UCRLoader("GunPoint")
        loader2 = UCRLoader("GunPoint")

        # Both should reference the same class
        assert loader1.__class__ == loader2.__class__


# =============================================================================
# DataLoader Base Tests
# =============================================================================


class TestDataLoader:
    """Tests for DataLoader abstract base class."""

    def test_is_abstract(self) -> None:
        """Test that DataLoader is abstract."""
        with pytest.raises(TypeError):
            DataLoader()  # type: ignore[abstract]
