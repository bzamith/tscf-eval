"""Time-series classification data container.

This module provides the :class:`TSCData` dataclass, an immutable container
for time-series classification datasets. It supports both univariate and
multivariate time series with utilities for conversion, persistence, and
basic dataset operations.

Classes
-------
TSCData
    Immutable container holding feature arrays ``X``, labels ``y``, and
    metadata (name, split). Provides factory methods for construction from
    arrays or DataFrames, properties for dataset introspection, and
    serialization via NumPy's ``.npz`` format.

Type Aliases
------------
Split
    Literal type for dataset splits: ``'train'`` or ``'test'``.

Examples
--------
>>> import numpy as np
>>> from tscf_eval.data_loader import TSCData
>>>
>>> # Create from arrays
>>> X = np.random.randn(100, 50)  # 100 instances, 50 time points
>>> y = np.random.randint(0, 2, 100)
>>> data = TSCData.from_arrays("my_dataset", "train", X, y)
>>>
>>> # Inspect properties
>>> print(data.n_instances, data.series_length, data.n_classes)
100 50 2
>>>
>>> # Save and load
>>> data.save("my_dataset.npz")
>>> loaded = TSCData.load("my_dataset.npz")

See Also
--------
tscf_eval.data_loader.DataLoader : Abstract loader interface.
tscf_eval.data_loader.UCRLoader : UCR archive dataset loader.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd

Split = Literal["train", "test"]


@dataclass(frozen=True)
class TSCData:
    """Immutable container for time-series classification data.

    A small, well-typed container for time-series classification datasets.
    """

    name: str
    split: Split
    X: np.ndarray
    y: np.ndarray

    @staticmethod
    def from_arrays(
        name: str,
        split: Split,
        X: np.ndarray,
        y: Iterable,
        *,
        squeeze_univariate: bool = True,
    ) -> "TSCData":
        """Create a ``TSCData`` instance from numpy arrays / array-likes.

        Parameters
        ----------
        name : str
            Dataset name.
        split : {'train', 'test'}
            Which split this instance belongs to.
        X : array-like
            Time-series data. Accepts either 2D ``(n, L)`` for univariate
            data or 3D ``(n, D, L)`` for multivariate data.
        y : array-like
            1D labels of length ``n``.
        squeeze_univariate : bool, optional
            If ``True`` and ``X`` is shape ``(n,1,L)``, squeeze the
            channel dimension to produce shape ``(n,L)``.

        Returns
        -------
        TSCData
            Constructed immutable container.

        Raises
        ------
        ValueError
            If ``X`` or ``y`` do not have expected dimensions or if the
            number of instances disagree.
        """
        X = np.asarray(X)
        y = np.asarray(list(y))

        if squeeze_univariate and X.ndim == 3 and X.shape[1] == 1:
            X = X[:, 0, :]

        if X.ndim not in (2, 3):
            raise ValueError("X must be 2D (n,L) or 3D (n,D,L).")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"n mismatch: X has {X.shape[0]} rows but y has {y.shape[0]}.")

        return TSCData(name=name, split=split, X=X, y=y)

    @staticmethod
    def from_dataframe(
        name: str,
        split: Split,
        df: pd.DataFrame,
        *,
        label_col: str,
        feature_cols: Sequence[str] | None = None,
    ) -> "TSCData":
        """Create a ``TSCData`` instance from a wide-format ``DataFrame``.

        The dataframe format expected is one row per instance, numeric
        columns representing time points (or channels flattened), and
        a column containing the label.

        Parameters
        ----------
        name : str
            Dataset name used in the resulting ``TSCData``.
        split : {'train', 'test'}
            Split label to set on the resulting object.
        df : pandas.DataFrame
            Source table.
        label_col : str
            Column name in ``df`` containing labels.
        feature_cols : sequence of str, optional
            Columns to use as features in the desired order. If ``None``,
            numeric columns except label/split columns are used.
        (label maps are not used; labels are returned in original form)

        Returns
        -------
        TSCData
            Constructed dataset object.

        Raises
        ------
        ValueError
            If ``label_col`` is missing or no numeric feature columns are
            found when ``feature_cols`` is not provided.
        """
        if label_col not in df.columns:
            raise ValueError(f"label_col='{label_col}' not in dataframe.")

        if feature_cols is None:
            drop_cols = {label_col}
            drop_cols |= {c for c in ["split", "Split", "SPLIT"] if c in df.columns}
            feature_cols = (
                df.drop(columns=list(drop_cols)).select_dtypes(include=[np.number]).columns.tolist()
            )
            if not feature_cols:
                raise ValueError("No numeric feature columns found; pass feature_cols=[...]")
        else:
            for c in feature_cols:
                if c not in df.columns:
                    raise ValueError(f"feature column '{c}' missing.")

        X = df[feature_cols].to_numpy(dtype=float)  # (n, L)
        y = df[label_col].to_numpy()
        return TSCData.from_arrays(name, split, X, y, squeeze_univariate=True)

    @property
    def n_instances(self) -> int:
        """Number of instances in the dataset.

        Returns
        -------
        int
            Number of rows / time-series instances (n).
        """
        return int(self.X.shape[0])

    @property
    def series_length(self) -> int:
        """Length of each time series in time points.

        For univariate data this is the second axis length of ``X`` when
        ``X`` has shape ``(n, L)``. For multivariate data (``X`` shape
        ``(n, D, L)``) this returns ``L``.

        Returns
        -------
        int
            Series length (L).
        """
        return int(self.X.shape[-1])

    @property
    def n_dims(self) -> int:
        """Number of dimensions (channels) per time series.

        Returns
        -------
        int
            ``1`` for univariate series (``X`` is 2D) or ``D`` for
            multivariate series (``X`` is 3D with shape ``(n, D, L)``).
        """
        return 1 if self.X.ndim == 2 else int(self.X.shape[1])

    @property
    def n_classes(self) -> int:
        """Number of unique class labels present in ``y``.

        Returns
        -------
        int
            The number of distinct labels (classes) in the label array.
        """
        return int(np.unique(self.y).size)

    @property
    def is_univariate(self) -> bool:
        """Whether the dataset is univariate.

        Returns
        -------
        bool
            True if each instance has a single channel (``D == 1``),
            False otherwise.
        """
        return self.n_dims == 1

    def describe(self) -> dict:
        """Return a small dictionary summarizing dataset properties.

        The dictionary contains basic metadata useful for logging or
        quick inspection: dataset name and split, shapes (instances,
        series length, dimensions), number of classes, class counts
        and the optional label mapping if present.

        Returns
        -------
        dict
            Summary dictionary with keys: 'name', 'split', 'n_instances',
            'series_length', 'n_dims', 'n_classes', 'class_counts'.
        """
        classes, counts = np.unique(self.y, return_counts=True)
        return {
            "name": self.name,
            "split": self.split,
            "n_instances": self.n_instances,
            "series_length": self.series_length,
            "n_dims": self.n_dims,
            "n_classes": self.n_classes,
            "class_counts": {str(c): int(n) for c, n in zip(classes, counts, strict=True)},
        }

    def to_dataframe(self, *, label_name: str = "label", prefix: str = "t_") -> pd.DataFrame:
        """Return a wide-format ``DataFrame`` representing the dataset.

        Parameters
        ----------
        label_name : str, optional
            Column name to use for labels in the returned dataframe.
        prefix : str, optional
            Prefix for generated numeric/time columns.

        Returns
        -------
        pandas.DataFrame
            Wide-format dataframe with numeric columns for each time
            point (and channel) and a final column with labels.
        """
        if self.is_univariate:
            cols = [f"{prefix}{i}" for i in range(self.series_length)]
            dfX = pd.DataFrame(self.X, columns=cols)
        else:
            n, d, L = self.X.shape
            cols = [f"{prefix}{t}_d{dim}" for dim in range(d) for t in range(L)]
            dfX = pd.DataFrame(self.X.reshape(n, d * L), columns=cols)
        df = dfX.copy()
        df[label_name] = self.y
        return df

    def map_labels(self, mapping: dict[int | str, int | str]) -> "TSCData":
        """Return a copy of this dataset with labels remapped.

        Parameters
        ----------
        mapping : dict
            Mapping from original labels to new labels. If a label is not
            present in ``mapping``, it is left unchanged.

        Returns
        -------
        TSCData
            New instance with remapped ``y``.
        """
        new_y = np.array([mapping.get(item, item) for item in self.y])
        return TSCData(self.name, self.split, self.X, new_y)

    def select_classes(self, keep: Iterable[int | str]) -> "TSCData":
        """Return a view of the dataset keeping only specified classes.

        Parameters
        ----------
        keep : iterable
            Labels to keep. Items not present in the dataset are ignored.

        Returns
        -------
        TSCData
            New instance containing only instances whose label is in
            ``keep``.
        """
        keep_set = set(keep)
        mask = np.array([item in keep_set for item in self.y], dtype=bool)
        return TSCData(self.name, self.split, self.X[mask], self.y[mask])

    def save(self, path: str | Path) -> None:
        """Save the dataset to a compressed NumPy ``.npz`` file.

        The file contains arrays for ``X``, ``y``, ``name`` and ``split``. Use
        :meth:`TSCData.load` to restore.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file path. The function will use ``numpy.savez_compressed``.
        """
        path = Path(path)
        np.savez_compressed(
            path,
            X=self.X,
            y=self.y,
            name=self.name,
            split=self.split,
        )

    @staticmethod
    def load(path: str | Path) -> "TSCData":
        """Load a ``TSCData`` instance previously written with :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to ``.npz`` file produced by :meth:`save`.

        Returns
        -------
        TSCData
            Restored dataset.
        """
        z = np.load(Path(path), allow_pickle=True)
        name = str(z["name"])
        split = str(z["split"])
        split = cast("Split", split)

        X = np.asarray(z["X"])
        y = np.asarray(z["y"])

        return TSCData(name=name, split=split, X=X, y=y)
