"""File-based dataset loader for CSV, TXT, and Excel files.

This module provides the :class:`FileLoader` class for loading time-series
classification datasets from local files. It supports wide-format tabular
data where each row represents a time-series instance.

Supported file formats:
- CSV (``.csv``, ``.txt``)
- Excel (``.xlsx``, ``.xls``)

The loader can operate in two modes:
1. **Two-file mode**: Separate files for train and test splits.
2. **Single-file mode**: One file with a column indicating split membership.

Classes
-------
FileLoader
    Loader for file-based datasets. Reads wide-format tables and returns
    data in the :class:`TSCData` container format.

Examples
--------
>>> from tscf_eval.data_loader import FileLoader
>>>
>>> # Two-file mode
>>> loader = FileLoader(
...     train_path="data/train.csv",
...     test_path="data/test.csv",
...     label_col="target"
... )
>>> train_data = loader.load("train")
>>> test_data = loader.load("test")
>>>
>>> # Single-file mode with split column
>>> loader = FileLoader(
...     data_path="data/full_dataset.csv",
...     split_col="split",
...     label_col="target"
... )
>>> train_data = loader.load("train")

See Also
--------
tscf_eval.data_loader.UCRLoader : For loading UCR archive datasets.
tscf_eval.data_loader.TSCData : Data container returned by loaders.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .base import DataLoader
from .tsc_data import Split, TSCData

if TYPE_CHECKING:
    from collections.abc import Sequence


class FileLoader(DataLoader):
    """Load a wide-format CSV/XLSX file (or pair of files) as ``TSCData``.

    Supports two modes:
      - Provide ``train_path`` and ``test_path`` (two-file mode).
      - Provide ``data_path`` and ``split_col`` indicating which rows
        belong to train/test (single-file mode).

    The table should be wide-format: one row per instance, numeric
    columns representing time points (or flattened channels), and a
    separate label column.
    """

    def __init__(
        self,
        *,
        train_path: str | Path | None = None,
        test_path: str | Path | None = None,
        data_path: str | Path | None = None,
        split_col: str | None = None,
        train_value: str = "train",
        test_value: str = "test",
        label_col: str | None = None,
        feature_cols: Sequence[str] | None = None,
        sheet_name: str | int | None = None,
        name: str = "local_wide",
    ):
        """Initialize a file-based loader.

        Parameters
        ----------
        train_path, test_path : str or pathlib.Path, optional
            Paths to separate train/test files (two-file mode). Mutually
            exclusive with ``data_path`` mode.
        data_path : str or pathlib.Path, optional
            Path to a single file containing both splits; requires
            ``split_col`` to be provided.
        split_col : str, optional
            Column name in ``data_path`` indicating split membership.
        train_value, test_value : str
            Values in ``split_col`` that indicate train/test rows.
        label_col : str
            Column name containing labels (required).
        feature_cols : sequence of str, optional
            Optional explicit list of feature columns to use.
        sheet_name : str or int, optional
            When reading Excel files, the sheet to use.
        name : str
            Dataset name to assign to produced ``TSCData`` objects.
        """
        two = (train_path is not None) and (test_path is not None) and (data_path is None)
        one = (
            (data_path is not None)
            and (train_path is None)
            and (test_path is None)
            and (split_col is not None)
        )
        if not two ^ one:
            raise ValueError("Provide either (train_path & test_path) XOR (data_path & split_col).")
        if label_col is None:
            raise ValueError("label_col is required.")

        self.train_path = Path(train_path).expanduser() if train_path else None
        self.test_path = Path(test_path).expanduser() if test_path else None
        self.data_path = Path(data_path).expanduser() if data_path else None
        self.split_col = split_col
        self.train_value = str(train_value).lower()
        self.test_value = str(test_value).lower()
        self.label_col = label_col
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.sheet_name = sheet_name
        self.name = name

    def load(self, split: Split, **kwargs) -> TSCData:
        """Load the requested split and return a :class:`TSCData`.

        Parameters
        ----------
        split : {'train', 'test'}
            Which split to load.
        **kwargs
            Additional options (not currently used).

        Returns
        -------
        TSCData
            Dataset container with feature arrays ``X`` and labels ``y``.
            ``X`` has shape ``(N, T)`` where ``N`` is the number of instances
            and ``T`` is the number of time points (feature columns).

        Raises
        ------
        ValueError
            If ``split_col`` is specified but not found in the data file.
        """
        if self.data_path is not None:
            df = self._read_table(self.data_path)
            if self.split_col not in df.columns:
                raise ValueError(f"split_col='{self.split_col}' not found in {self.data_path}")
            want = self.train_value if split == "train" else self.test_value
            df_split = df[df[self.split_col].astype(str).str.lower() == want]
            return TSCData.from_dataframe(
                name=self.name,
                split=split,
                df=df_split,
                label_col=self.label_col,
                feature_cols=self.feature_cols,
            )

        assert self.train_path is not None and self.test_path is not None
        path = self.train_path if split == "train" else self.test_path
        df = self._read_table(path)
        return TSCData.from_dataframe(
            name=self.name,
            split=split,
            df=df,
            label_col=self.label_col,
            feature_cols=self.feature_cols,
        )

    def describe(self) -> dict:
        """Return a concise description for the dataset(s) represented by this loader.

        The return value includes per-split metadata (via
        :meth:`TSCData.describe`) and an overall summary (combined
        number of classes across splits).

        Returns
        -------
        dict
            Dictionary with keys:

            - ``'name'``: Dataset name.
            - ``'splits'``: Dict mapping 'train'/'test' to their descriptions.
            - ``'overall'``: Dict with ``'n_classes'`` (total unique classes).
        """
        tr = self.load("train")
        te = self.load("test")
        n_classes = int(np.unique(np.concatenate([tr.y, te.y])).size)
        return {
            "name": self.name,
            "splits": {"train": tr.describe(), "test": te.describe()},
            "overall": {"n_classes": n_classes},
        }

    def _read_table(self, path: Path) -> pd.DataFrame:
        """Read a table from CSV or Excel into a DataFrame.

        Parameters
        ----------
        path : pathlib.Path
            Path to the file. Supported suffixes: .csv, .txt, .xlsx, .xls.

        Returns
        -------
        pandas.DataFrame
            Loaded table.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file extension is unsupported.
        """
        if not path.exists():
            raise FileNotFoundError(str(path))
        suffix = path.suffix.lower()
        if suffix in (".csv", ".txt"):
            return pd.read_csv(path)
        if suffix in (".xlsx", ".xls"):
            res = pd.read_excel(path, sheet_name=self.sheet_name)
            if isinstance(res, dict):
                return next(iter(res.values()))
            return res
        raise ValueError(f"Unsupported file type: {suffix}")
