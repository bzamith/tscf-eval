"""Data loader package for time series classification datasets.

This module provides utilities for loading and managing time series
classification datasets from various sources, including the UCR archive
and custom file formats.

Classes
-------
TSCData
    Immutable container for time series classification data. Holds the
    data arrays (X, y) along with metadata like dataset name and split.
DataLoader
    Abstract base class for implementing custom data loaders.
UCRLoader
    Loader for datasets from the UCR Time Series Classification Archive.
    Supports automatic download and caching via the aeon library.
FileLoader
    Loader for datasets stored in CSV, TXT, or Excel files. Supports
    both single-file and split-file configurations.

Example
-------
>>> from tscf_eval.data_loader import UCRLoader, TSCData
>>>
>>> # Load from UCR archive
>>> loader = UCRLoader("ItalyPowerDemand")
>>> train_data = loader.load("train")
>>> test_data = loader.load("test")
>>>
>>> # Inspect the data
>>> print(train_data.describe())
>>> print(f"Shape: {train_data.X.shape}")
>>> print(f"Classes: {train_data.n_classes}")

>>> # Load from custom file
>>> from tscf_eval.data_loader import FileLoader
>>> loader = FileLoader(
...     train_path="data/train.csv",
...     test_path="data/test.csv",
...     label_col="target"
... )
>>> train = loader.load("train")

See Also
--------
tscf_eval.evaluator : Evaluation metrics for counterfactuals.
tscf_eval.counterfactuals : Counterfactual generation algorithms.
"""

from .base import DataLoader
from .files import FileLoader
from .tsc_data import TSCData
from .ucr import UCRLoader

__all__ = ["DataLoader", "FileLoader", "TSCData", "UCRLoader"]
