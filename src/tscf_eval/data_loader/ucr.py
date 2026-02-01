"""UCR Time Series Classification Archive loader.

This module provides the :class:`UCRLoader` class for loading datasets from
the UCR Time Series Classification Archive. The loader delegates to the
``aeon`` library for data retrieval and caching.

The UCR archive contains over 100 univariate and multivariate time-series
classification datasets commonly used for benchmarking. Datasets are
automatically downloaded and cached on first access.

Classes
-------
UCRLoader
    Loader for UCR archive datasets. Wraps ``aeon.datasets.load_classification``
    and returns data in the :class:`TSCData` container format.

Examples
--------
>>> from tscf_eval.data_loader import UCRLoader
>>>
>>> # Load the ItalyPowerDemand dataset
>>> loader = UCRLoader("ItalyPowerDemand")
>>> train_data = loader.load("train")
>>> test_data = loader.load("test")
>>>
>>> print(train_data.describe())
{'name': 'ItalyPowerDemand', 'split': 'train', ...}
>>>
>>> # Load both splits at once
>>> train, test = loader.load_both()

References
----------
.. [1] Dau, H. A., et al. (2019). The UCR time series archive.
       IEEE/CAA Journal of Automatica Sinica, 6(6), 1293-1305.
       https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

See Also
--------
tscf_eval.data_loader.FileLoader : For loading custom file-based datasets.
tscf_eval.data_loader.TSCData : Data container returned by loaders.
"""

from __future__ import annotations

from aeon.datasets import load_classification
import numpy as np

from .base import DataLoader
from .tsc_data import Split, TSCData


class UCRLoader(DataLoader):
    """Loader for UCR time-series classification datasets from the UCR archive.

    This loader delegates to the ``aeon`` library's dataset utilities
    (``aeon.datasets.load_classification``). The ``aeon`` package must be
    installed for this loader to work.

    Parameters
    ----------
    dataset_name : str
        Name of the UCR dataset (e.g., 'ItalyPowerDemand', 'GunPoint').
    """

    def __init__(self, dataset_name: str):
        """Create a loader for a named UCR dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the UCR dataset (e.g., 'ItalyPowerDemand').
        """
        self.dataset_name = dataset_name

    def load(self, split: Split, **kwargs) -> TSCData:
        """Load a split ('train' or 'test') of the dataset using aeon.

        Parameters
        ----------
        split : {'train', 'test'}
            Which split to load.
        **kwargs
            Additional arguments forwarded to the underlying loader in aeon.

        Returns
        -------
        TSCData
            Dataset container with feature arrays ``X`` and labels ``y``.
            For univariate datasets, ``X`` has shape ``(N, T)``.
            For multivariate datasets, ``X`` has shape ``(N, C, T)``
            where ``C`` is the number of channels/dimensions.
        """
        # aeon versions differ: some return (X, y) when asked with
        # return_X_y=True, others return an object with attributes.
        res = load_classification(name=self.dataset_name, split=split)
        if isinstance(res, tuple) and len(res) == 2:
            X_raw, y_raw = res
        elif getattr(res, "data", None) is not None and getattr(res, "target", None) is not None:
            X_raw, y_raw = res.data, res.target
        elif getattr(res, "X", None) is not None and getattr(res, "y", None) is not None:
            X_raw, y_raw = res.X, res.y
        else:
            raise RuntimeError("Unexpected return value from aeon.datasets.load_classification")

        X = np.asarray(X_raw)
        y = np.asarray(y_raw)
        return TSCData.from_arrays(
            name=self.dataset_name, split=split, X=X, y=y, squeeze_univariate=True
        )

    def describe(self) -> dict:
        """Return a compact description for the dataset.

        The description contains per-split metadata (from
        :meth:`TSCData.describe`) and an overall summary (currently the
        combined number of classes observed across splits).

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
            "name": self.dataset_name,
            "splits": {"train": tr.describe(), "test": te.describe()},
            "overall": {"n_classes": n_classes},
        }
