"""Base interface for dataset loaders.

This module defines the abstract :class:`DataLoader` class which concrete
loaders should subclass. The interface exposes a consistent API for loading
time-series classification datasets regardless of their source (files, UCR
archive, databases, etc.).

Classes
-------
DataLoader
    Abstract base class defining the loader interface with ``load``,
    ``describe``, and ``load_both`` methods.

See Also
--------
tscf_eval.data_loader.ucr : UCR archive loader implementation.
tscf_eval.data_loader.files : File-based loader implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tsc_data import Split, TSCData


class DataLoader(ABC):
    """Abstract base class for dataset loaders.

    Subclasses implement dataset-specific loading logic. Implementations
    must provide :meth:`load` which returns a :class:`TSCData` for the
    requested split and :meth:`describe` which returns a small metadata
    dictionary suitable for discovery and logging. Use :meth:`load_both`
    as a convenience to obtain both train and test splits.
    """

    @abstractmethod
    def load(self, split: Split, **kwargs) -> TSCData:
        """Load and return a :class:`TSCData` for ``split``.

        Parameters
        ----------
        split : {'train', 'test'}
            Which split to load.
        **kwargs
            Loader-specific options forwarded to the concrete loader.

        Returns
        -------
        TSCData
            Loaded dataset for the requested split.

        Raises
        ------
        RuntimeError
            Implementations may raise when the split is not available or
            when underlying I/O fails.
        """

    @abstractmethod
    def describe(self) -> dict:
        """Return a small metadata dictionary describing available datasets.

        The returned dictionary should contain enough information for
        discovery and logging (for example, available dataset names,
        default paths, and per-split summaries). The exact structure is
        loader-specific but should be JSON-serializable.

        Returns
        -------
        dict
            Metadata dictionary with loader-specific structure.
        """

    def load_both(self, **kwargs) -> tuple[TSCData, TSCData]:
        """Load both train and test splits and return them as a tuple.

        Parameters
        ----------
        **kwargs
            Loader-specific options forwarded to :meth:`load`.

        Returns
        -------
        train : TSCData
            Training dataset.
        test : TSCData
            Test dataset.
        """
        return self.load("train", **kwargs), self.load("test", **kwargs)
