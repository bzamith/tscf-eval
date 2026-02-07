"""Configuration dataclasses for the benchmark framework.

This module provides immutable configuration containers for datasets,
models, and explainers used in benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from typing import Protocol

    class Counterfactual(Protocol):
        """Protocol for counterfactual explainer classes."""

        def __init__(
            self, model: Any, data: tuple[np.ndarray, np.ndarray], **kwargs: Any
        ) -> None: ...

        def explain(
            self, x: np.ndarray, y_pred: int | None = None
        ) -> tuple[np.ndarray, int, dict[str, Any]]: ...

        def explain_k(
            self, x: np.ndarray, k: int, y_pred: int | None = None
        ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]: ...


__all__ = [
    "DatasetConfig",
    "ExplainerConfig",
    "ModelConfig",
]


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a dataset in benchmarks.

    Parameters
    ----------
    name : str
        Unique identifier for this dataset.
    X_train : np.ndarray
        Training features, shape ``(n_train, series_length)`` or
        ``(n_train, n_channels, series_length)``.
    y_train : np.ndarray
        Training labels, shape ``(n_train,)``.
    X_test : np.ndarray
        Test features, same shape convention as X_train.
    y_test : np.ndarray, optional
        Test labels, shape ``(n_test,)``. Required for some metrics.

    Examples
    --------
    >>> from tscf_eval.benchmark import DatasetConfig
    >>>
    >>> dataset = DatasetConfig(
    ...     name="GunPoint",
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     X_test=X_test,
    ...     y_test=y_test,
    ... )
    """

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate and coerce array fields to numpy arrays."""
        object.__setattr__(self, "X_train", np.asarray(self.X_train))
        object.__setattr__(self, "y_train", np.asarray(self.y_train).ravel())
        object.__setattr__(self, "X_test", np.asarray(self.X_test))
        if self.y_test is not None:
            object.__setattr__(self, "y_test", np.asarray(self.y_test).ravel())

    @property
    def n_train(self) -> int:
        """Number of training instances.

        Returns
        -------
        int
            Length of ``X_train`` along axis 0.
        """
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        """Number of test instances.

        Returns
        -------
        int
            Length of ``X_test`` along axis 0.
        """
        return len(self.X_test)

    @property
    def series_length(self) -> int:
        """Return the length of each time series.

        Returns
        -------
        int
            Last dimension of ``X_train``.
        """
        return int(self.X_train.shape[-1])


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a classifier model in benchmarks.

    The model must be pre-fitted before being passed to the benchmark.

    Parameters
    ----------
    name : str
        Unique identifier for this model (e.g., "knn_dtw", "rocket").
    model : Any
        Pre-trained classifier with ``predict()`` method.
        Should also have ``predict_proba()`` for some metrics.

    Examples
    --------
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from tscf_eval.benchmark import ModelConfig
    >>>
    >>> knn = KNeighborsClassifier(n_neighbors=1)
    >>> knn.fit(X_train, y_train)
    >>>
    >>> model_config = ModelConfig("knn", knn)
    """

    name: str
    model: Any

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Parameters
        ----------
        X : np.ndarray
            Input instances, shape ``(n_samples, ...)``.

        Returns
        -------
        np.ndarray
            Predicted class labels, shape ``(n_samples,)``.
        """
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Predict class probabilities if available.

        Parameters
        ----------
        X : np.ndarray
            Input instances, shape ``(n_samples, ...)``.

        Returns
        -------
        np.ndarray or None
            Class probabilities of shape ``(n_samples, n_classes)``,
            or ``None`` if the model lacks ``predict_proba``.
        """
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X))
        return None


@dataclass(frozen=True)
class ExplainerConfig:
    """Configuration for a counterfactual explainer.

    Parameters
    ----------
    name : str
        Unique identifier for this configuration (used in results).
    explainer_class : type[Counterfactual]
        The counterfactual explainer class (e.g., COMTE, NativeGuide).
    params : dict, optional
        Parameters to pass to the explainer constructor.
        ``model`` and ``data`` are provided automatically by the runner.
    n_counterfactuals : int, default 1
        Number of counterfactuals to generate per instance.
        When > 1, uses ``explain_k()`` method.

    Examples
    --------
    >>> from tscf_eval import COMTE, NativeGuide
    >>> from tscf_eval.benchmark import ExplainerConfig
    >>>
    >>> configs = [
    ...     ExplainerConfig("comte_dtw", COMTE, {"distance": "dtw"}),
    ...     ExplainerConfig("ng_blend", NativeGuide, {"method": "blend"}),
    ... ]
    """

    name: str
    explainer_class: type[Counterfactual]
    params: dict[str, Any] = field(default_factory=dict)
    n_counterfactuals: int = 1

    def create_explainer(
        self,
        model: Any,
        data: tuple[np.ndarray, np.ndarray],
    ) -> Counterfactual:
        """Instantiate the explainer with model and training data.

        Parameters
        ----------
        model : Any
            Fitted classifier.
        data : tuple[np.ndarray, np.ndarray]
            Tuple of (X_train, y_train).

        Returns
        -------
        Counterfactual
            Configured explainer instance.
        """
        return self.explainer_class(model=model, data=data, **self.params)
