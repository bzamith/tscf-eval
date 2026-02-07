"""Utility helpers for counterfactual explainers.

This package provides small, focused helpers used by counterfactual explainers,
organized into the following modules:

Shape Utilities (_shape)
------------------------
ensure_batch_shape
    Ensures array has a leading batch dimension.
strip_batch
    Removes batch dimension added by ``ensure_batch_shape``.

Prediction Utilities (_predict)
-------------------------------
predict_proba_fn
    Wraps a classifier's ``predict_proba`` method for consistent interface.
soft_predict_proba_fn
    Wraps a classifier to provide soft probabilities for gradient-based methods.
    Handles ROCKET-style classifiers that return hard 0/1 probabilities.
supports_soft_probabilities
    Check if a classifier supports smooth probability estimates for gradients.

Distance Functions (_distance)
------------------------------
euclidean_cdist_flat
    Computes pairwise Euclidean distances after flattening.
dtw_distance_vec_multich
    Computes DTW distance averaged across channels.

DTW/DBA Utilities (_dba)
------------------------
weighted_dba_multich
    Weighted DTW barycenter averaging per channel (multivariate).
dba_barycenter_multich
    DTW barycenter averaging for multiple sequences.

Adam Optimizer (_adam)
---------------------
AdamState
    Stateful Adam optimizer for NumPy gradient-based optimization loops.

Nearest Unlike Neighbor (_nun)
------------------------------
find_nearest_unlike_neighbor
    Find the nearest instance(s) from a reference set belonging to a target class.

Notes
-----
Optional dependencies:
- ``tslearn``: Required for DTW-based functions. Falls back to Euclidean
  distance when unavailable.
- ``scipy``: Used for optimized pairwise distance computation when available.
"""

from ._adam import AdamState
from ._dba import (
    dba_barycenter_multich,
    weighted_dba_multich,
)
from ._distance import (
    dtw_distance_vec_multich,
    euclidean_cdist_flat,
)
from ._nun import find_nearest_unlike_neighbor
from ._predict import (
    has_expensive_transform,
    predict_proba_fn,
    soft_predict_proba_fn,
    supports_soft_probabilities,
)
from ._shape import ensure_batch_shape, strip_batch

__all__ = [
    "AdamState",
    "dba_barycenter_multich",
    "dtw_distance_vec_multich",
    "ensure_batch_shape",
    "euclidean_cdist_flat",
    "find_nearest_unlike_neighbor",
    "has_expensive_transform",
    "predict_proba_fn",
    "soft_predict_proba_fn",
    "strip_batch",
    "supports_soft_probabilities",
    "weighted_dba_multich",
]
