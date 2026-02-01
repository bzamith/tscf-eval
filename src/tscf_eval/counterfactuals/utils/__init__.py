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
dtw_pair_average
    DTW-based pairwise averaging of two sequences.
weighted_dba_pair
    Weighted DTW barycenter averaging for two sequences.
weighted_dba_multich
    Weighted DTW barycenter averaging per channel (multivariate).
dba_barycenter_multich
    DTW barycenter averaging for multiple sequences.
replace_topk_univariate
    Replace top-k important points in univariate series.
replace_topk_multivariate
    Replace top-k important entries in multivariate series.

Notes
-----
Optional dependencies:
- ``tslearn``: Required for DTW-based functions. Falls back to Euclidean
  distance when unavailable.
- ``scipy``: Used for optimized pairwise distance computation when available.
"""

from ._dba import (
    dba_barycenter_multich,
    dtw_pair_average,
    replace_topk_multivariate,
    replace_topk_univariate,
    weighted_dba_multich,
    weighted_dba_pair,
)
from ._distance import (
    dtw_distance_vec_multich,
    euclidean_cdist_flat,
)
from ._predict import predict_proba_fn, soft_predict_proba_fn, supports_soft_probabilities
from ._shape import ensure_batch_shape, strip_batch

__all__ = [
    "dba_barycenter_multich",
    "dtw_distance_vec_multich",
    "dtw_pair_average",
    "ensure_batch_shape",
    "euclidean_cdist_flat",
    "predict_proba_fn",
    "replace_topk_multivariate",
    "replace_topk_univariate",
    "soft_predict_proba_fn",
    "strip_batch",
    "supports_soft_probabilities",
    "weighted_dba_multich",
    "weighted_dba_pair",
]
