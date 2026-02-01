"""CoMTE counterfactual explainer implementation.

This module provides the ``COMTE`` class, an implementation of the CoMTE
(Counterfactual Explanations for Multivariate Time Series) algorithm for
generating counterfactual explanations for time series classification.

The algorithm was originally developed by Emre Ates, Burak Aksar, Vitus J. Leung,
and Ayse K. Coskun at Boston University's PEAC Lab.

Original implementation: https://github.com/peaclab/CoMTE

Classes
-------
COMTE
    CoMTE counterfactual generator using greedy channel substitution.

Algorithm Overview
------------------
CoMTE generates counterfactuals through a sequential greedy approach:

1. Select distractor candidates from the reference set that are predicted
   as the target class.
2. For each distractor, greedily swap channels from the distractor into
   the query series, selecting the channel that most increases the target
   class probability at each step.
3. Choose the best counterfactual across all distractors using the loss
   function: ``L = max(0, tau - f_c)^2 + lambda_reg * max(0, n_vars - delta)``

Examples
--------
>>> from tscf_eval.counterfactuals import COMTE
>>> import numpy as np
>>>
>>> # Assume clf is a trained classifier
>>> comte = COMTE(
...     model=clf,
...     data=(X_train, y_train),
...     distance="dtw",
...     n_distractors=10,
...     tau=0.95,
... )
>>>
>>> # Generate counterfactual for a test instance
>>> cf, cf_label, meta = comte.explain(x_test)
>>> print(f"Edited channels: {meta['edits_variables']}")
>>> print(f"Target probability: {meta['target_prob']:.3f}")

References
----------
.. [comte1] Ates, E., Aksar, B., Leung, V. J., & Coskun, A. K. (2021).
       Counterfactual Explanations for Multivariate Time Series.
       In Proceedings of the 2021 International Conference on Applied
       Artificial Intelligence (ICAPAI), pp. 1-8.
       DOI: 10.1109/ICAPAI49758.2021.9462056

.. [comte2] Hollig, J., Kulbach, C., & Thoma, S. (2023).
       TSInterpret: A Python Package for the Interpretability of Time Series
       Classification. Journal of Open Source Software, 8(85), 5220.
       https://doi.org/10.21105/joss.05220
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import warnings

import numpy as np

from .base import Counterfactual
from .utils import (
    dtw_distance_vec_multich,
    ensure_batch_shape,
    euclidean_cdist_flat,
    soft_predict_proba_fn,
    strip_batch,
)


@dataclass
class COMTE(Counterfactual):
    """CoMTE (Sequential Greedy) counterfactual generator for time-series.

    Implementation of the CoMTE algorithm by Ates et al. (2021) [comte1]_.

    Produces counterfactuals by greedily replacing whole variables (channels)
    from distractor series drawn from a reference set. Distractors are
    selected among reference instances predicted as the target class. For each
    distractor the algorithm performs a sequential greedy search that replaces
    channels one-by-one, choosing at each step the channel swap that most
    increases the model probability ``f_c`` of the target class. The best
    counterfactual across distractors is chosen using the paper's loss::

        L = max(0, tau - f_c)^2 + lambda_reg * max(0, n_vars - delta)

    Supported distances
    -------------------
    - 'dtw' : multivariate DTW via ``dtw_distance_vec_multich``
    - 'euclidean' : Euclidean distance using flattened pairwise distances

    Parameters
    ----------
    model : object
        A classifier with a probability estimator (``predict_proba`` or a
        compatible interface). The helper ``predict_proba_fn`` wraps model
        inference.
    data : tuple (X_ref, y_ref)
        Reference dataset used to select distractors.
    distance : {'euclidean', 'dtw'}
        Distance metric to find nearest distractors.
    n_distractors : int
        Maximum number of distractors to try.
    tau : float
        Target probability threshold for class ``c``.
    delta : int
        Preferred number of variable edits (paper's sweet spot).
    lambda_reg : float
        Regularization weight in the paper loss.
    random_state : Optional[int]
        Seed for reproducible distractor tie-breaking.

    References
    ----------
    .. [comte1] Ates, E., Aksar, B., Leung, V. J., & Coskun, A. K. (2021).
           Counterfactual Explanations for Multivariate Time Series.
           ICAPAI 2021. https://github.com/peaclab/CoMTE
    """

    model: Any
    data: tuple[np.ndarray, np.ndarray]  # (X_ref, y_ref)
    distance: Literal["euclidean", "dtw"] = "dtw"
    n_distractors: int = 10  # try up to n candidates
    tau: float = 0.95  # target prob threshold
    delta: int = 3  # min-len sweet spot in L
    lambda_reg: float = 0.8  # λ in the paper's loss
    random_state: int | None = 0

    def __post_init__(self):
        self.predict_proba = soft_predict_proba_fn(self.model)
        self.rng = np.random.default_rng(self.random_state)
        self.X_ref = np.asarray(self.data[0])
        self.y_ref = np.asarray(self.data[1]).ravel()
        # Pre-compute reference set predictions to avoid redundant calls in _select_distractors
        self._ref_probs = self.predict_proba(self.X_ref)
        self._ref_yhat = np.argmax(self._ref_probs, axis=1)

    def explain(
        self,
        x: np.ndarray,
        y_pred: int | None = None,
        *,
        class_of_interest: int | None = None,
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        """Generate a counterfactual toward a class of interest.

        Parameters
        ----------
        x : np.ndarray
            Input time series of shape ``(T,)`` for univariate or ``(C, T)``
            for multivariate data.
        y_pred : int, optional
            Base predicted class for ``x``. If ``None``, computed via the model.
        class_of_interest : int, optional
            Target class for the counterfactual. If ``None``, uses the
            highest-probability alternative to ``y_pred``.

        Returns
        -------
        cf : np.ndarray
            Counterfactual time series with the same shape as ``x``.
        cf_label : int
            Predicted class label for the counterfactual.
        meta : dict
            Metadata dictionary containing:

            - ``method``: Algorithm identifier (``'comte_greedy'``).
            - ``distance``: Distance metric used.
            - ``class_of_interest``: Target class.
            - ``tau``, ``delta``, ``lambda_reg``: Algorithm parameters.
            - ``distractor_index_in_ref``: Index of selected distractor.
            - ``distractor_distance``: Distance to selected distractor.
            - ``edits_variables``: List of edited channel indices.
            - ``target_prob``: Final target class probability.
            - ``loss``: Final loss value.
        """
        xb, added = ensure_batch_shape(x)
        x1 = strip_batch(xb, added)

        # Determine base prediction and target class c
        base_probs = self.predict_proba(xb)[0]
        base_label = int(np.argmax(base_probs)) if y_pred is None else int(y_pred)

        if class_of_interest is None:
            # choose the best alternative class (highest prob not equal to base)
            probs_sorted = np.argsort(-base_probs)
            class_of_interest = int(next(c for c in probs_sorted if c != base_label))

        # 1) choose distractor candidates predicted as the class of interest
        dmeta, distractors = self._select_distractors(x1, class_of_interest)

        # 2) run greedy search per distractor and pick the best by loss
        best = None
        for i, distractor in enumerate(distractors):
            cf, edits, fc = self._greedy_channel_search(x1, distractor, class_of_interest)
            loss = self._compute_loss(fc, len(edits))
            item = (loss, i, cf, edits, fc)
            if (best is None) or (loss < best[0]):
                best = item

        # Fallback if no distractors found: return x unchanged
        if best is None:
            warnings.warn(
                f"COMTE: No distractors found for target class {class_of_interest}. "
                f"This typically occurs when the classifier predicts all reference "
                f"samples as the same class (base class={base_label}). The original "
                f"instance is returned unchanged. Consider using a different dataset "
                f"or a classifier with more diverse predictions.",
                UserWarning,
                stacklevel=2,
            )
            return (
                x1,
                int(base_label),
                {
                    "method": "comte_greedy",
                    "distance": self.distance,
                    "class_of_interest": class_of_interest,
                    "validity": False,
                    "failure_reason": "no_distractors",
                    "note": "no distractors found; returning original unchanged",
                    "tau": self.tau,
                    "delta": self.delta,
                    "lambda": self.lambda_reg,
                },
            )

        loss, i, cf, edits, fc = best
        final_label = int(np.argmax(self.predict_proba(cf[None, ...])[0]))

        meta: dict[str, Any] = {
            "method": "comte_greedy",
            "distance": self.distance,
            "class_of_interest": class_of_interest,
            "tau": float(self.tau),
            "delta": int(self.delta),
            # keep backwards-compatible key name 'lambda' but also expose
            # explicit 'lambda_reg' to avoid confusion with the Python keyword
            "lambda": float(self.lambda_reg),
            "lambda_reg": float(self.lambda_reg),
            "distractor_index_in_ref": (dmeta["indices"][i] if dmeta["indices"] else None),
            "distractor_distance": (dmeta["distances"][i] if dmeta["distances"] else None),
            "edits_variables": edits,  # list[int] of channels swapped
            "target_prob": float(fc),
            "loss": float(loss),
        }
        return cf, final_label, meta

    def explain_k(
        self,
        x: np.ndarray,
        k: int = 5,
        y_pred: int | None = None,
        *,
        class_of_interest: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Generate k diverse counterfactuals using different distractors.

        COMTE naturally supports diverse counterfactual generation by using
        different distractor instances from the reference set. Each CF is
        generated using a different distractor, producing structurally
        diverse explanations.

        Parameters
        ----------
        x : np.ndarray
            Input time series.
        k : int, default 5
            Number of counterfactuals to generate.
        y_pred : int, optional
            Precomputed predicted label for ``x``.
        class_of_interest : int, optional
            Target class for counterfactuals.

        Returns
        -------
        cfs : np.ndarray
            Array of k counterfactuals.
        cf_labels : np.ndarray
            Array of k predicted labels.
        metas : list[dict]
            List of k metadata dictionaries.
        """
        xb, added = ensure_batch_shape(x)
        x1 = strip_batch(xb, added)

        # Determine base prediction and target class
        base_probs = self.predict_proba(xb)[0]
        base_label = int(np.argmax(base_probs)) if y_pred is None else int(y_pred)

        if class_of_interest is None:
            probs_sorted = np.argsort(-base_probs)
            class_of_interest = int(next(c for c in probs_sorted if c != base_label))

        # Get all distractors (request more than k to have options)
        orig_n_distractors = self.n_distractors
        self.n_distractors = max(k * 2, orig_n_distractors)
        dmeta, distractors = self._select_distractors(x1, class_of_interest)
        self.n_distractors = orig_n_distractors

        if not distractors:
            # No distractors found - return k copies of original with failure metadata
            cfs = np.array([x1 for _ in range(k)])
            cf_labels = np.array([base_label for _ in range(k)])
            metas = [
                {
                    "method": "comte_greedy",
                    "k_index": i,
                    "validity": False,
                    "failure_reason": "no_distractors",
                }
                for i in range(k)
            ]
            return cfs, cf_labels, metas

        # Generate CF for each distractor (up to k)
        results: list[tuple[np.ndarray, int, dict[str, Any]]] = []
        for i, distractor in enumerate(distractors[:k]):
            cf, edits, fc = self._greedy_channel_search(x1, distractor, class_of_interest)
            loss = self._compute_loss(fc, len(edits))
            final_label = int(np.argmax(self.predict_proba(cf[None, ...])[0]))
            meta = {
                "method": "comte_greedy",
                "k_index": i,
                "distance": self.distance,
                "class_of_interest": class_of_interest,
                "distractor_index_in_ref": dmeta["indices"][i] if dmeta["indices"] else None,
                "distractor_distance": dmeta["distances"][i] if dmeta["distances"] else None,
                "edits_variables": edits,
                "target_prob": float(fc),
                "loss": float(loss),
            }
            results.append((cf, final_label, meta))

        # If we have fewer than k distractors, pad with best result
        while len(results) < k:
            # Repeat the best (lowest loss) result
            best_idx = min(range(len(results)), key=lambda j: float(results[j][2]["loss"]))
            cf, label, meta = results[best_idx]
            new_meta = meta.copy()
            new_meta["k_index"] = len(results)
            new_meta["note"] = "duplicated from best result"
            results.append((cf.copy(), label, new_meta))

        cfs = np.array([r[0] for r in results])
        cf_labels = np.array([r[1] for r in results])
        metas = [r[2] for r in results]

        return cfs, cf_labels, metas

    def _select_distractors(self, x: np.ndarray, c: int) -> tuple[dict[str, Any], list[np.ndarray]]:
        """Select distractor candidates from the reference set.

        Finds instances in the reference set that are predicted as the target
        class and ranks them by distance to the query.

        Parameters
        ----------
        x : np.ndarray
            Query time series of shape ``(T,)`` or ``(C, T)``.
        c : int
            Target class index.

        Returns
        -------
        metadata : dict
            Dictionary with ``'indices'`` (reference set indices) and
            ``'distances'`` (distances to query).
        distractors : list of np.ndarray
            List of distractor time series arrays.
        """
        if self.X_ref.ndim not in (2, 3):
            raise ValueError(f"Unsupported X_ref shape {self.X_ref.shape}")

        # Use pre-computed reference set predictions (computed once in __post_init__)
        yhat = self._ref_yhat

        # Paper: choose training instances that are *correctly classified* as class c
        mask = (self.y_ref == c) & (yhat == c)
        if not np.any(mask):
            # fallback 1: keep predicted==c only
            mask = yhat == c
        if not np.any(mask):
            # fallback 2: keep ground-truth==c only
            mask = self.y_ref == c
        if not np.any(mask):
            return {"indices": [], "distances": []}, []

        Xc = self.X_ref[mask]

        if self.distance == "dtw":
            dvec = dtw_distance_vec_multich(x, Xc)
        else:
            xb, _ = ensure_batch_shape(x)
            dvec = euclidean_cdist_flat(xb, Xc).ravel()

        order = np.argsort(dvec)
        k = min(self.n_distractors, len(order))
        picks = order[:k]

        idx_in_ref = np.flatnonzero(mask)[picks]
        distractors = [Xc[j] for j in picks]

        return {
            "indices": idx_in_ref.tolist(),
            "distances": [float(dvec[j]) for j in picks],
        }, distractors

    def _greedy_channel_search(
        self, x: np.ndarray, distractor: np.ndarray, target_class: int
    ) -> tuple[np.ndarray, list[int], float]:
        """Greedily select channels to swap for maximum target class probability.

        At each step, selects the channel whose substitution from the distractor
        most increases the target class probability. Stops when probability
        reaches tau or no further improvement is possible.

        Parameters
        ----------
        x : np.ndarray
            Original time series of shape (T,) or (C, T).
        distractor : np.ndarray
            Distractor series to copy channels from, same shape as x.
        target_class : int
            Target class index to optimize probability for.

        Returns
        -------
        tuple
            (counterfactual, edited_channels, final_target_prob)
        """
        if x.ndim == 1:
            return self._greedy_search_univariate(x, distractor, target_class)
        return self._greedy_search_multivariate(x, distractor, target_class)

    def _greedy_search_univariate(
        self, x: np.ndarray, distractor: np.ndarray, target_class: int
    ) -> tuple[np.ndarray, list[int], float]:
        """Greedy search for univariate series.

        For univariate data (single channel), compares swapping the entire
        series vs keeping original, choosing whichever minimizes the loss.

        Parameters
        ----------
        x : np.ndarray
            Original univariate time series of shape ``(T,)``.
        distractor : np.ndarray
            Distractor series of shape ``(T,)``.
        target_class : int
            Target class index to optimize probability for.

        Returns
        -------
        cf : np.ndarray
            Counterfactual series of shape ``(T,)``.
        edited_channels : list of int
            List of edited channel indices (``[0]`` if swapped, ``[]`` if not).
        target_prob : float
            Final probability of the target class.
        """
        base_prob = float(self.predict_proba(x[None, ...])[0][target_class])
        cf_candidate = distractor.copy()
        cf_prob = float(self.predict_proba(cf_candidate[None, ...])[0][target_class])

        loss_orig = self._compute_loss(base_prob, 0)
        loss_swap = self._compute_loss(cf_prob, 1)

        if loss_swap < loss_orig:
            return cf_candidate, [0], cf_prob
        return x.copy(), [], base_prob

    def _greedy_search_multivariate(
        self, x: np.ndarray, distractor: np.ndarray, target_class: int
    ) -> tuple[np.ndarray, list[int], float]:
        """Greedy search across channels for multivariate series.

        Iteratively selects the channel swap that most increases the target
        class probability, stopping when probability reaches tau or no
        further improvement is possible.

        Parameters
        ----------
        x : np.ndarray
            Original multivariate time series of shape ``(C, T)``.
        distractor : np.ndarray
            Distractor series of shape ``(C, T)``.
        target_class : int
            Target class index to optimize probability for.

        Returns
        -------
        cf : np.ndarray
            Counterfactual series of shape ``(C, T)``.
        edited_channels : list of int
            List of edited channel indices in order of selection.
        target_prob : float
            Final probability of the target class.
        """
        C, _ = x.shape
        edited_channels: list[int] = []
        cf = x.copy()
        current_prob = float(self.predict_proba(cf[None, ...])[0][target_class])

        remaining_channels = set(range(C))
        MIN_GAIN = 1e-9

        while current_prob < self.tau and remaining_channels:
            best_gain = -np.inf
            best_channel = None
            best_prob = current_prob

            for channel in list(remaining_channels):
                candidate = cf.copy()
                candidate[channel, :] = distractor[channel, :]
                candidate_prob = float(self.predict_proba(candidate[None, ...])[0][target_class])
                gain = candidate_prob - current_prob
                if gain > best_gain:
                    best_gain = gain
                    best_channel = channel
                    best_prob = candidate_prob

            if best_channel is None or best_gain <= MIN_GAIN:
                # If no channel selected yet, try single-channel swap
                # that minimizes loss (even if gain is tiny)
                if not edited_channels:
                    best_channel = None
                    best_prob = current_prob
                    best_loss = self._compute_loss(current_prob, 0)
                    for channel in remaining_channels:
                        candidate = cf.copy()
                        candidate[channel, :] = distractor[channel, :]
                        candidate_prob = float(
                            self.predict_proba(candidate[None, ...])[0][target_class]
                        )
                        candidate_loss = self._compute_loss(candidate_prob, 1)
                        if candidate_loss < best_loss:
                            best_loss = candidate_loss
                            best_prob = candidate_prob
                            best_channel = channel
                    if best_channel is not None:
                        cf[best_channel, :] = distractor[best_channel, :]
                        edited_channels.append(best_channel)
                        remaining_channels.remove(best_channel)
                        current_prob = best_prob
                        continue
                break

            # Commit the best channel swap
            cf[best_channel, :] = distractor[best_channel, :]
            edited_channels.append(best_channel)
            remaining_channels.remove(best_channel)
            current_prob = best_prob

        return cf, edited_channels, current_prob

    def _compute_loss(self, target_prob: float, n_edits: int) -> float:
        """Compute the counterfactual loss balancing validity and sparsity.

        The loss combines two terms:
        1. Validity penalty: penalizes low target class probability
        2. Sparsity penalty: penalizes using more than delta variable edits

        Loss = max(0, tau - target_prob)^2 + lambda_reg * max(0, n_edits - delta)

        Parameters
        ----------
        target_prob : float
            Probability of the target class for the candidate counterfactual.
        n_edits : int
            Number of variables (channels) edited in the counterfactual.

        Returns
        -------
        float
            Combined loss value (lower is better).
        """
        validity_penalty = max(0.0, self.tau - target_prob) ** 2
        sparsity_penalty = float(self.lambda_reg) * max(0, n_edits - int(self.delta))
        return float(validity_penalty + sparsity_penalty)
