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

_MIN_GAIN = 1e-9


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

    **Supported distances:**

    - ``'dtw'`` : multivariate DTW via ``dtw_distance_vec_multich``
    - ``'euclidean'`` : Euclidean distance using flattened pairwise distances

    Parameters
    ----------
    model : object
        A classifier with a probability estimator (``predict_proba`` or a
        compatible interface). The helper ``predict_proba_fn`` wraps model
        inference.
    data : tuple (``X_ref``, ``y_ref``)
        Reference dataset used to select distractors.
    distance : {'euclidean', 'dtw'}, default 'dtw'
        Distance metric to find nearest distractors.

        - ``'euclidean'``: Euclidean distance on flattened vectors. Faster but
          ignores temporal alignment.
        - ``'dtw'``: Dynamic Time Warping distance (per-channel, averaged).
          Respects temporal shifts and is recommended for time series.
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
        """Initialise probability wrapper, RNG, reference data, and label mapping.

        Validates all hyperparameters and pre-computes reference-set
        predictions to avoid redundant calls during distractor selection.

        Raises
        ------
        ValueError
            If ``distance`` is not in ``{'euclidean', 'dtw'}``,
            ``n_distractors < 1``, ``tau`` is outside ``(0, 1]``,
            ``delta < 1``, or ``lambda_reg < 0``.
        """
        # Validate parameters
        if self.distance not in ("euclidean", "dtw"):
            raise ValueError("distance must be one of {'euclidean', 'dtw'}")
        if self.n_distractors < 1:
            raise ValueError("n_distractors must be >= 1")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError("tau must be in (0, 1]")
        if self.delta < 1:
            raise ValueError("delta must be >= 1")
        if self.lambda_reg < 0:
            raise ValueError("lambda_reg must be >= 0")

        self.predict_proba = soft_predict_proba_fn(self.model)
        self.rng = np.random.default_rng(self.random_state)
        self.X_ref = np.asarray(self.data[0])
        self.y_ref = np.asarray(self.data[1]).ravel()

        self._init_label_mapping(self.model, self.y_ref)

        # Pre-compute reference set predictions to avoid redundant calls
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

        base_probs = self.predict_proba(xb)[0]
        base_idx = int(np.argmax(base_probs)) if y_pred is None else self._label_to_idx(y_pred)
        target_idx = self._resolve_target_class(base_probs, base_idx, class_of_interest)

        # Step 1: Find distractor candidates predicted as the target class
        distractor_meta, distractors = self._find_target_class_distractors(x1, target_idx)

        # Step 2: Greedy channel swap per distractor, keep the best
        best = self._select_best_distractor_result(x1, distractors, target_idx)

        # Step 3: Assemble result
        if best is None:
            return self._no_distractor_fallback(x1, base_idx, target_idx)

        loss, i, cf, edits, fc = best
        cf_idx = self._predict_class_idx(cf)
        cf_label = self._idx_to_label(cf_idx)
        meta = self._build_meta(target_idx, distractor_meta, i, edits, fc, loss)
        return cf, cf_label, meta

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

        base_probs = self.predict_proba(xb)[0]
        base_idx = int(np.argmax(base_probs)) if y_pred is None else self._label_to_idx(y_pred)
        target_idx = self._resolve_target_class(base_probs, base_idx, class_of_interest)

        # Step 1: Get distractors (request more than k to have options)
        orig_n_distractors = self.n_distractors
        self.n_distractors = max(k * 2, orig_n_distractors)
        distractor_meta, distractors = self._find_target_class_distractors(x1, target_idx)
        self.n_distractors = orig_n_distractors

        if not distractors:
            base_label = self._idx_to_label(base_idx)
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

        # Step 2: Generate a counterfactual per distractor (up to k)
        results: list[tuple[np.ndarray, Any, dict[str, Any]]] = []
        for i, distractor in enumerate(distractors[:k]):
            cf, edits, fc = self._swap_channels_greedily(x1, distractor, target_idx)
            loss = self._compute_loss(fc, len(edits))
            cf_idx = self._predict_class_idx(cf)
            meta = self._build_meta(target_idx, distractor_meta, i, edits, fc, loss, k_index=i)
            results.append((cf, self._idx_to_label(cf_idx), meta))

        # Step 3: Pad with best result if fewer than k distractors
        while len(results) < k:
            best_idx = min(
                range(len(results)),
                key=lambda j: float(results[j][2]["loss"]),
            )
            cf, label, meta = results[best_idx]
            new_meta = meta.copy()
            new_meta["k_index"] = len(results)
            new_meta["note"] = "duplicated from best result"
            results.append((cf.copy(), label, new_meta))

        cfs = np.array([r[0] for r in results])
        cf_labels = np.array([r[1] for r in results])
        metas = [r[2] for r in results]

        return cfs, cf_labels, metas

    def _find_target_class_distractors(
        self, x: np.ndarray, target_idx: int
    ) -> tuple[dict[str, Any], list[np.ndarray]]:
        """Find distractor candidates from the reference set.

        Selects instances predicted as the target class, ranked by distance
        to the query. Uses a cascading fallback strategy:

        1. Correctly-classified instances of the target class.
        2. Any instance predicted as the target class.
        3. Any instance with ground-truth target class label.

        Parameters
        ----------
        x : np.ndarray
            Query time series of shape ``(T,)`` or ``(C, T)``.
        target_idx : int
            Target class index.

        Returns
        -------
        metadata : dict
            Dictionary with ``'indices'`` (reference set indices) and
            ``'distances'`` (distances to query).
        distractors : list of np.ndarray
            List of distractor time series arrays.

        Raises
        ------
        ValueError
            If ``X_ref`` has fewer than 2 or more than 3 dimensions.
        """
        if self.X_ref.ndim not in (2, 3):
            raise ValueError(f"Unsupported X_ref shape {self.X_ref.shape}")

        yhat = self._ref_yhat
        target_label = self._idx_to_label(target_idx)

        # Primary: correctly-classified instances of target class
        mask = (self.y_ref == target_label) & (yhat == target_idx)
        if not np.any(mask):
            # Fallback A: any instance predicted as target class
            mask = yhat == target_idx
        if not np.any(mask):
            # Fallback B: any instance with ground-truth target class
            mask = self.y_ref == target_label
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

    def _select_best_distractor_result(
        self,
        x: np.ndarray,
        distractors: list[np.ndarray],
        target_idx: int,
    ) -> tuple[float, int, np.ndarray, list[int], float] | None:
        """Run greedy channel swap per distractor, return the best by loss.

        Parameters
        ----------
        x : np.ndarray
            Query time series.
        distractors : list of np.ndarray
            Distractor candidates.
        target_idx : int
            Target class index.

        Returns
        -------
        tuple or None
            ``(loss, distractor_index, cf, edits, target_prob)`` for the
            best distractor, or ``None`` if no distractors were provided.
        """
        best = None
        for i, distractor in enumerate(distractors):
            cf, edits, fc = self._swap_channels_greedily(x, distractor, target_idx)
            loss = self._compute_loss(fc, len(edits))
            item = (loss, i, cf, edits, fc)
            if best is None or loss < best[0]:
                best = item
        return best

    def _swap_channels_greedily(
        self, x: np.ndarray, distractor: np.ndarray, target_idx: int
    ) -> tuple[np.ndarray, list[int], float]:
        """Greedily swap channels from distractor to maximize target class probability.

        At each step, selects the channel whose substitution most increases
        the target class probability. Stops when probability reaches tau or
        no further improvement is possible.

        Parameters
        ----------
        x : np.ndarray
            Original time series of shape ``(T,)`` or ``(C, T)``.
        distractor : np.ndarray
            Distractor series to copy channels from, same shape as ``x``.
        target_idx : int
            Target class index to optimize probability for.

        Returns
        -------
        cf : np.ndarray
            Counterfactual series.
        edited_channels : list of int
            Channel indices swapped, in order.
        target_prob : float
            Final probability of the target class.
        """
        # Univariate short-circuit: single swap-or-not decision
        if x.ndim == 1:
            return self._try_univariate_swap(x, distractor, target_idx)

        C, _ = x.shape
        edited: list[int] = []
        cf = x.copy()
        current_prob = self._predict_target_prob(cf, target_idx)

        # Only consider channels where the distractor actually differs
        remaining = {c for c in range(C) if not np.array_equal(x[c, :], distractor[c, :])}

        while current_prob < self.tau and remaining:
            # Step A: Find the channel swap with the highest probability gain
            best_ch, best_prob, best_gain = self._find_best_channel_swap(
                cf, distractor, target_idx, remaining, current_prob
            )

            # Step B: If no gain, try fallback (single swap minimizing loss)
            if best_ch is None or best_gain <= _MIN_GAIN:
                if not edited:
                    fallback = self._try_fallback_swap(
                        cf, distractor, target_idx, remaining, current_prob
                    )
                    if fallback is not None:
                        ch, prob = fallback
                        cf[ch, :] = distractor[ch, :]
                        edited.append(ch)
                        remaining.remove(ch)
                        current_prob = prob
                        continue
                break

            # Step C: Commit the best channel swap
            cf[best_ch, :] = distractor[best_ch, :]
            edited.append(best_ch)
            remaining.remove(best_ch)
            current_prob = best_prob

        return cf, edited, current_prob

    def _try_univariate_swap(
        self, x: np.ndarray, distractor: np.ndarray, target_idx: int
    ) -> tuple[np.ndarray, list[int], float]:
        """Try swapping the entire univariate series, keep if loss improves.

        Parameters
        ----------
        x : np.ndarray
            Original univariate time series of shape ``(T,)``.
        distractor : np.ndarray
            Distractor series of shape ``(T,)``.
        target_idx : int
            Target class index.

        Returns
        -------
        cf : np.ndarray
            Counterfactual series of shape ``(T,)``.
        edited_channels : list of int
            ``[0]`` if swapped, ``[]`` if not.
        target_prob : float
            Final probability of the target class.
        """
        if np.array_equal(x, distractor):
            base_prob = self._predict_target_prob(x, target_idx)
            return x.copy(), [], base_prob

        base_prob = self._predict_target_prob(x, target_idx)
        cf_prob = self._predict_target_prob(distractor, target_idx)

        if self._compute_loss(cf_prob, 1) < self._compute_loss(base_prob, 0):
            return distractor.copy(), [0], cf_prob
        return x.copy(), [], base_prob

    def _find_best_channel_swap(
        self,
        cf: np.ndarray,
        distractor: np.ndarray,
        target_idx: int,
        remaining: set[int],
        current_prob: float,
    ) -> tuple[int | None, float, float]:
        """Evaluate all remaining channels, return the best swap.

        Parameters
        ----------
        cf : np.ndarray
            Current counterfactual of shape ``(C, T)``.
        distractor : np.ndarray
            Distractor series of shape ``(C, T)``.
        target_idx : int
            Target class index.
        remaining : set of int
            Channel indices still available for swapping.
        current_prob : float
            Current target class probability.

        Returns
        -------
        best_channel : int or None
            Channel with the highest gain, or ``None`` if set is empty.
        best_prob : float
            Probability after swapping ``best_channel``.
        best_gain : float
            Probability gain from the swap.
        """
        best_gain = -np.inf
        best_channel = None
        best_prob = current_prob

        for ch in remaining:
            candidate = cf.copy()
            candidate[ch, :] = distractor[ch, :]
            prob = self._predict_target_prob(candidate, target_idx)
            gain = prob - current_prob
            if gain > best_gain:
                best_gain = gain
                best_channel = ch
                best_prob = prob

        return best_channel, best_prob, best_gain

    def _try_fallback_swap(
        self,
        cf: np.ndarray,
        distractor: np.ndarray,
        target_idx: int,
        remaining: set[int],
        current_prob: float,
    ) -> tuple[int, float] | None:
        """When no channel gives positive gain, try the swap that minimizes loss.

        This fallback is only used when no channels have been edited yet, to
        ensure at least one swap is attempted.

        Parameters
        ----------
        cf : np.ndarray
            Current counterfactual of shape ``(C, T)``.
        distractor : np.ndarray
            Distractor series of shape ``(C, T)``.
        target_idx : int
            Target class index.
        remaining : set of int
            Channel indices still available.
        current_prob : float
            Current target class probability.

        Returns
        -------
        tuple or None
            ``(channel, prob)`` if a loss-improving swap exists, else ``None``.
        """
        best_loss = self._compute_loss(current_prob, 0)
        best_channel = None
        best_prob = current_prob

        for ch in remaining:
            candidate = cf.copy()
            candidate[ch, :] = distractor[ch, :]
            prob = self._predict_target_prob(candidate, target_idx)
            loss = self._compute_loss(prob, 1)
            if loss < best_loss:
                best_loss = loss
                best_prob = prob
                best_channel = ch

        if best_channel is not None:
            return best_channel, best_prob
        return None

    def _predict_target_prob(self, arr: np.ndarray, target_idx: int) -> float:
        """Return the model's probability for target_idx.

        Parameters
        ----------
        arr : np.ndarray
            Time series of shape ``(T,)`` or ``(C, T)``.
        target_idx : int
            Probability column index of the target class.

        Returns
        -------
        float
            Predicted probability for the target class.
        """
        return float(self.predict_proba(arr[None, ...])[0][target_idx])

    def _predict_class_idx(self, arr: np.ndarray) -> int:
        """Return the predicted class index for a single instance.

        Parameters
        ----------
        arr : np.ndarray
            Time series of shape ``(T,)`` or ``(C, T)``.

        Returns
        -------
        int
            Argmax index of the model's probability vector.
        """
        return int(np.argmax(self.predict_proba(arr[None, ...])[0]))

    def _resolve_target_class(
        self,
        base_probs: np.ndarray,
        base_idx: int,
        class_of_interest: int | None,
    ) -> int:
        """Determine the target class index for counterfactual generation.

        If ``class_of_interest`` is provided, it is converted to an internal
        index. Otherwise, the highest-probability class other than
        ``base_idx`` is selected.

        Parameters
        ----------
        base_probs : np.ndarray
            Probability vector for the query instance.
        base_idx : int
            Probability column index of the base (original) class.
        class_of_interest : int or None
            User-specified target class label, or ``None`` for automatic
            selection.

        Returns
        -------
        int
            Probability column index for the target class.
        """
        if class_of_interest is not None:
            return self._label_to_idx(class_of_interest)
        probs_sorted = np.argsort(-base_probs)
        return int(next(c for c in probs_sorted if c != base_idx))

    def _compute_loss(self, target_prob: float, n_edits: int) -> float:
        """Compute the counterfactual loss balancing validity and sparsity.

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

    def _no_distractor_fallback(
        self, x: np.ndarray, base_idx: int, target_idx: int
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        """Return the original instance unchanged when no distractors exist.

        Emits a warning and returns a metadata dict with
        ``validity=False`` and ``failure_reason='no_distractors'``.

        Parameters
        ----------
        x : np.ndarray
            Original time series of shape ``(T,)`` or ``(C, T)``.
        base_idx : int
            Probability column index of the base class.
        target_idx : int
            Probability column index of the target class.

        Returns
        -------
        cf : np.ndarray
            The original ``x`` (unchanged).
        cf_label : int
            Base class label.
        meta : dict
            Metadata dictionary flagged as invalid.

        Warns
        -----
        UserWarning
            When no distractors are found for the target class.
        """
        warnings.warn(
            f"COMTE: No distractors found for target class "
            f"{self._idx_to_label(target_idx)}. "
            f"This typically occurs when the classifier predicts all reference "
            f"samples as the same class (base class="
            f"{self._idx_to_label(base_idx)}). The original "
            f"instance is returned unchanged. Consider using a different dataset "
            f"or a classifier with more diverse predictions.",
            UserWarning,
            stacklevel=3,
        )
        return (
            x,
            self._idx_to_label(base_idx),
            {
                "method": "comte_greedy",
                "distance": self.distance,
                "class_of_interest": self._idx_to_label(target_idx),
                "validity": False,
                "failure_reason": "no_distractors",
                "note": "no distractors found; returning original unchanged",
                "tau": self.tau,
                "delta": self.delta,
                "lambda": self.lambda_reg,
            },
        )

    def _build_meta(
        self,
        target_idx: int,
        distractor_meta: dict[str, Any],
        distractor_i: int,
        edits: list[int],
        target_prob: float,
        loss: float,
        *,
        k_index: int | None = None,
    ) -> dict[str, Any]:
        """Build the metadata dictionary for an explanation result.

        Parameters
        ----------
        target_idx : int
            Probability column index of the target class.
        distractor_meta : dict
            Metadata from distractor selection, containing ``'indices'``
            and ``'distances'``.
        distractor_i : int
            Index into ``distractor_meta`` lists identifying the selected
            distractor.
        edits : list of int
            Channel indices swapped, in order.
        target_prob : float
            Final probability of the target class.
        loss : float
            Computed loss for this counterfactual.
        k_index : int or None
            Index of this result within an ``explain_k`` batch. Omitted
            from the dict when ``None``.

        Returns
        -------
        dict
            Metadata dictionary with keys ``method``, ``distance``,
            ``class_of_interest``, ``tau``, ``delta``, ``lambda``,
            ``lambda_reg``, ``distractor_index_in_ref``,
            ``distractor_distance``, ``edits_variables``,
            ``target_prob``, and ``loss`` (plus ``k_index`` when
            applicable).
        """
        meta: dict[str, Any] = {
            "method": "comte_greedy",
            "distance": self.distance,
            "class_of_interest": self._idx_to_label(target_idx),
            "tau": float(self.tau),
            "delta": int(self.delta),
            "lambda": float(self.lambda_reg),
            "lambda_reg": float(self.lambda_reg),
            "distractor_index_in_ref": (
                distractor_meta["indices"][distractor_i] if distractor_meta["indices"] else None
            ),
            "distractor_distance": (
                distractor_meta["distances"][distractor_i] if distractor_meta["distances"] else None
            ),
            "edits_variables": edits,
            "target_prob": float(target_prob),
            "loss": float(loss),
        }
        if k_index is not None:
            meta["k_index"] = k_index
        return meta
