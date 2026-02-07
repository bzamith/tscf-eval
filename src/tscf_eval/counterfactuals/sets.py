"""SETS counterfactual explainer implementation.

This module provides the ``SETS`` class, an implementation of the SETS
(Shapelet Explainer for Time Series) algorithm for generating counterfactual
explanations for time series classification.

The algorithm was originally developed by Omar Bahri, Soukaina Filali
Boubrahimi, and Shah Muhammad Hamdi at Utah State University and New Mexico
State University.

Original implementation: https://github.com/omarbahri/SETS

Classes
-------
SETS
    SETS counterfactual generator using class-specific shapelet manipulation.

Algorithm Overview
------------------
SETS generates counterfactuals through class-specific shapelet manipulation:

1. Extract discriminative shapelets using the Random Shapelet Transform.
2. Compute an occlusion threshold from the bottom percentile of scaled
   shapelet-to-series distances.
3. Assign each shapelet to its exclusive class (discard multi-class ones).
4. Build occurrence heat maps describing typical shapelet positions.
5. For a test instance, per dimension (ordered by information gain):
   a. **Phase A - Remove original-class shapelets**: replace detected
      original-class shapelet regions with min-max rescaled segments from the
      nearest unlike neighbor (NUN).
   b. **Phase B - Introduce target-class shapelets**: insert target-class
      shapelets at heat-map-guided positions, min-max rescaled.
6. Check the classifier prediction after each edit; stop early if the target
   class is achieved.
7. If single dimensions fail, try combinations of perturbed dimensions.

Examples
--------
>>> from tscf_eval.counterfactuals import SETS
>>> import numpy as np
>>>
>>> # Assume clf is a trained classifier with predict_proba
>>> sets = SETS(
...     model=clf,
...     data=(X_train, y_train),
...     n_shapelet_samples=5000,
...     max_shapelets=200,
... )
>>>
>>> # Generate counterfactual for a test instance
>>> cf, cf_label, meta = sets.explain(x_test)
>>> print(f"Valid: {meta['validity']}")
>>> print(f"Dimensions modified: {meta['dimensions_modified']}")

References
----------
.. [sets1] Bahri, O., Filali Boubrahimi, S., & Hamdi, S. M. (2022).
       Shapelet-Based Counterfactual Explanations for Multivariate Time Series.
       In Proceedings of the ACM SIGKDD Workshop on Mining and Learning from
       Time Series (KDD-MiLeTS 2022).
       DOI: 10.48550/arXiv.2208.10462

See Also
--------
tscf_eval.counterfactuals.COMTE : CoMTE algorithm implementation.
tscf_eval.counterfactuals.NativeGuide : NativeGuide algorithm implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, NamedTuple
import warnings

import numpy as np

from .base import Counterfactual
from .utils import (
    ensure_batch_shape,
    predict_proba_fn,
    strip_batch,
)
from .utils._nun import find_nearest_unlike_neighbor

# Optional: aeon for shapelet transform
try:
    from aeon.transformations.collection.shapelet_based import (
        RandomShapeletTransform,
    )

    _AEON_SHAPELET_AVAILABLE = True
except ImportError:  # pragma: no cover
    _AEON_SHAPELET_AVAILABLE = False


class ShapeletInfo(NamedTuple):
    """Parsed shapelet metadata with both normalized and raw arrays."""

    idx: int  # Index in the transformer's shapelet list
    info_gain: float
    length: int
    start_pos: int
    channel: int
    series_id: int
    class_value: Any  # Matches y_ref dtype
    z_norm_array: np.ndarray  # Z-normalized (from aeon)
    raw_array: np.ndarray  # Raw values (from X_ref)


@dataclass
class SETS(Counterfactual):
    """SETS counterfactual generator using class-specific shapelets.

    Implementation of the SETS algorithm by Bahri et al. (2022) [sets1]_.

    SETS leverages the inherent interpretability of shapelets to produce
    counterfactual explanations with contiguous, visually meaningful
    perturbations.  The preprocessing phase discovers class-exclusive
    shapelets and their typical occurrence positions; the generation phase
    removes original-class shapelets and introduces target-class shapelets
    to flip the classifier prediction.

    Parameters
    ----------
    model : object
        A classifier with ``predict_proba`` (or compatible interface).
    data : tuple (``X_ref``, ``y_ref``)
        Reference dataset for shapelet extraction and NUN lookup.
    n_shapelet_samples : int, default 10000
        Number of candidate shapelets to evaluate during extraction.
    max_shapelets : int or None, default None
        Maximum shapelets to retain.  ``None`` uses aeon's default
        (``min(10 * n_cases, 1000)``).
    min_shapelet_length : int, default 3
        Minimum shapelet length.
    max_shapelet_length : int or None, default None
        Maximum shapelet length.  ``None`` uses the full series length.
    time_limit_in_minutes : float, default 0.0
        Time budget for shapelet extraction (0 = use ``n_shapelet_samples``).
    threshold_percentile : float, default 10.0
        Bottom percentile of per-shapelet scaled distances used as the
        occlusion threshold.  Lower values are stricter.
    max_combination_dims : int, default 3
        Maximum number of dimensions to combine when single-dimension
        edits fail.  Caps the combinatorial search at C(D, k) for
        k ≤ ``max_combination_dims``.
    random_state : int or None, default 0
        PRNG seed for reproducibility.
    n_jobs : int, default 1
        Number of parallel jobs for shapelet extraction.

    Attributes
    ----------
    predict_proba : callable
        Wrapped probability prediction function.
    rng : numpy.random.Generator
        Random number generator.
    X_ref : np.ndarray
        Reference dataset features.
    y_ref : np.ndarray
        Reference dataset labels.

    References
    ----------
    .. [sets1] Bahri, O., Filali Boubrahimi, S., & Hamdi, S. M. (2022).
           Shapelet-Based Counterfactual Explanations for Multivariate Time
           Series. In Proceedings of the ACM SIGKDD Workshop on Mining and
           Learning from Time Series (KDD-MiLeTS 2022).
           https://github.com/omarbahri/SETS
    """

    model: Any
    data: tuple[np.ndarray, np.ndarray]

    # Shapelet transform parameters
    n_shapelet_samples: int = 10000
    max_shapelets: int | None = None
    min_shapelet_length: int = 3
    max_shapelet_length: int | None = None
    time_limit_in_minutes: float = 0.0

    # SETS algorithm parameters
    threshold_percentile: float = 10.0
    max_combination_dims: int = 3
    random_state: int | None = 0
    n_jobs: int = 1

    # Internal state (not user-facing)
    _shapelets: list[ShapeletInfo] = field(default_factory=list, init=False, repr=False)
    _class_shapelets: dict[Any, list[int]] = field(default_factory=dict, init=False, repr=False)
    _thresholds: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _heat_maps: dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _dim_ig: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _n_channels: int = field(default=1, init=False, repr=False)
    _series_length: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        """Initialise prediction wrapper, reference data, and shapelet pipeline.

        Validates parameters, fits the shapelet transform, computes the
        occlusion threshold, assigns class-exclusive shapelets, builds
        heat maps, and computes per-channel information gain.
        """
        if not _AEON_SHAPELET_AVAILABLE:
            raise ImportError(
                "SETS requires aeon's RandomShapeletTransform. Install aeon: pip install aeon"
            )
        self._validate_params()
        self._preprocess_data()

        # Run full preprocessing pipeline
        self._fit_shapelets()
        if self._shapelets:
            self._compute_thresholds()
            self._assign_classes()
            self._build_heat_maps()
            self._compute_dim_ig()

    def _validate_params(self) -> None:
        """Validate user-facing parameters.

        Raises
        ------
        ValueError
            If any parameter is out of its valid range.
        """
        if self.n_shapelet_samples < 1:
            raise ValueError("n_shapelet_samples must be >= 1")
        if self.min_shapelet_length < 1:
            raise ValueError("min_shapelet_length must be >= 1")
        if (
            self.max_shapelet_length is not None
            and self.max_shapelet_length < self.min_shapelet_length
        ):
            raise ValueError("max_shapelet_length must be >= min_shapelet_length")
        if not (0.0 < self.threshold_percentile <= 100.0):
            raise ValueError("threshold_percentile must be in (0, 100]")
        if self.max_combination_dims < 1:
            raise ValueError("max_combination_dims must be >= 1")

    def _preprocess_data(self) -> None:
        """Initialise internal data structures from constructor arguments.

        Wraps the model's predict function, normalises ``X_ref`` to 3-D,
        and pre-computes reference predictions and label mapping.

        Raises
        ------
        ValueError
            If ``X_ref`` has an unsupported number of dimensions.
        """
        self.predict_proba = predict_proba_fn(self.model)
        self.rng = np.random.default_rng(self.random_state)
        self.X_ref = np.asarray(self.data[0])
        self.y_ref = np.asarray(self.data[1]).ravel()

        # Normalise to 3-D (N, C, T) for aeon
        if self.X_ref.ndim == 2:  # (N, T) univariate
            self._X_ref_3d = self.X_ref[:, np.newaxis, :]
            self._n_channels = 1
            self._series_length = self.X_ref.shape[1]
        elif self.X_ref.ndim == 3:  # (N, C, T) multivariate
            self._X_ref_3d = self.X_ref
            self._n_channels = self.X_ref.shape[1]
            self._series_length = self.X_ref.shape[2]
        else:
            raise ValueError(f"Unsupported X_ref shape: {self.X_ref.shape}")

        self._init_label_mapping(self.model, self.y_ref)

    def _fit_shapelets(self) -> None:
        """Extract shapelets using aeon's RandomShapeletTransform.

        Fits the transform on the 3-D reference data, extracts the raw
        shapelet table (values, lengths, dimensions, info gain, class
        value), parses each entry into a :class:`ShapeletInfo` named
        tuple, and stores the distance matrix for later thresholding.
        """
        max_len = self.max_shapelet_length
        if max_len is None:
            max_len = self._series_length

        transformer = RandomShapeletTransform(
            n_shapelet_samples=self.n_shapelet_samples,
            max_shapelets=self.max_shapelets,
            min_shapelet_length=self.min_shapelet_length,
            max_shapelet_length=max_len,
            time_limit_in_minutes=self.time_limit_in_minutes,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self._dist_matrix: np.ndarray = transformer.fit_transform(self._X_ref_3d, self.y_ref)
        self._transformer = transformer

        parsed: list[ShapeletInfo] = []
        for i, shp in enumerate(transformer.shapelets):
            ig, length, start_pos, channel, series_id, class_val, z_norm = shp
            length = int(length)
            start_pos = int(start_pos)
            channel = int(channel)
            series_id = int(series_id)
            end = start_pos + length
            raw = self._X_ref_3d[series_id, channel, start_pos:end].copy()
            parsed.append(
                ShapeletInfo(
                    idx=i,
                    info_gain=float(ig),
                    length=length,
                    start_pos=start_pos,
                    channel=channel,
                    series_id=series_id,
                    class_value=class_val,
                    z_norm_array=np.asarray(z_norm),
                    raw_array=raw,
                )
            )

        self._shapelets = parsed
        if not parsed:
            warnings.warn(
                "SETS: No shapelets extracted. Counterfactual generation will "
                "return the original instance unchanged.",
                UserWarning,
                stacklevel=2,
            )

    def _compute_thresholds(self) -> None:
        """Compute per-shapelet occlusion thresholds from raw distances.

        For each shapelet, takes the ``threshold_percentile``-th percentile
        of the distance column from the ST distance matrix.  This matches
        the original SETS implementation which applies a per-shapelet
        threshold on the raw sliding-window distances rather than a single
        global threshold on normalised distances.
        """
        D = self._dist_matrix  # (n_cases, n_shapelets)
        self._thresholds = {}
        for s in self._shapelets:
            col = D[:, s.idx]
            self._thresholds[s.idx] = float(np.percentile(col, self.threshold_percentile))

    def _assign_classes(self) -> None:
        """Filter to class-exclusive shapelets and build class mapping.

        Keeps only shapelets that occur exclusively in instances of a
        single class.  Multi-class shapelets are discarded.  Builds the
        ``_class_shapelets`` mapping from class label to shapelet indices.
        """
        class_shapelets: dict[Any, list[int]] = {}
        surviving_indices: set[int] = set()

        for s in self._shapelets:
            # Find training instances where this shapelet "occurs"
            threshold = self._thresholds[s.idx]
            occ_mask = self._dist_matrix[:, s.idx] <= threshold
            if not np.any(occ_mask):
                continue  # No occurrences — discard

            occ_classes = set(self.y_ref[occ_mask].tolist())
            if len(occ_classes) != 1:
                continue  # Multi-class — discard

            cls = occ_classes.pop()
            class_shapelets.setdefault(cls, []).append(s.idx)
            surviving_indices.add(s.idx)

        self._class_shapelets = class_shapelets
        # Keep only surviving shapelets
        self._shapelets = [s for s in self._shapelets if s.idx in surviving_indices]

        if not self._shapelets:
            warnings.warn(
                "SETS: No class-exclusive shapelets found after filtering. "
                "Try increasing n_shapelet_samples or threshold_percentile.",
                UserWarning,
                stacklevel=2,
            )

    def _build_heat_maps(self) -> None:
        """Build normalised occurrence-position heat maps per shapelet.

        For each shapelet, aggregates the best-match start positions
        across same-class training instances to produce a probability
        distribution over time positions.  Used to guide Phase B
        shapelet insertion.
        """
        for s in self._shapelets:
            heat = np.zeros(self._series_length, dtype=np.float64)
            threshold = self._thresholds[s.idx]
            # Iterate only over training instances of this shapelet's class
            class_mask = self.y_ref == s.class_value
            n_occ = 0
            for i in np.where(class_mask)[0]:
                if self._dist_matrix[i, s.idx] > threshold:
                    continue
                # Find occurrence position(s) in this instance
                channel_data = self._X_ref_3d[i, s.channel]
                positions = self._find_occurrence_positions(
                    channel_data, s.z_norm_array, s.length, s.idx
                )
                for p in positions:
                    end = min(p + s.length, self._series_length)
                    heat[p:end] += 1.0
                    n_occ += 1
            if n_occ > 0:
                heat /= n_occ
            self._heat_maps[s.idx] = heat

    def _compute_dim_ig(self) -> None:
        """Compute maximum information gain per channel.

        For each channel, stores the highest information gain across all
        class-exclusive shapelets assigned to that channel. Channels with
        no shapelets receive an information gain of ``0.0``. The result
        is stored in ``_dim_ig`` and used to order dimensions during
        counterfactual generation.
        """
        self._dim_ig = {}
        for c in range(self._n_channels):
            igs = [s.info_gain for s in self._shapelets if s.channel == c]
            self._dim_ig[c] = max(igs) if igs else 0.0

    @staticmethod
    def _sliding_window_distances(
        series_channel: np.ndarray,
        z_norm_shapelet: np.ndarray,
        length: int,
    ) -> np.ndarray:
        """Z-normalised sliding-window squared Euclidean distance profile.

        Parameters
        ----------
        series_channel : np.ndarray
            1-D array of shape ``(T,)``.
        z_norm_shapelet : np.ndarray
            Z-normalised shapelet of shape ``(L,)``.
        length : int
            Shapelet length.

        Returns
        -------
        np.ndarray
            Distance at each valid position, shape ``(T - L + 1,)``.
        """
        T = len(series_channel)
        n_pos = T - length + 1
        dists = np.empty(n_pos, dtype=np.float64)
        for p in range(n_pos):
            w = series_channel[p : p + length].astype(np.float64)
            std = w.std()
            w_norm = np.zeros_like(w) if std < 1e-8 else (w - w.mean()) / std
            dists[p] = float(np.sum((z_norm_shapelet - w_norm) ** 2)) / length
        return dists

    def _find_occurrence_positions(
        self,
        series_channel: np.ndarray,
        z_norm_shapelet: np.ndarray,
        length: int,
        shapelet_idx: int,
    ) -> list[int]:
        """Return positions where a shapelet occurs below its occlusion threshold.

        Parameters
        ----------
        series_channel : np.ndarray
            1-D channel data of shape ``(T,)``.
        z_norm_shapelet : np.ndarray
            Z-normalised shapelet of shape ``(L,)``.
        length : int
            Shapelet length.
        shapelet_idx : int
            Index of the shapelet, used to look up its per-shapelet
            threshold in ``_thresholds``.

        Returns
        -------
        list[int]
            Start positions where the distance is at or below the
            shapelet's threshold.
        """
        if length > len(series_channel):
            return []
        dists = self._sliding_window_distances(series_channel, z_norm_shapelet, length)
        threshold = self._thresholds[shapelet_idx]
        positions: list[int] = np.where(dists <= threshold)[0].tolist()
        return positions

    @staticmethod
    def _rescale_segment(
        source: np.ndarray,
        target_min: float,
        target_max: float,
    ) -> np.ndarray:
        """Min-max rescale *source* values into ``[target_min, target_max]``.

        Parameters
        ----------
        source : np.ndarray
            1-D array of values to rescale.
        target_min : float
            Desired minimum of the output range.
        target_max : float
            Desired maximum of the output range.

        Returns
        -------
        np.ndarray
            Rescaled array with the same shape as *source*.
        """
        s_min = float(source.min())
        s_max = float(source.max())
        if s_max - s_min < 1e-12:
            return np.full_like(source, (target_min + target_max) / 2.0)
        rescaled = (target_max - target_min) * (source - s_min) / (s_max - s_min) + target_min
        return rescaled

    @staticmethod
    def _best_position_from_heatmap(heat_map: np.ndarray, length: int) -> int:
        """Find insertion position from the center of the heat map's active region.

        Matches the original SETS implementation: computes the center of
        the non-zero region of the heat map, then positions the shapelet
        so that its midpoint aligns with that center.

        Parameters
        ----------
        heat_map : np.ndarray
            1-D heat map of shape ``(T,)``.
        length : int
            Shapelet length to insert.

        Returns
        -------
        int
            Start position for the shapelet insertion.
        """
        T = len(heat_map)
        nonzero = np.argwhere(heat_map > 0)
        if len(nonzero) == 0:
            return 0
        first = int(nonzero[0, 0])
        last = int(nonzero[-1, 0])
        center = (last - first) // 2 + first

        start = center - length // 2
        end = center + (length - length // 2)

        # Boundary adjustments (matching original)
        if start < 0:
            end = end - start
            start = 0
        if end > T:
            start = start - (end - T)
            end = T
            start = max(start, 0)

        return start

    def _find_nun(
        self,
        x_internal: np.ndarray,
        target_class: Any,
    ) -> tuple[np.ndarray, int]:
        """Find nearest unlike neighbor from *target_class*.

        Performs simple kNN search on training instances whose
        ground-truth label matches the target class, matching the
        original SETS implementation.

        Parameters
        ----------
        x_internal : np.ndarray
            Instance in ``(C, T)`` shape.
        target_class
            Target class label (probability index).

        Returns
        -------
        nun : np.ndarray
            Nearest unlike neighbor in ``(C, T)`` shape.
        nun_idx : int
            Index in ``X_ref``.
        """
        target_label = self._idx_to_label(target_class)
        nuns, indices = find_nearest_unlike_neighbor(
            x_internal,
            self._X_ref_3d,
            self.y_ref,
            target_label,
            k=1,
        )
        if nuns:
            return nuns[0], indices[0]

        # Fallback: no instances of target class in training data
        warnings.warn(
            f"SETS: No instances of target class {target_label!r} found in "
            f"training labels. Using closest instance regardless of class.",
            UserWarning,
            stacklevel=2,
        )
        nuns, indices = find_nearest_unlike_neighbor(
            x_internal,
            self._X_ref_3d,
            np.zeros(len(self.y_ref)),
            0,
            fallback_all=True,
            k=1,
        )
        return nuns[0], indices[0]

    def _predict_class_idx(self, x_internal: np.ndarray) -> int:
        """Predict class for a ``(C, T)`` internal representation.

        Parameters
        ----------
        x_internal : np.ndarray
            Time series in internal ``(C, T)`` format.

        Returns
        -------
        int
            Probability column index of the predicted class.
        """
        # Convert back to original dimensionality for model
        if self._n_channels == 1 and self.X_ref.ndim == 2:
            x_model = x_internal[0][np.newaxis, :]  # (1, T)
        else:
            x_model = x_internal[np.newaxis, ...]  # (1, C, T)
        probs = self.predict_proba(x_model)[0]
        return int(np.argmax(probs))

    def _generate_cf(
        self,
        x_internal: np.ndarray,
        orig_class: Any,
        target_class: Any,
        nun: np.ndarray,
        dim_order: list[int],
    ) -> tuple[np.ndarray, bool, dict[str, Any]]:
        """Core generation loop with Phase A/B and dimension combinations.

        For each dimension (ordered by information gain), applies Phase A
        (remove original-class shapelets by replacing with NUN segments)
        and Phase B (insert target-class shapelets at heat-map-guided
        positions).  After each dimension, if the single-dimension edit
        failed, immediately tries combinations of all dimensions processed
        so far.  This matches the original SETS implementation structure.

        Parameters
        ----------
        x_internal : np.ndarray
            Input series in ``(C, T)`` shape.
        orig_class : int
            Probability index of the original predicted class.
        target_class : int
            Probability index of the target class.
        nun : np.ndarray
            Nearest unlike neighbor in ``(C, T)`` shape.
        dim_order : list[int]
            Dimension indices ordered by information gain (descending).

        Returns
        -------
        cf : np.ndarray
            Counterfactual in ``(C, T)`` shape.
        success : bool
            Whether the target class was achieved.
        info : dict
            Edit information including dimensions modified, phase A/B
            edit counts.
        """
        per_dim_cfs: dict[int, np.ndarray] = {}
        total_a = 0
        total_b = 0

        # Map probability indices to original class labels for shapelet lookup
        orig_label = self._classes[orig_class] if orig_class < len(self._classes) else orig_class
        target_label = (
            self._classes[target_class] if target_class < len(self._classes) else target_class
        )

        # Collect shapelets per class, per channel for quick lookup
        orig_shps_by_ch: dict[int, list[ShapeletInfo]] = {}
        target_shps_by_ch: dict[int, list[ShapeletInfo]] = {}
        for s in self._shapelets:
            if s.class_value == orig_label and s.channel in dim_order:
                orig_shps_by_ch.setdefault(s.channel, []).append(s)
            if s.class_value == target_label and s.channel in dim_order:
                target_shps_by_ch.setdefault(s.channel, []).append(s)

        # Sort shapelets within each channel by info gain (descending)
        for ch_list in orig_shps_by_ch.values():
            ch_list.sort(key=lambda s: s.info_gain, reverse=True)
        for ch_list in target_shps_by_ch.values():
            ch_list.sort(key=lambda s: s.info_gain, reverse=True)

        for d in dim_order:
            working = x_internal.copy()

            # Phase A: Remove original-class shapelets in dimension d
            for s in orig_shps_by_ch.get(d, []):
                positions = self._find_occurrence_positions(
                    working[d], s.z_norm_array, s.length, s.idx
                )
                for p in positions:
                    end = min(p + s.length, self._series_length)
                    seg_len = end - p
                    local_min = float(working[d, p:end].min())
                    local_max = float(working[d, p:end].max())
                    nun_seg = nun[d, p:end]
                    working[d, p:end] = self._rescale_segment(
                        nun_seg[:seg_len], local_min, local_max
                    )
                    total_a += 1

                    if self._predict_class_idx(working) == target_class:
                        return (
                            working,
                            True,
                            {
                                "dimensions_modified": [d],
                                "phase_a_edits": total_a,
                                "phase_b_edits": total_b,
                            },
                        )

            # Phase B: Introduce target-class shapelets in dimension d
            for s in target_shps_by_ch.get(d, []):
                heat_map = self._heat_maps.get(s.idx)
                if heat_map is None:
                    continue
                insert_pos = self._best_position_from_heatmap(heat_map, s.length)
                end = min(insert_pos + s.length, self._series_length)
                seg_len = end - insert_pos
                local_min = float(working[d, insert_pos:end].min())
                local_max = float(working[d, insert_pos:end].max())
                working[d, insert_pos:end] = self._rescale_segment(
                    s.raw_array[:seg_len], local_min, local_max
                )
                total_b += 1

                if self._predict_class_idx(working) == target_class:
                    return (
                        working,
                        True,
                        {
                            "dimensions_modified": [d],
                            "phase_a_edits": total_a,
                            "phase_b_edits": total_b,
                        },
                    )

            per_dim_cfs[d] = working

            # After single-dim edit, check if prediction changed
            if self._predict_class_idx(working) == target_class:
                return (
                    working,
                    True,
                    {
                        "dimensions_modified": [d],
                        "phase_a_edits": total_a,
                        "phase_b_edits": total_b,
                    },
                )

            # Try combinations of all dimensions processed so far
            available_dims = [dd for dd in dim_order if dd in per_dim_cfs]
            max_k = min(self.max_combination_dims, len(available_dims))
            for n_dims in range(2, max_k + 1):
                for combo in combinations(available_dims, n_dims):
                    combined = x_internal.copy()
                    for cd in combo:
                        combined[cd] = per_dim_cfs[cd][cd]
                    if self._predict_class_idx(combined) == target_class:
                        return (
                            combined,
                            True,
                            {
                                "dimensions_modified": list(combo),
                                "phase_a_edits": total_a,
                                "phase_b_edits": total_b,
                            },
                        )

        # Failed — return the best single-dimension attempt
        best_cf = (
            per_dim_cfs.get(dim_order[0], x_internal.copy()) if per_dim_cfs else x_internal.copy()
        )
        return (
            best_cf,
            False,
            {
                "dimensions_modified": [],
                "phase_a_edits": total_a,
                "phase_b_edits": total_b,
            },
        )

    def explain(
        self,
        x: np.ndarray,
        y_pred: int | None = None,
        *,
        class_of_interest: int | None = None,
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        """Generate a counterfactual explanation using SETS.

        Parameters
        ----------
        x : np.ndarray
            Input time series of shape ``(T,)`` for univariate or ``(C, T)``
            for multivariate data.
        y_pred : int, optional
            Base predicted class for ``x``.  If ``None``, computed via model.
        class_of_interest : int, optional
            Target class.  If ``None``, uses the highest-probability
            alternative to ``y_pred``.

        Returns
        -------
        cf : np.ndarray
            Counterfactual time series with the same shape as ``x``.
        cf_label : int
            Predicted class label for the counterfactual.
        meta : dict
            Metadata dictionary containing:

            - ``method``: ``'sets'``
            - ``class_of_interest``: Target class.
            - ``nun_index_in_ref``: Index of the NUN used.
            - ``dimensions_modified``: Channels edited.
            - ``phase_a_edits``: Number of Phase A replacements.
            - ``phase_b_edits``: Number of Phase B insertions.
            - ``n_class_shapelets``: Total surviving class-exclusive shapelets.
            - ``validity``: Whether the target class was achieved.
            - ``failure_reason``: ``None`` if successful, description otherwise.
        """
        xb, added = ensure_batch_shape(x)
        x1 = strip_batch(xb, added)

        # Convert to (C, T) internally
        was_univariate = x1.ndim == 1
        x_internal = x1[np.newaxis, :] if was_univariate else x1.copy()

        # Determine base prediction and target class (all in index space)
        base_probs = self.predict_proba(xb)[0]
        base_idx = int(np.argmax(base_probs)) if y_pred is None else self._label_to_idx(y_pred)

        if class_of_interest is not None:
            target_idx = self._label_to_idx(class_of_interest)
        else:
            probs_sorted = np.argsort(-base_probs)
            target_idx = int(next(c for c in probs_sorted if c != base_idx))

        failure_reason: str | None = None

        # Edge case: no class-exclusive shapelets
        if not self._shapelets:
            failure_reason = "no_class_exclusive_shapelets"
            cf = x1.copy()
            cf_label = self._idx_to_label(base_idx)
            return (
                cf,
                cf_label,
                self._build_meta(
                    self._idx_to_label(target_idx),
                    None,
                    [],
                    0,
                    0,
                    False,
                    failure_reason,
                ),
            )

        # Sort dimensions by max info gain (descending)
        dim_order = sorted(
            range(self._n_channels),
            key=lambda c: self._dim_ig.get(c, 0.0),
            reverse=True,
        )

        # Find nearest unlike neighbor
        nun, nun_idx = self._find_nun(x_internal, target_idx)

        # Generate counterfactual
        cf_internal, success, edit_info = self._generate_cf(
            x_internal, base_idx, target_idx, nun, dim_order
        )

        if not success:
            failure_reason = "no_valid_cf_found"

        # Convert back to original shape
        cf = cf_internal[0] if was_univariate else cf_internal
        cf_idx = self._predict_class_idx(cf_internal)
        cf_label = self._idx_to_label(cf_idx)

        return (
            cf,
            cf_label,
            self._build_meta(
                self._idx_to_label(target_idx),
                nun_idx,
                edit_info.get("dimensions_modified", []),
                edit_info.get("phase_a_edits", 0),
                edit_info.get("phase_b_edits", 0),
                success,
                failure_reason,
            ),
        )

    def explain_k(
        self,
        x: np.ndarray,
        k: int = 5,
        y_pred: int | None = None,
        *,
        class_of_interest: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Generate k diverse counterfactuals using different NUNs.

        SETS supports diverse counterfactual generation by using different
        nearest unlike neighbors as the replacement source for Phase A.
        Each counterfactual is generated with a different NUN, producing
        structurally diverse explanations.

        Parameters
        ----------
        x : np.ndarray
            Input time series of shape ``(T,)`` or ``(C, T)``.
        k : int, default 5
            Number of counterfactuals to generate.
        y_pred : int, optional
            Precomputed predicted label for ``x``.
        class_of_interest : int, optional
            Target class for counterfactuals.

        Returns
        -------
        cfs : np.ndarray
            Array of k counterfactuals with shape ``(k, ...)``.
        cf_labels : np.ndarray
            Array of k predicted labels.
        metas : list[dict]
            List of k metadata dictionaries.
        """
        xb, added = ensure_batch_shape(x)
        x1 = strip_batch(xb, added)

        was_univariate = x1.ndim == 1
        x_internal = x1[np.newaxis, :] if was_univariate else x1.copy()

        # Determine base prediction and target class (all in index space)
        base_probs = self.predict_proba(xb)[0]
        base_idx = int(np.argmax(base_probs)) if y_pred is None else self._label_to_idx(y_pred)

        if class_of_interest is not None:
            target_idx = self._label_to_idx(class_of_interest)
        else:
            probs_sorted = np.argsort(-base_probs)
            target_idx = int(next(c for c in probs_sorted if c != base_idx))

        target_label = self._idx_to_label(target_idx)

        # Edge case: no class-exclusive shapelets
        if not self._shapelets:
            base_label = self._idx_to_label(base_idx)
            cfs_out = np.array([x1.copy() for _ in range(k)])
            labels_out = np.array([base_label] * k)
            metas_out = [
                {
                    **self._build_meta(
                        target_label, None, [], 0, 0, False, "no_class_exclusive_shapelets"
                    ),
                    "k_index": i,
                }
                for i in range(k)
            ]
            return cfs_out, labels_out, metas_out

        dim_order = sorted(
            range(self._n_channels),
            key=lambda c: self._dim_ig.get(c, 0.0),
            reverse=True,
        )

        # Find k NUNs for diversity
        nuns, nun_indices = self._find_k_nuns(x_internal, target_idx, k)

        cfs: list[np.ndarray] = []
        cf_labels: list[Any] = []
        metas: list[dict[str, Any]] = []

        for i, (nun, nun_idx) in enumerate(zip(nuns, nun_indices, strict=True)):
            cf_internal, success, edit_info = self._generate_cf(
                x_internal, base_idx, target_idx, nun, dim_order
            )
            failure_reason = None if success else "no_valid_cf_found"
            cf = cf_internal[0] if was_univariate else cf_internal
            cf_idx = self._predict_class_idx(cf_internal)
            cf_label = self._idx_to_label(cf_idx)

            meta = self._build_meta(
                target_label,
                nun_idx,
                edit_info.get("dimensions_modified", []),
                edit_info.get("phase_a_edits", 0),
                edit_info.get("phase_b_edits", 0),
                success,
                failure_reason,
            )
            meta["k_index"] = i
            cfs.append(cf)
            cf_labels.append(cf_label)
            metas.append(meta)

        # Pad with best result if fewer NUNs than k
        while len(cfs) < k:
            best_idx = 0
            cf = cfs[best_idx].copy()
            cf_label = cf_labels[best_idx]
            new_meta = metas[best_idx].copy()
            new_meta["k_index"] = len(cfs)
            new_meta["note"] = "duplicated from nearest neighbor"
            cfs.append(cf)
            cf_labels.append(cf_label)
            metas.append(new_meta)

        return np.array(cfs), np.array(cf_labels), metas

    def _find_k_nuns(
        self,
        x_internal: np.ndarray,
        target_class: Any,
        k: int,
    ) -> tuple[list[np.ndarray], list[int]]:
        """Find k nearest unlike neighbors from *target_class*.

        Parameters
        ----------
        x_internal : np.ndarray
            Instance in ``(C, T)`` shape.
        target_class
            Target class label (probability index).
        k : int
            Number of NUNs to retrieve.

        Returns
        -------
        nuns : list[np.ndarray]
            Up to k NUNs in ``(C, T)`` shape.
        nun_indices : list[int]
            Indices in ``X_ref``.
        """
        target_label = self._idx_to_label(target_class)
        nuns, indices = find_nearest_unlike_neighbor(
            x_internal,
            self._X_ref_3d,
            self.y_ref,
            target_label,
            k=k,
        )
        if nuns:
            return nuns, indices

        # Fallback: no instances of target class in training data
        return find_nearest_unlike_neighbor(
            x_internal,
            self._X_ref_3d,
            np.zeros(len(self.y_ref)),
            0,
            fallback_all=True,
            k=k,
        )

    def _build_meta(
        self,
        class_of_interest: int,
        nun_idx: int | None,
        dims_modified: list[int],
        phase_a: int,
        phase_b: int,
        validity: bool,
        failure_reason: str | None,
    ) -> dict[str, Any]:
        """Build the metadata dictionary returned by ``explain``.

        Parameters
        ----------
        class_of_interest : int
            Target class label.
        nun_idx : int or None
            Index of the NUN in ``X_ref``, or ``None`` if unavailable.
        dims_modified : list[int]
            Channel indices that were edited.
        phase_a : int
            Number of Phase A (removal) edits applied.
        phase_b : int
            Number of Phase B (insertion) edits applied.
        validity : bool
            Whether the target class was achieved.
        failure_reason : str or None
            Description of the failure, or ``None`` on success.

        Returns
        -------
        dict[str, Any]
            Metadata dictionary suitable for the ``explain`` return value.
        """
        return {
            "method": "sets",
            "class_of_interest": class_of_interest,
            "nun_index_in_ref": nun_idx,
            "dimensions_modified": dims_modified,
            "phase_a_edits": phase_a,
            "phase_b_edits": phase_b,
            "n_class_shapelets": len(self._shapelets),
            "validity": validity,
            "failure_reason": failure_reason,
        }
