"""TSEvo counterfactual explainer implementation.

This module provides the ``TSEvo`` class, an implementation of the TSEvo
(Evolutionary Counterfactual Explanations for Time Series Classification)
algorithm using multi-objective evolutionary optimization.

The algorithm was originally developed by Jacqueline Höllig, Cedric Kulbach,
and Steffen Thoma at the Karlsruhe Institute of Technology (KIT).

Original implementation: https://github.com/JHoelli/TSEvo

Classes
-------
TSEvo
    TSEvo counterfactual generator using multi-objective evolutionary optimization.

Algorithm Overview
------------------
TSEvo generates counterfactuals through multi-objective evolutionary optimization:

1. Initialize a population of candidate solutions derived from the reference set
   (series predicted as the target class).
2. Evolve the population using NSGA-II with three mutation operators:

   - **authentic_opposing_information**: Replace temporal windows with segments
     from reference series (preserves realistic patterns).
   - **frequency_band_mapping**: Replace frequency bands using FFT transformation
     (captures frequency-domain characteristics).
   - **gaussian_perturbation**: Apply Gaussian noise based on reference set statistics.

3. Optimize three objectives simultaneously:

   - **Output distance**: Minimize distance to target class prediction.
   - **Input distance**: Minimize L1 distance from original series.
   - **Sparsity**: Minimize proportion of changed features.

4. Return the Pareto-optimal solutions from the Hall of Fame.

Examples
--------
>>> from tscf_eval.counterfactuals import TSEvo
>>> import numpy as np
>>>
>>> # Assume clf is a trained classifier
>>> tsevo = TSEvo(
...     model=clf,
...     data=(X_train, y_train),
...     transformer="authentic",  # or "frequency", "gaussian", "all"
...     n_generations=100,
...     population_size=50,
... )
>>>
>>> # Generate counterfactual for a test instance
>>> cf, cf_label, meta = tsevo.explain(x_test)
>>> print(f"Final objectives: {meta['objectives']}")
>>> print(f"Generations: {meta['n_generations']}")

References
----------
.. [tsevo1] Höllig, J., Kulbach, C., & Thoma, S. (2022).
       TSEvo: Evolutionary Counterfactual Explanations for Time Series
       Classification. In Proceedings of the 21st IEEE International Conference
       on Machine Learning and Applications (ICMLA 2022), pp. 29-36.
       DOI: 10.1109/ICMLA55696.2022.00013

.. [tsevo2] Hollig, J., Kulbach, C., & Thoma, S. (2023).
       TSInterpret: A Python Package for the Interpretability of Time Series
       Classification. Journal of Open Source Software, 8(85), 5220.
       https://doi.org/10.21105/joss.05220

Notes
-----
This implementation requires the ``deap`` package for evolutionary computation.
If not installed, a helpful error message will guide installation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .base import Counterfactual
from .utils import (
    ensure_batch_shape,
    soft_predict_proba_fn,
    strip_batch,
)

try:
    from deap import base, creator, tools

    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False


TransformerType = Literal["authentic", "frequency", "gaussian", "all"]


@dataclass
class TSEvo(Counterfactual):
    """TSEvo counterfactual generator using multi-objective evolutionary optimization.

    Implementation of the TSEvo algorithm by Höllig et al. (2022) [tsevo1]_.

    TSEvo uses NSGA-II (Non-dominated Sorting Genetic Algorithm II) to evolve
    counterfactual explanations that balance three objectives: changing the
    model's prediction (validity), minimizing perturbation (proximity), and
    keeping changes sparse (sparsity).

    The algorithm supports three mutation strategies that can be used
    individually or combined:

    - **authentic**: Replace windows with segments from reference series
    - **frequency**: Replace frequency bands via FFT transformation
    - **gaussian**: Apply Gaussian perturbation based on reference statistics

    Parameters
    ----------
    model : object
        A classifier with a probability estimator (``predict_proba`` or a
        compatible interface). The helper ``predict_proba_fn`` wraps model
        inference.
    data : tuple (X_ref, y_ref)
        Reference dataset used for mutation operations and population
        initialization. Series predicted as the target class are used.
    transformer : {'authentic', 'frequency', 'gaussian', 'all'}, default 'authentic'
        Mutation strategy to use:

        - 'authentic': Authentic opposing information (window replacement)
        - 'frequency': Frequency band mapping via FFT
        - 'gaussian': Gaussian perturbation from reference statistics
        - 'all': Randomly select among all strategies per individual
    n_generations : int, default 100
        Number of evolutionary generations.
    population_size : int, default 50
        Population size (μ in NSGA-II).
    crossover_prob : float, default 0.9
        Probability of applying crossover between individuals.
    mutation_prob : float, default 0.6
        Probability of applying mutation to an individual.
    window_sizes : tuple of int, default (5, 10, 20)
        Candidate window sizes for authentic mutation operator.
    random_state : int or None, default 0
        PRNG seed for reproducible evolution.
    verbose : int, default 0
        Verbosity level (0=silent, 1=progress, 2=detailed).

    Attributes
    ----------
    predict_proba : callable
        Wrapped probability prediction function.
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    X_ref : np.ndarray
        Reference dataset features.
    y_ref : np.ndarray
        Reference dataset labels.

    References
    ----------
    .. [tsevo1] Höllig, J., Kulbach, C., & Thoma, S. (2022).
           TSEvo: Evolutionary Counterfactual Explanations for Time Series
           Classification. ICMLA 2022. https://github.com/JHoelli/TSEvo
    """

    model: Any
    data: tuple[np.ndarray, np.ndarray]
    transformer: TransformerType = "authentic"
    n_generations: int = 100
    population_size: int = 50
    crossover_prob: float = 0.9
    mutation_prob: float = 0.6
    window_sizes: tuple[int, ...] = (5, 10, 20)
    random_state: int | None = 0
    verbose: int = 0

    # Internal state (not exposed as parameters)
    _deap_initialized: bool = field(default=False, init=False, repr=False)
    _ref_predictions: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not DEAP_AVAILABLE:
            raise ImportError(
                "TSEvo requires the 'deap' package for evolutionary computation. "
                "Install it with: pip install deap"
            )

        self.predict_proba = soft_predict_proba_fn(self.model)
        self.rng = np.random.default_rng(self.random_state)
        self.X_ref = np.asarray(self.data[0])
        self.y_ref = np.asarray(self.data[1]).ravel()

        # Validate parameters
        if self.transformer not in ("authentic", "frequency", "gaussian", "all"):
            raise ValueError(
                "transformer must be one of {'authentic', 'frequency', 'gaussian', 'all'}"
            )
        if self.n_generations < 1:
            raise ValueError("n_generations must be >= 1")
        if self.population_size < 4:
            raise ValueError("population_size must be >= 4 for NSGA-II")
        if self.population_size % 4 != 0:
            # Round up to nearest multiple of 4 for selTournamentDCD
            self.population_size = ((self.population_size + 3) // 4) * 4
        if not (0.0 <= self.crossover_prob <= 1.0):
            raise ValueError("crossover_prob must be in [0, 1]")
        if not (0.0 <= self.mutation_prob <= 1.0):
            raise ValueError("mutation_prob must be in [0, 1]")

    def explain(
        self,
        x: np.ndarray,
        y_pred: int | None = None,
        *,
        class_of_interest: int | None = None,
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        """Generate a counterfactual explanation using evolutionary optimization.

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
            Best counterfactual time series with the same shape as ``x``.
        cf_label : int
            Predicted class label for the counterfactual.
        meta : dict
            Metadata dictionary containing:

            - ``method``: Algorithm identifier (``'tsevo'``).
            - ``transformer``: Mutation strategy used.
            - ``class_of_interest``: Target class.
            - ``n_generations``: Number of generations evolved.
            - ``population_size``: Population size used.
            - ``objectives``: Final objective values (output_dist, input_dist, sparsity).
            - ``pareto_front_size``: Number of solutions in Pareto front.
            - ``validity``: Whether prediction changed (True/False).
        """
        xb, added = ensure_batch_shape(x)
        x1 = strip_batch(xb, added)

        # Determine base prediction and target class
        base_probs = self.predict_proba(xb)[0]
        base_label = int(np.argmax(base_probs)) if y_pred is None else int(y_pred)

        if class_of_interest is None:
            probs_sorted = np.argsort(-base_probs)
            class_of_interest = int(next(c for c in probs_sorted if c != base_label))

        # Get reference set: series predicted as target class
        reference_set = self._get_reference_set(class_of_interest)
        if len(reference_set) == 0:
            # Fallback: use all series not predicted as base_label
            reference_set = self._get_reference_set_fallback(base_label)

        if len(reference_set) == 0:
            return (
                x1,
                base_label,
                {
                    "method": "tsevo",
                    "note": "no reference set found; returning original",
                    "transformer": self.transformer,
                },
            )

        # Run evolutionary optimization
        best_individual, pareto_front, _logbook = self._evolve(
            x1, base_label, class_of_interest, reference_set
        )

        # Extract best counterfactual
        cf = np.array(best_individual).reshape(x1.shape)
        cf_probs = self.predict_proba(cf[None, ...])[0]
        cf_label = int(np.argmax(cf_probs))

        # Compute final objectives for metadata
        objectives = self._evaluate_objectives(cf, x1, base_label, class_of_interest, cf_probs)

        meta: dict[str, Any] = {
            "method": "tsevo",
            "transformer": self.transformer,
            "class_of_interest": class_of_interest,
            "n_generations": self.n_generations,
            "population_size": self.population_size,
            "objectives": {
                "output_distance": float(objectives[0]),
                "input_distance": float(objectives[1]),
                "sparsity": float(objectives[2]),
            },
            "pareto_front_size": len(pareto_front),
            "validity": cf_label != base_label,
        }

        return cf, cf_label, meta

    def _get_ref_predictions(self) -> np.ndarray:
        """Get cached predictions for reference set.

        Computes predictions once and caches them for reuse across
        multiple explain() calls.

        Returns
        -------
        np.ndarray
            Predicted class labels for all reference samples.
        """
        if self._ref_predictions is None:
            probs = self.predict_proba(self.X_ref)
            self._ref_predictions = np.argmax(probs, axis=1)
        return self._ref_predictions

    def _get_reference_set(self, target_class: int) -> np.ndarray:
        """Get reference series predicted as target class.

        Parameters
        ----------
        target_class : int
            Target class to filter reference set by.

        Returns
        -------
        np.ndarray
            Subset of reference data predicted as target class.
        """
        yhat = self._get_ref_predictions()
        mask = yhat == target_class
        result: np.ndarray = self.X_ref[mask]
        return result

    def _get_reference_set_fallback(self, base_label: int) -> np.ndarray:
        """Fallback: get any series not predicted as base class.

        Parameters
        ----------
        base_label : int
            Base class label to exclude.

        Returns
        -------
        np.ndarray
            Subset of reference data not predicted as base_label.
        """
        yhat = self._get_ref_predictions()
        mask = yhat != base_label
        result: np.ndarray = self.X_ref[mask]
        return result

    def _setup_deap(self, n_features: int):
        """Initialize DEAP creator and toolbox for this problem.

        Parameters
        ----------
        n_features : int
            Total number of features (flattened time series length).
        """
        # Clean up any previous DEAP definitions
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        # 3 objectives: output_distance, input_distance, sparsity (all minimize)
        # Use list as base type to avoid numpy array comparison issues in ParetoFront
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self._deap_initialized = True

    def _evolve(
        self,
        x: np.ndarray,
        base_label: int,
        target_class: int,
        reference_set: np.ndarray,
    ) -> tuple[np.ndarray, list, Any]:
        """Run the evolutionary optimization.

        Parameters
        ----------
        x : np.ndarray
            Original time series of shape ``(T,)`` or ``(C, T)``.
        base_label : int
            Original predicted class label.
        target_class : int
            Target class for counterfactual.
        reference_set : np.ndarray
            Reference series to sample from for mutations.

        Returns
        -------
        best_individual : np.ndarray
            Best counterfactual found.
        pareto_front : list
            All Pareto-optimal individuals.
        logbook : deap.tools.Logbook
            Evolution statistics.
        """
        n_features = x.size
        self._setup_deap(n_features)

        toolbox = base.Toolbox()

        # Individual initialization: sample from reference set
        def init_individual():
            idx = self.rng.integers(0, len(reference_set))
            # Use list for DEAP compatibility, convert to numpy for operations
            ind = creator.Individual(reference_set[idx].flatten().tolist())
            # Attach metadata for mutation operator
            ind.window_size = self.rng.choice(self.window_sizes)
            ind.transformer_type = self._select_transformer()
            return ind

        toolbox.register("individual", init_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Genetic operators
        toolbox.register(
            "mate",
            self._crossover,
            x_shape=x.shape,
        )
        toolbox.register(
            "mutate",
            self._mutate,
            x_original=x,
            reference_set=reference_set,
        )
        toolbox.register("select", tools.selNSGA2)

        # Initialize population
        pop = toolbox.population(n=self.population_size)

        # Evaluate initial population using batch prediction
        fitnesses = self._batch_evaluate(pop, x, base_label, target_class)
        for ind, fit in zip(pop, fitnesses, strict=True):
            ind.fitness.values = fit

        # Assign crowding distance for initial population
        pop = toolbox.select(pop, len(pop))

        # Hall of Fame for Pareto front
        hof = tools.ParetoFront()
        hof.update(pop)

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min, axis=0)
        stats.register("avg", np.mean, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals", *stats.fields]

        record = stats.compile(pop)
        logbook.record(gen=0, nevals=len(pop), **record)

        if self.verbose > 0:
            print(logbook.stream)

        # Main evolution loop (NSGA-II style)
        for gen in range(1, self.n_generations + 1):
            # Select offspring using tournament selection with crowding distance
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            # Apply crossover
            for i in range(0, len(offspring) - 1, 2):
                if self.rng.random() < self.crossover_prob:
                    toolbox.mate(offspring[i], offspring[i + 1])
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values

            # Apply mutation
            for mutant in offspring:
                if self.rng.random() < self.mutation_prob:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate individuals with invalid fitness using batch prediction
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                fitnesses = self._batch_evaluate(invalid_ind, x, base_label, target_class)
                for ind, fit in zip(invalid_ind, fitnesses, strict=True):
                    ind.fitness.values = fit

            # Select next generation using NSGA-II (computes crowding distance)
            pop = toolbox.select(pop + offspring, self.population_size)
            hof.update(pop)

            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            if self.verbose > 0:
                print(logbook.stream)

        # Select best individual from Pareto front
        # Prioritize validity (low output_distance), then proximity
        best = self._select_best_from_pareto(hof, x, base_label, target_class)

        return best, list(hof), logbook

    def _batch_evaluate(
        self,
        individuals: list,
        x_original: np.ndarray,
        base_label: int,
        target_class: int,
    ) -> list[tuple[float, float, float]]:
        """Evaluate multiple individuals using a single batch prediction.

        This is significantly faster than evaluating individuals one-by-one,
        especially for classifiers with high per-prediction overhead.

        Parameters
        ----------
        individuals : list
            List of DEAP individuals to evaluate.
        x_original : np.ndarray
            Original time series.
        base_label : int
            Original predicted class.
        target_class : int
            Target class for counterfactual.

        Returns
        -------
        list of tuple
            List of (output_distance, input_distance, sparsity) for each individual.
        """
        if not individuals:
            return []

        # Stack all individuals into a batch
        cf_batch = np.array([np.array(ind).reshape(x_original.shape) for ind in individuals])

        # Single batch prediction call
        probs_batch = self.predict_proba(cf_batch)

        # Compute objectives for each individual
        results = []
        for i, _ind in enumerate(individuals):
            cf = cf_batch[i]
            cf_probs = probs_batch[i]
            objectives = self._evaluate_objectives(
                cf, x_original, base_label, target_class, cf_probs
            )
            results.append(objectives)

        return results

    def _evaluate_objectives(
        self,
        cf: np.ndarray,
        x_original: np.ndarray,
        base_label: int,
        target_class: int,
        cf_probs: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute the three objective values.

        Parameters
        ----------
        cf : np.ndarray
            Candidate counterfactual.
        x_original : np.ndarray
            Original time series.
        base_label : int
            Original predicted class.
        target_class : int
            Target class for counterfactual.
        cf_probs : np.ndarray
            Model probability output for cf.

        Returns
        -------
        tuple of float
            (output_distance, input_distance, sparsity)
        """
        # Objective 1: Output distance
        # If still predicting base class, penalize heavily
        pred_label = int(np.argmax(cf_probs))
        if pred_label == base_label:
            output_dist = 1.0 - cf_probs[target_class]
        else:
            # Already flipped, minimize further
            output_dist = max(0.0, 0.5 - cf_probs[target_class])

        # Objective 2: Input distance (normalized L1)
        diff = np.abs(cf.flatten() - x_original.flatten())
        input_dist = float(np.mean(diff))

        # Objective 3: Sparsity (proportion of changed features)
        # Use tolerance-based comparison to avoid false positives from
        # floating-point rounding (e.g., after inverse FFT in frequency mutation)
        n_changed = np.count_nonzero(~np.isclose(cf.flatten(), x_original.flatten(), atol=1e-8))
        sparsity = n_changed / cf.size

        return (output_dist, input_dist, sparsity)

    def _select_transformer(self) -> str:
        """Select mutation transformer type based on settings.

        Returns
        -------
        str
            Selected transformer type.
        """
        if self.transformer == "all":
            choice: str = self.rng.choice(["authentic", "frequency", "gaussian"])
            return choice
        return self.transformer

    def _crossover(
        self,
        ind1: list,
        ind2: list,
        x_shape: tuple,
    ) -> tuple[list, list]:
        """Perform crossover between two individuals.

        Uses window-based uniform crossover respecting temporal structure.

        Parameters
        ----------
        ind1 : list
            First parent individual (flattened time series as list).
        ind2 : list
            Second parent individual (flattened time series as list).
        x_shape : tuple
            Original time series shape.

        Returns
        -------
        tuple
            Modified (ind1, ind2) after crossover.
        """
        window_size = getattr(ind1, "window_size", 10)
        T = x_shape[-1] if len(x_shape) > 1 else x_shape[0]

        # Create window indices
        n_windows = max(1, T // window_size)

        for w in range(n_windows):
            if self.rng.random() < 0.5:
                start = w * window_size
                end = min(start + window_size, T)

                if len(x_shape) == 1:
                    # Univariate: swap window segments
                    ind1[start:end], ind2[start:end] = (
                        list(ind2[start:end]),
                        list(ind1[start:end]),
                    )
                else:
                    # Multivariate: swap across all channels
                    C = x_shape[0]
                    for c in range(C):
                        c_start = c * T + start
                        c_end = c * T + end
                        ind1[c_start:c_end], ind2[c_start:c_end] = (
                            list(ind2[c_start:c_end]),
                            list(ind1[c_start:c_end]),
                        )

        return ind1, ind2

    def _mutate(
        self,
        individual: list,
        x_original: np.ndarray,
        reference_set: np.ndarray,
    ) -> tuple[list]:
        """Apply mutation based on transformer type.

        Parameters
        ----------
        individual : list
            Individual to mutate (modified in-place).
        x_original : np.ndarray
            Original time series for reference.
        reference_set : np.ndarray
            Reference set for authentic mutation.

        Returns
        -------
        tuple
            (individual,) as required by DEAP.
        """
        transformer_type = getattr(individual, "transformer_type", "authentic")

        if transformer_type == "authentic":
            self._mutate_authentic(individual, x_original, reference_set)
        elif transformer_type == "frequency":
            self._mutate_frequency(individual, x_original, reference_set)
        elif transformer_type == "gaussian":
            self._mutate_gaussian(individual, reference_set)

        # Occasionally change the transformer type
        if self.transformer == "all" and self.rng.random() < 0.1:
            individual.transformer_type = self._select_transformer()  # type: ignore[attr-defined]

        # Occasionally change window size
        if self.rng.random() < 0.1:
            individual.window_size = self.rng.choice(self.window_sizes)  # type: ignore[attr-defined]

        return (individual,)

    def _mutate_authentic(
        self,
        individual: list,
        x_original: np.ndarray,
        reference_set: np.ndarray,
    ) -> None:
        """Authentic opposing information mutation.

        Replaces a temporal window with the corresponding segment from a
        reference series. This preserves realistic temporal patterns.

        Parameters
        ----------
        individual : list
            Flattened individual to mutate in-place.
        x_original : np.ndarray
            Original time series (shape info).
        reference_set : np.ndarray
            Reference series to sample from.
        """
        # Select random reference series
        ref_idx = self.rng.integers(0, len(reference_set))
        ref_series = reference_set[ref_idx].flatten().tolist()

        window_size = getattr(individual, "window_size", 10)
        T = x_original.shape[-1] if x_original.ndim > 1 else x_original.shape[0]

        # Select random window position
        if window_size >= T:
            start = 0
            end = T
        else:
            start = self.rng.integers(0, T - window_size + 1)
            end = start + window_size

        if x_original.ndim == 1:
            # Univariate
            individual[start:end] = ref_series[start:end]
        else:
            # Multivariate: mutate random channel
            C = x_original.shape[0]
            channel = self.rng.integers(0, C)
            c_start = channel * T + start
            c_end = channel * T + end
            ref_c_start = channel * T + start
            ref_c_end = channel * T + end
            individual[c_start:c_end] = ref_series[ref_c_start:ref_c_end]

    def _mutate_frequency(
        self,
        individual: list,
        x_original: np.ndarray,
        reference_set: np.ndarray,
    ) -> None:
        """Frequency band mapping mutation.

        Replaces frequency bands using FFT transformation, capturing
        frequency-domain characteristics from reference series.

        Parameters
        ----------
        individual : list
            Flattened individual to mutate in-place.
        x_original : np.ndarray
            Original time series (shape info).
        reference_set : np.ndarray
            Reference series for frequency content.
        """
        # Select random reference series
        ref_idx = self.rng.integers(0, len(reference_set))
        ref_series = reference_set[ref_idx].flatten()

        T = x_original.shape[-1] if x_original.ndim > 1 else x_original.shape[0]

        if x_original.ndim == 1:
            # Univariate FFT mutation - convert to numpy for FFT
            result = self._fft_band_replace(np.array(individual[:T]), ref_series[:T], T)
            individual[:T] = result.tolist()
        else:
            # Multivariate: mutate random channel
            C = x_original.shape[0]
            channel = self.rng.integers(0, C)
            c_start = channel * T
            c_end = (channel + 1) * T
            result = self._fft_band_replace(
                np.array(individual[c_start:c_end]),
                ref_series[c_start:c_end],
                T,
            )
            individual[c_start:c_end] = result.tolist()

    def _fft_band_replace(
        self,
        signal: np.ndarray,
        reference: np.ndarray,
        T: int,
    ) -> np.ndarray:
        """Replace a frequency band from reference signal.

        Parameters
        ----------
        signal : np.ndarray
            Signal to modify.
        reference : np.ndarray
            Reference signal for frequency content.
        T : int
            Length of signals.

        Returns
        -------
        np.ndarray
            Modified signal.
        """
        # Compute FFT
        fft_signal = np.fft.rfft(signal)
        fft_reference = np.fft.rfft(reference)

        n_freqs = len(fft_signal)

        # Define frequency bands (quadratic scaling as in original paper)
        # Band widths increase quadratically with frequency
        n_bands = min(5, n_freqs)
        band_edges = np.linspace(0, n_freqs, n_bands + 1, dtype=int)

        # Select random band (skip DC component at index 0)
        if n_bands > 1:
            band_idx = self.rng.integers(1, n_bands)
            start = band_edges[band_idx]
            end = band_edges[band_idx + 1]

            # Replace band
            fft_signal[start:end] = fft_reference[start:end]

        # Inverse FFT
        result = np.fft.irfft(fft_signal, n=T)
        return result

    def _mutate_gaussian(
        self,
        individual: list,
        reference_set: np.ndarray,
    ) -> None:
        """Gaussian perturbation based on reference set statistics.

        Applies point-wise Gaussian noise scaled by the reference set's
        mean and standard deviation.

        Parameters
        ----------
        individual : list
            Flattened individual to mutate in-place.
        reference_set : np.ndarray
            Reference set for computing statistics.
        """
        # Compute reference statistics
        ref_flat = reference_set.reshape(len(reference_set), -1)
        means = ref_flat.mean(axis=0)
        stds = ref_flat.std(axis=0) + 1e-8  # Avoid division by zero

        # Per-gene mutation probability
        indpb = 0.1

        for i in range(len(individual)):
            if self.rng.random() < indpb:
                individual[i] = float(self.rng.normal(means[i % len(means)], stds[i % len(stds)]))

    def _select_best_from_pareto(
        self,
        pareto_front: list,
        x_original: np.ndarray,
        base_label: int,
        target_class: int,
    ) -> np.ndarray:
        """Select the best individual from the Pareto front.

        Prioritizes validity (prediction changed), then proximity.
        Uses batch prediction for efficiency.

        Parameters
        ----------
        pareto_front : list
            Pareto-optimal individuals.
        x_original : np.ndarray
            Original time series.
        base_label : int
            Original predicted class.
        target_class : int
            Target class for counterfactual.

        Returns
        -------
        np.ndarray
            Best individual from Pareto front.
        """
        if len(pareto_front) == 0:
            # Should not happen, but fallback
            return x_original.flatten()

        # Batch predict all Pareto front individuals at once
        cf_batch = np.array([np.array(ind).reshape(x_original.shape) for ind in pareto_front])
        probs_batch = self.predict_proba(cf_batch)
        pred_labels = np.argmax(probs_batch, axis=1)

        # Find valid counterfactuals (prediction changed)
        valid_individuals = []
        for i, ind in enumerate(pareto_front):
            if pred_labels[i] != base_label:
                cf = cf_batch[i]
                # Compute proximity for ranking
                proximity = float(np.mean(np.abs(cf.flatten() - x_original.flatten())))
                valid_individuals.append((ind, proximity, probs_batch[i, target_class]))

        if valid_individuals:
            # Sort by proximity (ascending), then by target probability (descending)
            valid_individuals.sort(key=lambda x: (x[1], -x[2]))
            return np.array(valid_individuals[0][0])

        # No valid counterfactual found: return the one closest to flipping
        best_ind = None
        best_output_dist = float("inf")
        for ind in pareto_front:
            output_dist = ind.fitness.values[0]
            if output_dist < best_output_dist:
                best_output_dist = output_dist
                best_ind = ind

        return np.array(best_ind) if best_ind is not None else x_original.flatten()
