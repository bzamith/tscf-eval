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

1. Initialize a population of candidate solutions from the original instance.
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

from dataclasses import dataclass
from typing import Any, Literal
import warnings

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
    data : tuple (``X_ref``, ``y_ref``)
        Reference dataset used for mutation operations. Series predicted as
        the target class are used during evolution.
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

    def __post_init__(self):
        """Initialise probability wrapper, RNG, reference data, and label mapping.

        Validates all hyperparameters and ensures the ``deap`` package is
        available for evolutionary computation. Rounds ``population_size``
        up to the nearest multiple of four as required by NSGA-II
        tournament selection.
        """
        if not DEAP_AVAILABLE:
            raise ImportError(
                "TSEvo requires the 'deap' package for evolutionary computation. "
                "Install it with: pip install deap"
            )

        self.predict_proba = soft_predict_proba_fn(self.model)
        self.rng = np.random.default_rng(self.random_state)
        self.X_ref = np.asarray(self.data[0])
        self.y_ref = np.asarray(self.data[1]).ravel()

        self._init_label_mapping(self.model, self.y_ref)

        # Pre-compute reference set predictions (matches NativeGuide/CoMTE)
        self._ref_probs = self.predict_proba(self.X_ref)
        self._ref_yhat = np.argmax(self._ref_probs, axis=1)

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

        base_probs = self.predict_proba(xb)[0]
        base_idx = int(np.argmax(base_probs)) if y_pred is None else self._label_to_idx(y_pred)
        target_idx = self._resolve_target_class(base_probs, base_idx, class_of_interest)

        # Step 1: Build reference set from series predicted as target class
        reference_set = self._build_reference_set(target_idx, fallback_exclude=base_idx)
        if len(reference_set) == 0:
            return self._no_reference_set_fallback(x1, base_idx, target_idx)

        # Step 2: Evolve counterfactuals via NSGA-II
        best, pareto_front = self._run_evolution(x1, base_idx, target_idx, reference_set)

        # Step 3: Assemble result
        cf = np.array(best).reshape(x1.shape)
        cf_probs = self.predict_proba(cf[None, ...])[0]
        cf_idx = int(np.argmax(cf_probs))
        objectives = self._evaluate_objectives(cf, x1, base_idx, target_idx, cf_probs)

        cf_label = self._idx_to_label(cf_idx)
        meta = self._build_meta(target_idx, objectives, len(pareto_front), cf_idx != base_idx)
        return cf, cf_label, meta

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

    def _build_reference_set(self, target_idx: int, fallback_exclude: int) -> np.ndarray:
        """Get reference series predicted as target class, with fallback.

        Primary: series predicted as ``target_idx``.
        Fallback: any series NOT predicted as ``fallback_exclude``.

        Parameters
        ----------
        target_idx : int
            Target class index to filter reference set by.
        fallback_exclude : int
            Base class index to exclude in the fallback.

        Returns
        -------
        np.ndarray
            Subset of reference data (may be empty if no candidates exist).
        """
        mask = self._ref_yhat == target_idx
        if np.any(mask):
            result: np.ndarray = self.X_ref[mask]
            return result
        mask = self._ref_yhat != fallback_exclude
        result = self.X_ref[mask]
        return result

    def _no_reference_set_fallback(
        self, x: np.ndarray, base_idx: int, target_idx: int
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        """Return the original instance unchanged when no reference set exists.

        Emits a warning and returns a metadata dict with
        ``validity=False`` and ``failure_reason='no_reference_set'``.

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
            When no reference series are found for the target class.
        """
        warnings.warn(
            f"TSEvo: No reference series found for target class "
            f"{self._idx_to_label(target_idx)}. "
            f"The classifier predicts no reference samples as the target class "
            f"(base class={self._idx_to_label(base_idx)}). The original instance "
            f"is returned unchanged.",
            UserWarning,
            stacklevel=3,
        )
        return (
            x,
            self._idx_to_label(base_idx),
            {
                "method": "tsevo",
                "transformer": self.transformer,
                "class_of_interest": self._idx_to_label(target_idx),
                "validity": False,
                "failure_reason": "no_reference_set",
                "note": "no reference set found; returning original unchanged",
            },
        )

    def _run_evolution(
        self,
        x: np.ndarray,
        base_idx: int,
        target_idx: int,
        reference_set: np.ndarray,
    ) -> tuple[np.ndarray, list]:
        """Run NSGA-II evolutionary optimization.

        Parameters
        ----------
        x : np.ndarray
            Original time series of shape ``(T,)`` or ``(C, T)``.
        base_idx : int
            Original predicted class index.
        target_idx : int
            Target class index for counterfactual.
        reference_set : np.ndarray
            Reference series used for mutation operators.

        Returns
        -------
        best_individual : np.ndarray
            Best counterfactual found.
        pareto_front : list
            All Pareto-optimal individuals.
        """
        # Step A: Register DEAP types and build toolbox
        self._register_deap_types(n_features=x.size)
        toolbox = self._build_toolbox(x, reference_set)

        # Step B: Create and evaluate initial population
        pop = toolbox.population(n=self.population_size)
        fitnesses = self._evaluate_population(pop, x, base_idx, target_idx)
        for ind, fit in zip(pop, fitnesses, strict=True):
            ind.fitness.values = fit
        pop = toolbox.select(pop, len(pop))

        hof = tools.ParetoFront()
        hof.update(pop)

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

        # Step C: Main evolution loop (NSGA-II)
        for gen in range(1, self.n_generations + 1):
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

            # Evaluate individuals with invalidated fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                fitnesses = self._evaluate_population(invalid_ind, x, base_idx, target_idx)
                for ind, fit in zip(invalid_ind, fitnesses, strict=True):
                    ind.fitness.values = fit

            # Select next generation (NSGA-II computes crowding distance)
            pop = toolbox.select(pop + offspring, self.population_size)
            hof.update(pop)

            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            if self.verbose > 0:
                print(logbook.stream)

        # Step D: Pick best from Pareto front
        best = self._pick_best_from_pareto(hof, x, base_idx, target_idx)
        return best, list(hof)

    def _register_deap_types(self, n_features: int) -> None:
        """Register DEAP creator types (FitnessMin, Individual) for this problem.

        Parameters
        ----------
        n_features : int
            Total number of features (flattened time series length).
        """
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        # 3 objectives: output_distance, input_distance, sparsity (all minimize)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)

    def _build_toolbox(
        self,
        x: np.ndarray,
        reference_set: np.ndarray,
    ) -> base.Toolbox:
        """Build a DEAP toolbox with individual factory and genetic operators.

        Parameters
        ----------
        x : np.ndarray
            Original time series used for population initialisation and
            crossover/mutation shape info.
        reference_set : np.ndarray
            Reference series used for mutation operators.

        Returns
        -------
        base.Toolbox
            Fully configured DEAP toolbox.
        """
        toolbox = base.Toolbox()

        x_flat = x.flatten().tolist()

        def init_individual():
            """Create one DEAP individual from the original instance.

            Following the original TSEvo paper, each individual in the
            initial population is a copy of the query instance ``x``.
            Diversity is introduced through mutation operators, not
            through population initialisation.

            Returns
            -------
            creator.Individual
                Flattened original series wrapped as a DEAP individual
                with ``window_size`` and ``transformer_type`` attributes.
            """
            ind = creator.Individual(x_flat)
            ind.window_size = self.rng.choice(self.window_sizes)
            ind.transformer_type = self._choose_mutation_strategy()
            return ind

        toolbox.register("individual", init_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", self._crossover_windows, x_shape=x.shape)
        toolbox.register("mutate", self._apply_mutation, x_original=x, reference_set=reference_set)
        toolbox.register("select", tools.selNSGA2)
        return toolbox

    def _evaluate_population(
        self,
        individuals: list,
        x_original: np.ndarray,
        base_idx: int,
        target_idx: int,
    ) -> list[tuple[float, float, float]]:
        """Evaluate a batch of individuals using a single batch prediction.

        Parameters
        ----------
        individuals : list
            List of DEAP individuals to evaluate.
        x_original : np.ndarray
            Original time series.
        base_idx : int
            Original predicted class index.
        target_idx : int
            Target class index for counterfactual.

        Returns
        -------
        list of tuple
            List of (output_distance, input_distance, sparsity) per individual.
        """
        if not individuals:
            return []

        cf_batch = np.array([np.array(ind).reshape(x_original.shape) for ind in individuals])
        probs_batch = self.predict_proba(cf_batch)

        results = []
        for i in range(len(individuals)):
            objectives = self._evaluate_objectives(
                cf_batch[i], x_original, base_idx, target_idx, probs_batch[i]
            )
            results.append(objectives)

        return results

    def _evaluate_objectives(
        self,
        cf: np.ndarray,
        x_original: np.ndarray,
        base_idx: int,
        target_idx: int,
        cf_probs: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute the three fitness objectives.

        Parameters
        ----------
        cf : np.ndarray
            Candidate counterfactual.
        x_original : np.ndarray
            Original time series.
        base_idx : int
            Original predicted class index.
        target_idx : int
            Target class index for counterfactual.
        cf_probs : np.ndarray
            Model probability output for cf.

        Returns
        -------
        tuple of float
            (output_distance, input_distance, sparsity)
        """
        # Objective 1: Output distance — penalize if still predicting base class
        pred_idx = int(np.argmax(cf_probs))
        if pred_idx == base_idx:
            output_dist = 1.0 - cf_probs[target_idx]
        else:
            output_dist = max(0.0, 0.5 - cf_probs[target_idx])

        # Objective 2: Input distance (normalized L1)
        diff = np.abs(cf.flatten() - x_original.flatten())
        input_dist = float(np.mean(diff))

        # Objective 3: Sparsity (proportion of changed features)
        # Tolerance-based to avoid false positives from floating-point rounding
        n_changed = np.count_nonzero(~np.isclose(cf.flatten(), x_original.flatten(), atol=1e-8))
        sparsity = n_changed / cf.size

        return (output_dist, input_dist, sparsity)

    def _pick_best_from_pareto(
        self,
        pareto_front: list,
        x_original: np.ndarray,
        base_idx: int,
        target_idx: int,
    ) -> np.ndarray:
        """Pick the best individual from the Pareto front.

        Prioritizes valid counterfactuals (prediction changed), ranked by
        proximity then target probability. Falls back to the individual
        closest to flipping if none are valid.

        Parameters
        ----------
        pareto_front : list
            Pareto-optimal individuals.
        x_original : np.ndarray
            Original time series.
        base_idx : int
            Original predicted class index.
        target_idx : int
            Target class index for counterfactual.

        Returns
        -------
        np.ndarray
            Best individual from Pareto front.
        """
        if len(pareto_front) == 0:
            return x_original.flatten()

        # Batch predict all Pareto front individuals
        cf_batch = np.array([np.array(ind).reshape(x_original.shape) for ind in pareto_front])
        probs_batch = self.predict_proba(cf_batch)
        pred_labels = np.argmax(probs_batch, axis=1)

        # Prefer valid counterfactuals, ranked by proximity then target prob
        valid_individuals = []
        for i, ind in enumerate(pareto_front):
            if pred_labels[i] != base_idx:
                proximity = float(np.mean(np.abs(cf_batch[i].flatten() - x_original.flatten())))
                valid_individuals.append((ind, proximity, probs_batch[i, target_idx]))

        if valid_individuals:
            valid_individuals.sort(key=lambda v: (v[1], -v[2]))
            return np.array(valid_individuals[0][0])

        # No valid counterfactual: return the one closest to flipping
        best_ind = None
        best_output_dist = float("inf")
        for ind in pareto_front:
            output_dist = ind.fitness.values[0]
            if output_dist < best_output_dist:
                best_output_dist = output_dist
                best_ind = ind

        return np.array(best_ind) if best_ind is not None else x_original.flatten()

    def _crossover_windows(
        self,
        ind1: list,
        ind2: list,
        x_shape: tuple,
    ) -> tuple[list, list]:
        """Window-based uniform crossover respecting temporal structure.

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

        n_windows = max(1, T // window_size)

        for w in range(n_windows):
            if self.rng.random() < 0.5:
                start = w * window_size
                end = min(start + window_size, T)

                if len(x_shape) == 1:
                    ind1[start:end], ind2[start:end] = (
                        list(ind2[start:end]),
                        list(ind1[start:end]),
                    )
                else:
                    C = x_shape[0]
                    for c in range(C):
                        c_start = c * T + start
                        c_end = c * T + end
                        ind1[c_start:c_end], ind2[c_start:c_end] = (
                            list(ind2[c_start:c_end]),
                            list(ind1[c_start:c_end]),
                        )

        return ind1, ind2

    def _apply_mutation(
        self,
        individual: list,
        x_original: np.ndarray,
        reference_set: np.ndarray,
    ) -> tuple[list]:
        """Dispatch mutation based on the individual's assigned strategy.

        Parameters
        ----------
        individual : list
            Individual to mutate (modified in-place).
        x_original : np.ndarray
            Original time series for shape reference.
        reference_set : np.ndarray
            Reference set for mutation operators.

        Returns
        -------
        tuple
            (individual,) as required by DEAP.
        """
        strategy = getattr(individual, "transformer_type", "authentic")

        if strategy == "authentic":
            self._mutate_by_window_replacement(individual, x_original, reference_set)
        elif strategy == "frequency":
            self._mutate_by_frequency_band(individual, x_original, reference_set)
        elif strategy == "gaussian":
            self._mutate_by_gaussian_noise(individual, reference_set)

        # Occasionally change strategy or window size
        if self.transformer == "all" and self.rng.random() < 0.1:
            individual.transformer_type = self._choose_mutation_strategy()  # type: ignore[attr-defined]
        if self.rng.random() < 0.1:
            individual.window_size = self.rng.choice(self.window_sizes)  # type: ignore[attr-defined]

        return (individual,)

    def _choose_mutation_strategy(self) -> str:
        """Choose which mutation strategy to assign to an individual.

        Returns
        -------
        str
            Selected mutation strategy name.
        """
        if self.transformer == "all":
            choice: str = self.rng.choice(["authentic", "frequency", "gaussian"])
            return choice
        return self.transformer

    def _mutate_by_window_replacement(
        self,
        individual: list,
        x_original: np.ndarray,
        reference_set: np.ndarray,
    ) -> None:
        """Replace a temporal window with the corresponding segment from a reference series.

        Parameters
        ----------
        individual : list
            Flattened individual to mutate in-place.
        x_original : np.ndarray
            Original time series (shape info).
        reference_set : np.ndarray
            Reference series to sample from.
        """
        ref_idx = self.rng.integers(0, len(reference_set))
        ref_series = reference_set[ref_idx].flatten().tolist()

        window_size = getattr(individual, "window_size", 10)
        T = x_original.shape[-1] if x_original.ndim > 1 else x_original.shape[0]

        if window_size >= T:
            start = 0
            end = T
        else:
            start = self.rng.integers(0, T - window_size + 1)
            end = start + window_size

        if x_original.ndim == 1:
            individual[start:end] = ref_series[start:end]
        else:
            C = x_original.shape[0]
            channel = self.rng.integers(0, C)
            c_start = channel * T + start
            c_end = channel * T + end
            ref_c_start = channel * T + start
            ref_c_end = channel * T + end
            individual[c_start:c_end] = ref_series[ref_c_start:ref_c_end]

    def _mutate_by_frequency_band(
        self,
        individual: list,
        x_original: np.ndarray,
        reference_set: np.ndarray,
    ) -> None:
        """Replace frequency bands via FFT from a reference series.

        Parameters
        ----------
        individual : list
            Flattened individual to mutate in-place.
        x_original : np.ndarray
            Original time series (shape info).
        reference_set : np.ndarray
            Reference series for frequency content.
        """
        ref_idx = self.rng.integers(0, len(reference_set))
        ref_series = reference_set[ref_idx].flatten()

        T = x_original.shape[-1] if x_original.ndim > 1 else x_original.shape[0]

        if x_original.ndim == 1:
            result = self._replace_fft_band(np.array(individual[:T]), ref_series[:T], T)
            individual[:T] = result.tolist()
        else:
            C = x_original.shape[0]
            channel = self.rng.integers(0, C)
            c_start = channel * T
            c_end = (channel + 1) * T
            result = self._replace_fft_band(
                np.array(individual[c_start:c_end]),
                ref_series[c_start:c_end],
                T,
            )
            individual[c_start:c_end] = result.tolist()

    def _replace_fft_band(
        self,
        signal: np.ndarray,
        reference: np.ndarray,
        T: int,
    ) -> np.ndarray:
        """Replace a random frequency band in signal with content from reference.

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
        fft_signal = np.fft.rfft(signal)
        fft_reference = np.fft.rfft(reference)

        n_freqs = len(fft_signal)
        n_bands = min(5, n_freqs)
        band_edges = np.linspace(0, n_freqs, n_bands + 1, dtype=int)

        # Select random band (skip DC component at index 0)
        if n_bands > 1:
            band_idx = self.rng.integers(1, n_bands)
            start = band_edges[band_idx]
            end = band_edges[band_idx + 1]
            fft_signal[start:end] = fft_reference[start:end]

        result = np.fft.irfft(fft_signal, n=T)
        return result

    def _mutate_by_gaussian_noise(
        self,
        individual: list,
        reference_set: np.ndarray,
    ) -> None:
        """Apply point-wise Gaussian perturbation scaled by reference statistics.

        Parameters
        ----------
        individual : list
            Flattened individual to mutate in-place.
        reference_set : np.ndarray
            Reference set for computing mean and std.
        """
        ref_flat = reference_set.reshape(len(reference_set), -1)
        means = ref_flat.mean(axis=0)
        stds = ref_flat.std(axis=0) + 1e-8

        indpb = 0.1
        for i in range(len(individual)):
            if self.rng.random() < indpb:
                individual[i] = float(self.rng.normal(means[i % len(means)], stds[i % len(stds)]))

    def _build_meta(
        self,
        target_idx: int,
        objectives: tuple[float, float, float],
        pareto_front_size: int,
        validity: bool,
    ) -> dict[str, Any]:
        """Build the metadata dictionary for an explanation result.

        Parameters
        ----------
        target_idx : int
            Probability column index of the target class.
        objectives : tuple of float
            Final objective values ``(output_distance, input_distance,
            sparsity)`` for the selected counterfactual.
        pareto_front_size : int
            Number of individuals in the final Pareto front.
        validity : bool
            Whether the counterfactual's predicted class differs from
            the original.

        Returns
        -------
        dict[str, Any]
            Metadata dictionary with keys ``method``, ``transformer``,
            ``class_of_interest``, ``n_generations``, ``population_size``,
            ``objectives``, ``pareto_front_size``, and ``validity``.
        """
        return {
            "method": "tsevo",
            "transformer": self.transformer,
            "class_of_interest": self._idx_to_label(target_idx),
            "n_generations": self.n_generations,
            "population_size": self.population_size,
            "objectives": {
                "output_distance": float(objectives[0]),
                "input_distance": float(objectives[1]),
                "sparsity": float(objectives[2]),
            },
            "pareto_front_size": pareto_front_size,
            "validity": validity,
        }
