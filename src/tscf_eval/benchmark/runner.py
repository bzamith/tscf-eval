"""Benchmark runner with parallel execution support.

This module provides the BenchmarkRunner class that orchestrates
the execution of counterfactual generation and evaluation across
multiple datasets, models, and explainers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any
import warnings

import numpy as np

from tscf_eval.evaluator import (
    Composition,
    Confidence,
    Contiguity,
    Diversity,
    Efficiency,
    Evaluator,
    Plausibility,
    Proximity,
    Robustness,
    Sparsity,
    Validity,
)

from .results import BenchmarkResults, ExplainerResult
from .selection import SelectionStrategy, select_instances

if TYPE_CHECKING:
    from .config import DatasetConfig, ExplainerConfig, ModelConfig

__all__ = ["BenchmarkRunner"]

# Check for optional dependencies
try:
    from joblib import Parallel, delayed

    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

try:
    from tqdm.auto import tqdm

    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


def _limit_numba_threads() -> None:
    """Limit numba threads to avoid contention in parallel execution."""
    try:
        from numba import set_num_threads  # noqa: PLC0415

        set_num_threads(1)
    except ImportError:
        pass


def _default_evaluator() -> Evaluator:
    """Create default evaluator with all metrics."""
    return Evaluator(
        [
            Validity(),
            Proximity(p=1),
            Proximity(p=2),
            Proximity(p=float("inf")),
            Sparsity(),
            Plausibility(method="lof"),
            Plausibility(method="if"),
            Plausibility(method="mp_ocsvm"),
            Diversity(),
            Contiguity(),
            Composition(),
            Confidence(),
            Robustness(),
            Efficiency(),
        ]
    )


def _run_single_task(
    dataset: DatasetConfig,
    model: ModelConfig,
    explainer_config: ExplainerConfig,
    evaluator: Evaluator,
    n_instances: int | None,
    instance_selection: SelectionStrategy,
    random_state: int | None,
    verbose: bool,
) -> ExplainerResult:
    """Run a single (dataset, model, explainer) combination.

    This function is designed to be called in parallel.
    """
    # Limit numba threads to avoid contention
    _limit_numba_threads()

    # Select test instances
    X_test, y_test = select_instances(
        dataset=dataset,
        model=model,
        n_instances=n_instances,
        strategy=instance_selection,
        random_state=random_state,
    )

    n_test = len(X_test)
    k = explainer_config.n_counterfactuals

    # Pre-allocate arrays
    X_cf: np.ndarray
    y_cf: np.ndarray
    if k == 1:
        X_cf = np.full_like(X_test, np.nan, dtype=float)
        y_cf = np.full(n_test, -1, dtype=int)
    else:
        X_cf = np.full((n_test, k, *X_test.shape[1:]), np.nan, dtype=float)
        y_cf = np.full((n_test, k), -1, dtype=int)

    success_mask = np.zeros(n_test, dtype=bool)
    generation_times: list[float] = []
    metadata: list[dict[str, Any]] = []

    # Create explainer
    explainer = explainer_config.create_explainer(
        model=model.model,
        data=(dataset.X_train, dataset.y_train),
    )

    # Pre-compute predictions for efficiency
    y_pred = model.predict(X_test)

    # Generate counterfactuals
    n_silent_failures = 0
    for i in range(n_test):
        x = X_test[i]
        start = time.perf_counter()

        try:
            if k == 1:
                cf, cf_label, meta = explainer.explain(x, y_pred=int(y_pred[i]))

                # Detect silent failure: CF is identical to x
                if np.allclose(cf, x, atol=1e-8):
                    n_silent_failures += 1
                    meta["_silent_failure"] = True
                    meta["_failure_reason"] = "counterfactual identical to original"
                    metadata.append(meta)
                    # Leave pre-allocated NaN in X_cf[i] and -1 in y_cf[i]
                else:
                    X_cf[i] = cf
                    y_cf[i] = cf_label
                    metadata.append(meta)
                    success_mask[i] = True
            else:
                cfs, cf_labels, metas = explainer.explain_k(x, k=k, y_pred=int(y_pred[i]))
                meta = {"k_metas": metas}

                # Detect silent failure: all k CFs are identical to x
                all_identical = all(np.allclose(cfs[j], x, atol=1e-8) for j in range(cfs.shape[0]))
                if all_identical:
                    n_silent_failures += 1
                    meta["_silent_failure"] = True
                    meta["_failure_reason"] = "all counterfactuals identical to original"
                    metadata.append(meta)
                    # Leave pre-allocated NaN in X_cf[i] and -1 in y_cf[i]
                else:
                    X_cf[i] = cfs
                    y_cf[i] = cf_labels
                    metadata.append(meta)
                    success_mask[i] = True

        except Exception as e:
            metadata.append({"error": str(e)})

        generation_times.append(time.perf_counter() - start)

    if n_silent_failures > 0:
        warnings.warn(
            f"{explainer_config.name}: {n_silent_failures}/{n_test} instances "
            f"returned counterfactuals identical to the original (silent failure). "
            f"These are excluded from evaluation metrics. This typically occurs "
            f"when the classifier is too confident for the explainer to flip the "
            f"prediction.",
            UserWarning,
            stacklevel=2,
        )

    # Evaluate
    if np.any(success_mask):
        X_orig = X_test[success_mask]
        X_cf_success = X_cf[success_mask]
        y_cf_success = y_cf[success_mask]
        times_success = [generation_times[i] for i in range(n_test) if success_mask[i]]

        # Build kwargs for evaluator
        eval_kwargs: dict[str, Any] = {
            "model": model.model,
            "X_train": dataset.X_train,
            "y": y_test[success_mask] if y_test is not None else None,
            "time_per_instance": times_success,
        }

        # For k > 1, use first CF for most metrics but pass all CFs for diversity
        if k > 1:
            X_cf_eval = X_cf_success[:, 0]
            y_cf_eval = y_cf_success[:, 0]
            # Pass all k counterfactuals for Diversity metric
            eval_kwargs["_X_cf_all"] = X_cf_success
        else:
            X_cf_eval = X_cf_success
            y_cf_eval = y_cf_success

        eval_kwargs["y_cf"] = y_cf_eval

        metrics = evaluator.evaluate(X_orig, X_cf_eval, **eval_kwargs)
    else:
        metrics = {"_note": "No successful counterfactuals generated"}

    return ExplainerResult(
        explainer_name=explainer_config.name,
        dataset_name=dataset.name,
        model_name=model.name,
        X_cf=X_cf,
        y_cf=y_cf,
        success_mask=success_mask,
        metrics=metrics,
        generation_times=generation_times,
        metadata=metadata,
    )


@dataclass
class BenchmarkRunner:
    """Orchestrates benchmark execution across datasets, models, and explainers.

    Supports parallel execution of independent tasks and provides progress
    tracking when tqdm is available.

    Parameters
    ----------
    datasets : list[DatasetConfig]
        Datasets to benchmark on.
    models : list[ModelConfig]
        Fitted classifier models to use.
    explainers : list[ExplainerConfig]
        Explainer configurations to evaluate.
    evaluator : Evaluator, optional
        Evaluator with metrics. Defaults to all available metrics.
    n_instances : int, optional
        Number of test instances per dataset. None uses all.
    instance_selection : {"random", "stratified_confidence"}, default "random"
        Strategy for selecting test instances.

        - ``"random"``: Uniform random sampling.
        - ``"stratified_confidence"``: Stratified sampling based on model
          prediction confidence. Divides instances into quantile-based
          confidence bins and samples from each bin, ensuring coverage
          of both high-confidence and low-confidence instances.
    n_jobs : int, default 1
        Number of parallel jobs. Use -1 for all CPUs.
        Requires joblib to be installed.
    verbose : bool, default True
        Show progress information.
    random_state : int, optional
        Random seed for reproducibility when subsampling.

    Examples
    --------
    >>> from tscf_eval.benchmark import (
    ...     BenchmarkRunner, DatasetConfig, ModelConfig, ExplainerConfig
    ... )
    >>> from tscf_eval import COMTE, NativeGuide
    >>>
    >>> datasets = [DatasetConfig("DS1", X_tr, y_tr, X_te, y_te)]
    >>> models = [ModelConfig("knn", fitted_knn)]
    >>> explainers = [
    ...     ExplainerConfig("comte", COMTE),
    ...     ExplainerConfig("ng", NativeGuide),
    ... ]
    >>>
    >>> runner = BenchmarkRunner(datasets, models, explainers, n_jobs=-1)
    >>> results = runner.run()
    >>> print(results.summary())
    """

    datasets: list[DatasetConfig]
    models: list[ModelConfig]
    explainers: list[ExplainerConfig]
    evaluator: Evaluator | None = None
    n_instances: int | None = None
    instance_selection: SelectionStrategy = "random"
    n_jobs: int = 1
    verbose: bool = True
    random_state: int | None = None

    # Internal
    _evaluator: Evaluator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration and set up evaluator."""
        if not self.datasets:
            raise ValueError("At least one dataset is required.")
        if not self.models:
            raise ValueError("At least one model is required.")
        if not self.explainers:
            raise ValueError("At least one explainer is required.")

        # Check for duplicate names
        self._check_unique_names("dataset", [d.name for d in self.datasets])
        self._check_unique_names("model", [m.name for m in self.models])
        self._check_unique_names("explainer", [e.name for e in self.explainers])

        # Set up evaluator
        self._evaluator = self.evaluator or _default_evaluator()

    def _check_unique_names(self, kind: str, names: list[str]) -> None:
        """Check that names are unique."""
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate {kind} names: {set(duplicates)}")

    def _build_tasks(
        self,
    ) -> list[tuple[DatasetConfig, ModelConfig, ExplainerConfig]]:
        """Build list of all (dataset, model, explainer) combinations."""
        tasks = []
        for dataset in self.datasets:
            for model in self.models:
                for explainer in self.explainers:
                    tasks.append((dataset, model, explainer))
        return tasks

    def run(self) -> BenchmarkResults:
        """Execute the benchmark.

        Returns
        -------
        BenchmarkResults
            Results container with all evaluation metrics.
        """
        tasks = self._build_tasks()
        n_tasks = len(tasks)

        if self.verbose:
            print(
                f"Running benchmark: {len(self.datasets)} datasets x "
                f"{len(self.models)} models x {len(self.explainers)} explainers "
                f"= {n_tasks} tasks"
            )

        if self.n_jobs != 1 and _JOBLIB_AVAILABLE:
            results_list = self._run_parallel(tasks)
        else:
            if self.n_jobs != 1 and not _JOBLIB_AVAILABLE:
                warnings.warn(
                    "joblib not installed, falling back to sequential execution.",
                    stacklevel=2,
                )
            results_list = self._run_sequential(tasks)

        # Collect results
        results = BenchmarkResults()
        for result in results_list:
            results.add(result)

        if self.verbose:
            print(f"Benchmark complete: {len(results)} results collected")

        return results

    def _run_sequential(
        self,
        tasks: list[tuple[DatasetConfig, ModelConfig, ExplainerConfig]],
    ) -> list[ExplainerResult]:
        """Run tasks sequentially."""
        results = []

        iterator: Any = tasks
        if self.verbose and _TQDM_AVAILABLE:
            iterator = tqdm(tasks, desc="Running benchmark", unit="task")

        for dataset, model, explainer in iterator:
            if self.verbose and _TQDM_AVAILABLE:
                iterator.set_postfix(
                    dataset=dataset.name,
                    model=model.name,
                    explainer=explainer.name,
                )

            result = _run_single_task(
                dataset=dataset,
                model=model,
                explainer_config=explainer,
                evaluator=self._evaluator,
                n_instances=self.n_instances,
                instance_selection=self.instance_selection,
                random_state=self.random_state,
                verbose=False,
            )
            results.append(result)

        return results

    def _run_parallel(
        self,
        tasks: list[tuple[DatasetConfig, ModelConfig, ExplainerConfig]],
    ) -> list[ExplainerResult]:
        """Run tasks in parallel using joblib."""
        if self.verbose:
            print(f"Running {len(tasks)} tasks in parallel (n_jobs={self.n_jobs})")

        results = Parallel(
            n_jobs=self.n_jobs,
            verbose=10 if self.verbose else 0,
        )(
            delayed(_run_single_task)(
                dataset=dataset,
                model=model,
                explainer_config=explainer,
                evaluator=self._evaluator,
                n_instances=self.n_instances,
                instance_selection=self.instance_selection,
                random_state=self.random_state,
                verbose=False,
            )
            for dataset, model, explainer in tasks
        )

        return list(results)
