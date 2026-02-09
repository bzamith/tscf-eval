Benchmark Module
================

The benchmark module provides a framework for systematic evaluation of
counterfactual explanation methods across multiple datasets and classifiers.

Features
--------

- **Parallel Execution**: Run multiple explainers in parallel using ``n_jobs`` parameter
- **Progress Tracking**: Built-in progress bars with tqdm
- **Evaluation Metrics**: 10 built-in metrics in six quality dimensions
- **Pareto Analysis**: Multi-criteria optimization with visualization
- **Result Export**: Save/load results in JSON format

BenchmarkRunner
---------------

.. autoclass:: tscf_eval.BenchmarkRunner
   :members:
   :undoc-members:
   :show-inheritance:

Instance Selection
------------------

The benchmark supports multiple strategies for selecting test instances via
the ``instance_selection`` parameter on ``BenchmarkRunner``.

SelectionStrategy
~~~~~~~~~~~~~~~~~

.. autodata:: tscf_eval.benchmark.SelectionStrategy

Two strategies are available:

- ``"random"`` (default): Uniform random sampling without replacement.
- ``"stratified_confidence"``: Instances are grouped into 4 quantile-based
  confidence bins (25th, 50th, 75th percentiles of max predicted probability),
  with an equal number of instances sampled from each bin. This ensures coverage
  of both high-confidence and uncertain instances.

Falls back to random selection with a warning if the model does not support
``predict_proba`` or if ``n_instances < 4``.

**Example**:

.. code-block:: python

   from tscf_eval.benchmark import BenchmarkRunner

   runner = BenchmarkRunner(
       ...,
       instance_selection="stratified_confidence",
   )

select_instances
~~~~~~~~~~~~~~~~

.. autofunction:: tscf_eval.benchmark.selection.select_instances

Configuration Classes
---------------------

ExplainerConfig
~~~~~~~~~~~~~~~

.. autoclass:: tscf_eval.ExplainerConfig
   :members:
   :undoc-members:
   :show-inheritance:

DatasetConfig
~~~~~~~~~~~~~

.. autoclass:: tscf_eval.DatasetConfig
   :members:
   :undoc-members:
   :show-inheritance:

ModelConfig
~~~~~~~~~~~

.. autoclass:: tscf_eval.ModelConfig
   :members:
   :undoc-members:
   :show-inheritance:

Result Classes
--------------

ExplainerResult
~~~~~~~~~~~~~~~

.. autoclass:: tscf_eval.ExplainerResult
   :members:
   :show-inheritance:

BenchmarkResults
~~~~~~~~~~~~~~~~

.. autoclass:: tscf_eval.BenchmarkResults
   :members:
   :exclude-members: results, config, timestamp
   :show-inheritance:


Pareto Analysis
---------------

ParetoAnalyzer
~~~~~~~~~~~~~~

Multi-criteria Pareto analysis with visualization support.

.. autoclass:: tscf_eval.ParetoAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

Visualization and Analysis Methods
"""""""""""""""""""""""""""""""""""

``ParetoAnalyzer`` provides these methods:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - ``pareto_front()``
     - Identify Pareto-optimal (non-dominated) solutions
   * - ``dominance_ranking()``
     - Full dominance ranking table with metric values
   * - ``plot_front()``
     - 2D scatter plot showing Pareto front between two objectives
   * - ``consistency()``
     - Cross-dataset Pareto consistency matrix
   * - ``plot_consistency_heatmap()``
     - Heatmap of Pareto consistency across datasets
   * - ``to_latex()``
     - Generate LaTeX table of the dominance ranking

**Examples**:

.. code-block:: python

   from tscf_eval.benchmark import ParetoAnalyzer
   import matplotlib.pyplot as plt

   # Create analyzer with metric names (directions are inferred)
   analyzer = ParetoAnalyzer(metrics=[
       "validity_soft", "proximity_dtw", "sparsity", "efficiency_time_s",
   ])

   # Identify Pareto-optimal methods
   pareto_methods = analyzer.pareto_front(results)
   print(f"Pareto-optimal: {pareto_methods}")

   # Full dominance ranking table
   ranking = analyzer.dominance_ranking(results)
   print(ranking)

   # 2D Pareto front plot
   ax = analyzer.plot_front(
       results,
       x_metric="proximity_dtw",
       y_metric="validity_soft",
       annotate=True,
   )
   plt.savefig("pareto_front.png")

   # Cross-dataset consistency analysis
   results_by_dataset = {
       ds: results.filter(datasets=[ds])
       for ds in results.datasets
   }
   consistency_df = analyzer.consistency(results_by_dataset)
   analyzer.plot_consistency_heatmap(consistency_df)
   plt.savefig("consistency_heatmap.png")

   # Export to LaTeX
   latex = analyzer.to_latex(results, caption="Pareto Ranking", label="tab:pareto")

Weighted Scalarization
----------------------

WeightedScalarizer
~~~~~~~~~~~~~~~~~~

Min-max normalized weighted composite scoring for ranking methods.

.. autoclass:: tscf_eval.WeightedScalarizer
   :members:
   :undoc-members:
   :show-inheritance:

**Example**:

.. code-block:: python

   from tscf_eval.benchmark import WeightedScalarizer

   # Equal-weight composite across metrics
   scalarizer = WeightedScalarizer(metrics=[
       "validity_soft", "proximity_dtw", "sparsity",
   ])
   scores = scalarizer.score(results)

   # Custom weights emphasizing validity
   scalarizer = WeightedScalarizer(
       metrics=["validity_soft", "proximity_dtw", "sparsity"],
       weights={"validity_soft": 3.0, "proximity_dtw": 1.0, "sparsity": 1.0},
   )

   # Sensitivity analysis
   sens_df = scalarizer.sensitivity(results, vary_metric="validity_soft", n_steps=11)
   scalarizer.plot_sensitivity(sens_df)

Statistical Testing
-------------------

friedman_test
~~~~~~~~~~~~~

.. autofunction:: tscf_eval.benchmark.friedman_test

**Example**:

.. code-block:: python

   from tscf_eval.benchmark import friedman_test

   fr = friedman_test(results, metric="validity_soft")
   print(f"Statistic: {fr.statistic:.3f}, p-value: {fr.p_value:.4f}")
   print(fr.rankings)

FriedmanResult
~~~~~~~~~~~~~~

A ``NamedTuple`` with three fields:

- ``statistic`` (*float*) -- Friedman chi-squared statistic.
- ``p_value`` (*float*) -- p-value of the test.
- ``rankings`` (*pd.DataFrame*) -- Mean ranks per explainer for each metric.

LaTeX Table Generation
----------------------

.. autofunction:: tscf_eval.benchmark.format_latex_table
