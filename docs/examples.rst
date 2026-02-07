Examples
========

This guide provides examples for common use cases with tscf-eval.

.. contents:: Table of Contents
   :local:
   :depth: 2

Evaluating Your Own Counterfactuals
-----------------------------------

If you have implemented your own counterfactual generation method and want to
evaluate it using tscf-eval's metrics, follow these examples.

Basic Evaluation with Labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest case is when you have pre-computed counterfactuals and their labels.
Metrics like Plausibility and Diversity require ``X_train``.

.. code-block:: python

   import numpy as np
   from tscf_eval.evaluator import (
       Evaluator, Validity, Proximity, Sparsity,
       Plausibility, Contiguity, Efficiency
   )

   # Your original instances and counterfactuals
   X = np.random.randn(100, 50)  # 100 instances, 50 time points
   X_cf = X + np.random.randn(100, 50) * 0.1  # Your generated counterfactuals

   # Original and counterfactual labels
   y = np.array([0] * 50 + [1] * 50)  # Original labels
   y_cf = np.array([1] * 50 + [0] * 50)  # Counterfactual labels (flipped)

   # Create evaluator with desired metrics
   evaluator = Evaluator([
       Validity(),
      Proximity(p=2, distance="lp"),
       Sparsity(),
       Plausibility(method="lof"),
       Contiguity(),
   ])

   # Evaluate
   results = evaluator.evaluate(X, X_cf, y=y, y_cf=y_cf, X_train=X)

   # Print results
   print(f"Validity: {results['validity']:.2%}")
   print(f"Proximity (L2): {results['proximity_l2']:.4f}")
   print(f"Sparsity: {results['sparsity']:.2%}")
   print(f"Plausibility (LOF): {results['plausibility_lof']:.4f}")
   print(f"Contiguity: {results['contiguity']:.4f}")

Evaluation with a Trained Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For metrics that depend on model predictions (Validity, Controllability, Confidence),
provide a fitted classifier:

.. code-block:: python

   from sklearn.neighbors import KNeighborsClassifier
   from tscf_eval.evaluator import (
       Evaluator, Validity, Proximity, Sparsity,
       Controllability, Confidence, Efficiency
   )

   # Train your classifier
   clf = KNeighborsClassifier(n_neighbors=5)
   clf.fit(X_train, y_train)

   # Your counterfactuals
   X_test = ...  # Original test instances
   X_cf = ...    # Your generated counterfactuals

   # Create evaluator with model-dependent metrics
   evaluator = Evaluator([
       Validity(),
      Proximity(p=2, distance="lp"),
       Sparsity(),
       Controllability(),
       Confidence(),
   ])

   # Evaluate with model
   results = evaluator.evaluate(
       X_test, X_cf,
       model=clf,
       X_train=X_train,
   )

   # Validity is computed from model predictions
   print(f"Validity: {results['validity']:.2%}")
   print(f"Controllability: {results['controllability']:.4f}")
   print(f"Confidence: {results['confidence']}")

Evaluation with Timing Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you tracked generation time per instance, include it for the Efficiency metric:

.. code-block:: python

   import time
   from tscf_eval.evaluator import Evaluator, Validity, Proximity, Efficiency

   # Generate counterfactuals and track time
   X_cf_list = []
   times = []
   for x in X_test:
       start = time.perf_counter()
       cf = your_cf_method(x)  # Your counterfactual generation
       elapsed = time.perf_counter() - start
       X_cf_list.append(cf)
       times.append(elapsed)

   X_cf = np.array(X_cf_list)

   # Evaluator with Efficiency metric
   evaluator = Evaluator([
       Validity(),
      Proximity(p=2, distance="lp"),
       Efficiency(),
   ])

   # Include time_per_instance
   results = evaluator.evaluate(
       X_test, X_cf,
       model=clf,
       time_per_instance=times,
   )

   print(f"Mean time per instance: {results['efficiency_time_s']:.4f}s")

Full Evaluation (All Metric Classes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For full evaluation, use all metric classes:

.. code-block:: python

   from tscf_eval.evaluator import (
       Evaluator, Validity, Proximity, Sparsity,
       Plausibility, Diversity, Controllability, Confidence,
       Composition, Contiguity, Robustness, Efficiency
   )

   # Complete evaluator with all metrics
   evaluator = Evaluator([
       # Core metrics
       Validity(),
      Proximity(p=1, distance="lp"),         # L1 distance
      Proximity(p=2, distance="lp"),         # L2 distance
      Proximity(p=float("inf"), distance="lp"),  # L-infinity distance
       Sparsity(),

       # Distribution metrics
       Plausibility(method="lof"),  # Local Outlier Factor
       Plausibility(method="if"),   # Isolation Forest
       Diversity(),

       # Model-dependent metrics
       Controllability(),
       Confidence(),

       # Structure metrics
       Composition(),
       Contiguity(),

       # Stability and performance
       Robustness(),
       Efficiency(),
   ])

   # Full evaluation
   results = evaluator.evaluate(
       X_test, X_cf,
       model=clf,
       X_train=X_train,
       y=y_test,
       y_cf=y_cf,
       time_per_instance=times,
   )

   # Print all results
   for key, value in results.items():
       if not key.startswith("_"):
           print(f"{key}: {value}")


Implementing a Custom Counterfactual Method
-------------------------------------------

To integrate your counterfactual method with tscf-eval's benchmarking framework,
implement the ``Counterfactual`` interface.

Basic Implementation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from tscf_eval.counterfactuals import Counterfactual

   class MyCounterfactual(Counterfactual):
       """My custom counterfactual explainer.

       Parameters
       ----------
       model : object
           Fitted classifier with predict/predict_proba methods.
       data : tuple
           Tuple of (X_train, y_train) for reference data.
       perturbation_scale : float
           Scale of random perturbations.
       """

       def __init__(
           self,
           model,
           data: tuple[np.ndarray, np.ndarray],
           perturbation_scale: float = 0.1,
       ):
           self.model = model
           self.X_train, self.y_train = data
           self.perturbation_scale = perturbation_scale

       def explain(
           self,
           x: np.ndarray,
           y_pred: int | None = None,
       ) -> tuple[np.ndarray, int, dict]:
           """Generate a counterfactual for instance x.

           Parameters
           ----------
           x : np.ndarray
               Input time series of shape (T,) or (C, T).
           y_pred : int, optional
               Precomputed prediction for x. If None, computed internally.

           Returns
           -------
           cf : np.ndarray
               Counterfactual time series with same shape as x.
           cf_label : int
               Predicted class for the counterfactual.
           meta : dict
               Metadata about the generation process.
           """
           # Ensure x is properly shaped
           x = np.asarray(x).squeeze()

           # Get original prediction if not provided
           if y_pred is None:
               y_pred = int(self.model.predict(x.reshape(1, -1))[0])

           # Your counterfactual generation logic here
           # Example: iterative perturbation until prediction changes
           cf = x.copy()
           max_iterations = 100
           for i in range(max_iterations):
               # Add small perturbation
               cf = cf + np.random.randn(*cf.shape) * self.perturbation_scale

               # Check if prediction changed
               cf_label = int(self.model.predict(cf.reshape(1, -1))[0])
               if cf_label != y_pred:
                   break

           # Build metadata
           meta = {
               "method": "my_counterfactual",
               "iterations": i + 1,
               "original_label": y_pred,
               "perturbation_scale": self.perturbation_scale,
               "converged": cf_label != y_pred,
           }

           return cf, cf_label, meta

Using Your Custom Method
~~~~~~~~~~~~~~~~~~~~~~~~

Once implemented, use it like any built-in explainer:

.. code-block:: python

   from sklearn.neighbors import KNeighborsClassifier

   # Train classifier
   clf = KNeighborsClassifier(n_neighbors=5)
   clf.fit(X_train, y_train)

   # Create your explainer
   explainer = MyCounterfactual(
       model=clf,
       data=(X_train, y_train),
       perturbation_scale=0.05,
   )

   # Generate counterfactual for a single instance
   x = X_test[0]
   cf, cf_label, meta = explainer.explain(x)

   print(f"Original shape: {x.shape}")
   print(f"Counterfactual shape: {cf.shape}")
   print(f"New label: {cf_label}")
   print(f"Metadata: {meta}")


Benchmarking Against Built-in Methods
-------------------------------------

Compare your custom method against tscf-eval's built-in explainers using the
benchmarking framework.

Single-Dataset Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tscf_eval import Evaluator, Validity, Proximity, Sparsity
   from tscf_eval.benchmark import (
       BenchmarkRunner, DatasetConfig, ModelConfig, ExplainerConfig,
   )
   from tscf_eval.counterfactuals import COMTE, NativeGuide, Glacier
   from tscf_eval.data_loader import UCRLoader
   from aeon.classification.convolution_based import RocketClassifier

   # Load data
   loader = UCRLoader("ItalyPowerDemand")
   train, test = loader.load("train"), loader.load("test")

   # Train classifier
   model = RocketClassifier(n_kernels=500, random_state=42)
   model.fit(train.X, train.y)

   # Define explainer configurations including your custom method
   explainer_configs = [
       ExplainerConfig("my_method", MyCounterfactual, {"perturbation_scale": 0.05}),
       ExplainerConfig("comte_dtw", COMTE, {"distance": "dtw"}),
       ExplainerConfig("ng_blend", NativeGuide, {"method": "blend"}),
       ExplainerConfig(
           "glacier_uniform", Glacier,
           {"weight_type": "uniform", "pred_margin_weight": 0.5},
       ),
   ]

   # Create and run benchmark
   runner = BenchmarkRunner(
       datasets=[DatasetConfig("ItalyPowerDemand", train.X, train.y, test.X, test.y)],
       models=[ModelConfig("rocket", model)],
       explainers=explainer_configs,
       evaluator=Evaluator([Validity(), Proximity(p=2), Sparsity()]),
       n_instances=20,
       instance_selection="stratified_confidence",
       verbose=True,
   )

   results = runner.run()

   # View results as DataFrame
   df = results.to_dataframe()
   print(df)

   # Aggregate by explainer
   print(results.aggregate(by="explainer"))

Multi-Dataset Benchmark
~~~~~~~~~~~~~~~~~~~~~~~

For evaluation across multiple datasets, pass multiple ``DatasetConfig`` and
``ModelConfig`` entries to ``BenchmarkRunner``:

.. code-block:: python

   from tscf_eval.benchmark import (
       BenchmarkRunner, DatasetConfig, ModelConfig, ExplainerConfig,
   )
   from tscf_eval.data_loader import UCRLoader
   from aeon.classification.convolution_based import RocketClassifier

   # Load multiple datasets and train models
   datasets = []
   models_by_dataset = {}
   for name in ["ItalyPowerDemand", "GunPoint", "ECG200"]:
       loader = UCRLoader(name)
       train, test = loader.load("train"), loader.load("test")
       datasets.append(DatasetConfig(name, train.X, train.y, test.X, test.y))

       clf = RocketClassifier(n_kernels=500, random_state=42)
       clf.fit(train.X, train.y)
       models_by_dataset[name] = clf

   model_configs = [ModelConfig("rocket", models_by_dataset[ds.name]) for ds in datasets]

   # Explainer configs (including your custom method)
   explainer_configs = [
       ExplainerConfig("my_method", MyCounterfactual, {"perturbation_scale": 0.05}),
       ExplainerConfig("comte_dtw", COMTE, {"distance": "dtw"}),
       ExplainerConfig("ng_blend", NativeGuide, {"method": "blend"}),
   ]

   # Run multi-dataset benchmark
   runner = BenchmarkRunner(
       datasets=datasets,
       models=model_configs,
       explainers=explainer_configs,
       n_instances=10,
       verbose=True,
   )

   results = runner.run()

   # Aggregate by explainer (mean across all datasets)
   print(results.aggregate(by="explainer"))

   # Filter results for a specific dataset
   gunpoint_results = results.filter(datasets=["GunPoint"])
   print(gunpoint_results.to_dataframe())

Parallel Benchmark Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Speed up benchmarks by running explainers in parallel:

.. code-block:: python

   runner = BenchmarkRunner(
       datasets=datasets,
       models=model_configs,
       explainers=explainer_configs,
       n_instances=20,
       n_jobs=-1,  # Use all CPUs (-1) or specific number (e.g., 4)
       verbose=True,
   )

   results = runner.run()

.. note::

   Parallel execution requires ``joblib`` to be installed:
   ``pip install joblib``

   If joblib is not available, the benchmark falls back to sequential execution.

Pareto Analysis
~~~~~~~~~~~~~~~

Identify Pareto-optimal methods based on multiple criteria:

.. code-block:: python

   from tscf_eval.benchmark import ParetoAnalyzer

   # Create analyzer with metric names (directions are inferred)
   analyzer = ParetoAnalyzer(metrics=[
       "validity", "proximity_l2", "sparsity", "efficiency_time_s",
   ])

   # Find Pareto-optimal methods
   pareto_methods = analyzer.pareto_front(results)
   print(f"Pareto-optimal methods: {pareto_methods}")

   # Full dominance ranking table
   ranking = analyzer.dominance_ranking(results)
   print(ranking)

   # Export to LaTeX
   latex = analyzer.to_latex(results, caption="Pareto Ranking", label="tab:pareto")

Pareto Visualization
~~~~~~~~~~~~~~~~~~~~

``ParetoAnalyzer`` provides methods for visualizing trade-offs:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - ``plot_front()``
     - 2D scatter plot showing Pareto front between two objectives
   * - ``plot_consistency_heatmap()``
     - Heatmap of Pareto consistency across datasets

**2D Pareto Front Plot**:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Scatter plot with Pareto front highlighted
   ax = analyzer.plot_front(
       results,
       x_metric="proximity_l2",
       y_metric="validity",
       annotate=True,
   )
   plt.savefig("pareto_front.png")
   plt.show()

**Cross-Dataset Consistency Heatmap**:

.. code-block:: python

   # Build per-dataset results dict
   results_by_dataset = {
       ds: results.filter(datasets=[ds])
       for ds in results.datasets
   }

   # Compute consistency matrix
   consistency_df = analyzer.consistency(results_by_dataset)
   print(consistency_df)

   # Plot heatmap
   ax = analyzer.plot_consistency_heatmap(consistency_df)
   plt.savefig("consistency_heatmap.png")
   plt.show()

.. note::

   Plotting requires ``matplotlib`` to be installed.

Weighted Scalarization
~~~~~~~~~~~~~~~~~~~~~~

Aggregate metrics into a composite score for ranking:

.. code-block:: python

   from tscf_eval.benchmark import WeightedScalarizer

   # Equal-weight composite
   scalarizer = WeightedScalarizer(metrics=[
       "validity", "proximity_l2", "sparsity",
   ])
   scores = scalarizer.score(results)
   print(scores)

   # Custom weights emphasizing validity
   scalarizer = WeightedScalarizer(
       metrics=["validity", "proximity_l2", "sparsity"],
       weights={"validity": 3.0, "proximity_l2": 1.0, "sparsity": 1.0},
   )

   # Sensitivity analysis: how ranking changes as one metric's weight varies
   sens_df = scalarizer.sensitivity(results, vary_metric="validity", n_steps=11)
   scalarizer.plot_sensitivity(sens_df)
   plt.show()

Statistical Testing
~~~~~~~~~~~~~~~~~~~

Use the Friedman test to compare explainers across datasets:

.. code-block:: python

   from tscf_eval.benchmark import friedman_test

   fr = friedman_test(results, metric="validity")
   print(f"Statistic: {fr.statistic:.3f}, p-value: {fr.p_value:.4f}")
   print(fr.rankings)


Advanced: Custom Metrics
------------------------

You can also implement custom evaluation metrics.

Implementing a Custom Metric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from tscf_eval.evaluator import Metric

   class MyCustomMetric(Metric):
       """My custom evaluation metric.

       Parameters
       ----------
       threshold : float
           Threshold for the metric computation.
       """

       def __init__(self, threshold: float = 0.1):
           self.threshold = threshold

       def name(self) -> str:
           """Return the metric name."""
           return f"my_metric_t{self.threshold}"

       def compute(
           self,
           X: np.ndarray,
           X_cf: np.ndarray,
           **kwargs,
       ) -> float:
           """Compute the metric.

           Parameters
           ----------
           X : np.ndarray
               Original instances, shape (N, ...).
           X_cf : np.ndarray
               Counterfactual instances, same shape as X.
           **kwargs
               Additional arguments (model, X_train, y, y_cf, etc.).

           Returns
           -------
           float
               The computed metric value.
           """
           # Example: fraction of instances where max change exceeds threshold
           diff = np.abs(X - X_cf)
           max_changes = np.max(diff.reshape(X.shape[0], -1), axis=1)
           return float(np.mean(max_changes > self.threshold))

Using Custom Metrics
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tscf_eval.evaluator import Evaluator, Validity, Proximity

   evaluator = Evaluator([
       Validity(),
       Proximity(p=2),
       MyCustomMetric(threshold=0.1),
       MyCustomMetric(threshold=0.5),
   ])

   results = evaluator.evaluate(X, X_cf, y=y, y_cf=y_cf)

   print(results["my_metric_t0.1"])
   print(results["my_metric_t0.5"])


Complete Example: End-to-End Workflow
-------------------------------------

A complete example showing the full workflow:

.. code-block:: python

   import numpy as np
   from sklearn.neighbors import KNeighborsClassifier

   from tscf_eval.counterfactuals import Counterfactual, COMTE, NativeGuide
   from tscf_eval.evaluator import Evaluator, Validity, Proximity, Sparsity
   from tscf_eval.benchmark import (
       BenchmarkRunner, DatasetConfig, ModelConfig, ExplainerConfig,
       ParetoAnalyzer, WeightedScalarizer,
   )
   from tscf_eval.data_loader import UCRLoader


   # 1. Define your custom counterfactual method
   class GradientFreeOptimizer(Counterfactual):
       def __init__(self, model, data, n_iterations=50):
           self.model = model
           self.X_train, self.y_train = data
           self.n_iterations = n_iterations

       def explain(self, x, y_pred=None):
           x = np.asarray(x).squeeze()
           if y_pred is None:
               y_pred = int(self.model.predict(x.reshape(1, -1))[0])

           # Find nearest unlike neighbor
           preds = self.model.predict(self.X_train)
           unlike_mask = preds != y_pred
           unlike_samples = self.X_train[unlike_mask]

           # Use it as target
           distances = np.linalg.norm(
               unlike_samples.reshape(len(unlike_samples), -1) - x.flatten(),
               axis=1
           )
           target = unlike_samples[np.argmin(distances)]

           # Interpolate toward target
           cf = x.copy()
           for i in range(self.n_iterations):
               alpha = (i + 1) / self.n_iterations
               cf = (1 - alpha) * x + alpha * target.squeeze()
               cf_label = int(self.model.predict(cf.reshape(1, -1))[0])
               if cf_label != y_pred:
                   break

           meta = {"method": "gradient_free", "iterations": i + 1, "alpha": alpha}
           return cf, cf_label, meta


   # 2. Load data
   loader = UCRLoader("ItalyPowerDemand")
   train = loader.load("train")
   test = loader.load("test")


   # 3. Train classifier
   clf = KNeighborsClassifier(n_neighbors=5)
   clf.fit(train.X, train.y)


   # 4. Configure and run benchmark
   evaluator = Evaluator([Validity(), Proximity(p=2), Sparsity()])

   runner = BenchmarkRunner(
       datasets=[DatasetConfig("ItalyPowerDemand", train.X, train.y, test.X, test.y)],
       models=[ModelConfig("knn", clf)],
       explainers=[
           ExplainerConfig("my_optimizer", GradientFreeOptimizer, {"n_iterations": 50}),
           ExplainerConfig("comte", COMTE, {"distance": "euclidean"}),
           ExplainerConfig("ng_blend", NativeGuide, {"method": "blend"}),
       ],
       evaluator=evaluator,
       n_instances=20,
       instance_selection="stratified_confidence",
       verbose=True,
   )

   results = runner.run()


   # 5. Analyze results
   print("\n=== Results DataFrame ===")
   print(results.to_dataframe())

   print("\n=== Aggregated by Explainer ===")
   print(results.aggregate(by="explainer"))

   print("\n=== Per-Result Details ===")
   for result in results:
       print(f"\n{result.explainer_name}:")
       print(f"  Success rate: {result.success_rate:.0%}")
       print(f"  Mean time: {result.mean_time:.4f}s")
       print(f"  Validity: {result.metrics.get('validity', 'N/A')}")


   # 6. Pareto analysis
   analyzer = ParetoAnalyzer(metrics=["validity", "proximity_l2", "sparsity"])
   pareto_methods = analyzer.pareto_front(results)
   print(f"\nPareto-optimal: {pareto_methods}")
   print(analyzer.dominance_ranking(results))


   # 7. Weighted scalarization
   scalarizer = WeightedScalarizer(metrics=["validity", "proximity_l2", "sparsity"])
   print(scalarizer.score(results))


   # 8. Save for later analysis
   import json
   with open("my_benchmark_results.json", "w") as f:
       json.dump(results.to_dict(), f, indent=2)
