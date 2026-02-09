Examples
========

This guide provides comprehensive examples for common use cases with TSCFEval,
from generating counterfactuals to running benchmarks and analyzing results.

.. contents:: Table of Contents
   :local:
   :depth: 2


Generating Counterfactuals
--------------------------

TSCFEval provides 7 built-in counterfactual methods covering different generation
strategies: instance-based (NativeGuide, COMTE), evolutionary (TSEvo), gradient-based
(Glacier, LatentCF), saliency-based (CELS), and shapelet-based (SETS).

All methods follow a unified interface:

1. Initialize with a fitted classifier and training data tuple ``(X_train, y_train)``
2. Call ``explain(x)`` to generate a counterfactual for instance ``x``
3. Returns a tuple ``(cf, cf_label, meta)`` containing the counterfactual,
   its predicted label, and method-specific metadata

Using NativeGuide
~~~~~~~~~~~~~~~~~

NativeGuide is an instance-based method that generates counterfactuals by guiding
the original instance toward its nearest unlike neighbor (NUN) - the closest training
instance with a different predicted class. It supports four blending strategies:

- ``blend``: Linear interpolation toward NUN until prediction flips
- ``ng``: Native Guide with weighted averaging
- ``dtw_dba``: DTW Barycentric Averaging for time-series-aware blending
- ``cam``: Class Activation Map weighted guidance

.. code-block:: python

   from sklearn.neighbors import KNeighborsClassifier
   from tscf_eval import UCRLoader, NativeGuide

   # Load data and train classifier
   loader = UCRLoader("ItalyPowerDemand")
   train, test = loader.load("train"), loader.load("test")
   clf = KNeighborsClassifier(n_neighbors=3)
   clf.fit(train.X, train.y)

   # Create explainer (methods: "blend", "ng", "dtw_dba", "cam")
   explainer = NativeGuide(clf, (train.X, train.y), method="blend")

   # Generate counterfactual for a single instance
   x = test.X[0]
   cf, cf_label, meta = explainer.explain(x)

   print(f"Original prediction: {clf.predict(x.reshape(1, -1))[0]}")
   print(f"Counterfactual prediction: {cf_label}")

Using COMTE
~~~~~~~~~~~

COMTE (Counterfactual Multivariate Time-series Explanations) generates counterfactuals
by greedily substituting channels from a "distractor" series - a training instance
from a different class. It iteratively replaces channels until the prediction flips,
producing sparse, interpretable explanations that highlight which channels are most
important for the classification decision. Works with both univariate and multivariate
time series, using Euclidean or DTW distance for distractor selection:

.. code-block:: python

   from tscf_eval import UCRLoader, COMTE

   explainer = COMTE(clf, (train.X, train.y), distance="dtw")
   cf, cf_label, meta = explainer.explain(test.X[0])

Using TSEvo
~~~~~~~~~~~

TSEvo uses multi-objective evolutionary optimization (NSGA-II) to generate
counterfactuals that balance validity, proximity, and plausibility. It applies
mutation operators to evolve a population of candidate counterfactuals over
multiple generations. Three transformer types control how mutations are applied:

- ``authentic``: Mutations based on authentic patterns from training data
- ``frequency``: Frequency-domain perturbations
- ``gaussian``: Random Gaussian noise perturbations

.. code-block:: python

   from tscf_eval import UCRLoader, TSEvo

   # Transformers: "authentic", "frequency", "gaussian"
   explainer = TSEvo(clf, (train.X, train.y), transformer="authentic")
   cf, cf_label, meta = explainer.explain(test.X[0])

Using Glacier
~~~~~~~~~~~~~

Glacier (Guided Locally Constrained Counterfactual Explanations) uses gradient-based
optimization with importance-weighted proximity constraints. It optimizes in the input
space while penalizing changes to important time points more heavily. Requires a
differentiable classifier (e.g., neural networks). The ``weight_type`` parameter
controls how importance weights are computed:

- ``uniform``: Equal weight for all time points
- ``local``: Weights based on local gradients (instance-specific)
- ``global``: Weights based on global feature importance

.. code-block:: python

   from tscf_eval import UCRLoader, Glacier

   # Weight types: "uniform", "local", "global"
   explainer = Glacier(clf, (train.X, train.y), weight_type="uniform")
   cf, cf_label, meta = explainer.explain(test.X[0])

Using SETS and CELS
~~~~~~~~~~~~~~~~~~~

SETS and CELS use different strategies for identifying discriminative regions:

- **SETS** (Shapelet-based Explanations for Time Series): Identifies class-discriminative
  shapelets and generates counterfactuals by manipulating these subsequences. Produces
  contiguous, localized perturbations that are often more interpretable.

- **CELS** (Counterfactual Explanations via Learned Saliency): Uses learned saliency maps
  to identify important time points, then blends the original instance with its nearest
  unlike neighbor weighted by the saliency scores. Produces smooth counterfactuals that
  focus changes on the most discriminative regions.

.. code-block:: python

   from tscf_eval import UCRLoader, SETS, CELS

   # SETS: Shapelet-based explanations
   explainer_sets = SETS(clf, (train.X, train.y))
   cf, cf_label, meta = explainer_sets.explain(test.X[0])

   # CELS: Saliency map blending
   explainer_cels = CELS(clf, (train.X, train.y))
   cf, cf_label, meta = explainer_cels.explain(test.X[0])


Evaluating Counterfactuals
--------------------------

TSCFEval provides 11 metrics across 6 quality dimensions for comprehensive
counterfactual evaluation:

1. **Core Quality**: Validity, Proximity, Sparsity
2. **Distribution Alignment**: Plausibility, Diversity
3. **Structural Properties**: Contiguity, Composition
4. **Model Behavior**: Confidence, Controllability
5. **Stability**: Robustness
6. **Performance**: Efficiency

The ``Evaluator`` class provides a flexible interface for computing any combination
of these metrics. Each metric has specific requirements (e.g., some need the model,
others need training data) which are detailed in the API reference.

Basic Evaluation
~~~~~~~~~~~~~~~~

The core metrics (Validity, Proximity, Sparsity) measure fundamental counterfactual
quality. Validity checks if the prediction changed, Proximity measures how close
the counterfactual is to the original, and Sparsity quantifies the fraction of
changed features:

.. code-block:: python

   from sklearn.neighbors import KNeighborsClassifier
   from tscf_eval import UCRLoader, NativeGuide
   from tscf_eval.evaluator import Evaluator, Validity, Proximity, Sparsity

   # Load data and train classifier
   loader = UCRLoader("ItalyPowerDemand")
   train, test = loader.load("train"), loader.load("test")
   clf = KNeighborsClassifier(n_neighbors=3)
   clf.fit(train.X, train.y)

   # Generate counterfactuals
   explainer = NativeGuide(clf, (train.X, train.y), method="blend")
   X, X_cf, y, y_cf = [], [], [], []
   for x in test.X[:10]:
       cf, cf_label, _ = explainer.explain(x)
       X.append(x)
       X_cf.append(cf)
       y.append(clf.predict(x.reshape(1, -1))[0])
       y_cf.append(cf_label)

   # Create evaluator
   evaluator = Evaluator([
       Validity(),
       Proximity(p=2, distance="lp"),
       Proximity(distance="dtw"),
       Sparsity(),
   ])

   # Evaluate
   results = evaluator.evaluate(X, X_cf, y=y, y_cf=y_cf)

   print(f"Validity: {results['validity_soft']:.2%}")
   print(f"Proximity (L2): {results['proximity_l2']:.4f}")
   print(f"Proximity (DTW): {results['proximity_dtw']:.4f}")
   print(f"Sparsity: {results['sparsity']:.2%}")

Using Model-Dependent Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some metrics require access to the classifier to compute their values:

- **Validity**: When labels aren't provided, predictions are inferred from the model
- **Controllability**: Measures how easily the counterfactual changes can be reverted
  by modifying a single feature (requires making predictions on modified instances)
- **Confidence**: Reports the model's predicted probabilities for both the original
  instance and the counterfactual (requires ``predict_proba``)

.. code-block:: python

   from tscf_eval.evaluator import (
       Evaluator, Validity, Proximity, Sparsity,
       Controllability, Confidence
   )

   evaluator = Evaluator([
       Validity(),
       Proximity(distance="dtw"),
       Sparsity(),
       Controllability(),
       Confidence(),
   ])

   # Pass the model to evaluate()
   results = evaluator.evaluate(X, X_cf, model=clf, X_train=train.X)

   print(f"Validity: {results['validity_soft']:.2%}")
   print(f"Controllability: {results['controllability']:.4f}")
   print(f"Mean CF confidence: {results['mean_conf_cf']:.4f}")

Using Distribution Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

Distribution metrics assess whether counterfactuals are realistic and diverse:

- **Plausibility**: Measures whether counterfactuals lie within the training data
  distribution using outlier detection. High plausibility means the counterfactual
  resembles real training instances. Methods include LOF (Local Outlier Factor),
  Isolation Forest, and DTW-based LOF for time-series-aware detection.

- **Diversity**: When generating multiple counterfactuals per instance, measures
  the variety among them using Determinantal Point Processes (DPP). Higher diversity
  means the counterfactuals explore different regions of the feature space.

Both metrics require ``X_train`` to be passed to ``evaluate()``:

.. code-block:: python

   from tscf_eval.evaluator import (
       Evaluator, Plausibility, Diversity, Contiguity
   )

   evaluator = Evaluator([
       Plausibility(method="lof"),       # Local Outlier Factor
       Plausibility(method="dtw_lof"),   # DTW-based LOF
       Diversity(distance="euclidean"),
       Diversity(distance="dtw"),
       Contiguity(),
   ])

   # Pass X_train for distribution metrics
   results = evaluator.evaluate(X, X_cf, y=y, y_cf=y_cf, X_train=train.X)

Measuring Efficiency
~~~~~~~~~~~~~~~~~~~~

The Efficiency metric tracks how long it takes to generate each counterfactual.
This is important for comparing methods in practical applications where generation
time matters. You must measure the time yourself and pass it to the evaluator:

.. code-block:: python

   import time
   from tscf_eval import TSEvo
   from tscf_eval.evaluator import Evaluator, Validity, Proximity, Efficiency

   explainer = TSEvo(clf, (train.X, train.y), transformer="authentic")
   X, X_cf, times = [], [], []
   for x in test.X[:5]:
       start = time.perf_counter()
       cf, _, _ = explainer.explain(x)
       times.append(time.perf_counter() - start)
       X.append(x)
       X_cf.append(cf)

   evaluator = Evaluator([Validity(), Proximity(distance="dtw"), Efficiency()])
   results = evaluator.evaluate(X, X_cf, model=clf, time_per_instance=times)

   print(f"Mean time: {results['efficiency_time_s']:.4f}s")

Full Evaluation with All Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For comprehensive evaluation, you can use all available metrics together. Note that
this requires providing all optional parameters (``model``, ``X_train``, ``y``, ``y_cf``,
``time_per_instance``) to satisfy each metric's requirements:

.. code-block:: python

   import time
   from tscf_eval import UCRLoader, Glacier
   from tscf_eval.evaluator import (
       Evaluator, Validity, Proximity, Sparsity,
       Plausibility, Diversity, Controllability, Confidence,
       Composition, Contiguity, Robustness, Efficiency
   )

   evaluator = Evaluator([
       # Core
       Validity(),
       Proximity(p=2, distance="lp"),
       Proximity(distance="dtw"),
       Sparsity(),
       # Distribution
       Plausibility(method="lof"),
       Plausibility(method="dtw_lof"),
       Diversity(distance="dtw"),
       # Model behavior
       Controllability(),
       Confidence(),
       # Structure
       Composition(),
       Contiguity(),
       # Stability and performance
       Robustness(distance="dtw"),
       Efficiency(),
   ])

   results = evaluator.evaluate(
       X, X_cf,
       model=clf,
       X_train=train.X,
       y=y,
       y_cf=y_cf,
       time_per_instance=times,
   )


Running Benchmarks
------------------

The ``BenchmarkRunner`` class provides a structured framework for systematically
comparing counterfactual methods. It handles:

- **Instance selection**: Random or confidence-stratified sampling of test instances
- **Parallel execution**: Run multiple explainers in parallel with ``n_jobs``
- **Progress tracking**: Built-in progress bars with tqdm
- **Result aggregation**: Aggregate results by explainer, dataset, or model

TSCFEval supports three benchmarking scenarios:

1. **Single dataset, multiple CF methods**: Compare explainer algorithms on a fixed dataset
2. **Single dataset, multiple classifiers**: Study how the classifier affects CF quality
3. **Multiple datasets, fixed classifier**: Assess generalization across datasets

Single-Dataset Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~

The most common scenario: compare multiple counterfactual methods on a single dataset
with a fixed classifier. Use ``instance_selection="stratified_confidence"`` to ensure
coverage of both high-confidence and uncertain instances near the decision boundary:

.. code-block:: python

   from sklearn.neighbors import KNeighborsClassifier
   from tscf_eval import Evaluator, Validity, Proximity, Sparsity
   from tscf_eval.benchmark import (
       BenchmarkRunner, DatasetConfig, ModelConfig, ExplainerConfig,
   )
   from tscf_eval.counterfactuals import COMTE, NativeGuide, Glacier
   from tscf_eval.data_loader import UCRLoader

   # Load data
   loader = UCRLoader("ItalyPowerDemand")
   train, test = loader.load("train"), loader.load("test")

   # Train classifier
   clf = KNeighborsClassifier(n_neighbors=3)
   clf.fit(train.X, train.y)

   # Configure explainers
   explainer_configs = [
       ExplainerConfig("comte", COMTE, {"distance": "dtw"}),
       ExplainerConfig("ng_blend", NativeGuide, {"method": "blend"}),
       ExplainerConfig("glacier", Glacier, {"weight_type": "uniform"}),
   ]

   # Configure evaluator
   evaluator = Evaluator([
       Validity(),
       Proximity(distance="dtw"),
       Sparsity(),
   ])

   # Run benchmark
   runner = BenchmarkRunner(
       datasets=[DatasetConfig("ItalyPowerDemand", train.X, train.y, test.X, test.y)],
       models=[ModelConfig("knn", clf)],
       explainers=explainer_configs,
       evaluator=evaluator,
       n_instances=20,
       instance_selection="stratified_confidence",
       verbose=True,
   )
   results = runner.run()

   # View results
   print(results.to_dataframe())
   print(results.aggregate(by="explainer"))

Multi-Dataset Benchmark
~~~~~~~~~~~~~~~~~~~~~~~

To assess how well counterfactual methods generalize, run benchmarks across multiple
datasets. This enables statistical testing (e.g., Friedman test) to determine if
performance differences are significant across problem domains:

.. code-block:: python

   from tscf_eval.benchmark import (
       BenchmarkRunner, DatasetConfig, ModelConfig, ExplainerConfig,
   )
   from tscf_eval.counterfactuals import COMTE, NativeGuide
   from tscf_eval.data_loader import UCRLoader

   # Load datasets and train models
   dataset_names = ["ItalyPowerDemand", "GunPoint", "ECG200"]
   datasets, model_configs = [], []

   for name in dataset_names:
       loader = UCRLoader(name)
       train, test = loader.load("train"), loader.load("test")
       datasets.append(DatasetConfig(name, train.X, train.y, test.X, test.y))

       clf = KNeighborsClassifier(n_neighbors=3)
       clf.fit(train.X, train.y)
       model_configs.append(ModelConfig("knn", clf))

   # Run benchmark
   runner = BenchmarkRunner(
       datasets=datasets,
       models=model_configs,
       explainers=[
           ExplainerConfig("comte", COMTE, {"distance": "dtw"}),
           ExplainerConfig("ng_blend", NativeGuide, {"method": "blend"}),
       ],
       n_instances=10,
       n_jobs=-1,  # Parallel execution
       verbose=True,
   )
   results = runner.run()

   # Aggregate across datasets
   print(results.aggregate(by="explainer"))


Analyzing Results
-----------------

Counterfactual evaluation is inherently multi-objective: high validity may come at
the cost of low proximity, and sparse explanations may sacrifice plausibility.
TSCFEval provides tools for principled multi-criteria analysis.

Pareto Analysis
~~~~~~~~~~~~~~~

Pareto analysis identifies methods that are not dominated by any other method on
the selected metrics. A method is Pareto-optimal if no other method is better on
all metrics simultaneously. This avoids the need to specify metric weights upfront:

.. code-block:: python

   from tscf_eval.benchmark import ParetoAnalyzer

   analyzer = ParetoAnalyzer(metrics=[
       "validity_soft", "proximity_dtw", "sparsity",
   ])

   # Find non-dominated methods
   pareto_methods = analyzer.pareto_front(results)
   print(f"Pareto-optimal: {pareto_methods}")

   # Full ranking table
   print(analyzer.dominance_ranking(results))

   # Export to LaTeX
   latex = analyzer.to_latex(results, caption="Results", label="tab:results")

Visualizing Pareto Fronts
~~~~~~~~~~~~~~~~~~~~~~~~~

Pareto front visualizations help understand the trade-offs between metrics.
The 2D plot shows which methods lie on the Pareto front (non-dominated solutions)
for any pair of metrics. Consistency heatmaps show how often each method appears
on the Pareto front across different datasets:

.. code-block:: python

   import matplotlib.pyplot as plt

   # 2D Pareto front plot
   ax = analyzer.plot_front(
       results,
       x_metric="proximity_dtw",
       y_metric="validity_soft",
       annotate=True,
   )
   plt.savefig("pareto_front.png")

   # Cross-dataset consistency heatmap
   results_by_dataset = {
       ds: results.filter(datasets=[ds])
       for ds in results.datasets
   }
   consistency_df = analyzer.consistency(results_by_dataset)
   analyzer.plot_consistency_heatmap(consistency_df)
   plt.savefig("consistency.png")

Weighted Scalarization
~~~~~~~~~~~~~~~~~~~~~~

When you need a single ranking of methods, weighted scalarization combines metrics
into a composite score. Each metric is min-max normalized to [0, 1] with direction
awareness (maximize metrics are higher-is-better, minimize metrics are inverted),
then combined via weighted sum. This enables customizable rankings based on your
priorities:

.. code-block:: python

   from tscf_eval.benchmark import WeightedScalarizer

   # Equal weights
   scalarizer = WeightedScalarizer(metrics=[
       "validity_soft", "proximity_dtw", "sparsity",
   ])
   print(scalarizer.score(results))

   # Custom weights
   scalarizer = WeightedScalarizer(
       metrics=["validity_soft", "proximity_dtw", "sparsity"],
       weights={"validity_soft": 3.0, "proximity_dtw": 1.0, "sparsity": 1.0},
   )

   # Sensitivity analysis
   sens_df = scalarizer.sensitivity(results, vary_metric="validity_soft", n_steps=11)
   scalarizer.plot_sensitivity(sens_df)

Statistical Testing
~~~~~~~~~~~~~~~~~~~

When benchmarking across multiple datasets, the Friedman test determines if there
are statistically significant differences between methods. It's a non-parametric
alternative to repeated-measures ANOVA, ranking methods within each dataset and
testing if the average ranks differ significantly:

.. code-block:: python

   from tscf_eval.benchmark import friedman_test

   fr = friedman_test(results, metric="validity_soft")
   print(f"Statistic: {fr.statistic:.3f}, p-value: {fr.p_value:.4f}")
   print(fr.rankings)


Extending TSCFEval
------------------

TSCFEval is designed to be extensible. You can add your own counterfactual methods
and evaluation metrics that integrate seamlessly with the benchmarking framework.

Custom Counterfactual Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add a new counterfactual method, inherit from the ``Counterfactual`` base class
and implement the ``explain`` method. The method receives a single instance ``x``
and returns a tuple ``(cf, cf_label, meta)``:

- ``cf``: The generated counterfactual (same shape as input)
- ``cf_label``: The predicted class label for the counterfactual
- ``meta``: A dictionary with method-specific metadata (e.g., generation parameters)

Here's an example of a simple interpolation-based method:

.. code-block:: python

   import numpy as np
   from tscf_eval.counterfactuals import Counterfactual

   class MyCounterfactual(Counterfactual):
       """Custom counterfactual using nearest unlike neighbor interpolation."""

       def __init__(self, model, data, n_steps=50):
           self.model = model
           self.X_train, self.y_train = data
           self.n_steps = n_steps

       def explain(self, x, y_pred=None):
           x = np.asarray(x).squeeze()
           if y_pred is None:
               y_pred = int(self.model.predict(x.reshape(1, -1))[0])

           # Find nearest unlike neighbor
           preds = self.model.predict(self.X_train)
           unlike_mask = preds != y_pred
           unlike_samples = self.X_train[unlike_mask]
           distances = np.linalg.norm(
               unlike_samples.reshape(len(unlike_samples), -1) - x.flatten(),
               axis=1
           )
           target = unlike_samples[np.argmin(distances)]

           # Interpolate toward target until prediction flips
           cf = x.copy()
           for i in range(self.n_steps):
               alpha = (i + 1) / self.n_steps
               cf = (1 - alpha) * x + alpha * target.squeeze()
               cf_label = int(self.model.predict(cf.reshape(1, -1))[0])
               if cf_label != y_pred:
                   break

           meta = {"method": "my_cf", "steps": i + 1, "alpha": alpha}
           return cf, cf_label, meta

   # Use in benchmarks
   from tscf_eval.benchmark import ExplainerConfig

   config = ExplainerConfig("my_method", MyCounterfactual, {"n_steps": 50})

Custom Evaluation Metric
~~~~~~~~~~~~~~~~~~~~~~~~

To add a new evaluation metric, inherit from the ``Metric`` base class and implement:

- ``name()``: Returns the metric key used in results dictionaries
- ``compute(X, X_cf, **kwargs)``: Computes and returns the metric value

The ``compute`` method receives the original instances ``X``, counterfactuals ``X_cf``,
and any additional keyword arguments passed to ``evaluate()`` (e.g., ``model``,
``X_train``, ``y``, ``y_cf``). Here's an example metric that measures the maximum
per-instance change:

.. code-block:: python

   import numpy as np
   from tscf_eval.evaluator import Metric

   class MaxChangeMetric(Metric):
       """Fraction of instances where max change exceeds threshold."""

       def __init__(self, threshold=0.1):
           self.threshold = threshold

       def name(self):
           return f"max_change_t{self.threshold}"

       def compute(self, X, X_cf, **kwargs):
           diff = np.abs(np.array(X) - np.array(X_cf))
           max_changes = np.max(diff.reshape(len(X), -1), axis=1)
           return float(np.mean(max_changes > self.threshold))

   # Use in evaluator
   from tscf_eval.evaluator import Evaluator, Validity

   evaluator = Evaluator([
       Validity(),
       MaxChangeMetric(threshold=0.1),
       MaxChangeMetric(threshold=0.5),
   ])


Complete Workflow
-----------------

This end-to-end example demonstrates a typical TSCFEval workflow: loading data,
training a classifier, running a benchmark, and analyzing results with multiple
analysis tools. The results are saved to JSON for later analysis or visualization:

.. code-block:: python

   import json
   from sklearn.neighbors import KNeighborsClassifier
   from tscf_eval import UCRLoader
   from tscf_eval.counterfactuals import COMTE, NativeGuide
   from tscf_eval.evaluator import Evaluator, Validity, Proximity, Sparsity
   from tscf_eval.benchmark import (
       BenchmarkRunner, DatasetConfig, ModelConfig, ExplainerConfig,
       ParetoAnalyzer, WeightedScalarizer, friedman_test,
   )

   # 1. Load data
   loader = UCRLoader("ItalyPowerDemand")
   train, test = loader.load("train"), loader.load("test")

   # 2. Train classifier
   clf = KNeighborsClassifier(n_neighbors=5)
   clf.fit(train.X, train.y)

   # 3. Run benchmark
   runner = BenchmarkRunner(
       datasets=[DatasetConfig("ItalyPowerDemand", train.X, train.y, test.X, test.y)],
       models=[ModelConfig("knn", clf)],
       explainers=[
           ExplainerConfig("comte", COMTE, {"distance": "dtw"}),
           ExplainerConfig("ng_blend", NativeGuide, {"method": "blend"}),
       ],
       evaluator=Evaluator([Validity(), Proximity(distance="dtw"), Sparsity()]),
       n_instances=20,
       instance_selection="stratified_confidence",
       verbose=True,
   )
   results = runner.run()

   # 4. View results
   print(results.to_dataframe())
   print(results.aggregate(by="explainer"))

   # 5. Pareto analysis
   analyzer = ParetoAnalyzer(metrics=["validity_soft", "proximity_dtw", "sparsity"])
   print(f"Pareto-optimal: {analyzer.pareto_front(results)}")
   print(analyzer.dominance_ranking(results))

   # 6. Weighted ranking
   scalarizer = WeightedScalarizer(metrics=["validity_soft", "proximity_dtw", "sparsity"])
   print(scalarizer.score(results))

   # 7. Save results
   with open("results.json", "w") as f:
       json.dump(results.to_dict(), f, indent=2)
