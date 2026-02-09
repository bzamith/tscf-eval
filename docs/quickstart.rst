Quick Start Guide
=================

This guide will help you get started with tscf-eval for evaluating
counterfactual explanations for time series classification.

Loading Data
------------

tscf-eval provides utilities for loading time series data.

From the UCR Archive
~~~~~~~~~~~~~~~~~~~~

The easiest way to get started is using the UCR Time Series Archive:

.. code-block:: python

   from tscf_eval import UCRLoader

   loader = UCRLoader("ItalyPowerDemand")
   train_data = loader.load("train")
   test_data = loader.load("test")

   print(f"Train: {train_data.X.shape}, Test: {test_data.X.shape}")
   print(train_data.describe())

From NumPy Arrays
~~~~~~~~~~~~~~~~~

You can also create data containers from your own arrays:

.. code-block:: python

   from tscf_eval import TSCData
   import numpy as np

   X = np.random.randn(100, 50)  # 100 instances, 50 time points
   y = np.array([0] * 50 + [1] * 50)

   data = TSCData.from_arrays(
       name="my_dataset",
       split="train",
       X=X,  # Shape: (n, T) or (n, C, T)
       y=y,  # Shape: (n,)
   )

Basic Usage
-----------

Evaluating Counterfactuals
~~~~~~~~~~~~~~~~~~~~~~~~~~

The core functionality of tscf-eval is evaluating counterfactual quality
using the :class:`~tscf_eval.Evaluator` class:

.. code-block:: python

   from sklearn.neighbors import KNeighborsClassifier
   from tscf_eval import (
       Evaluator, Validity, Proximity, Sparsity,
       UCRLoader, NativeGuide,
   )

   # Load data
   loader = UCRLoader("ItalyPowerDemand")
   train, test = loader.load("train"), loader.load("test")

   # Train classifier
   clf = KNeighborsClassifier(n_neighbors=3)
   clf.fit(train.X, train.y)

   # Generate counterfactuals using NativeGuide
   explainer = NativeGuide(clf, (train.X, train.y), method="blend")
   X, X_cf, y, y_cf = [], [], [], []
   for x in test.X[:10]:
       cf, cf_label, _ = explainer.explain(x)
       X.append(x)
       X_cf.append(cf)
       y.append(clf.predict(x.reshape(1, -1))[0])
       y_cf.append(cf_label)

   # Create evaluator with desired metrics
   evaluator = Evaluator([
       Validity(),
       Proximity(p=2, distance="lp"),
       Proximity(distance="dtw"),  # DTW-based proximity
       Sparsity(),
   ])

   # Run evaluation
   results = evaluator.evaluate(X, X_cf, y=y, y_cf=y_cf)

   # Access results
   print(f"Validity: {results['validity_soft']:.2f}")
   print(f"Proximity (L2): {results['proximity_l2']:.2f}")
   print(f"Proximity (DTW): {results['proximity_dtw']:.2f}")
   print(f"Sparsity: {results['sparsity']:.2f}")

Using a Classifier
~~~~~~~~~~~~~~~~~~

For metrics like Validity and Controllability, you can provide a fitted
classifier instead of labels:

.. code-block:: python

   from sklearn.neighbors import KNeighborsClassifier
   from tscf_eval import (
       Evaluator, Validity, Proximity, Sparsity,
       UCRLoader, COMTE,
   )

   # Load data
   loader = UCRLoader("ItalyPowerDemand")
   train, test = loader.load("train"), loader.load("test")

   # Train a classifier
   clf = KNeighborsClassifier(n_neighbors=3)
   clf.fit(train.X, train.y)

   # Generate counterfactuals using COMTE
   explainer = COMTE(clf, (train.X, train.y), distance="dtw")
   X, X_cf = [], []
   for x in test.X[:10]:
       cf, _, _ = explainer.explain(x)
       X.append(x)
       X_cf.append(cf)

   # Evaluate using the classifier (labels inferred from model)
   evaluator = Evaluator([
       Validity(),
       Proximity(p=2, distance="lp"),
       Proximity(distance="dtw"),
       Sparsity(),
   ])
   results = evaluator.evaluate(X, X_cf, model=clf)

Some metrics require additional inputs:

- ``model``: Validity, Controllability, Confidence
- ``X_train``: Plausibility, Diversity
- ``time_per_instance``: Efficiency

Available Metrics
-----------------

tscf-eval provides 11 metric classes organized into six quality dimensions:

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Metric
     - Description
     - Range
   * - Validity
     - Fraction of CFs that change prediction
     - [0, 1]
   * - Proximity(p)
     - Proximity score ``1 / (1 + d)``, where ``d`` is distance
     - [0, 1]
   * - Sparsity
     - Fraction of changed features
     - [0, 1]
   * - Plausibility
     - Outlier detection score
     - [0, 1]
   * - Diversity
     - DPP-based diversity score
     - [0, +inf)
   * - Controllability
     - Ease of reverting changes
     - [0, 1]
   * - Confidence
     - Model confidence statistics
     - dict
   * - Composition
     - Edit segment statistics
     - dict
   * - Contiguity
     - Edit contiguity score
     - [0, 1]
   * - Robustness
     - Lipschitz-like stability
     - [0, +inf)
   * - Efficiency
     - Mean time per instance
     - seconds

Next Steps
----------

- See :doc:`examples` for more detailed usage examples
- Explore the :doc:`api/evaluator` for all available metrics
- Check :doc:`api/counterfactuals` for counterfactual generation methods
