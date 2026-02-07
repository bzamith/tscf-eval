Quick Start Guide
=================

This guide will help you get started with tscf-eval for evaluating
counterfactual explanations for time series classification.

Basic Usage
-----------

Evaluating Counterfactuals
~~~~~~~~~~~~~~~~~~~~~~~~~~

The core functionality of tscf-eval is evaluating counterfactual quality
using the :class:`~tscf_eval.Evaluator` class:

.. code-block:: python

   from tscf_eval import Evaluator, Validity, Proximity, Sparsity
   import numpy as np

   # Your original instances and counterfactuals
   X = np.random.randn(100, 50)  # 100 instances, 50 time points
   X_cf = X + np.random.randn(100, 50) * 0.1

   # Labels (original and counterfactual)
   y = np.zeros(100)
   y_cf = np.ones(100)

   # Create evaluator with desired metrics
   evaluator = Evaluator([
       Validity(),
       Proximity(p=2, distance="lp"),
       Sparsity(),
   ])

   # Run evaluation
   results = evaluator.evaluate(X, X_cf, y=y, y_cf=y_cf)

   # Access results
   print(f"Validity: {results['validity']:.2f}")
   print(f"Proximity (L2): {results['proximity_l2']:.2f}")
   print(f"Sparsity: {results['sparsity']:.2f}")

Using a Classifier
~~~~~~~~~~~~~~~~~~

For metrics like Validity and Controllability, you can provide a fitted
classifier instead of labels:

.. code-block:: python

   from sklearn.neighbors import KNeighborsClassifier

   # Train a classifier
   clf = KNeighborsClassifier(n_neighbors=3)
   clf.fit(X_train, y_train)

   # Evaluate using the classifier
   results = evaluator.evaluate(X_test, X_cf, model=clf)

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

Loading Data
------------

tscf-eval provides utilities for loading time series data:

From NumPy Arrays
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tscf_eval import TSCData

   data = TSCData.from_arrays(
       name="my_dataset",
       split="train",
       X=X,  # Shape: (n, T) or (n, C, T)
       y=y,  # Shape: (n,)
   )

From the UCR Archive
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tscf_eval import UCRLoader

   loader = UCRLoader("ItalyPowerDemand")
   train_data = loader.load("train")
   test_data = loader.load("test")

   print(train_data.describe())

Next Steps
----------

- See :doc:`examples` for more detailed usage examples
- Explore the :doc:`api/evaluator` for all available metrics
- Check :doc:`api/counterfactuals` for counterfactual generation methods
