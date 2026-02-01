TSCFEval
========

**TSCFEval** is a model-agnostic Python framework for systematic evaluation
of counterfactual explanations in Time Series Classification (TSC).

Unlike existing libraries that focus on counterfactual generation, TSCFEval
is specifically designed for counterfactual evaluation, consolidating
fragmented evaluation practices from the TSC counterfactual literature into
a unified, extensible toolkit.

Given a time series classifier and counterfactual explanations, TSCFEval provides:

- **10 evaluation metrics** organized into **six quality dimensions** (core quality, distribution alignment, structural properties, model behavior, stability, and computational performance)
- **Weighted scalarization** for aggregating metrics into composite scores, enabling customizable method ranking
- **Confidence-stratified instance selection** for benchmarking across the decision boundary
- **Three benchmarking scenarios**: single dataset with multiple CF methods, single dataset with multiple classifiers, and multiple datasets with a fixed classifier
- **5 built-in CF methods** for generating counterfactuals
- **Pareto and Friedman analysis** for principled multi-criteria comparison

Installation
------------

.. code-block:: bash

   pip install tscf-eval

With optional dependencies:

.. code-block:: bash

   pip install tscf-eval[dtw]   # DTW distance support
   pip install tscf-eval[full]  # All features

Available Methods and Metrics
-----------------------------

Counterfactual Methods
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 15 40 30

   * - Method
     - Strategy
     - Description
     - Reference
   * - ``NativeGuide``
     - Instance-based
     - Nearest unlike neighbor guidance (blend, ng, dtw_dba, cam)
     - Delaney et al., 2021
   * - ``COMTE``
     - Instance-based
     - Greedy channel substitution for multivariate TS
     - Ates et al., 2021
   * - ``TSEvo``
     - Evolutionary
     - Multi-objective optimization via NSGA-II
     - Hollig et al., 2022
   * - ``Glacier``
     - Gradient-based
     - Gradient optimization with importance-weighted proximity
     - Wang et al., 2024
   * - ``LatentCF`` (LatentCF++)
     - Gradient-based
     - Latent space optimization with local/global weighting
     - Wang et al., 2021

Evaluation Metrics
~~~~~~~~~~~~~~~~~~

TSCFEval implements 10 metrics organized into six quality dimensions:

.. list-table::
   :header-rows: 1
   :widths: 20 20 35 25

   * - Dimension
     - Metric
     - Description
     - Direction
   * - Core Quality
     - ``Validity``
     - Fraction of CFs that flip the prediction
     - maximize
   * - Core Quality
     - ``Proximity``
     - Closeness to original instance (L1, L2, L-inf)
     - maximize
   * - Core Quality
     - ``Sparsity``
     - Fraction of changed features
     - minimize
   * - Distribution
     - ``Plausibility``
     - Whether CFs lie within data distribution (LOF, IF, MP-OCSVM)
     - maximize
   * - Distribution
     - ``Diversity``
     - Variety among multiple CFs (DPP-based)
     - maximize
   * - Structure
     - ``Contiguity``
     - How contiguous the edits are
     - maximize
   * - Structure
     - ``Composition``
     - Number and length of edit segments
     - minimize
   * - Model Behavior
     - ``Confidence``
     - Model confidence on original and CF predictions
     - maximize
   * - Stability
     - ``Robustness``
     - Local Lipschitz-like stability to input perturbations
     - minimize
   * - Performance
     - ``Efficiency``
     - Generation time per instance
     - minimize

Quick Start
-----------

Evaluating Counterfactuals
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from tscf_eval import Evaluator, Validity, Proximity, Sparsity

   # Your data
   X = np.random.randn(10, 100)       # Original instances
   X_cf = X + np.random.randn(10, 100) * 0.1  # Counterfactuals
   y = np.zeros(10)                   # Original labels
   y_cf = np.ones(10)                 # CF labels

   # Create evaluator
   evaluator = Evaluator([
       Validity(),
       Proximity(p=2),
       Sparsity(),
   ])

   # Evaluate
   results = evaluator.evaluate(X, X_cf, y=y, y_cf=y_cf)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/evaluator
   api/counterfactuals
   api/data_loader
   api/benchmark

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
