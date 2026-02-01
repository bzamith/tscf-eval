Evaluator Module
================

The evaluator module provides metrics and orchestration for assessing
counterfactual quality.

The metrics implemented follow established counterfactual evaluation literature,
with core metrics based on Wachter et al. (2017) and Mothilal et al. (2020).

Evaluator Class
---------------

.. autoclass:: tscf_eval.Evaluator
   :members:
   :undoc-members:
   :show-inheritance:

Metric Base Class
-----------------

.. autoclass:: tscf_eval.evaluator.base.Metric
   :members:
   :undoc-members:
   :show-inheritance:

Built-in Metrics
----------------

Validity
~~~~~~~~

Fraction of counterfactuals that change the model prediction.
Based on Li et al. (2023).

.. autoclass:: tscf_eval.Validity
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: direction

Proximity
~~~~~~~~~

Proximity score between original and counterfactual instances, computed as
``1 / (1 + d)`` where ``d`` is the L-p distance. Higher values indicate
counterfactuals closer to the originals.
Based on Delaney et al. (2021) and Bahri et al. (2022).

.. autoclass:: tscf_eval.Proximity
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: direction

Sparsity
~~~~~~~~

Fraction of features/time-points changed between original and counterfactual.
Lower values indicate sparser (more targeted) edits.
Based on Mothilal et al. (2020).

.. autoclass:: tscf_eval.Sparsity
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: direction

Plausibility
~~~~~~~~~~~~

Whether counterfactuals lie within the training data distribution,
scored via outlier detection. Supports three backends:
LOF (Breunig et al., 2000), Isolation Forest (Liu et al., 2008),
and Matrix Profile + OneClassSVM (Yeh et al., 2016).

.. autoclass:: tscf_eval.Plausibility
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: direction

Diversity
~~~~~~~~~

Diversity among multiple counterfactuals for the same query, using a
DPP-inspired log-determinant measure. Based on Mothilal et al. (2020)
and Kulesza & Taskar (2012).

.. autoclass:: tscf_eval.Diversity
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: direction

Controllability
~~~~~~~~~~~~~~~

Ease of reverting counterfactual changes via single-feature edits.
Based on Verma et al. (2024).

.. autoclass:: tscf_eval.Controllability
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: direction

Confidence
~~~~~~~~~~

Model confidence (maximum predicted probability) statistics for
original and counterfactual instances.
Based on Le et al. (2023).

.. autoclass:: tscf_eval.Confidence
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: direction

Composition
~~~~~~~~~~~

Segment-based statistics measuring contiguous runs of edits, relevant
for time series interpretability.
Based on Delaney et al. (2021) and Ates et al. (2021).

.. autoclass:: tscf_eval.Composition
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: direction

Contiguity
~~~~~~~~~~

Scalar measure of how concentrated edits are in a single block.
Based on Delaney et al. (2021) and Ates et al. (2021).

.. autoclass:: tscf_eval.Contiguity
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: direction

Robustness
~~~~~~~~~~

Local Lipschitz-like stability estimate using k-nearest neighbor
analysis. Based on Ates et al. (2021).

.. autoclass:: tscf_eval.Robustness
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: direction

Efficiency
~~~~~~~~~~

Mean per-instance generation time.
Based on Li et al. (2023).

.. autoclass:: tscf_eval.Efficiency
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: direction

References
----------

The metrics in this module are based on the following works:

**Validity**

- Li, P., Bahri, O., Boubrahimi, S. F., & Hamdi, S. M. (2023). "CELS: Counterfactual Explanations for Time Series Data via Learned Saliency Maps." In *2023 IEEE International Conference on Big Data (BigData)*, pp. 718-727. `[Paper] <https://doi.org/10.1109/BigData59044.2023.10386229>`_

**Proximity**

- Delaney, E., Greene, D., & Keane, M. T. (2021). "Instance-Based Counterfactual Explanations for Time Series Classification." *arXiv:2009.13211*. `[Paper] <https://arxiv.org/abs/2009.13211>`_
- Bahri, O., Boubrahimi, S. F., & Hamdi, S. M. (2022). "Shapelet-Based Counterfactual Explanations for Multivariate Time Series."

**Sparsity**

- Mothilal, R. K., Sharma, A., & Tan, C. (2020). "Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations." In *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency (FAT* '20)*, pp. 607-617. `[Paper] <https://doi.org/10.1145/3351095.3372850>`_

**Plausibility**

- Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). "LOF: Identifying Density-Based Local Outliers." In *Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data*, pp. 93-104. `[Paper] <https://doi.org/10.1145/342009.335388>`_
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). "Isolation Forest." In *2008 Eighth IEEE International Conference on Data Mining*, pp. 413-422. `[Paper] <https://doi.org/10.1109/ICDM.2008.17>`_
- Yeh, C.-C. M., Zhu, Y., Ulanova, L., Begum, N., Ding, Y., Dau, H. A., Silva, D. F., Mueen, A., & Keogh, E. (2016). "Matrix Profile I: All Pairs Similarity Joins for Time Series." In *2016 IEEE 16th International Conference on Data Mining (ICDM)*, pp. 1317-1322. `[Paper] <https://doi.org/10.1109/ICDM.2016.0179>`_

**Diversity**

- Mothilal, R. K., Sharma, A., & Tan, C. (2020). "Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations." In *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency (FAT* '20)*, pp. 607-617. `[Paper] <https://doi.org/10.1145/3351095.3372850>`_
- Kulesza, A., & Taskar, B. (2012). "Determinantal Point Processes for Machine Learning." *Foundations and Trends in Machine Learning*, 5(2-3), 123-286. `[Paper] <https://doi.org/10.1561/2200000044>`_

**Confidence**

- Le, T., Miller, T., Singh, R., & Sonenberg, L. (2023). "Explaining model confidence using counterfactuals." In *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(10), pp. 12101-12109. `[Paper] <https://doi.org/10.1609/aaai.v37i10.26399>`_

**Controllability**

- Verma, S., Boonsanong, V., Hoang, M., Hines, K. E., Dickerson, J. P., & Shah, C. (2024). "Counterfactual Explanations for Machine Learning: Challenges Revisited." *ACM Computing Surveys*, 56(12), Article 304. `[Paper] <https://doi.org/10.1145/3677119>`_

**Composition, Contiguity**

- Delaney, E., Greene, D., & Keane, M. T. (2021). "Instance-Based Counterfactual Explanations for Time Series Classification." In *Case-Based Reasoning Research and Development (ICCBR 2021)*, pp. 32-47. Springer. `[Paper] <https://doi.org/10.1007/978-3-030-86957-1_3>`_
- Ates, E., Aksar, B., Leung, V. J., & Coskun, A. K. (2021). "Counterfactual Explanations for Multivariate Time Series." In *Proceedings of the 2021 International Conference on Applied Artificial Intelligence (ICAPAI)*, pp. 1-8. `[Paper] <https://doi.org/10.1109/ICAPAI49758.2021.9462056>`_

**Robustness**

- Ates, E., Aksar, B., Leung, V. J., & Coskun, A. K. (2021). "Counterfactual Explanations for Multivariate Time Series." In *Proceedings of the 2021 International Conference on Applied Artificial Intelligence (ICAPAI)*, pp. 1-8. `[Paper] <https://doi.org/10.1109/ICAPAI49758.2021.9462056>`_

**Efficiency**

- Li, P., Bahri, O., Boubrahimi, S. F., & Hamdi, S. M. (2023). "Attention-Based Counterfactual Explanation for Multivariate Time Series." In *Data Warehousing and Knowledge Discovery (DaWaK 2023)*, Lecture Notes in Computer Science, vol 14148, pp. 287-293. Springer. `[Paper] <https://doi.org/10.1007/978-3-031-39831-5_26>`_
