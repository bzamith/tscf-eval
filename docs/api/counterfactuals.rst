Counterfactuals Module
======================

The counterfactuals module provides implementations of counterfactual
explanation algorithms for time series classification.

All implementations wrap existing methods from the literature and provide
a unified interface for benchmarking and evaluation.

Base Class
----------

.. autoclass:: tscf_eval.Counterfactual
   :members:
   :undoc-members:
   :show-inheritance:

Implementations
---------------

CoMTE
~~~~~

Counterfactual explanations for Multivariate Time series using greedy channel
substitution from distractor series. Based on Ates et al. (2021).

.. autoclass:: tscf_eval.COMTE
   :members:
   :undoc-members:
   :show-inheritance:

NativeGuide
~~~~~~~~~~~

Instance-based counterfactual explanations using nearest unlike neighbor guidance
with DTW barycenter averaging. Based on Delaney et al. (2021).

.. autoclass:: tscf_eval.NativeGuide
   :members:
   :undoc-members:
   :show-inheritance:

TSEvo
~~~~~

Evolutionary counterfactual generation using multi-objective optimization (NSGA-II)
with three mutation strategies: authentic, frequency, and gaussian.
Based on Höllig et al. (2022).

.. autoclass:: tscf_eval.TSEvo
   :members:
   :undoc-members:
   :show-inheritance:

Glacier
~~~~~~~

Gradient-based counterfactual generation with guided locally constrained
optimization using importance-weighted proximity. Based on Wang et al. (2024).

.. autoclass:: tscf_eval.Glacier
   :members:
   :undoc-members:
   :show-inheritance:

LatentCF++
~~~~~~~~~~

Gradient-based counterfactual generation with importance-weighted proximity
constraints, optimizing directly in the input space. Based on Wang et al. (2021).

.. autoclass:: tscf_eval.LatentCF
   :members:
   :undoc-members:
   :show-inheritance:

References
----------

The counterfactual methods implemented in this module are based on the following papers:

- Ates, E., Aksar, B., Leung, V. J., & Coskun, A. K. (2021). "Counterfactual Explanations for Multivariate Time Series." In *Proceedings of the 2021 International Conference on Applied Artificial Intelligence (ICAPAI)*, pp. 1-8. `[Paper] <https://doi.org/10.1109/ICAPAI49758.2021.9462056>`_ `[Code] <https://github.com/peaclab/CoMTE>`_
- Delaney, E., Greene, D., & Keane, M. T. (2021). "Instance-Based Counterfactual Explanations for Time Series Classification." In *Case-Based Reasoning Research and Development (ICCBR 2021)*, pp. 32-47. Springer. `[Paper] <https://doi.org/10.1007/978-3-030-86957-1_3>`_ `[Code] <https://github.com/e-delaney/Instance-Based_CFE_TSC>`_
- Höllig, J., Kulbach, C., & Thoma, S. (2022). "TSEvo: Evolutionary Counterfactual Explanations for Time Series Classification." In *Proceedings of the 21st IEEE International Conference on Machine Learning and Applications (ICMLA 2022)*, pp. 29-36. `[Paper] <https://doi.org/10.1109/ICMLA55696.2022.00013>`_ `[Code] <https://github.com/JHoelli/TSEvo>`_
- Wang, Z., Samsten, I., Miliou, I., Mochaourab, R., & Papapetrou, P. (2024). "Glacier: Guided Locally Constrained Counterfactual Explanations for Time Series Classification." *Machine Learning*, 113(3). `[Paper] <https://doi.org/10.1007/s10994-023-06502-x>`_ `[Code] <https://github.com/zhendong3wang/learning-time-series-counterfactuals>`_
- Wang, Z., Samsten, I., Mochaourab, R., & Papapetrou, P. (2021). "Learning Time Series Counterfactuals via Latent Space Representations." In *International Conference on Discovery Science (DS 2021)*, Lecture Notes in Computer Science, vol 12986, pp. 369-384. Springer. `[Paper] <https://doi.org/10.1007/978-3-030-88942-5_29>`_ `[Code] <https://github.com/zhendong3wang/learning-time-series-counterfactuals>`_

The implementations also use TSInterpret as a foundation:

- Hollig, J., Kulbach, C., & Thoma, S. (2023). "TSInterpret: A Python Package for the Interpretability of Time Series Classification." *Journal of Open Source Software*, 8(85), 5220. `[Paper] <https://doi.org/10.21105/joss.05220>`_
