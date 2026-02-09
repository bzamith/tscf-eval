# TSCFEval

[![PyPI version](https://badge.fury.io/py/tscf-eval.svg)](https://badge.fury.io/py/tscf-eval)
[![Python 3.10-3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TSCFEval** is a model-agnostic Python framework for systematic evaluation of counterfactual explanations in Time Series Classification (TSC). Unlike existing libraries that focus on counterfactual generation, TSCFEval is specifically designed for counterfactual evaluation, consolidating fragmented evaluation practices from the TSC counterfactual literature into a unified, extensible toolkit.

Given a time series classifier and a set of counterfactual explanations, TSCFEval provides:
- **11 evaluation metrics** organized into **six quality dimensions** (core quality, distribution alignment, structural properties, model behavior, stability, and computational performance)
- **Weighted scalarization** for aggregating metrics into composite scores, enabling customizable method ranking
- **Confidence-stratified instance selection** for benchmarking across the decision boundary
- **Three benchmarking scenarios**: single dataset with multiple CF methods, single dataset with multiple classifiers, and multiple datasets with a fixed classifier
- **7 built-in CF methods** for generating counterfactuals
- **Multi-criteria analysis** for principled multi-criteria comparison

## Table of Contents

- [Installation](#installation)
- [Available Methods and Metrics](#available-methods-and-metrics)
- [Evaluating Counterfactuals](#evaluating-counterfactuals)
- [Benchmarking Multiple Methods](#benchmarking-multiple-methods)
- [Weighted Scalarization](#weighted-scalarization)
- [Adding Your Own CF Method](#adding-your-own-cf-method)
- [Citation](#citation)
- [References](#references)

---

## Installation

```bash
pip install tscf-eval
```

With optional dependencies:

```bash
pip install tscf-eval[dtw]   # DTW distance support
pip install tscf-eval[full]  # All features
```

From source:

```bash
git clone https://github.com/bzamith/tscf-eval.git
cd tscf-eval && pip install -e .
```

---

## Available Methods and Metrics

### Counterfactual Methods

| Method | Strategy | Description | Reference |
|--------|----------|-------------|-----------|
| `CELS` | Saliency map | Learned saliency map blending with nearest unlike neighbor | Li et al., 2023 |
| `NativeGuide` | Instance-based | Nearest unlike neighbor guidance with four variants (blend, ng, dtw_dba, cam) | Delaney et al., 2021 |
| `COMTE` | Instance-based | Greedy channel substitution for multivariate TS | Ates et al., 2021 |
| `SETS` | Shapelet-based | Class-specific shapelet manipulation with contiguous perturbations | Bahri et al., 2022 |
| `TSEvo` | Evolutionary | Multi-objective optimization via NSGA-II with three mutation operators | Hollig et al., 2022 |
| `Glacier` | Gradient-based | Gradient optimization with importance-weighted proximity constraints | Wang et al., 2024 |
| `LatentCF` | Gradient-based | Latent space optimization with local/global importance weighting | Wang et al., 2021 |

### Evaluation Metrics

TSCFEval implements 11 metrics organized into six quality dimensions:

| Dimension | Metric | Description | Direction | Reference |
|-----------|--------|-------------|-----------|-----------|
| **Core Quality** | `Validity` | Fraction of CFs that flip the prediction (hard or soft mode) | maximize | Li et al., 2023 |
| | `Proximity` | Closeness to original instance (L1, L2, L-inf, DTW) | maximize | Delaney et al., 2021; Bahri et al., 2022 |
| | `Sparsity` | Fraction of changed features | minimize | Mothilal et al., 2020 |
| **Distribution** | `Plausibility` | Whether CFs lie within data distribution (LOF, IF, MP-OCSVM, DTW-LOF) | maximize | Breunig et al., 2000; Liu et al., 2008 |
| | `Diversity` | Variety among multiple CFs via DPP (Euclidean or DTW) | maximize | Mothilal et al., 2020 |
| **Structure** | `Contiguity` | How contiguous the edits are | maximize | Delaney et al., 2021; Ates et al., 2021 |
| | `Composition` | Number and length of edit segments | minimize | Delaney et al., 2021; Ates et al., 2021 |
| **Model Behavior** | `Confidence` | Model confidence on original and CF predictions | maximize | Le et al., 2023 |
| | `Controllability` | Ease of reverting CF changes via single-feature edits | maximize | Verma et al., 2024 |
| **Stability** | `Robustness` | Local Lipschitz-like stability to input perturbations (Euclidean or DTW) | minimize | Ates et al., 2021 |
| **Performance** | `Efficiency` | Generation time per instance | minimize | Li et al., 2023 |

---

## Evaluating Counterfactuals

### Simple Evaluation

```python
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

# Evaluate counterfactual quality
evaluator = Evaluator([
    Validity(),
    Proximity(p=2, distance="lp"),
    Proximity(distance="dtw"),
    Sparsity(),
])
results = evaluator.evaluate(X, X_cf, y=y, y_cf=y_cf)
print(f"Validity: {results['validity_soft']:.2f}")
print(f"Proximity (L2): {results['proximity_l2']:.2f}")
print(f"Proximity (DTW): {results['proximity_dtw']:.2f}")
print(f"Sparsity: {results['sparsity']:.2f}")
```

### Using Built-in CF Methods

```python
from tscf_eval import COMTE, NativeGuide
from tscf_eval.data_loader import UCRLoader
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
loader = UCRLoader("ItalyPowerDemand")
train = loader.load("train")
test = loader.load("test")

# Train classifier
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(train.X, train.y)

# Generate counterfactual with CoMTE
comte = COMTE(
    model=clf,
    data=(train.X, train.y),
    distance="dtw",
)

cf, cf_label, meta = comte.explain(test.X[0])
print(f"CF label: {cf_label}, Edits: {meta['edits_variables']}")
```

---

## Benchmarking Multiple Methods

TSCFEval supports three benchmarking scenarios for systematic evaluation:

1. **Single dataset, multiple CF methods** -- compare explainer algorithms on a fixed dataset and model
2. **Single dataset, multiple classifiers** -- study how the classifier affects CF quality
3. **Multiple datasets, fixed classifier** -- assess generalization across datasets

```python
from tscf_eval import Evaluator, Validity, Proximity, Sparsity, COMTE, NativeGuide, TSEvo
from tscf_eval.benchmark import BenchmarkRunner, DatasetConfig, ModelConfig, ExplainerConfig
from tscf_eval.data_loader import UCRLoader

# Load data
loader = UCRLoader("ArrowHead")
train, test = loader.load("train"), loader.load("test")

# Train a classifier
from aeon.classification.convolution_based import RocketClassifier
clf = RocketClassifier(n_kernels=500, random_state=42)
clf.fit(train.X, train.y)

# Configure benchmark
runner = BenchmarkRunner(
    datasets=[DatasetConfig("ArrowHead", train.X, train.y, test.X, test.y)],
    models=[ModelConfig("rocket", clf)],
    explainers=[
        ExplainerConfig("comte_dtw", COMTE, {"distance": "dtw"}),
        ExplainerConfig("ng_blend", NativeGuide, {"method": "blend"}),
        ExplainerConfig("tsevo", TSEvo, {"n_generations": 50}),
    ],
    evaluator=Evaluator([Validity(), Proximity(p=2, distance="lp"), Proximity(distance="dtw"), Sparsity()]),
    n_instances=12,
    instance_selection="stratified_confidence",  # Confidence-stratified selection
)

results = runner.run()

# Aggregate results by explainer
df = results.aggregate(by="explainer")
print(df)
```

### Pareto Analysis

Find Pareto-optimal methods balancing multiple objectives:

```python
from tscf_eval.benchmark import ParetoAnalyzer

analyzer = ParetoAnalyzer(metrics=[
    "validity_soft", "proximity_dtw", "sparsity",
])

# Dominance ranking
ranking = analyzer.dominance_ranking(results)
print(ranking)

# Visualize trade-offs
analyzer.plot_front(results, x_metric="validity_soft", y_metric="proximity_dtw")
```

---

## Weighted Scalarization

Since evaluation metrics often conflict (e.g., maximizing validity may reduce proximity), TSCFEval implements weighted sum scalarization for aggregating metrics into a composite score. Each metric is min-max normalized to [0, 1] with direction awareness, then combined via weighted sum:

```python
from tscf_eval.benchmark import WeightedScalarizer

# Equal-weight composite across all metrics
scalarizer = WeightedScalarizer(metrics=[
    "validity_soft", "proximity_dtw", "sparsity",
])
scores = scalarizer.score(results)
print(scores)  # Ranked by composite score

# Custom weights emphasizing validity
scalarizer = WeightedScalarizer(
    metrics=["validity_soft", "proximity_dtw", "sparsity"],
    weights={"validity_soft": 3.0, "proximity_dtw": 1.0, "sparsity": 1.0},
)

# Sensitivity analysis: how ranking changes as one metric's weight varies
sens_df = scalarizer.sensitivity(results, vary_metric="validity_soft", n_steps=11)
scalarizer.plot_sensitivity(sens_df)
```

---

## Adding Your Own CF Method

To integrate your own counterfactual method with TSCFEval, inherit from the `Counterfactual` base class and implement the `explain` method:

```python
from tscf_eval.counterfactuals.base import Counterfactual
import numpy as np

class MyCFMethod(Counterfactual):
    """Custom counterfactual generator."""

    def __init__(self, model, data, my_param=1.0):
        self.model = model
        self.X_ref, self.y_ref = data
        self.my_param = my_param

    def explain(self, x, y_pred=None):
        """Generate counterfactual for instance x.

        Parameters
        ----------
        x : np.ndarray
            Single instance, shape (T,) or (C, T).
        y_pred : int, optional
            Predicted label for x (computed if None).

        Returns
        -------
        cf : np.ndarray
            Counterfactual instance.
        cf_label : int
            Predicted label for the counterfactual.
        meta : dict
            Metadata about the generation process.
        """
        # Your CF generation logic here
        cf = x.copy()
        # ... modify cf ...

        cf_label = int(self.model.predict(cf[None, ...])[0])

        meta = {
            "method": "my_cf_method",
            "param_used": self.my_param,
        }

        return cf, cf_label, meta
```

Use your method with the evaluator and benchmarking tools:

```python
# Direct evaluation
my_method = MyCFMethod(model=clf, data=(train.X, train.y))
cf, label, meta = my_method.explain(test.X[0])

# In benchmarks
configs = [
    ExplainerConfig("my_method", MyCFMethod, {"my_param": 2.0}),
    ExplainerConfig("comte", COMTE, {}),
]
```

---

## Citation

If you use TSCFEval in your research, please cite our paper:

```bibtex
@inproceedings{santos2025tscfeval,
  title     = {{TSCFEval}: A Model-Agnostic Framework for Evaluating Time Series Classification Counterfactuals},
  author    = {Santos, Bruna Zamith and Lira, Maira Farias Andrade and Cerri, Ricardo and Prud{\^{e}}ncio, Ricardo Bastos Cavalcante},
  year      = {2025},
  note      = {Under review}
}
```

---

## References

### Counterfactual Methods

- **CELS**: Li, P., Tang, B., & Ning, Y. (2023). *CELS: Counterfactual Explanation of Time-Series via Learned Saliency Maps*. IEEE International Conference on Big Data 2023, pp. 1952-1957. [[Paper](https://doi.org/10.1109/BigData59044.2023.10386404)] [[Code](https://github.com/Luckilyeee/CELS)]

- **CoMTE**: Ates, E., Aksar, B., Leung, V. J., & Coskun, A. K. (2021). *Counterfactual Explanations for Multivariate Time Series*. ICAPAI 2021. [[Paper](https://doi.org/10.1109/ICAPAI49758.2021.9462056)] [[Code](https://github.com/peaclab/CoMTE)]

- **NativeGuide**: Delaney, E., Greene, D., & Keane, M. T. (2021). *Instance-Based Counterfactual Explanations for Time Series Classification*. ICCBR 2021. [[Paper](https://doi.org/10.1007/978-3-030-86957-1_3)] [[Code](https://github.com/e-delaney/Instance-Based_CFE_TSC)]

- **SETS**: Bahri, O., Filali Boubrahimi, S., & Hamdi, S. M. (2022). *Shapelet-Based Counterfactual Explanations for Multivariate Time Series*. KDD-MiLeTS 2022. [[Paper](https://arxiv.org/abs/2208.10462)] [[Code](https://github.com/omarbahri/SETS)]

- **TSEvo**: Hollig, J., Kulbach, C., & Thoma, S. (2022). *TSEvo: Evolutionary Counterfactual Explanations for Time Series Classification*. ICMLA 2022. [[Paper](https://doi.org/10.1109/ICMLA55696.2022.00013)] [[Code](https://github.com/JHoelli/TSEvo)]

- **Glacier**: Wang, Z., Samsten, I., Miliou, I., Mochaourab, R., & Papapetrou, P. (2024). *Glacier: Guided Locally Constrained Counterfactual Explanations for Time Series Classification*. Machine Learning, 113(3). [[Paper](https://doi.org/10.1007/s10994-023-06502-x)] [[Code](https://github.com/zhendong3wang/learning-time-series-counterfactuals)]

- **LatentCF++**: Wang, Z., Samsten, I., Mochaourab, R., & Papapetrou, P. (2021). *Learning Time Series Counterfactuals via Latent Space Representations*. DS 2021. [[Paper](https://doi.org/10.1007/978-3-030-88942-5_29)] [[Code](https://github.com/zhendong3wang/learning-time-series-counterfactuals)]

### Evaluation Metrics

- **Validity**: Li, P., Bahri, O., Boubrahimi, S. F., & Hamdi, S. M. (2023). *CELS: Counterfactual Explanations for Time Series Data via Learned Saliency Maps*. IEEE BigData 2023. [[Paper](https://doi.org/10.1109/BigData59044.2023.10386229)]

- **Proximity**: Delaney, E., Greene, D., & Keane, M. T. (2021). ICCBR 2021; Bahri, O., Boubrahimi, S. F., & Hamdi, S. M. (2022). *Shapelet-Based Counterfactual Explanations for Multivariate Time Series*.

- **Sparsity**: Mothilal, R. K., Sharma, A., & Tan, C. (2020). *Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations*. FAT* 2020. [[Paper](https://doi.org/10.1145/3351095.3372850)]

- **Plausibility (LOF)**: Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). *LOF: Identifying Density-Based Local Outliers*. ACM SIGMOD 2000. [[Paper](https://doi.org/10.1145/342009.335388)]

- **Plausibility (Isolation Forest)**: Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest*. ICDM 2008. [[Paper](https://doi.org/10.1109/ICDM.2008.17)]

- **Diversity**: Mothilal, R. K., Sharma, A., & Tan, C. (2020). FAT* 2020; Kulesza, A., & Taskar, B. (2012). *Determinantal Point Processes for Machine Learning*. [[Paper](https://doi.org/10.1561/2200000044)]

- **Composition, Contiguity**: Delaney et al. (2021). ICCBR 2021; Ates et al. (2021). ICAPAI 2021.

- **Confidence**: Le, T., Miller, T., Singh, R., & Sonenberg, L. (2023). *Explaining model confidence using counterfactuals*. AAAI 2023. [[Paper](https://doi.org/10.1609/aaai.v37i10.26399)]

- **Controllability**: Verma, S., Boonsanong, V., Hoang, M., Hines, K. E., Dickerson, J. P., & Shah, C. (2024). *Counterfactual Explanations for Machine Learning: Challenges Revisited*. ACM Computing Surveys, 56(12), Article 304. [[Paper](https://doi.org/10.1145/3677119)]

- **Robustness**: Ates et al. (2021). ICAPAI 2021.

- **Efficiency**: Li, P., Bahri, O., Boubrahimi, S. F., & Hamdi, S. M. (2023). *Attention-Based Counterfactual Explanation for Multivariate Time Series*. DaWaK 2023. [[Paper](https://doi.org/10.1007/978-3-031-39831-5_26)]

### Data and Tools

- **UCR Archive**: Dau, H. A., et al. (2019). *The UCR Time Series Archive*. IEEE/CAA Journal of Automatica Sinica, 6(6), 1293-1305. [[Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)]

- **aeon**: Middlehurst, M., et al. (2024). *aeon: a Python toolkit for learning from time series*. JMLR. [[Code](https://github.com/aeon-toolkit/aeon)]

---

## License

MIT License - see [LICENSE](LICENSE).
