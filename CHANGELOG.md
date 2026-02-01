# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-31

Initial release of TSCFEval.

### Added

- **Counterfactual Explainers** (5 methods):
  - `COMTE`: Counterfactual Multivariate Time-series Explanations using greedy
    channel substitution (Ates et al., 2021)
  - `NativeGuide`: Instance-based counterfactual explanations using nearest
    unlike neighbor guidance (Delaney et al., 2021)
  - `TSEvo`: Multi-objective evolutionary optimization using NSGA-II with
    three mutation strategies (Hollig et al., 2022)
  - `Glacier`: Gradient-based optimization with guided locally constrained
    proximity (Wang et al., 2024)
  - `LatentCF`: Gradient-based optimization with importance-weighted proximity
    constraints (Wang et al., 2021)

- **Evaluation Metrics** (10 metrics in 6 quality dimensions):
  - Core Quality: `Validity`, `Proximity` (L1, L2, L-inf), `Sparsity`
  - Distribution Alignment: `Plausibility` (LOF, IF, MP-OCSVM), `Diversity` (DPP)
  - Structural Properties: `Contiguity`, `Composition`
  - Model Behavior: `Confidence` (orig, cf, delta)
  - Stability: `Robustness` (Lipschitz)
  - Computational Performance: `Efficiency` (timing)

- **Benchmarking Framework**:
  - `BenchmarkRunner` with three evaluation scenarios (single dataset/multiple
    methods, single dataset/multiple models, multiple datasets/fixed model)
  - `BenchmarkResults` container with filter, aggregate, and serialization
  - Confidence-stratified instance selection covering high- and low-confidence predictions
  - `ParetoAnalyzer` for multi-criteria dominance analysis
  - `WeightedScalarizer` for customizable metric aggregation with sensitivity analysis
  - `friedman_test` for statistical comparison across datasets
  - LaTeX table export for publication-ready results

- **Data Loaders**:
  - `UCRLoader`: UCR Time Series Archive loader
  - `FileLoader`: CSV and Excel file loaders
  - `TSCData`: Immutable data container

[0.1.0]: https://github.com/bzamith/tscf-eval/releases/tag/v0.1.0
