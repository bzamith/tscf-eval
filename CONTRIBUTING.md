# Contributing to TSCF-Eval

Thank you for your interest in contributing to TSCF-Eval! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/tscf-eval.git
   cd tscf-eval
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/bzamith/tscf-eval.git
   ```

## Development Setup

### Prerequisites

- Python 3.10-3.13
- Git

### Installation

1. **Create a virtual environment**:
   ```bash
   make venv
   ```

2. **Install development dependencies**:
   ```bash
   make install-dev
   ```

3. **Install pre-commit hooks**:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Verify Setup

Run the test suite to verify everything is working:
```bash
make test
```

## Making Changes

### Branch Naming

Create a descriptive branch name:
- `feature/add-new-metric` - For new features
- `fix/proximity-edge-case` - For bug fixes
- `docs/update-readme` - For documentation
- `refactor/evaluator-cleanup` - For refactoring

### Workflow

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** with clear, atomic commits

4. **Run checks before committing**:
   ```bash
   make check  # Runs format, lint, typecheck
   make test   # Runs tests
   ```

## Code Style

We use automated tools to maintain consistent code style:

### Formatting

- **Ruff** for formatting (line length: 100)
- Run: `make format`

### Linting

- **Ruff** for linting
- **Pylint** for additional checks
- Run: `make lint`

### Type Hints

- All public functions should have type hints
- Use `from __future__ import annotations` for modern syntax
- Run: `make typecheck`

### Docstrings

Follow NumPy docstring style:

```python
def compute_metric(X: np.ndarray, X_cf: np.ndarray) -> float:
    """Compute the evaluation metric.

    Parameters
    ----------
    X : np.ndarray
        Original instances, shape ``(N, T)`` or ``(N, C, T)``.
    X_cf : np.ndarray
        Counterfactual instances, same shape as ``X``.

    Returns
    -------
    float
        Metric value in range ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``X`` and ``X_cf`` have different shapes.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.zeros((10, 50))
    >>> X_cf = np.ones((10, 50))
    >>> compute_metric(X, X_cf)
    0.5
    """
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_metrics.py -v

# Run specific test
pytest tests/test_metrics.py::TestValidity::test_with_labels -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Use pytest fixtures from `conftest.py`
- Aim for >60% code coverage

Example test:

```python
import pytest
import numpy as np
from tscf_eval.evaluator import Proximity

class TestProximity:
    """Tests for the Proximity metric."""

    def test_name(self):
        """Test metric name."""
        assert Proximity(p=2).name() == "proximity_l2"

    def test_identical_inputs(self, counterfactual_pair):
        """Test that identical inputs give zero distance."""
        X, _, _, _ = counterfactual_pair
        metric = Proximity(p=2)
        assert metric.compute(X, X) == 0.0
```

## Documentation

### Building Docs

```bash
cd docs
make html
```

### Docstring Coverage

All public modules, classes, and functions should have docstrings.

## Submitting Changes

### Pull Request Process

1. **Update your branch** with the latest main:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Open a Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what and why
   - Reference to any related issues

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] New code has appropriate test coverage
- [ ] Documentation is updated (if applicable)
- [ ] CHANGELOG.md is updated (for user-facing changes)
- [ ] Commit messages are clear and descriptive

### Review Process

1. Maintainers will review your PR
2. Address any feedback
3. Once approved, your PR will be merged

## Release Process

Releases are managed by maintainers following semantic versioning:

- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Version Bumping

1. Update version in `pyproject.toml`
2. Update version in `src/tscf_eval/__init__.py`
3. Update `CHANGELOG.md`
4. Create a git tag: `git tag v0.2.0`
5. Push tag: `git push origin v0.2.0`

The CI/CD pipeline will automatically build and publish to PyPI.

## Adding New Features

### Adding a New Metric

1. Create a new class in `src/tscf_eval/evaluator/metrics.py`:
   ```python
   class NewMetric(Metric):
       def name(self) -> str:
           return "new_metric"

       def compute(self, X, X_cf, **kwargs) -> float:
           # Implementation
           pass
   ```

2. Export in `src/tscf_eval/evaluator/__init__.py`

3. Export in `src/tscf_eval/__init__.py`

4. Add tests in `tests/test_metrics.py`

5. Update documentation

### Adding a New Counterfactual Method

1. Create a new class in `src/tscf_eval/counterfactuals/`:
   ```python
   from .base import Counterfactual

   class NewMethod(Counterfactual):
       def explain(self, x, y_pred=None):
           # Implementation
           pass
   ```

2. Export in `src/tscf_eval/counterfactuals/__init__.py`

3. Add tests in `tests/test_counterfactuals.py`

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas

Thank you for contributing!
