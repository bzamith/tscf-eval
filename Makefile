.PHONY: help install install-dev install-full format lint typecheck check test test-cov clean build publish info venv docs pre-commit security

# Virtual environment
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Find compatible Python (3.10-3.13)
SYSTEM_PYTHON := $(shell \
	for py in python3.12 python3.11 python3.13 python3.10; do \
		if command -v $$py >/dev/null 2>&1; then \
			echo $$py; \
			exit 0; \
		fi; \
	done; \
	echo "python3" \
)

# Check Python version compatibility
check-python:
	@version=$$($(SYSTEM_PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"); \
	major=$$(echo $$version | cut -d. -f1); \
	minor=$$(echo $$version | cut -d. -f2); \
	if [ "$$major" -ne 3 ] || [ "$$minor" -lt 10 ] || [ "$$minor" -gt 13 ]; then \
		echo "Error: Python 3.10-3.13 required, found Python $$version"; \
		echo "Install a compatible version: brew install python@3.12 (macOS) or apt install python3.12 (Ubuntu)"; \
		exit 1; \
	fi; \
	echo "Using $(SYSTEM_PYTHON) (Python $$version)"

# Create virtual environment if it doesn't exist
$(VENV)/bin/activate: check-python
	@echo "Creating virtual environment with $(SYSTEM_PYTHON)..."
	$(SYSTEM_PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

venv: $(VENV)/bin/activate

# Default target
help:
	@echo "TSCF-Eval Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  make venv          - Create virtual environment"
	@echo "  make install       - Install package (production dependencies only)"
	@echo "  make install-dev   - Install package with dev dependencies"
	@echo "  make install-full  - Install package with all optional dependencies"
	@echo "  make pre-commit    - Install pre-commit hooks"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format        - Format code with ruff"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make typecheck     - Type check with mypy"
	@echo "  make security      - Security scan with bandit"
	@echo "  make check         - Run all checks (format, lint, typecheck, security)"
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run tests with pytest"
	@echo "  make test-cov      - Run tests with coverage (requires 60% minimum)"
	@echo "  make test-fast     - Run tests excluding slow tests"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          - Build Sphinx documentation"
	@echo "  make docs-serve    - Build and serve docs locally"
	@echo ""
	@echo "Build & Release:"
	@echo "  make build         - Build the package"
	@echo "  make publish       - Publish to PyPI (requires credentials)"
	@echo "  make publish-test  - Publish to TestPyPI"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean         - Remove build artifacts and cache"
	@echo "  make clean-all     - Remove build artifacts, cache, and virtual environment"
	@echo "  make info          - Show project information"

# Install package (production only)
install: $(VENV)/bin/activate
	@echo "Installing tscf-eval..."
	$(PIP) install -e .
	@echo "Done!"

# Install with dev dependencies
install-dev: $(VENV)/bin/activate
	@echo "Installing tscf-eval with dev dependencies..."
	$(PIP) install -e ".[dev]"
	@echo "Done!"

# Install with all optional dependencies
install-full: $(VENV)/bin/activate
	@echo "Installing tscf-eval with all dependencies..."
	$(PIP) install -e ".[full,dev,docs]"
	@echo "Done!"

# Install pre-commit hooks
pre-commit: $(VENV)/bin/activate
	@echo "Installing pre-commit hooks..."
	$(PIP) install pre-commit
	$(VENV)/bin/pre-commit install
	@echo "Pre-commit hooks installed!"

# Format code with ruff
format: $(VENV)/bin/activate
	@echo "Formatting code with ruff..."
	$(VENV)/bin/ruff format src/ tests/
	$(VENV)/bin/ruff check --fix src/ tests/
	@echo "Code formatted!"

# Lint code with ruff
lint: $(VENV)/bin/activate
	@echo "Linting code with ruff..."
	$(VENV)/bin/ruff check src/ tests/
	@echo "Linting complete!"

# Type check with mypy
typecheck: $(VENV)/bin/activate
	@echo "Type checking with mypy..."
	$(VENV)/bin/mypy src/tscf_eval/ --ignore-missing-imports
	@echo "Type check complete!"

# Security scan with bandit
security: $(VENV)/bin/activate
	@echo "Running security scan with bandit..."
	$(VENV)/bin/bandit -c pyproject.toml -r src/
	@echo "Security scan complete!"

# Run all checks (format, lint, typecheck, security)
check: format lint typecheck security
	@echo "All checks passed!"

# Run tests
test: $(VENV)/bin/activate
	@echo "Running tests..."
	$(VENV)/bin/pytest tests/ -v
	@echo "Tests completed!"

# Run tests with coverage
test-cov: $(VENV)/bin/activate
	@echo "Running tests with coverage..."
	$(VENV)/bin/pytest tests/ \
		--cov=src/tscf_eval \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-fail-under=70
	@echo "Tests completed! Coverage report available in htmlcov/index.html"

# Run fast tests (excluding slow tests)
test-fast: $(VENV)/bin/activate
	@echo "Running fast tests..."
	$(VENV)/bin/pytest tests/ -v -m "not slow"
	@echo "Fast tests completed!"

# Build documentation
docs: $(VENV)/bin/activate
	@echo "Building documentation..."
	cd docs && $(MAKE) html SPHINXBUILD=../$(VENV)/bin/sphinx-build
	@echo "Documentation built! Open docs/_build/html/index.html"

# Serve documentation locally
docs-serve: docs
	@echo "Serving documentation at http://localhost:8000..."
	$(PYTHON) -m http.server 8000 --directory docs/_build/html

# Build the package
build: clean $(VENV)/bin/activate
	@echo "Building package..."
	$(PIP) install build
	$(PYTHON) -m build
	@echo "Build complete! Distribution files in dist/"

# Publish to PyPI
publish: build
	@echo "Publishing to PyPI..."
	$(PIP) install twine
	$(PYTHON) -m twine upload dist/*
	@echo "Published!"

# Publish to TestPyPI
publish-test: build
	@echo "Publishing to TestPyPI..."
	$(PIP) install twine
	$(PYTHON) -m twine upload --repository testpypi dist/*
	@echo "Published to TestPyPI!"

# Clean build artifacts and cache
clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true
	rm -rf docs/_build/ 2>/dev/null || true
	@echo "Cleanup complete!"

# Clean everything including virtual environment
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@echo "Full cleanup complete!"

# Show project info
info: $(VENV)/bin/activate
	@echo "Project Information:"
	@echo "  Name: tscf-eval"
	@echo "  Python version: $$($(PYTHON) --version)"
	@$(PIP) show tscf-eval 2>/dev/null || echo "  Package: Not installed"
	@echo ""
	@echo "Installed dependencies:"
	@$(PIP) list | grep -E "numpy|pandas|scikit-learn|aeon|tqdm|tslearn|stumpy|scipy|ruff|mypy|pytest" || true
