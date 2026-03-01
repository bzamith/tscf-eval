"""Utility functions for benchmark notebooks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metrics: list[tuple[str, dict[str, Any] | None]],
) -> dict[str, float]:
    """
    Evaluate model with multiple metrics.

    Parameters
    ----------
    model : Any
        Trained classifier with ``score`` method.
    X : np.ndarray
        Test features.
    y : np.ndarray
        Test labels.
    metrics : list[tuple[str, dict[str, Any] | None]]
        List of ``(metric_name, metric_params)`` tuples.

    Returns
    -------
    dict[str, float]
        Dictionary mapping metric names to scores.
    """
    return {
        name: model.score(X, y, metric=name, metric_params=params)
        for name, params in metrics
    }


def sample_params(
    param_dist: dict[str, list[Any]],
    n_iter: int,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """
    Sample n_iter random parameter combinations from distributions.

    Parameters
    ----------
    param_dist : dict[str, list[Any]]
        Dictionary mapping parameter names to lists of possible values.
    n_iter : int
        Number of parameter combinations to sample.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    list[dict[str, Any]]
        List of sampled parameter combinations.
    """
    rng = random.Random(random_state)

    keys = list(param_dist.keys())
    sampled = []

    for _ in range(n_iter):
        params = {}
        for key in keys:
            params[key] = rng.choice(param_dist[key])
        sampled.append(params)

    return sampled


def hpo_random_search(
    classifier_class: type[Any],
    param_dist: dict[str, list[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 10,
    cv: int = 3,
    random_state: int = 42,
    verbose_model: bool = True,
) -> tuple[Any, dict[str, Any], float]:
    """
    Perform HPO using Random Search with cross-validation, optimizing for F1 score (macro).

    Parameters
    ----------
    classifier_class : type[Any]
        Classifier class to instantiate (e.g., ``RocketClassifier``).
    param_dist : dict[str, list[Any]]
        Dictionary mapping parameter names to lists of possible values.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    n_iter : int, default=10
        Number of random parameter combinations to try.
    cv : int, default=3
        Number of cross-validation folds.
    random_state : int, default=42
        Random seed for reproducibility.
    verbose_model : bool, default=True
        Whether to allow verbose output from models.

    Returns
    -------
    best_model : Any
        Best model refitted on full training data.
    best_params : dict[str, Any]
        Best parameter combination found.
    best_score : float
        Best cross-validation F1 score.
    """
    # Sample random parameter combinations
    param_combinations = sample_params(param_dist, n_iter, random_state)

    best_score: float = -1
    best_params: dict[str, Any] = {}

    # Cross-validation splitter
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    for idx, params in enumerate(param_combinations, 1):
        scores = []

        for train_idx, val_idx in cv_splitter.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            # Build model params
            model_params = {"random_state": random_state, **params}
            if not verbose_model:
                model_params["verbose"] = 0
            model = classifier_class(**model_params)
            model.fit(X_tr, y_tr)

            # Compute F1 score (macro)
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average="macro")
            scores.append(score)

        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

        print(f"  [{idx}/{n_iter}] F1={mean_score:.2%} (best={best_score:.2%})", end="\r", flush=True)

    print(" " * 60, end="\r", flush=True)  # clear progress line

    # Refit on full training data with best params
    model_params = {"random_state": random_state, **best_params}
    if not verbose_model:
        model_params["verbose"] = 0
    best_model = classifier_class(**model_params)
    best_model.fit(X_train, y_train)

    return best_model, best_params, best_score


def hpo_random_search_catch22(
    catch22_class: type[Any],
    param_dist: dict[str, list[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 10,
    cv: int = 3,
    random_state: int = 42,
) -> tuple[Any, dict[str, str], float]:
    """
    Random Search HPO for Catch22 (special handling for estimator objects).
    Optimizes for F1 score (macro).

    Parameters
    ----------
    catch22_class : type[Any]
        Catch22Classifier class.
    param_dist : dict[str, list[Any]]
        Dictionary with ``"estimator"`` key containing list of estimator objects.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    n_iter : int, default=10
        Number of random parameter combinations to try.
    cv : int, default=3
        Number of cross-validation folds.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    best_model : Any
        Best model refitted on full training data.
    best_params : dict[str, str]
        Best parameter combination found (with estimator name).
    best_score : float
        Best cross-validation F1 score.
    """
    # Sample random estimators
    rng = random.Random(random_state)
    sampled_estimators = [rng.choice(param_dist["estimator"]) for _ in range(n_iter)]

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    best_score: float = -1
    best_estimator: Any = sampled_estimators[0]

    for estimator in sampled_estimators:
        scores = []

        for train_idx, val_idx in cv_splitter.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            est_clone = clone(estimator)
            model = catch22_class(estimator=est_clone, random_state=random_state)
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average="macro")
            scores.append(score)

        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_estimator = estimator

    # Refit on full training data
    best_model = catch22_class(estimator=clone(best_estimator), random_state=random_state)
    best_model.fit(X_train, y_train)

    estimator_name = type(best_estimator).__name__

    # Build a descriptive string based on available attributes
    if hasattr(best_estimator, "n_estimators"):
        estimator_desc = f"{estimator_name}(n={best_estimator.n_estimators})"
    elif hasattr(best_estimator, "C"):
        estimator_desc = f"{estimator_name}(C={best_estimator.C})"
    else:
        estimator_desc = estimator_name

    return best_model, {"estimator": estimator_desc}, best_score


def save_models(
    models: dict[str, dict[str, Any]],
    hpo_results: dict[str, dict[str, dict[str, Any]]],
    model_types: list[str],
    dataset_names: list[str],
    output_dir: str | Path = "trained_models",
) -> Path:
    """
    Save trained models and HPO results to disk.

    Parameters
    ----------
    models : dict[str, dict[str, Any]]
        Nested dictionary: ``models[model_type][dataset_name] = model``.
    hpo_results : dict[str, dict[str, dict[str, Any]]]
        Nested dictionary: ``hpo_results[model_type][dataset_name] = {"params": ..., "cv_score": ...}``.
    model_types : list[str]
        List of model type names.
    dataset_names : list[str]
        List of dataset names.
    output_dir : str | Path, default="trained_models"
        Directory to save models to.

    Returns
    -------
    Path
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("Saving trained models...")
    print("=" * 80)

    for model_type in model_types:
        for dataset_name in dataset_names:
            model = models[model_type][dataset_name]
            model_path = output_dir / f"{model_type}_{dataset_name}.joblib"
            joblib.dump(model, model_path)
            print(f"  Saved: {model_path}")

    # Save HPO results as JSON
    hpo_results_path = output_dir / "hpo_results.json"

    # Convert hpo_results to JSON-serializable format
    hpo_results_serializable = {}
    for model_type in model_types:
        hpo_results_serializable[model_type] = {}
        for dataset_name in dataset_names:
            result = hpo_results[model_type][dataset_name]
            # Convert params to serializable format
            params_serializable = {}
            for k, v in result["params"].items():
                if hasattr(v, "__name__"):
                    params_serializable[k] = v.__name__
                else:
                    params_serializable[k] = v
            hpo_results_serializable[model_type][dataset_name] = {
                "params": params_serializable,
                "cv_score": float(result["cv_score"]),
            }

    with open(hpo_results_path, "w") as f:
        json.dump(hpo_results_serializable, f, indent=2)

    print(f"\nSaved HPO results: {hpo_results_path}")
    print(f"Total models saved: {len(model_types) * len(dataset_names)}")

    return output_dir


def load_models(
    model_types: list[str],
    dataset_names: list[str],
    models_dir: str | Path = "trained_models",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """
    Load previously trained models from disk.

    Parameters
    ----------
    model_types : list[str]
        List of model type names.
    dataset_names : list[str]
        List of dataset names.
    models_dir : str | Path, default="trained_models"
        Directory containing saved models.

    Returns
    -------
    models : dict[str, dict[str, Any]]
        Nested dictionary: ``models[model_type][dataset_name] = model``.
    hpo_results : dict[str, Any] | None
        HPO results if available, ``None`` otherwise.
    """
    models_dir = Path(models_dir)

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    loaded_models = {model_type: {} for model_type in model_types}

    for model_type in model_types:
        for dataset_name in dataset_names:
            model_path = models_dir / f"{model_type}_{dataset_name}.joblib"
            if model_path.exists():
                loaded_models[model_type][dataset_name] = joblib.load(model_path)
            else:
                print(f"  Warning: {model_path} not found")

    # Load HPO results
    hpo_path = models_dir / "hpo_results.json"
    loaded_hpo = None
    if hpo_path.exists():
        with open(hpo_path) as f:
            loaded_hpo = json.load(f)

    return loaded_models, loaded_hpo
