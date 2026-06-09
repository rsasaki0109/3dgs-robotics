"""Configuration loading and management.

This module handles loading YAML configuration files for datasets and
training hyperparameters, merging them with CLI overrides, and providing
a unified config object for the pipeline.

Expected config files:
- configs/datasets.yaml: Dataset metadata (URLs, descriptions, splits)
- configs/training.yaml: 3DGS training hyperparameters
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return the project root directory (where pyproject.toml lives)."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found in parents).")


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_datasets_config() -> dict[str, Any]:
    """Load the datasets configuration from configs/datasets.yaml.

    Returns:
        Dictionary of dataset configurations keyed by dataset name.
    """
    root = get_project_root()
    return load_config(root / "configs" / "datasets.yaml")


# Mirrors configs/training.yaml so pip-installed packages (no repo checkout,
# e.g. the Hugging Face Space / Colab) can train without the YAML file.
_DEFAULT_TRAINING_CONFIG: dict[str, Any] = {
    "num_iterations": 30000,
    "batch_size": 1,
    "seed": 42,
    "learning_rate": {
        "position": 0.00016,
        "feature": 0.0025,
        "opacity": 0.05,
        "scaling": 0.005,
        "rotation": 0.001,
    },
    "lr_schedule": {
        "position_lr_init": 0.00016,
        "position_lr_final": 0.0000016,
        "position_lr_delay_mult": 0.01,
        "position_lr_max_steps": 30000,
    },
    "sh_degree": 3,
    "densify_from_iter": 500,
    "densify_until_iter": 15000,
    "densify_interval": 100,
    "densify_grad_threshold": 0.0002,
    "opacity_reset_interval": 3000,
    "min_opacity": 0.005,
    "percent_dense": 0.01,
    "max_screen_size": 20,
    "lambda_dssim": 0.2,
    "save_iterations": [7000, 15000, 30000],
    "test_iterations": [7000, 15000, 30000],
    "resolution": -1,
    "white_background": False,
}


def load_training_config() -> dict[str, Any]:
    """Load the training configuration from configs/training.yaml.

    Falls back to the built-in defaults when the repo checkout (and thus
    configs/training.yaml) is not available, e.g. for pip installs.

    Returns:
        Dictionary of training hyperparameters.
    """
    try:
        root = get_project_root()
        return load_config(root / "configs" / "training.yaml")
    except (RuntimeError, FileNotFoundError):
        return dict(_DEFAULT_TRAINING_CONFIG)


def get_dataset_config(name: str) -> dict[str, Any]:
    """Get configuration for a specific dataset by name.

    Args:
        name: Dataset identifier (e.g. "covla", "mcd", "autoware_leo_drive_bagN").

    Returns:
        Dictionary of dataset configuration.

    Raises:
        ValueError: If the dataset name is not found in the configuration.
    """
    datasets = load_datasets_config()
    if name not in datasets:
        available = ", ".join(sorted(datasets.keys()))
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")
    return datasets[name]
