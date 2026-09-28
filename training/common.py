"""Shared training plumbing: paths, PPO construction, logging helpers.

Keeps ``train_duopoly.py`` and ``train_monopoly.py`` short and identical in
how they build models, where they save them, and how they log.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import config

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure console logging once for CLI entry points."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch, and pin torch to the configured threads."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.TORCH_THREADS)


def results_path(results_dir: Path | None, *parts: str) -> Path:
    """Path under the results directory (default ``config.RESULTS_DIR``)."""
    return Path(results_dir or config.RESULTS_DIR).joinpath(*parts)


def model_file(results_dir: Path | None, mode: str, tag: str, platform: str) -> Path:
    """Where a model lives, e.g. ``results/models/duopoly/final/platform_A/model.zip``."""
    return results_path(results_dir, config.MODELS_SUBDIR, mode, tag, f"platform_{platform}", config.MODEL_FILENAME)


def make_ppo(env: gym.Env, seed: int, results_dir: Path | None) -> PPO:
    """Independent PPO learner: default ``MlpPolicy`` (2x64), random init, CPU."""
    return PPO(
        "MlpPolicy",
        Monitor(env),
        seed=seed,
        device="cpu",
        verbose=0,
        tensorboard_log=str(results_path(results_dir, config.LOGS_SUBDIR, "tensorboard")),
        **config.PPO_KWARGS,
    )


def save_model(model: PPO, path: Path) -> None:
    """Save a model zip, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def log_scalars(model: PPO, values: dict[str, float]) -> None:
    """Write evaluation scalars to the model's TensorBoard run."""
    for key, value in values.items():
        model.logger.record(key, value)
    model.logger.dump(model.num_timesteps)


def append_training_log(rows: list[dict], results_dir: Path | None) -> Path:
    """Append rows to ``results/csv/training_log.csv`` (header written once)."""
    path = results_path(results_dir, config.CSV_SUBDIR, config.TRAINING_LOG_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)
    return path


def clear_training_log(results_dir: Path | None, mode: str) -> None:
    """Drop earlier rows for ``mode`` so a rerun does not mix old and new rounds."""
    path = results_path(results_dir, config.CSV_SUBDIR, config.TRAINING_LOG_CSV)
    if path.exists():
        df = pd.read_csv(path)
        df[df["mode"] != mode].to_csv(path, index=False)
