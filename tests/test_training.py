"""Smoke tests for training, snapshots and model loading (tiny settings)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO

import config
from envs.platform_env import PlatformEnv
from training.baselines import BaselinePolicy
from training.common import model_file
from training.policies import FrozenPolicy, trained_policies
from training.rollout import evaluate, eval_seeds
from training.train_duopoly import train_duopoly
from training.train_monopoly import train_monopoly

TINY_STEPS = config.PPO_KWARGS["n_steps"]  # one rollout per round


@pytest.fixture(scope="module")
def trained_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train tiny duopoly and monopoly models once for this module."""
    out = tmp_path_factory.mktemp("results")
    train_duopoly(rounds=1, steps=TINY_STEPS, seed=0, results_dir=out)
    train_monopoly(rounds=1, steps=TINY_STEPS, seed=0, results_dir=out)
    return out


def test_duopoly_smoke_writes_models_and_log(trained_dir: Path) -> None:
    for tag in ("untrained", "round_01", "final"):
        for name in config.PLATFORM_NAMES:
            assert model_file(trained_dir, "duopoly", tag, name).exists()
    log = pd.read_csv(trained_dir / config.CSV_SUBDIR / config.TRAINING_LOG_CSV)
    duo = log[log["mode"] == "duopoly"]
    assert set(duo["round"]) == {0, 1}
    assert set(duo["opponent"]) == {"baseline", "rival"}
    assert (trained_dir / config.LOGS_SUBDIR / "tensorboard").exists()


def test_monopoly_smoke_writes_model(trained_dir: Path) -> None:
    assert model_file(trained_dir, "monopoly", "final", config.MONOPOLY_NAME).exists()


def test_trained_policies_load_and_play(trained_dir: Path) -> None:
    for mode, n in (("duopoly", 2), ("monopoly", 1)):
        policies = trained_policies(mode, trained_dir)
        df = evaluate(policies, n, eval_seeds(2))
        assert len(df) == 2 and np.isfinite(df["total_welfare"]).all()


def test_missing_models_give_instructions(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train_duopoly"):
        trained_policies("duopoly", tmp_path)


def test_frozen_policy_matches_sb3_predict() -> None:
    env = PlatformEnv(0, {1: BaselinePolicy(2)}, 2, seed=0)
    model = PPO("MlpPolicy", env, seed=0, device="cpu")
    model.learn(256)  # move weights away from init so the check is meaningful
    frozen = FrozenPolicy.from_model(model)
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs = rng.random(config.OBS_DIM).astype(np.float32)
        expected, _ = model.predict(obs, deterministic=True)
        assert (frozen(obs) == expected).all()


def test_snapshot_is_independent_of_later_training() -> None:
    env = PlatformEnv(0, None, 1, seed=0)
    model = PPO("MlpPolicy", env, seed=0, device="cpu", n_steps=144, batch_size=144)
    frozen = FrozenPolicy.from_model(model)
    before = [w.copy() for w, _ in frozen.layers]
    model.learn(288)
    assert all(np.array_equal(a, w) for a, (w, _) in zip(before, frozen.layers, strict=True))
