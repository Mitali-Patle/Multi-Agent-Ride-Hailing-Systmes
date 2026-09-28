"""Tests for the Gymnasium wrapper."""

from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

import config
from envs.platform_env import PlatformEnv
from training.baselines import BaselinePolicy, RandomPolicy


def _duopoly_env(**kwargs) -> PlatformEnv:
    return PlatformEnv(0, {1: BaselinePolicy(2)}, n_platforms=2, seed=0, **kwargs)


@pytest.mark.parametrize("random_shocks", [False, True])
def test_check_env_duopoly(random_shocks: bool) -> None:
    check_env(_duopoly_env(random_shocks=random_shocks), warn=True)


def test_check_env_duopoly_as_platform_b() -> None:
    check_env(PlatformEnv(1, {0: RandomPolicy(1)}, n_platforms=2, seed=0), warn=True)


def test_check_env_monopoly() -> None:
    check_env(PlatformEnv(0, None, n_platforms=1, seed=0, random_shocks=True), warn=True)


def test_missing_opponent_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="opponent"):
        PlatformEnv(0, None, n_platforms=2)
    with pytest.raises(ValueError):
        PlatformEnv(1, None, n_platforms=1)


def test_episode_length_and_reward_scale() -> None:
    env = _duopoly_env()
    obs, _ = env.reset(seed=3)
    total, steps, done = 0.0, 0, False
    while not done:
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        assert not truncated and env.observation_space.contains(obs)
        total += reward * config.REWARD_SCALE
        steps += 1
    assert steps == config.EPISODE_TICKS
    assert total == pytest.approx(env.market.metrics.summary()["profit_p0"])


def test_env_seeding_is_reproducible() -> None:
    def rollout() -> list[float]:
        env = _duopoly_env(random_shocks=True)
        env.reset(seed=11)
        env.action_space.seed(0)
        return [env.step(env.action_space.sample())[1] for _ in range(100)]

    assert rollout() == rollout()


def test_baseline_policy_outputs_valid_actions() -> None:
    policy = BaselinePolicy(2)
    for obs in (np.zeros(config.OBS_DIM), np.ones(config.OBS_DIM), np.full(config.OBS_DIM, 0.3)):
        fare, bonus = policy(obs)
        assert 0 <= fare < len(config.FARE_MULTIPLIERS) and 0 <= bonus < len(config.BONUS_LEVELS)
