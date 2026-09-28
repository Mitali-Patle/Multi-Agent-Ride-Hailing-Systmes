"""Stress test: many random episodes, random shocks, both modes, no exceptions."""

from __future__ import annotations

import numpy as np

import config
from envs.platform_env import PlatformEnv
from sim.shocks import ShockKind
from training.baselines import RandomPolicy
from tests.helpers import assert_conserved

N_EPISODES = 200


def test_200_random_episodes_with_random_shocks() -> None:
    rng = np.random.default_rng(0)
    envs = {
        1: PlatformEnv(0, None, n_platforms=1, seed=0, random_shocks=True),
        2: PlatformEnv(0, {1: RandomPolicy(1)}, n_platforms=2, seed=0, random_shocks=True),
    }
    kinds = list(ShockKind)
    for episode in range(N_EPISODES):
        env = envs[1 + episode % 2]
        env.reset(seed=episode)
        env.action_space.seed(episode)
        done, steps = False, 0
        while not done:
            if rng.random() < 0.02:  # extra live shocks on top of the env's own random ones
                env.market.trigger_shock(kinds[rng.integers(len(kinds))], int(rng.integers(env.n_platforms)))
            obs, reward, done, _, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward) and env.observation_space.contains(obs)
            assert_conserved(env.market)
            steps += 1
        assert steps == config.EPISODE_TICKS
        assert np.isfinite(list(env.market.metrics.summary().values())).all()
