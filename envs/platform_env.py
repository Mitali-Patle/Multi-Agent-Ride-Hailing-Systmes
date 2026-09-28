"""Gymnasium wrapper that exposes ONE platform's view of the shared market.

Stable-Baselines3 trains single-agent environments, so each platform gets its
own :class:`PlatformEnv`. The other platform is folded into the environment
as a callable ``policy(obs) -> action``: a frozen snapshot of the rival's PPO
model, or a rule-based baseline. This is how independent PPO turns a
two-player market into two single-agent problems.

Observation: the 11-dim partial view built by ``MarketSim.observe``.
Action:      ``MultiDiscrete([n_fare_levels, n_bonus_levels])``.
Reward:      this platform's profit for the tick / ``REWARD_SCALE``.

With ``random_shocks=True`` (used in training) episodes randomly contain
shocks, so the learned policies have seen demand spikes, outages and blind
rival feeds before they meet them live in the demo.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config
from sim.market import MarketSim
from sim.shocks import ShockKind

logger = logging.getLogger(__name__)

Policy = Callable[[np.ndarray], np.ndarray]


class PlatformEnv(gym.Env):
    """One platform's single-agent view of the market."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        platform_id: int = 0,
        opponent_policies: Mapping[int, Policy] | None = None,
        n_platforms: int | None = None,
        seed: int | None = None,
        random_shocks: bool = False,
    ) -> None:
        """Create the environment.

        Args:
            platform_id: Which platform the learner controls.
            opponent_policies: ``{platform_id: policy}`` for every other
                platform. Required in duopoly, ignored in monopoly.
            n_platforms: 1 or 2; defaults to ``config.N_PLATFORMS``.
            seed: Seed used by the first ``reset()`` if none is passed.
            random_shocks: Inject random shocks into episodes (training).
        """
        super().__init__()
        self.n_platforms = config.N_PLATFORMS if n_platforms is None else int(n_platforms)
        if not 0 <= platform_id < self.n_platforms:
            raise ValueError(f"platform_id {platform_id} invalid for {self.n_platforms} platform(s)")
        self.platform_id = platform_id
        self.opponents = dict(opponent_policies or {})
        missing = [p for p in range(self.n_platforms) if p != platform_id and p not in self.opponents]
        if missing:
            raise ValueError(f"no opponent policy supplied for platform(s) {missing}")
        self.random_shocks = random_shocks
        self._initial_seed = seed
        self._shock_plan: dict[int, list[tuple[ShockKind, int | None]]] = {}
        self.market = MarketSim(self.n_platforms, seed=seed)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(config.OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([len(config.FARE_MULTIPLIERS), len(config.BONUS_LEVELS)])

    def set_opponents(self, opponent_policies: Mapping[int, Policy]) -> None:
        """Swap in new opponent policies (e.g. the rival's latest snapshot)."""
        self.opponents.update(opponent_policies)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        """Start a new episode. The first reset uses the constructor seed."""
        if seed is None and self._initial_seed is not None:
            seed, self._initial_seed = self._initial_seed, None
        super().reset(seed=seed)
        market_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.market.reset(market_seed)
        self._shock_plan = self._plan_shocks() if self.random_shocks else {}
        return self.market.observe(self.platform_id), {}

    def _plan_shocks(self) -> dict[int, list[tuple[ShockKind, int | None]]]:
        """Randomly schedule 0..N shocks for a training episode."""
        plan: dict[int, list[tuple[ShockKind, int | None]]] = {}
        if self.np_random.random() >= config.TRAIN_RANDOM_SHOCK_PROB:
            return plan
        kinds = list(ShockKind)
        if self.n_platforms < 2:
            kinds.remove(ShockKind.SENSOR_FAILURE)
        for _ in range(int(self.np_random.integers(1, config.TRAIN_SHOCKS_PER_EPISODE_MAX + 1))):
            tick = int(self.np_random.integers(0, config.EPISODE_TICKS))
            kind = kinds[int(self.np_random.integers(len(kinds)))]
            target = int(self.np_random.integers(self.n_platforms)) if kind is ShockKind.SENSOR_FAILURE else None
            plan.setdefault(tick, []).append((kind, target))
        return plan

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one tick with the learner's action and the opponents' actions."""
        for kind, target in self._shock_plan.get(self.market.tick, []):
            self.market.trigger_shock(kind, target)
        actions = np.zeros((self.n_platforms, 2), dtype=int)
        for p in range(self.n_platforms):
            if p == self.platform_id:
                actions[p] = np.asarray(action, dtype=int).reshape(-1)[:2]
            else:
                actions[p] = np.asarray(self.opponents[p](self.market.observe(p)), dtype=int).reshape(-1)[:2]
        rec = self.market.step(actions)
        own = rec.platforms[self.platform_id]
        reward = float(own.profit / config.REWARD_SCALE)
        terminated = self.market.done  # the market closes; nothing follows
        info = {"profit": own.profit, "completed": own.completed, "tick": rec.tick}
        return self.market.observe(self.platform_id), reward, terminated, False, info
