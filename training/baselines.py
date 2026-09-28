"""Fixed rule-based policies: the bootstrap opponent and evaluation yardstick.

:class:`BaselinePolicy` is a hand-written surge rule a pricing analyst might
deploy. It never learns, so "reward against the baseline" is a fixed
yardstick for whether a PPO policy is actually improving.

* Fare: surge when riders are queueing, raise when utilisation is high,
  discount when it is low, otherwise the standard fare.
* Bonus: pay the low bonus only when the platform's active fleet has fallen
  below ``BASELINE_FLEET_SHARE_FLOOR`` of an equal share of all drivers.

It only reads the platform's own observation, like the learners do.
"""

from __future__ import annotations

import numpy as np

import config


class BaselinePolicy:
    """Rule-based pricing policy with the same call signature as a learner."""

    def __init__(self, n_platforms: int | None = None) -> None:
        """Create the rule for a market with ``n_platforms`` platforms."""
        self.n_platforms = config.N_PLATFORMS if n_platforms is None else int(n_platforms)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Map an observation to ``[fare_index, bonus_index]``."""
        obs = np.asarray(obs, dtype=float).reshape(-1)
        idle, busy, queue = obs[0], obs[1], obs[2]
        active = idle + busy
        utilisation = busy / active if active > 0 else 1.0
        if queue >= config.BASELINE_QUEUE_SURGE:
            fare = config.BASELINE_FARE_SURGE
        elif utilisation >= config.BASELINE_HIGH_UTIL:
            fare = config.BASELINE_FARE_HIGH
        elif utilisation <= config.BASELINE_LOW_UTIL:
            fare = config.BASELINE_FARE_LOW
        else:
            fare = config.BASELINE_FARE_MID
        fair_share = 1.0 / self.n_platforms
        recruiting = active < config.BASELINE_FLEET_SHARE_FLOOR * fair_share
        bonus = config.BASELINE_BONUS_RECRUIT if recruiting else config.BASELINE_BONUS_DEFAULT
        return np.array([fare, bonus], dtype=np.int64)


class RandomPolicy:
    """Uniformly random actions (a sanity-check reference, seeded)."""

    def __init__(self, seed: int | None = None) -> None:
        """Create a seeded random policy."""
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Ignore the observation and return a random action."""
        return np.array(
            [self.rng.integers(len(config.FARE_MULTIPLIERS)), self.rng.integers(len(config.BONUS_LEVELS))],
            dtype=np.int64,
        )
