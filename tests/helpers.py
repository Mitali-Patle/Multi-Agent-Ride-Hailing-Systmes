"""Shared helpers for the test suite."""

from __future__ import annotations

import numpy as np

import config
from sim.market import MarketSim


def random_actions(rng: np.random.Generator, n_platforms: int) -> np.ndarray:
    """Uniformly random (fare, bonus) pair per platform."""
    fares = rng.integers(0, len(config.FARE_MULTIPLIERS), n_platforms)
    bonuses = rng.integers(0, len(config.BONUS_LEVELS), n_platforms)
    return np.stack([fares, bonuses], axis=1)


def assert_conserved(market: MarketSim) -> None:
    """Driver count is conserved and no count is negative."""
    counts = market.driver_counts()
    assert (counts >= 0).all(), counts
    assert counts.sum() == market.n_drivers, counts
    assert all(q >= 0 for q in market.queue_lengths())


def run_episode(market: MarketSim, seed: int = 0, check: bool = True) -> None:
    """Play a full episode with random actions, checking invariants every tick."""
    rng = np.random.default_rng(seed)
    while not market.done:
        market.step(random_actions(rng, market.n_platforms))
        if check:
            assert_conserved(market)
