"""Run whole episodes with given policies and collect episode metrics.

Evaluation runs policies directly against :class:`MarketSim` (no Gymnasium
wrapper needed), always on explicit seeds so results are reproducible and
comparable across rounds, modes and policies.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import config
from envs.platform_env import Policy
from sim.market import MarketSim
from sim.metrics import EpisodeMetrics

ShockSchedule = Sequence[tuple[int, str, int | None]]


def eval_seeds(n: int) -> list[int]:
    """The fixed evaluation seeds (the same every round and every run)."""
    return list(range(config.EVAL_SEED_START, config.EVAL_SEED_START + n))


def play_episode(
    policies: Mapping[int, Policy], n_platforms: int, seed: int, shocks: ShockSchedule = ()
) -> EpisodeMetrics:
    """Play one full episode and return its metrics.

    Args:
        policies: ``{platform_id: policy}`` covering every platform.
        n_platforms: 1 or 2.
        seed: Market seed.
        shocks: ``(tick, kind, target_platform)`` triggers, applied just before that tick.
    """
    market = MarketSim(n_platforms, seed=seed)
    by_tick: dict[int, list[tuple[str, int | None]]] = {}
    for tick, kind, target in shocks:
        by_tick.setdefault(int(tick), []).append((kind, target))
    while not market.done:
        for kind, target in by_tick.get(market.tick, []):
            market.trigger_shock(kind, target)
        actions = np.array([policies[p](market.observe(p)) for p in range(n_platforms)])
        market.step(actions)
    return market.metrics


def evaluate(
    policies: Mapping[int, Policy], n_platforms: int, seeds: Iterable[int], shocks: ShockSchedule = ()
) -> pd.DataFrame:
    """One row of episode summary metrics per seed."""
    rows = []
    for seed in seeds:
        summary = play_episode(policies, n_platforms, seed, shocks).summary()
        summary["seed"] = seed
        rows.append(summary)
    return pd.DataFrame(rows)


def mean_profit(
    policies: Mapping[int, Policy], n_platforms: int, platform: int, seeds: Iterable[int]
) -> tuple[float, float]:
    """Mean and std of one platform's episode profit over ``seeds``."""
    profits = evaluate(policies, n_platforms, seeds)[f"profit_p{platform}"]
    return float(profits.mean()), float(profits.std(ddof=0))
