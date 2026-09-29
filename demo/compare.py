"""Monopoly vs duopoly, live and side by side, for the dashboard.

:class:`ComparisonSimulation` runs two inference-only markets in lockstep:
the trained monopolist, and the two trained competing platforms. Both use
the same seed with a shared rider-arrival stream (common random numbers), so
the same riders arrive at the same moments in both markets, and every shock
hits both markets on the same tick. Any gap between the two lines on a chart
is therefore caused by market structure (one platform vs two), not by luck.

:func:`chart_frames` turns both histories into the dashboard series: average
fare, share of riders served, rides served, total welfare, average market
share (all day-so-far), and rider wait time (the one trade-off).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
from demo.live_sim import LiveSimulation
from sim.shocks import ShockKind

WORLDS = ("monopoly", "duopoly")
MONO, DUO = config.MONOPOLY_NAME, "Duopoly"
DUO_A, DUO_B = config.PLATFORM_NAMES[0], config.PLATFORM_NAMES[1]  # the duopoly's two platforms
WAIT_WINDOW = 6  # ticks (30 min) the wait-time line is averaged over


class ComparisonSimulation:
    """A monopoly market and a duopoly market stepped and shocked together."""

    def __init__(self, seed: int = config.SEED, policy_source: str = "trained",
                 results_dir: Path | None = None) -> None:
        """Load both sets of saved policies (raises FileNotFoundError if missing)."""
        self.worlds = {m: LiveSimulation(m, seed, policy_source, results_dir, common_demand=True) for m in WORLDS}

    @property
    def mono(self) -> LiveSimulation:
        """The monopoly market."""
        return self.worlds["monopoly"]

    @property
    def duo(self) -> LiveSimulation:
        """The duopoly market."""
        return self.worlds["duopoly"]

    @property
    def seed(self) -> int:
        """Seed shared by both markets."""
        return self.duo.seed

    @property
    def tick(self) -> int:
        """Next tick to simulate (the markets are always in lockstep)."""
        return self.duo.tick

    @property
    def done(self) -> bool:
        """True once the 12-hour day is over."""
        return self.duo.done

    @property
    def policy_label(self) -> str:
        """'trained PPO' or 'rule-based baseline'."""
        return self.duo.policy_label

    def reset(self, seed: int | None = None) -> None:
        """Restart both markets on the same (optionally new) seed."""
        for world in self.worlds.values():
            world.reset(seed)

    def step(self) -> bool:
        """Advance both markets one tick; returns False if the day is over."""
        if self.done:
            return False
        for world in self.worlds.values():
            world.step()
        return True

    def advance(self, ticks: int) -> int:
        """Fast-forward up to ``ticks`` ticks; returns how many were simulated."""
        n = 0
        while n < ticks and self.step():
            n += 1
        return n

    def trigger_shock(self, kind: ShockKind | str, target_platform: int | None = None) -> str:
        """Queue a shock in both markets and return a status message for the UI.

        A sensor failure only exists in the duopoly (a monopolist has no rival
        price feed to lose), so it is applied there alone.
        """
        kind = ShockKind(kind)
        if kind is ShockKind.SENSOR_FAILURE:
            return self.duo.trigger_shock(kind, target_platform) + " (duopoly only: a monopoly has no rival)"
        messages = [world.trigger_shock(kind) for world in self.worlds.values()]
        return messages[-1] + " in both markets"

    def active_shocks(self) -> list[str]:
        """Labels of shocks active right now in either market (duopoly is a superset)."""
        return [] if self.done else self.duo.market.shocks.active_labels()

    def pending_shocks(self) -> list[str]:
        """Labels of shocks queued for the next tick."""
        return [s.label() for s in self.duo.market.shocks.pending]

    def shock_windows(self) -> list[tuple[int, int, str]]:
        """``(start, end, kind)`` of every shock so far (for chart shading)."""
        return self.duo.shock_windows()

    def events_frame(self) -> pd.DataFrame:
        """Both markets' event logs merged, newest first."""
        frames = []
        for mode, world in self.worlds.items():
            df = world.events_frame()
            df.insert(0, "market", MONO if mode == "monopoly" else DUO)
            frames.append(df)
        merged = pd.concat(frames, ignore_index=True)
        return merged.sort_values("tick", ascending=False, kind="stable").reset_index(drop=True)


def _per_tick(world: LiveSimulation) -> pd.DataFrame:
    """Market-wide totals per tick for one world (summing over platforms)."""
    hist = world.history()
    cols = ["fares_collected", "completed", "matched", "wait_ticks", "rider_surplus", "driver_earnings", "profit"]
    per = hist.groupby("tick")[cols].sum()
    per["arrivals"] = hist.groupby("tick")["arrivals"].first()  # market-wide, repeated on each platform row
    return per


def _long(values: pd.Series, series: str) -> pd.DataFrame:
    """Tidy rows ``tick, hour, series, value`` for one line."""
    ticks = values.index.to_numpy()
    return pd.DataFrame({
        "tick": ticks,
        "hour": config.START_HOUR + (ticks + 1) * config.TICK_MINUTES / 60.0,  # value at the END of each tick
        "series": series,
        "value": values.to_numpy(dtype=float),
    })


def _shares(world: LiveSimulation) -> pd.DataFrame:
    """Average (day-so-far) share of completed rides per platform."""
    hist = world.history()
    completed = hist.pivot(index="tick", columns="name", values="completed").cumsum()
    total = completed.sum(axis=1).replace(0, np.nan)
    frames = []
    for name in completed.columns:
        label = MONO if world.mode == "monopoly" else name
        frames.append(_long(completed[name] / total, label))
    return pd.concat(frames, ignore_index=True)


def chart_frames(sim: ComparisonSimulation) -> dict[str, pd.DataFrame]:
    """Dashboard series, each a tidy frame with columns tick/hour/series/value.

    All day-so-far values except ``wait`` (last 30 minutes): ``fare`` (average
    fare paid), ``served`` (share of arriving riders who got a ride),
    ``rides`` (rides served), ``welfare`` (rider surplus + driver earnings +
    platform profit), ``share`` (average market share) and ``wait``.
    """
    if sim.tick == 0:
        return {}
    out: dict[str, list[pd.DataFrame]] = {k: [] for k in ("fare", "served", "rides", "welfare", "share", "wait")}
    for mode, world in sim.worlds.items():
        per = _per_tick(world)
        cum = per.cumsum()
        label = MONO if mode == "monopoly" else DUO
        wait = (per["wait_ticks"].rolling(WAIT_WINDOW, min_periods=1).sum()
                / per["matched"].rolling(WAIT_WINDOW, min_periods=1).sum().replace(0, np.nan)) * config.TICK_MINUTES
        out["fare"].append(_long(cum["fares_collected"] / cum["completed"].replace(0, np.nan), label))
        out["served"].append(_long(cum["matched"] / cum["arrivals"].replace(0, np.nan), label))
        out["rides"].append(_long(cum["matched"], label))
        out["welfare"].append(_long(cum["rider_surplus"] + cum["driver_earnings"] + cum["profit"], label))
        out["wait"].append(_long(wait, label))
        out["share"].append(_shares(world))
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}
