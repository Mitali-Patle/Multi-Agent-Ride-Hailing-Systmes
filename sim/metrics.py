"""Per-tick and per-episode metric tracking.

The market writes one :class:`TickRecord` per tick. :class:`EpisodeMetrics`
keeps the list of records and turns it into the welfare numbers that the
hypotheses are about:

* rider surplus   = sum over served riders of (WTP - price - WAIT_COST * wait),
  minus the waiting cost already borne by riders who abandoned;
* driver earnings = sum of (1 - commission) * fare + bonus over completed rides;
* platform profit = sum of commission * fare - bonus over completed rides;
* total welfare   = rider surplus + driver earnings + platform profit.

Fares and bonuses are transfers, so total welfare only depends on which
riders get served and how long everyone waits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

import config


@dataclass
class PlatformTick:
    """What happened on one platform during one tick."""

    fare_idx: int = 0
    bonus_idx: int = 0
    fare: float = 0.0
    bonus: float = 0.0
    idle: int = 0
    busy: int = 0
    offline: int = 0
    queue: int = 0
    requests: int = 0  # riders who chose this platform this tick
    matched: int = 0  # riders picked up this tick (from queue or new)
    completed: int = 0  # rides that finished and paid out this tick
    abandoned: int = 0
    fares_collected: float = 0.0
    bonus_paid: float = 0.0
    profit: float = 0.0
    driver_earnings: float = 0.0
    rider_surplus: float = 0.0
    wait_ticks: float = 0.0  # summed over riders matched this tick
    switches_in: int = 0
    switches_out: int = 0
    feed_ok: bool = True


@dataclass
class TickRecord:
    """Market-wide snapshot of one tick."""

    tick: int
    arrivals: int = 0
    outside: int = 0  # riders who chose not to ride
    demand_mult: float = 1.0
    active_shocks: list[str] = field(default_factory=list)
    platforms: list[PlatformTick] = field(default_factory=list)


def _safe_div(num: float, den: float) -> float:
    """Division that returns 0.0 instead of failing on an empty denominator."""
    return float(num) / float(den) if den else 0.0


class EpisodeMetrics:
    """Accumulates tick records and summarises an episode."""

    def __init__(self, n_platforms: int, n_drivers: int) -> None:
        """Create an empty tracker for a market of the given size."""
        self.n_platforms = n_platforms
        self.n_drivers = n_drivers
        self.records: list[TickRecord] = []

    def add(self, record: TickRecord) -> None:
        """Store one tick."""
        self.records.append(record)

    def totals(self, name: str) -> np.ndarray:
        """Sum of a :class:`PlatformTick` field over the episode, per platform."""
        out = np.zeros(self.n_platforms)
        for rec in self.records:
            for p, pt in enumerate(rec.platforms):
                out[p] += getattr(pt, name)
        return out

    def summary(self) -> dict[str, float]:
        """Episode-level metrics used by evaluation, charts and hypotheses."""
        completed = self.totals("completed")
        matched = self.totals("matched")
        requests = self.totals("requests")
        profit = self.totals("profit")
        driver_earn = self.totals("driver_earnings").sum()
        surplus = self.totals("rider_surplus").sum()
        arrivals = sum(r.arrivals for r in self.records)
        out = {
            "avg_fare": _safe_div(self.totals("fares_collected").sum(), completed.sum()),
            "avg_wait_min": _safe_div(self.totals("wait_ticks").sum(), matched.sum()) * config.TICK_MINUTES,
            "abandonment_rate": _safe_div(self.totals("abandoned").sum(), requests.sum()),
            "completion_rate": _safe_div(completed.sum(), requests.sum()),
            "market_coverage": _safe_div(completed.sum(), arrivals),
            "arrivals": float(arrivals),
            "requests": float(requests.sum()),
            "completed": float(completed.sum()),
            "driver_earnings_total": float(driver_earn),
            "driver_earnings_per_driver": _safe_div(driver_earn, self.n_drivers),
            "platform_profit": float(profit.sum()),
            "bonus_paid": float(self.totals("bonus_paid").sum()),
            "rider_surplus": float(surplus),
            "total_welfare": float(surplus + driver_earn + profit.sum()),
        }
        for p in range(self.n_platforms):
            out[f"profit_p{p}"] = float(profit[p])
            out[f"share_p{p}"] = _safe_div(completed[p], completed.sum())
        return out

    def to_frame(self) -> pd.DataFrame:
        """Long-format table: one row per (tick, platform)."""
        rows = []
        for rec in self.records:
            for p, pt in enumerate(rec.platforms):
                row = {
                    "tick": rec.tick,
                    "platform": p,
                    "arrivals": rec.arrivals,
                    "outside": rec.outside,
                    "demand_mult": rec.demand_mult,
                    "active_shocks": "; ".join(rec.active_shocks),
                }
                row.update(asdict(pt))
                rows.append(row)
        return pd.DataFrame(rows)
