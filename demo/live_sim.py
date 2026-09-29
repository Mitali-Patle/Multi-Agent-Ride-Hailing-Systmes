"""Inference-only live simulation shared by the dashboard and the CLI demo.

:class:`LiveSimulation` wraps a :class:`MarketSim` plus loaded policies. It
never trains: it only runs saved models (or the rule baseline as a clearly
labelled fallback). Each tick it also turns raw state changes into a
human-readable event log (the demo's telemetry): shocks starting and ending,
bonus changes, big fare moves, drivers migrating between platforms, queues
building up, and platforms dipping into losses.

Keeping this free of Streamlit means the demo logic is unit-tested.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import config
from envs.platform_env import Policy
from sim.market import MarketSim, clock_time
from sim.metrics import TickRecord
from sim.shocks import Shock, ShockKind
from training.policies import baseline_policies, trained_policies

logger = logging.getLogger(__name__)
MODES = ("duopoly", "monopoly")


@dataclass
class Event:
    """One line of telemetry."""

    tick: int
    clock: str
    kind: str  # "shock", "agent" or "market"
    message: str


def load_policies(mode: str, source: str, results_dir: Path | None = None) -> tuple[dict[int, Policy], str]:
    """Return ``(policies, label)``. ``source`` is 'trained' or 'baseline'.

    Raises:
        FileNotFoundError: trained models are missing (message says how to train).
    """
    if source == "baseline":
        return baseline_policies(mode), "rule-based baseline"
    return trained_policies(mode, results_dir), "trained PPO"


class LiveSimulation:
    """A resettable, steppable market with telemetry, driven by fixed policies."""

    def __init__(self, mode: str = "duopoly", seed: int | None = config.SEED, policy_source: str = "trained",
                 results_dir: Path | None = None, common_demand: bool = False) -> None:
        """Load policies for ``mode`` and start an episode with ``seed``.

        ``common_demand`` gives the market its own rider-arrival stream so two
        simulations with the same seed see identical riders (see MarketSim).
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.n_platforms = 2 if mode == "duopoly" else 1
        self.policies, self.policy_label = load_policies(mode, policy_source, results_dir)
        self.names = list(config.PLATFORM_NAMES[: self.n_platforms]) if self.n_platforms > 1 else [config.MONOPOLY_NAME]
        self.seed = config.SEED if seed is None else int(seed)
        self.market = MarketSim(self.n_platforms, seed=self.seed, common_demand=common_demand)
        self.reset()

    # ----------------------------------------------------------------- control
    def reset(self, seed: int | None = None) -> None:
        """Restart the episode (optionally with a new seed)."""
        self.seed = self.seed if seed is None else int(seed)
        self.market.reset(self.seed)
        self.events: deque[Event] = deque(maxlen=config.EVENT_LOG_MAX)
        self._prev_actions: list[tuple[int, int]] | None = None
        self._net_flow = 0  # net drivers moved from platform 0 to 1 since the last switch event
        self._queue_alert = [False] * self.n_platforms
        self._losing = [False] * self.n_platforms
        self._log("market", f"Episode started ({self.mode}, seed {self.seed}, policies: {self.policy_label})")

    @property
    def done(self) -> bool:
        """True when the 12-hour episode is over."""
        return self.market.done

    @property
    def tick(self) -> int:
        """Next tick to be simulated."""
        return self.market.tick

    def step(self) -> TickRecord | None:
        """Simulate one tick; returns ``None`` (no error) if the episode is over."""
        if self.market.done:
            return None
        actions = [tuple(int(x) for x in self.policies[p](self.market.observe(p))[:2]) for p in range(self.n_platforms)]
        rec = self.market.step(actions)
        self._record_events(actions, rec)
        self._prev_actions = actions
        if self.market.done:
            self._log("market", "Market closed: episode finished")
        return rec

    def trigger_shock(self, kind: ShockKind | str, target_platform: int | None = None) -> str:
        """Queue a shock for the next tick and return a status message for the UI."""
        kind = ShockKind(kind)
        shock = self.market.trigger_shock(kind, target_platform)
        if shock is None:
            reason = "episode is over" if self.done else "a monopoly has no rival price feed"
            msg = f"{kind.value.replace('_', ' ').capitalize()} ignored: {reason}"
            self._log("shock", msg)
            return msg
        return f"{shock.label().capitalize()} queued for tick {self.tick}"

    # --------------------------------------------------------------- telemetry
    def _log(self, kind: str, message: str, tick: int | None = None) -> None:
        """Append an event."""
        t = self.market.tick if tick is None else tick
        self.events.append(Event(t, clock_time(t), kind, message))

    def _shock_text(self, shock: Shock, started: bool) -> str:
        """Describe a shock starting or ending."""
        if shock.kind is ShockKind.DEMAND_SPIKE:
            return (f"Demand spike started: arrivals x{config.DEMAND_SPIKE_MULT:g} for {shock.duration} ticks"
                    if started else "Demand spike ended")
        if shock.kind is ShockKind.DRIVER_OUTAGE:
            n = len(shock.affected_drivers)
            return (f"Driver outage started: {n} drivers logging off for {shock.duration} ticks" if started
                    else f"Driver outage ended: {n} drivers back online")
        name = self.names[shock.target_platform or 0]
        return (f"Sensor failure: Platform {name}'s rival price feed frozen for {shock.duration} ticks" if started
                else f"Sensor restored: Platform {name} sees rival prices again")

    def _record_events(self, actions: list[tuple[int, int]], rec: TickRecord) -> None:
        """Turn this tick's changes into events."""
        t = rec.tick
        for shock in self.market.expired_shocks:
            self._log("shock", self._shock_text(shock, started=False), t)
        for shock in self.market.started_shocks:
            self._log("shock", self._shock_text(shock, started=True), t)
        if self._prev_actions is not None:
            for p, ((f0, b0), (f1, b1)) in enumerate(zip(self._prev_actions, actions, strict=True)):
                who = f"Platform {self.names[p]}" if self.n_platforms > 1 else config.MONOPOLY_NAME
                if b1 != b0:
                    verb = "raised" if b1 > b0 else "cut"
                    self._log("agent", f"{who} {verb} driver bonus from {config.BONUS_LABELS[b0]} to "
                                       f"{config.BONUS_LABELS[b1]}", t)
                if abs(f1 - f0) >= config.FARE_EVENT_MIN_JUMP:
                    verb = "raised" if f1 > f0 else "cut"
                    self._log("agent", f"{who} {verb} fare x{config.FARE_MULTIPLIERS[f0]:g} -> "
                                       f"x{config.FARE_MULTIPLIERS[f1]:g}", t)
        self._switch_events(rec)
        self._queue_and_loss_events(rec)

    def _switch_events(self, rec: TickRecord) -> None:
        """Log driver migration once enough drivers have (net) moved."""
        if self.n_platforms < 2:
            return
        self._net_flow += rec.platforms[0].switches_out - rec.platforms[0].switches_in
        if abs(self._net_flow) >= config.SWITCH_EVENT_MIN:
            src, dst = (0, 1) if self._net_flow > 0 else (1, 0)
            self._log("agent", f"{abs(self._net_flow)} drivers switched from {self.names[src]} to {self.names[dst]}",
                      rec.tick)
            self._net_flow = 0

    def _queue_and_loss_events(self, rec: TickRecord) -> None:
        """Log queues crossing the alert threshold and platforms dipping into losses."""
        for p, pt in enumerate(rec.platforms):
            who = f"Platform {self.names[p]}" if self.n_platforms > 1 else config.MONOPOLY_NAME
            if pt.queue >= config.QUEUE_EVENT_THRESHOLD and not self._queue_alert[p]:
                self._log("market", f"{who}: {pt.queue} riders queueing", rec.tick)
                self._queue_alert[p] = True
            elif pt.queue < config.QUEUE_EVENT_THRESHOLD / 2:
                self._queue_alert[p] = False
            if pt.profit < 0 and not self._losing[p]:
                self._log("market", f"{who} lost money this tick ({pt.profit:,.0f})", rec.tick)
            self._losing[p] = pt.profit < 0

    # ------------------------------------------------------------------ views
    def history(self) -> pd.DataFrame:
        """Long-format per-tick, per-platform history (empty before the first tick)."""
        df = self.market.metrics.to_frame()
        if not df.empty:
            df["name"] = df["platform"].map(dict(enumerate(self.names)))
            df["clock"] = df["tick"].map(clock_time)
        return df

    def events_frame(self) -> pd.DataFrame:
        """Event log, newest first."""
        rows = [{"tick": e.tick, "time": e.clock, "type": e.kind, "event": e.message} for e in reversed(self.events)]
        return pd.DataFrame(rows, columns=["tick", "time", "type", "event"])

    def shock_windows(self) -> list[tuple[int, int, str]]:
        """``(start, end, kind)`` for every shock that has started (for charts)."""
        shocks = self.market.shocks.history + self.market.shocks.active
        return [(s.start_tick, min(s.end_tick, config.EPISODE_TICKS), s.kind.value) for s in shocks
                if s.start_tick is not None]
