"""Environmental shocks that perturb the shared market.

Three shock types model the "environmental shock" the demo must survive:

* ``DEMAND_SPIKE``   - arrival rate is multiplied (a concert just ended).
* ``DRIVER_OUTAGE``  - a fraction of drivers log off (heavy rain).
* ``SENSOR_FAILURE`` - one platform's feed of the rival's prices freezes and
  its ``rival_feed_ok`` observation flag drops to 0.

The :class:`ShockManager` only keeps the bookkeeping (what is active, when it
started, when it ends). The market applies the effects, because only the
market knows about drivers and observations. Shocks stack: two demand spikes
multiply, two outages each take their own drivers offline.

Lifecycle: ``trigger`` queues a shock; at the start of the next market tick
``begin_tick`` activates it. A shock with duration ``D`` activated at tick
``t0`` is active on ticks ``t0 .. t0 + D - 1`` and expires at the start of
tick ``t0 + D``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import config

logger = logging.getLogger(__name__)


class ShockKind(str, Enum):
    """The supported shock types."""

    DEMAND_SPIKE = "demand_spike"
    DRIVER_OUTAGE = "driver_outage"
    SENSOR_FAILURE = "sensor_failure"


def default_duration(kind: ShockKind) -> int:
    """Return the configured duration in ticks for a shock kind."""
    return {
        ShockKind.DEMAND_SPIKE: config.DEMAND_SPIKE_TICKS,
        ShockKind.DRIVER_OUTAGE: config.OUTAGE_TICKS,
        ShockKind.SENSOR_FAILURE: config.SENSOR_FAIL_TICKS,
    }[kind]


@dataclass
class Shock:
    """One shock instance.

    Attributes:
        shock_id: Unique id within an episode.
        kind: Which shock this is.
        duration: Number of ticks the shock stays active.
        target_platform: Platform whose feed fails (sensor failure only).
        start_tick: Tick the shock became active, ``None`` while pending.
        affected_drivers: Driver ids taken offline (driver outage only).
    """

    shock_id: int
    kind: ShockKind
    duration: int
    target_platform: int | None = None
    start_tick: int | None = None
    affected_drivers: list[int] = field(default_factory=list)

    @property
    def end_tick(self) -> int | None:
        """First tick on which the shock is no longer active."""
        return None if self.start_tick is None else self.start_tick + self.duration

    def label(self) -> str:
        """Short human-readable description for logs and the dashboard."""
        name = self.kind.value.replace("_", " ")
        if self.kind is ShockKind.SENSOR_FAILURE and self.target_platform is not None:
            name += f" ({config.PLATFORM_NAMES[self.target_platform]} blind)"
        return name


class ShockManager:
    """Tracks pending, active and finished shocks for one episode."""

    def __init__(self) -> None:
        """Create an empty manager."""
        self.pending: list[Shock] = []
        self.active: list[Shock] = []
        self.history: list[Shock] = []
        self._next_id = 0

    def reset(self) -> None:
        """Forget all shocks (used on market reset, including mid-episode)."""
        self.pending.clear()
        self.active.clear()
        self.history.clear()
        self._next_id = 0

    def trigger(
        self,
        kind: ShockKind | str,
        target_platform: int | None = None,
        duration: int | None = None,
    ) -> Shock:
        """Queue a shock; it becomes active at the start of the next tick.

        Args:
            kind: Shock type (enum or its string value).
            target_platform: Platform that loses its rival feed (sensor only).
            duration: Override the configured duration, in ticks (min 1).

        Returns:
            The queued :class:`Shock`.
        """
        kind = ShockKind(kind)
        if duration is None:
            duration = default_duration(kind)
        shock = Shock(
            shock_id=self._next_id,
            kind=kind,
            duration=max(1, int(duration)),
            target_platform=target_platform,
        )
        self._next_id += 1
        self.pending.append(shock)
        logger.debug("queued shock %s", shock.label())
        return shock

    def begin_tick(self, tick: int) -> tuple[list[Shock], list[Shock]]:
        """Expire finished shocks, then activate pending ones.

        Expiring first means back-to-back shocks of the same kind hand over
        cleanly (e.g. drivers from the first outage return before the second
        outage picks its drivers).

        Returns:
            ``(started, expired)`` lists of shocks.
        """
        expired = [s for s in self.active if s.end_tick is not None and tick >= s.end_tick]
        self.active = [s for s in self.active if s not in expired]
        self.history.extend(expired)
        started = list(self.pending)
        for shock in started:
            shock.start_tick = tick
        self.active.extend(started)
        self.pending.clear()
        return started, expired

    def demand_multiplier(self) -> float:
        """Product of all active demand-spike multipliers (1.0 if none)."""
        mult = 1.0
        for shock in self.active:
            if shock.kind is ShockKind.DEMAND_SPIKE:
                mult *= config.DEMAND_SPIKE_MULT
        return mult

    def feed_ok(self, platform_id: int) -> bool:
        """False while a sensor failure targets ``platform_id``."""
        return not any(
            s.kind is ShockKind.SENSOR_FAILURE and s.target_platform == platform_id
            for s in self.active
        )

    def active_labels(self) -> list[str]:
        """Labels of all active shocks."""
        return [s.label() for s in self.active]
