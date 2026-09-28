"""Rule-based driver agents and the pool that manages them.

Each :class:`Driver` is an individual agent with its own affiliation, state,
current ride, earnings, and private beliefs. Beliefs are an exponential
moving average (EMA) of expected earnings per tick on each platform, built
only from public information: the posted fare, posted bonus, the commission,
and each platform's recently observed utilisation.

Drivers are heterogeneous: each has its own EMA speed and its own inertia
(a utility bonus for staying with the current platform). Only idle drivers
may switch, only a random fraction reconsider each tick, and the choice is a
logit over platforms, so the fleet drifts rather than flipping at once.

Driver count is conserved by construction: drivers are never created or
destroyed after reset, only moved between states and platforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

import config


class DriverState(IntEnum):
    """What a driver is doing right now."""

    IDLE = 0
    BUSY = 1
    OFFLINE = 2


@dataclass
class Ride:
    """A ride in progress. Fare is locked when the rider chose; bonus at match."""

    platform: int
    fare: float
    bonus: float
    ticks_remaining: int


@dataclass
class Driver:
    """An individual driver agent.

    Attributes:
        driver_id: Stable id.
        platform: Currently affiliated platform.
        ema_alpha: How fast this driver updates beliefs (0-1).
        inertia: Utility bonus (currency/tick) for staying on ``platform``.
        beliefs: EMA of expected earnings per tick, one entry per platform.
        state: Idle, busy, or offline.
        ride: The ride in progress, if busy.
        logoff_pending: Go offline as soon as the current ride ends.
        earnings: Total earnings so far this episode.
        rides_completed: Rides completed this episode.
    """

    driver_id: int
    platform: int
    ema_alpha: float
    inertia: float
    beliefs: np.ndarray
    state: DriverState = DriverState.IDLE
    ride: Ride | None = None
    logoff_pending: bool = False
    earnings: float = 0.0
    rides_completed: int = 0

    def update_beliefs(self, signal: np.ndarray) -> None:
        """Blend the latest public earnings signal into the private EMA (in place)."""
        self.beliefs += self.ema_alpha * (np.asarray(signal, dtype=float) - self.beliefs)

    def choose_platform(self, rng: np.random.Generator) -> int:
        """Logit choice over platforms with inertia toward the current one."""
        utility = self.beliefs.copy()
        utility[self.platform] += self.inertia
        return int(rng.choice(len(utility), p=softmax(utility, config.DRIVER_LOGIT_TEMP)))


@dataclass
class Completion:
    """A ride that finished this tick, for payout accounting."""

    driver_id: int
    platform: int
    fare: float
    bonus: float
    driver_pay: float


@dataclass
class Switch:
    """A driver moving from one platform to another."""

    driver_id: int
    from_platform: int
    to_platform: int


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically stable softmax; ``temperature <= 0`` gives argmax with ties shared."""
    values = np.asarray(values, dtype=float)
    if temperature <= 0:
        best = values == values.max()
        return best / best.sum()
    z = (values - values.max()) / temperature
    expz = np.exp(z)
    return expz / expz.sum()


def driver_pay_per_ride(fare: float, bonus: float) -> float:
    """What a driver earns for one completed ride."""
    return (1.0 - config.COMMISSION) * fare + bonus


@dataclass
class DriverPool:
    """All drivers in the market plus the operations the market needs."""

    n_platforms: int
    rng: np.random.Generator
    n_drivers: int = field(default_factory=lambda: config.N_DRIVERS)
    drivers: list[Driver] = field(init=False)

    def __post_init__(self) -> None:
        """Create drivers, spread evenly across platforms (round robin).

        Each driver's ``beliefs`` is a row view into ``belief_matrix`` so the
        whole fleet can be updated in one vectorised step while every driver
        still owns (and decides from) its private beliefs.
        """
        self.alphas = self.rng.uniform(config.DRIVER_EMA_ALPHA_MIN, config.DRIVER_EMA_ALPHA_MAX, self.n_drivers)
        inertias = self.rng.uniform(config.DRIVER_INERTIA_MIN, config.DRIVER_INERTIA_MAX, self.n_drivers)
        self.belief_matrix = np.zeros((self.n_drivers, self.n_platforms))
        self.drivers = [
            Driver(
                driver_id=i,
                platform=i % self.n_platforms,
                ema_alpha=float(self.alphas[i]),
                inertia=float(inertias[i]),
                beliefs=self.belief_matrix[i],
            )
            for i in range(self.n_drivers)
        ]

    # ------------------------------------------------------------------ counts
    def counts(self) -> np.ndarray:
        """Matrix ``[platform, state]`` of driver counts (idle, busy, offline)."""
        out = np.zeros((self.n_platforms, 3), dtype=int)
        for d in self.drivers:
            out[d.platform, int(d.state)] += 1
        return out

    def idle_ids(self, platform: int) -> list[int]:
        """Ids of idle drivers on ``platform``."""
        return [d.driver_id for d in self.drivers if d.platform == platform and d.state is DriverState.IDLE]

    # ------------------------------------------------------------ behaviour
    def init_beliefs(self, signal: np.ndarray) -> None:
        """Start every driver's beliefs at the same public signal."""
        self.belief_matrix[:] = np.asarray(signal, dtype=float)

    def update_beliefs(self, signal: np.ndarray) -> None:
        """Every driver (busy or not) observes the public signal.

        Vectorised equivalent of calling :meth:`Driver.update_beliefs` on
        each driver; updates the per-driver views in place.
        """
        signal = np.asarray(signal, dtype=float)
        self.belief_matrix += self.alphas[:, None] * (signal[None, :] - self.belief_matrix)

    def consider_switching(self) -> list[Switch]:
        """Let a random subset of idle drivers re-choose their platform."""
        switches: list[Switch] = []
        if self.n_platforms < 2:
            return switches
        for d in self.drivers:
            if d.state is not DriverState.IDLE or self.rng.random() >= config.DRIVER_CONSIDER_PROB:
                continue
            target = d.choose_platform(self.rng)
            if target != d.platform:
                switches.append(Switch(d.driver_id, d.platform, target))
                d.platform = target
        return switches

    def assign_ride(self, driver_id: int, fare: float, bonus: float) -> None:
        """Make an idle driver busy with a new ride of random length."""
        d = self.drivers[driver_id]
        if d.state is not DriverState.IDLE:
            raise ValueError(f"driver {driver_id} is not idle")
        ticks = int(self.rng.integers(config.RIDE_TICKS_MIN, config.RIDE_TICKS_MAX + 1))
        d.ride = Ride(platform=d.platform, fare=fare, bonus=bonus, ticks_remaining=ticks)
        d.state = DriverState.BUSY

    def advance(self) -> list[Completion]:
        """Advance every ride by one tick and pay out rides that finish."""
        done: list[Completion] = []
        for d in self.drivers:
            if d.state is DriverState.BUSY and d.ride is not None:
                d.ride.ticks_remaining -= 1
                if d.ride.ticks_remaining <= 0:
                    done.append(self._complete(d))
        return done

    def settle_all(self) -> list[Completion]:
        """Complete every ride still in progress (used when the market closes)."""
        return [self._complete(d) for d in self.drivers if d.state is DriverState.BUSY and d.ride is not None]

    def _complete(self, d: Driver) -> Completion:
        """Pay out and free one driver."""
        ride = d.ride
        assert ride is not None
        pay = driver_pay_per_ride(ride.fare, ride.bonus)
        d.earnings += pay
        d.rides_completed += 1
        d.ride = None
        d.state = DriverState.OFFLINE if d.logoff_pending else DriverState.IDLE
        d.logoff_pending = False
        return Completion(d.driver_id, ride.platform, ride.fare, ride.bonus, pay)

    # ------------------------------------------------------------ outages
    def take_offline(self, count: int) -> list[int]:
        """Send ``count`` random online drivers offline.

        Idle drivers leave at once; busy drivers finish their current ride
        first (``logoff_pending``). Drivers already affected by another
        outage are not picked twice.

        Returns:
            Ids of the affected drivers.
        """
        eligible = [
            d.driver_id for d in self.drivers if d.state is not DriverState.OFFLINE and not d.logoff_pending
        ]
        count = max(0, min(count, len(eligible)))
        if count == 0:
            return []
        chosen = [int(i) for i in self.rng.choice(eligible, size=count, replace=False)]
        for i in chosen:
            d = self.drivers[i]
            if d.state is DriverState.IDLE:
                d.state = DriverState.OFFLINE
            else:
                d.logoff_pending = True
        return chosen

    def bring_online(self, driver_ids: list[int]) -> None:
        """End an outage for the given drivers: offline -> idle, cancel pending logoffs."""
        for i in driver_ids:
            d = self.drivers[i]
            d.logoff_pending = False
            if d.state is DriverState.OFFLINE:
                d.state = DriverState.IDLE
