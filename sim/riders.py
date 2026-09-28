"""Rule-based rider agents: arrivals, quotes, choice, and queues.

Riders arrive by a Poisson process whose mean follows a daily curve (a base
rate plus a morning and an evening Gaussian peak). Each rider draws a
willingness to pay (WTP), receives a quote (price, estimated wait) from every
platform, and makes a logit choice between the platforms and the outside
option ("don't ride", utility 0).

A rider who picks a platform with no idle driver joins that platform's FIFO
queue and abandons after ``RIDER_PATIENCE`` ticks without a pickup.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

import config
from sim.drivers import softmax

OUTSIDE_OPTION = -1


@dataclass
class Rider:
    """An individual rider agent.

    Attributes:
        rider_id: Unique id within an episode.
        arrival_tick: Tick the rider appeared.
        wtp: Willingness to pay for one ride.
        platform: Chosen platform, or ``OUTSIDE_OPTION``.
        price: Price locked in when the rider chose.
    """

    rider_id: int
    arrival_tick: int
    wtp: float
    platform: int = OUTSIDE_OPTION
    price: float = 0.0


def arrival_rate(tick: int) -> float:
    """Poisson mean of rider arrivals at ``tick`` (before shocks)."""
    morning = config.MORNING_PEAK_HEIGHT * np.exp(
        -0.5 * ((tick - config.MORNING_PEAK_TICK) / config.MORNING_PEAK_WIDTH) ** 2
    )
    evening = config.EVENING_PEAK_HEIGHT * np.exp(
        -0.5 * ((tick - config.EVENING_PEAK_TICK) / config.EVENING_PEAK_WIDTH) ** 2
    )
    return float(config.ARRIVAL_BASE_RATE + morning + evening)


def draw_arrivals(rng: np.random.Generator, tick: int, demand_mult: float, first_id: int) -> list[Rider]:
    """Sample this tick's new riders and their willingness to pay."""
    n = int(rng.poisson(max(0.0, arrival_rate(tick) * demand_mult)))
    wtp = np.clip(rng.normal(config.WTP_MEAN, config.WTP_STD, n), config.WTP_MIN, config.WTP_MAX)
    return [Rider(first_id + i, tick, float(w)) for i, w in enumerate(wtp)]


def estimate_wait(idle: int, busy: int, queue_len: int) -> float:
    """Quoted wait in ticks for a platform, from public supply information.

    With an idle driver the wait is just the pickup time. Otherwise the rider
    waits for their turn in the queue while busy drivers free up at a rate of
    ``busy / MEAN_RIDE_TICKS`` per tick. No active drivers at all gives the
    long ``MAX_WAIT_ESTIMATE`` rather than a division by zero.
    """
    if idle > 0:
        return config.PICKUP_TICKS
    service_rate = busy / config.MEAN_RIDE_TICKS
    if service_rate <= 0:
        return config.MAX_WAIT_ESTIMATE
    return float(min(config.MAX_WAIT_ESTIMATE, config.PICKUP_TICKS + (queue_len + 1) / service_rate))


def choose_platform(
    rider: Rider, prices: np.ndarray, waits: np.ndarray, rng: np.random.Generator
) -> int:
    """Logit choice over platforms plus the outside option.

    Utility per platform is ``wtp - price - WAIT_COST * wait``; the outside
    option has utility 0. Sampling from the softmax breaks ties randomly
    (tied options get equal probability).

    Returns:
        Chosen platform index, or ``OUTSIDE_OPTION``.
    """
    utilities = np.append(rider.wtp - prices - config.WAIT_COST * waits, 0.0)
    probs = softmax(utilities, config.RIDER_LOGIT_TEMP)
    choice = int(np.searchsorted(np.cumsum(probs), rng.random() * probs.sum(), side="right"))
    choice = min(choice, len(utilities) - 1)
    return OUTSIDE_OPTION if choice == len(prices) else choice


def rider_surplus(rider: Rider, wait_ticks: float) -> float:
    """Surplus of a served rider: WTP minus price minus the cost of waiting."""
    return rider.wtp - rider.price - config.WAIT_COST * wait_ticks


class RiderQueue:
    """FIFO queue of riders waiting for one platform."""

    def __init__(self) -> None:
        """Create an empty queue."""
        self._riders: deque[Rider] = deque()

    def __len__(self) -> int:
        """Number of waiting riders."""
        return len(self._riders)

    def push(self, rider: Rider) -> None:
        """Add a rider at the back."""
        self._riders.append(rider)

    def pop(self) -> Rider:
        """Remove and return the rider at the front."""
        return self._riders.popleft()

    def pop_abandoned(self, tick: int) -> list[Rider]:
        """Remove riders who have waited longer than their patience."""
        gone = [r for r in self._riders if tick - r.arrival_tick > config.RIDER_PATIENCE]
        if gone:
            self._riders = deque(r for r in self._riders if tick - r.arrival_tick <= config.RIDER_PATIENCE)
        return gone

    def clear(self) -> list[Rider]:
        """Empty the queue and return whoever was in it."""
        riders = list(self._riders)
        self._riders.clear()
        return riders
