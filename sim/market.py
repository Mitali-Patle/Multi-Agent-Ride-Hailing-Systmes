"""MarketSim: the shared world that one or two platforms compete in.

The market owns every rule-based agent (drivers, riders), the shock manager,
and the metric tracker. Platforms are external: each tick they hand in an
action ``(fare_index, bonus_index)`` and read back an observation. The same
code runs a monopoly (``n_platforms=1``) and a duopoly (``n_platforms=2``).

Order of events within one tick (``MarketSim.step``):

1. Apply active shocks: expire finished shocks (outage drivers return),
   then activate newly triggered ones (outage drivers log off).
2. Platforms post their actions (fare level, driver bonus).
3. Every driver updates its earnings beliefs from the posted terms; a random
   subset of idle drivers may switch platform (duopoly only).
4. Riders: queued riders past their patience abandon, remaining queued
   riders are served first (FIFO), then new arrivals get quotes, choose a
   platform or the outside option, and are matched or queued. On the final
   tick, riders still queued when the market closes are counted as
   abandoned and charged the waiting cost up to closing time.
5. Busy drivers advance one tick; finished rides complete and pay out. On
   the final tick every ride still in progress is completed (the market
   closes), so no matched ride goes unpaid.
6. Record metrics and refresh each platform's observation (a platform whose
   rival feed has failed keeps the last rival prices it saw).

Invariants, checked by the tests: idle + busy + offline over all platforms
equals the number of drivers after every tick, and no count is negative; by
closing time every rider who chose a platform was either served or abandoned.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

import config
from sim.drivers import DriverPool, driver_pay_per_ride
from sim.metrics import EpisodeMetrics, PlatformTick, TickRecord
from sim.riders import OUTSIDE_OPTION, Rider, RiderQueue, choose_platform, draw_arrivals, estimate_wait, rider_surplus
from sim.shocks import Shock, ShockKind, ShockManager

logger = logging.getLogger(__name__)


def clock_time(tick: int) -> str:
    """Wall-clock label (HH:MM) for a tick, e.g. tick 0 -> '07:00'."""
    minutes = int(round(config.START_HOUR * 60 + tick * config.TICK_MINUTES))
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


def fare_price(fare_idx: int) -> float:
    """Posted price for a fare level."""
    return config.BASE_FARE * config.FARE_MULTIPLIERS[fare_idx]


class MarketSim:
    """The shared ride-hailing market (1 zone, 1 or 2 platforms)."""

    def __init__(self, n_platforms: int | None = None, seed: int | None = None, n_drivers: int | None = None) -> None:
        """Build a market.

        Args:
            n_platforms: 1 (monopoly) or 2 (duopoly); defaults to config.
            seed: Seed for all randomness in this market.
            n_drivers: Override ``config.N_DRIVERS`` (mainly for tests).
        """
        self.n_platforms = config.N_PLATFORMS if n_platforms is None else int(n_platforms)
        if not 1 <= self.n_platforms <= config.MAX_PLATFORMS:
            raise ValueError(f"n_platforms must be 1..{config.MAX_PLATFORMS}, got {self.n_platforms}")
        self.n_drivers = config.N_DRIVERS if n_drivers is None else int(n_drivers)
        if self.n_drivers < 0:
            raise ValueError("n_drivers must be non-negative")
        self.shocks = ShockManager()
        self.reset(seed)

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int | None = None) -> None:
        """Start a fresh episode. Safe to call at any time, even mid-episode."""
        P = self.n_platforms
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.pool = DriverPool(P, self.rng, self.n_drivers)
        self.queues = [RiderQueue() for _ in range(P)]
        self.shocks.reset()
        self.fare_idx = [config.DEFAULT_FARE_INDEX] * P
        self.bonus_idx = [config.DEFAULT_BONUS_INDEX] * P
        self.utilization = np.full(P, config.UTILIZATION_PRIOR)
        self.observed_rival = [(config.DEFAULT_FARE_INDEX, config.DEFAULT_BONUS_INDEX)] * P
        self.feed_ok = [True] * P
        self.last_arrivals = 0
        self.last_abandoned = [0] * P
        self.last_switches: list = []
        self.started_shocks: list[Shock] = []
        self.expired_shocks: list[Shock] = []
        self.metrics = EpisodeMetrics(P, self.n_drivers)
        self.last_record: TickRecord | None = None
        self._next_rider_id = 0
        self._counts = self.pool.counts()
        self.pool.init_beliefs(self._earnings_signal())

    @property
    def done(self) -> bool:
        """True once all ``EPISODE_TICKS`` ticks have been simulated."""
        return self.tick >= config.EPISODE_TICKS

    def driver_counts(self) -> np.ndarray:
        """Matrix ``[platform, (idle, busy, offline)]`` as of the last tick."""
        return self._counts.copy()

    def queue_lengths(self) -> list[int]:
        """Number of riders waiting on each platform."""
        return [len(q) for q in self.queues]

    # ----------------------------------------------------------------- shocks
    def trigger_shock(
        self, kind: ShockKind | str, target_platform: int | None = None, duration: int | None = None
    ) -> Shock | None:
        """Queue a shock that takes effect at the start of the next tick.

        Returns ``None`` (and logs why) when the shock cannot have any effect:
        the episode is already over, or a sensor failure in monopoly mode
        (there is no rival feed to lose).
        """
        kind = ShockKind(kind)
        if self.done:
            logger.warning("episode finished; ignoring %s shock", kind.value)
            return None
        if kind is ShockKind.SENSOR_FAILURE:
            if self.n_platforms < 2:
                logger.info("monopoly has no rival feed; ignoring sensor failure")
                return None
            target_platform = 0 if target_platform is None else int(target_platform)
            if not 0 <= target_platform < self.n_platforms:
                raise ValueError(f"target_platform {target_platform} out of range")
        else:
            target_platform = None
        return self.shocks.trigger(kind, target_platform, duration)

    # ------------------------------------------------------------------- step
    def step(self, actions: Sequence[Sequence[int]] | np.ndarray) -> TickRecord:
        """Advance the market by one tick. See the module docstring for the order.

        Args:
            actions: One ``(fare_index, bonus_index)`` pair per platform.
                Out-of-range or non-finite values are clipped, not fatal.

        Returns:
            The :class:`TickRecord` for this tick.
        """
        if self.done:
            raise RuntimeError("episode finished; call reset() before stepping again")
        acts = self._validate_actions(actions)
        t = self.tick
        rec = TickRecord(tick=t, platforms=[PlatformTick() for _ in range(self.n_platforms)])
        self._apply_shocks(t, rec)
        self._post_actions(acts, rec)
        self._driver_switching(rec)
        final = t == config.EPISODE_TICKS - 1
        self._match_riders(t, rec)
        if final:
            self._close_queues(rec)
        self._advance_rides(rec, final=final)
        self._finish_tick(rec)
        self.tick += 1
        return rec

    def _validate_actions(self, actions: Sequence[Sequence[int]] | np.ndarray) -> list[tuple[int, int]]:
        """Coerce actions to one in-range ``(fare, bonus)`` int pair per platform."""
        arr = np.asarray(actions, dtype=float).reshape(-1)
        if arr.size != 2 * self.n_platforms:
            raise ValueError(f"expected {self.n_platforms} (fare, bonus) pairs, got shape {np.shape(actions)}")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).reshape(self.n_platforms, 2)
        hi = np.array([len(config.FARE_MULTIPLIERS) - 1, len(config.BONUS_LEVELS) - 1])
        clipped = np.clip(np.rint(arr), 0, hi).astype(int)
        if not np.array_equal(clipped, arr):
            logger.debug("clipped actions %s -> %s", arr.tolist(), clipped.tolist())
        return [(int(f), int(b)) for f, b in clipped]

    def _apply_shocks(self, t: int, rec: TickRecord) -> None:
        """Step 1: expire and start shocks, applying outage effects to drivers."""
        started, expired = self.shocks.begin_tick(t)
        for shock in expired:
            if shock.kind is ShockKind.DRIVER_OUTAGE:
                self.pool.bring_online(shock.affected_drivers)
        for shock in started:
            if shock.kind is ShockKind.DRIVER_OUTAGE:
                shock.affected_drivers = self.pool.take_offline(round(config.OUTAGE_FRACTION * self.n_drivers))
        self.started_shocks, self.expired_shocks = started, expired
        rec.demand_mult = self.shocks.demand_multiplier()
        rec.active_shocks = self.shocks.active_labels()

    def _post_actions(self, acts: list[tuple[int, int]], rec: TickRecord) -> None:
        """Step 2: platforms post fare and bonus."""
        for p, (f, b) in enumerate(acts):
            self.fare_idx[p], self.bonus_idx[p] = f, b
            pt = rec.platforms[p]
            pt.fare_idx, pt.bonus_idx = f, b
            pt.fare, pt.bonus = fare_price(f), config.BONUS_LEVELS[b]

    def _earnings_signal(self) -> np.ndarray:
        """Public estimate of driver earnings per tick on each platform."""
        pay = np.array(
            [driver_pay_per_ride(fare_price(self.fare_idx[p]), config.BONUS_LEVELS[self.bonus_idx[p]])
             for p in range(self.n_platforms)]
        )
        return pay * self.utilization / config.MEAN_RIDE_TICKS

    def _driver_switching(self, rec: TickRecord) -> None:
        """Step 3: drivers update beliefs; some idle drivers switch."""
        self.pool.update_beliefs(self._earnings_signal())
        self.last_switches = self.pool.consider_switching()
        for sw in self.last_switches:
            rec.platforms[sw.from_platform].switches_out += 1
            rec.platforms[sw.to_platform].switches_in += 1

    def _match_riders(self, t: int, rec: TickRecord) -> None:
        """Step 4: abandonment, queue service, then new arrivals."""
        P = self.n_platforms
        idle = [list(self.rng.permutation(self.pool.idle_ids(p))) for p in range(P)]
        busy = self.pool.counts()[:, 1].copy()
        for p in range(P):
            pt = rec.platforms[p]
            for r in self.queues[p].pop_abandoned(t):
                pt.abandoned += 1
                pt.rider_surplus -= config.WAIT_COST * (t - r.arrival_tick)
            while idle[p] and len(self.queues[p]):
                self._match(self.queues[p].pop(), int(idle[p].pop()), t, pt)
                busy[p] += 1
        riders = draw_arrivals(self.rng, t, rec.demand_mult, self._next_rider_id)
        self._next_rider_id += len(riders)
        rec.arrivals = len(riders)
        prices = np.array([fare_price(f) for f in self.fare_idx])
        for rider in riders:
            waits = np.array([estimate_wait(len(idle[p]), busy[p], len(self.queues[p])) for p in range(P)])
            choice = choose_platform(rider, prices, waits, self.rng)
            if choice == OUTSIDE_OPTION:
                rec.outside += 1
                continue
            rider.platform, rider.price = choice, float(prices[choice])
            pt = rec.platforms[choice]
            pt.requests += 1
            if idle[choice]:
                self._match(rider, int(idle[choice].pop()), t, pt)
                busy[choice] += 1
            else:
                self.queues[choice].push(rider)

    def _close_queues(self, rec: TickRecord) -> None:
        """Market closes: riders still queued go unserved and count as abandoned.

        Like riders who ran out of patience, they bear the waiting cost of the
        time they queued, here up to closing time.
        """
        for p, queue in enumerate(self.queues):
            pt = rec.platforms[p]
            for r in queue.clear():
                pt.abandoned += 1
                pt.rider_surplus -= config.WAIT_COST * (config.EPISODE_TICKS - r.arrival_tick)

    def _match(self, rider: Rider, driver_id: int, t: int, pt: PlatformTick) -> None:
        """Pair a rider with an idle driver; the bonus is locked at match time."""
        wait = config.PICKUP_TICKS + (t - rider.arrival_tick)
        self.pool.assign_ride(driver_id, rider.price, config.BONUS_LEVELS[self.bonus_idx[rider.platform]])
        pt.matched += 1
        pt.wait_ticks += wait
        pt.rider_surplus += rider_surplus(rider, wait)

    def _advance_rides(self, rec: TickRecord, final: bool) -> None:
        """Step 5: rides progress; completed rides pay drivers and platforms."""
        completions = self.pool.advance()
        if final:
            completions += self.pool.settle_all()
        for c in completions:
            pt = rec.platforms[c.platform]
            pt.completed += 1
            pt.fares_collected += c.fare
            pt.bonus_paid += c.bonus
            pt.driver_earnings += c.driver_pay
            pt.profit += config.COMMISSION * c.fare - c.bonus

    def _finish_tick(self, rec: TickRecord) -> None:
        """Step 6: snapshot counts, utilisation, rival feeds; store metrics."""
        self._counts = self.pool.counts()
        for p, pt in enumerate(rec.platforms):
            pt.idle, pt.busy, pt.offline = (int(x) for x in self._counts[p])
            pt.queue = len(self.queues[p])
            active = pt.idle + pt.busy
            if active > 0:
                self.utilization[p] = min(1.0, (pt.busy + pt.queue) / active)
            else:
                self.utilization[p] = 1.0 if pt.queue > 0 else config.UTILIZATION_PRIOR
            ok = self.shocks.feed_ok(p)
            self.feed_ok[p] = pt.feed_ok = ok
            if ok and self.n_platforms > 1:
                rival = 1 - p
                self.observed_rival[p] = (self.fare_idx[rival], self.bonus_idx[rival])
        self.last_arrivals = rec.arrivals
        self.last_abandoned = [pt.abandoned for pt in rec.platforms]
        self.metrics.add(rec)
        self.last_record = rec

    # ------------------------------------------------------------ observation
    def observe(self, platform_id: int) -> np.ndarray:
        """Partial observation for one platform, normalised to [0, 1].

        Layout (``config.OBS_DIM`` = 11): own idle, own busy, own queue,
        own fare idx, own bonus idx, rival fare idx, rival bonus idx,
        rival_feed_ok, arrivals last tick, own abandoned last tick,
        time of day. The rival's drivers and internals are never visible.
        """
        n_f, n_b = len(config.FARE_MULTIPLIERS) - 1, len(config.BONUS_LEVELS) - 1
        n = max(1, self.n_drivers)
        if self.n_platforms > 1:
            rf, rb = self.observed_rival[platform_id]
            rival = [rf / n_f, rb / n_b, 1.0 if self.feed_ok[platform_id] else 0.0]
        else:
            rival = [0.0, 0.0, 1.0]
        obs = [
            self._counts[platform_id, 0] / n,
            self._counts[platform_id, 1] / n,
            len(self.queues[platform_id]) / config.QUEUE_NORM,
            self.fare_idx[platform_id] / n_f,
            self.bonus_idx[platform_id] / n_b,
            *rival,
            self.last_arrivals / config.ARRIVALS_NORM,
            self.last_abandoned[platform_id] / config.ABANDONED_NORM,
            self.tick / config.EPISODE_TICKS,
        ]
        return np.clip(np.asarray(obs, dtype=np.float32), 0.0, 1.0)
