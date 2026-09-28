"""Tests for the shared market: invariants, determinism, and edge cases."""

from __future__ import annotations

import numpy as np
import pytest

import config
from sim.drivers import DriverState
from sim.market import MarketSim, clock_time
from sim.riders import estimate_wait
from tests.helpers import assert_conserved, random_actions, run_episode


@pytest.mark.parametrize("n_platforms", [1, 2])
def test_driver_conservation_every_tick(n_platforms: int) -> None:
    for seed in range(20):
        run_episode(MarketSim(n_platforms, seed=seed), seed=seed)


@pytest.mark.parametrize("n_platforms", [1, 2])
def test_same_seed_same_result(n_platforms: int) -> None:
    summaries = []
    for _ in range(2):
        m = MarketSim(n_platforms, seed=123)
        run_episode(m, seed=7, check=False)
        summaries.append(m.metrics.summary())
    assert summaries[0] == summaries[1]


def test_different_seed_differs() -> None:
    a, b = MarketSim(2, seed=1), MarketSim(2, seed=2)
    run_episode(a, check=False)
    run_episode(b, check=False)
    assert a.metrics.summary() != b.metrics.summary()


def test_monopoly_drivers_never_switch() -> None:
    m = MarketSim(1, seed=3)
    run_episode(m)
    assert sum(pt.switches_in + pt.switches_out for r in m.metrics.records for pt in r.platforms) == 0
    assert all(d.platform == 0 for d in m.pool.drivers)


def test_duopoly_drivers_do_switch_toward_bonus() -> None:
    m = MarketSim(2, seed=5)
    while not m.done:
        m.step([[2, 2], [2, 0]])  # A pays the high bonus, B pays none
    counts = m.driver_counts().sum(axis=1)
    assert counts[0] > counts[1]


def test_zero_idle_drivers_and_zero_drivers() -> None:
    assert estimate_wait(idle=0, busy=0, queue_len=5) == config.MAX_WAIT_ESTIMATE
    for n in (0, 1, 2):
        m = MarketSim(2, seed=11, n_drivers=n)
        run_episode(m)
        s = m.metrics.summary()
        assert np.isfinite(list(s.values())).all()


def test_all_drivers_on_one_platform() -> None:
    m = MarketSim(2, seed=4)
    for d in m.pool.drivers:
        d.platform = 0
    m.step([[2, 0], [0, 2]])
    assert_conserved(m)
    obs_b = m.observe(1)
    assert obs_b.shape == (config.OBS_DIM,)
    assert obs_b[0] == 0.0 or m.driver_counts()[1].sum() > 0
    run_episode(m)


def test_negative_profit_is_possible_and_handled() -> None:
    m = MarketSim(2, seed=8)
    while not m.done:
        m.step([[0, 2], [2, 0]])  # A: cheapest fare with the highest bonus
    s = m.metrics.summary()
    assert s["profit_p0"] < 0
    assert np.isfinite(s["total_welfare"])


def test_reset_mid_episode_restores_clean_state() -> None:
    m = MarketSim(2, seed=9)
    for _ in range(40):
        m.step([[1, 1], [3, 2]])
    m.trigger_shock("driver_outage")
    m.step([[1, 1], [3, 2]])
    m.reset(seed=9)
    assert m.tick == 0 and not m.shocks.active and not m.metrics.records
    assert all(d.state is DriverState.IDLE for d in m.pool.drivers)
    fresh = MarketSim(2, seed=9)
    run_episode(m, seed=1, check=False)
    run_episode(fresh, seed=1, check=False)
    assert m.metrics.summary() == fresh.metrics.summary()


def test_step_after_done_raises_clear_error() -> None:
    m = MarketSim(1, seed=0)
    run_episode(m, check=False)
    with pytest.raises(RuntimeError, match="reset"):
        m.step([[2, 0]])


def test_bad_actions_are_clipped_not_fatal() -> None:
    m = MarketSim(2, seed=0)
    m.step([[99, -5], [np.nan, 1.7]])
    assert m.fare_idx == [len(config.FARE_MULTIPLIERS) - 1, 0]
    assert m.bonus_idx == [0, 2]
    with pytest.raises(ValueError):
        m.step([[1, 1]])  # wrong number of platforms


def test_invalid_platform_count() -> None:
    with pytest.raises(ValueError):
        MarketSim(3)


def test_all_matched_rides_are_paid_by_close() -> None:
    m = MarketSim(2, seed=2)
    run_episode(m, check=False)
    s = m.metrics.summary()
    matched = m.metrics.totals("matched").sum()
    assert s["completed"] == matched
    assert all(d.state is not DriverState.BUSY for d in m.pool.drivers)


@pytest.mark.parametrize("n_platforms", [1, 2])
def test_every_request_is_served_or_abandoned_by_close(n_platforms: int) -> None:
    m = MarketSim(n_platforms, seed=3, n_drivers=5)  # scarce drivers, so riders are queued at closing time
    run_episode(m, check=False)
    assert m.metrics.records[-1].platforms[0].abandoned > 0  # the closing flush actually happened
    assert m.queue_lengths() == [0] * n_platforms
    totals = m.metrics.totals
    assert totals("requests").sum() == totals("matched").sum() + totals("abandoned").sum()


def test_observation_is_normalised_and_hides_rival_supply() -> None:
    m = MarketSim(2, seed=1)
    rng = np.random.default_rng(0)
    for _ in range(50):
        m.step(random_actions(rng, 2))
        for p in range(2):
            obs = m.observe(p)
            assert obs.dtype == np.float32 and obs.shape == (config.OBS_DIM,)
            assert (obs >= 0).all() and (obs <= 1).all()
    mono = MarketSim(1, seed=1)
    obs = mono.observe(0)
    assert obs[5] == 0 and obs[6] == 0 and obs[7] == 1


def test_welfare_identity() -> None:
    m = MarketSim(2, seed=6)
    run_episode(m, check=False)
    s = m.metrics.summary()
    assert s["total_welfare"] == pytest.approx(s["rider_surplus"] + s["driver_earnings_total"] + s["platform_profit"])


def test_clock_time_labels() -> None:
    assert clock_time(0) == "07:00"
    assert clock_time(config.EPISODE_TICKS) == "19:00"
