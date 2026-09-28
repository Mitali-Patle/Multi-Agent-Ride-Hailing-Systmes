"""Tests for every shock type, stacking, timing, and the sensor failure."""

from __future__ import annotations

import pytest

import config
from sim.market import MarketSim
from sim.shocks import ShockKind, ShockManager
from tests.helpers import assert_conserved, run_episode

HOLD = [[2, 1], [2, 1]]


def _offline(m: MarketSim) -> int:
    return int(m.driver_counts()[:, 2].sum())


def test_manager_lifecycle_and_stacking() -> None:
    mgr = ShockManager()
    mgr.trigger("demand_spike", duration=2)
    mgr.trigger(ShockKind.DEMAND_SPIKE, duration=4)
    started, expired = mgr.begin_tick(10)
    assert len(started) == 2 and not expired
    assert mgr.demand_multiplier() == pytest.approx(config.DEMAND_SPIKE_MULT**2)
    mgr.begin_tick(11)
    started, expired = mgr.begin_tick(12)
    assert len(expired) == 1 and mgr.demand_multiplier() == pytest.approx(config.DEMAND_SPIKE_MULT)
    mgr.begin_tick(14)
    assert mgr.demand_multiplier() == 1.0 and len(mgr.history) == 2


def test_demand_spike_raises_arrivals() -> None:
    base, spiked = MarketSim(2, seed=1), MarketSim(2, seed=1)
    for m in (base, spiked):
        for _ in range(60):
            m.step(HOLD)
    spiked.trigger_shock("demand_spike")
    arr_base = arr_spike = 0
    for _ in range(config.DEMAND_SPIKE_TICKS):
        arr_base += base.step(HOLD).arrivals
        arr_spike += spiked.step(HOLD).arrivals
    assert arr_spike > arr_base * 1.5
    rec = spiked.step(HOLD)
    assert rec.demand_mult == 1.0 and not rec.active_shocks


@pytest.mark.parametrize("n_platforms", [1, 2])
def test_driver_outage_and_recovery(n_platforms: int) -> None:
    m = MarketSim(n_platforms, seed=2)
    for _ in range(5):
        m.step([[2, 1]] * n_platforms)
    m.trigger_shock("driver_outage")
    expected = round(config.OUTAGE_FRACTION * config.N_DRIVERS)
    peak = 0
    for _ in range(config.OUTAGE_TICKS):
        m.step([[2, 1]] * n_platforms)
        assert_conserved(m)
        peak = max(peak, _offline(m))
    assert 0 < peak <= expected
    m.step([[2, 1]] * n_platforms)
    assert _offline(m) == 0


def test_back_to_back_and_overlapping_outages() -> None:
    m = MarketSim(2, seed=3)
    m.trigger_shock("driver_outage", duration=3)
    for _ in range(3):
        m.step(HOLD)
    m.trigger_shock("driver_outage", duration=3)  # starts the tick the first expires
    m.step(HOLD)
    assert len(m.shocks.active) == 1 and len(m.expired_shocks) == 1
    m.trigger_shock("driver_outage", duration=5)  # overlaps the second
    m.trigger_shock("demand_spike", duration=5)
    for _ in range(10):
        m.step(HOLD)
        assert_conserved(m)
    assert _offline(m) == 0
    run_episode(m)


def test_shock_on_final_tick_and_after_end() -> None:
    m = MarketSim(2, seed=4)
    while m.tick < config.EPISODE_TICKS - 1:
        m.step(HOLD)
    for kind in ShockKind:
        assert m.trigger_shock(kind, target_platform=1) is not None
    rec = m.step(HOLD)
    assert m.done and len(rec.active_shocks) == 3
    assert_conserved(m)
    assert m.trigger_shock("demand_spike") is None  # episode over: ignored, no crash


def test_sensor_failure_freezes_rival_feed_then_restores() -> None:
    m = MarketSim(2, seed=5)
    m.step([[2, 0], [1, 1]])
    frozen = m.observe(0)[5:7].copy()
    m.trigger_shock("sensor_failure", target_platform=0, duration=4)
    for _ in range(4):
        m.step([[2, 0], [4, 2]])  # B changes prices while A is blind
        obs_a, obs_b = m.observe(0), m.observe(1)
        assert obs_a[7] == 0.0 and (obs_a[5:7] == frozen).all()
        assert obs_b[7] == 1.0  # B's own feed is unaffected
    m.step([[2, 0], [4, 2]])
    obs_a = m.observe(0)
    assert obs_a[7] == 1.0 and obs_a[5] == 1.0 and obs_a[6] == 1.0


def test_sensor_failure_ignored_in_monopoly() -> None:
    m = MarketSim(1, seed=6)
    assert m.trigger_shock("sensor_failure") is None
    m.step([[2, 0]])
    assert m.observe(0)[7] == 1.0


def test_sensor_failure_bad_target_rejected() -> None:
    m = MarketSim(2, seed=6)
    with pytest.raises(ValueError):
        m.trigger_shock("sensor_failure", target_platform=5)
