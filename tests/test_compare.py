"""Tests for the side-by-side monopoly vs duopoly simulation behind the dashboard."""

from __future__ import annotations

import numpy as np

import config
from demo.compare import DUO, DUO_A, DUO_B, MONO, ComparisonSimulation, chart_frames
from sim.market import MarketSim
from tests.helpers import random_actions


def _arrivals(market: MarketSim) -> list[int]:
    return [r.arrivals for r in market.metrics.records]


def test_common_demand_gives_identical_riders_across_market_structures() -> None:
    mono, duo = MarketSim(1, seed=5, common_demand=True), MarketSim(2, seed=5, common_demand=True)
    rng = np.random.default_rng(0)
    for market in (mono, duo):
        market.trigger_shock("demand_spike")
    while not duo.done:
        if duo.tick == 70:
            for market in (mono, duo):
                market.trigger_shock("driver_outage")
        mono.step(random_actions(rng, 1))
        duo.step(random_actions(rng, 2))
    assert _arrivals(mono) == _arrivals(duo)


def test_common_demand_is_off_by_default() -> None:
    default, legacy = MarketSim(2, seed=3), MarketSim(2, seed=3)
    assert default.demand_rng is default.rng
    for market in (default, legacy):
        while not market.done:
            market.step([[2, 0], [2, 0]])
    assert default.metrics.summary() == legacy.metrics.summary()


def test_comparison_steps_and_shocks_both_markets_in_lockstep() -> None:
    sim = ComparisonSimulation(seed=1, policy_source="baseline")
    sim.trigger_shock("driver_outage")
    assert "duopoly only" in sim.trigger_shock("sensor_failure", 1)
    assert sim.advance(5) == 5
    assert sim.mono.tick == sim.duo.tick == 5
    assert len(sim.duo.market.shocks.active) == 2 and len(sim.mono.market.shocks.active) == 1
    assert _arrivals(sim.mono.market) == _arrivals(sim.duo.market)
    assert sim.advance(10_000) == config.EPISODE_TICKS - 5 and sim.done and not sim.step()
    assert set(sim.events_frame()["market"]) == {MONO, DUO}
    sim.reset(2)
    assert sim.tick == 0 and sim.seed == 2 and chart_frames(sim) == {}


def test_chart_frames_have_every_metric_for_both_markets() -> None:
    sim = ComparisonSimulation(seed=4, policy_source="baseline")
    sim.advance(30)
    frames = chart_frames(sim)
    for key in ("fare", "served", "rides", "welfare", "wait"):
        assert set(frames[key]["series"]) == {MONO, DUO}, key
        assert len(frames[key]) == 2 * 30
    assert set(frames["share"]["series"]) == {MONO, DUO_A, DUO_B}
    shares = frames["share"].pivot(index="tick", columns="series", values="value")
    assert np.allclose(shares[[DUO_A, DUO_B]].sum(axis=1, min_count=1).dropna(), 1.0)
    assert np.allclose(shares[MONO].dropna(), 1.0)
    served = frames["served"]["value"].dropna()
    assert ((served >= 0) & (served <= 1)).all()
    rides = frames["rides"].pivot(index="tick", columns="series", values="value")
    assert (rides.diff().dropna() >= 0).all().all()  # cumulative, never decreasing
