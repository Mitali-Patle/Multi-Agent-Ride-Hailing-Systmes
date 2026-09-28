"""Tests for the demo layer: live simulation, CLI scenarios, and the dashboard."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import config
from demo.live_sim import LiveSimulation
from demo.run_scenario import SCENARIOS, run_scenario, schedule_for
from sim.shocks import ShockKind

DASHBOARD = str(Path(__file__).resolve().parents[1] / "demo" / "dashboard.py")


@pytest.mark.parametrize("mode", ["duopoly", "monopoly"])
def test_run_scenario_all_completes(mode: str, tmp_path: Path) -> None:
    summary = run_scenario("all", mode, seed=3, policy="baseline", out_dir=tmp_path, verbose=False)
    assert summary["completed"] > 0
    assert (tmp_path / f"scenario_all_{mode}.png").exists()


def test_every_scenario_has_a_valid_schedule() -> None:
    for name in SCENARIOS:
        for tick, kind, _ in schedule_for(name):
            assert 0 <= tick < config.EPISODE_TICKS
            ShockKind(kind)
    ticks = [t for t, _, _ in schedule_for("all")]
    assert config.EPISODE_TICKS - 1 in ticks  # a shock on the final tick


def test_live_sim_events_and_ignored_shocks() -> None:
    sim = LiveSimulation("monopoly", seed=1, policy_source="baseline")
    assert "ignored" in sim.trigger_shock("sensor_failure")
    sim.trigger_shock("driver_outage")
    sim.trigger_shock("demand_spike")
    while sim.step() is not None:
        pass
    assert sim.done and sim.step() is None
    kinds = set(sim.events_frame()["type"])
    assert {"shock", "market"} <= kinds
    assert "ignored" in sim.trigger_shock("demand_spike")
    sim.reset(2)
    assert sim.tick == 0 and len(sim.events) == 1


def test_live_sim_history_and_windows() -> None:
    sim = LiveSimulation("duopoly", seed=4, policy_source="baseline")
    sim.trigger_shock("sensor_failure", 1)
    for _ in range(5):
        sim.step()
    hist = sim.history()
    assert set(hist["name"]) == set(config.PLATFORM_NAMES)
    assert sim.shock_windows()[0][2] == "sensor_failure"


def _app() -> AppTest:
    return AppTest.from_file(DASHBOARD, default_timeout=60)


def test_dashboard_missing_models_shows_instructions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    at = _app().run()
    assert not at.exception
    assert any("train_duopoly" in e.value for e in at.error)


def test_dashboard_controls_and_shocks_never_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)  # no models: use the labelled baseline fallback
    at = _app().run()
    at.checkbox[0].check().run()
    for label in ("Step 1 tick", "Demand spike (concert ends)", "Driver outage (heavy rain)",
                  "Sensor failure (rival feed down)", "Step 1 tick", "Step 1 tick"):
        next(b for b in at.button if b.label == label).click().run()
        assert not at.exception, at.exception
    sim = at.session_state["sim"]
    assert sim.tick == 3 and sim.market.shocks.active
    at.radio[0].set_value("monopoly").run()
    next(b for b in at.button if b.label == "Step 1 tick").click().run()
    next(b for b in at.button if b.label == "Reset").click().run()
    assert not at.exception
    assert at.session_state["sim"].mode == "monopoly"
