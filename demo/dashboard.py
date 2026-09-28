"""Streamlit live demo: watch trained platforms compete, and hit them with shocks.

Launch with::

    streamlit run demo/dashboard.py

The dashboard only runs inference on saved models (it never trains). The
simulation lives in ``st.session_state`` between reruns. Autoplay runs inside
an ``st.fragment`` that re-executes on a timer, so sidebar buttons (including
the shock buttons) stay responsive while the simulation is playing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make project modules importable

import logging  # noqa: E402

import altair as alt  # noqa: E402
import streamlit as st  # noqa: E402

import config  # noqa: E402
from demo import panels  # noqa: E402
from demo.live_sim import LiveSimulation  # noqa: E402
from sim.market import clock_time  # noqa: E402
from sim.shocks import ShockKind  # noqa: E402

logger = logging.getLogger(__name__)
TRAIN_HELP = """No trained models found. From the project folder run:

```
python training/train_duopoly.py
python training/train_monopoly.py
```
then reload this page. (Or tick *Use rule-based baseline* in the sidebar to demo the market without RL.)"""


def build_sim(mode: str, seed: int, use_baseline: bool) -> LiveSimulation | None:
    """Return the session's simulation, rebuilding it when mode or policy source changes."""
    key = (mode, use_baseline)
    if st.session_state.get("sim_key") != key or "sim" not in st.session_state:
        try:
            st.session_state.sim = LiveSimulation(mode, seed, "baseline" if use_baseline else "trained")
            st.session_state.sim_key = key
        except FileNotFoundError as exc:
            st.session_state.pop("sim", None)
            st.session_state.pop("sim_key", None)
            st.error(f"**{exc}**")
            st.markdown(TRAIN_HELP)
            return None
    return st.session_state.sim


def sidebar() -> tuple[LiveSimulation | None, bool, float]:
    """Controls. Returns ``(sim, autoplay, ticks_per_second)``."""
    sb = st.sidebar
    sb.header("Controls")
    mode = sb.radio("Market", ["duopoly", "monopoly"], horizontal=True)
    seed = int(sb.number_input("Seed", min_value=0, max_value=1_000_000, value=config.SEED, step=1))
    use_baseline = sb.checkbox("Use rule-based baseline", value=False,
                               help="Fallback if trained models are missing; the demo is labelled accordingly.")
    sim = build_sim(mode, seed, use_baseline)
    if sim is None:
        return None, False, config.DASHBOARD_DEFAULT_SPEED
    col1, col2 = sb.columns(2)
    if col1.button("Reset"):
        sim.reset(seed)
    if col2.button("Step 1 tick", disabled=sim.done):
        sim.step()
    autoplay = sb.toggle("Autoplay", value=False)
    speed = sb.slider("Speed (ticks / second)", config.DASHBOARD_MIN_SPEED, config.DASHBOARD_MAX_SPEED,
                      config.DASHBOARD_DEFAULT_SPEED, 0.5)

    sb.subheader("Environmental shocks")
    if sb.button("Demand spike (concert ends)"):
        st.toast(sim.trigger_shock(ShockKind.DEMAND_SPIKE))
    if sb.button("Driver outage (heavy rain)"):
        st.toast(sim.trigger_shock(ShockKind.DRIVER_OUTAGE))
    options = list(range(sim.n_platforms)) if sim.n_platforms > 1 else [0]
    target = sb.selectbox("Platform losing its rival price feed", options, format_func=lambda p: sim.names[p],
                          disabled=sim.n_platforms < 2)
    if sb.button("Sensor failure (rival feed down)", disabled=sim.n_platforms < 2,
                 help="Only meaningful in duopoly: the platform's view of rival prices freezes."):
        st.toast(sim.trigger_shock(ShockKind.SENSOR_FAILURE, target))
    sb.caption(f"Shock sizes: demand x{config.DEMAND_SPIKE_MULT:g} for {config.DEMAND_SPIKE_TICKS} ticks; "
               f"{config.OUTAGE_FRACTION:.0%} of drivers offline for {config.OUTAGE_TICKS} ticks; "
               f"feed frozen for {config.SENSOR_FAIL_TICKS} ticks.")
    return sim, autoplay, speed


def header(sim: LiveSimulation) -> None:
    """Clock, status and active shocks."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Tick", f"{sim.tick} / {config.EPISODE_TICKS}")
    c2.metric("Time of day", clock_time(sim.tick))
    c3.metric("Market", sim.mode.capitalize())
    st.caption(f"Policies: **{sim.policy_label}** (inference only, no training) · seed {sim.seed}")
    active = sim.market.shocks.active_labels() if not sim.done else []
    pending = [s.label() for s in sim.market.shocks.pending]
    if active:
        st.error("ACTIVE SHOCKS: " + " · ".join(a.upper() for a in active))
    elif sim.done:
        st.info("Episode finished. Press Reset (or change the seed) to run again.")
    else:
        st.success("No active shocks")
    if pending and not sim.done:
        st.warning("Queued for next tick: " + ", ".join(pending))


def render(sim: LiveSimulation) -> None:
    """All live panels."""
    header(sim)
    hist = sim.history()
    if hist.empty:
        st.info("Press **Step 1 tick** or switch on **Autoplay** to start the day.")
        st.dataframe(sim.events_frame(), hide_index=True)
        return
    hist = panels.with_derived(hist)
    names, windows = sim.names, sim.shock_windows()
    ts = lambda col, title, y, step=True, fmt=",.0f": panels.timeseries(  # noqa: E731
        hist, col, title, y, names, windows, step, fmt)

    row = st.columns(3)
    row[0].altair_chart(panels.driver_states(hist, names))
    row[1].altair_chart(ts("fare_mult", "Fare multiplier posted", "x base fare", fmt=".2f"))
    row[2].altair_chart(ts("bonus", "Driver bonus posted", "per ride"))
    row = st.columns(3)
    row[0].altair_chart(ts("queue", "Riders queueing", "riders"))
    row[1].altair_chart(ts("cum_abandoned", "Riders who gave up (cumulative)", "riders"))
    row[2].altair_chart(ts("cum_profit", "Cumulative platform profit", "currency", step=False))
    row = st.columns(3)
    row[0].altair_chart(ts("share", "Market share of completed rides", "share", step=False, fmt=".0%"))
    row[1].altair_chart(ts("idle", "Idle drivers", "drivers"))
    with row[2]:
        w = panels.welfare_numbers(hist, sim.market.n_drivers)
        st.markdown("**Rider welfare and driver earnings (so far)**")
        a, b = st.columns(2)
        a.metric("Avg fare", f"{w['avg_fare']:,.0f}")
        b.metric("Wait (min)", f"{w['avg_wait']:.1f}")
        a.metric("Gave up", f"{w['abandon']:.1%}")
        b.metric("Pay/driver", f"{w['earn_per_driver']:,.0f}")
        st.metric("Rider surplus", f"{w['rider_surplus']:,.0f}")
    st.subheader("Event log")
    st.dataframe(sim.events_frame(), hide_index=True, height=280)


def live_panel(autoplay: bool) -> None:
    """Advance one tick when autoplaying, then redraw. Runs as a timed fragment."""
    sim: LiveSimulation = st.session_state.sim
    if autoplay and not sim.done:
        try:
            sim.step()
        except Exception as exc:  # the live demo must never crash on stage
            st.error(f"Simulation step failed: {exc}. Press Reset to continue.")
            return
        if sim.done:
            st.rerun()  # full rerun switches the autoplay timer off
    try:
        render(sim)
    except Exception as exc:  # a drawing glitch must not stop the simulation
        logger.exception("render failed")
        st.warning(f"Could not draw a panel ({exc}); the simulation keeps running.")


def main() -> None:
    """Page entry point."""
    st.set_page_config(page_title="Ride-hailing MARL live demo", layout="wide")
    alt.data_transformers.disable_max_rows()
    st.title("Competitive ride-hailing: live multi-agent simulation")
    st.caption("Two independently trained PPO platforms (or one monopolist) set fares and driver bonuses each "
               "5-minute tick; 50 rule-based drivers and Poisson-arriving riders react in real time.")
    sim, autoplay, speed = sidebar()
    if sim is None:
        return
    interval = 1.0 / speed if autoplay and not sim.done else None
    st.fragment(live_panel, run_every=interval)(autoplay)


main()
