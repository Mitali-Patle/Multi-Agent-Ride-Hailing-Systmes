"""Streamlit live demo: monopoly vs duopoly, side by side, under live shocks.

Launch with::

    streamlit run demo/dashboard.py

Two markets run in lockstep: one trained monopolist, and two trained
competing platforms. They get the same riders (same seed, shared arrival
stream) and every shock button hits both at the same moment, so any gap
between the lines comes from market structure alone. The page shows six
charts: average fare, riders who got a ride, rides served, total welfare,
average market share, and rider wait time (the one trade-off).

The dashboard only runs inference on saved models (it never trains). The
simulation lives in ``st.session_state`` between reruns; autoplay runs in an
``st.fragment`` on a timer, so the sidebar buttons stay responsive while the
day is playing.
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
from demo.compare import ComparisonSimulation, chart_frames  # noqa: E402
from sim.market import clock_time  # noqa: E402
from sim.shocks import ShockKind  # noqa: E402

logger = logging.getLogger(__name__)
TRAIN_HELP = """Train the models first. From the project folder run:

```
python training/train_duopoly.py
python training/train_monopoly.py
```
then reload this page."""
SHOCK_HELP = {
    ShockKind.DEMAND_SPIKE: (f"A concert just ended: {config.DEMAND_SPIKE_MULT:g}x more riders for "
                             f"{config.DEMAND_SPIKE_TICKS * config.TICK_MINUTES} minutes."),
    ShockKind.DRIVER_OUTAGE: (f"Heavy rain: {config.OUTAGE_FRACTION:.0%} of drivers log off for "
                              f"{config.OUTAGE_TICKS * config.TICK_MINUTES} minutes."),
    ShockKind.SENSOR_FAILURE: (f"The chosen platform's feed of rival prices freezes for "
                               f"{config.SENSOR_FAIL_TICKS * config.TICK_MINUTES} minutes (duopoly only)."),
}
COMPACT_CSS = ("<style>.block-container, [data-testid='stMainBlockContainer']"
               "{padding-top:2rem; padding-bottom:1rem}</style>")  # fit all five charts on one screen
BADGE = ('<span style="background:{bg};color:{fg};padding:3px 10px;border-radius:12px;'
         'margin-right:6px;font-size:0.9rem;font-weight:600">{text}</span>')


def load_sim(seed: int) -> ComparisonSimulation | None:
    """The session's side-by-side simulation, or ``None`` (with instructions) if models are missing."""
    if "sim" in st.session_state:
        return st.session_state.sim
    source = "baseline" if st.session_state.get("use_baseline") else "trained"
    try:
        st.session_state.sim = ComparisonSimulation(seed, source)
    except FileNotFoundError as exc:
        st.error(f"**{exc}**")
        st.markdown(TRAIN_HELP)
        if st.button("Run with the rule-based baseline instead"):
            st.session_state.use_baseline = True
            st.rerun()
        return None
    return st.session_state.sim


def sidebar() -> tuple[ComparisonSimulation | None, bool, float]:
    """Controls. Returns ``(sim, autoplay, ticks_per_second)``."""
    sb = st.sidebar
    sb.header("Simulation")
    seed = int(sb.number_input("Seed", min_value=0, max_value=1_000_000, value=config.SEED, step=1,
                               help="Same seed = same riders. Press Reset to apply a new seed."))
    sim = load_sim(seed)
    if sim is None:
        return None, False, config.DASHBOARD_DEFAULT_SPEED
    reset, tick, hour = sb.columns(3)
    if reset.button("Reset", help="Restart the day at 07:00 with this seed"):
        sim.reset(seed)
    if tick.button("+1 tick", disabled=sim.done, help="Advance 5 minutes"):
        sim.step()
    if hour.button("+1 hour", disabled=sim.done, help="Fast-forward one hour, e.g. to reach rush hour"):
        sim.advance(config.DASHBOARD_FAST_FORWARD_TICKS)
    autoplay = sb.toggle("Autoplay", value=False)
    speed = sb.slider("Speed (ticks per second)", config.DASHBOARD_MIN_SPEED, config.DASHBOARD_MAX_SPEED,
                      config.DASHBOARD_DEFAULT_SPEED, 0.5)

    sb.header("Shocks")
    sb.caption("Each shock hits both markets at the same moment. Biggest effect at rush hour "
               "(08:00-09:30, 16:30-18:30).")
    if sb.button("Demand spike", help=SHOCK_HELP[ShockKind.DEMAND_SPIKE]):
        st.toast(sim.trigger_shock(ShockKind.DEMAND_SPIKE))
    if sb.button("Driver outage", help=SHOCK_HELP[ShockKind.DRIVER_OUTAGE]):
        st.toast(sim.trigger_shock(ShockKind.DRIVER_OUTAGE))
    blind = sb.radio("Platform that loses its rival feed", config.PLATFORM_NAMES, horizontal=True)
    if sb.button("Sensor failure", help=SHOCK_HELP[ShockKind.SENSOR_FAILURE]):
        st.toast(sim.trigger_shock(ShockKind.SENSOR_FAILURE, config.PLATFORM_NAMES.index(blind)))
    return sim, autoplay, speed


def status(sim: ComparisonSimulation) -> None:
    """Clock, day progress, and shock badges in one compact row."""
    clock, bar = st.columns([1, 5], vertical_alignment="center")
    clock.metric("Time", clock_time(sim.tick))
    with bar:
        st.progress(sim.tick / config.EPISODE_TICKS,
                    text=f"Tick {sim.tick} of {config.EPISODE_TICKS}  ·  policies: {sim.policy_label}")
        badges = [BADGE.format(bg="#FEE2E2", fg="#B91C1C", text=f"ACTIVE: {s}") for s in sim.active_shocks()]
        badges += [BADGE.format(bg="#FEF3C7", fg="#92400E", text=f"Next tick: {s}") for s in sim.pending_shocks()]
        if sim.done:
            badges = [BADGE.format(bg="#E0E7FF", fg="#3730A3", text="Day over: press Reset to run again")]
        elif not badges:
            badges = [BADGE.format(bg="#F3F4F6", fg="#4B5563", text="No active shocks")]
        st.markdown("".join(badges), unsafe_allow_html=True)


def panel(container: st.delta_generator.DeltaGenerator, title: str, readout: str, chart: alt.TopLevelMixin) -> None:
    """Title, one-line monopoly-vs-duopoly readout, then the chart."""
    with container:
        st.markdown(f"**{title}**  \n<span style='color:#4B5563'>{readout}</span>", unsafe_allow_html=True)
        st.altair_chart(chart)


def charts(sim: ComparisonSimulation) -> None:
    """Six comparison charts: what competition gives riders, then the one trade-off (waiting)."""
    frames = chart_frames(sim)
    if not frames:
        st.info("Press **+1 tick**, **+1 hour** or switch on **Autoplay** to start the day.")
        return
    shocks = sim.shock_windows()
    chart, readout = panels.comparison_chart, panels.gap_readout
    day = "Day so far"
    top = st.columns(3)
    panel(top[0], "Average fare paid", readout(frames["fare"], "fare", day),
          chart(frames["fare"], "fare per ride", ",.0f", shocks))
    panel(top[1], "Riders who got a ride", readout(frames["served"], "percent", day),
          chart(frames["served"], "share of arriving riders", ".0%", shocks))
    panel(top[2], "Rides served", readout(frames["rides"], "count", day),
          chart(frames["rides"], "rides (cumulative)", ",.0f", shocks))
    bottom = st.columns(3)
    panel(bottom[0], "Total welfare", readout(frames["welfare"], "money", day),
          chart(frames["welfare"], "riders + drivers + platforms", ",.0f", shocks))
    panel(bottom[1], "Average market share", panels.share_readout(frames["share"]),
          chart(frames["share"], "share of rides (day so far)", ".0%", shocks))
    panel(bottom[2], "Rider wait time: the trade-off", readout(frames["wait"], "minutes"),
          chart(frames["wait"], "minutes (30-min avg)", ".1f", shocks))


def live_panel(autoplay: bool) -> None:
    """Advance one tick when autoplaying, then redraw. Runs as a timed fragment."""
    sim: ComparisonSimulation = st.session_state.sim
    if autoplay and not sim.done:
        try:
            sim.step()
        except Exception as exc:  # the live demo must never crash on stage
            logger.exception("step failed")
            st.error(f"Simulation step failed: {exc}. Press Reset to continue.")
            return
        if sim.done:
            st.rerun()  # full rerun switches the autoplay timer off
    try:
        status(sim)
        charts(sim)
        with st.expander("Event log"):
            st.dataframe(sim.events_frame(), hide_index=True, height=260)
    except Exception as exc:  # a drawing glitch must not stop the simulation
        logger.exception("render failed")
        st.warning(f"Could not draw a panel ({exc}); the simulation keeps running.")


def main() -> None:
    """Page entry point."""
    st.set_page_config(page_title="Monopoly vs duopoly: live", layout="wide")
    alt.data_transformers.disable_max_rows()
    st.markdown(COMPACT_CSS, unsafe_allow_html=True)
    st.markdown("## Monopoly vs duopoly, live")
    st.caption("Same riders, drivers and shocks in both markets: one trained platform (navy) vs two competing "
               "trained platforms (purple). Percentages = duopoly vs monopoly.")
    sim, autoplay, speed = sidebar()
    if sim is None:
        return
    interval = 1.0 / speed if autoplay and not sim.done else None
    st.fragment(live_panel, run_every=interval)(autoplay)


main()
