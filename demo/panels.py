"""Chart and table builders for the Streamlit dashboard.

Pure functions from the simulation history (a pandas frame) to Altair charts
or small frames, so the dashboard file only does layout and control flow.
Colours follow ``config``: Platform A, Platform B, monopoly; shock windows
are shaded in the shock colour on every time-series chart.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

import config

STATE_COLORS = {"busy": config.COLOR_A, "idle": "#C4B5FD", "offline": config.COLOR_SHOCK}
CHART_HEIGHT = 300  # total height incl. title, legend and axes


def name_colors(names: list[str]) -> list[str]:
    """Colour per platform name, in the given order."""
    lookup = {config.PLATFORM_NAMES[0]: config.COLOR_A, config.PLATFORM_NAMES[1]: config.COLOR_B,
              config.MONOPOLY_NAME: config.COLOR_MONOPOLY}
    return [lookup.get(n, config.COLOR_BASELINE) for n in names]


def _shock_layer(windows: list[tuple[int, int, str]]) -> alt.Chart | None:
    """Translucent rectangles marking when shocks were active."""
    if not windows:
        return None
    frame = pd.DataFrame(windows, columns=["start", "end", "shock"])
    return alt.Chart(frame).mark_rect(color=config.COLOR_SHOCK, opacity=0.12).encode(
        x="start:Q", x2="end:Q", tooltip=["shock:N", "start:Q", "end:Q"])


def timeseries(hist: pd.DataFrame, value: str, title: str, y_title: str, names: list[str],
               windows: list[tuple[int, int, str]], step: bool = True, fmt: str = ",.0f") -> alt.LayerChart | alt.Chart:
    """Per-platform line over ticks with shock windows shaded behind it."""
    data = hist[["tick", "clock", "name", value]]
    line = alt.Chart(data).mark_line(strokeWidth=2, interpolate="step-after" if step else "linear").encode(
        x=alt.X("tick:Q", scale=alt.Scale(domain=[0, config.EPISODE_TICKS], nice=False), title="tick (5 min)"),
        y=alt.Y(f"{value}:Q", title=y_title, axis=alt.Axis(format=fmt)),
        color=alt.Color("name:N", scale=alt.Scale(domain=names, range=name_colors(names)),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["tick:Q", "clock:N", "name:N", alt.Tooltip(f"{value}:Q", format=fmt)],
    )
    shocks = _shock_layer(windows)
    chart = line if shocks is None else alt.layer(shocks, line)
    return chart.properties(title=title, height=CHART_HEIGHT, width="container")


def driver_states(hist: pd.DataFrame, names: list[str]) -> alt.Chart:
    """Stacked bars: busy / idle / offline drivers per platform at the latest tick."""
    last = hist[hist["tick"] == hist["tick"].max()]
    data = last.melt(id_vars=["name"], value_vars=list(STATE_COLORS), var_name="state", value_name="drivers")
    states = list(STATE_COLORS)
    data["order"] = data["state"].map({s: i for i, s in enumerate(states)})
    return alt.Chart(data).mark_bar(size=48).encode(
        x=alt.X("name:N", sort=names, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=14)),
        y=alt.Y("drivers:Q", title="drivers", scale=alt.Scale(domain=[0, config.N_DRIVERS])),
        color=alt.Color("state:N", scale=alt.Scale(domain=states, range=list(STATE_COLORS.values())),
                        legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("order:Q"),
        tooltip=["name:N", "state:N", "drivers:Q"],
    ).properties(title="Drivers per platform (now)", height=CHART_HEIGHT, width="container")


def with_derived(hist: pd.DataFrame) -> pd.DataFrame:
    """Add fare multiplier and cumulative columns used by the charts."""
    hist = hist.sort_values(["platform", "tick"]).copy()
    hist["fare_mult"] = hist["fare_idx"].map(dict(enumerate(config.FARE_MULTIPLIERS)))
    grp = hist.groupby("platform")
    hist["cum_profit"] = grp["profit"].cumsum()
    hist["cum_abandoned"] = grp["abandoned"].cumsum()
    hist["cum_completed"] = grp["completed"].cumsum()
    total = hist.groupby("tick")["cum_completed"].transform("sum")
    hist["share"] = (hist["cum_completed"] / total.where(total > 0)).fillna(0.0)
    return hist


def welfare_numbers(hist: pd.DataFrame, n_drivers: int) -> dict[str, float]:
    """Running rider-welfare and driver numbers for the headline metrics."""
    matched = hist["matched"].sum()
    completed = hist["completed"].sum()
    requests = hist["requests"].sum()
    return {
        "avg_fare": hist["fares_collected"].sum() / completed if completed else 0.0,
        "avg_wait": hist["wait_ticks"].sum() / matched * config.TICK_MINUTES if matched else 0.0,
        "abandon": hist["abandoned"].sum() / requests if requests else 0.0,
        "earn_per_driver": hist["driver_earnings"].sum() / max(1, n_drivers),
        "rider_surplus": hist["rider_surplus"].sum(),
    }
