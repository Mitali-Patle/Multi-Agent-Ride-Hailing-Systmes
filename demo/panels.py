"""Altair chart builders for the side-by-side dashboard.

Every chart compares the monopoly (navy) with the duopoly (purple; its two
platforms in two purples on the market-share chart) over the 12-hour day,
with shock windows shaded red. The readout helpers produce the one-line
"Monopoly X · Duopoly Y (gap)" text the page shows above each chart.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

import config
from demo.compare import DUO, DUO_A, DUO_B, MONO
from sim.market import clock_time

SERIES_ORDER = [MONO, DUO, DUO_A, DUO_B]
SERIES_COLORS = {MONO: config.COLOR_MONOPOLY, DUO: config.COLOR_A, DUO_A: config.COLOR_A, DUO_B: config.COLOR_B}
CHART_HEIGHT = 210
DAY_START = config.START_HOUR
DAY_END = config.START_HOUR + config.EPISODE_TICKS * config.TICK_MINUTES / 60.0


def _hour(tick: float) -> float:
    """Clock time in hours for the start of a tick."""
    return config.START_HOUR + tick * config.TICK_MINUTES / 60.0


def _shock_layer(windows: list[tuple[int, int, str]]) -> alt.Chart | None:
    """Translucent red bands for the periods when shocks were active."""
    if not windows:
        return None
    frame = pd.DataFrame(
        [(_hour(s), _hour(e), k.replace("_", " "), clock_time(s), clock_time(e)) for s, e, k in windows],
        columns=["start", "end", "shock", "from", "to"],
    )
    return alt.Chart(frame).mark_rect(color=config.COLOR_SHOCK, opacity=0.12).encode(
        x="start:Q", x2="end:Q", tooltip=["shock:N", "from:N", "to:N"])


def comparison_chart(frame: pd.DataFrame, y_title: str, fmt: str, windows: list[tuple[int, int, str]],
                     step: bool = False) -> alt.LayerChart | alt.Chart:
    """One metric over the day, one line per market, shocks shaded behind (title is drawn by the page)."""
    series = [s for s in SERIES_ORDER if s in set(frame["series"])]
    data = frame.assign(time=[clock_time(t + 1) for t in frame["tick"]])
    hours = list(range(int(DAY_START), int(DAY_END) + 1, 3))
    line = alt.Chart(data).mark_line(strokeWidth=2.5, interpolate="step-after" if step else "linear").encode(
        x=alt.X("hour:Q", title=None, scale=alt.Scale(domain=[DAY_START, DAY_END], nice=False),
                axis=alt.Axis(values=hours, labelExpr="datum.value + ':00'")),
        y=alt.Y("value:Q", title=y_title, axis=alt.Axis(format=fmt)),
        color=alt.Color("series:N", scale=alt.Scale(domain=series, range=[SERIES_COLORS[s] for s in series]),
                        legend=alt.Legend(orient="top", title=None, labelFontSize=13)),
        tooltip=["time:N", "series:N", alt.Tooltip("value:Q", format=fmt)],
    )
    shocks = _shock_layer(windows)
    chart = line if shocks is None else alt.layer(shocks, line)
    return chart.properties(height=CHART_HEIGHT, width="container")


def latest(frame: pd.DataFrame) -> dict[str, float]:
    """Most recent non-missing value of each series."""
    clean = frame.dropna(subset=["value"]).sort_values("tick")
    return clean.groupby("series")["value"].last().to_dict()


def _compact(value: float, kind: str) -> str:
    """Short number for a subtitle: '54.3k', '6.5 min', '12'."""
    if kind == "minutes":
        return f"{value:.1f} min"
    if kind == "percent":
        return f"{value:.0%}"
    if kind == "fare":
        return f"{value:,.1f}"
    if kind == "money" and abs(value) >= 10_000:
        return f"{value / 1000:,.1f}k"
    return f"{value:,.0f}"


def _gap(mono: float, duo: float) -> str:
    """Duopoly relative to monopoly, e.g. '(-27%)'; empty when undefined."""
    return f" ({(duo - mono) / abs(mono):+.0%})" if abs(mono) > 1e-9 else ""


def gap_readout(frame: pd.DataFrame, kind: str, label: str = "Now", day: pd.DataFrame | None = None) -> str:
    """'Now: Monopoly X · Duopoly Y (+Z%)', plus the whole-day gap when ``day`` totals are given.

    ``kind`` is 'money', 'minutes' or 'count' (controls number formatting).
    """
    now = latest(frame)
    if MONO not in now or DUO not in now:
        return "waiting for data"
    mono, duo = now[MONO], now[DUO]
    gap = f" ({(duo - mono) * 100:+.0f} pts)" if kind == "percent" else _gap(mono, duo)
    text = f"{label}: Monopoly {_compact(mono, kind)} · Duopoly {_compact(duo, kind)}{gap}"
    if day is not None:
        totals = latest(day)
        if MONO in totals and DUO in totals:
            text += f"  |  day so far{_gap(totals[MONO], totals[DUO]) or ': n/a'}"
    return text


def share_readout(frame: pd.DataFrame) -> str:
    """Average (day-so-far) market shares."""
    now = latest(frame)
    if DUO_A not in now:
        return "waiting for data"
    return f"Day so far: A {now[DUO_A]:.0%}  ·  B {now.get(DUO_B, 0.0):.0%}  ·  Monopoly {now.get(MONO, 1.0):.0%}"
