"""Hypothesis tests comparing the trained monopoly and the trained duopoly.

For each metric we report means and standard deviations over the fixed
evaluation seeds, the difference (duopoly minus monopoly), and a Welch
t-statistic with a normal-approximation two-sided p-value (fine for the 30+
episodes used; avoids a SciPy dependency). A difference counts as
significant when ``|t| > config.SIGNIFICANCE_T``.

Verdicts are computed from the data, never assumed:

* H1 is supported if the duopoly has a significantly lower average fare AND
  a significantly shorter average wait; "partly" if only one holds.
* H2 is supported unless competition clearly raises driver earnings by at
  least as much as it costs platforms in profit.
* H3 is supported if total welfare is significantly lower in the duopoly.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import config

METRICS = {
    "H1": [("avg_fare", "lower"), ("avg_wait_min", "lower"), ("abandonment_rate", "lower")],
    "H2": [("driver_earnings_per_driver", "not higher"), ("platform_profit", "lower"), ("bonus_paid", "context")],
    "H3": [("total_welfare", "lower"), ("rider_surplus", "context"), ("completion_rate", "context")],
}
STATEMENTS = {
    "H1": "Duopoly gives lower fares and shorter waits than monopoly",
    "H2": "Competition does not straightforwardly raise driver earnings; bonus wars erode profit",
    "H3": "Total welfare is lower under duopoly than an equally resourced monopoly",
}


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch t-statistic of ``b - a`` and a two-sided normal-approximation p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) if len(a) > 1 and len(b) > 1 else 0.0
    diff = b.mean() - a.mean()
    if se == 0.0:
        return (0.0, 1.0) if diff == 0 else (math.copysign(math.inf, diff), 0.0)
    t = diff / se
    return float(t), float(math.erfc(abs(t) / math.sqrt(2)))


def metric_row(hyp: str, metric: str, expected: str, mono: pd.DataFrame, duo: pd.DataFrame) -> dict:
    """Statistics for one metric."""
    t, p = welch(mono[metric], duo[metric])
    m_mean, d_mean = float(mono[metric].mean()), float(duo[metric].mean())
    sig = abs(t) > config.SIGNIFICANCE_T
    if expected == "lower":
        holds = sig and d_mean < m_mean
    elif expected == "not higher":
        holds = not (sig and d_mean > m_mean)
    else:
        holds = None
    return {
        "hypothesis": hyp, "metric": metric, "expected_duopoly": expected,
        "monopoly_mean": m_mean, "monopoly_std": float(mono[metric].std(ddof=0)),
        "duopoly_mean": d_mean, "duopoly_std": float(duo[metric].std(ddof=0)),
        "difference": d_mean - m_mean, "relative_difference": (d_mean - m_mean) / abs(m_mean) if m_mean else np.nan,
        "welch_t": t, "p_value_approx": p, "significant": sig,
        "supports": "context" if holds is None else ("yes" if holds else "no"),
    }


def _direction(row: pd.Series, lower: str, higher: str) -> str:
    """Plain-English direction of a duopoly-minus-monopoly difference."""
    if not row["significant"]:
        return "no significant difference"
    rel = row["relative_difference"]
    pct = f" ({rel:+.0%})" if pd.notna(rel) else ""
    return (lower if row["difference"] < 0 else higher) + pct


def verdicts(rows: pd.DataFrame, mono: pd.DataFrame, duo: pd.DataFrame) -> dict[str, str]:
    """Overall verdict per hypothesis with a one-line reason."""
    get = rows.set_index("metric")
    fare_ok, wait_ok = get.loc["avg_fare", "supports"] == "yes", get.loc["avg_wait_min", "supports"] == "yes"
    h1 = "SUPPORTED" if fare_ok and wait_ok else ("PARTLY SUPPORTED" if fare_ok or wait_ok else "NOT SUPPORTED")
    h1 += (f": fare {_direction(get.loc['avg_fare'], 'significantly lower', 'significantly higher')}, "
           f"wait {_direction(get.loc['avg_wait_min'], 'significantly shorter', 'significantly longer')}")

    driver_gain = duo["driver_earnings_total"].mean() - mono["driver_earnings_total"].mean()
    profit_loss = mono["platform_profit"].mean() - duo["platform_profit"].mean()
    earn = get.loc["driver_earnings_per_driver"]
    clearly_higher = bool(earn["significant"]) and earn["difference"] > 0
    proportional = clearly_higher and driver_gain >= profit_loss
    h2 = "NOT SUPPORTED" if proportional else "SUPPORTED"
    duo_bonus = duo["bonus_paid"].mean()
    h2 += (f": driver earnings change {driver_gain:+,.0f} vs platform profit change {-profit_loss:+,.0f} "
           f"(bonus paid {duo_bonus:,.0f} duopoly vs {mono['bonus_paid'].mean():,.0f} monopoly)")
    if duo_bonus < 0.01 * max(1.0, abs(duo["platform_profit"].mean())):
        h2 += "; note: no bonus war emerged, so any effect comes through fares, not bonuses"

    welfare_ok = get.loc["total_welfare", "supports"] == "yes"
    diff = get.loc["total_welfare", "difference"]
    t_welfare = get.loc["total_welfare", "welch_t"]
    h3 = ("SUPPORTED" if welfare_ok else "NOT SUPPORTED") + (
        f": duopoly welfare {_direction(get.loc['total_welfare'], 'lower', 'higher')}, "
        f"difference {diff:+,.0f} (t={t_welfare:.2f})")
    return {"H1": h1, "H2": h2, "H3": h3}


def hypotheses_table(mono: pd.DataFrame, duo: pd.DataFrame) -> pd.DataFrame:
    """Per-metric rows plus one overall row per hypothesis."""
    rows = pd.DataFrame([metric_row(h, m, e, mono, duo) for h, specs in METRICS.items() for m, e in specs])
    overall = verdicts(rows, mono, duo)
    summary = pd.DataFrame([
        {"hypothesis": h, "metric": "OVERALL", "expected_duopoly": STATEMENTS[h], "supports": overall[h]}
        for h in METRICS
    ])
    return pd.concat([rows, summary], ignore_index=True)
