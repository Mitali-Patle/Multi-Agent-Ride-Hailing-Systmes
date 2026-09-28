"""Projector-friendly matplotlib charts for the slides.

Every function takes tidy pandas data and writes one PNG. Styling is shared
through :func:`apply_style`: large fonts, hairline grids, the fixed colour
scheme from ``config`` (Platform A, Platform B, monopoly), a legend whenever
there are two or more series, and direct value labels on bars so identity
never relies on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: works on servers, CI and every OS

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402

SERIES_COLORS = {
    "A": config.COLOR_A,
    "B": config.COLOR_B,
    config.MONOPOLY_NAME: config.COLOR_MONOPOLY,
    "Duopoly": config.COLOR_A,
    "Baseline": config.COLOR_BASELINE,
}


def apply_style() -> None:
    """Shared rcParams for every chart."""
    fs = config.CHART_FONT_SIZE
    plt.rcParams.update({
        "font.size": fs, "axes.titlesize": fs + 2, "axes.labelsize": fs, "legend.fontsize": fs - 2,
        "xtick.labelsize": fs - 2, "ytick.labelsize": fs - 2, "figure.titlesize": fs + 6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#6B7280",
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E5E7EB", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "savefig.bbox": "tight", "savefig.dpi": config.CHART_DPI,
    })


def _save(fig: plt.Figure, path: Path) -> Path:
    """Write a figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


THOUSANDS = FuncFormatter(lambda v, _: f"{v:,.0f}")


def _label_bars(ax: plt.Axes, bars, fmt: str, errors=None) -> None:
    """Value labels just above each bar (and above its error bar, if any)."""
    errors = np.zeros(len(bars)) if errors is None else np.nan_to_num(np.asarray(errors, dtype=float))
    for bar, err in zip(bars, errors, strict=True):
        h = bar.get_height()
        ax.annotate(fmt.format(h), (bar.get_x() + bar.get_width() / 2, h + err), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=config.CHART_FONT_SIZE - 3,
                    color="#111827")


def learning_curves(log: pd.DataFrame, path: Path) -> Path:
    """Mean episode profit per round: duopoly vs baseline, vs rival, and the monopoly."""
    apply_style()
    panels = [("duopoly", "baseline", "Duopoly vs fixed baseline"), ("duopoly", "rival", "Duopoly vs current rival"),
              ("monopoly", "none", "Monopoly (no rival)")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, (mode, opponent, title) in zip(axes, panels, strict=True):
        sub = log[(log["mode"] == mode) & (log["opponent"] == opponent)]
        for name, grp in sub.groupby("platform", sort=True):
            grp = grp.sort_values("round")
            color = SERIES_COLORS.get(str(name), config.COLOR_MONOPOLY)
            label = f"Platform {name}" if name in config.PLATFORM_NAMES else str(name)
            ax.plot(grp["round"], grp["mean_profit"], marker="o", markersize=8, linewidth=2, color=color, label=label)
            ax.fill_between(grp["round"], grp["mean_profit"] - grp["std_profit"],
                            grp["mean_profit"] + grp["std_profit"], color=color, alpha=0.15, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Training round (0 = random init)")
        if not sub.empty:
            ax.set_xticks(sorted(sub["round"].unique()))
            ax.legend(loc="lower right", frameon=False)
    axes[0].set_ylabel("Mean episode profit (±1 sd)")
    axes[0].yaxis.set_major_formatter(THOUSANDS)
    fig.suptitle("Learning curves: profit on fixed evaluation seeds")
    return _save(fig, path)


def learning_proof(df: pd.DataFrame, path: Path) -> Path:
    """Grouped bars: untrained vs trained vs rule baseline, per platform, same opponent."""
    apply_style()
    groups = list(dict.fromkeys(df["group"]))
    variants = ["Untrained (random init)", "Trained PPO", "Rule-based baseline"]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    width, x = 0.26, np.arange(len(groups))
    for i, variant in enumerate(variants):
        sub = df[df["variant"] == variant].set_index("group").reindex(groups)
        colors = [SERIES_COLORS.get(g.split()[-1], config.COLOR_MONOPOLY) for g in groups]
        if variant == "Rule-based baseline":
            colors = [config.COLOR_BASELINE] * len(groups)
        bars = ax.bar(x + (i - 1) * width, sub["mean"], width * 0.92, yerr=sub["std"], capsize=5, color=colors,
                      alpha=0.45 if variant.startswith("Untrained") else 1.0,
                      hatch="//" if variant.startswith("Untrained") else None, edgecolor="white", linewidth=0,
                      error_kw={"elinewidth": 1.2, "ecolor": "#374151"}, label=variant)
        _label_bars(ax, bars, "{:,.0f}", sub["std"].to_numpy())
    ax.set_xticks(x, groups)
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.set_ylabel("Mean episode profit (±1 sd)")
    ax.axhline(0, color="#6B7280", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left", ncols=3)
    ax.set_title("Learning proof: same opponent (rule baseline), same seeds")
    ax.margins(y=0.15)
    return _save(fig, path)


MVD_METRICS = [
    ("avg_fare", "H1 · Average fare paid", "{:,.1f}"),
    ("avg_wait_min", "H1 · Average wait (min)", "{:,.2f}"),
    ("abandonment_rate", "H1 · Abandonment rate", "{:.1%}"),
    ("driver_earnings_per_driver", "H2 · Earnings per driver", "{:,.0f}"),
    ("platform_profit", "H2 · Platform profit (total)", "{:,.0f}"),
    ("completion_rate", "H3 · Completed / requested", "{:.1%}"),
    ("total_welfare", "H3 · Total welfare", "{:,.0f}"),
    ("rider_surplus", "H3 · Rider surplus", "{:,.0f}"),
]


def monopoly_vs_duopoly(mono: pd.DataFrame, duo: pd.DataFrame, path: Path) -> Path:
    """Small multiples, one metric per panel; duopoly profit stacked by platform."""
    apply_style()
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, (metric, title, fmt) in zip(axes.flat, MVD_METRICS, strict=True):
        means = [mono[metric].mean(), duo[metric].mean()]
        stds = [mono[metric].std(ddof=0), duo[metric].std(ddof=0)]
        if metric == "platform_profit":
            a, b = duo["profit_p0"].mean(), duo["profit_p1"].mean()
            ax.bar(0, means[0], 0.6, color=config.COLOR_MONOPOLY, label=config.MONOPOLY_NAME)
            ax.bar(1, a, 0.6, color=config.COLOR_A)
            ax.bar(1, b, 0.6, bottom=a, color=config.COLOR_B)
            for y, text, ink in ((a / 2, "A", "white"), (a + b / 2, "B", "#111827")):
                ax.text(1, y, text, ha="center", va="center", color=ink, fontsize=config.CHART_FONT_SIZE, weight="bold")
            ax.errorbar([0, 1], means, yerr=stds, fmt="none", ecolor="#374151", capsize=5, elinewidth=1.2)
            bars = ax.bar([0, 1], means, 0.6, color="none")
        else:
            bars = ax.bar([0, 1], means, 0.6, yerr=stds, capsize=5, color=[config.COLOR_MONOPOLY, config.COLOR_A],
                          error_kw={"elinewidth": 1.2, "ecolor": "#374151"})
        _label_bars(ax, bars, fmt, stds)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0) if "%" in fmt else THOUSANDS)
        ax.set_xticks([0, 1], [config.MONOPOLY_NAME, "Duopoly"])
        ax.set_title(title)
        ax.margins(y=0.18)
        if min(means) < 0:
            ax.axhline(0, color="#6B7280", linewidth=0.8)
    fig.suptitle(f"Monopoly vs duopoly (trained PPO, {len(mono)} fixed seeds, bars = mean ± 1 sd)")
    fig.tight_layout()
    return _save(fig, path)


def scenario_summary(frame: pd.DataFrame, shocks: list[tuple[int, int, str]], title: str, path: Path) -> Path:
    """Four stacked panels over one episode with shock windows shaded."""
    apply_style()
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    panels = [("fare", "Fare posted"), ("bonus", "Driver bonus"), ("active", "Active drivers (idle+busy)"),
              ("queue", "Riders queued")]
    frame = frame.assign(active=frame["idle"] + frame["busy"])
    for ax, (col, label) in zip(axes, panels, strict=True):
        for p, grp in frame.groupby("platform"):
            name = config.MONOPOLY_NAME if frame["platform"].nunique() == 1 else f"Platform {config.PLATFORM_NAMES[p]}"
            color = config.COLOR_MONOPOLY if name == config.MONOPOLY_NAME else SERIES_COLORS[config.PLATFORM_NAMES[p]]
            ax.step(grp["tick"], grp[col], where="post", linewidth=2, color=color, label=name)
        for start, end, _kind in shocks:
            ax.axvspan(start, end, color=config.COLOR_SHOCK, alpha=0.08, linewidth=0)
        ax.set_ylabel(label)
    for i, (start, _end, kind) in enumerate(sorted(shocks)):
        axes[0].annotate(kind.replace("_", " "), (start, 1.0 + 0.09 * (i % 2)), xycoords=("data", "axes fraction"),
                         fontsize=11, color=config.COLOR_SHOCK, va="bottom")
    axes[1].set_ylim(-1, max(config.BONUS_LEVELS) * 1.1)
    axes[0].legend(frameon=False, loc="center right", ncols=1)
    axes[-1].set_xlabel("Tick (5 min each)")
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, path)
