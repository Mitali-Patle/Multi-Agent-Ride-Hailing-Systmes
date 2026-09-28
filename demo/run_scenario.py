"""Headless CLI demo with terminal telemetry (GUI backup and test harness).

Plays one 12-hour episode with the saved policies (inference only), prints a
compact per-tick table, highlights shocks as they start and end, prints the
event log, and saves a summary chart to ``results/charts/``.

Usage::

    python demo/run_scenario.py --scenario all --mode duopoly --seed 7

Scenarios: ``steady`` (no shocks), ``demand_spike``, ``driver_outage``,
``sensor_failure`` (one shock at a fixed tick), and ``all`` (every shock at
fixed ticks, including two overlapping shocks and one on the final tick).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python demo/run_scenario.py`

import argparse  # noqa: E402
import logging  # noqa: E402

import config  # noqa: E402
from demo.live_sim import LiveSimulation  # noqa: E402
from sim.market import clock_time  # noqa: E402
from sim.metrics import TickRecord  # noqa: E402
from training import charts  # noqa: E402
from training.common import results_path, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)
SCENARIOS = ("steady", "demand_spike", "driver_outage", "sensor_failure", "all")
SENSOR_TARGET = 1  # the platform that goes blind in the single-shock scenario


def schedule_for(scenario: str) -> list[tuple[int, str, int | None]]:
    """Shock triggers ``(tick, kind, target)`` for a named scenario."""
    if scenario == "steady":
        return []
    if scenario == "all":
        return [tuple(item) for item in config.SCENARIO_ALL_SCHEDULE]
    target = SENSOR_TARGET if scenario == "sensor_failure" else None
    return [(config.SCENARIO_SHOCK_TICKS[scenario], scenario, target)]


def _color(text: str, code: str, enabled: bool) -> str:
    """Wrap text in an ANSI colour when writing to a terminal."""
    return f"\033[{code}m{text}\033[0m" if enabled else text


def header(sim: LiveSimulation) -> str:
    """Column header for the per-tick table."""
    cols = ["tick", "time "]
    for n in sim.names:
        tag = n[:4]
        cols += [f"fare{tag:>4}", f"bon{tag:>4}", f"drv{tag:>4}", f"que{tag:>4}", f"ride{tag:>3}", f"profit{tag:>5}"]
    return "  ".join(f"{c:>8}" for c in cols)


def row(sim: LiveSimulation, rec: TickRecord, cum_profit: list[float]) -> str:
    """One formatted table row."""
    cells = [f"{rec.tick:>8}", f"{clock_time(rec.tick):>8}"]
    for p, pt in enumerate(rec.platforms):
        fare = "x" + format(config.FARE_MULTIPLIERS[pt.fare_idx], ".2f")
        cells += [f"{fare:>8}", f"{pt.bonus:>8.0f}", f"{pt.idle + pt.busy:>8}",
                  f"{pt.queue:>8}", f"{pt.completed:>8}", f"{cum_profit[p]:>11,.0f}"]
    return "  ".join(cells)


def run_scenario(scenario: str, mode: str, seed: int, policy: str = "trained", out_dir: Path | None = None,
                 verbose: bool = True) -> dict:
    """Play the scenario and return the episode summary metrics."""
    try:
        sim = LiveSimulation(mode, seed, policy)
    except FileNotFoundError as exc:
        print(_color(f"WARNING: {exc}\nFalling back to the rule-based baseline policies.", "33", sys.stdout.isatty()))
        sim = LiveSimulation(mode, seed, "baseline")
    tty = sys.stdout.isatty()
    plan: dict[int, list[tuple[str, int | None]]] = {}
    for tick, kind, target in schedule_for(scenario):
        plan.setdefault(int(tick), []).append((kind, target))
    print(f"Scenario '{scenario}' | {mode} | seed {seed} | policies: {sim.policy_label}")
    if verbose:
        print(header(sim))
    cum = [0.0] * sim.n_platforms
    while not sim.done:
        for kind, target in plan.get(sim.tick, []):
            sim.trigger_shock(kind, target)
        n_events = len(sim.events)
        rec = sim.step()
        cum = [c + pt.profit for c, pt in zip(cum, rec.platforms, strict=True)]
        new_events = list(sim.events)[n_events:]
        for ev in new_events:
            if ev.kind == "shock":
                print(_color(f">>> tick {ev.tick} ({ev.clock}) {ev.message}", "1;31", tty))
        if verbose:
            print(_color(row(sim, rec, cum), "31", tty) if rec.active_shocks else row(sim, rec, cum))
    summary = sim.market.metrics.summary()
    print_summary(sim, summary)
    out = results_path(None, config.CHARTS_SUBDIR) if out_dir is None else Path(out_dir)
    path = charts.scenario_summary(sim.history(), sim.shock_windows(),
                                   f"Scenario: {scenario} ({mode}, seed {seed}, {sim.policy_label})",
                                   out / f"scenario_{scenario}_{mode}.png")
    print(f"Chart saved to {path}")
    return summary


def print_summary(sim: LiveSimulation, summary: dict) -> None:
    """Event log and headline metrics."""
    print("\nEvent log:")
    for _, ev in sim.events_frame().iloc[::-1].iterrows():
        print(f"  tick {ev['tick']:>3} {ev['time']}  [{ev['type']:<6}] {ev['event']}")
    print("\nEpisode summary:")
    for key in ("avg_fare", "avg_wait_min", "abandonment_rate", "completed", "driver_earnings_per_driver",
                "platform_profit", "rider_surplus", "total_welfare"):
        print(f"  {key:<28} {summary[key]:>14,.3f}")
    for p, name in enumerate(sim.names):
        print(f"  profit {name:<21} {summary[f'profit_p{p}']:>14,.0f}   share {summary[f'share_p{p}']:.1%}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    setup_logging(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", choices=SCENARIOS, default="all")
    parser.add_argument("--mode", choices=("duopoly", "monopoly"), default="duopoly")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--policy", choices=("trained", "baseline"), default="trained",
                        help="saved PPO models (default) or the rule-based baseline")
    parser.add_argument("--out-dir", type=Path, default=None, help="where to save the chart (default: results/charts)")
    parser.add_argument("--quiet", action="store_true", help="skip the per-tick table")
    args = parser.parse_args(argv)
    run_scenario(args.scenario, args.mode, args.seed, args.policy, args.out_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
