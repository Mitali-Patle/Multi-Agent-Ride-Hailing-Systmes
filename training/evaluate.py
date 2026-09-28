"""Final evaluation: metrics, hypothesis tests and slide charts.

Runs every saved policy on the same fixed evaluation seeds and writes:

* ``results/charts/learning_curves.png``      reward per round (from the training log)
* ``results/charts/learning_proof.png``       untrained vs trained vs rule baseline
* ``results/charts/monopoly_vs_duopoly.png``  metrics grouped by hypothesis
* ``results/charts/hypotheses_summary.csv``   verdicts with means, stds, Welch t
* ``results/csv/eval_*.csv``                  per-episode raw numbers

Usage::

    python training/evaluate.py [--episodes N]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python training/<script>.py`

import argparse  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402

import pandas as pd  # noqa: E402

import config  # noqa: E402
from training import charts  # noqa: E402
from training.baselines import BaselinePolicy  # noqa: E402
from training.common import model_file, results_path, seed_everything, setup_logging  # noqa: E402
from training.hypotheses import hypotheses_table  # noqa: E402
from training.policies import load_frozen, trained_policies  # noqa: E402
from training.rollout import evaluate, eval_seeds  # noqa: E402

logger = logging.getLogger(__name__)


def _stats(group: str, variant: str, profits: pd.Series) -> dict:
    """One bar of the learning-proof chart."""
    return {"group": group, "variant": variant, "mean": float(profits.mean()), "std": float(profits.std(ddof=0))}


def learning_proof_table(results_dir: Path | None, seeds: list[int]) -> pd.DataFrame:
    """Untrained vs trained vs rule baseline, each against the rule baseline opponent."""
    rows = []
    trained = trained_policies("duopoly", results_dir)
    for p, name in enumerate(config.PLATFORM_NAMES):
        untrained = load_frozen(model_file(results_dir, "duopoly", "untrained", name))
        for variant, policy in (("Untrained (random init)", untrained), ("Trained PPO", trained[p]),
                                ("Rule-based baseline", BaselinePolicy(2))):
            df = evaluate({p: policy, 1 - p: BaselinePolicy(2)}, 2, seeds)
            rows.append(_stats(f"Platform {name}", variant, df[f"profit_p{p}"]))
    mono_trained = trained_policies("monopoly", results_dir)[0]
    mono_untrained = load_frozen(model_file(results_dir, "monopoly", "untrained", config.MONOPOLY_NAME))
    for variant, policy in (("Untrained (random init)", mono_untrained), ("Trained PPO", mono_trained),
                            ("Rule-based baseline", BaselinePolicy(1))):
        rows.append(_stats(config.MONOPOLY_NAME, variant, evaluate({0: policy}, 1, seeds)["profit_p0"]))
    return pd.DataFrame(rows)


def print_report(proof: pd.DataFrame, hyp: pd.DataFrame, mono: pd.DataFrame, duo: pd.DataFrame) -> None:
    """Human-readable console summary."""
    print("\nLearning proof (mean episode profit vs rule baseline opponent):")
    print(proof.pivot(index="group", columns="variant", values="mean").round(0).to_string())
    cols = ["avg_fare", "avg_wait_min", "abandonment_rate", "completion_rate", "driver_earnings_per_driver",
            "platform_profit", "bonus_paid", "rider_surplus", "total_welfare"]
    table = pd.DataFrame({"monopoly": mono[cols].mean(), "duopoly": duo[cols].mean()})
    print("\nMonopoly vs duopoly (means over evaluation seeds):")
    print(table.round(3).to_string())
    print("\nHypotheses:")
    for _, row in hyp[hyp["metric"] == "OVERALL"].iterrows():
        print(f"  {row['hypothesis']}: {row['supports']}")


def run_evaluation(results_dir: Path | None = None, n_episodes: int = config.N_EVAL_EPISODES_FINAL) -> pd.DataFrame:
    """Evaluate all saved models and write CSVs and charts. Returns the hypotheses table."""
    seed_everything(config.SEED)
    seeds = eval_seeds(n_episodes)
    charts_dir = results_path(results_dir, config.CHARTS_SUBDIR)
    csv_dir = results_path(results_dir, config.CSV_SUBDIR)
    csv_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    duo = evaluate(trained_policies("duopoly", results_dir), 2, seeds)
    mono = evaluate(trained_policies("monopoly", results_dir), 1, seeds)
    duo.to_csv(csv_dir / "eval_duopoly.csv", index=False)
    mono.to_csv(csv_dir / "eval_monopoly.csv", index=False)
    proof = learning_proof_table(results_dir, seeds)
    proof.to_csv(csv_dir / "learning_proof.csv", index=False)
    hyp = hypotheses_table(mono, duo)
    charts_dir.mkdir(parents=True, exist_ok=True)
    hyp.to_csv(charts_dir / "hypotheses_summary.csv", index=False)

    log_path = csv_dir / config.TRAINING_LOG_CSV
    if log_path.exists():
        charts.learning_curves(pd.read_csv(log_path), charts_dir / "learning_curves.png")
    else:
        logger.warning("no training log at %s; skipping learning_curves.png", log_path)
    charts.learning_proof(proof, charts_dir / "learning_proof.png")
    charts.monopoly_vs_duopoly(mono, duo, charts_dir / "monopoly_vs_duopoly.png")
    print_report(proof, hyp, mono, duo)
    elapsed = time.perf_counter() - start
    print(f"\n[evaluate] {n_episodes} seeds per setting, done in {elapsed:.1f}s; charts in {charts_dir}")
    return hyp


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--episodes", type=int, default=config.N_EVAL_EPISODES_FINAL,
                        help="evaluation seeds per setting")
    parser.add_argument("--results-dir", type=Path, default=None, help="results folder (default: results/)")
    args = parser.parse_args(argv)
    try:
        run_evaluation(args.results_dir, max(2, args.episodes))
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
