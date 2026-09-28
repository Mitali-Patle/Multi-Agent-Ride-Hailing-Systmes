"""Alternating self-play training of two independent PPO platforms.

Round 1: train A against the fixed rule-based baseline, then train B against
A's fresh snapshot. Later rounds: A trains against B's latest snapshot, then
B against A's latest snapshot. Each platform has its own PPO model (no shared
weights), both start from random initialisation.

After every round each platform is evaluated on the SAME fixed seeds against
(1) the fixed baseline, proving the policy itself improves, and (2) the
current rival, showing the competitive dynamic. Results go to
``results/csv/training_log.csv`` and TensorBoard.

Usage::

    python training/train_duopoly.py [--rounds N] [--steps N] [--seed N]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python training/<script>.py`

import argparse  # noqa: E402
import logging  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import pandas as pd  # noqa: E402

import config  # noqa: E402
from envs.platform_env import PlatformEnv  # noqa: E402
from training.baselines import BaselinePolicy  # noqa: E402
from training.common import (  # noqa: E402
    append_training_log, clear_training_log, log_scalars, make_ppo, model_file, results_path, save_model,
    seed_everything, setup_logging,
)
from training.policies import FrozenPolicy  # noqa: E402
from training.rollout import eval_seeds, mean_profit  # noqa: E402

logger = logging.getLogger(__name__)
MODE = "duopoly"
NAMES = config.PLATFORM_NAMES


def evaluate_round(snaps: dict, baseline: BaselinePolicy, rnd: int, timesteps: int, elapsed: float) -> list[dict]:
    """Evaluate both platforms vs the fixed baseline and vs each other."""
    seeds = eval_seeds(config.N_EVAL_EPISODES_TRAINING)
    rows = []
    for p in (0, 1):
        policies = {p: snaps[p], 1 - p: baseline}
        for opponent, pols in (("baseline", policies), ("rival", snaps)):
            mean, std = mean_profit(pols, 2, p, seeds)
            rows.append({
                "mode": MODE, "round": rnd, "platform": NAMES[p], "opponent": opponent,
                "mean_profit": mean, "std_profit": std, "mean_reward": mean / config.REWARD_SCALE,
                "timesteps": timesteps, "elapsed_s": round(elapsed, 1),
            })
    return rows


def summary_line(rows: list[dict], rnd: int, rounds: int, elapsed: float) -> str:
    """One-line human summary of a round's evaluation."""
    get = {(r["platform"], r["opponent"]): r["mean_profit"] for r in rows}
    parts = [
        f"{n}: vs baseline {get[(n, 'baseline')]:>8,.0f}  vs rival {get[(n, 'rival')]:>8,.0f}" for n in NAMES
    ]
    return f"[duopoly] round {rnd}/{rounds} | " + " | ".join(parts) + f" | {elapsed:6.1f}s"


def train_duopoly(
    rounds: int = config.N_ROUNDS,
    steps: int = config.STEPS_PER_ROUND,
    seed: int = config.SEED,
    results_dir: Path | None = None,
) -> pd.DataFrame:
    """Run alternating self-play and save checkpoints, final models and logs.

    Returns:
        The evaluation rows written to the training log.
    """
    seed_everything(seed)
    clear_training_log(results_dir, MODE)
    for old in results_path(results_dir, config.LOGS_SUBDIR, "tensorboard").glob(f"{MODE}_*"):
        shutil.rmtree(old, ignore_errors=True)
    baseline = BaselinePolicy(2)
    envs = {p: PlatformEnv(p, {1 - p: baseline}, 2, seed=seed + p, random_shocks=True) for p in (0, 1)}
    models = {p: make_ppo(envs[p], seed + p, results_dir) for p in (0, 1)}
    snaps = {p: FrozenPolicy.from_model(models[p]) for p in (0, 1)}
    for p in (0, 1):
        save_model(models[p], model_file(results_dir, MODE, "untrained", NAMES[p]))

    start = time.perf_counter()
    all_rows = evaluate_round(snaps, baseline, 0, 0, 0.0)
    append_training_log(all_rows, results_dir)
    print(summary_line(all_rows, 0, rounds, 0.0), flush=True)
    for rnd in range(1, rounds + 1):
        for p in (0, 1):
            opponent = baseline if (rnd == 1 and p == 0) else snaps[1 - p]
            envs[p].set_opponents({1 - p: opponent})
            models[p].learn(steps, reset_num_timesteps=False, tb_log_name=f"{MODE}_{NAMES[p]}")
            snaps[p] = FrozenPolicy.from_model(models[p])
        elapsed = time.perf_counter() - start
        rows = evaluate_round(snaps, baseline, rnd, int(models[0].num_timesteps), elapsed)
        for p in (0, 1):
            mine = {r["opponent"]: r["mean_profit"] for r in rows if r["platform"] == NAMES[p]}
            log_scalars(models[p], {f"eval/profit_vs_{k}": v for k, v in mine.items()})
            save_model(models[p], model_file(results_dir, MODE, f"round_{rnd:02d}", NAMES[p]))
        append_training_log(rows, results_dir)
        all_rows += rows
        print(summary_line(rows, rnd, rounds, elapsed), flush=True)

    for p in (0, 1):
        save_model(models[p], model_file(results_dir, MODE, "final", NAMES[p]))
    total = time.perf_counter() - start
    final_dir = model_file(results_dir, MODE, "final", NAMES[0]).parent.parent
    print(f"[duopoly] done in {total:.1f}s; final models in {final_dir}")
    return pd.DataFrame(all_rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rounds", type=int, default=config.N_ROUNDS, help="self-play rounds")
    parser.add_argument("--steps", type=int, default=config.STEPS_PER_ROUND, help="timesteps per platform per round")
    parser.add_argument("--seed", type=int, default=config.SEED, help="master random seed")
    parser.add_argument("--results-dir", type=Path, default=None, help="output folder (default: results/)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    setup_logging()
    args = parse_args(argv)
    if args.rounds < 1 or args.steps < 1:
        raise SystemExit("--rounds and --steps must be positive")
    train_duopoly(args.rounds, args.steps, args.seed, args.results_dir)


if __name__ == "__main__":
    main()
