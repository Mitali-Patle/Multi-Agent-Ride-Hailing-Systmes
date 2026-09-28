"""Single-platform PPO training for the monopoly ablation.

Same PPO setup, same market, same drivers and demand as the duopoly, but
``N_PLATFORMS = 1``. It trains for the same total timesteps as ONE duopoly
platform (``rounds * steps``), split into the same number of chunks so the
learning curve is comparable, and saves to ``results/models/monopoly/final/``.

Usage::

    python training/train_monopoly.py [--rounds N] [--steps N] [--seed N]
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
from training.common import (  # noqa: E402
    append_training_log, clear_training_log, log_scalars, make_ppo, model_file, results_path, save_model,
    seed_everything, setup_logging,
)
from training.policies import FrozenPolicy  # noqa: E402
from training.rollout import eval_seeds, mean_profit  # noqa: E402

logger = logging.getLogger(__name__)
MODE = "monopoly"
NAME = config.MONOPOLY_NAME


def evaluate_chunk(policy: FrozenPolicy, rnd: int, timesteps: int, elapsed: float) -> dict:
    """Evaluate the monopolist on the fixed seeds."""
    mean, std = mean_profit({0: policy}, 1, 0, eval_seeds(config.N_EVAL_EPISODES_TRAINING))
    return {
        "mode": MODE, "round": rnd, "platform": NAME, "opponent": "none",
        "mean_profit": mean, "std_profit": std, "mean_reward": mean / config.REWARD_SCALE,
        "timesteps": timesteps, "elapsed_s": round(elapsed, 1),
    }


def train_monopoly(
    rounds: int = config.N_ROUNDS,
    steps: int = config.STEPS_PER_ROUND,
    seed: int = config.SEED,
    results_dir: Path | None = None,
) -> pd.DataFrame:
    """Train the monopolist; returns the evaluation rows written to the log."""
    seed_everything(seed)
    clear_training_log(results_dir, MODE)
    for old in results_path(results_dir, config.LOGS_SUBDIR, "tensorboard").glob(f"{MODE}_*"):
        shutil.rmtree(old, ignore_errors=True)
    env = PlatformEnv(0, None, 1, seed=seed, random_shocks=True)
    model = make_ppo(env, seed, results_dir)
    save_model(model, model_file(results_dir, MODE, "untrained", NAME))

    start = time.perf_counter()
    rows = [evaluate_chunk(FrozenPolicy.from_model(model), 0, 0, 0.0)]
    print(f"[monopoly] round 0/{rounds} | profit {rows[0]['mean_profit']:>8,.0f} |    0.0s", flush=True)
    for rnd in range(1, rounds + 1):
        model.learn(steps, reset_num_timesteps=False, tb_log_name=f"{MODE}_{NAME}")
        elapsed = time.perf_counter() - start
        row = evaluate_chunk(FrozenPolicy.from_model(model), rnd, int(model.num_timesteps), elapsed)
        log_scalars(model, {"eval/profit": row["mean_profit"]})
        save_model(model, model_file(results_dir, MODE, f"round_{rnd:02d}", NAME))
        rows.append(row)
        print(f"[monopoly] round {rnd}/{rounds} | profit {row['mean_profit']:>8,.0f} | {elapsed:6.1f}s", flush=True)
    append_training_log(rows, results_dir)
    save_model(model, model_file(results_dir, MODE, "final", NAME))
    print(f"[monopoly] done in {time.perf_counter() - start:.1f}s")
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rounds", type=int, default=config.N_ROUNDS, help="training chunks (match the duopoly)")
    parser.add_argument("--steps", type=int, default=config.STEPS_PER_ROUND, help="timesteps per chunk")
    parser.add_argument("--seed", type=int, default=config.SEED, help="master random seed")
    parser.add_argument("--results-dir", type=Path, default=None, help="output folder (default: results/)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    setup_logging()
    args = parse_args(argv)
    if args.rounds < 1 or args.steps < 1:
        raise SystemExit("--rounds and --steps must be positive")
    train_monopoly(args.rounds, args.steps, args.seed, args.results_dir)


if __name__ == "__main__":
    main()
