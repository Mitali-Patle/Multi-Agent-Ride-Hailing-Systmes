"""Frozen policy snapshots and loaders for trained models.

:class:`FrozenPolicy` copies the actor half of an SB3 ``MlpPolicy`` into
plain NumPy arrays. That gives us:

* a true snapshot: later training of the source model cannot change it,
  which is exactly what alternating self-play needs;
* fast inference (no torch call per tick) for opponents inside the env,
  evaluation, and the live demo.

Deterministic mode takes the arg-max of each action head, matching
``model.predict(obs, deterministic=True)`` (checked in the tests).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

import config
from envs.platform_env import Policy
from training.baselines import BaselinePolicy
from training.common import model_file

logger = logging.getLogger(__name__)

_ACTIVATIONS = {torch.nn.Tanh: np.tanh, torch.nn.ReLU: lambda x: np.maximum(x, 0.0)}


class FrozenPolicy:
    """NumPy copy of a PPO actor for a ``MultiDiscrete`` action space."""

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]], activation, head: tuple[np.ndarray, np.ndarray],
                 nvec: np.ndarray, deterministic: bool = True, seed: int | None = None) -> None:
        """Build from raw weights; use :meth:`from_model` in practice."""
        self.layers = layers
        self.activation = activation
        self.head = head
        self.splits = np.cumsum(nvec)[:-1]
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_model(cls, model: PPO, deterministic: bool = config.OPPONENT_DETERMINISTIC) -> FrozenPolicy:
        """Snapshot the current weights of an SB3 PPO ``MlpPolicy``."""
        policy = model.policy
        layers, activation = [], np.tanh
        for module in policy.mlp_extractor.policy_net:
            if isinstance(module, torch.nn.Linear):
                weight, bias = module.weight.detach().cpu().numpy(), module.bias.detach().cpu().numpy()
                layers.append((weight.T.copy(), bias.copy()))
            elif type(module) in _ACTIVATIONS:
                activation = _ACTIVATIONS[type(module)]
            else:
                raise TypeError(f"unsupported layer in policy net: {module}")
        head = policy.action_net
        return cls(
            layers,
            activation,
            (head.weight.detach().cpu().numpy().T.copy(), head.bias.detach().cpu().numpy().copy()),
            np.asarray(model.action_space.nvec),
            deterministic=deterministic,
        )

    def logits(self, obs: np.ndarray) -> list[np.ndarray]:
        """Per-head action logits for one observation."""
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        for w, b in self.layers:
            x = self.activation(x @ w + b)
        return np.split(x @ self.head[0] + self.head[1], self.splits)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Return ``[fare_index, bonus_index]`` for one observation."""
        heads = self.logits(obs)
        if self.deterministic:
            return np.array([int(np.argmax(h)) for h in heads], dtype=np.int64)
        out = []
        for h in heads:
            p = np.exp(h - h.max())
            out.append(int(self.rng.choice(len(h), p=p / p.sum())))
        return np.array(out, dtype=np.int64)


def load_frozen(path: Path) -> FrozenPolicy:
    """Load a saved PPO zip and snapshot it."""
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return FrozenPolicy.from_model(PPO.load(path, device="cpu"))


def trained_policies(mode: str, results_dir: Path | None = None, tag: str = "final") -> dict[int, Policy]:
    """Load the saved policies for ``mode`` ('duopoly' or 'monopoly').

    Raises:
        FileNotFoundError: with the command that creates the missing models.
    """
    if mode == "duopoly":
        files = {p: model_file(results_dir, "duopoly", tag, config.PLATFORM_NAMES[p]) for p in range(2)}
        script = "python training/train_duopoly.py"
    elif mode == "monopoly":
        files = {0: model_file(results_dir, "monopoly", tag, config.MONOPOLY_NAME)}
        script = "python training/train_monopoly.py"
    else:
        raise ValueError(f"unknown mode {mode!r}")
    missing = [str(f) for f in files.values() if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing trained {mode} model(s): {', '.join(missing)}. Train them first with: {script}")
    return {p: load_frozen(f) for p, f in files.items()}


def baseline_policies(mode: str) -> dict[int, Policy]:
    """Rule-based policies for every platform in ``mode``."""
    n = 2 if mode == "duopoly" else 1
    return {p: BaselinePolicy(n) for p in range(n)}
