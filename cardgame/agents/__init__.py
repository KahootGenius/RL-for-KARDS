"""Agents and a small factory: make_agent("random" | "greedy" | "<ckpt>.pt" | "ppo:<ckpt>")."""
from __future__ import annotations

from typing import Optional

from ..cards import GameConfig, load_ruleset
from .base import Agent
from .greedy_agent import GreedyAgent
from .random_agent import RandomAgent


def make_agent(spec: str, config: Optional[GameConfig] = None, **kwargs) -> Agent:
    config = config if config is not None else load_ruleset()
    if spec == "random":
        return RandomAgent(config, **kwargs)
    if spec == "greedy":
        return GreedyAgent(config, **kwargs)
    path = spec[4:] if spec.startswith("ppo:") else spec
    if path.endswith(".pt"):
        from ..rl.agent import PPOAgent  # lazy: torch is only needed for PPO agents
        return PPOAgent.from_checkpoint(path, config, **kwargs)
    raise ValueError(f"unknown agent spec {spec!r} (expected random, greedy, or a .pt checkpoint)")


__all__ = ["Agent", "GreedyAgent", "RandomAgent", "make_agent"]
