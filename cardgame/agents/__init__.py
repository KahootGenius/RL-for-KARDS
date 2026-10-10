"""Agents and a small factory: make_agent("random" | "greedy" | "lookahead" | "<ckpt>.pt" | "ppo:<ckpt>").

Ask any agent for a move with `choose_action(agent, game)` (SPEC.md section 9).
"""
from __future__ import annotations

from typing import Optional

from ..cards import GameConfig, load_ruleset
from .base import Agent, GameAgent, choose_action
from .greedy_agent import GreedyAgent
from .lookahead_agent import LookaheadAgent
from .random_agent import RandomAgent


def make_agent(spec: str, config: Optional[GameConfig] = None, **kwargs) -> Agent:
    config = config if config is not None else load_ruleset()
    if spec == "random":
        return RandomAgent(config, **kwargs)
    if spec == "greedy":
        return GreedyAgent(config, **kwargs)
    if spec == "lookahead":
        return LookaheadAgent(config, **kwargs)
    path = spec[4:] if spec.startswith("ppo:") else spec
    if path.endswith(".pt"):
        from ..rl.agent import PPOAgent  # lazy: torch is only needed for PPO agents
        return PPOAgent.from_checkpoint(path, config, **kwargs)
    raise ValueError(f"unknown agent spec {spec!r} (expected random, greedy, lookahead, or a .pt checkpoint)")


__all__ = ["Agent", "GameAgent", "GreedyAgent", "LookaheadAgent", "RandomAgent", "choose_action", "make_agent"]
