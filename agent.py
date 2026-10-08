"""PPO policy wrapped as an Agent (Observation + legal actions -> action)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from ..cards import GameConfig, load_ruleset
from ..engine import Observation
from ..features import ObservationEncoder
from .network import NumpyPolicy, PolicyValueNet


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def net_from_checkpoint(ckpt: dict) -> PolicyValueNet:
    spec = ckpt["net"]
    net = PolicyValueNet(spec["obs_dim"], spec["n_actions"], spec["hidden"])
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net


class PPOAgent:
    def __init__(self, net: PolicyValueNet, config: Optional[GameConfig] = None, deterministic: bool = False,
                 name: str = "ppo", seed: Optional[int] = None):
        self.config = config if config is not None else load_ruleset()
        self.encoder = ObservationEncoder(self.config)
        if net.obs_dim != self.encoder.dim:
            raise ValueError(f"checkpoint expects obs_dim={net.obs_dim}, ruleset encodes {self.encoder.dim}")
        self.policy = NumpyPolicy(net)
        self.n_actions = net.n_actions
        self.deterministic = deterministic
        self.name = name
        self.rng = np.random.default_rng(seed)
        self._x = np.zeros(self.encoder.dim, dtype=np.float32)
        self._mask = np.zeros(self.n_actions, dtype=bool)

    @classmethod
    def from_checkpoint(cls, path: str, config: Optional[GameConfig] = None, deterministic: bool = False,
                        name: Optional[str] = None, seed: Optional[int] = None) -> "PPOAgent":
        net = net_from_checkpoint(load_checkpoint(path))
        return cls(net, config, deterministic=deterministic, name=name or f"ppo:{path}", seed=seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def action_probs(self, obs: Observation, legal_actions: Sequence[int]) -> np.ndarray:
        """Probabilities over `legal_actions` (same order)."""
        x, mask = self._x, self._mask
        x[:] = 0.0
        mask[:] = False
        mask[list(legal_actions)] = True
        self.encoder.encode_into(obs, x, mask)
        logits = self.policy.logits(x[None, :])[0][list(legal_actions)].astype(np.float64)
        logits -= logits.max()
        p = np.exp(logits)
        return p / p.sum()

    def act(self, obs: Observation, legal_actions: Sequence[int]) -> int:
        if len(legal_actions) == 1:
            return legal_actions[0]
        p = self.action_probs(obs, legal_actions)
        if self.deterministic:
            return legal_actions[int(np.argmax(p))]
        return legal_actions[int(self.rng.choice(len(p), p=p))]
