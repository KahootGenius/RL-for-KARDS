"""PPO policy wrapped as an Agent (Observation + legal actions -> action). CPU inference only."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from ..cards import GameConfig, load_ruleset
from ..engine import Observation
from ..features import ObservationEncoder
from .network import build_net


class CheckpointError(ValueError):
    pass


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def net_from_checkpoint(ckpt: dict, encoder: Optional[ObservationEncoder] = None, allow_pool_mismatch: bool = False):
    """Rebuild the network of a checkpoint (eval mode, CPU) and check it matches the encoder."""
    spec = ckpt.get("net", {})
    if "kind" not in spec:
        raise CheckpointError("Stage 1 checkpoint: not loadable in Stage 2 (different rules and encoding)")
    net = build_net(spec)
    net.load_state_dict(ckpt["model"])
    net.eval()
    if encoder is not None:
        if net.obs_dim != encoder.dim:
            raise CheckpointError(f"checkpoint expects obs_dim={net.obs_dim}, this ruleset encodes {encoder.dim}")
        fp = spec.get("layout", {}).get("fingerprint")
        if fp is not None and fp != encoder.fingerprint and not allow_pool_mismatch:
            raise CheckpointError(f"checkpoint was trained on another card pool/decks (fingerprint {fp} != "
                                  f"{encoder.fingerprint}); pass allow_pool_mismatch=True to load it anyway")
    return net


class PPOAgent:
    def __init__(self, net, config: Optional[GameConfig] = None, deterministic: bool = False,
                 name: str = "ppo", seed: Optional[int] = None):
        self.config = config if config is not None else load_ruleset()
        self.encoder = ObservationEncoder(self.config)
        if net.obs_dim != self.encoder.dim:
            raise CheckpointError(f"network expects obs_dim={net.obs_dim}, this ruleset encodes {self.encoder.dim}")
        self.net = net.to("cpu").eval()
        self.n_actions = self.encoder.n_actions
        self.deterministic = deterministic
        self.name = name
        self.rng = np.random.default_rng(seed)
        self._x = np.zeros((1, self.encoder.dim), dtype=np.float32)
        self._mask = np.zeros(self.n_actions, dtype=bool)

    @classmethod
    def from_checkpoint(cls, path: str, config: Optional[GameConfig] = None, deterministic: bool = False,
                        name: Optional[str] = None, seed: Optional[int] = None,
                        allow_pool_mismatch: bool = False) -> "PPOAgent":
        config = config if config is not None else load_ruleset()
        net = net_from_checkpoint(load_checkpoint(path), ObservationEncoder(config), allow_pool_mismatch)
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
        self.encoder.encode_into(obs, x[0], mask)
        x[:] = x.astype(np.float16)  # training inputs are float16-rounded (rollout protocol)
        with torch.inference_mode():
            logits = self.net.policy_logits(torch.from_numpy(x), None)[0].numpy()
        logits = logits[list(legal_actions)].astype(np.float64)
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
