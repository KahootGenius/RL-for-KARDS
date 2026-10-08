"""Masked policy MLP + value MLP."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

MASK_VALUE = -1e9  # finite so masked entries give p=0 and p*log(p)=0 (no NaNs)


def mlp(in_dim: int, hidden: Sequence[int], out_dim: int, out_gain: float) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, gain=np.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.ReLU()]
        d = h
    out = nn.Linear(d, out_dim)
    nn.init.orthogonal_(out.weight, gain=out_gain)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class PolicyValueNet(nn.Module):
    """Separate policy and value MLPs over the same encoded observation."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        self.obs_dim, self.n_actions, self.hidden = obs_dim, n_actions, tuple(hidden)
        self.policy = mlp(obs_dim, hidden, n_actions, out_gain=0.01)
        self.value = mlp(obs_dim, hidden, 1, out_gain=1.0)

    def spec(self) -> dict:
        return {"obs_dim": self.obs_dim, "n_actions": self.n_actions, "hidden": list(self.hidden)}

    def masked_logits(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.policy(obs).masked_fill(~mask, MASK_VALUE)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor, deterministic: bool = False):
        logits = self.masked_logits(obs, mask)
        logp_all = torch.log_softmax(logits, dim=-1)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = torch.multinomial(logp_all.exp(), 1).squeeze(-1)
        logp = logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        return actions, logp, self.get_value(obs)

    def evaluate(self, obs: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor):
        logits = self.masked_logits(obs, mask)
        logp_all = torch.log_softmax(logits, dim=-1)
        logp = logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(logp_all.exp() * logp_all).sum(-1)
        return logp, entropy, self.get_value(obs)


class NumpyPolicy:
    """Fast batch-1 inference of the policy MLP with numpy (used by PPOAgent for eval/play)."""

    def __init__(self, net: PolicyValueNet):
        self.layers = [(m.weight.detach().cpu().numpy().T.copy(), m.bias.detach().cpu().numpy().copy())
                       for m in net.policy if isinstance(m, nn.Linear)]

    def logits(self, x: np.ndarray) -> np.ndarray:
        last = len(self.layers) - 1
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < last:
                np.maximum(x, 0.0, out=x)
        return x
