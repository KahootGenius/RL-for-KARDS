"""PPO policy wrapped as an Agent (Observation + legal actions -> action). CPU inference only.

Checkpoints are `{"model": state_dict, "net": net.spec(), ...}` (SPEC §7/§8). `net_from_checkpoint`
rejects Stage 1 (no network kind) and Stage 2 (kind "entity", or an encoder version other than
`features.ENCODER_VERSION`) checkpoints with a `CheckpointError`, and checks the card-pool fingerprint
against the encoder.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from ..cards import GameConfig, load_ruleset
from ..engine import Observation
from ..features import ENCODER_VERSION, ObservationEncoder
from .network import NET_KINDS, STAGE2_KINDS, build_net


class CheckpointError(ValueError):
    pass


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def spec_encoder_version(spec: dict):
    """Encoder version a network spec was built for (layout nets: layout["version"]; mlp: "version")."""
    layout = spec.get("layout")
    return layout.get("version") if isinstance(layout, dict) else spec.get("version")


def spec_fingerprint(spec: dict):
    layout = spec.get("layout")
    return layout.get("fingerprint") if isinstance(layout, dict) else spec.get("fingerprint")


def spec_obs_dim(spec: dict):
    layout = spec.get("layout")
    return layout.get("dim") if isinstance(layout, dict) else spec.get("obs_dim")


def _rebase(spec: dict, encoder: ObservationEncoder) -> dict:
    """The spec rebuilt on the encoder's layout (allow_pool_mismatch): the network then reads the live
    card table instead of the training pool's, exactly as the encoder sees the live pool."""
    spec = dict(spec)
    if isinstance(spec.get("layout"), dict):
        spec["layout"] = encoder.layout()
    else:
        spec["fingerprint"] = encoder.fingerprint
    return spec


def net_from_checkpoint(ckpt: dict, encoder: Optional[ObservationEncoder] = None, allow_pool_mismatch: bool = False):
    """Rebuild the network of a checkpoint (eval mode, CPU) and check it matches the encoder.

    Raises CheckpointError for Stage 1 / Stage 2 checkpoints, an encoder version or input size that
    differs from this code, weights that do not fit the spec, and (unless `allow_pool_mismatch`) a
    card pool/deck fingerprint that differs from the encoder's. With `allow_pool_mismatch` the network
    is rebuilt on the encoder's layout (live card table); its parameter shapes must still fit."""
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise CheckpointError("not a PPO checkpoint (expected a dict with 'model' and 'net')")
    spec = ckpt.get("net") or {}
    if not isinstance(spec, dict) or "kind" not in spec:
        raise CheckpointError("Stage 1 checkpoint (network spec without 'kind'): not loadable in Stage 3 "
                              "(different rules and encoding)")
    kind = spec["kind"]
    if kind in STAGE2_KINDS:
        raise CheckpointError(f"Stage 2 checkpoint (network kind {kind!r}, encoder v4): not loadable in Stage 3 "
                              f"(different rules, encoding and architecture); retrain with train.py")
    if kind not in NET_KINDS:
        raise CheckpointError(f"unknown network kind {kind!r} (expected one of {NET_KINDS})")
    version = spec_encoder_version(spec)
    if version != ENCODER_VERSION:
        raise CheckpointError(f"checkpoint was built for encoder version {version}, this code encodes version "
                              f"{ENCODER_VERSION} (Stage 2 checkpoints have version 4 or none): not loadable")
    if encoder is not None:
        if spec_obs_dim(spec) != encoder.dim:
            raise CheckpointError(f"checkpoint expects obs_dim={spec_obs_dim(spec)}, this ruleset encodes "
                                  f"{encoder.dim}")
        fp = spec_fingerprint(spec)
        if fp != encoder.fingerprint:
            if not allow_pool_mismatch:
                raise CheckpointError(f"checkpoint was trained on another card pool/decks (fingerprint {fp} != "
                                      f"{encoder.fingerprint}); pass allow_pool_mismatch=True to load it anyway")
            spec = _rebase(spec, encoder)
    try:
        net = build_net(spec)
    except (ValueError, KeyError, TypeError) as exc:
        raise CheckpointError(f"cannot rebuild the checkpoint network: {exc}") from exc
    try:
        net.load_state_dict(ckpt["model"])
    except RuntimeError as exc:
        where = " on this card pool" if encoder is not None else ""
        raise CheckpointError(f"checkpoint weights do not fit its network spec{where}: {exc}") from exc
    return net.to("cpu").eval()


class PPOAgent:
    """Acts with a policy network: encodes the observer's own decision with its legal mask, rounds the
    input to float16 exactly as the rollout workers do (the protocol the learner trains on), then
    samples (or, `deterministic`, takes the argmax) over the legal actions."""

    def __init__(self, net, config: Optional[GameConfig] = None, deterministic: bool = False,
                 name: str = "ppo", seed: Optional[int] = None):
        self.config = config if config is not None else load_ruleset()
        self.encoder = ObservationEncoder(self.config)
        if net.obs_dim != self.encoder.dim or net.n_actions != self.encoder.n_actions:
            raise CheckpointError(f"network expects obs_dim={net.obs_dim}, n_actions={net.n_actions}; this ruleset "
                                  f"encodes {self.encoder.dim}, {self.encoder.n_actions}")
        self.net = net.to("cpu").eval()
        self.n_actions = self.encoder.n_actions
        self.deterministic = deterministic
        self.name = name
        self.rng = np.random.default_rng(seed)
        self._x = np.zeros((1, self.encoder.dim), dtype=np.float32)
        self._mask = np.zeros((1, self.n_actions), dtype=bool)

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
        """Probabilities over `legal_actions` (same order). `obs` is the acting player's observation."""
        legal = list(legal_actions)
        x, mask = self._x, self._mask
        x[:] = 0.0
        mask[:] = False
        mask[0, legal] = True
        self.encoder.encode_into(obs, x[0], mask[0])
        x[:] = x.astype(np.float16)  # training inputs are float16-rounded (rollout protocol)
        with torch.inference_mode():
            logits = self.net.policy_logits(torch.from_numpy(x), torch.from_numpy(mask))[0].numpy()
        logits = logits[legal].astype(np.float64)
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
