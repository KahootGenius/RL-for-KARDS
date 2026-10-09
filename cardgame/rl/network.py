"""Policy/value networks over the encoded observation.

* `EntityPolicyNet` (Stage 2 default): shared per-card encoder + learned card-id embedding, masked
  mean/max pooling per zone, optional self-attention, and pointer-style action heads that score
  every PLAY / MOVE / ATTACK(attacker, target) from the slots involved (SPEC §7).
* `PolicyValueNet` (Stage 1): separate policy and value MLPs over the flat vector (baseline).

Both expose: `masked_logits(x, mask)`, `get_value(x)`, `act(x, mask, deterministic)`,
`evaluate(x, mask, actions)` and `spec()` (constructor kwargs, stored in checkpoints).
"""
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


class _ActMixin:
    """Sampling / evaluation shared by both architectures (they implement `policy_logits` and `value`)."""

    def forward(self, obs: torch.Tensor, mask=None):
        return self.policy_logits(obs, mask), self.value(obs)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor, deterministic: bool = False, generator=None):
        """Sample (or argmax) actions; returns (actions, logp). Uses the policy only."""
        logits = self.policy_logits(obs, mask)
        logp_all = torch.log_softmax(logits, dim=-1)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = torch.multinomial(logp_all.exp(), 1, generator=generator).squeeze(-1)
        return actions, logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    def evaluate(self, obs: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(obs, mask)
        logp_all = torch.log_softmax(logits, dim=-1)
        logp = logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(logp_all.exp() * logp_all).sum(-1)
        return logp, entropy, value

    def masked_logits(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.policy_logits(obs, mask)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value(obs)


class PolicyValueNet(_ActMixin, nn.Module):
    """Stage 1 baseline: separate policy and value MLPs over the flat encoded observation."""

    kind = "mlp"

    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        self.obs_dim, self.n_actions, self.hidden = obs_dim, n_actions, tuple(hidden)
        self.policy = mlp(obs_dim, hidden, n_actions, out_gain=0.01)
        self.value_net = mlp(obs_dim, hidden, 1, out_gain=1.0)

    def spec(self) -> dict:
        return {"kind": self.kind, "obs_dim": self.obs_dim, "n_actions": self.n_actions, "hidden": list(self.hidden)}

    def policy_logits(self, obs: torch.Tensor, mask=None) -> torch.Tensor:
        logits = self.policy(obs)
        return logits if mask is None else logits.masked_fill(~mask, MASK_VALUE)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_net(obs).squeeze(-1)


class EntityTower(nn.Module):
    """Per-card encoder (shared by hand and zones) + masked pooling -> (slot embeddings, context)."""

    def __init__(self, layout: dict, d_model: int, id_dim: int, ctx_dim: int, attention_layers: int, heads: int):
        super().__init__()
        self.G, self.E, self.F = layout["G"], layout["E"], layout["F"]
        self.H, self.Z, self.n_cards = layout["H"], layout["Z"], layout["n_cards"]
        self.n_preview = 2 * self.Z * (2 * self.Z + 1) * layout.get("P", 0)
        d = d_model
        self.card_embed = nn.Embedding(self.n_cards + 1, id_dim, padding_idx=0)  # id = card index + 1
        self.slot_enc = nn.Sequential(nn.Linear(self.F + id_dim, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.glob_enc = nn.Sequential(nn.Linear(self.G, d), nn.ReLU())
        self.attn = None
        if attention_layers:
            layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True)
            self.attn = nn.TransformerEncoder(layer, attention_layers, enable_nested_tensor=False)
        H, Z = self.H, self.Z
        self.groups = ((0, H), (H, H + Z), (H + Z, H + 2 * Z), (H + 2 * Z, H + 3 * Z))
        n_in = d + len(self.groups) * (3 * d + 1)
        self.ctx = nn.Sequential(nn.Linear(n_in, ctx_dim), nn.ReLU(), nn.Linear(ctx_dim, ctx_dim), nn.ReLU())

    def split(self, x: torch.Tensor):
        G, E, F = self.G, self.E, self.F
        g = x[:, :G]
        feats = x[:, G:G + E * F].reshape(-1, E, F)
        off_ids = G + E * F + self.n_preview  # attack previews sit between the slots and the ids
        ids = x[:, off_ids:off_ids + E].long()
        present = x[:, off_ids + E:] > 0.5
        return g, feats, ids, present

    @staticmethod
    def _pool(h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Masked mean, max, sum (scaled by capacity) and fill fraction; empty groups pool to zeros."""
        mf = m.unsqueeze(-1).to(h.dtype)
        count = mf.sum(1)
        total = (h * mf).sum(1)
        mean = total / count.clamp(min=1.0)
        mx = h.masked_fill(~m.unsqueeze(-1), -1e4).max(1).values
        mx = torch.where(count > 0, mx, torch.zeros_like(mx))
        return torch.cat([mean, mx, total / m.shape[1], count / m.shape[1]], dim=-1)

    def forward(self, x: torch.Tensor):
        g, feats, ids, present = self.split(x)
        h = self.slot_enc(torch.cat([feats, self.card_embed(ids)], dim=-1))
        h = h * present.unsqueeze(-1).to(h.dtype)
        ge = self.glob_enc(g)
        if self.attn is not None:  # globals token first: never padded, so no row is fully masked
            tokens = torch.cat([ge.unsqueeze(1), h], dim=1)
            pad = torch.cat([torch.zeros_like(present[:, :1]), ~present], dim=1)
            h = self.attn(tokens, src_key_padding_mask=pad)[:, 1:]
            h = h * present.unsqueeze(-1).to(h.dtype)
        pooled = [self._pool(h[:, a:b], present[:, a:b]) for a, b in self.groups]
        return h, self.ctx(torch.cat([ge] + pooled, dim=-1)), g


class EntityPolicyNet(_ActMixin, nn.Module):
    """Policy tower with pointer-style heads + value tower (separate parameters unless shared_trunk).

    `layout` comes from `ObservationEncoder.layout()`. Slot groups: hand [0, H), my backline
    [H, H+Z), frontline [H+Z, H+2Z), opponent backline [H+2Z, H+3Z). Attacker slot a maps to my
    backline (a < Z) or the frontline (a >= Z); target slot t to the opponent backline (t < Z),
    the frontline (Z <= t < 2Z) or the base (t = 2Z), whose token is built from the enemy base HP.
    """

    kind = "entity"

    def __init__(self, layout: dict, d_model: int = 128, id_dim: int = 16, ctx_dim: int = 256,
                 pair_dim: int = 128, pair_mlp_dim: int = 32, attention_layers: int = 0, heads: int = 4,
                 shared_trunk: bool = False):
        super().__init__()
        self.layout = dict(layout)
        L = self.layout
        self.H, self.Z = L["H"], L["Z"]
        self.n_actions, self.obs_dim = L["n_actions"], L["dim"]
        self.opp_base_hp = L["opp_base_hp_index"]
        self.P = L.get("P", 0)
        self.off_preview = L["G"] + L["E"] * L["F"]
        self.d_model, self.id_dim, self.ctx_dim = d_model, id_dim, ctx_dim
        self.pair_dim, self.attention_layers, self.heads = pair_dim, attention_layers, heads
        self.pair_mlp_dim = pair_mlp_dim
        self.shared_trunk = shared_trunk
        tower = lambda: EntityTower(L, d_model, id_dim, ctx_dim, attention_layers, heads)  # noqa: E731
        self.policy_tower = tower()
        self.value_tower = self.policy_tower if shared_trunk else tower()
        d, k = d_model, pair_dim
        # pointer heads. PLAY/MOVE: w . relu(U slot + V ctx). ATTACK(a, t): bilinear attention between
        # a context-conditioned attacker query and a target key ((B, 2Z, k) x (B, k, 2Z+1)), plus a
        # narrow additive pair MLP w . relu(A h_a + T h_t + C ctx) of width pair_mlp_dim, which can
        # express thresholds such as "attack >= remaining HP" that a bilinear form cannot; plus a
        # per-target bias.
        self.end_head = nn.Sequential(nn.Linear(ctx_dim, k), nn.ReLU(), nn.Linear(k, 1))
        self.play_slot, self.play_ctx, self.play_out = nn.Linear(d, k), nn.Linear(ctx_dim, k), nn.Linear(k, 1)
        self.move_slot, self.move_ctx, self.move_out = nn.Linear(d, k), nn.Linear(ctx_dim, k), nn.Linear(k, 1)
        self.att_q_slot, self.att_q_ctx, self.att_q_out = nn.Linear(d, k), nn.Linear(ctx_dim, k), nn.Linear(k, k)
        self.att_key, self.att_bias = nn.Linear(d, k), nn.Linear(d, 1)
        m = pair_mlp_dim
        self.pair_src, self.pair_dst, self.pair_ctx = nn.Linear(d, m), nn.Linear(d, m), nn.Linear(ctx_dim, m)
        self.pair_out = nn.Linear(m, 1)
        if self.P:  # engine attack previews (kills / attacker dies / damage) feed the attack head
            self.prev_pair, self.prev_logit = nn.Linear(self.P, m), nn.Linear(self.P, 1)
        self.base_enc = nn.Linear(2, d)  # [opp base hp, 1] -> base target token
        self.value_head = mlp(ctx_dim, (ctx_dim // 2,), 1, out_gain=1.0)
        for head in (self.end_head[-1], self.play_out, self.move_out, self.att_q_out, self.att_bias, self.pair_out):
            nn.init.orthogonal_(head.weight, gain=0.01)  # near-uniform initial policy
            nn.init.zeros_(head.bias)

    def spec(self) -> dict:
        return {"kind": self.kind, "layout": self.layout, "d_model": self.d_model, "id_dim": self.id_dim,
                "ctx_dim": self.ctx_dim, "pair_dim": self.pair_dim, "pair_mlp_dim": self.pair_mlp_dim,
                "attention_layers": self.attention_layers, "heads": self.heads, "shared_trunk": self.shared_trunk}

    def policy_logits(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        H, Z = self.H, self.Z
        h, ctx, g = self.policy_tower(x)
        B = x.shape[0]
        end = self.end_head(ctx)                                                             # (B, 1)
        play = self.play_out(torch.relu(self.play_slot(h[:, :H]) + self.play_ctx(ctx)[:, None])).squeeze(-1)
        my_back, front, opp_back = h[:, H:H + Z], h[:, H + Z:H + 2 * Z], h[:, H + 2 * Z:H + 3 * Z]
        move = self.move_out(torch.relu(self.move_slot(my_back) + self.move_ctx(ctx)[:, None])).squeeze(-1)
        base_in = torch.cat([g[:, self.opp_base_hp:self.opp_base_hp + 1], torch.ones_like(g[:, :1])], dim=-1)
        base = self.base_enc(base_in).unsqueeze(1)                                           # (B, 1, d)
        attackers = torch.cat([my_back, front], dim=1)                                       # (B, 2Z, d)
        targets = torch.cat([opp_back, front, base], dim=1)                                  # (B, 2Z+1, d)
        q = self.att_q_out(torch.relu(self.att_q_slot(attackers) + self.att_q_ctx(ctx)[:, None]))  # (B, 2Z, k)
        key = self.att_key(targets)                                                          # (B, 2Z+1, k)
        attack = torch.bmm(q, key.transpose(1, 2)) * self.pair_dim ** -0.5                  # (B, 2Z, 2Z+1)
        pre = self.pair_src(attackers)[:, :, None] + self.pair_dst(targets)[:, None] + self.pair_ctx(ctx)[:, None, None]
        if self.P:
            n_a, n_t = 2 * Z, 2 * Z + 1
            preview = x[:, self.off_preview:self.off_preview + n_a * n_t * self.P].reshape(B, n_a, n_t, self.P)
            pre = pre + self.prev_pair(preview)
            attack = attack + self.prev_logit(preview).squeeze(-1)
        attack = attack + self.pair_out(torch.relu(pre)).squeeze(-1)                         # (B, 2Z, 2Z+1)
        attack = (attack + self.att_bias(targets).transpose(1, 2)).reshape(B, -1)            # (B, 2Z(2Z+1))
        logits = torch.cat([end, play, move, attack], dim=-1)
        return logits if mask is None else logits.masked_fill(~mask, MASK_VALUE)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        _, ctx, _ = self.value_tower(x)
        return self.value_head(ctx).squeeze(-1)


def build_net(spec: dict) -> nn.Module:
    """Rebuild a network from `net.spec()` (as stored in checkpoints)."""
    spec = dict(spec)
    kind = spec.pop("kind", "mlp")
    if kind == "entity":
        return EntityPolicyNet(**spec)
    if kind == "mlp":
        return PolicyValueNet(spec["obs_dim"], spec["n_actions"], spec["hidden"])
    raise ValueError(f"unknown network kind {kind!r}")
