"""Policy/value networks over the v5 token encoding (SPEC §7).

* `TransformerPolicyNet` (kind "transformer", the Stage 3 default): card MLP + token MLP -> pre-norm
  Transformer encoder over the T tokens -> one pointer scorer for every action type.
* `PooledPolicyNet` (kind "pooled", baseline): the same token embeddings, masked mean/max/sum/fill
  pooling per token group and a context MLP in place of attention; same heads.
* `PolicyValueNet` (kind "mlp", sanity baseline): MLPs over the flat vector.

Common interface (all three):
    policy_logits(x, mask=None) -> (B, N) logits, illegal actions at MASK_VALUE
    value(x, priv=None)         -> (B,)   (priv = the opponent's true hand counts, privileged critic only)
    belief_logits(x)            -> (B, n_cards) multi-label logits of the opponent's hand, or None
    act(x, mask, deterministic=False, generator=None) -> (actions, logp)        (policy only, no grad)
    evaluate(x, mask, actions, priv=None) -> (logp, entropy, value, belief_logits or None)
    spec()                      -> JSON-safe constructor kwargs (with "kind"); `build_net(spec)` rebuilds
    attributes obs_dim, n_actions, n_cards, kind

Token embedding (shared by the two token nets; one copy per tower):
    cardvec_c = card_mlp([card_table[c], Embedding(c + 1)])          for every card of the pool
    token_i   = token_mlp([cardvec[id_i] (0 for id 0), features_i])
i.e. MLP([card_table[id], features, Embedding(id)]) with the card-only part factored out, so it runs
once per forward over the n_cards pool rows instead of once per token, and the same `cardvec` feeds
the deck token and the privileged critic input. The global token row gets `+ Linear(globals)`, the
deck token row `+ Linear(sum_c counts_c * cardvec_c / 40)`; in a privileged value tower the global row
also gets `+ Linear(sum_c priv_c * cardvec_c / 10)`. The policy tower has no privileged input module
at all, so the actor cannot see `priv`.

Transformer tower: pre-norm `nn.TransformerEncoder` (dropout 0, final LayerNorm), key padding from
token presence (global, base and deck tokens are forced present, so no row is ever fully masked).
Each row's present tokens are packed to the front and the batch is truncated to its longest row
before the encoder (no positional encoding, so this only saves compute and activation memory).

Pointer scorer: every action index n has a (source token, target token, type) triple fixed by the
layout (`PointerScorer`), so the whole (B, N) logit row comes out in action-index order:
    score = q(src, g, type) . k(tgt) / sqrt(key_dim) + w . relu(A src + T tgt + C g + E_type + P preview)
            + b(tgt)
with q(src, g, type) = Q_out relu(Q_src src + Q_g g + Q_type) computed once per distinct (source, type)
pair and scored against every token with one bmm, g = the global token output (pooled net: the
context vector), and P a separate linear map for attack and for choose previews.
"""
from __future__ import annotations

import json
import math
from typing import NamedTuple, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tF

MASK_VALUE = -1e9  # finite so masked entries give p=0 and p*log(p)=0 (no NaNs)
PRIV_SCALE = 10.0  # privileged critic input: sum_c opp_hand_c * cardvec_c / 10 (SPEC §7)
NET_KINDS = ("transformer", "pooled", "mlp")
STAGE2_KINDS = ("entity",)
ACTION_TYPES = ("END_TURN", "PLAY", "MOVE", "ATTACK", "CHOOSE", "MULLIGAN", "CONFIRM")
NULL_TARGET_TYPES = ("END_TURN", "PLAY", "MOVE", "MULLIGAN", "CONFIRM")  # types scored against a learned null
ALWAYS_PRESENT_GROUPS = ("global", "my_base", "opp_base", "deck")       # SPEC §6: never absent
POOLED_GROUPS = ("hand", "my_back", "front", "opp_back", "revealed")    # variable-size groups pooled
SINGLE_TOKENS = ("global", "my_base", "opp_base", "pending", "deck")    # fixed tokens concatenated
_LAYOUT_KEYS = ("version", "fingerprint", "G", "T", "F", "S", "P", "PC", "H", "Z", "n_cards", "dim", "n_actions",
                "offsets", "groups", "attacker_tokens", "attack_target_tokens", "choose_tokens", "END_TURN", "PLAY0",
                "MOVE0", "ATTACK0", "CHOOSE0", "MULLIGAN0", "CONFIRM", "n_attackers", "n_targets", "n_choose",
                "card_table")


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


def _small(layer: nn.Linear, gain: float = 0.01) -> nn.Linear:
    """Near-zero output layer (near-uniform initial policy, as in Stage 2)."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def _json_copy(obj):
    """Private, JSON-safe copy (spec() must be JSON-serialisable and compare equal after a round trip)."""
    return json.loads(json.dumps(obj))


# ====================================================================== layout helpers
class Parts(NamedTuple):
    globals: torch.Tensor         # (B, G)
    features: torch.Tensor        # (B, T, F)
    ids: torch.Tensor             # (B, T) long, card index + 1 (0 = none)
    present: torch.Tensor         # (B, T) bool, always-present tokens forced True
    deck_counts: torch.Tensor     # (B, n_cards) raw remaining counts
    attack_preview: torch.Tensor  # (B, n_attackers * n_targets, P), ATTACK-block order
    choose_preview: torch.Tensor  # (B, n_choose, PC)


class _Encoding:
    """Plain-Python view of an encoder layout (sizes, offsets, groups) with a torch `split`."""

    def __init__(self, layout: dict):
        missing = [k for k in _LAYOUT_KEYS if k not in layout]
        if missing:
            raise ValueError(f"layout lacks {missing}: not a Stage 3 (encoder v5) layout; build it with "
                             f"ObservationEncoder(config).layout()")
        L = layout
        self.G, self.T, self.F, self.S = L["G"], L["T"], L["F"], L["S"]
        self.P, self.PC, self.H, self.Z = L["P"], L["PC"], L["H"], L["Z"]
        self.n_cards, self.dim, self.n_actions = L["n_cards"], L["dim"], L["n_actions"]
        self.n_att, self.n_tgt, self.n_choose = L["n_attackers"], L["n_targets"], L["n_choose"]
        off = L["offsets"]
        self.o_g, self.o_f, self.o_id = off["globals"], off["features"], off["ids"]
        self.o_pr, self.o_dk = off["present"], off["deck_counts"]
        self.o_ap, self.o_cp = off["attack_preview"], off["choose_preview"]
        sizes = [(self.o_g, self.G), (self.o_f, self.T * self.F), (self.o_id, self.T), (self.o_pr, self.T),
                 (self.o_dk, self.n_cards), (self.o_ap, self.n_att * self.n_tgt * self.P),
                 (self.o_cp, self.n_choose * self.PC)]
        if sorted(sizes) != sizes or any(a + n != b for (a, n), (b, _) in zip(sizes, sizes[1:])) \
                or sizes[0][0] != 0 or sizes[-1][0] + sizes[-1][1] != self.dim:
            raise ValueError("layout offsets do not tile the flat vector in SPEC §6 order")
        self.groups = {k: tuple(v) for k, v in L["groups"].items()}
        for g in ALWAYS_PRESENT_GROUPS + ("hand", "my_back", "front", "opp_back", "pending", "revealed"):
            if g not in self.groups:
                raise ValueError(f"layout has no token group {g!r}")
        self.tok_global = self.groups["global"][0]
        self.tok_deck = self.groups["deck"][0]
        self.tok_pending = self.groups["pending"][0]
        self.always = sorted({t for g in ALWAYS_PRESENT_GROUPS for t in range(*self.groups[g])})
        deck_scale = L.get("scales", {}).get("deck", 40.0)
        self.deck_div = float(deck_scale) / float(L.get("deck_counts_scale", 1.0))  # raw counts / 40

    def split(self, x: torch.Tensor) -> Parts:
        if x.dim() != 2 or x.shape[1] != self.dim:
            raise ValueError(f"expected encodings of shape (B, {self.dim}), got {tuple(x.shape)}")
        B, T = x.shape[0], self.T
        g = x[:, self.o_g:self.o_f]
        feats = x[:, self.o_f:self.o_id].reshape(B, T, self.F)
        ids = x[:, self.o_id:self.o_pr].round().long()
        present = x[:, self.o_pr:self.o_dk] > 0.5
        deck = x[:, self.o_dk:self.o_ap]
        ap = x[:, self.o_ap:self.o_cp].reshape(B, self.n_att * self.n_tgt, self.P)
        cp = x[:, self.o_cp:self.dim].reshape(B, self.n_choose, self.PC)
        return Parts(g, feats, ids, present, deck, ap, cp)


# ====================================================================== shared mixin
class _ActMixin:
    """Sampling / evaluation shared by every architecture."""

    def forward(self, x: torch.Tensor, mask=None, priv=None):
        return self.policy_logits(x, mask), self.value(x, priv)

    @torch.no_grad()
    def act(self, x: torch.Tensor, mask: torch.Tensor, deterministic: bool = False, generator=None):
        """Sample (or argmax) actions from the policy; returns (actions, logp). Never takes `priv`."""
        logits = self.policy_logits(x, mask)
        logp_all = torch.log_softmax(logits, dim=-1)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = torch.multinomial(logp_all.exp(), 1, generator=generator).squeeze(-1)
        return actions, logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _masked(logits: torch.Tensor, mask) -> torch.Tensor:
        return logits if mask is None else logits.masked_fill(~mask, MASK_VALUE)

    @staticmethod
    def _logp_entropy(logits: torch.Tensor, actions: torch.Tensor):
        logp_all = torch.log_softmax(logits, dim=-1)
        logp = logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(logp_all.exp() * logp_all).sum(-1)
        return logp, entropy

    def _check_priv(self, priv, batch: int):
        """The privileged input (or None when this net has no privileged critic; `priv` is then ignored)."""
        if not self.privileged_critic:
            return None
        if priv is None:
            raise ValueError("this network has a privileged critic: value()/evaluate() need priv = the opponent's "
                             "true hand counts, a float tensor of shape (B, n_cards)")
        if priv.dim() != 2 or priv.shape != (batch, self.n_cards):
            raise ValueError(f"priv must have shape ({batch}, {self.n_cards}), got {tuple(priv.shape)}")
        return priv.to(torch.float32) if not torch.is_floating_point(priv) else priv


# ====================================================================== flat MLP baseline
class PolicyValueNet(_ActMixin, nn.Module):
    """Sanity baseline: policy and value MLPs over the flat encoded vector (kind "mlp").

    Optional heads mirror the token nets: `belief` = Linear(last policy hidden layer) -> n_cards;
    `privileged_critic` = the value MLP also reads priv / 3 (the counts scale)."""

    kind = "mlp"

    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (256, 256), n_cards: int = 0,
                 belief: bool = False, privileged_critic: bool = False, version: Optional[int] = None,
                 fingerprint: Optional[str] = None):
        super().__init__()
        if (belief or privileged_critic) and n_cards <= 0:
            raise ValueError("belief / privileged_critic need n_cards > 0")
        self.obs_dim, self.n_actions, self.hidden = int(obs_dim), int(n_actions), tuple(int(h) for h in hidden)
        self.n_cards, self.belief, self.privileged_critic = int(n_cards), bool(belief), bool(privileged_critic)
        self.version, self.fingerprint = version, fingerprint
        self.shared_trunk = False
        body = mlp(obs_dim, self.hidden, 1, out_gain=1.0)
        self.policy_body = body[:-1] if self.hidden else nn.Identity()  # Linear/ReLU stack
        last = self.hidden[-1] if self.hidden else obs_dim
        self.policy_out = _small(nn.Linear(last, n_actions))
        self.belief_head = _small(nn.Linear(last, self.n_cards)) if self.belief else None
        self.value_net = mlp(obs_dim + (self.n_cards if self.privileged_critic else 0), self.hidden, 1, out_gain=1.0)

    @classmethod
    def from_layout(cls, layout: dict, hidden: Sequence[int] = (256, 256), belief: bool = True,
                    privileged_critic: bool = False) -> "PolicyValueNet":
        return cls(layout["dim"], layout["n_actions"], hidden, n_cards=layout["n_cards"], belief=belief,
                   privileged_critic=privileged_critic, version=layout.get("version"),
                   fingerprint=layout.get("fingerprint"))

    def spec(self) -> dict:
        return {"kind": self.kind, "obs_dim": self.obs_dim, "n_actions": self.n_actions, "hidden": list(self.hidden),
                "n_cards": self.n_cards, "belief": self.belief, "privileged_critic": self.privileged_critic,
                "version": self.version, "fingerprint": self.fingerprint}

    def policy_logits(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        return self._masked(self.policy_out(self.policy_body(x)), mask)

    def value(self, x: torch.Tensor, priv=None) -> torch.Tensor:
        priv = self._check_priv(priv, x.shape[0])
        inp = x if priv is None else torch.cat([x, priv / 3.0], dim=-1)
        return self.value_net(inp).squeeze(-1)

    def belief_logits(self, x: torch.Tensor):
        return None if self.belief_head is None else self.belief_head(self.policy_body(x))

    def evaluate(self, x: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor, priv=None):
        h = self.policy_body(x)
        logp, entropy = self._logp_entropy(self._masked(self.policy_out(h), mask), actions)
        belief = None if self.belief_head is None else self.belief_head(h)
        return logp, entropy, self.value(x, priv), belief


# ====================================================================== token nets
class TokenEmbedder(nn.Module):
    """Card MLP + token MLP + global / deck (/ privileged) inputs -> (B, T, d) token embeddings."""

    def __init__(self, enc: _Encoding, card_table: np.ndarray, d: int, id_dim: int, privileged: bool):
        super().__init__()
        self.enc = enc
        table = np.zeros((enc.n_cards + 1, enc.S), dtype=np.float32)  # row 0 = no card
        table[1:] = card_table
        self.register_buffer("card_table", torch.from_numpy(table), persistent=False)  # rebuilt from the layout
        self.card_embed = nn.Embedding(enc.n_cards + 1, id_dim, padding_idx=0)
        self.card_mlp = mlp(enc.S + id_dim, (d,), d, out_gain=1.0)
        self.token_mlp = mlp(d + enc.F, (d,), d, out_gain=1.0)
        self.global_in = nn.Linear(enc.G, d)
        self.deck_in = nn.Linear(d, d)
        self.priv_in = nn.Linear(d, d) if privileged else None

    def cardvecs(self) -> torch.Tensor:
        """(n_cards + 1, d) card vectors, row 0 (no card) = 0. Depends only on parameters: computed once
        per forward and shared by every token, the deck token and the privileged input."""
        cv = self.card_mlp(torch.cat([self.card_table[1:], self.card_embed.weight[1:]], dim=-1))
        return torch.cat([cv.new_zeros(1, cv.shape[1]), cv], dim=0)

    def forward(self, parts: Parts, priv: Optional[torch.Tensor] = None) -> torch.Tensor:
        enc = self.enc
        cv = self.cardvecs()
        h = self.token_mlp(torch.cat([tF.embedding(parts.ids, cv), parts.features], dim=-1))  # (B, T, d)
        g_add = self.global_in(parts.globals)
        if priv is not None:
            if self.priv_in is None:
                raise ValueError("this tower has no privileged input")
            g_add = g_add + self.priv_in(priv @ cv[1:] / PRIV_SCALE)
        deck_add = self.deck_in(parts.deck_counts @ cv[1:] / enc.deck_div)
        extra = torch.zeros_like(h)
        extra[:, enc.tok_global] = g_add
        extra[:, enc.tok_deck] = deck_add
        return h + extra


class _TransformerTower(nn.Module):
    """Token embeddings -> pre-norm Transformer encoder (key padding from token presence, no positional
    encoding) -> (token outputs (B, T, d), global token output (B, d))."""

    def __init__(self, enc: _Encoding, card_table, d: int, layers: int, heads: int, ff: int, id_dim: int,
                 privileged: bool):
        super().__init__()
        self.embed = TokenEmbedder(enc, card_table, d, id_dim, privileged)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=ff, dropout=0.0, activation="relu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.tok_global = enc.tok_global

    @staticmethod
    def pack_order(present: torch.Tensor) -> torch.Tensor:
        """(B, L) token indices with each row's present tokens first (in token order), L = the largest
        present count in the batch. Sort-free (cumsum + scatter), so it runs on every device."""
        B, T = present.shape
        p = present.long()
        count = p.sum(1, keepdim=True)
        dest = torch.where(present, p.cumsum(1) - 1, count + (1 - p).cumsum(1) - 1)  # packed slot of each token
        order = torch.empty_like(dest).scatter_(1, dest, torch.arange(T, device=present.device).expand(B, T))
        return order[:, :int(count.max())]

    def forward(self, parts: Parts, priv=None):
        present = parts.present
        x = self.embed(parts, priv)
        # Only present tokens matter (absent ones are key-padded and zeroed), and with no positional
        # encoding the token order is irrelevant: move each row's present tokens to the front and drop the
        # trailing columns absent in every row. Same outputs, a fraction of the cost and activation memory
        # (on average ~23 of the 50 tokens are present; a minibatch sorted by present count packs best).
        order = self.pack_order(present)
        idx = order.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        out = self.encoder(x.gather(1, idx), src_key_padding_mask=~present.gather(1, order))
        # The buffer takes the encoder output's dtype: under CUDA (and MPS) bf16 autocast the final
        # LayerNorm runs in fp32 while the token embedding `x` is bf16, and scatter needs equal dtypes.
        h = out.new_zeros(x.shape).scatter(1, idx, out)
        h = h * present.unsqueeze(-1).to(h.dtype)
        return h, h[:, self.tok_global]


class _PooledTower(nn.Module):
    """Token embeddings -> masked pooling per group + fixed tokens -> context MLP (no attention)."""

    def __init__(self, enc: _Encoding, card_table, d: int, ctx_dim: int, id_dim: int, privileged: bool):
        super().__init__()
        self.embed = TokenEmbedder(enc, card_table, d, id_dim, privileged)
        self.pool_groups = tuple(enc.groups[g] for g in POOLED_GROUPS)
        self.single = [enc.groups[g][0] for g in SINGLE_TOKENS]
        for g in SINGLE_TOKENS:
            if enc.groups[g][1] - enc.groups[g][0] != 1:
                raise ValueError(f"token group {g!r} must be a single token")
        n_in = len(self.single) * d + len(self.pool_groups) * (3 * d + 1)
        self.ctx = mlp(n_in, (ctx_dim,), ctx_dim, out_gain=np.sqrt(2))
        self.register_buffer("single_idx", torch.tensor(self.single, dtype=torch.long), persistent=False)

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

    def forward(self, parts: Parts, priv=None):
        present = parts.present
        h = self.embed(parts, priv)
        h = h * present.unsqueeze(-1).to(h.dtype)
        B = h.shape[0]
        singles = h.index_select(1, self.single_idx).reshape(B, -1)
        pooled = [self._pool(h[:, a:b], present[:, a:b]) for a, b in self.pool_groups]
        ctx = torch.relu(self.ctx(torch.cat([singles] + pooled, dim=-1)))
        return h, ctx


class PointerScorer(nn.Module):
    """One scorer for every action type; returns the (B, N) logits in action-index order.

    Wiring (from the layout): ATTACK(a, t) = (attacker_tokens[a], attack_target_tokens[t]) with the
    attack preview; CHOOSE(t) = (pending token, choose_tokens[t]) with the choose preview; PLAY(i) and
    MULLIGAN(i) = (hand token i, null target of the type); MOVE(j) = (my backline token j, null);
    END_TURN and CONFIRM = (global token, null). Null targets are learned vectors in token space."""

    def __init__(self, enc: _Encoding, layout: dict, d_tok: int, d_g: int, key_dim: int, pair_dim: int):
        super().__init__()
        N, T, H, Z = enc.n_actions, enc.T, enc.H, enc.Z
        n_att, n_tgt, n_ch = enc.n_att, enc.n_tgt, enc.n_choose
        src, tgt, typ = np.full(N, -1), np.full(N, -1), np.full(N, -1)
        null = {t: T + k for k, t in enumerate(NULL_TARGET_TYPES)}
        ty = {t: k for k, t in enumerate(ACTION_TYPES)}
        hand0, back0 = enc.groups["hand"][0], enc.groups["my_back"][0]

        def put(n: int, s: int, t: int, kind: str) -> None:
            if not 0 <= n < N or src[n] >= 0:
                raise ValueError(f"layout action blocks overlap or leave [0, {N}) at index {n}")
            src[n], tgt[n], typ[n] = s, t, ty[kind]

        put(layout["END_TURN"], enc.tok_global, null["END_TURN"], "END_TURN")
        put(layout["CONFIRM"], enc.tok_global, null["CONFIRM"], "CONFIRM")
        for i in range(H):
            put(layout["PLAY0"] + i, hand0 + i, null["PLAY"], "PLAY")
            put(layout["MULLIGAN0"] + i, hand0 + i, null["MULLIGAN"], "MULLIGAN")
        for j in range(Z):
            put(layout["MOVE0"] + j, back0 + j, null["MOVE"], "MOVE")
        att, tgts, chs = layout["attacker_tokens"], layout["attack_target_tokens"], layout["choose_tokens"]
        if (len(att), len(tgts), len(chs)) != (n_att, n_tgt, n_ch):
            raise ValueError("layout slot->token maps do not match n_attackers / n_targets / n_choose")
        for a in range(n_att):
            for t in range(n_tgt):
                put(layout["ATTACK0"] + a * n_tgt + t, att[a], tgts[t], "ATTACK")
        for t in range(n_ch):
            put(layout["CHOOSE0"] + t, enc.tok_pending, chs[t], "CHOOSE")
        if (src < 0).any():
            raise ValueError(f"layout leaves action indices {np.flatnonzero(src < 0).tolist()} unwired")
        # distinct (source, type) pairs: one query each, scored against every bank token with one bmm, so
        # no (B, N, key_dim) tensor is ever built (that would be ~0.6 GB per tensor at an 8192 minibatch)
        pairs = sorted(set(zip(src.tolist(), typ.tolist())))
        row = {p: k for k, p in enumerate(pairs)}
        q_row = np.array([row[(s, t)] for s, t in zip(src.tolist(), typ.tolist())])
        n_bank = T + len(NULL_TARGET_TYPES)
        for name, arr in (("src_idx", src), ("tgt_idx", tgt), ("type_idx", typ),
                          ("q_src_idx", [p[0] for p in pairs]), ("q_type_idx", [p[1] for p in pairs]),
                          ("bilinear_idx", q_row * n_bank + tgt)):
            self.register_buffer(name, torch.tensor(np.asarray(arr), dtype=torch.long), persistent=False)
        self.att0, self.n_att_pairs, self.ch0, self.n_ch = layout["ATTACK0"], n_att * n_tgt, layout["CHOOSE0"], n_ch
        self.P, self.PC, self.N = enc.P, enc.PC, N
        n_types = len(ACTION_TYPES)
        self.null = nn.Parameter(torch.randn(len(NULL_TARGET_TYPES), d_tok))
        # bilinear part
        self.q_src, self.q_g = nn.Linear(d_tok, key_dim), nn.Linear(d_g, key_dim, bias=False)
        self.q_type = nn.Parameter(torch.zeros(n_types, key_dim))
        self.q_out = _small(nn.Linear(key_dim, key_dim))
        self.key = nn.Linear(d_tok, key_dim)
        self.scale = 1.0 / math.sqrt(key_dim)
        # additive pair MLP (thresholds such as "attack >= remaining hp" that a bilinear form cannot express)
        self.p_src, self.p_tgt = nn.Linear(d_tok, pair_dim), nn.Linear(d_tok, pair_dim, bias=False)
        self.p_g = nn.Linear(d_g, pair_dim, bias=False)
        self.p_type = nn.Parameter(torch.zeros(n_types, pair_dim))
        self.p_prev_attack = nn.Linear(self.P, pair_dim, bias=False)  # attack previews (ATTACK block)
        self.p_prev_choose = nn.Linear(self.PC, pair_dim, bias=False)  # choose previews (CHOOSE block)
        self.p_out = _small(nn.Linear(pair_dim, 1))
        self.t_bias = _small(nn.Linear(d_tok, 1))

    def forward(self, h: torch.Tensor, g: torch.Tensor, parts: Parts) -> torch.Tensor:
        B = h.shape[0]
        bank = torch.cat([h, self.null.unsqueeze(0).expand(B, -1, -1)], dim=1)       # (B, T + n_null, d)
        q = torch.relu(self.q_src(bank).index_select(1, self.q_src_idx) + self.q_g(g).unsqueeze(1)
                       + self.q_type[self.q_type_idx])                                 # (B, U, k)
        scores = torch.bmm(self.q_out(q), self.key(bank).transpose(1, 2))             # (B, U, T + n_null)
        bilinear = scores.reshape(B, -1).index_select(1, self.bilinear_idx) * self.scale  # (B, N)
        pre = self.p_src(bank).index_select(1, self.src_idx) + self.p_tgt(bank).index_select(1, self.tgt_idx)
        pre = pre + (self.p_g(g).unsqueeze(1) + self.p_type[self.type_idx])          # (B, N, pair)
        pre[:, self.att0:self.att0 + self.n_att_pairs] += self.p_prev_attack(parts.attack_preview)
        pre[:, self.ch0:self.ch0 + self.n_ch] += self.p_prev_choose(parts.choose_preview)
        additive = self.p_out(torch.relu(pre)).squeeze(-1)
        bias = self.t_bias(bank).squeeze(-1).index_select(1, self.tgt_idx)
        return bilinear + additive + bias


class _PointerNet(_ActMixin, nn.Module):
    """Shared plumbing of the two token nets: towers, pointer scorer, value and belief heads."""

    kind = ""

    def _setup(self, layout: dict, d_g: int, key_dim: int, pair_dim: int, belief: bool, privileged_critic: bool,
               shared_trunk: bool, make_tower, value_hidden: int) -> None:
        if shared_trunk and privileged_critic:
            raise ValueError("shared_trunk cannot be combined with privileged_critic: the actor would see the "
                             "opponent's hand through the shared trunk")
        self.layout = _json_copy(layout)
        self.enc = enc = _Encoding(self.layout)
        card_table = np.asarray(self.layout["card_table"], dtype=np.float32).reshape(enc.n_cards, enc.S)
        self.obs_dim, self.n_actions, self.n_cards = enc.dim, enc.n_actions, enc.n_cards
        self.belief, self.privileged_critic = bool(belief), bool(privileged_critic)
        self.shared_trunk = bool(shared_trunk)
        self.key_dim, self.pair_dim = int(key_dim), int(pair_dim)
        self.register_buffer("always_present", torch.zeros(enc.T, dtype=torch.bool), persistent=False)
        self.always_present[enc.always] = True
        self.policy_tower = make_tower(enc, card_table, False)
        self.value_tower = self.policy_tower if shared_trunk else make_tower(enc, card_table, privileged_critic)
        self.scorer = PointerScorer(enc, self.layout, self.d_model, d_g, key_dim, pair_dim)
        self.value_head = mlp(d_g, (value_hidden,), 1, out_gain=1.0)
        self.belief_head = _small(nn.Linear(d_g, enc.n_cards)) if self.belief else None

    def _parts(self, x: torch.Tensor) -> Parts:
        p = self.enc.split(x)
        return p._replace(present=p.present | self.always_present)  # global/base/deck: never masked

    def policy_logits(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        parts = self._parts(x)
        h, g = self.policy_tower(parts)
        return self._masked(self.scorer(h, g, parts), mask)

    def value(self, x: torch.Tensor, priv=None) -> torch.Tensor:
        priv = self._check_priv(priv, x.shape[0])
        _, g = self.value_tower(self._parts(x), priv)
        return self.value_head(g).squeeze(-1)

    def belief_logits(self, x: torch.Tensor):
        if self.belief_head is None:
            return None
        return self.belief_head(self.policy_tower(self._parts(x))[1])

    def evaluate(self, x: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor, priv=None):
        priv = self._check_priv(priv, x.shape[0])
        parts = self._parts(x)
        h, g = self.policy_tower(parts)
        logp, entropy = self._logp_entropy(self._masked(self.scorer(h, g, parts), mask), actions)
        gv = g if self.shared_trunk else self.value_tower(parts, priv)[1]
        belief = None if self.belief_head is None else self.belief_head(g)
        return logp, entropy, self.value_head(gv).squeeze(-1), belief


class TransformerPolicyNet(_PointerNet):
    """Stage 3 default (SPEC §7): token embeddings -> pre-norm Transformer -> pointer scorer.

    The global token output g conditions the scorer and feeds the value head (value tower) and the
    belief head (policy tower)."""

    kind = "transformer"

    def __init__(self, layout: dict, d_model: int = 128, layers: int = 3, heads: int = 4, ff: int = 256,
                 id_dim: int = 16, belief: bool = True, privileged_critic: bool = False, shared_trunk: bool = False,
                 key_dim: Optional[int] = None, pair_dim: int = 32):
        super().__init__()
        if d_model % heads:
            raise ValueError(f"d_model={d_model} must be divisible by heads={heads}")
        if layers < 1:
            raise ValueError("layers must be >= 1")
        self.d_model, self.layers, self.heads, self.ff, self.id_dim = d_model, layers, heads, ff, id_dim
        key_dim = d_model if key_dim is None else key_dim

        def make_tower(enc, card_table, privileged):
            return _TransformerTower(enc, card_table, d_model, layers, heads, ff, id_dim, privileged)

        self._setup(layout, d_model, key_dim, pair_dim, belief, privileged_critic, shared_trunk, make_tower,
                    value_hidden=d_model)

    def spec(self) -> dict:
        return {"kind": self.kind, "layout": self.layout, "d_model": self.d_model, "layers": self.layers,
                "heads": self.heads, "ff": self.ff, "id_dim": self.id_dim, "belief": self.belief,
                "privileged_critic": self.privileged_critic, "shared_trunk": self.shared_trunk,
                "key_dim": self.key_dim, "pair_dim": self.pair_dim}


class PooledPolicyNet(_PointerNet):
    """Baseline (SPEC §7): the same token embeddings, masked mean/max/sum/fill pooling per variable token
    group (hand, my backline, frontline, opponent backline, revealed) plus the fixed tokens (global, both
    bases, pending, deck) concatenated -> context MLP. The context plays the role of g; the pointer scorer
    reads the per-token embeddings (no attention)."""

    kind = "pooled"

    def __init__(self, layout: dict, d_model: int = 128, ctx_dim: int = 256, id_dim: int = 16, belief: bool = True,
                 privileged_critic: bool = False, shared_trunk: bool = False, key_dim: Optional[int] = None,
                 pair_dim: int = 32):
        super().__init__()
        self.d_model, self.ctx_dim, self.id_dim = d_model, ctx_dim, id_dim
        key_dim = d_model if key_dim is None else key_dim

        def make_tower(enc, card_table, privileged):
            return _PooledTower(enc, card_table, d_model, ctx_dim, id_dim, privileged)

        self._setup(layout, ctx_dim, key_dim, pair_dim, belief, privileged_critic, shared_trunk, make_tower,
                    value_hidden=ctx_dim // 2)

    def spec(self) -> dict:
        return {"kind": self.kind, "layout": self.layout, "d_model": self.d_model, "ctx_dim": self.ctx_dim,
                "id_dim": self.id_dim, "belief": self.belief, "privileged_critic": self.privileged_critic,
                "shared_trunk": self.shared_trunk, "key_dim": self.key_dim, "pair_dim": self.pair_dim}


class EntityPolicyNet:
    """Removed Stage 2 network (encoder v4). Kept as a name only so Stage 2 imports fail with a clear
    message instead of an ImportError; use TransformerPolicyNet or PooledPolicyNet."""

    kind = "entity"

    def __init__(self, *args, **kwargs):
        raise ValueError("EntityPolicyNet is the Stage 2 network (encoder v4) and was removed in Stage 3; use "
                         "TransformerPolicyNet (default) or PooledPolicyNet (baseline)")


def build_net(spec: dict) -> nn.Module:
    """Rebuild a network from `net.spec()` (as stored in checkpoints)."""
    spec = dict(spec)
    if "kind" not in spec:
        raise ValueError("network spec without 'kind': a Stage 1 checkpoint, not loadable in Stage 3")
    kind = spec.pop("kind")
    if kind in STAGE2_KINDS:
        raise ValueError(f"network kind {kind!r} is the Stage 2 architecture (encoder v4), not loadable in Stage 3")
    if kind == "transformer":
        return TransformerPolicyNet(**spec)
    if kind == "pooled":
        return PooledPolicyNet(**spec)
    if kind == "mlp":
        if spec.get("version") is None:
            raise ValueError("'mlp' spec without an encoder version: a Stage 2 checkpoint, not loadable in Stage 3")
        return PolicyValueNet(**spec)
    raise ValueError(f"unknown network kind {kind!r} (expected one of {NET_KINDS})")
