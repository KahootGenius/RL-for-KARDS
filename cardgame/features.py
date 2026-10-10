"""Observation -> token encoding for the Stage 3 Transformer (SPEC §6, ENCODER_VERSION 5).

The observation becomes T = 50 tokens in a fixed order (global, hand H, my backline Z, frontline Z,
opponent backline Z, my base, opponent base, pending, opponent-revealed cards R, own deck). Every
token has one feature row of the common schema `TOKEN_FEATURES` (unused entries 0), an id (card
index + 1, 0 = none) and a presence bit. The network looks static card data up by id in the card
table (`layout()["card_table"]`), so token rows only carry what changes during a game.

One flat float32 vector keeps the PPO buffers simple:
    [globals G | token features T*F | token ids T | token present T | deck counts n_cards
     | attack previews 2Z*(2Z+1)*4 | choose previews (3Z+2)*4]
`split()` recovers the named parts; `layout()` carries every size, offset and index a network needs.

Only the Observation and the observer's legal mask are read: nothing hidden can leak in. Every
"can it act now" hint is read from the mask, never re-derived from the rules; attack outcomes come
from `engine.combat_damage` and choice outcomes are copied from the engine's `PendingView.previews`.
Scales are fixed constants (never derived from the card pool), so retuning cards does not silently
rescale the inputs of existing checkpoints; `pool_fingerprint()` detects such changes instead.
"""
from __future__ import annotations

import hashlib
import json
from typing import NamedTuple, Optional

import numpy as np

from .actions import ActionSpace
from .cards import FAST, RANGED, TRAIT_BITS, TROOP, UNCAPPED_BASE_HP, GameConfig
from .engine import Observation, PendingView, UnitView, combat_damage

ENCODER_VERSION = 5  # bump whenever the layout or the meaning of a feature changes

# ---------------------------------------------------------------- fixed scales (SPEC §6)
COST_SCALE, ATK_SCALE, HP_SCALE, MOVE_SCALE, ARMOR_SCALE = 8.0, 10.0, 10.0, 4.0, 4.0
COIN_SCALE, ROUND_SCALE, BASE_SCALE, HAND_SCALE, DECK_SCALE = 10.0, 50.0, 20.0, 10.0, 40.0
COUNT_SCALE, TURN_SCALE, ATTACKS_SCALE = 3.0, 100.0, 2.0
AMOUNT_SCALE = HP_SCALE   # effect amounts (damage, heal, buff, draw, coins, copies) and choose previews
BURNED_SCALE = HAND_SCALE
PIN_SCALE = 2.0           # pin_turns (at most 2 in play)
HISTORY_TURN_SCALE, HISTORY_GAME_SCALE = 5.0, 20.0  # public history counters: this turn / this game
SCALES = {"cost": COST_SCALE, "atk": ATK_SCALE, "hp": HP_SCALE, "move_cost": MOVE_SCALE, "armor": ARMOR_SCALE,
          "coins": COIN_SCALE, "round": ROUND_SCALE, "base": BASE_SCALE, "hand": HAND_SCALE, "deck": DECK_SCALE,
          "counts": COUNT_SCALE, "turn": TURN_SCALE, "attacks": ATTACKS_SCALE, "amount": AMOUNT_SCALE,
          "burned": BURNED_SCALE, "deck_counts": 1.0, "pin_turns": PIN_SCALE, "history_turn": HISTORY_TURN_SCALE,
          "history_game": HISTORY_GAME_SCALE}
# UNCAPPED_BASE_HP (cards): an "uncapped" heal lets a base exceed base_hp up to this cap; the engine's choose
# previews use it, so it is part of the fingerprint
N_REVEALED = 20           # R: opponent-revealed card tokens
N_EFFECT_SLOTS = 3        # SPEC §1.1: at most 3 effects per card

# ---------------------------------------------------------------- effect vocabulary (SPEC §1.2, §1.2b)
# Fixed here (not read from cards.py) so the card-table columns never move when the loader grows; a
# card that uses a name outside this vocabulary is rejected when the encoder is built.
ENC_TRIGGERS = ("on_play", "on_deploy", "on_death", "on_attack", "on_damaged", "on_move", "on_kill",
                "start_of_turn", "end_of_turn", "on_attacked", "static")
ENC_SCOPES = ("self", "friendly", "enemy", "any")
ENC_ACTIONS = ("damage", "heal", "buff", "destroy", "draw", "gain_coins", "increase_max_coins", "add_trait",
               "remove_trait", "summon", "return_to_hand", "pin", "discard", "add_card", "retreat")
ENC_SELECTS = ("chosen", "random", "all", "self", "event", "prev", "adjacent")
ENC_SIDES = ("friendly", "enemy", "any")
ENC_KINDS = ("unit", "base", "player", "unit_or_base")
ENC_ZONES = ("board", "backline", "frontline")
ENC_TRAITS = ("defense", "armor", "blitz", "smokescreen", "fury", "ambush", "shock", "immune")

# ---------------------------------------------------------------- feature schemas
TOKEN_TYPES = ("global", "hand", "my_back", "front_mine", "front_opp", "opp_back", "my_base", "opp_base",
               "pending", "revealed", "deck")
# unit state, from the UnitView field of the same name, except `damaged` = hp < max_hp and the signed per-trait
# `lapse_<trait>` flags: +1 = granted until the end of this turn (UnitView.temp_traits bit), -1 = removed until
# the end of this turn (UnitView.temp_removed bit), 0 = no change lapses (a trait is never in both masks)
LAPSE_FEATURES = tuple(f"lapse_{t}" for t in ENC_TRAITS)
UNIT_FEATURES = ("atk", "hp", "max_hp", "armor", "defense", "blitz", "smokescreen", "fury", "pinned", "summoned",
                 "moved", "attacks", "can_move", "can_attack", "temp_atk", "temp_hp", "move_cost", "token",
                 "ambush", "shock", "immune", "ambush_ready", "pin_turns", "temp_move_cost", "damaged") + LAPSE_FEATURES
HAND_FEATURES = ("playable", "mulligan_marked", "mulligan_legal", "known_to_opp")
HINT_FEATURES = ("move_legal", "attack_ready", "attack_targetable", "incoming_atk", "choose_legal")
BASE_FEATURES = ("base_hp",)  # base tokens also use attack_targetable / incoming_atk / choose_legal
REVEALED_FEATURES = ("revealed", "known_in_hand", "graveyard", "discard", "on_board")
PENDING_FEATURES = ("effect0", "effect1", "effect2", "pending_amount", "pending_full", "pending_atk",
                    "pending_hp")
TOKEN_FEATURES = (tuple(f"type_{t}" for t in TOKEN_TYPES) + UNIT_FEATURES + HAND_FEATURES + HINT_FEATURES
                  + BASE_FEATURES + REVEALED_FEATURES + PENDING_FEATURES)
# history globals: Observation.my_history / opp_history = (ops, deployed, died) this turn, then this game
HISTORY_FEATURES = tuple(f"{who}_{event}_{window}" for who in ("my", "opp") for window in ("turn", "game")
                         for event in ("ops", "deployed", "died"))
GLOBAL_FEATURES = ("is_my_turn", "went_first", "round", "turn", "phase_mulligan", "phase_main", "phase_choice",
                   "my_coins", "opp_coins", "my_coin_bonus", "opp_coin_bonus", "my_base_hp", "opp_base_hp",
                   "my_hand_size", "opp_hand_size", "my_deck_size", "opp_deck_size", "my_burned", "opp_burned",
                   "front_mine", "front_empty", "front_opp", "n_legal", "base_damage_ready") + HISTORY_FEATURES
ATTACK_PREVIEW_FEATURES = ("kills", "attacker_dies", "dealt", "taken")
CHOOSE_PREVIEW_FEATURES = ("kills", "dealt", "healed", "other")
# static card table: CARD_FEATURES, then N_EFFECT_SLOTS blocks of EFFECT_FEATURES (prefix "e{k}_"), then one
# "tag_<name>" column per tag of the pool's vocabulary (layout()["tags"]), so every fixed column has a
# pool-independent index.
CARD_FEATURES = ("is_unit", "is_operation", "token", "cost", "atk", "hp", "move_cost", "troop", "fast",
                 "ranged") + ENC_TRAITS
EFFECT_FEATURES = (("present",) + tuple(f"trigger_{t}" for t in ENC_TRIGGERS)
                   + tuple(f"scope_{s}" for s in ENC_SCOPES) + tuple(f"action_{a}" for a in ENC_ACTIONS)
                   + tuple(f"select_{s}" for s in ENC_SELECTS) + tuple(f"side_{s}" for s in ENC_SIDES)
                   + tuple(f"kind_{k}" for k in ENC_KINDS) + tuple(f"zone_{z}" for z in ENC_ZONES)
                   + ("amount", "amount_full", "amount_is_expr", "uncapped", "buff_atk", "buff_hp", "buff_move_cost",
                      "duration_turn", "has_condition", "has_filter", "has_else", "has_event_filter", "repeat",
                      "count")
                   + tuple(f"trait_{t}" for t in ENC_TRAITS))
CARD_TABLE_FEATURES = CARD_FEATURES + tuple(f"e{k}_{n}" for k in range(N_EFFECT_SLOTS) for n in EFFECT_FEATURES)

F = len(TOKEN_FEATURES)
G = len(GLOBAL_FEATURES)
P = len(ATTACK_PREVIEW_FEATURES)
PC = len(CHOOSE_PREVIEW_FEATURES)
TF = {name: i for i, name in enumerate(TOKEN_FEATURES)}     # token feature column by name
GF = {name: i for i, name in enumerate(GLOBAL_FEATURES)}    # global index by name
CF = {name: i for i, name in enumerate(CARD_TABLE_FEATURES)}  # fixed card-table column by name
_EF = {name: i for i, name in enumerate(EFFECT_FEATURES)}
assert len(TF) == F and len(GF) == G and len(CF) == len(CARD_TABLE_FEATURES)

_UNIT_SCALES = {"atk": ATK_SCALE, "hp": HP_SCALE, "max_hp": HP_SCALE, "armor": ARMOR_SCALE, "attacks": ATTACKS_SCALE,
                "temp_atk": ATK_SCALE, "temp_hp": HP_SCALE, "move_cost": MOVE_SCALE, "pin_turns": PIN_SCALE,
                "temp_move_cost": MOVE_SCALE}
_NATURE_NAMES = {TROOP: "troop", FAST: "fast", RANGED: "ranged"}
FULL_AMOUNT = -1  # PendingView.amount of heal "full"
_LAPSE_BITS = np.array([TRAIT_BITS[t] for t in ENC_TRAITS], dtype=np.int64)  # bit of lapse_<trait>
# choose preview scales: kills is a flag, dealt / healed / other are hp-like amounts
_CHOOSE_PREVIEW_SCALE = np.array([1.0] + [1.0 / AMOUNT_SCALE] * 3, dtype=np.float32)


def _amount_parts(am) -> tuple:
    """(literal value, is_full, is_expr) of an AmountDef (None = no amount)."""
    if am is None:
        return 0, 0.0, 0.0
    if isinstance(am, int):
        return am, 0.0, 0.0
    kind = getattr(am, "kind", "literal")
    if kind == "literal":
        return am.value, 0.0, 0.0
    if kind == "full":
        return 0, 1.0, 0.0
    return 0, 0.0, 1.0


def _effect_row(card_id: str, eff) -> np.ndarray:
    """EFFECT_FEATURES of one parsed effect (static card data only)."""
    out = np.zeros(len(EFFECT_FEATURES), dtype=np.float32)
    tgt = eff.target

    def onehot(prefix: str, value: str, vocab: tuple, what: str) -> None:
        if value not in vocab:
            raise ValueError(f"card {card_id!r}: {what} {value!r} is not in the encoder vocabulary {vocab}; "
                             f"extend features.py (and bump ENCODER_VERSION)")
        out[_EF[f"{prefix}_{value}"]] = 1.0

    out[_EF["present"]] = 1.0
    onehot("trigger", eff.trigger, ENC_TRIGGERS, "trigger")
    onehot("scope", eff.scope, ENC_SCOPES, "scope")
    onehot("action", eff.action, ENC_ACTIONS, "action")
    onehot("select", tgt.select, ENC_SELECTS, "select")
    if tgt.side:
        onehot("side", tgt.side, ENC_SIDES, "side")
    onehot("kind", tgt.kind, ENC_KINDS, "kind")
    onehot("zone", tgt.zone, ENC_ZONES, "zone")
    value, full, expr = _amount_parts(eff.amount)
    out[_EF["amount"]] = value / AMOUNT_SCALE
    out[_EF["amount_full"]] = full
    out[_EF["uncapped"]] = float(bool(getattr(eff, "uncapped", False)))
    buff = [getattr(eff, "atk", None), getattr(eff, "hp", None), getattr(eff, "move_cost", None)]
    for name, am, scale in zip(("buff_atk", "buff_hp", "buff_move_cost"), buff,
                               (ATK_SCALE, HP_SCALE, MOVE_SCALE)):
        v, _, e = _amount_parts(am)
        out[_EF[name]] = v / scale
        expr = max(expr, e)
    out[_EF["amount_is_expr"]] = expr
    out[_EF["duration_turn"]] = float(getattr(eff, "duration", "permanent") == "turn")
    out[_EF["has_condition"]] = float(bool(eff.condition))
    out[_EF["has_filter"]] = float(tgt.filter is not None or getattr(eff, "target_condition", None) is not None)
    out[_EF["has_else"]] = float(getattr(eff, "else_", None) is not None)
    out[_EF["has_event_filter"]] = float(getattr(eff, "event_filter", None) is not None)
    out[_EF["repeat"]] = getattr(eff, "repeat", 1) / COUNT_SCALE
    out[_EF["count"]] = (tgt.count / COUNT_SCALE) if tgt.select == "random" else 0.0
    for name in getattr(eff, "trait", ()) or ():
        onehot("trait", name, ENC_TRAITS, "trait")
    return out


def card_table(config: GameConfig) -> tuple:
    """(n_cards x S float32 table, tag vocabulary) of the static per-card features (SPEC §6)."""
    cards = config.cards.cards
    tags = sorted({t for c in cards for t in (getattr(c, "tags", ()) or ())})
    tag_col = {t: len(CARD_TABLE_FEATURES) + i for i, t in enumerate(tags)}
    table = np.zeros((len(cards), len(CARD_TABLE_FEATURES) + len(tags)), dtype=np.float32)
    n_slot = len(EFFECT_FEATURES)
    for c in cards:
        row = table[c.index]
        unit = c.is_unit
        row[CF["is_unit"]] = float(unit)
        row[CF["is_operation"]] = float(not unit)
        row[CF["token"]] = float(c.token)
        row[CF["cost"]] = c.cost / COST_SCALE
        if unit:
            row[CF["atk"]] = c.attack / ATK_SCALE
            row[CF["hp"]] = c.health / HP_SCALE
            row[CF["move_cost"]] = c.move_cost / MOVE_SCALE
            row[CF[_NATURE_NAMES[c.nature]]] = 1.0
            for name in ENC_TRAITS:
                if name == "armor":
                    row[CF["armor"]] = c.armor / ARMOR_SCALE
                else:
                    row[CF[name]] = float(bool(getattr(c, name, False)))
        if len(c.effects) > N_EFFECT_SLOTS:
            raise ValueError(f"card {c.id!r}: more than {N_EFFECT_SLOTS} effects")
        for k, eff in enumerate(c.effects):
            start = len(CARD_FEATURES) + k * n_slot
            row[start:start + n_slot] = _effect_row(c.id, eff)
        for t in getattr(c, "tags", ()) or ():
            row[tag_col[t]] = 1.0
    return table, tags


def pool_fingerprint(config: GameConfig) -> str:
    """Hash of everything the encoding depends on: ENCODER_VERSION and the feature schemas (including the
    engine's Observation / UnitView / PendingView fields the encoder reads), the scales, the card definitions
    and card table, the fixed decks and the rule constants features are read against. Gameplay switches that
    do not change what a feature means (`mulligan`, the loop guard) are left out, so a checkpoint trained
    with the mulligan also loads for mulligan-free scenarios."""
    table, tags = card_table(config)
    blob = json.dumps({
        "version": ENCODER_VERSION,
        "schema": {"tokens": list(TOKEN_TYPES), "token_features": list(TOKEN_FEATURES),
                   "globals": list(GLOBAL_FEATURES), "card_table": list(CARD_TABLE_FEATURES),
                   "attack_preview": list(ATTACK_PREVIEW_FEATURES), "choose_preview": list(CHOOSE_PREVIEW_FEATURES),
                   "R": N_REVEALED, "effect_slots": N_EFFECT_SLOTS, "observation": list(Observation._fields),
                   "unit_view": list(UnitView._fields), "pending_view": list(PendingView._fields)},
        "scales": SCALES, "uncapped_base_hp": UNCAPPED_BASE_HP,
        "cards": [repr(c) for c in config.cards.cards],
        "card_table": hashlib.sha256(np.ascontiguousarray(table).tobytes()).hexdigest(), "tags": tags,
        "decks": [list(d) for d in config.decks],
        "rules": {"H": config.max_hand_size, "Z": config.zone_capacity, "base_hp": config.base_hp,
                  "max_rounds": config.max_rounds, "coin_cap": config.coin_cap,
                  "opening_hand": list(config.opening_hand), "deck_size": config.deck_size,
                  "max_copies": config.max_copies},
    }, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class EncodedParts(NamedTuple):
    globals: object         # (..., G)
    features: object        # (..., T, F)
    ids: object             # (..., T) int: card index + 1, 0 = none
    present: object         # (..., T) bool
    deck_counts: object     # (..., n_cards) remaining own deck contents (raw counts)
    attack_preview: object  # (..., 2Z, 2Z+1, 4) zero for illegal attacks
    choose_preview: object  # (..., 3Z+2, 4) zero for illegal / non-previewed choices


class ObservationEncoder:
    """Stage 3 token encoder (SPEC §6). See the module docstring and `layout()`."""

    def __init__(self, config: GameConfig):
        self.config = config
        pool = config.cards
        self.n_cards = n = len(pool)
        self.H, self.Z = H, Z = config.max_hand_size, config.zone_capacity
        self.R = R = N_REVEALED
        sp = self.space = ActionSpace(H, Z)
        self.n_actions = sp.n
        self.G, self.F, self.P, self.PC = G, F, P, PC
        # ---- token order (SPEC §6)
        self.tok_global = 0
        self.tok_hand = 1
        self.tok_my_back = self.tok_hand + H
        self.tok_front = self.tok_my_back + Z
        self.tok_opp_back = self.tok_front + Z
        self.tok_my_base = self.tok_opp_back + Z
        self.tok_opp_base = self.tok_my_base + 1
        self.tok_pending = self.tok_opp_base + 1
        self.tok_revealed = self.tok_pending + 1
        self.tok_deck = self.tok_revealed + R
        self.T = T = self.tok_deck + 1
        self.groups = {"global": (0, 1), "hand": (self.tok_hand, self.tok_hand + H),
                       "my_back": (self.tok_my_back, self.tok_my_back + Z),
                       "front": (self.tok_front, self.tok_front + Z),
                       "opp_back": (self.tok_opp_back, self.tok_opp_back + Z),
                       "bases": (self.tok_my_base, self.tok_opp_base + 1),
                       "my_base": (self.tok_my_base, self.tok_my_base + 1),
                       "opp_base": (self.tok_opp_base, self.tok_opp_base + 1),
                       "pending": (self.tok_pending, self.tok_pending + 1),
                       "revealed": (self.tok_revealed, self.tok_revealed + R),
                       "deck": (self.tok_deck, self.tok_deck + 1)}
        # action slots -> tokens
        self.attacker_tokens = np.arange(self.tok_my_back, self.tok_my_back + 2 * Z)  # my backline, frontline
        self.target_tokens = np.array([self.tok_opp_back + j for j in range(Z)] + [self.tok_front + j for j in range(Z)]
                                      + [self.tok_opp_base])
        self.choose_tokens = np.concatenate([self.target_tokens, np.arange(self.tok_my_back, self.tok_my_back + Z),
                                             [self.tok_my_base]])
        assert len(self.attacker_tokens) == sp.n_attackers and len(self.target_tokens) == sp.n_targets
        assert len(self.choose_tokens) == sp.n_choose
        # ---- flat offsets
        self.off_globals = 0
        self.off_features = G
        self.off_ids = self.off_features + T * F
        self.off_present = self.off_ids + T
        self.off_deck = self.off_present + T
        self.off_attack_preview = self.off_deck + n
        self.off_choose_preview = self.off_attack_preview + sp.n_attackers * sp.n_targets * P
        self.dim = self.off_choose_preview + sp.n_choose * PC
        # ---- static data
        self.card_table, self.tags = card_table(config)
        self.S = self.card_table.shape[1]
        self.fingerprint = pool_fingerprint(config)
        # UnitView field -> unit feature column (fields the engine does not expose yet stay 0)
        uv = UnitView._fields
        self._uv_card, self._uv_atk, self._uv_hp, self._uv_max_hp = (uv.index(k)
                                                                     for k in ("card", "atk", "hp", "max_hp"))
        pairs = [(TF[name], uv.index(name), _UNIT_SCALES.get(name, 1.0)) for name in UNIT_FEATURES
                 if name != "damaged" and name in uv]
        self._uv_dst = np.array([p[0] for p in pairs], dtype=np.intp)
        self._uv_src = np.array([p[1] for p in pairs], dtype=np.intp)
        self._uv_scale = np.array([1.0 / p[2] for p in pairs], dtype=np.float32)
        u0 = TF[UNIT_FEATURES[0]]
        self._uv_contig = bool(np.array_equal(self._uv_dst, np.arange(u0, u0 + len(pairs))))
        self._uv_lo, self._uv_hi = u0, u0 + len(pairs)
        self._uv_temp = np.array([uv.index("temp_traits"), uv.index("temp_removed")], dtype=np.intp)
        self._lapse_lo = TF[LAPSE_FEATURES[0]]
        self._lapse_hi = self._lapse_lo + len(LAPSE_FEATURES)
        assert TF[LAPSE_FEATURES[-1]] == self._lapse_hi - 1
        self._hist0 = GF[HISTORY_FEATURES[0]]
        assert GF[HISTORY_FEATURES[-1]] == G - 1  # the history globals close the global block
        self._hist_scale = np.array(([1.0 / HISTORY_TURN_SCALE] * 3 + [1.0 / HISTORY_GAME_SCALE] * 3) * 2,
                                    dtype=np.float32)
        # constant entries: type one-hot and presence of the always-present tokens
        const = []
        for tok, typ in ((self.tok_global, "global"), (self.tok_my_base, "my_base"), (self.tok_opp_base, "opp_base"),
                         (self.tok_deck, "deck")):
            const += [self.off_features + tok * F + TF[f"type_{typ}"], self.off_present + tok]
        self._const_idx = np.array(const, dtype=np.intp)
        self._inc_scale = np.array([1.0 / HP_SCALE] * (2 * Z) + [1.0 / BASE_SCALE], dtype=np.float32)
        self._unit_cache: dict = {}
        self._rev_cols = np.array([TF[k] for k in REVEALED_FEATURES[:4]], dtype=np.intp)

    # ------------------------------------------------------------------ layout
    def layout(self) -> dict:
        """Everything a network needs to interpret the flat encoding (plain Python types; stored in
        checkpoints). `card_table[c]` holds card index c, i.e. token id c + 1 (id 0 = no card)."""
        sp = self.space
        return {
            "version": ENCODER_VERSION, "fingerprint": self.fingerprint,
            "G": self.G, "T": self.T, "F": self.F, "S": self.S, "P": self.P, "PC": self.PC,
            "H": self.H, "Z": self.Z, "R": self.R, "n_cards": self.n_cards, "n_tags": len(self.tags),
            "dim": self.dim, "n_actions": self.n_actions,
            "offsets": {"globals": self.off_globals, "features": self.off_features, "ids": self.off_ids,
                        "present": self.off_present, "deck_counts": self.off_deck,
                        "attack_preview": self.off_attack_preview, "choose_preview": self.off_choose_preview},
            "groups": {k: list(v) for k, v in self.groups.items()},
            "token_types": list(TOKEN_TYPES),
            "attacker_tokens": self.attacker_tokens.tolist(), "attack_target_tokens": self.target_tokens.tolist(),
            "choose_tokens": self.choose_tokens.tolist(),
            "END_TURN": sp.END_TURN, "PLAY0": sp.PLAY0, "MOVE0": sp.MOVE0, "ATTACK0": sp.ATTACK0,
            "CHOOSE0": sp.CHOOSE0, "MULLIGAN0": sp.MULLIGAN0, "CONFIRM": sp.CONFIRM,
            "n_attackers": sp.n_attackers, "n_targets": sp.n_targets, "n_choose": sp.n_choose,
            "BASE_TARGET": sp.BASE_TARGET, "ENEMY_BASE_CHOICE": sp.ENEMY_BASE_CHOICE,
            "OWN_BASE_CHOICE": sp.OWN_BASE_CHOICE,
            "global_features": list(GLOBAL_FEATURES), "token_features": list(TOKEN_FEATURES),
            "card_table_features": list(CARD_TABLE_FEATURES) + [f"tag_{t}" for t in self.tags],
            "attack_preview_features": list(ATTACK_PREVIEW_FEATURES),
            "choose_preview_features": list(CHOOSE_PREVIEW_FEATURES),
            "card_table": self.card_table.tolist(), "tags": list(self.tags),
            "card_ids": [c.id for c in self.config.cards.cards], "scales": dict(SCALES),
            "deck_counts_scale": 1.0,
        }

    # ------------------------------------------------------------------ public API
    def encode(self, obs: Observation, mask: Optional[np.ndarray] = None) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        self.encode_into(obs, out, mask)
        return out

    def encode_batch(self, observations, out: Optional[np.ndarray] = None, masks=None) -> np.ndarray:
        if out is None:
            out = np.zeros((len(observations), self.dim), dtype=np.float32)
        else:
            out[:len(observations)] = 0.0
        for i, obs in enumerate(observations):
            self.encode_into(obs, out[i], None if masks is None else masks[i])
        return out

    def split(self, x) -> EncodedParts:
        """Named parts of flat encodings (numpy or torch, any leading batch shape)."""
        lead = tuple(x.shape[:-1])
        sp = self.space
        g = x[..., :self.off_features]
        feats = x[..., self.off_features:self.off_ids].reshape(*lead, self.T, self.F)
        ids = x[..., self.off_ids:self.off_present]
        present = x[..., self.off_present:self.off_deck] > 0.5
        deck = x[..., self.off_deck:self.off_attack_preview]
        ap = x[..., self.off_attack_preview:self.off_choose_preview].reshape(*lead, sp.n_attackers, sp.n_targets,
                                                                            self.P)
        cp = x[..., self.off_choose_preview:self.dim].reshape(*lead, sp.n_choose, self.PC)
        ids = ids.long() if hasattr(ids, "long") else ids.astype(np.int64)
        return EncodedParts(g, feats, ids, present, deck, ap, cp)

    def _check_mask(self, obs: Observation, mask) -> None:
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != (self.n_actions,):
            raise TypeError(f"mask must be the observer's legal mask: a bool ndarray of shape ({self.n_actions},), "
                            f"got {type(mask).__name__} {getattr(mask, 'dtype', '')} {getattr(mask, 'shape', '')}")
        if not obs.is_my_turn and mask.any():
            raise ValueError("a legal mask was given for an observation that is not the observer's turn "
                             "(the mover's mask reflects their hidden hand)")

    def _unit_slots(self, nb: int, nf: int, no: int, mine_front: bool) -> tuple:
        """(token indices, type columns) of the unit tokens for zone sizes (nb, nf, no)."""
        key = (nb, nf, no, mine_front)
        hit = self._unit_cache.get(key)
        if hit is None:
            toks = ([self.tok_my_back + j for j in range(nb)] + [self.tok_front + j for j in range(nf)]
                    + [self.tok_opp_back + j for j in range(no)])
            front_type = TF["type_front_mine"] if mine_front else TF["type_front_opp"]
            types = [TF["type_my_back"]] * nb + [front_type] * nf + [TF["type_opp_back"]] * no
            hit = self._unit_cache[key] = (np.array(toks, dtype=np.intp), np.array(types, dtype=np.intp))
        return hit

    def encode_into(self, obs: Observation, row: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        """Write the encoding of `obs` into a zeroed float32 row of length `dim`. `mask` is the observer's
        legal mask for their own decision (bool ndarray of shape (n_actions,)) or None."""
        if mask is not None:
            self._check_mask(obs, mask)
        if row.shape != (self.dim,):
            raise ValueError(f"row must have shape ({self.dim},), got {row.shape}")
        if not row.flags.c_contiguous or row.dtype != np.float32:
            tmp = np.zeros(self.dim, dtype=np.float32)
            self.encode_into(obs, tmp, mask)
            row[:] = tmp
            return
        H, Z, T, sp = self.H, self.Z, self.T, self.space
        oF, oI, oP = self.off_features, self.off_ids, self.off_present
        feats = row[oF:oI].reshape(T, F)
        ids = row[oI:oP]
        present = row[oP:self.off_deck]
        row[self._const_idx] = 1.0
        fo = obs.front_owner
        mine_front = fo > 0

        # ---- hand
        hand = obs.hand
        nh = len(hand)
        h0 = self.tok_hand
        if nh:
            feats[h0:h0 + nh, TF["type_hand"]] = 1.0
            ids[h0:h0 + nh] = hand
            ids[h0:h0 + nh] += 1.0
            present[h0:h0 + nh] = 1.0
            marks = obs.mulligan_marks
            if marks:
                feats[h0:h0 + len(marks), TF["mulligan_marked"]] = marks
            known = obs.my_known_hand
            if known and any(known):
                seen: dict = {}
                col = TF["known_to_opp"]
                for i, c in enumerate(hand):
                    k = seen.get(c, 0)
                    if k < known[c]:
                        feats[h0 + i, col] = 1.0
                    seen[c] = k + 1

        # ---- units
        mb, fr, ob = obs.my_backline, obs.frontline, obs.opp_backline
        nb, nf, no = len(mb), len(fr), len(ob)
        units = mb + fr + ob
        A = None
        if units:
            toks, types = self._unit_slots(nb, nf, no, mine_front)
            A = np.array(units, dtype=np.float32)
            vals = A[:, self._uv_src] * self._uv_scale
            if self._uv_contig:
                feats[toks, self._uv_lo:self._uv_hi] = vals
            else:
                feats[toks[:, None], self._uv_dst[None, :]] = vals
            feats[toks, TF["damaged"]] = A[:, self._uv_hp] < A[:, self._uv_max_hp]
            temp = A[:, self._uv_temp]
            if temp.any():  # signed "lapses this turn" flags per trait
                bits = temp.astype(np.int64)
                feats[toks, self._lapse_lo:self._lapse_hi] = (
                    ((bits[:, :1] & _LAPSE_BITS) != 0).astype(np.float32)
                    - ((bits[:, 1:] & _LAPSE_BITS) != 0).astype(np.float32))
            feats[toks, types] = 1.0
            ids[toks] = A[:, self._uv_card] + 1.0
            present[toks] = 1.0

        # ---- bases
        feats[self.tok_my_base, TF["base_hp"]] = obs.my_base_hp / BASE_SCALE
        feats[self.tok_opp_base, TF["base_hp"]] = obs.opp_base_hp / BASE_SCALE

        # ---- pending choice (public)
        pend = obs.pending
        if pend is not None:
            tp = self.tok_pending
            r = feats[tp]
            r[TF["type_pending"]] = 1.0
            r[TF["effect0"] + pend.effect] = 1.0
            amount = pend.amount
            if pend.action == "heal" and amount == FULL_AMOUNT:
                r[TF["pending_full"]] = 1.0
            else:
                r[TF["pending_amount"]] = amount / AMOUNT_SCALE
            r[TF["pending_atk"]] = getattr(pend, "atk", 0) / ATK_SCALE
            r[TF["pending_hp"]] = getattr(pend, "hp", 0) / HP_SCALE
            ids[tp] = pend.card + 1
            present[tp] = 1.0

        # ---- opponent-revealed cards: unique card indices with any public presence, ascending, first R
        rv, kh, gy, dc = obs.opp_revealed, obs.opp_known_hand, obs.opp_graveyard, obs.opp_discard
        if rv:
            C = np.array((rv, kh, gy, dc), dtype=np.float32)
            sel = np.flatnonzero(C.any(axis=0))[:self.R]
            k = len(sel)
            if k:
                r0 = self.tok_revealed
                rt = slice(r0, r0 + k)
                feats[rt, TF["type_revealed"]] = 1.0
                feats[r0:r0 + k, self._rev_cols[0]:self._rev_cols[-1] + 1] = C[:, sel].T * (1.0 / COUNT_SCALE)
                opp_units = ob + (() if mine_front else fr)
                if opp_units:
                    on_board = np.bincount([u.card for u in opp_units], minlength=self.n_cards)[sel]
                    feats[rt, TF["on_board"]] = on_board * (1.0 / COUNT_SCALE)
                ids[rt] = sel + 1
                present[rt] = 1.0

        # ---- own deck: remaining contents (raw counts; the network averages them over the deck size)
        if obs.my_deck_counts:
            row[self.off_deck:self.off_attack_preview] = obs.my_deck_counts

        # ---- mask-derived hints and previews (own decision only)
        n_legal = 0
        base_ready = 0.0
        if mask is not None:
            n_legal = int(np.count_nonzero(mask))
            if n_legal:
                base_ready = self._write_hints(obs, row, feats, mask, A, nb, nf, mine_front)
                feats *= present[:, None]  # hints never mark an absent token (only a malformed mask could)

        # ---- globals
        ph = obs.phase
        h0 = self._hist0
        row[:h0] = (obs.is_my_turn, obs.went_first, obs.round / ROUND_SCALE, obs.turn / TURN_SCALE,
                    ph == 0, ph == 1, ph == 2, obs.my_coins / COIN_SCALE, obs.opp_coins / COIN_SCALE,
                    obs.my_coin_bonus / COIN_SCALE, obs.opp_coin_bonus / COIN_SCALE,
                    obs.my_base_hp / BASE_SCALE, obs.opp_base_hp / BASE_SCALE, nh / HAND_SCALE,
                    obs.opp_hand_size / HAND_SCALE, obs.my_deck_size / DECK_SCALE, obs.opp_deck_size / DECK_SCALE,
                    obs.my_burned / BURNED_SCALE, obs.opp_burned / BURNED_SCALE, fo > 0, fo == 0, fo < 0,
                    n_legal / self.n_actions, base_ready)
        mh, oh = obs.my_history, obs.opp_history  # () (the default) encodes as zeros
        if mh or oh:
            n = len(HISTORY_FEATURES) // 2
            if mh:
                row[h0:h0 + n] = mh
            if oh:
                row[h0 + n:G] = oh
            row[h0:G] *= self._hist_scale

    def _write_hints(self, obs, row, feats, mask, A, nb: int, nf: int, mine_front: bool) -> float:
        """Mask-derived token hints, attack and choose previews. Returns base_damage_ready."""
        H, Z, sp = self.H, self.Z, self.space
        h0, b0 = self.tok_hand, self.tok_my_back
        feats[h0:h0 + H, TF["playable"]] = mask[sp.PLAY0:sp.PLAY0 + H]
        feats[h0:h0 + H, TF["mulligan_legal"]] = mask[sp.MULLIGAN0:sp.MULLIGAN0 + H]
        feats[b0:b0 + Z, TF["move_legal"]] = mask[sp.MOVE0:sp.MOVE0 + Z]
        base_ready = 0.0
        attack = mask[sp.ATTACK0:sp.CHOOSE0].reshape(sp.n_attackers, sp.n_targets)
        if A is not None and attack.any():
            feats[self.attacker_tokens, TF["attack_ready"]] = attack.any(axis=1)
            feats[self.target_tokens, TF["attack_targetable"]] = attack.any(axis=0)
            my_atk = np.zeros(sp.n_attackers, dtype=np.float32)
            my_atk[:nb] = A[:nb, self._uv_atk]
            if mine_front:
                my_atk[Z:Z + nf] = A[nb:nb + nf, self._uv_atk]
            incoming = my_atk @ attack  # per target: combined attack that can legally reach it
            feats[self.target_tokens, TF["incoming_atk"]] = incoming * self._inc_scale
            base_ready = float(incoming[sp.BASE_TARGET]) / BASE_SCALE
            self._attack_previews(obs, row, attack, mine_front)
        choose = mask[sp.CHOOSE0:sp.CHOOSE0 + sp.n_choose]
        if choose.any():
            feats[self.choose_tokens, TF["choose_legal"]] = choose
            self._choose_previews(obs, row, choose)
        return base_ready

    def _attack_previews(self, obs, row, attack, mine_front: bool) -> None:
        """(kills, attacker_dies, dealt, taken) of every legal attack, from engine.combat_damage."""
        Z, sp = self.Z, self.space
        mb, fr, ob = obs.my_backline, obs.frontline, obs.opp_backline
        nb, nf, no = len(mb), len(fr), len(ob)
        base, n_t = sp.BASE_TARGET, sp.n_targets
        ap = row[self.off_attack_preview:self.off_choose_preview]
        opp_hp = obs.opp_base_hp
        for a, t in zip(*np.nonzero(attack)):
            a, t = int(a), int(t)
            if a < Z:
                att = mb[a] if a < nb else None
            else:
                att = fr[a - Z] if mine_front and a - Z < nf else None
            if att is None:
                raise ValueError(f"mask allows {sp.describe(sp.attack(a, t))} but attacker slot {a} is empty: "
                                 f"the mask does not belong to this observation")
            r = (a * n_t + t) * P
            if t == base:
                ap[r:r + P] = (att.atk >= opp_hp, 0.0, att.atk / HP_SCALE, 0.0)
                continue
            if t < Z:
                tgt = ob[t] if t < no else None
            else:
                tgt = fr[t - Z] if not mine_front and t - Z < nf else None
            if tgt is None:
                raise ValueError(f"mask allows {sp.describe(sp.attack(a, t))} but target slot {t} holds no enemy "
                                 f"unit: the mask does not belong to this observation")
            dealt, taken = combat_damage(att, tgt)
            ap[r:r + P] = (dealt >= tgt.hp, taken >= att.hp, dealt / HP_SCALE, taken / HP_SCALE)

    def _choose_previews(self, obs, row, choose) -> None:
        """Copy the engine's (kills, dealt, healed, other) previews (PendingView.previews, SPEC §5) of every legal
        CHOOSE; dealt / healed / other are scaled like amounts. No rule is re-derived here: the engine applies
        target_condition / else, immune and the resolved amounts."""
        Z = self.Z
        nb, nf, no = len(obs.my_backline), len(obs.frontline), len(obs.opp_backline)
        for t in np.flatnonzero(choose).tolist():  # the mask must name options that exist in the observation
            if t < Z:
                ok = t < no
            elif t < 2 * Z:
                ok = t - Z < nf
            elif t == 2 * Z or t == 3 * Z + 1:
                ok = True
            else:
                ok = t - 2 * Z - 1 < nb
            if not ok:
                raise ValueError(f"mask allows CHOOSE({self.space.choice_slot_name(t)}) but that slot is empty: "
                                 f"the mask does not belong to this observation")
        pend = obs.pending
        previews = getattr(pend, "previews", ()) if pend is not None else ()
        if not previews:
            return
        if len(previews) != self.space.n_choose:
            raise ValueError(f"PendingView.previews has {len(previews)} slots, expected {self.space.n_choose}")
        cp = row[self.off_choose_preview:self.dim].reshape(self.space.n_choose, PC)
        cp[choose] = np.asarray(previews, dtype=np.float32)[choose] * _CHOOSE_PREVIEW_SCALE
