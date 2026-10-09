"""Observation -> feature-based entity encoding for neural agents (SPEC §6).

Every card (in hand or on the board) becomes one slot with a feature vector, a card id (for a
learned embedding) and a presence bit. Slots: hand (H) | my backline (Z) | frontline (Z) |
opponent backline (Z). The whole encoding is one flat float32 vector so the PPO buffers stay
simple: [globals (G) | features (E*F) | ids (E) | mask (E)]; `split()` recovers the parts.

Only the Observation and the engine's legal mask are used: nothing hidden can leak in, and every
"can it act now" hint is read from the mask instead of re-deriving a rule. Feature scales are
fixed constants (never derived from the card pool), so retuning cards does not silently rescale
the inputs of existing checkpoints; `pool_fingerprint()` detects such changes instead.
"""
from __future__ import annotations

import hashlib
import json
from typing import NamedTuple, Optional

import numpy as np

from .actions import ActionSpace
from .cards import FAST, RANGED, TROOP, GameConfig
from .engine import Observation, combat_damage

ENCODER_VERSION = 4  # bump whenever the layout or the meaning of a feature changes
# fixed scales (SPEC §6)
COST_SCALE, ATK_SCALE, HP_SCALE, MOVE_SCALE, ARMOR_SCALE = 8.0, 10.0, 10.0, 4.0, 4.0
COIN_SCALE, ROUND_SCALE, BASE_SCALE, HAND_SCALE, DECK_SCALE, PLAYED_SCALE = 10.0, 50.0, 20.0, 10.0, 40.0, 3.0

SLOT_FEATURES = ("present", "cost", "atk", "max_hp", "hp", "move_cost", "troop", "fast", "ranged", "defense",
                 "has_armor", "armor", "summoned", "moved", "attacked", "can_move", "can_attack",
                 "playable", "move_legal", "attack_ready", "targetable", "incoming_atk",
                 "zone_hand", "zone_my_back", "zone_front", "zone_opp_back", "mine")
GLOBAL_FEATURES = ("is_my_turn", "went_first", "round", "my_coins", "opp_coins", "my_base_hp", "opp_base_hp",
                   "hand_size", "opp_hand_size", "my_deck_size", "opp_deck_size", "front_mine", "front_empty",
                   "front_opp", "base_targetable", "n_legal", "base_damage_ready")
# ... followed by: own decklist summary (DECK_SUMMARY + cost histogram), my_played (n_cards), opp_played (n_cards)
DECK_SUMMARY = ("mean_cost", "mean_atk", "mean_hp", "mean_move_cost", "frac_troop", "frac_fast", "frac_ranged",
                "frac_defense", "frac_armor", "mean_armor")
COST_BINS = 8  # fraction of the deck at cost 1..8 (last bin: 8+)
# Attack previews: for every legal ATTACK(a, t), the outcome the engine's combat rule gives
# (engine.combat_damage), so the policy does not have to re-learn the combat arithmetic.
PREVIEW_FEATURES = ("kills", "attacker_dies", "damage_dealt", "damage_taken")
F = len(SLOT_FEATURES)
P = len(PREVIEW_FEATURES)
_FI = {name: i for i, name in enumerate(SLOT_FEATURES)}


def pool_fingerprint(config: GameConfig) -> str:
    """Hash of everything the encoding depends on: encoder version and schema, scales, card stats,
    decks and the rule constants features are normalised against."""
    cards = [(c.id, c.cost, c.attack, c.health, c.nature, c.defense, c.armor, c.move_cost)
             for c in config.cards.cards]
    decks = [list(d) for d in config.decks]
    blob = json.dumps({"version": ENCODER_VERSION, "cards": cards, "decks": decks, "H": config.max_hand_size,
                       "Z": config.zone_capacity, "rules": [config.base_hp, config.max_rounds, config.coin_cap,
                                                           list(config.opening_hand), config.deck_size],
                       "slots": list(SLOT_FEATURES), "globals": list(GLOBAL_FEATURES) + list(DECK_SUMMARY),
                       "preview": list(PREVIEW_FEATURES),
                       "scales": [COST_SCALE, ATK_SCALE, HP_SCALE, MOVE_SCALE, ARMOR_SCALE, COIN_SCALE,
                                  ROUND_SCALE, BASE_SCALE, HAND_SCALE, DECK_SCALE, PLAYED_SCALE, COST_BINS]},
                      sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class EncodedParts(NamedTuple):
    globals: object   # (..., G)
    features: object  # (..., E, F)
    preview: object   # (..., 2Z, 2Z+1, P) attack previews (zero for illegal attacks)
    ids: object       # (..., E) embedding ids: card index + 1, 0 = empty slot
    mask: object      # (..., E) bool, True = slot present


class ObservationEncoder:
    def __init__(self, config: GameConfig):
        self.config = config
        pool = config.cards
        self.n_cards = len(pool)
        self.n_decks = config.n_decks
        self.H, self.Z = config.max_hand_size, config.zone_capacity
        self.E = self.H + 3 * self.Z
        self.F = F
        self.n_base_globals = len(GLOBAL_FEATURES)
        self.n_deck_summary = len(DECK_SUMMARY) + COST_BINS
        self.G = self.n_base_globals + self.n_deck_summary + 2 * self.n_cards
        self.slot_hand, self.slot_my_back = 0, self.H
        self.slot_front, self.slot_opp_back = self.H + self.Z, self.H + 2 * self.Z
        sp = ActionSpace(config.max_hand_size, config.zone_capacity)
        self.space = sp
        self.P = P
        self.off_feat = self.G
        self.off_preview = self.G + self.E * F
        self.off_ids = self.off_preview + sp.n_attackers * sp.n_targets * P
        self.off_mask = self.off_ids + self.E
        self.dim = self.off_mask + self.E
        self.pad_id = 0  # embedding id = card index + 1 (stable when cards are appended)
        self.n_actions = sp.n
        self.fingerprint = pool_fingerprint(config)
        # static per-card prefix for hand slots: present .. armor (12 values)
        self.card_static = [[
            1.0, c.cost / COST_SCALE, c.attack / ATK_SCALE, c.health / HP_SCALE, c.health / HP_SCALE,
            c.move_cost / MOVE_SCALE, float(c.nature == TROOP), float(c.nature == FAST), float(c.nature == RANGED),
            float(c.defense), float(c.armor > 0), c.armor / ARMOR_SCALE] for c in pool.cards]
        self.card_cost = [c.cost / COST_SCALE for c in pool.cards]
        self.deck_summary = [self._summarize([pool.cards[i] for i in deck]) for deck in config.decks]

    @staticmethod
    def _summarize(cards) -> list:
        n = max(1, len(cards))
        hist = [0.0] * COST_BINS
        for c in cards:
            hist[min(max(c.cost, 1), COST_BINS) - 1] += 1.0 / n
        return [sum(c.cost for c in cards) / n / COST_SCALE, sum(c.attack for c in cards) / n / ATK_SCALE,
                sum(c.health for c in cards) / n / HP_SCALE, sum(c.move_cost for c in cards) / n / MOVE_SCALE,
                sum(c.nature == TROOP for c in cards) / n, sum(c.nature == FAST for c in cards) / n,
                sum(c.nature == RANGED for c in cards) / n, sum(c.defense for c in cards) / n,
                sum(c.armor > 0 for c in cards) / n, sum(c.armor for c in cards) / n / ARMOR_SCALE] + hist

    def layout(self) -> dict:
        """Everything a network needs to interpret the flat encoding (stored in checkpoints)."""
        return {"G": self.G, "E": self.E, "F": self.F, "P": self.P, "H": self.H, "Z": self.Z,
                "n_cards": self.n_cards, "dim": self.dim, "n_actions": self.n_actions, "fingerprint": self.fingerprint,
                "opp_base_hp_index": GLOBAL_FEATURES.index("opp_base_hp"),
                "slot_features": list(SLOT_FEATURES), "card_ids": [c.id for c in self.config.cards.cards]}

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
        """Recover (globals, features, ids, mask) from flat encodings (numpy or torch, any batch shape)."""
        lead = x.shape[:-1]
        g = x[..., :self.G]
        feats = x[..., self.off_feat:self.off_preview].reshape(*lead, self.E, F)
        sp = self.space
        preview = x[..., self.off_preview:self.off_ids].reshape(*lead, sp.n_attackers, sp.n_targets, P)
        ids = x[..., self.off_ids:self.off_mask]
        present = x[..., self.off_mask:] > 0.5
        ids = ids.long() if hasattr(ids, "long") else ids.astype(np.int64)
        return EncodedParts(g, feats, preview, ids, present)

    def _check_mask(self, mask) -> None:
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != (self.n_actions,):
            raise TypeError(f"mask must be the engine's legal mask: a bool ndarray of shape ({self.n_actions},), "
                            f"got {type(mask).__name__} {getattr(mask, 'dtype', '')} {getattr(mask, 'shape', '')}")

    def encode_into(self, obs: Observation, row: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        """Write the encoding into a zeroed row. `mask` is the observer's legal-action mask (own turn) or None."""
        cfg_H, Z, sp = self.H, self.Z, self.space
        fo = obs.front_owner
        if mask is not None:
            self._check_mask(mask)
            if not obs.is_my_turn and mask.any():
                raise ValueError("a legal mask was given for an observation that is not the observer's turn")
            attack = mask[sp.ATTACK0:].reshape(sp.n_attackers, sp.n_targets)
            attack_ready = attack.any(axis=1)
            targetable = attack.any(axis=0)
            n_legal = int(mask.sum())
            # attack of each of my attacker slots (backline, then frontline when I hold it)
            my_atk = np.zeros(sp.n_attackers, dtype=np.float32)
            for j, u in enumerate(obs.my_backline):
                my_atk[j] = u.atk
            if fo > 0:
                for j, u in enumerate(obs.frontline):
                    my_atk[Z + j] = u.atk
            incoming = my_atk @ attack            # per target: combined attack that can legally reach it
            self._write_previews(obs, row, attack)
        else:
            attack_ready = targetable = incoming = None
            n_legal = 0
        nb = self.n_base_globals
        row[:nb] = (float(obs.is_my_turn), float(obs.went_first), obs.round / ROUND_SCALE,
                    obs.my_coins / COIN_SCALE, obs.opp_coins / COIN_SCALE, obs.my_base_hp / BASE_SCALE,
                    obs.opp_base_hp / BASE_SCALE, len(obs.hand) / HAND_SCALE, obs.opp_hand_size / HAND_SCALE,
                    obs.my_deck_size / DECK_SCALE, obs.opp_deck_size / DECK_SCALE, float(fo > 0), float(fo == 0),
                    float(fo < 0), float(targetable[sp.BASE_TARGET]) if mask is not None else 0.0,
                    n_legal / self.n_actions,
                    float(incoming[sp.BASE_TARGET]) / BASE_SCALE if mask is not None else 0.0)
        row[nb:nb + self.n_deck_summary] = self.deck_summary[obs.my_deck]
        b = nb + self.n_deck_summary
        row[b:b + self.n_cards] = obs.my_played
        row[b + self.n_cards:b + 2 * self.n_cards] = obs.opp_played
        row[b:b + 2 * self.n_cards] /= PLAYED_SCALE

        off_feat, off_ids, off_mask = self.off_feat, self.off_ids, self.off_mask
        I = _FI
        play0, move0 = sp.PLAY0, sp.MOVE0
        for i, card in enumerate(obs.hand):
            slot = self.slot_hand + i
            r = off_feat + slot * F
            row[r:r + 12] = self.card_static[card]
            if mask is not None and mask[play0 + i]:
                row[r + I["playable"]] = 1.0
            row[r + I["zone_hand"]] = 1.0
            row[r + I["mine"]] = 1.0
            row[off_ids + slot] = card + 1
            row[off_mask + slot] = 1.0

        mine_front = fo > 0
        # (units, first slot, zone feature, mine, attacker slot offset or None, target slot offset or None)
        zones = ((obs.my_backline, self.slot_my_back, "zone_my_back", True, 0, None),
                 (obs.frontline, self.slot_front, "zone_front", mine_front,
                  Z if mine_front else None, None if mine_front else Z),
                 (obs.opp_backline, self.slot_opp_back, "zone_opp_back", False, None, 0))
        for units, first, zone_name, mine, att_off, tgt_off in zones:
            zone_i = I[zone_name]
            for j in range(Z):
                slot = first + j
                if j >= len(units):
                    continue  # empty slot: all zeros (id 0 = empty)
                u = units[j]
                r = off_feat + slot * F
                row[r:r + 12] = (
                    1.0, self.card_cost[u.card], u.atk / ATK_SCALE, u.max_hp / HP_SCALE, u.hp / HP_SCALE,
                    u.move_cost / MOVE_SCALE, float(u.nature == TROOP), float(u.nature == FAST),
                    float(u.nature == RANGED), float(u.defense), float(u.armor > 0), u.armor / ARMOR_SCALE)
                # flags are refreshed at their owner's END_TURN, so they are meaningful for both sides
                row[r + 12:r + 17] = (float(u.summoned), float(u.moved), float(u.attacked),
                                      float(u.can_move), float(u.can_attack))
                if mine:
                    row[r + I["mine"]] = 1.0
                if mask is not None:
                    if zone_name == "zone_my_back" and mask[move0 + j]:
                        row[r + I["move_legal"]] = 1.0
                    if att_off is not None and attack_ready[att_off + j]:
                        row[r + I["attack_ready"]] = 1.0
                    if tgt_off is not None and targetable[tgt_off + j]:
                        row[r + I["targetable"]] = 1.0
                        row[r + I["incoming_atk"]] = incoming[tgt_off + j] / HP_SCALE
                row[r + zone_i] = 1.0
                row[off_ids + slot] = u.card + 1
                row[off_mask + slot] = 1.0

    def _write_previews(self, obs: Observation, row: np.ndarray, attack: np.ndarray) -> None:
        """Outcome of every legal attack, from the engine's combat rule (no rule is re-derived here)."""
        Z, sp = self.Z, self.space
        mine_front = obs.front_owner > 0
        attackers = list(obs.my_backline) + [None] * (Z - len(obs.my_backline))
        attackers += list(obs.frontline) if mine_front else []
        targets = list(obs.opp_backline) + [None] * (Z - len(obs.opp_backline))
        targets += list(obs.frontline) if not mine_front else []
        base, n_t, off = sp.BASE_TARGET, sp.n_targets, self.off_preview
        for a, t in zip(*np.nonzero(attack)):
            att = attackers[a]
            r = off + (a * n_t + t) * P
            if t == base:
                row[r:r + P] = (float(att.atk >= obs.opp_base_hp), 0.0, att.atk / HP_SCALE, 0.0)
            else:
                tgt = targets[t]
                dealt, taken = combat_damage(att, tgt)
                row[r:r + P] = (float(dealt >= tgt.hp), float(taken >= att.hp), dealt / HP_SCALE, taken / HP_SCALE)
