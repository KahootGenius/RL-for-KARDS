"""Observation -> fixed-size float vector for neural agents (egocentric, uses only Observation)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .actions import ActionSpace
from .cards import GameConfig
from .engine import Observation


class ObservationEncoder:
    """Layout: globals | hand slots | hand card counts | my backline | frontline | opp backline.

    Globals    = [is_my_turn, went_first, seat0, seat1, round, coins x2, base hp x2, hand sizes x2,
                  deck sizes x2, front owner one-hot x3, zone fill x3, board atk/hp totals x4]
    Hand slot  = [present, cost, atk, hp, playable, one-hot card]
    Unit slot  = [present, atk, hp, ready, one-hot card]

    The seat is visible to the observer and fixes which deck each side plays (SPEC §1).
    `playable` is read from the engine's legal-action mask (pass `legal`), so no rule is
    re-implemented here; it is 0 when no mask is given (e.g. when it is not the observer's turn).
    """

    N_GLOBAL = 23

    def __init__(self, config: GameConfig):
        self.config = config
        cards = config.cards
        self.n_cards = len(cards)
        self.H = config.max_hand_size
        self.Z = config.zone_capacity
        self.hand_width = 5 + self.n_cards
        self.unit_width = 4 + self.n_cards
        self.off_hand = self.N_GLOBAL
        self.off_counts = self.off_hand + self.H * self.hand_width
        self.off_my_back = self.off_counts + self.n_cards
        self.off_front = self.off_my_back + self.Z * self.unit_width
        self.off_opp_back = self.off_front + self.Z * self.unit_width
        self.dim = self.off_opp_back + self.Z * self.unit_width
        self.play0 = ActionSpace(config.max_hand_size, config.zone_capacity).PLAY0

        self.inv_cost = 1.0 / max(1, cards.max_cost)
        self.inv_atk = 1.0 / max(1, cards.max_attack)
        self.inv_hp = 1.0 / max(1, cards.max_health)
        self.inv_base = 1.0 / config.base_hp
        self.inv_rounds = 1.0 / config.max_rounds
        self.inv_deck = 1.0 / config.deck_size
        self.inv_H = 1.0 / self.H
        self.inv_Z = 1.0 / self.Z
        self.card_atk = [c.attack * self.inv_atk for c in cards.cards]
        self.card_hp = [c.health * self.inv_hp for c in cards.cards]
        self.card_cost = [c.cost * self.inv_cost for c in cards.cards]

    def encode(self, obs: Observation, legal: Optional[Sequence[bool]] = None) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        self.encode_into(obs, out, legal)
        return out

    def encode_batch(self, observations: Sequence[Observation], out: Optional[np.ndarray] = None,
                     legals: Optional[Sequence[Sequence[bool]]] = None) -> np.ndarray:
        if out is None:
            out = np.zeros((len(observations), self.dim), dtype=np.float32)
        else:
            out[:] = 0.0
        for i, obs in enumerate(observations):
            self.encode_into(obs, out[i], None if legals is None else legals[i])
        return out

    def encode_into(self, obs: Observation, row: np.ndarray, legal: Optional[Sequence[bool]] = None) -> None:
        """Write the encoding into a zeroed row. `legal` is the observer's legal-action mask."""
        inv_atk, inv_hp, Z = self.inv_atk, self.inv_hp, self.Z
        my_atk = sum(u.atk for u in obs.my_backline)
        my_hp = sum(u.hp for u in obs.my_backline)
        opp_atk = sum(u.atk for u in obs.opp_backline)
        opp_hp = sum(u.hp for u in obs.opp_backline)
        front_atk = sum(u.atk for u in obs.frontline)
        front_hp = sum(u.hp for u in obs.frontline)
        if obs.front_owner > 0:
            my_atk += front_atk
            my_hp += front_hp
        elif obs.front_owner < 0:
            opp_atk += front_atk
            opp_hp += front_hp

        g = (
            float(obs.is_my_turn), float(obs.went_first), float(obs.player == 0), float(obs.player == 1),
            obs.round * self.inv_rounds,
            obs.my_coins * 0.1, obs.opp_coins * 0.1,
            obs.my_base_hp * self.inv_base, obs.opp_base_hp * self.inv_base,
            len(obs.hand) * self.inv_H, obs.opp_hand_size * self.inv_H,
            obs.my_deck_size * self.inv_deck, obs.opp_deck_size * self.inv_deck,
            float(obs.front_owner > 0), float(obs.front_owner == 0), float(obs.front_owner < 0),
            len(obs.my_backline) * self.inv_Z, len(obs.frontline) * self.inv_Z, len(obs.opp_backline) * self.inv_Z,
            my_atk * inv_atk * 0.2, my_hp * inv_hp * 0.2, opp_atk * inv_atk * 0.2, opp_hp * inv_hp * 0.2,
        )
        row[:self.N_GLOBAL] = g

        play0 = self.play0
        hw, off = self.hand_width, self.off_hand
        counts = self.off_counts
        for i, card in enumerate(obs.hand):
            b = off + i * hw
            row[b] = 1.0
            row[b + 1] = self.card_cost[card]
            row[b + 2] = self.card_atk[card]
            row[b + 3] = self.card_hp[card]
            if legal is not None and legal[play0 + i]:
                row[b + 4] = 1.0
            row[b + 5 + card] = 1.0
            row[counts + card] += 0.25

        uw = self.unit_width
        for units, off in ((obs.my_backline, self.off_my_back), (obs.frontline, self.off_front),
                           (obs.opp_backline, self.off_opp_back)):
            for j, u in enumerate(units):
                b = off + j * uw
                row[b] = 1.0
                row[b + 1] = u.atk * inv_atk
                row[b + 2] = u.hp * inv_hp
                row[b + 3] = 1.0 if u.ready else 0.0
                row[b + 4 + u.card] = 1.0
