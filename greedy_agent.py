"""Rule-based greedy baseline (SPEC.md section 6). One action per call; re-evaluated after every step."""
from __future__ import annotations

from typing import Optional, Sequence

from ..actions import ActionSpace
from ..cards import GameConfig, load_ruleset
from ..engine import Observation


class GreedyAgent:
    """Priorities: play the costliest card > best favorable trade > push > hit the base > end turn.

    1. Play the highest-cost affordable card (ties: higher atk+hp, then lower hand slot).
    2. Favorable trade: the attacker kills the target and survives, or the target costs more.
       Maximize gain = target cost - (attacker cost if the attacker dies else 0);
       ties: higher target cost, then lower action index.
    3. Move the ready backline unit with the highest attack (ties: lower slot).
    4. A ready frontline unit attacks the enemy base (lowest slot first).
    5. END_TURN.
    """

    name = "greedy"

    def __init__(self, config: Optional[GameConfig] = None, seed: Optional[int] = None):
        self.config = config if config is not None else load_ruleset()
        cards = self.config.cards.cards
        self.cost = tuple(c.cost for c in cards)
        self.stats = tuple(c.attack + c.health for c in cards)
        sp = ActionSpace(self.config.max_hand_size, self.config.zone_capacity)
        self.action_space = sp
        self.play0, self.move0, self.base0 = sp.PLAY0, sp.MOVE0, sp.BASE0
        self.front0, self.back0 = sp.FRONT0, sp.BACK0
        # Per-action slot params (a = acting slot / hand slot, b = target slot), decoded once.
        self.arg_a = tuple(act.a for act in sp.actions)
        self.arg_b = tuple(act.b for act in sp.actions)

    def reset(self, seed: Optional[int] = None) -> None:
        """Deterministic agent: nothing to reset."""

    def act(self, obs: Observation, legal_actions: Sequence[int]) -> int:
        cost, stats, arg_a, arg_b = self.cost, self.stats, self.arg_a, self.arg_b
        play0, move0, base0, front0, back0 = self.play0, self.move0, self.base0, self.front0, self.back0

        best_play, play_key = -1, None
        best_trade, trade_key = -1, None
        best_move, move_key = -1, None
        best_base = -1
        for a in legal_actions:
            if a < play0:
                continue  # END_TURN
            if a < move0:
                card = obs.hand[a - play0]
                key = (cost[card], stats[card], -a)
                if play_key is None or key > play_key:
                    best_play, play_key = a, key
            elif best_play >= 0:
                continue  # a card will be played; the rest only matters without one
            elif a < base0:
                key = (obs.my_backline[a - move0].atk, -a)
                if move_key is None or key > move_key:
                    best_move, move_key = a, key
            elif a < front0:
                if best_base < 0 or a < best_base:
                    best_base = a
            else:
                if a < back0:
                    attacker, target = obs.frontline[arg_a[a]], obs.opp_backline[arg_b[a]]
                else:
                    attacker, target = obs.my_backline[arg_a[a]], obs.frontline[arg_b[a]]
                if attacker.atk < target.hp:
                    continue  # no kill
                t_cost = cost[target.card]
                if target.atk < attacker.hp:
                    gain = t_cost  # attacker survives
                else:
                    a_cost = cost[attacker.card]
                    if t_cost <= a_cost:
                        continue  # even or losing trade
                    gain = t_cost - a_cost
                key = (gain, t_cost, -a)
                if trade_key is None or key > trade_key:
                    best_trade, trade_key = a, key

        if best_play >= 0:
            return best_play
        if best_trade >= 0:
            return best_trade
        if best_move >= 0:
            return best_move
        if best_base >= 0:
            return best_base
        return self.action_space.END_TURN
