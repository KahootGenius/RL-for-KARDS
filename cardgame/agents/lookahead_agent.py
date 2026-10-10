"""One-step lookahead baseline (SPEC.md section 9).

Each decision samples a world consistent with what the player knows (`game.determinize(p, rng)`),
tries every legal action on a clone of that sample, resolves the player's own follow-up choices
greedily, and scores the result with a fixed material evaluation. All rules (combat, effects, random
outcomes, the opponent's next turn start) come from the engine; nothing is re-derived here.
"""
from __future__ import annotations

import random
from typing import Optional, Sequence

from ..actions import ActionKind, ActionSpace
from ..cards import GameConfig, load_ruleset
from ..engine import CHOICE, DRAW, MULLIGAN, Game, Observation

# Tie-break among equal values: ATTACK > PLAY > MOVE > CHOOSE > END_TURN, then the lower index.
KIND_PRIORITY = {ActionKind.ATTACK: 4, ActionKind.PLAY: 3, ActionKind.MOVE: 2, ActionKind.CHOOSE: 1,
                 ActionKind.END_TURN: 0, ActionKind.MULLIGAN: -1, ActionKind.CONFIRM: -1}
WIN_VALUE = 1000.0


class LookaheadAgent:
    """One-step lookahead over every legal action of a determinized copy of the game.

    1. `sim = game.determinize(p, rng)` (the real game's hidden state is never read).
    2. For each legal action a: step a on a clone of `sim`; while the result has a choice pending for
       p, step the CHOOSE with the best value, greedily (at most `max_choice_depth` = 3 deep).
    3. Value: a finished game is +-1000 (draw 0), otherwise
       V = 1.0 (base_p - base_o) + 0.5 (sum_p(atk + hp) - sum_o(atk + hp)) + 1.0 (hand_p - hand_o)
       over the units on the board and the hand sizes. END_TURN is scored as the current position
       (not stepped): the opponent's draw and turn-start effects follow whatever p does, so stepping
       it would bias the bot against ever ending its turn.
    4. The max V wins; ties go by kind ATTACK > PLAY > MOVE > CHOOSE > END_TURN, then the lower index.

    Mulligan: MULLIGAN every card with cost >= 5 (lowest slot first), then CONFIRM.

    Shortcuts that do not change the choice: with a single legal action it is returned without
    determinizing (the RNG is then not used), and the last candidate action (or choice option) is
    stepped on the parent copy itself instead of on a fresh clone. Values are computed in half-points
    (integers), so ties are exact.
    """

    name = "lookahead"
    needs_game = True
    MULLIGAN_MIN_COST = 5

    def __init__(self, config: Optional[GameConfig] = None, seed: Optional[int] = None,
                 max_choice_depth: int = 3):
        self.config = config if config is not None else load_ruleset()
        if type(max_choice_depth) is not int or max_choice_depth < 0:
            raise ValueError(f"max_choice_depth must be a non-negative integer, got {max_choice_depth!r}")
        self.max_choice_depth = max_choice_depth
        self.rng = random.Random(seed)
        sp = ActionSpace(self.config.max_hand_size, self.config.zone_capacity)
        self.action_space = sp
        self.cost = tuple(c.cost for c in self.config.cards.cards)
        self.priority = tuple(KIND_PRIORITY[act.kind] for act in sp.actions)
        self._win2 = int(2 * WIN_VALUE)

    def reset(self, seed: Optional[int] = None) -> None:
        """Seed the determinization RNG (None keeps the current stream)."""
        if seed is not None:
            self.rng.seed(seed)

    def act(self, obs: Observation, legal_actions: Sequence[int]) -> int:
        raise TypeError("LookaheadAgent simulates the game (needs_game = True): ask for its move with "
                        "cardgame.agents.choose_action(agent, game) or agent.act_game(game, player)")

    # ------------------------------------------------------------------ decisions
    def act_game(self, game: Game, player: int) -> int:
        legal = self._check(game, player)
        if game.phase == MULLIGAN:
            return self._mulligan(game.observe(player), legal)
        if len(legal) == 1:
            return legal[0]
        sim = game.determinize(player, self.rng)
        best, best_key = -1, None
        prio = self.priority
        end = self.action_space.END_TURN
        stay = self._value2(sim, player)  # END_TURN's value, read before `sim` is stepped below
        last = len(legal) - 1
        for k, a in enumerate(legal):
            v = stay if a == end else self._value_after(sim if k == last else sim.clone(), a, player)
            key = (v, prio[a], -a)
            if best_key is None or key > best_key:
                best, best_key = a, key
        return best

    def action_values(self, game: Game, player: int) -> dict:
        """{action: V} for every legal action, on one determinization (diagnostics and tests).

        Uses the agent's RNG like `act_game` (one determinization), but always determinizes, also with
        a single legal action. Not defined during the mulligan (the mulligan is a fixed rule)."""
        legal = self._check(game, player)
        if game.phase == MULLIGAN:
            raise ValueError("the mulligan is decided by a fixed rule, not by values")
        sim = game.determinize(player, self.rng)
        end = self.action_space.END_TURN
        return {a: (self._value2(sim, player) if a == end else self._value_after(sim.clone(), a, player)) / 2
                for a in legal}

    def evaluate(self, game: Game, player: int) -> float:
        """V of a position for `player` (SPEC 9 step 3). Only meant for simulated (determinized) games."""
        return self._value2(game, player) / 2

    # ------------------------------------------------------------------ internals
    def _check(self, game: Game, player: int) -> list:
        if game.done:
            raise ValueError("the game is over: there is no action to choose")
        if player != game.current_player():
            raise ValueError(f"player {player} is not the player to act ({game.current_player()})")
        return game.legal_actions()

    def _mulligan(self, obs: Observation, legal: Sequence[int]) -> int:
        legal_set = set(legal)
        m0, cost, min_cost = self.action_space.MULLIGAN0, self.cost, self.MULLIGAN_MIN_COST
        for i, c in enumerate(obs.hand):
            if cost[c] >= min_cost and m0 + i in legal_set:
                return m0 + i
        return self.action_space.CONFIRM

    def _value_after(self, g: Game, action: int, p: int) -> int:
        """Step `action` on `g` (a disposable copy), resolve p's pending choices greedily, return 2V."""
        g.step(action)
        for _ in range(self.max_choice_depth):
            if g.done or g.phase != CHOICE or g.current != p:
                break
            options = g.legal_actions()
            best_g, best_key = None, None
            last = len(options) - 1
            for k, c in enumerate(options):
                h = g if k == last else g.clone()
                h.step(c)
                key = (self._value2(h, p), -c)
                if best_key is None or key > best_key:
                    best_g, best_key = h, key
            g = best_g
        return self._value2(g, p)

    def _value2(self, g: Game, p: int) -> int:
        """2V (integer half-points): terminal +-2000 (draw 0), else
        2 (base_p - base_o) + (sum_p(atk + hp) - sum_o(atk + hp)) + 2 (hand_p - hand_o)."""
        if g.done:
            w = g.winner()
            return 0 if w == DRAW else (self._win2 if w == p else -self._win2)
        o = 1 - p
        v = 2 * (g.base_hp[p] - g.base_hp[o] + len(g.hands[p]) - len(g.hands[o]))
        for u in g.backline[p]:
            v += u.atk + u.hp
        for u in g.backline[o]:
            v -= u.atk + u.hp
        fo = g.front_owner
        if fo is not None:
            s = 0
            for u in g.frontline:
                s += u.atk + u.hp
            v += s if fo == p else -s
        return v
