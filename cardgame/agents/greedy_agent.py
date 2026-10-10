"""Rule-based greedy baseline v2 (SPEC.md section 9), kept as a legacy diagnostic. One action per call;
re-evaluated after every step."""
from __future__ import annotations

from typing import Optional, Sequence

from ..actions import ActionSpace
from ..cards import FAST, RANGED, GameConfig, load_ruleset
from ..engine import MAIN, Observation, combat_damage


class GreedyAgent:
    """First rule that yields an action wins (value of a unit = its card cost; kills and survival
    come from `engine.combat_damage`, so greedy never re-implements the combat rules):

    0. Lethal: if the own units with a legal base attack have total atk >= enemy base HP, the
       highest-atk one hits the base (ties: lower action index).
    1. Fast advance: MOVE a fast unit that can still attack (higher atk, lower slot).
    2. Play the highest-cost affordable card (ties: higher atk+hp, then lower hand slot).
    3. Kill a Defense unit that shields a non-Defense unit in its zone; max by
       (attacker survives, target value, -attacker value, -action index).
    4. Favorable trade: kill where the attacker survives or the target is worth more; max by
       (gain, target value, -action index), gain = target value - attacker value if it dies.
    5. Ranged units (higher atk, lower slot first): best unit target with damage > 0 by
       (kills, target value, damage, -action index), else the base.
    6. Advance: MOVE a troop/fast unit (higher atk, lower slot); ranged units stay back.
    7. A troop/fast frontline unit hits the base (lowest frontline slot).
    8. END_TURN.

    Stage 3 phases: in the mulligan it keeps its hand (CONFIRM); at a pending choice it takes the
    first legal option (lowest CHOOSE slot). Operations are plays like any other card (rule 2: by
    cost; their atk+hp is 0). Card effects are otherwise ignored.
    """

    name = "greedy"

    def __init__(self, config: Optional[GameConfig] = None, seed: Optional[int] = None):
        self.config = config if config is not None else load_ruleset()
        cards = self.config.cards.cards
        self.cost = tuple(c.cost for c in cards)
        self.stats = tuple(c.attack + c.health for c in cards)
        sp = ActionSpace(self.config.max_hand_size, self.config.zone_capacity)
        self.action_space = sp
        self.Z = sp.zone_capacity
        self.play0, self.move0, self.attack0, self.n = sp.PLAY0, sp.MOVE0, sp.ATTACK0, sp.n
        self.choose0, self.mulligan0, self.confirm = sp.CHOOSE0, sp.MULLIGAN0, sp.CONFIRM
        self.base_target = sp.BASE_TARGET
        # Per-action slot params (a = hand / backline / attacker slot, b = target slot), decoded once.
        self.arg_a = tuple(act.a for act in sp.actions)
        self.arg_b = tuple(act.b for act in sp.actions)

    def reset(self, seed: Optional[int] = None) -> None:
        """Deterministic agent: nothing to reset."""

    def act(self, obs: Observation, legal_actions: Sequence[int]) -> int:
        if obs.phase != MAIN:
            # mulligan: keep the hand; pending choice: the first legal option
            if self.confirm in legal_actions:
                return self.confirm
            return min(a for a in legal_actions if self.choose0 <= a < self.mulligan0)
        cost, Z, base_t = self.cost, self.Z, self.base_target
        play0, move0, attack0, attack_end = self.play0, self.move0, self.attack0, self.choose0
        arg_a, arg_b = self.arg_a, self.arg_b
        back, front, opp_back = obs.my_backline, obs.frontline, obs.opp_backline

        plays, moves = [], []
        base_hits = []   # (action, attacker slot, attacker)
        unit_hits = []   # (action, attacker slot, attacker, target slot, target)
        for act in legal_actions:
            if act < play0:
                continue  # END_TURN
            if act < move0:
                plays.append(act)
            elif act < attack0:
                moves.append(act)
            elif act < attack_end:  # attacks only (CHOOSE/MULLIGAN/CONFIRM are not MAIN actions)
                a, t = arg_a[act], arg_b[act]
                att = back[a] if a < Z else front[a - Z]
                if t == base_t:
                    base_hits.append((act, a, att))
                else:
                    unit_hits.append((act, a, att, t, opp_back[t] if t < Z else front[t - Z]))

        # 0. Lethal on the base.
        if base_hits and sum(h[2].atk for h in base_hits) >= obs.opp_base_hp:
            return max(base_hits, key=lambda h: (h[2].atk, -h[0]))[0]

        # 1. Fast advance: a fast unit that can still attack after moving.
        best, best_key = -1, None
        for act in moves:
            j = act - move0
            u = back[j]
            if u.nature == FAST and u.can_attack:
                key = (u.atk, -j)
                if best_key is None or key > best_key:
                    best, best_key = act, key
        if best >= 0:
            return best

        # 2. Play the costliest affordable card.
        if plays:
            hand, stats = obs.hand, self.stats
            return max(plays, key=lambda act: (cost[hand[act - play0]], stats[hand[act - play0]], -act))

        if unit_hits:
            # Combat outcomes of every legal unit attack, judged with the engine's own combat rule
            # (armor, no return damage for ranged attackers): (kills, attacker survives, damage).
            outcomes = []
            for act, a, att, t, tgt in unit_hits:
                dmg, ret = combat_damage(att, tgt)
                outcomes.append((dmg >= tgt.hp, ret < att.hp, dmg))

            # 3. Kill a Defense unit that shields a non-Defense unit.
            best, best_key = -1, None
            for (act, a, att, t, tgt), (kills, survives, _) in zip(unit_hits, outcomes):
                if kills and tgt.defense:
                    zone = opp_back if t < Z else front
                    if any(not u.defense for u in zone):
                        key = (survives, cost[tgt.card], -cost[att.card], -act)
                        if best_key is None or key > best_key:
                            best, best_key = act, key
            if best >= 0:
                return best

            # 4. Favorable trade.
            for (act, a, att, t, tgt), (kills, survives, _) in zip(unit_hits, outcomes):
                if kills:
                    t_val = cost[tgt.card]
                    if survives:
                        gain = t_val
                    else:
                        a_val = cost[att.card]
                        if t_val <= a_val:
                            continue
                        gain = t_val - a_val
                    key = (gain, t_val, -act)
                    if best_key is None or key > best_key:
                        best, best_key = act, key
            if best >= 0:
                return best
        else:
            outcomes = []

        # 5. Ranged units: best unit target, else the base; first ranged unit with an action wins.
        shooters = [(-u.atk, j) for j, u in enumerate(back) if u.nature == RANGED and u.can_attack]
        if obs.front_owner == 1:
            shooters += [(-u.atk, Z + k) for k, u in enumerate(front) if u.nature == RANGED and u.can_attack]
        if shooters:
            shooters.sort()
            for _, slot in shooters:
                best, best_key = -1, None
                for (act, a, att, t, tgt), (kills, _, dmg) in zip(unit_hits, outcomes):
                    if a == slot and dmg > 0:
                        key = (kills, cost[tgt.card], dmg, -act)
                        if best_key is None or key > best_key:
                            best, best_key = act, key
                if best >= 0:
                    return best
                for act, a, _ in base_hits:
                    if a == slot:
                        return act

        # 6. Advance a troop/fast unit.
        best, best_key = -1, None
        for act in moves:
            j = act - move0
            u = back[j]
            if u.nature != RANGED:
                key = (u.atk, -j)
                if best_key is None or key > best_key:
                    best, best_key = act, key
        if best >= 0:
            return best

        # 7. A troop/fast frontline unit hits the base.
        best = -1
        for act, a, att in base_hits:
            if a >= Z and att.nature != RANGED and (best < 0 or act < best):
                best = act
        if best >= 0:
            return best

        return self.action_space.END_TURN
