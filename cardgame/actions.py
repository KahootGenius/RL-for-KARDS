"""Fixed-size action space. Indices are stable for a given (max_hand_size, zone_capacity).

Layout (SPEC §3): END_TURN | PLAY(hand slot) x H | MOVE(backline slot) x Z | ATTACK(a, t) x 2Z(2Z+1)
| CHOOSE(t) x (3Z+2) | MULLIGAN(hand slot) x H | CONFIRM. Stage 2 indices are unchanged.
Attacker slot a: own backline 0..Z-1, then frontline Z..2Z-1.
Target slot t: enemy backline 0..Z-1, frontline Z..2Z-1, enemy base 2Z.
Choice slot t: the 2Z+1 target slots (frontline either owner), then own backline 2Z+1..3Z, own base 3Z+1.
"""
from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple


class ActionKind(IntEnum):
    END_TURN = 0
    PLAY = 1      # a = hand slot
    MOVE = 2      # a = own backline slot (advance to the frontline)
    ATTACK = 3    # a = attacker slot, b = target slot
    CHOOSE = 4    # a = choice slot (pending effect target)
    MULLIGAN = 5  # a = opening-hand slot to replace
    CONFIRM = 6   # finish the mulligan


class Action(NamedTuple):
    kind: ActionKind
    a: int = -1
    b: int = -1

    def __str__(self) -> str:
        if self.a < 0:
            return self.kind.name
        if self.b < 0:
            return f"{self.kind.name}({self.a})"
        return f"{self.kind.name}({self.a},{self.b})"


class ActionSpace:
    def __init__(self, max_hand_size: int = 10, zone_capacity: int = 5):
        H, Z = max_hand_size, zone_capacity
        self.max_hand_size, self.zone_capacity = H, Z
        self.n_attackers = 2 * Z        # own backline + frontline slots
        self.n_targets = 2 * Z + 1      # enemy backline + frontline slots + base
        self.BASE_TARGET = 2 * Z
        self.n_choose = 3 * Z + 2       # target slots + own backline + own base
        self.ENEMY_BASE_CHOICE = 2 * Z
        self.OWN_BASE_CHOICE = 3 * Z + 1
        self.END_TURN = 0
        self.PLAY0 = 1
        self.MOVE0 = self.PLAY0 + H
        self.ATTACK0 = self.MOVE0 + Z
        self.CHOOSE0 = self.ATTACK0 + self.n_attackers * self.n_targets
        self.MULLIGAN0 = self.CHOOSE0 + self.n_choose
        self.CONFIRM = self.MULLIGAN0 + H
        self.n = self.CONFIRM + 1

        actions = [Action(ActionKind.END_TURN)]
        actions += [Action(ActionKind.PLAY, i) for i in range(H)]
        actions += [Action(ActionKind.MOVE, j) for j in range(Z)]
        actions += [Action(ActionKind.ATTACK, a, t) for a in range(self.n_attackers) for t in range(self.n_targets)]
        actions += [Action(ActionKind.CHOOSE, t) for t in range(self.n_choose)]
        actions += [Action(ActionKind.MULLIGAN, i) for i in range(H)]
        actions += [Action(ActionKind.CONFIRM)]
        assert len(actions) == self.n
        self.actions = tuple(actions)
        self._index = {a: i for i, a in enumerate(actions)}

    def __len__(self) -> int:
        return self.n

    def attack(self, a: int, t: int) -> int:
        return self.ATTACK0 + a * self.n_targets + t

    def choose(self, t: int) -> int:
        return self.CHOOSE0 + t

    def mulligan(self, i: int) -> int:
        return self.MULLIGAN0 + i

    def decode(self, index: int) -> Action:
        return self.actions[index]

    def encode(self, kind: ActionKind, a: int = -1, b: int = -1) -> int:
        return self._index[Action(ActionKind(kind), a, b)]

    def choice_slot_name(self, t: int) -> str:
        Z = self.zone_capacity
        if t < Z:
            return f"enemy_back{t}"
        if t < 2 * Z:
            return f"front{t - Z}"
        if t == self.ENEMY_BASE_CHOICE:
            return "enemy_base"
        if t < self.OWN_BASE_CHOICE:
            return f"own_back{t - 2 * Z - 1}"
        return "own_base"

    def describe(self, index: int) -> str:
        act = self.actions[index]
        if act.kind == ActionKind.CHOOSE:
            return f"CHOOSE({self.choice_slot_name(act.a)})"
        if act.kind != ActionKind.ATTACK:
            return str(act)
        Z = self.zone_capacity
        src = f"back{act.a}" if act.a < Z else f"front{act.a - Z}"
        dst = "base" if act.b == self.BASE_TARGET else (f"back{act.b}" if act.b < Z else f"front{act.b - Z}")
        return f"ATTACK({src}->{dst})"
