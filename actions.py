"""Fixed-size action space. Indices are stable for a given (max_hand_size, zone_capacity).

Layout (SPEC §3): END_TURN | PLAY(hand slot) x H | MOVE(backline slot) x Z | ATTACK(a, t) x 2Z(2Z+1).
Attacker slot a: own backline 0..Z-1, then frontline Z..2Z-1.
Target slot t: enemy backline 0..Z-1, frontline Z..2Z-1, enemy base 2Z.
"""
from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple


class ActionKind(IntEnum):
    END_TURN = 0
    PLAY = 1    # a = hand slot
    MOVE = 2    # a = own backline slot (advance to the frontline)
    ATTACK = 3  # a = attacker slot, b = target slot


class Action(NamedTuple):
    kind: ActionKind
    a: int = -1
    b: int = -1

    def __str__(self) -> str:
        if self.kind == ActionKind.END_TURN:
            return "END_TURN"
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
        self.END_TURN = 0
        self.PLAY0 = 1
        self.MOVE0 = self.PLAY0 + H
        self.ATTACK0 = self.MOVE0 + Z
        self.n = self.ATTACK0 + self.n_attackers * self.n_targets

        actions = [Action(ActionKind.END_TURN)]
        actions += [Action(ActionKind.PLAY, i) for i in range(H)]
        actions += [Action(ActionKind.MOVE, j) for j in range(Z)]
        actions += [Action(ActionKind.ATTACK, a, t) for a in range(self.n_attackers) for t in range(self.n_targets)]
        assert len(actions) == self.n
        self.actions = tuple(actions)
        self._index = {a: i for i, a in enumerate(actions)}

    def __len__(self) -> int:
        return self.n

    def attack(self, a: int, t: int) -> int:
        return self.ATTACK0 + a * self.n_targets + t

    def decode(self, index: int) -> Action:
        return self.actions[index]

    def encode(self, kind: ActionKind, a: int = -1, b: int = -1) -> int:
        return self._index[Action(ActionKind(kind), a, b)]

    def describe(self, index: int) -> str:
        act = self.actions[index]
        if act.kind != ActionKind.ATTACK:
            return str(act)
        Z = self.zone_capacity
        src = f"back{act.a}" if act.a < Z else f"front{act.a - Z}"
        dst = "base" if act.b == self.BASE_TARGET else (f"back{act.b}" if act.b < Z else f"front{act.b - Z}")
        return f"ATTACK({src}->{dst})"
