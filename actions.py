"""Fixed-size action space. Indices are stable for a given (max_hand_size, zone_capacity)."""
from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple


class ActionKind(IntEnum):
    END_TURN = 0
    PLAY = 1          # a = hand slot
    MOVE = 2          # a = own backline slot (advance to frontline)
    ATTACK_BASE = 3   # a = frontline slot
    FRONT_ATTACK = 4  # a = frontline slot, b = enemy backline slot
    BACK_ATTACK = 5   # a = own backline slot, b = frontline slot


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
        self.END_TURN = 0
        self.PLAY0 = 1
        self.MOVE0 = self.PLAY0 + H
        self.BASE0 = self.MOVE0 + Z
        self.FRONT0 = self.BASE0 + Z
        self.BACK0 = self.FRONT0 + Z * Z
        self.n = self.BACK0 + Z * Z

        actions = [Action(ActionKind.END_TURN)]
        actions += [Action(ActionKind.PLAY, i) for i in range(H)]
        actions += [Action(ActionKind.MOVE, j) for j in range(Z)]
        actions += [Action(ActionKind.ATTACK_BASE, j) for j in range(Z)]
        actions += [Action(ActionKind.FRONT_ATTACK, j, k) for j in range(Z) for k in range(Z)]
        actions += [Action(ActionKind.BACK_ATTACK, j, k) for j in range(Z) for k in range(Z)]
        assert len(actions) == self.n
        self.actions = tuple(actions)
        self._index = {a: i for i, a in enumerate(actions)}

    def __len__(self) -> int:
        return self.n

    def decode(self, index: int) -> Action:
        return self.actions[index]

    def encode(self, kind: ActionKind, a: int = -1, b: int = -1) -> int:
        return self._index[Action(ActionKind(kind), a, b)]

    def describe(self, index: int) -> str:
        return str(self.actions[index])
