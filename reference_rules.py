"""Independent reference model of the Stage 1 rules, written from SPEC.md §2-§3 only.

It shares no code with `cardgame.engine`: it reads the documented internal state (SPEC §4)
into plain tuples and re-derives legality and transitions from the spec tables, so the
tests can compare the engine against a second, deliberately naive implementation.
"""
from __future__ import annotations

from bisect import insort
from typing import NamedTuple, Optional

DRAW = -1  # SPEC §4: winner() == DRAW (-1) for a draw

# Action kinds, in SPEC §3 order.
END_TURN, PLAY, MOVE, ATTACK_BASE, FRONT_ATTACK, BACK_ATTACK = range(6)
KIND_NAMES = ("END_TURN", "PLAY", "MOVE", "ATTACK_BASE", "FRONT_ATTACK", "BACK_ATTACK")


class RefUnit(NamedTuple):
    card: int
    atk: int
    hp: int
    owner: int
    ready: bool


class RefState(NamedTuple):
    """Plain, immutable copy of the documented engine state (SPEC §4)."""
    first_player: int
    current: int
    round: int
    coins: tuple
    base_hp: tuple
    hands: tuple       # (tuple, tuple), sorted card indices
    decks: tuple       # (tuple, tuple), top = end
    burned: tuple
    backline: tuple    # (tuple[RefUnit], tuple[RefUnit])
    frontline: tuple   # tuple[RefUnit]
    front_owner: Optional[int]
    done: bool
    winner: Optional[int]


def extract_state(game) -> RefState:
    """Read the documented internal state of an engine `Game` into a `RefState`."""
    def units(zone):
        return tuple(RefUnit(u.card, u.atk, u.hp, u.owner, bool(u.ready)) for u in zone)

    return RefState(
        first_player=game.first_player, current=game.current, round=game.round,
        coins=tuple(game.coins), base_hp=tuple(game.base_hp),
        hands=tuple(tuple(h) for h in game.hands), decks=tuple(tuple(d) for d in game.decks),
        burned=tuple(game.burned), backline=tuple(units(z) for z in game.backline),
        frontline=units(game.frontline), front_owner=game.front_owner,
        done=bool(game.done), winner=game.winner(),
    )


# ---------------------------------------------------------------- action indices (SPEC §3)
def num_actions(H: int, Z: int) -> int:
    return 1 + H + 2 * Z + 2 * Z * Z


def action_index(kind: int, a: int, b: int, H: int, Z: int) -> int:
    """Index formula from the SPEC §3 table."""
    if kind == END_TURN:
        return 0
    if kind == PLAY:
        return 1 + a
    if kind == MOVE:
        return 1 + H + a
    if kind == ATTACK_BASE:
        return 1 + H + Z + a
    if kind == FRONT_ATTACK:
        return 1 + H + 2 * Z + a * Z + b
    if kind == BACK_ATTACK:
        return 1 + H + 2 * Z + Z * Z + a * Z + b
    raise ValueError(kind)


def decode_index(index: int, H: int, Z: int) -> tuple:
    """Inverse of `action_index`: (kind, a, b), unused params are -1."""
    if index == 0:
        return END_TURN, -1, -1
    i = index - 1
    if i < H:
        return PLAY, i, -1
    i -= H
    if i < Z:
        return MOVE, i, -1
    i -= Z
    if i < Z:
        return ATTACK_BASE, i, -1
    i -= Z
    if i < Z * Z:
        return FRONT_ATTACK, i // Z, i % Z
    i -= Z * Z
    if i < Z * Z:
        return BACK_ATTACK, i // Z, i % Z
    raise ValueError(index)


# ---------------------------------------------------------------- legality (SPEC §2 table)
def legal_from_state(s: RefState, config) -> list:
    """Sorted legal action indices for state `s`."""
    if s.done:
        return []
    H, Z = config.max_hand_size, config.zone_capacity
    p = s.current
    o = 1 - p
    my_back, front, fo = s.backline[p], s.frontline, s.front_owner
    legal = [action_index(END_TURN, -1, -1, H, Z)]
    if len(my_back) < Z:
        for i, card in enumerate(s.hands[p]):
            if config.cards[card].cost <= s.coins[p]:
                legal.append(action_index(PLAY, i, -1, H, Z))
    if fo in (None, p) and len(front) < Z:
        for j, u in enumerate(my_back):
            if u.ready:
                legal.append(action_index(MOVE, j, -1, H, Z))
    if fo == p:
        for j, u in enumerate(front):
            if u.ready:
                legal.append(action_index(ATTACK_BASE, j, -1, H, Z))
                for k in range(len(s.backline[o])):
                    legal.append(action_index(FRONT_ATTACK, j, k, H, Z))
    if fo == o:
        for j, u in enumerate(my_back):
            if u.ready:
                for k in range(len(front)):
                    legal.append(action_index(BACK_ATTACK, j, k, H, Z))
    return sorted(legal)


def reference_legal(game) -> list:
    """Sorted legal action indices of an engine `Game`, re-derived from SPEC.md."""
    return legal_from_state(extract_state(game), game.config)


# ---------------------------------------------------------------- transitions (SPEC §2)
def reference_step(s: RefState, action: int, config) -> RefState:
    """The state SPEC.md says follows `s` after `action` (which must be legal)."""
    H, Z = config.max_hand_size, config.zone_capacity
    if action not in legal_from_state(s, config):
        raise ValueError(f"reference: action {action} is illegal")
    kind, a, b = decode_index(action, H, Z)
    p = s.current
    o = 1 - p
    st = {
        "current": s.current, "round": s.round, "front_owner": s.front_owner,
        "done": s.done, "winner": s.winner,
    }
    coins, base_hp, burned = list(s.coins), list(s.base_hp), list(s.burned)
    hands = [list(h) for h in s.hands]
    decks = [list(d) for d in s.decks]
    backline = [[u._asdict() for u in z] for z in s.backline]
    frontline = [u._asdict() for u in s.frontline]

    def combat(attacker: dict, target: dict) -> None:
        attacker_atk, target_atk = attacker["atk"], target["atk"]
        target["hp"] -= attacker_atk
        attacker["hp"] -= target_atk
        attacker["ready"] = False
        for zone in (backline[0], backline[1], frontline):
            zone[:] = [u for u in zone if u["hp"] > 0]
        if not frontline:
            st["front_owner"] = None

    if kind == END_TURN:
        coins[p] = 0
        st["current"] = o
        new_round = s.round + 1 if o == s.first_player else s.round
        if new_round > config.max_rounds:
            # The game ends after 50 full rounds; `round` stays at the last round played.
            st["done"], st["winner"] = True, DRAW
        else:
            st["round"] = new_round
            coins[o] = new_round if config.coin_cap is None else min(new_round, config.coin_cap)
            if decks[o]:
                card = decks[o].pop()
                if len(hands[o]) >= H:
                    burned[o] += 1
                else:
                    insort(hands[o], card)
            for zone in (backline[o], frontline):
                for u in zone:
                    if u["owner"] == o:
                        u["ready"] = True
    elif kind == PLAY:
        card = hands[p].pop(a)
        cdef = config.cards[card]
        coins[p] -= cdef.cost
        backline[p].append({"card": card, "atk": cdef.attack, "hp": cdef.health, "owner": p, "ready": False})
    elif kind == MOVE:
        unit = backline[p].pop(a)
        unit["ready"] = False
        frontline.append(unit)
        st["front_owner"] = p
    elif kind == ATTACK_BASE:
        unit = frontline[a]
        unit["ready"] = False
        base_hp[o] -= unit["atk"]
        if base_hp[o] <= 0:
            st["done"], st["winner"] = True, p
    elif kind == FRONT_ATTACK:
        combat(frontline[a], backline[o][b])
    elif kind == BACK_ATTACK:
        combat(backline[p][a], frontline[b])

    return RefState(
        first_player=s.first_player, current=st["current"], round=st["round"],
        coins=tuple(coins), base_hp=tuple(base_hp),
        hands=tuple(tuple(h) for h in hands), decks=tuple(tuple(d) for d in decks),
        burned=tuple(burned), backline=tuple(tuple(RefUnit(**u) for u in z) for z in backline),
        frontline=tuple(RefUnit(**u) for u in frontline), front_owner=st["front_owner"],
        done=st["done"], winner=st["winner"],
    )


def comparable(s: RefState) -> RefState:
    """Drop fields the spec leaves open: whose turn it is once the game is over."""
    return s._replace(current=-1) if s.done else s
