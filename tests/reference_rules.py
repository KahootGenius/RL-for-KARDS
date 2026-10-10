"""Independent reference model of the Stage 2 rules, written from SPEC.md §2-§3 only.

It shares no code with `cardgame.engine`: it reads the documented internal state (SPEC §4)
into plain tuples and re-derives setup, legality and transitions from the spec text and tables,
so the tests compare the engine against a second, deliberately naive implementation.
Only the card pool and the config numbers are taken from the `GameConfig`.

Scope (SPEC §2.11): effect-free (vanilla) cards with `mulligan=False`, i.e. phase MAIN only. The action
index formulas cover the whole Stage 3 layout (CHOOSE / MULLIGAN / CONFIRM are appended after ATTACK).
"""
from __future__ import annotations

import operator
import random
from bisect import insort
from typing import NamedTuple, Optional

DRAW = -1  # SPEC §4: winner() == DRAW (-1) for a draw
TROOP, FAST, RANGED = 0, 1, 2  # SPEC §4 nature codes

# Action kinds, in SPEC §3 order.
END_TURN, PLAY, MOVE, ATTACK, CHOOSE, MULLIGAN, CONFIRM = range(7)
KIND_NAMES = ("END_TURN", "PLAY", "MOVE", "ATTACK", "CHOOSE", "MULLIGAN", "CONFIRM")

UNIT_FIELDS = ("card", "owner", "uid", "atk", "hp", "max_hp", "armor", "defense", "nature", "move_cost",
               "summoned", "moved", "attacked")


class RefUnit(NamedTuple):
    card: int
    owner: int
    uid: int
    atk: int
    hp: int
    max_hp: int
    armor: int
    defense: bool
    nature: int
    move_cost: int
    summoned: bool
    moved: bool
    attacked: bool


class RefState(NamedTuple):
    """Plain, immutable copy of the documented engine state (SPEC §4)."""
    first_player: int
    current: int
    round: int
    deck_ids: tuple
    coins: tuple
    base_hp: tuple
    hands: tuple       # (tuple, tuple), sorted card indices
    deck_cards: tuple  # (tuple, tuple), top = end
    played: tuple      # (tuple, tuple), copies played per card index
    burned: tuple
    backline: tuple    # (tuple[RefUnit], tuple[RefUnit])
    frontline: tuple   # tuple[RefUnit]
    front_owner: Optional[int]
    next_uid: int
    done: bool
    winner: Optional[int]


_unit_fields = operator.attrgetter(*UNIT_FIELDS)


def ref_unit(u) -> RefUnit:
    return RefUnit._make(_unit_fields(u))


def extract_state(game) -> RefState:
    """Read the documented internal state of an engine `Game` into a `RefState`."""
    def units(zone):
        return tuple(ref_unit(u) for u in zone)

    return RefState(
        first_player=game.first_player, current=game.current, round=game.round,
        deck_ids=tuple(game.deck_ids), coins=tuple(game.coins), base_hp=tuple(game.base_hp),
        hands=tuple(tuple(h) for h in game.hands), deck_cards=tuple(tuple(d) for d in game.deck_cards),
        played=tuple(tuple(x) for x in game.played), burned=tuple(game.burned),
        backline=tuple(units(z) for z in game.backline), frontline=units(game.frontline),
        front_owner=game.front_owner, next_uid=game.next_uid, done=bool(game.done), winner=game.winner(),
    )


def comparable(s: RefState) -> RefState:
    """Drop what the spec leaves open: whose turn it is once the game is over."""
    return s._replace(current=-1) if s.done else s


# ---------------------------------------------------------------- action indices (SPEC §3)
def num_actions(H: int, Z: int) -> int:
    return 1 + H + Z + (2 * Z) * (2 * Z + 1) + (3 * Z + 2) + H + 1


def action_index(kind: int, a: int, b: int, H: int, Z: int) -> int:
    """Index formula from SPEC §3."""
    choose0 = 1 + H + Z + (2 * Z) * (2 * Z + 1)
    if kind == END_TURN:
        return 0
    if kind == PLAY:
        return 1 + a
    if kind == MOVE:
        return 1 + H + a
    if kind == ATTACK:
        return 1 + H + Z + a * (2 * Z + 1) + b
    if kind == CHOOSE:
        return choose0 + a
    if kind == MULLIGAN:
        return choose0 + 3 * Z + 2 + a
    if kind == CONFIRM:
        return choose0 + 3 * Z + 2 + H
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
    if i < 2 * Z * (2 * Z + 1):
        return ATTACK, i // (2 * Z + 1), i % (2 * Z + 1)
    i -= 2 * Z * (2 * Z + 1)
    if i < 3 * Z + 2:
        return CHOOSE, i, -1
    i -= 3 * Z + 2
    if i < H:
        return MULLIGAN, i, -1
    if i == H:
        return CONFIRM, -1, -1
    raise ValueError(index)


# ---------------------------------------------------------------- action economy (SPEC §2 table)
def can_move(u: RefUnit) -> bool:
    if u.nature == FAST:
        return not u.summoned and not u.moved
    return not u.summoned and not u.moved and not u.attacked  # troop and ranged


def can_attack(u: RefUnit) -> bool:
    if u.nature == FAST:
        return not u.summoned and not u.attacked
    return not u.summoned and not u.moved and not u.attacked  # troop and ranged


# ---------------------------------------------------------------- legality (SPEC §2 "Actions")
def attackers(s: RefState, Z: int) -> list:
    """(attacker slot, unit) for every unit of the current player, as SPEC §3 numbers them."""
    p = s.current
    out = [(j, u) for j, u in enumerate(s.backline[p])]
    if s.front_owner == p:
        out += [(Z + j, u) for j, u in enumerate(s.frontline)]
    return out


def targets(s: RefState, Z: int) -> list:
    """(target slot, unit or None for the base, zone units) for every enemy target."""
    o = 1 - s.current
    out = [(k, u, s.backline[o]) for k, u in enumerate(s.backline[o])]
    if s.front_owner == o:
        out += [(Z + k, u, s.frontline) for k, u in enumerate(s.frontline)]
    out.append((2 * Z, None, ()))
    return out


def reaches(attacker_slot: int, attacker: RefUnit, target_slot: int, Z: int) -> bool:
    if attacker.nature == RANGED:
        return True
    if attacker_slot < Z:  # troop/fast in the backline: enemy frontline units only
        return Z <= target_slot < 2 * Z
    return target_slot < Z or target_slot == 2 * Z  # in the frontline: enemy backline or base


def defense_allows(target: Optional[RefUnit], zone: tuple) -> bool:
    if target is None:  # the base is never protected
        return True
    return target.defense or not any(u.defense for u in zone)


def legal_from_state(s: RefState, config) -> list:
    """Sorted legal action indices for state `s`."""
    if s.done:
        return []
    H, Z = config.max_hand_size, config.zone_capacity
    p = s.current
    coins = s.coins[p]
    legal = [action_index(END_TURN, -1, -1, H, Z)]
    if len(s.backline[p]) < Z:
        for i, card in enumerate(s.hands[p]):
            if config.cards[card].cost <= coins:
                legal.append(action_index(PLAY, i, -1, H, Z))
    if s.front_owner in (None, p) and len(s.frontline) < Z:
        for j, u in enumerate(s.backline[p]):
            if can_move(u) and u.move_cost <= coins:
                legal.append(action_index(MOVE, j, -1, H, Z))
    for a, att in attackers(s, Z):
        if not can_attack(att):
            continue
        for t, tgt, zone in targets(s, Z):
            if reaches(a, att, t, Z) and defense_allows(tgt, zone):
                legal.append(action_index(ATTACK, a, t, H, Z))
    return sorted(legal)


def reference_legal(game) -> list:
    """Sorted legal action indices of an engine `Game`, re-derived from SPEC.md."""
    return legal_from_state(extract_state(game), game.config)


# ---------------------------------------------------------------- transitions (SPEC §2)
def _start_turn(st: dict, p: int, config) -> None:
    """Turn start: coins = round (capped), draw 1 (empty deck: none; full hand: burned)."""
    rnd = st["round"]
    st["coins"][p] = rnd if config.coin_cap is None else min(rnd, config.coin_cap)
    deck = st["deck_cards"][p]
    if deck:
        card = deck.pop()
        if len(st["hands"][p]) >= config.max_hand_size:
            st["burned"][p] += 1
        else:
            insort(st["hands"][p], card)


def _thaw(s: RefState) -> dict:
    return {
        "current": s.current, "round": s.round, "front_owner": s.front_owner, "next_uid": s.next_uid,
        "done": s.done, "winner": s.winner, "coins": list(s.coins), "base_hp": list(s.base_hp),
        "burned": list(s.burned), "hands": [list(h) for h in s.hands],
        "deck_cards": [list(d) for d in s.deck_cards], "played": [list(x) for x in s.played],
        "backline": [[u._asdict() for u in z] for z in s.backline],
        "frontline": [u._asdict() for u in s.frontline],
    }


def _freeze(s: RefState, st: dict) -> RefState:
    return RefState(
        first_player=s.first_player, current=st["current"], round=st["round"], deck_ids=s.deck_ids,
        coins=tuple(st["coins"]), base_hp=tuple(st["base_hp"]),
        hands=tuple(tuple(h) for h in st["hands"]), deck_cards=tuple(tuple(d) for d in st["deck_cards"]),
        played=tuple(tuple(x) for x in st["played"]), burned=tuple(st["burned"]),
        backline=tuple(tuple(RefUnit(**u) for u in z) for z in st["backline"]),
        frontline=tuple(RefUnit(**u) for u in st["frontline"]), front_owner=st["front_owner"],
        next_uid=st["next_uid"], done=st["done"], winner=st["winner"],
    )


def reference_step(s: RefState, action: int, config, check: bool = True) -> RefState:
    """The state SPEC.md says follows `s` after `action` (which must be legal; `check=False` skips
    re-deriving the legal set when the caller already compared it)."""
    H, Z = config.max_hand_size, config.zone_capacity
    if check and action not in legal_from_state(s, config):
        raise ValueError(f"reference: action {action} is illegal")
    kind, a, b = decode_index(action, H, Z)
    p = s.current
    o = 1 - p
    st = _thaw(s)
    coins, backline, frontline = st["coins"], st["backline"], st["frontline"]

    if kind == END_TURN:
        coins[p] = 0
        for zone in (backline[0], backline[1], frontline):
            for u in zone:
                if u["owner"] == p:
                    u["summoned"] = u["moved"] = u["attacked"] = False
        if o == s.first_player:
            if s.round == config.max_rounds:
                st["done"], st["winner"] = True, DRAW  # round stays at the last round
                return _freeze(s, st)
            st["round"] = s.round + 1
        st["current"] = o
        _start_turn(st, o, config)
    elif kind == PLAY:
        card = st["hands"][p].pop(a)
        c = config.cards[card]
        coins[p] -= c.cost
        backline[p].append(RefUnit(
            card=card, owner=p, uid=st["next_uid"], atk=c.attack, hp=c.health, max_hp=c.health,
            armor=c.armor, defense=c.defense, nature=c.nature, move_cost=c.move_cost,
            summoned=True, moved=False, attacked=False)._asdict())
        st["next_uid"] += 1
        st["played"][p][card] += 1
    elif kind == MOVE:
        unit = backline[p].pop(a)
        coins[p] -= unit["move_cost"]
        unit["moved"] = True
        frontline.append(unit)
        st["front_owner"] = p
    elif kind == ATTACK:
        att = backline[p][a] if a < Z else frontline[a - Z]
        att["attacked"] = True
        if b == 2 * Z:
            st["base_hp"][o] -= att["atk"]  # no armor, no return damage
            if st["base_hp"][o] <= 0:
                st["done"], st["winner"] = True, p
        else:
            tgt = backline[o][b] if b < Z else frontline[b - Z]
            att_atk, tgt_atk = att["atk"], tgt["atk"]  # pre-combat values, simultaneous
            tgt["hp"] -= max(0, att_atk - tgt["armor"])
            if att["nature"] != RANGED:
                att["hp"] -= max(0, tgt_atk - att["armor"])
            for zone in (backline[0], backline[1], frontline):
                zone[:] = [u for u in zone if u["hp"] > 0]
            if not frontline:
                st["front_owner"] = None
    return _freeze(s, st)


# ---------------------------------------------------------------- setup (SPEC §2 "Setup")
def sample_decks(seed: int, n_decks: int) -> tuple:
    r = random.Random(f"decks:{seed}")
    return (r.randrange(n_decks), r.randrange(n_decks))


def reference_reset(seed: int, config, decks=None) -> tuple:
    """(RefState, rng state) right after `reset(seed, decks)`."""
    if decks is None:
        decks = sample_decks(seed, len(config.decks))
    rng = random.Random(seed)
    first = rng.randrange(2)
    deck_cards = [list(config.decks[decks[0]]), list(config.decks[decks[1]])]
    rng.shuffle(deck_cards[0])
    rng.shuffle(deck_cards[1])
    hands = [[], []]
    for p, n in ((first, config.opening_hand[0]), (1 - first, config.opening_hand[1])):
        for _ in range(n):
            if deck_cards[p]:
                insort(hands[p], deck_cards[p].pop())
    n_cards = len(config.cards)
    st = {"round": 1, "coins": [0, 0], "deck_cards": deck_cards, "hands": hands, "burned": [0, 0]}
    _start_turn(st, first, config)
    state = RefState(
        first_player=first, current=first, round=1, deck_ids=tuple(decks), coins=tuple(st["coins"]),
        base_hp=(config.base_hp, config.base_hp), hands=tuple(tuple(h) for h in hands),
        deck_cards=tuple(tuple(d) for d in deck_cards), played=((0,) * n_cards, (0,) * n_cards),
        burned=tuple(st["burned"]), backline=((), ()), frontline=(), front_owner=None, next_uid=0,
        done=False, winner=None,
    )
    return state, rng.getstate()
