"""Headless, deterministic game engine (Stage 3). All rules live here; see SPEC.md §2-§5.

Stage 3 adds card effects (an event queue with listener order, a damage/death step, chained triggers
and a loop guard), operation cards, pending choices (phase CHOICE, CHOOSE actions), the mulligan
(phase MULLIGAN), keywords (blitz, fury, smokescreen, pin), random decks and `determinize`.
Phase 1b (SPEC §1.2b) adds tags, the ambush / shock / immune traits, `on_attacked`, static (continuous)
effects, watcher event filters, clause batches (`prev`), `adjacent` targets, compare / history / prev
conditions, `target_condition` + `else`, `repeat` and a few action parameters.
With `mulligan=False` and effect-free cards the engine plays (and consumes the RNG) exactly like
Stage 2. Per-card trigger tables are precomputed once per card pool, and the effect machinery is
skipped entirely when the pool has no effects (the static recompute when no static source is on the
board).
"""
from __future__ import annotations

import copy
import operator
import random
from bisect import insort
from typing import NamedTuple, Optional, Sequence

import numpy as np

from .actions import ActionSpace
from .cards import (AMBUSH_BIT, ARMOR_BIT, BLITZ_BIT, BOOL_TRAITS, DEFENSE_BIT, FAST, FULL, FURY_BIT, HISTORY_EVENTS,
                    IMMUNE_BIT, NATURES, RANGED, SCOPES, SHOCK_BIT, SMOKESCREEN_BIT, TRAIT_BITS, TRIGGER_INDEX,
                    TRIGGERS, TROOP, UNCAPPED_BASE_HP, CardDef, CardPool, GameConfig, generate_deck, load_ruleset,
                    sample_decks)

DRAW = -1
MULLIGAN, MAIN, CHOICE = 0, 1, 2  # phases (SPEC §2.6)
PHASE_NAMES = ("MULLIGAN", "MAIN", "CHOICE")

__all__ = ["DRAW", "TROOP", "FAST", "RANGED", "MULLIGAN", "MAIN", "CHOICE", "IllegalActionError", "UnitView",
           "PendingView", "Observation", "Unit", "Game", "combat_damage", "ambush_fires"]

# trigger codes (index into cards.TRIGGERS) and scope codes (index into cards.SCOPES)
(T_PLAY, T_DEPLOY, T_DEATH, T_ATTACK, T_DAMAGED, T_MOVE, T_KILL, T_START, T_END, T_ATTACKED,
 T_STATIC) = range(len(TRIGGERS))
S_SELF, S_FRIENDLY, S_ENEMY, S_ANY = range(len(SCOPES))
assert TRIGGERS[T_END] == "end_of_turn" and TRIGGERS[T_STATIC] == "static" and SCOPES[S_ANY] == "any"
H_OPS, H_DEPLOYED, H_DIED = range(len(HISTORY_EVENTS))  # history counter slots
HISTORY_INDEX = {name: i for i, name in enumerate(HISTORY_EVENTS)}
_BOOL_TRAIT_LIST = tuple((name, TRAIT_BITS[name]) for name in BOOL_TRAITS)
_BOOL_TRAIT_BITS = dict(_BOOL_TRAIT_LIST)
_BOOL_MASK = sum(bit for _, bit in _BOOL_TRAIT_LIST)
_NO_RECORD = ((), False)  # batch record of a clause that hit nothing
_NO_PREVIEW = (0, 0, 0, 0)  # PendingView.previews entry of a slot that is not an option (or gets nothing)


def ambush_fires(attacker, target) -> bool:
    """Whether the target's Ambush strikes first (SPEC §1.2b): it has an unused Ambush this turn and the
    attacker would take return damage (> 0: not ranged, no shock, not immune, armor below the target's atk).
    Works on `Unit` and `UnitView` alike."""
    return bool(target.ambush and target.ambush_ready and attacker.nature != RANGED and not attacker.shock
                and not attacker.immune and target.atk > attacker.armor)


def combat_damage(attacker, target) -> tuple:
    """(damage to the target, damage to the attacker) of one attack between two units.

    The single source of the combat rules (SPEC §2, §1.2b): armor reduces each hit (never below 0), ranged
    and shock attackers take no return damage, immune units take no damage, and a ready Ambush strikes
    first: if that strike kills the attacker, the target takes nothing. Works on `Unit` and `UnitView`
    alike, so agents can reason about trades without re-implementing the rules.
    """
    if target.immune:
        to_target = 0
    else:
        to_target = attacker.atk - target.armor
        if to_target < 0:
            to_target = 0
    if attacker.nature == RANGED or attacker.shock or attacker.immune:
        return to_target, 0
    to_attacker = target.atk - attacker.armor
    if to_attacker < 0:
        to_attacker = 0
    if target.ambush and to_attacker >= attacker.hp and target.ambush_ready:
        return 0, to_attacker  # the ambush strike kills the attacker first
    return to_target, to_attacker


class IllegalActionError(ValueError):
    pass


class UnitView(NamedTuple):
    card: int
    atk: int
    hp: int
    max_hp: int
    armor: int
    defense: bool
    nature: int
    move_cost: int
    summoned: bool     # deployed/summoned this round
    moved: bool
    attacked: bool     # attacks > 0
    can_move: bool     # action economy only (coins, space and frontline control not included)
    can_attack: bool   # action economy only (reach, Defense and targets not included)
    blitz: bool = False
    smokescreen: bool = False
    fury: bool = False
    pinned: bool = False
    attacks: int = 0
    temp_atk: int = 0
    temp_hp: int = 0
    token: bool = False
    ambush: bool = False
    shock: bool = False
    immune: bool = False
    ambush_ready: bool = False  # ambush and not yet used this turn
    # ---- observation fields (SPEC §5): what lapses or ends on its own
    pin_turns: int = 0       # owner turns (from the current one) the pin still covers; 0 = not pinned
    temp_move_cost: int = 0  # move-cost change that lapses at the end of this turn
    temp_traits: int = 0     # trait bits granted until the end of this turn (armor bit: a "turn" armor grant)
    temp_removed: int = 0    # trait bits removed until the end of this turn (armor bit: a "turn" armor removal)


class PendingView(NamedTuple):
    """The effect waiting for CHOOSE (public: the played card is revealed). Strings are the JSON names."""
    card: int          # card index of the effect's card
    effect: int        # effect slot within the card
    action: str
    amount: int        # resolved `amount`; -1 = heal "full"; 0 when the action has no amount (buff included)
    select: str
    side: str
    kind: str
    atk: int = 0       # buff: the resolved atk change (0 for other actions)
    hp: int = 0        # buff: the resolved hp change (0 for other actions)
    # one (kills, dealt, healed, other) tuple per CHOOSE slot of the chooser (the turn player), zeros for slots
    # that are not options; computed with the body that would apply to that option (SPEC §5)
    previews: tuple = ()


class Observation(NamedTuple):
    """Everything one player may see, from that player's point of view."""
    player: int
    is_my_turn: bool
    went_first: bool
    round: int
    my_deck: int       # own fixed deck index (-1 for a random deck); the opponent's deck is hidden
    my_coins: int
    opp_coins: int
    my_base_hp: int
    opp_base_hp: int
    hand: tuple
    opp_hand_size: int
    my_deck_size: int
    opp_deck_size: int
    my_played: tuple   # copies of each card index the observer has played this game (public)
    opp_played: tuple  # same for the opponent
    my_backline: tuple
    opp_backline: tuple
    frontline: tuple
    front_owner: int  # +1 observer, -1 opponent, 0 empty
    done: bool
    result: int       # +1 observer won, -1 lost, 0 draw/ongoing
    # ---- Stage 3 (appended with defaults)
    phase: int = MAIN
    mulligan_marks: tuple = ()   # own marks per hand slot during the observer's mulligan, else ()
    pending: Optional[PendingView] = None
    turn: int = 0
    my_coin_bonus: int = 0
    opp_coin_bonus: int = 0
    my_decklist: tuple = ()      # counts per card index
    my_deck_counts: tuple = ()   # remaining deck contents (order hidden)
    my_known_hand: tuple = ()    # observer's hand cards the opponent knows
    opp_known_hand: tuple = ()   # opponent's hand cards the observer knows
    opp_revealed: tuple = ()     # non-token copies seen from the opponent's deck
    my_discard: tuple = ()
    opp_discard: tuple = ()
    my_graveyard: tuple = ()
    opp_graveyard: tuple = ()
    my_burned: int = 0
    opp_burned: int = 0
    # public counters (operations played, units deployed, units died) this turn, then the same three this game
    my_history: tuple = ()
    opp_history: tuple = ()


_new_tuple = tuple.__new__  # builds a NamedTuple from a full tuple without the Python-level __new__


class Unit:
    """A unit on the board. All fields are immutable scalars (trait bookkeeping uses int bitmasks), so
    copying a unit stays a cheap slot copy; subclasses must keep any per-unit collection immutable.

    `attacked` is derived (`attacks > 0`) and kept for Stage 2 code; assigning it sets `attacks`.
    Effective values live in the fields. "turn" changes are also recorded in `temp_*` so they can be
    undone: `temp_atk`/`temp_hp`/`temp_armor`/`temp_move_cost` are the applied deltas, `temp_traits`/
    `temp_removed` mark traits granted/removed for the turn, and `base_traits` holds the permanent traits
    they revert to (own traits = (base & ~temp_removed) | temp_traits). Static effects (SPEC §2.12) add
    `static_atk`/`static_hp`/`static_move_cost` (applied deltas) and `static_traits` on top: the effective
    traits are own | static. `ambush_used` marks an Ambush spent this turn.
    """
    __slots__ = ("card", "owner", "uid", "atk", "hp", "max_hp", "armor", "defense", "nature", "move_cost",
                 "summoned", "moved", "attacks", "blitz", "smokescreen", "fury", "pinned", "pin_until",
                 "temp_atk", "temp_hp", "temp_armor", "temp_traits", "temp_removed", "base_traits", "token",
                 "ambush", "shock", "immune", "ambush_used", "temp_move_cost", "static_atk", "static_hp",
                 "static_move_cost", "static_traits")

    def __init__(self, card: int, owner: int, *, atk: int, hp: int, max_hp: Optional[int] = None, armor: int = 0,
                 defense: bool = False, nature: int = TROOP, move_cost: int = 1, uid: int = -1,
                 summoned: bool = False, moved: bool = False, attacked: bool = False, attacks: int = 0,
                 blitz: bool = False, smokescreen: bool = False, fury: bool = False, pinned: bool = False,
                 pin_until: int = 0, temp_atk: int = 0, temp_hp: int = 0, temp_armor: int = 0,
                 temp_traits: int = 0, temp_removed: int = 0, base_traits: Optional[int] = None,
                 token: bool = False, ambush: bool = False, shock: bool = False, immune: bool = False,
                 ambush_used: bool = False, temp_move_cost: int = 0, static_atk: int = 0, static_hp: int = 0,
                 static_move_cost: int = 0, static_traits: int = 0):
        self.card, self.owner, self.uid, self.atk, self.hp = card, owner, uid, atk, hp
        self.max_hp = hp if max_hp is None else max_hp
        self.armor, self.defense, self.nature, self.move_cost = armor, defense, nature, move_cost
        self.summoned, self.moved = summoned, moved
        self.attacks = 1 if attacked and not attacks else attacks
        self.blitz, self.smokescreen, self.fury = blitz, smokescreen, fury
        self.pinned, self.pin_until = pinned, pin_until
        self.temp_atk, self.temp_hp, self.temp_armor = temp_atk, temp_hp, temp_armor
        self.temp_traits, self.temp_removed = temp_traits, temp_removed
        self.token = token
        self.ambush, self.shock, self.immune, self.ambush_used = ambush, shock, immune, ambush_used
        self.temp_move_cost = temp_move_cost
        self.static_atk, self.static_hp, self.static_move_cost = static_atk, static_hp, static_move_cost
        self.static_traits = static_traits
        if base_traits is None:  # the flags minus what "turn" grants and static effects add, plus turn removals
            flags = self.trait_mask() & ~ARMOR_BIT
            base_traits = ((flags & ~(temp_traits | static_traits)) | (temp_removed & _BOOL_MASK)
                           | (ARMOR_BIT if armor - temp_armor > 0 else 0))
        self.base_traits = base_traits

    @property
    def attacked(self) -> bool:
        return self.attacks > 0

    @attacked.setter
    def attacked(self, value: bool) -> None:
        self.attacks = (self.attacks or 1) if value else 0

    @property
    def ambush_ready(self) -> bool:
        """Ambush not yet used this turn (SPEC §1.2b)."""
        return self.ambush and not self.ambush_used

    @classmethod
    def from_card(cls, c: CardDef, owner: int, uid: int = -1) -> "Unit":
        """A freshly deployed unit (cannot act this round unless it has blitz)."""
        if not c.is_unit:
            raise ValueError(f"card {c.id!r} is not a unit")
        return cls(c.index, owner, atk=c.attack, hp=c.health, max_hp=c.health, armor=c.armor, defense=c.defense,
                   nature=c.nature, move_cost=c.move_cost, uid=uid, summoned=True, blitz=c.blitz,
                   smokescreen=c.smokescreen, fury=c.fury, token=c.token, ambush=c.ambush, shock=c.shock,
                   immune=c.immune, base_traits=c.trait_mask)

    def trait_mask(self) -> int:
        """Bitmask of the unit's current traits (armor bit when armor > 0)."""
        return ((DEFENSE_BIT if self.defense else 0) | (ARMOR_BIT if self.armor > 0 else 0)
                | (BLITZ_BIT if self.blitz else 0) | (SMOKESCREEN_BIT if self.smokescreen else 0)
                | (FURY_BIT if self.fury else 0) | (AMBUSH_BIT if self.ambush else 0)
                | (SHOCK_BIT if self.shock else 0) | (IMMUNE_BIT if self.immune else 0))

    def ready(self) -> bool:
        return not self.pinned and (not self.summoned or self.blitz)

    def can_move(self) -> bool:
        return (not self.pinned and (not self.summoned or self.blitz) and not self.moved
                and (self.nature == FAST or self.attacks == 0))

    def can_attack(self) -> bool:
        return (not self.pinned and (not self.summoned or self.blitz) and self.attacks < (2 if self.fury else 1)
                and (self.nature == FAST or not self.moved))

    def copy(self) -> "Unit":
        cls = type(self)
        u = cls.__new__(cls)
        if cls is Unit:  # fast path: plain slot copies (one statement each: ~2x faster than a tuple unpack)
            u.card = self.card
            u.owner = self.owner
            u.uid = self.uid
            u.atk = self.atk
            u.hp = self.hp
            u.max_hp = self.max_hp
            u.armor = self.armor
            u.defense = self.defense
            u.nature = self.nature
            u.move_cost = self.move_cost
            u.summoned = self.summoned
            u.moved = self.moved
            u.attacks = self.attacks
            u.blitz = self.blitz
            u.smokescreen = self.smokescreen
            u.fury = self.fury
            u.pinned = self.pinned
            u.pin_until = self.pin_until
            u.temp_atk = self.temp_atk
            u.temp_hp = self.temp_hp
            u.temp_armor = self.temp_armor
            u.temp_traits = self.temp_traits
            u.temp_removed = self.temp_removed
            u.base_traits = self.base_traits
            u.token = self.token
            u.ambush = self.ambush
            u.shock = self.shock
            u.immune = self.immune
            u.ambush_used = self.ambush_used
            u.temp_move_cost = self.temp_move_cost
            u.static_atk = self.static_atk
            u.static_hp = self.static_hp
            u.static_move_cost = self.static_move_cost
            u.static_traits = self.static_traits
        else:  # subclasses may add slots
            for klass in cls.__mro__:
                for name in getattr(klass, "__slots__", ()):
                    if hasattr(self, name):
                        setattr(u, name, copy.deepcopy(getattr(self, name)))
        return u

    def pin_turns(self, turn: int, current: int) -> int:
        """Owner turns, counted from the current turn, during which the unit stays pinned (SPEC §5): the pin
        is lifted at END_TURN of turn max(turn, pin_until) (SPEC §2.4); 0 when not pinned."""
        if not self.pinned:
            return 0
        last = self.pin_until if self.pin_until > turn else turn
        first = turn if current == self.owner else turn + 1
        return (last - first) // 2 + 1 if last >= first else 0

    def view(self, turn: Optional[int] = None, current: Optional[int] = None) -> UnitView:
        """The public view of the unit. `pin_turns` needs the game's turn and current player
        (`Game.observe` passes them); without them it is 1 for any pinned unit."""
        attacks, moved, nature = self.attacks, self.moved, self.nature
        pinned = self.pinned
        ready = not pinned and (not self.summoned or self.blitz)
        ambush = self.ambush
        ta = self.temp_armor
        return _new_tuple(UnitView, (
            self.card, self.atk, self.hp, self.max_hp, self.armor, self.defense, nature, self.move_cost,
            self.summoned, moved, attacks > 0, ready and not moved and (nature == FAST or attacks == 0),
            ready and attacks < (2 if self.fury else 1) and (nature == FAST or not moved),
            self.blitz, self.smokescreen, self.fury, pinned, attacks, self.temp_atk, self.temp_hp, self.token,
            ambush, self.shock, self.immune, ambush and not self.ambush_used,
            (0 if not pinned else 1 if turn is None else self.pin_turns(turn, current)), self.temp_move_cost,
            self.temp_traits | (ARMOR_BIT if ta > 0 else 0), self.temp_removed | (ARMOR_BIT if ta < 0 else 0)))

    def __repr__(self) -> str:
        flags = "".join(f for f, on in (("S", self.summoned), ("M", self.moved), ("A", self.attacks > 0),
                                        ("P", self.pinned)) if on)
        traits = "".join(f" {n}" for n in BOOL_TRAITS if getattr(self, n))
        nature = NATURES[self.nature] if 0 <= self.nature < len(NATURES) else "?"
        return (f"Unit(card={self.card}, p{self.owner}, uid={self.uid}, {nature} {self.atk}/{self.hp}"
                f"{traits}{f' armor{self.armor}' if self.armor else ''}{' ' + flags if flags else ''})")


# ---------------------------------------------------------------- per-pool tables (shared by clones)
def _bodies(e) -> tuple:
    """An effect and its else body."""
    return (e,) if e.else_ is None else (e, e.else_)


def _uses_prev(e) -> bool:
    """Whether an effect (or its else body) reads the previous clause of its batch (SPEC §2.8b)."""
    for b in _bodies(e):
        t = b.target
        if t.select == "prev" or (t.select == "adjacent" and t.of == "prev"):
            return True
        if any(a is not None and a.kind == "stat" and a.of == "prev" for a in (b.amount, b.atk, b.hp)):
            return True
    for c in e.condition:
        if c.type == "prev":
            return True
        if c.type == "compare" and any(a.kind == "stat" and a.of == "prev" for a in (c.left, c.right)):
            return True
    return False


class _Tables:
    """Per-card lookup tables, computed once per card pool."""
    __slots__ = ("cost", "is_op", "is_token", "unit_tpl", "self_eff", "watch_eff", "watch_any", "op_chosen",
                 "has_effects", "any_op", "tags", "static_eff", "any_static", "trig_any", "self_batch",
                 "watch_batch", "uses_ambush", "uses_history")

    def __init__(self, pool: CardPool):
        cards = pool.cards
        n_t = len(TRIGGERS)
        self.cost = tuple(c.cost for c in cards)
        self.is_op = tuple(c.is_operation for c in cards)
        self.is_token = tuple(c.token for c in cards)
        self.tags = tuple(frozenset(c.tags) for c in cards)
        self.any_op = any(self.is_op)
        self.unit_tpl = tuple(
            (c.attack, c.health, c.health, c.armor, c.defense, c.nature, c.move_cost, c.blitz, c.smokescreen,
             c.fury, c.trait_mask, c.token, c.ambush, c.shock, c.immune) if c.is_unit else None for c in cards)
        self_eff, watch_eff, static_eff, self_batch, watch_batch = [], [], [], [], []
        watch_any = [False] * n_t
        trig_any = [False] * n_t
        for c in cards:
            s = [[] for _ in range(n_t)]
            w = [[] for _ in range(n_t)]
            for e in c.effects:
                t = TRIGGER_INDEX[e.trigger]
                trig_any[t] = True
                if t == T_STATIC:
                    continue
                scope = SCOPES.index(e.scope)
                if scope == S_SELF:
                    s[t].append(e)
                else:
                    w[t].append((e, scope))
                    watch_any[t] = True
            self_eff.append(tuple(tuple(x) for x in s))
            watch_eff.append(tuple(tuple(x) for x in w))
            static_eff.append(tuple(e for e in c.effects if e.trigger == "static"))
            # a batch list is created for an event only when a clause of the card reads `prev`
            self_batch.append(tuple(any(_uses_prev(e) for e in x) for x in s))
            watch_batch.append(tuple(any(_uses_prev(e) for e, _ in x) for x in w))
        self.self_eff = tuple(self_eff)
        self.watch_eff = tuple(watch_eff)
        self.watch_any = tuple(watch_any)
        self.trig_any = tuple(trig_any)
        self.static_eff = tuple(static_eff)
        self.any_static = any(static_eff)
        self.self_batch = tuple(self_batch)
        self.watch_batch = tuple(watch_batch)
        self.op_chosen = tuple(
            tuple(e for e in c.effects if any(b.target.select == "chosen" for b in _bodies(e))) if c.is_operation
            else () for c in cards)
        self.has_effects = any(c.effects for c in cards)
        bodies = [b for c in cards for e in c.effects for b in _bodies(e)]
        self.uses_ambush = (any(c.ambush for c in cards)
                            or any(b.action == "add_trait" and b.trait == ("ambush",) for b in bodies))
        self.uses_history = any(cd.type == "history" for c in cards for e in c.effects for cd in e.condition)


def _tables(pool: CardPool) -> _Tables:
    t = pool.__dict__.get("_engine_tables")
    if t is None:
        t = _Tables(pool)
        object.__setattr__(pool, "_engine_tables", t)
    return t


# Attributes that are immutable for the lifetime of a Game and may be shared by clones.
_SHARED_ATTRS = frozenset({"config", "action_space", "num_actions", "_cost", "_card_defs", "_t", "_is_op",
                           "_is_token", "_unit_tpl", "_self_eff", "_watch_eff", "_watch_any", "_op_chosen", "_tags",
                           "_static_eff", "_self_batch", "_watch_batch"})
_SCALAR_TYPES = (int, float, str, bool, type(None))
_LIST_OF_LISTS = ("deck_cards", "hands", "played", "discard", "graveyard", "known_hand", "revealed",
                  "history_turn", "history_game")
_FLAT_LISTS = ("burned", "base_hp", "coins", "coin_bonus", "mulligan_done")
# Every attribute clone() knows: shared tables, immutable scalars/tuples and the mutable fields it copies.
_KNOWN_ATTRS = _SHARED_ATTRS | frozenset(_LIST_OF_LISTS) | frozenset(_FLAT_LISTS) | frozenset({
    "_has_effects", "_any_op", "done", "_winner", "current", "round", "turn", "phase", "num_steps", "pending",
    "_legal", "_mask", "seed", "rng", "deck_ids", "decklists", "next_uid", "first_player", "backline",
    "frontline", "front_owner", "queue", "_combat", "guard_trips", "mulligan_marks", "_events", "_tripped",
    "_scan_units", "_dlc", "_any_static", "_static_on", "_dead_log", "_uses_ambush", "_uses_history",
    "_attacked_any", "_defer"})


def _as_index(value, what: str) -> int:
    """Strict integer conversion: ints and numpy ints, never bools/floats/None."""
    if type(value) is int:
        return value
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{what} must be an integer, got {value!r}")
    return operator.index(value)


def _int_tuple(v) -> bool:
    return type(v) is tuple and all(type(x) is int for x in v)


class Game:
    def __init__(self, config: Optional[GameConfig] = None):
        self.config = config if config is not None else load_ruleset()
        cfg = self.config
        self.action_space = ActionSpace(cfg.max_hand_size, cfg.zone_capacity)
        self.num_actions = self.action_space.n
        self._card_defs = cfg.cards.cards
        t = self._t = _tables(cfg.cards)
        self._cost, self._is_op, self._is_token, self._unit_tpl = t.cost, t.is_op, t.is_token, t.unit_tpl
        self._self_eff, self._watch_eff, self._watch_any, self._op_chosen = (t.self_eff, t.watch_eff, t.watch_any,
                                                                             t.op_chosen)
        self._has_effects = t.has_effects
        self._any_op = t.any_op
        self._tags, self._static_eff, self._self_batch, self._watch_batch = (t.tags, t.static_eff, t.self_batch,
                                                                             t.watch_batch)
        self._any_static = t.any_static
        self._uses_ambush, self._uses_history = t.uses_ambush, t.uses_history
        self._attacked_any = t.trig_any[T_ATTACKED]
        self._static_on = False  # some unit carries static contributions (the recompute cannot be skipped)
        self._dead_log = None    # units removed by the clause being resolved (batch records), else None
        self._defer = None       # damage steps deferred by a target_condition split (see _hit), else None
        # Not started: queries are safe (no legal actions, no winner) until reset().
        self.done = True
        self._winner = None
        self.current = 0
        self.round = 0
        self.turn = 0
        self.phase = MAIN
        self.num_steps = 0
        self.pending = None
        self._legal = []
        self._mask = bytearray(self.num_actions)

    # ------------------------------------------------------------------ setup
    def _parse_decks(self, decks) -> tuple:
        """(deck_ids, decklists) from a pair of deck specs: fixed index or a legal 40-card sequence."""
        cfg = self.config
        if isinstance(decks, (str, bytes)) or not hasattr(decks, "__len__"):
            raise TypeError(f"decks must be a pair of deck specs, got {decks!r}")
        if len(decks) != 2:
            raise ValueError(f"decks must be a pair of deck specs, got {decks!r}")
        ids, lists = [], []
        n_cards = len(self._card_defs)
        for d in decks:
            if isinstance(d, (int, np.integer)) or isinstance(d, (bool, np.bool_)):
                d = _as_index(d, "deck index")
                if not 0 <= d < cfg.n_decks:
                    raise ValueError(f"deck index must be in [0, {cfg.n_decks}), got {d}")
                ids.append(d)
                lists.append(cfg.decks[d])
                continue
            if isinstance(d, (str, bytes)) or not hasattr(d, "__len__"):
                raise TypeError(f"a deck spec is a deck index or a tuple of card indices, got {d!r}")
            cards = tuple(sorted(_as_index(c, "card index") for c in d))
            if len(cards) != cfg.deck_size:
                raise ValueError(f"a deck must have {cfg.deck_size} cards, got {len(cards)}")
            counts = {}
            for c in cards:
                if not 0 <= c < n_cards:
                    raise ValueError(f"card index {c} out of range")
                if self._is_token[c]:
                    raise ValueError(f"token card {self._card_defs[c].id!r} cannot be in a deck")
                counts[c] = counts.get(c, 0) + 1
                if counts[c] > cfg.max_copies:
                    raise ValueError(f"deck has more than {cfg.max_copies} copies of {self._card_defs[c].id!r}")
            ids.append(-1)
            lists.append(cards)
        return tuple(ids), tuple(lists)

    def reset(self, seed: int, decks: Optional[Sequence] = None) -> None:
        seed = _as_index(seed, "seed")  # ints and numpy ints; rejects None/bools/floats (determinism)
        if seed < 0:
            raise ValueError("seed must be a non-negative integer")  # Random(-s) == Random(s)
        cfg = self.config
        if decks is None:  # own stream: reset(s) == reset(s, decks=sample_decks(s, n))
            decks = sample_decks(seed, cfg.n_decks)
        ids, lists = self._parse_decks(decks)
        n = len(self._card_defs)
        self.seed = seed
        self.rng = random.Random(seed)
        self.deck_ids = ids
        self.decklists = lists
        self.next_uid = 0
        self.first_player = self.rng.randrange(2)
        self.deck_cards = [list(lists[0]), list(lists[1])]
        self.rng.shuffle(self.deck_cards[0])
        self.rng.shuffle(self.deck_cards[1])
        self.hands = [[], []]
        self.played = [[0] * n, [0] * n]
        self.discard = [[0] * n, [0] * n]
        self.graveyard = [[0] * n, [0] * n]
        self.known_hand = [[0] * n, [0] * n]
        self.revealed = [[0] * n, [0] * n]
        self.burned = [0, 0]
        self.base_hp = [cfg.base_hp, cfg.base_hp]
        self.coins = [0, 0]
        self.coin_bonus = [0, 0]
        self.backline = [[], []]
        self.frontline = []
        self.front_owner = None
        self.round = 1
        self.turn = 0
        self.current = self.first_player
        self.done = False
        self._winner = None
        self.num_steps = 0
        self.queue = []
        self.pending = None
        self._combat = None
        self.guard_trips = 0
        self.mulligan_marks = set()
        self.mulligan_done = [False, False]
        self._events = 0
        self._tripped = False
        self._scan_units = False
        self._static_on = False
        self._dead_log = None
        self._defer = None
        self.history_turn = [[0, 0, 0], [0, 0, 0]]  # per player: operations played, units deployed, units died
        self.history_game = [[0, 0, 0], [0, 0, 0]]
        self._legal = self._mask = None
        first = self.first_player
        self._draw(first, cfg.opening_hand[0])
        self._draw(1 - first, cfg.opening_hand[1])
        if cfg.mulligan:
            self.phase = MULLIGAN
        else:
            self.phase = MAIN
            self._start_turn(first)

    def _draw(self, p: int, n: int) -> None:
        deck, hand = self.deck_cards[p], self.hands[p]
        for _ in range(n):
            if not deck:
                return  # no fatigue
            card = deck.pop()
            if len(hand) >= self.config.max_hand_size:
                self.burned[p] += 1
            else:
                insort(hand, card)  # hand kept sorted by card index (canonical order)

    def _start_turn(self, p: int) -> None:
        self.turn += 1
        self.current = p
        self.phase = MAIN
        self.history_turn = [[0, 0, 0], [0, 0, 0]]  # public counters (SPEC §5): kept even without history cards
        if self._uses_ambush or self._scan_units:  # every Ambush refreshes at every turn start
            for zone in (self.backline[0], self.backline[1], self.frontline):
                for u in zone:
                    if u.ambush_used:
                        u.ambush_used = False
        c = self.config.coins_for_round(self.round) + self.coin_bonus[p]
        self.coins[p] = c if c > 0 else 0
        self._draw(p, 1)
        if self._has_effects:
            self._fire_turn(T_START, p)
            if self.queue:
                self._drain()
        self._legal = self._mask = None

    # ------------------------------------------------------------------ queries
    def current_player(self) -> int:
        return self.current

    def winner(self) -> Optional[int]:
        return self._winner

    @staticmethod
    def _targetable(zone: list) -> list:
        """Slots of `zone` that may be attacked: no smokescreen, and the Defense units if any (SPEC §2.4)."""
        guards = [k for k, u in enumerate(zone) if u.defense and not u.smokescreen]
        if guards:
            return guards
        return [k for k, u in enumerate(zone) if not u.smokescreen]

    def _compute_legal(self) -> None:
        sp = self.action_space
        legal = []
        if not self.done:
            phase = self.phase
            if phase == MAIN:
                self._legal_main(legal)
            elif phase == CHOICE:
                c0 = sp.CHOOSE0
                legal = [c0 + t for t in self.pending[1]]
            else:
                marks = self.mulligan_marks
                m0 = sp.MULLIGAN0
                legal = [m0 + i for i in range(len(self.hands[self.current])) if i not in marks]
                legal.append(sp.CONFIRM)
        mask = bytearray(self.num_actions)
        for a in legal:
            mask[a] = 1
        self._legal, self._mask = legal, mask

    def _legal_main(self, legal: list) -> None:
        sp = self.action_space
        Z = self.config.zone_capacity
        p = self.current
        o = 1 - p
        legal.append(sp.END_TURN)
        back = self.backline[p]
        front = self.frontline
        fo = self.front_owner
        coins = self.coins[p]
        room = len(back) < Z
        if room or self._any_op:
            cost, is_op = self._cost, self._is_op
            for i, card in enumerate(self.hands[p]):
                if cost[card] <= coins:
                    if is_op[card]:
                        if self._op_playable(card, p):
                            legal.append(sp.PLAY0 + i)
                    elif room:
                        legal.append(sp.PLAY0 + i)
        if (fo is None or fo == p) and len(front) < Z:
            for j, u in enumerate(back):  # inlined Unit.can_move()
                if (u.move_cost <= coins and not u.moved and (not u.summoned or u.blitz) and not u.pinned
                        and (u.nature == FAST or not u.attacks)):
                    legal.append(sp.MOVE0 + j)
        # inlined Unit.can_attack()
        attackers = [(a, u) for a, u in enumerate(back)
                     if (not u.summoned or u.blitz) and not u.pinned and (not u.attacks or (u.fury and u.attacks < 2))
                     and (u.nature == FAST or not u.moved)]
        if fo == p:
            attackers += [(Z + j, u) for j, u in enumerate(front)
                          if (not u.summoned or u.blitz) and not u.pinned
                          and (not u.attacks or (u.fury and u.attacks < 2)) and (u.nature == FAST or not u.moved)]
        if attackers:
            enemy_back = self._targetable(self.backline[o])
            enemy_front = [Z + k for k in self._targetable(front)] if fo == o else []
            base_t = sp.BASE_TARGET
            ranged_targets = enemy_back + enemy_front + [base_t]
            front_melee = enemy_back + [base_t]
            n_t = sp.n_targets
            for a, u in attackers:
                if u.nature == RANGED:
                    targets = ranged_targets
                elif a < Z:
                    targets = enemy_front
                else:
                    targets = front_melee
                first = sp.ATTACK0 + a * n_t
                legal += [first + t for t in targets]

    def _op_playable(self, card: int, p: int) -> bool:
        """An operation is playable only if each `chosen` body that would resolve now (the effect if its
        condition holds, else its `else` body) has an option."""
        for eff in self._op_chosen[card]:
            inst = (eff, card, p, -1, None, -1, None, None, 0)
            body = eff
            if eff.condition and not self._conditions_hold(eff.condition, inst):
                body = eff.else_
                if body is None:
                    continue
            if body.target.select == "chosen" and not self._choice_options(body.target, inst):
                return False
        return True

    def legal_actions(self) -> list:
        if self._legal is None:
            self._compute_legal()
        return list(self._legal)

    def legal_mask(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Bool mask of shape (num_actions,). Writes into `out` (bool array) when given."""
        if self._mask is None:
            self._compute_legal()
        mask = np.frombuffer(self._mask, dtype=bool)
        if out is None:
            return mask.copy()
        out[:] = mask
        return out

    def invalidate(self) -> None:
        """Drop cached legal actions after mutating state directly (tests, tools).

        Also checks the structural invariants the fixed action space relies on, then recomputes static
        contributions (SPEC §2.12), so a hand-built position sees its auras (and its legal actions follow them).
        """
        self._legal = self._mask = None
        self._scan_units = True  # direct edits may set pins, ambush or "turn" fields on a pool without effects
        cfg = self.config
        if not self.done:
            for p in (0, 1):
                if len(self.hands[p]) > cfg.max_hand_size:
                    raise ValueError(f"player {p} hand exceeds max_hand_size")
                if len(self.backline[p]) > cfg.zone_capacity:
                    raise ValueError(f"player {p} backline exceeds zone_capacity")
                if any(u.owner != p for u in self.backline[p]):
                    raise ValueError(f"player {p} backline holds an enemy unit")
            if len(self.frontline) > cfg.zone_capacity:
                raise ValueError("frontline exceeds zone_capacity")
            owners = {u.owner for u in self.frontline}
            if (self.front_owner is None) != (not self.frontline) or (owners and owners != {self.front_owner}):
                raise ValueError("front_owner inconsistent with frontline units")
            if self.phase == CHOICE and self.pending is None:
                raise ValueError("phase CHOICE without a pending effect")
            if self._any_static and hasattr(self, "rng"):
                self._refresh_statics()

    def is_legal(self, action: int) -> bool:
        try:
            action = _as_index(action, "action")
        except TypeError:
            return False
        if self._mask is None:
            self._compute_legal()
        return 0 <= action < self.num_actions and self._mask[action] == 1

    # ------------------------------------------------------------------ transitions
    def step(self, action: int) -> None:
        if self.done:
            raise IllegalActionError("game is over")
        try:
            action = _as_index(action, "action")
        except TypeError:
            raise IllegalActionError(f"action must be an integer, got {action!r}") from None
        if self._mask is None:
            self._compute_legal()
        if not (0 <= action < self.num_actions and self._mask[action]):
            raise IllegalActionError(f"illegal action {action} ({self.describe(action)})")
        sp = self.action_space
        p = self.current
        self._events = 0  # loop guard: instances resolved during this action
        self._tripped = False
        phase = self.phase
        fresh = False  # the statics were recomputed after the action's last change (see _unit_event)
        if phase == MAIN:
            if action == sp.END_TURN:
                self._end_turn(p)
            elif action < sp.MOVE0:
                fresh = self._play(p, action - sp.PLAY0)
            elif action < sp.ATTACK0:
                fresh = self._move(p, action - sp.MOVE0)
            elif action < sp.CHOOSE0:
                a, t = divmod(action - sp.ATTACK0, sp.n_targets)
                self._attack(p, a, t)
            else:
                raise IllegalActionError(f"no handler for action {action}")
        elif phase == CHOICE:
            self._choose(action - sp.CHOOSE0)
        elif action == sp.CONFIRM:
            self._confirm(p)
        else:
            self.mulligan_marks.add(action - sp.MULLIGAN0)
        if self._any_static and not self.done and not fresh:
            self._refresh_statics()
        self.num_steps += 1
        self._legal = self._mask = None

    def _note_left_hand(self, p: int, card: int) -> None:
        """revealed / known_hand bookkeeping when p plays or discards `card` (SPEC §5)."""
        kh = self.known_hand[p]
        if kh[card] > 0:
            kh[card] -= 1           # the known copy left the hand
        elif not self._is_token[card]:
            self.revealed[p][card] += 1  # a newly seen copy from p's deck

    def _new_unit(self, card: int, owner: int) -> Unit:
        (atk, hp, max_hp, armor, defense, nature, move_cost, blitz, smoke, fury, base, token, ambush, shock,
         immune) = self._unit_tpl[card]
        u = Unit.__new__(Unit)  # one statement per slot: ~2x faster than a 34-target tuple unpack
        u.card = card
        u.owner = owner
        u.uid = self.next_uid
        u.atk = atk
        u.hp = hp
        u.max_hp = max_hp
        u.armor = armor
        u.defense = defense
        u.nature = nature
        u.move_cost = move_cost
        u.summoned = True
        u.moved = False
        u.attacks = 0
        u.blitz = blitz
        u.smokescreen = smoke
        u.fury = fury
        u.pinned = False
        u.pin_until = 0
        u.temp_atk = u.temp_hp = u.temp_armor = u.temp_traits = u.temp_removed = u.temp_move_cost = 0
        u.base_traits = base
        u.token = token
        u.ambush = ambush
        u.shock = shock
        u.immune = immune
        u.ambush_used = False
        u.static_atk = u.static_hp = u.static_move_cost = u.static_traits = 0
        self.next_uid += 1
        return u

    def _play(self, p: int, i: int) -> bool:
        card = self.hands[p].pop(i)
        self.coins[p] -= self._cost[card]
        self.played[p][card] += 1
        self._note_left_hand(p, card)
        k = H_OPS if self._is_op[card] else H_DEPLOYED
        self.history_turn[p][k] += 1
        self.history_game[p][k] += 1
        if self._is_op[card]:
            self.discard[p][card] += 1
            self._fire_play(card, p)
            self._drain()
            return False
        u = self._new_unit(card, p)
        self.backline[p].append(u)
        return self._has_effects and self._unit_event(T_DEPLOY, u)

    def _move(self, p: int, j: int) -> bool:
        u = self.backline[p].pop(j)
        self.coins[p] -= u.move_cost
        u.moved = True
        if u.smokescreen or u.base_traits & SMOKESCREEN_BIT:
            self._lose_smokescreen(u)
        self.frontline.append(u)
        self.front_owner = p
        return self._has_effects and self._unit_event(T_MOVE, u)

    def _unit_event(self, trig: int, u: Unit) -> bool:
        """Fire on_deploy / on_move of u and resolve its listeners. The static contributions are recomputed
        first, so event filters see the unit's current values (SPEC 2.13: checked when the event fires).
        That recompute replaces the one the queue would run before its first instance, or the after-action
        one when nothing listens (firing changes nothing it reads), so the number of recomputes is unchanged.
        Returns True when the caller's after-action recompute is redundant."""
        statics = self._any_static
        if statics:
            self._refresh_statics()
        self._fire_unit(trig, u, None)
        if self.queue:
            self._drain(statics)
            return False
        return statics

    @staticmethod
    def _lose_smokescreen(u: Unit) -> None:
        """Moving or attacking loses smokescreen for good (also when a "turn" removal suppresses it now,
        so the expiry does not bring it back). A static grant keeps it while the source grants it."""
        u.smokescreen = bool(u.static_traits & SMOKESCREEN_BIT)
        u.base_traits &= ~SMOKESCREEN_BIT
        u.temp_traits &= ~SMOKESCREEN_BIT

    def _attack(self, p: int, a: int, t: int) -> None:
        Z = self.config.zone_capacity
        attacker = self.backline[p][a] if a < Z else self.frontline[a - Z]
        attacker.attacks += 1
        if attacker.smokescreen or attacker.base_traits & SMOKESCREEN_BIT:
            self._lose_smokescreen(attacker)
        if t == self.action_space.BASE_TARGET:
            target = None
        else:
            target = self.backline[1 - p][t] if t < Z else self.frontline[t - Z]
        if self._has_effects:
            # combat is deferred until the on_attack triggers (and their chains) have resolved, then the
            # target's on_attacked triggers (and their chains); stage 0 = on_attacked not fired yet. The
            # statics are recomputed before on_attack fires (the attacker may have lost smokescreen), in place
            # of the queue's first recompute (see _unit_event).
            statics = self._any_static
            if statics:
                self._refresh_statics()
            self._fire_unit(T_ATTACK, attacker, target)
            self._combat = (attacker.uid, -1 if target is None else target.uid, 0)
            self._drain(statics)
            return
        self._do_combat(attacker, target)

    def _resume_combat(self, c: tuple) -> None:
        attacker = self._find(c[0])
        if attacker is None:
            return  # the attacker left the board: the attack ends
        target = None
        if c[1] >= 0:
            target = self._find(c[1])
            if target is None:
                return
            if c[2] == 0 and self._attacked_any:
                self._fire_unit(T_ATTACKED, target, attacker, 0, True)
                if self.queue:  # combat follows once the on_attacked chains have resolved
                    self._combat = (c[0], c[1], 1)
                    return
                # nothing listens: combat follows at once, so the recompute schedule (and with it the
                # result of one-pass auras) does not depend on on_attacked cards elsewhere in the pool
        self._do_combat(attacker, target)

    def _do_combat(self, attacker: Unit, target: Optional[Unit]) -> None:
        if target is None:
            o = 1 - attacker.owner
            self.base_hp[o] -= attacker.atk  # no armor, no return damage
            if self.base_hp[o] <= 0:
                self._end_by_bases()
            return
        to_target, to_attacker = combat_damage(attacker, target)  # simultaneous: pre-combat values
        if target.ambush and ambush_fires(attacker, target):
            target.ambush_used = True  # spent for this turn, whatever the strike does
        target.hp -= to_target
        attacker.hp -= to_attacker
        if self._has_effects:
            damaged = []
            if to_target > 0:
                damaged.append((target, attacker, to_target))
            if to_attacker > 0:
                damaged.append((attacker, target, to_attacker))
            kills = []
            if to_target > 0 and target.hp <= 0:
                kills.append((attacker, target, to_target))
            if to_attacker > 0 and attacker.hp <= 0:
                kills.append((target, attacker, to_attacker))
            self._damage_step(damaged, None, kills)
        elif attacker.hp <= 0 or target.hp <= 0:
            self._remove_dead(None)

    def _end_by_bases(self) -> None:
        b0, b1 = self.base_hp
        self.done = True
        self._winner = DRAW if (b0 <= 0 and b1 <= 0) else (1 if b0 <= 0 else 0)
        self.queue.clear()
        self.pending = None
        self._combat = None

    def _remove_dead(self, destroyed) -> list:
        """Remove units with hp <= 0 (or uid in `destroyed`) in board order; returns them in removal order."""
        cur = self.current
        o = 1 - cur
        fo = self.front_owner
        dead = []
        zones = [self.backline[cur], self.frontline, self.backline[o]] if fo == cur else (
            [self.backline[cur], self.backline[o], self.frontline])
        for zone in zones:
            if any(u.hp <= 0 or (destroyed is not None and u.uid in destroyed) for u in zone):
                keep = []
                for u in zone:
                    if u.hp <= 0 or (destroyed is not None and u.uid in destroyed):
                        dead.append(u)
                    else:
                        keep.append(u)
                zone[:] = keep
        if fo is not None and not self.frontline:
            self.front_owner = None
        gy = self.graveyard
        for u in dead:
            gy[u.owner][u.card] += 1
        if dead:
            ht, hg = self.history_turn, self.history_game
            for u in dead:
                ht[u.owner][H_DIED] += 1
                hg[u.owner][H_DIED] += 1
            if self._dead_log is not None:
                self._dead_log += dead
        return dead

    def _damage_step(self, damaged, destroyed, kills) -> None:
        """SPEC §2.8 damage step. `damaged`: (unit, event unit, damage) of the units that took > 0 damage, in
        on_damaged order; `destroyed`: uids marked by `destroy` (or None); `kills`: (killer, victim, lethal
        damage) combat kills. Static contributions are recomputed after the removal (SPEC §2.12)."""
        if self.base_hp[0] <= 0 or self.base_hp[1] <= 0:
            self._end_by_bases()
            return
        dead = self._remove_dead(destroyed)
        if not self._has_effects:
            return
        if self._any_static:
            self._refresh_statics()
        fire = self._fire_unit
        for u, ev, dmg in damaged:
            if u.hp > 0:
                fire(T_DAMAGED, u, ev, dmg)
        for killer, victim, dmg in kills:
            fire(T_KILL, killer, victim, dmg, True)
        for u in dead:
            fire(T_DEATH, u, None)

    def _end_turn(self, p: int) -> None:
        if self._has_effects:
            self._fire_turn(T_END, p)
            if self.queue:
                self._drain()
                if self.done:
                    return
        if self._has_effects or self._scan_units:
            self._expire_and_unpin()
            if self._any_static:
                self._refresh_statics()
        self.coins[p] = 0  # unused coins are lost
        # Refresh the ending player's units now, so during the opponent's turn their flags already
        # describe what they can do on their owner's next turn (legality is unaffected).
        for u in self.backline[p]:
            u.summoned = u.moved = False
            u.attacks = 0
        if self.front_owner == p:
            for u in self.frontline:
                u.summoned = u.moved = False
                u.attacks = 0
        o = 1 - p
        # A unit the opponent gained during p's turn (an on_death summon, a summon for the enemy) was not
        # created on o's turn, so it may act on o's next turn (SPEC 2.4: only the creation turn is lost).
        # o's units cannot move or attack on p's turn, so only `summoned` can be set. (Without effects no
        # unit is created on the other player's turn.)
        if self._has_effects:
            for u in self.backline[o]:
                if u.summoned:
                    u.summoned = False
            if self.front_owner == o:
                for u in self.frontline:
                    if u.summoned:
                        u.summoned = False
        if o == self.first_player:
            if self.round >= self.config.max_rounds:
                self.done = True
                self._winner = DRAW
                return
            self.round += 1
        self._start_turn(o)

    def _expire_and_unpin(self) -> None:
        """END_TURN: every "turn" duration expires; pins whose expiry has come are lifted."""
        turn = self.turn
        for zone in (self.backline[0], self.backline[1], self.frontline):
            for u in zone:
                if (u.temp_atk or u.temp_hp or u.temp_armor or u.temp_traits or u.temp_removed
                        or u.temp_move_cost):
                    self._expire(u)
                if u.pinned and u.pin_until <= turn:
                    u.pinned = False

    @staticmethod
    def _expire(u: Unit) -> None:
        if u.temp_atk:
            if u.static_atk:
                _shift_atk(u, -u.temp_atk)
            else:
                atk = u.atk - u.temp_atk
                u.atk = atk if atk > 0 else 0
            u.temp_atk = 0
        if u.temp_hp:
            u.max_hp -= u.temp_hp
            u.hp = max(1, min(u.hp, u.max_hp))  # expiry never kills
            u.temp_hp = 0
        if u.temp_armor:
            armor = u.armor - u.temp_armor
            u.armor = armor if armor > 0 else 0
            u.temp_armor = 0
        if u.temp_move_cost:
            _shift_move_cost(u, -u.temp_move_cost)
            u.temp_move_cost = 0
        m = u.temp_traits | u.temp_removed
        if m:
            b = u.base_traits | u.static_traits  # the traits it would have had
            for name, bit in _BOOL_TRAIT_LIST:
                if m & bit:
                    setattr(u, name, bool(b & bit))
            u.temp_traits = u.temp_removed = 0

    def _confirm(self, p: int) -> None:
        """CONFIRM: draw k replacements, shuffle the marked cards into the deck, re-sort the hand (SPEC §2.2)."""
        marks = sorted(self.mulligan_marks)
        hand, deck = self.hands[p], self.deck_cards[p]
        k = min(len(marks), len(deck))
        if k:
            marks = marks[:k]  # only as many as the deck can replace (never happens with 40-card decks)
            drawn = [deck.pop() for _ in range(k)]
            marked = set(marks)
            deck.extend(hand[i] for i in marks)
            self.rng.shuffle(deck)
            self.hands[p] = sorted([c for i, c in enumerate(hand) if i not in marked] + drawn)
        self.mulligan_marks = set()
        self.mulligan_done[p] = True
        if p == self.first_player:
            self.current = 1 - p
        else:
            self._start_turn(self.first_player)

    def _choose(self, t: int) -> None:
        inst, _, amounts = self.pending
        self.pending = None
        self.phase = MAIN
        eff = inst[0]  # the effect, or its else body when the condition failed
        target = self._slot_target(t, inst[2])
        batch = inst[7]
        if batch is None:
            if eff.target_condition is None:
                self._apply(eff, inst, [target], amounts)
            else:
                self._hit(eff, inst, [target], amounts, False)
        else:
            self._dead_log = log = []
            hit = self._hit(eff, inst, [target], amounts, True)
            self._dead_log = None
            batch.append(_record(hit, log))
        self._drain()

    # ------------------------------------------------------------------ effects: listeners and the queue
    def _board_zones(self) -> list:
        """(zone list, owner, is_frontline) in board order: turn player's backline, frontline (if theirs),
        opponent's backline, frontline (if theirs)."""
        cur = self.current
        o = 1 - cur
        fo = self.front_owner
        if fo == cur:
            return [(self.backline[cur], cur, False), (self.frontline, cur, True), (self.backline[o], o, False)]
        if fo == o:
            return [(self.backline[cur], cur, False), (self.backline[o], o, False), (self.frontline, o, True)]
        return [(self.backline[cur], cur, False), (self.backline[o], o, False)]

    def _board_order(self) -> list:
        cur = self.current
        o = 1 - cur
        fo = self.front_owner
        if fo == cur:
            return self.backline[cur] + self.frontline + self.backline[o]
        if fo == o:
            return self.backline[cur] + self.backline[o] + self.frontline
        return self.backline[cur] + self.backline[o]

    def _find(self, uid: int) -> Optional[Unit]:
        if uid < 0:
            return None
        for zone in (self.backline[0], self.backline[1], self.frontline):
            for u in zone:
                if u.uid == uid:
                    return u
        return None

    # instance = (effect, card index, controller, source uid, source unit, event uid, event unit, batch, damage):
    # the unit references double as last-known views (a unit object is never updated after it leaves the
    # board); `batch` is the clause batch shared by the card's instances of one event (a list of records,
    # SPEC §2.8b; None when no clause of the card reads `prev`); `damage` is the event's damage
    # (on_damaged: taken, on_kill: the lethal hit), else 0.
    def _fire_unit(self, trig: int, x: Unit, self_ev: Optional[Unit], dmg: int = 0, shared_ev: bool = False) -> None:
        """Enqueue the listeners of an event about unit x: x's own effects (event unit `self_ev`), then
        watchers on the board in board order whose scope matches x's side and whose event_filter accepts
        their event unit: x itself, or `self_ev` with `shared_ev` (on_kill: the victim, on_attacked: the
        attacker; SPEC §1.2b, §2.13)."""
        q = self.queue
        effs = self._self_eff[x.card][trig]
        ev_uid = -1 if self_ev is None else self_ev.uid
        if effs:
            batch = [] if self._self_batch[x.card][trig] else None
            for e in effs:
                q.append((e, x.card, x.owner, x.uid, x, ev_uid, self_ev, batch, dmg))
        if self._watch_any[trig]:
            watch, wbatch = self._watch_eff, self._watch_batch
            xo, xuid = x.owner, x.uid
            wev, wev_uid = (self_ev, ev_uid) if shared_ev else (x, xuid)
            for w in self._board_order():
                ws = watch[w.card][trig]
                if ws and w.uid != xuid:
                    same = w.owner == xo
                    batch = [] if wbatch[w.card][trig] else None
                    for e, scope in ws:
                        if (scope == S_ANY or (scope == S_FRIENDLY) == same) and (
                                e.event_filter is None or self._matches(wev, e.event_filter, w.uid)):
                            q.append((e, w.card, w.owner, w.uid, w, wev_uid, wev, batch, dmg))

    def _fire_play(self, card: int, p: int) -> None:
        """An operation of player p is played: its own on_play effects, then watching units."""
        q = self.queue
        effs = self._self_eff[card][T_PLAY]
        if effs:
            batch = [] if self._self_batch[card][T_PLAY] else None
            for e in effs:
                q.append((e, card, p, -1, None, -1, None, batch, 0))
        if self._watch_any[T_PLAY]:
            watch, wbatch = self._watch_eff, self._watch_batch
            for w in self._board_order():
                ws = watch[w.card][T_PLAY]
                if ws:
                    batch = [] if wbatch[w.card][T_PLAY] else None
                    for e, scope in ws:
                        if (scope == S_ANY or (scope == S_FRIENDLY) == (w.owner == p)) and (
                                e.event_filter is None or self._card_matches(card, e.event_filter)):
                            q.append((e, w.card, w.owner, w.uid, w, -1, None, batch, 0))

    def _fire_turn(self, trig: int, p: int) -> None:
        """start_of_turn / end_of_turn of player p: matching effects of all units in board order."""
        if not self._watch_any[trig]:
            return
        q = self.queue
        watch, wbatch = self._watch_eff, self._watch_batch
        for w in self._board_order():
            ws = watch[w.card][trig]
            if ws:
                batch = [] if wbatch[w.card][trig] else None
                for e, scope in ws:
                    if scope == S_ANY or (scope == S_FRIENDLY) == (w.owner == p):
                        q.append((e, w.card, w.owner, w.uid, w, -1, None, batch, 0))

    def _drain(self, fresh: bool = False) -> None:
        """Resolve the queue (FIFO) until it is empty, the game is over or a choice is pending; then run
        a deferred attack (on_attack, then on_attacked triggers resolve before combat). Loop guard: SPEC
        §2.8. Static contributions are recomputed before every instance and before combat (SPEC §2.12);
        `fresh`: the caller has just recomputed them (when its event fired), so the first one is skipped."""
        q = self.queue
        limit = self.config.max_effect_events
        statics = self._any_static
        while not self.done and self.pending is None:
            if q:
                if self._events >= limit:
                    q.clear()
                    if not self._tripped:
                        self._tripped = True
                        self.guard_trips += 1
                    continue
                self._events += 1
                if statics:
                    if fresh:
                        fresh = False
                    else:
                        self._refresh_statics()
                self._resolve(q.pop(0))
            elif self._combat is not None:
                c = self._combat
                self._combat = None
                if statics:
                    if fresh:
                        fresh = False
                    else:
                        self._refresh_statics()
                self._resume_combat(c)
            else:
                break
        if self.pending is None and self.phase == CHOICE:
            self.phase = MAIN

    def _resolve(self, inst: tuple) -> None:
        """SPEC §2.8 resolution: the condition (else body when it fails), then `repeat` x (selection, amounts,
        action, damage step); `chosen` opens a choice instead."""
        eff = inst[0]
        if eff.condition and not self._conditions_hold(eff.condition, inst):
            alt = eff.else_
            if alt is None:
                if inst[7] is not None:
                    inst[7].append(_NO_RECORD)
                return
            self._run(alt, (alt,) + inst[1:], 1)  # the else body runs once
            return
        tgt = eff.target
        if inst[7] is None and eff.repeat == 1 and eff.target_condition is None and tgt.select != "chosen":
            targets = self._select_targets(tgt, inst)  # the common case, inlined
            if targets:
                self._apply(eff, inst, targets, self._effect_amounts(eff, inst))
            return
        self._run(eff, inst, eff.repeat)

    def _run(self, eff, inst: tuple, reps: int) -> None:
        tgt = eff.target
        batch = inst[7]
        if tgt.select == "chosen":
            opts = self._choice_options(tgt, inst)
            if not opts:
                if batch is not None:
                    batch.append(_NO_RECORD)
                return  # fizzle
            self.pending = (inst, tuple(opts), self._effect_amounts(eff, inst))
            self.phase = CHOICE
            return
        if batch is None:
            for _ in range(reps):
                targets = self._select_targets(tgt, inst)
                if targets:
                    self._hit(eff, inst, targets, None, False)
                    if self.done:
                        return
            return
        self._dead_log = log = []
        hit = []
        for _ in range(reps):
            targets = self._select_targets(tgt, inst)
            if targets:
                hit += self._hit(eff, inst, targets, None, True)
                if self.done:
                    break
        self._dead_log = None
        batch.append(_record(hit, log))

    def _hit(self, eff, inst: tuple, targets: list, amounts, record: bool) -> Optional[list]:
        """Apply `eff` to its selected targets. With a target_condition, matching targets get the action and
        the others the else body (its own target, or those targets). With `record`, returns the targets that
        received something, tagged for the batch record (players as -1 - p).

        The split is still one effect (SPEC 2.8): both bodies' targets and amounts are evaluated first, both
        actions are applied, then a single damage step runs for their damage and destroys together."""
        tc = eff.target_condition
        if tc is None:
            self._apply(eff, inst, targets, self._effect_amounts(eff, inst) if amounts is None else amounts)
            return _tagged(targets, eff.target.kind) if record else None
        src = inst[3]
        yes, no = [], []
        for t in targets:
            (yes if type(t) is int or self._matches(t, tc, src) else no).append(t)
        alt = eff.else_
        alt_t = ()
        if no and alt is not None:
            alt_t = self._select_targets(alt.target, (alt,) + inst[1:]) if alt.own_target else no
        if yes and amounts is None:
            amounts = self._effect_amounts(eff, inst)
        alt_amounts = self._effect_amounts(alt, inst) if alt_t else None
        self._defer = steps = []  # deferred damage steps: (damaged, destroyed uids)
        try:
            if yes:
                self._apply(eff, inst, yes, amounts)
            if alt_t:
                self._apply(alt, inst, alt_t, alt_amounts)
        finally:
            self._defer = None
        if steps:
            if len(steps) == 1:
                damaged, destroyed = steps[0]
            else:
                (d1, x1), (d2, x2) = steps
                damaged = _merge_damaged(targets, d1, d2)
                destroyed = x1 if x2 is None else (x2 if x1 is None else x1 | x2)
            self._damage_step(damaged, destroyed, ())
        if not record:
            return None
        out = _tagged(yes, eff.target.kind)
        if alt_t:
            out += _tagged(alt_t, alt.target.kind)
        return out

    # ------------------------------------------------------------------ effects: static (SPEC §2.12)
    def _refresh_statics(self) -> None:
        """Recompute every unit's static contributions from scratch: each static effect of each unit on the
        board whose condition holds (else: its else body) adds to its targets. Conditions and filters read
        the state before this recompute; all contributions are then applied at once. Skipped when no
        static source is on the board and no unit carries contributions."""
        se = self._static_eff
        zones = (self.backline[0], self.backline[1], self.frontline)
        sources = [u for z in zones for u in z if se[u.card]]
        if not sources and not self._static_on:
            return
        contrib: dict = {}
        for s in sources:
            for e in se[s.card]:
                inst = (e, s.card, s.owner, s.uid, s, -1, None, None, 0)
                body = e
                if e.condition and not self._conditions_hold(e.condition, inst):
                    body = e.else_
                    if body is None:
                        continue
                targets = self._select_targets(body.target, inst)
                if not targets:
                    continue
                tc = body.target_condition
                if tc is None:
                    self._add_static(contrib, body, targets)
                    continue
                yes, no = [], []
                for u in targets:
                    (yes if self._matches(u, tc, s.uid) else no).append(u)
                if yes:
                    self._add_static(contrib, body, yes)
                alt = body.else_
                if no and alt is not None:
                    alt_t = self._select_targets(alt.target, inst) if alt.own_target else no
                    if alt_t:
                        self._add_static(contrib, alt, alt_t)
        on = False
        for z in zones:
            for u in z:
                c = contrib.get(id(u))
                if c is None:
                    if not (u.static_atk or u.static_hp or u.static_move_cost or u.static_traits):
                        continue
                    _set_static(u, 0, 0, 0, 0)
                else:
                    _set_static(u, c[0], c[1], c[2], c[3])
                if u.static_atk or u.static_hp or u.static_move_cost or u.static_traits:
                    on = True
        self._static_on = on

    @staticmethod
    def _add_static(contrib: dict, body, targets: list) -> None:
        """Accumulate one static body's contribution ([atk, hp, move_cost, trait bits]) on its targets."""
        if body.action == "buff":
            a, h, m, t = body.atk.value, body.hp.value, body.move_cost, 0
        else:
            a = h = m = 0
            t = TRAIT_BITS[body.trait[0]]
        for u in targets:
            c = contrib.get(id(u))
            if c is None:
                contrib[id(u)] = [a, h, m, t]
            else:
                c[0] += a
                c[1] += h
                c[2] += m
                c[3] |= t

    # ------------------------------------------------------------------ effects: filters, conditions, amounts
    def _matches(self, u: Unit, f, src_uid: int) -> bool:
        if f is None:
            return True
        if f.natures and u.nature not in f.natures:
            return False
        if f.traits or f.not_traits:
            m = u.trait_mask()
            if (m & f.traits) != f.traits or (m & f.not_traits):
                return False
        if f.damaged is not None and (u.hp < u.max_hp) != f.damaged:
            return False
        if f.min_cost is not None or f.max_cost is not None:
            c = self._cost[u.card]
            if (f.min_cost is not None and c < f.min_cost) or (f.max_cost is not None and c > f.max_cost):
                return False
        if (f.min_atk is not None and u.atk < f.min_atk) or (f.max_atk is not None and u.atk > f.max_atk):
            return False
        if (f.min_hp is not None and u.hp < f.min_hp) or (f.max_hp is not None and u.hp > f.max_hp):
            return False
        if f.token is not None and u.token != f.token:
            return False
        if f.other and u.uid == src_uid:
            return False
        if f.tags and self._tags[u.card].isdisjoint(f.tags):
            return False
        if f.not_tags and not self._tags[u.card].isdisjoint(f.not_tags):
            return False
        if f.pinned is not None and u.pinned != f.pinned:
            return False
        return True

    def _card_matches(self, card: int, f) -> bool:
        """event_filter of an on_play watcher, on the played card (cost and tags)."""
        c = self._cost[card]
        if (f.min_cost is not None and c < f.min_cost) or (f.max_cost is not None and c > f.max_cost):
            return False
        tags = self._tags[card]
        if f.tags and tags.isdisjoint(f.tags):
            return False
        if f.not_tags and not tags.isdisjoint(f.not_tags):
            return False
        return True

    def _units(self, side: str, zone: str, flt, ctrl: int, src_uid: int) -> list:
        """Units in board order whose owner matches `side` (relative to ctrl), in `zone`, passing `flt`."""
        out = []
        for z, owner, front in self._board_zones():
            if not z or (zone == "backline" and front) or (zone == "frontline" and not front):
                continue
            if side != "any" and (side == "friendly") != (owner == ctrl):
                continue
            if flt is None:
                out += z
            else:
                out += [u for u in z if self._matches(u, flt, src_uid)]
        return out

    def _conditions_hold(self, conds: tuple, inst: tuple) -> bool:
        ctrl = inst[2]
        for c in conds:
            t = c.type
            if t == "control":
                n = len(self._units(c.side, c.zone, c.filter, ctrl, inst[3]))
                if n < c.min or (c.max is not None and n > c.max):
                    return False
            elif t == "frontline":
                fo = self.front_owner
                want = None if c.owner == "none" else (ctrl if c.owner == "friendly" else 1 - ctrl)
                if fo != want:
                    return False
            elif t == "source_zone":
                u = self._find(inst[3])
                if u is None:
                    return False  # the source is not on the board
                in_front = any(x is u for x in self.frontline)
                if in_front != (c.zone == "frontline"):
                    return False
            elif t == "base_hp" or t == "hand_size":
                p = ctrl if c.side == "friendly" else 1 - ctrl
                v = self.base_hp[p] if t == "base_hp" else len(self.hands[p])
                if (c.min is not None and v < c.min) or (c.max is not None and v > c.max):
                    return False
            elif t == "turn":
                if (self.current == ctrl) != (c.whose == "own"):
                    return False
            elif t == "prev":
                if not self._prev_holds(c, inst):
                    return False
            elif t == "compare":
                a, b = self._amount(c.left, inst), self._amount(c.right, inst)
                op = c.op
                if not (a > b if op == ">" else a >= b if op == ">=" else a < b if op == "<" else
                        a <= b if op == "<=" else a == b):
                    return False
            else:  # history
                counts = self.history_turn if c.window == "turn" else self.history_game
                k = HISTORY_INDEX[c.event]
                side = c.side
                v = (counts[ctrl][k] if side == "friendly" else counts[1 - ctrl][k] if side == "enemy"
                     else counts[0][k] + counts[1][k])
                if v < c.min or (c.max is not None and v > c.max):
                    return False
        return True

    def _prev_holds(self, c, inst: tuple) -> bool:
        """{"type": "prev"}: killed = whether any prev unit target was killed by that clause; filter = some
        prev unit target matches (current values on the board, else last-known); no key = it hit something."""
        batch = inst[7]
        targets, killed = batch[-1] if batch else _NO_RECORD
        if c.killed is None and c.filter is None:
            return bool(targets)
        if c.killed is not None and killed != c.killed:
            return False
        if c.filter is not None:
            src = inst[3]
            if not any(type(t) is not int and self._matches(t, c.filter, src) for t in targets):
                return False
        return True

    @staticmethod
    def _first_prev_unit(inst: tuple) -> Optional[Unit]:
        batch = inst[7]
        if batch:
            for t in batch[-1][0]:
                if type(t) is not int:
                    return t
        return None

    def _amount(self, am, inst: tuple) -> int:
        kind = am.kind
        if kind == "literal" or kind == "full":
            return am.value
        if kind == "stat":
            of = am.of
            u = inst[4] if of == "self" else (inst[6] if of == "event" else self._first_prev_unit(inst))
            stat = am.stat
            if u is None:  # an operation's own cost, or no event / prev unit
                v = self._cost[inst[1]] if (of == "self" and stat == "cost") else 0
            elif stat == "atk":
                v = u.atk
            elif stat == "hp":
                v = u.hp
            elif stat == "max_hp":
                v = u.max_hp
            elif stat == "move_cost":
                v = u.move_cost
            else:
                v = self._cost[u.card]
        elif kind == "event":
            v = inst[8]
        else:
            ctrl = inst[2]
            what = am.count
            if what == "units":
                v = len(self._units(am.side, am.zone, am.filter, ctrl, inst[3]))
            else:
                side = am.side
                players = (ctrl,) if side == "friendly" else ((1 - ctrl,) if side == "enemy" else (ctrl, 1 - ctrl))
                if what == "hand":
                    v = sum(len(self.hands[p]) for p in players)
                elif what == "coins":
                    v = sum(self.coins[p] for p in players)
                elif what == "deck":
                    v = sum(len(self.deck_cards[p]) for p in players)
                else:  # base_hp
                    v = sum(self.base_hp[p] for p in players)
        v = am.times * v + am.plus
        return v if v > 0 else 0

    def _effect_amounts(self, eff, inst: tuple) -> tuple:
        a = 0 if eff.amount is None else self._amount(eff.amount, inst)
        if eff.atk is not None:
            return (a, self._amount(eff.atk, inst), self._amount(eff.hp, inst))
        return (a, 0, 0)

    # ------------------------------------------------------------------ effects: targets
    def _select_targets(self, tgt, inst: tuple) -> list:
        sel = tgt.select
        if sel == "self" or sel == "event":
            u = self._find(inst[3] if sel == "self" else inst[5])
            if u is None or (tgt.filter is not None and not self._matches(u, tgt.filter, inst[3])):
                return []
            return [u]
        if sel == "prev":
            return self._prev_targets(tgt, inst)
        if sel == "adjacent":
            return self._adjacent(tgt, inst)
        ctrl = inst[2]
        kind = tgt.kind
        cands = []
        if kind == "unit" or kind == "unit_or_base":
            cands = self._units(tgt.side, tgt.zone, tgt.filter, ctrl, inst[3])
        if kind != "unit":  # bases / players: controller first, then the opponent
            if tgt.side != "enemy":
                cands.append(ctrl)
            if tgt.side != "friendly":
                cands.append(1 - ctrl)
        if sel == "random" and len(cands) > tgt.count:
            cands = self.rng.sample(cands, tgt.count)
        return cands

    def _prev_targets(self, tgt, inst: tuple) -> list:
        """The previous clause's targets (SPEC §1.2b prev): units still on the board (passing the filter),
        bases or players, as the target kind asks."""
        batch = inst[7]
        if not batch:
            return []
        kind, flt, src = tgt.kind, tgt.filter, inst[3]
        units = kind == "unit" or kind == "unit_or_base"
        out = []
        for t in batch[-1][0]:
            if type(t) is int:
                if t >= 0:
                    if kind == "base" or kind == "unit_or_base":
                        out.append(t)
                elif kind == "player":
                    out.append(-1 - t)
            elif units and self._find(t.uid) is t and (flt is None or self._matches(t, flt, src)):
                out.append(t)
        return out

    def _adjacent(self, tgt, inst: tuple) -> list:
        """The units next to the reference unit (self, event or the first prev unit still on the board) in its
        zone: slot - 1 (left), slot + 1 (right). The reference must be on the board."""
        of = tgt.of
        if of == "self":
            ref = self._find(inst[3])
        elif of == "event":
            ref = self._find(inst[5])
        else:
            ref = None
            batch = inst[7]
            if batch:
                for t in batch[-1][0]:
                    if type(t) is not int and self._find(t.uid) is t:
                        ref = t
                        break
        if ref is None:
            return []
        side = tgt.side
        if side and side != "any" and (side == "friendly") != (ref.owner == inst[2]):
            return []  # a zone has one owner, so its neighbours share the reference's side
        zone = self.frontline if any(x is ref for x in self.frontline) else self.backline[ref.owner]
        i = next(k for k, x in enumerate(zone) if x is ref)
        pos = tgt.position
        out = []
        if pos != "right" and i > 0:
            out.append(zone[i - 1])
        if pos != "left" and i + 1 < len(zone):
            out.append(zone[i + 1])
        flt = tgt.filter
        if flt is not None:
            out = [u for u in out if self._matches(u, flt, inst[3])]
        return out

    def _choice_options(self, tgt, inst: tuple) -> list:
        """CHOOSE slots (ascending) for a `chosen` target of controller inst[2] (SPEC §2.9)."""
        p = inst[2]
        o = 1 - p
        Z = self.config.zone_capacity
        side, zone, flt, kind = tgt.side, tgt.zone, tgt.filter, tgt.kind
        src = inst[3]
        m = self._matches
        opts = []
        units = kind != "base"
        bases = kind != "unit"
        if units and side != "friendly" and zone != "frontline":
            opts += [j for j, u in enumerate(self.backline[o]) if flt is None or m(u, flt, src)]
        fo = self.front_owner
        if units and zone != "backline" and fo is not None and (side == "any" or (side == "friendly") == (fo == p)):
            opts += [Z + j for j, u in enumerate(self.frontline) if flt is None or m(u, flt, src)]
        if bases and side != "friendly":
            opts.append(2 * Z)
        if units and side != "enemy" and zone != "frontline":
            opts += [2 * Z + 1 + j for j, u in enumerate(self.backline[p]) if flt is None or m(u, flt, src)]
        if bases and side != "enemy":
            opts.append(3 * Z + 1)
        return opts

    def _slot_target(self, t: int, p: int):
        Z = self.config.zone_capacity
        if t < Z:
            return self.backline[1 - p][t]
        if t < 2 * Z:
            return self.frontline[t - Z]
        if t == 2 * Z:
            return 1 - p
        if t <= 3 * Z:
            return self.backline[p][t - 2 * Z - 1]
        return p

    # ------------------------------------------------------------------ effects: actions (SPEC §2.10)
    def _leave_board(self, u: Unit) -> None:
        front = self.frontline
        for i, x in enumerate(front):
            if x is u:
                del front[i]
                if not front:
                    self.front_owner = None
                return
        back = self.backline[u.owner]
        for i, x in enumerate(back):
            if x is u:
                del back[i]
                return

    def _to_hand(self, p: int, card: int) -> None:
        """A card returns to p's hand and becomes known to the opponent (a full hand burns it)."""
        hand = self.hands[p]
        if len(hand) >= self.config.max_hand_size:
            self.burned[p] += 1
        else:
            insort(hand, card)
            self.known_hand[p][card] += 1

    def _apply(self, eff, inst: tuple, targets: list, amounts: tuple) -> None:
        """Apply one action to its targets. damage / destroy end with the damage step, or, while `_defer` is
        a list (a target_condition split, see _hit), append (damaged, destroyed uids) to it instead."""
        act = eff.action
        n = amounts[0]
        if act == "damage":
            src = inst[4]
            damaged = []
            for t in targets:
                if type(t) is int:
                    self.base_hp[t] -= n
                elif n > 0 and not t.immune:
                    t.hp -= n  # effect damage ignores armor
                    damaged.append((t, src, n))
            defer = self._defer
            if defer is None:
                self._damage_step(damaged, None, ())
            else:
                defer.append((damaged, None))
        elif act == "destroy":
            defer = self._defer
            if defer is None:
                self._damage_step((), {u.uid for u in targets}, ())
            else:
                defer.append(((), {u.uid for u in targets}))
        elif act == "heal":
            cap = self.config.base_hp
            top = max(cap, UNCAPPED_BASE_HP) if eff.uncapped else cap
            for t in targets:
                if type(t) is int:
                    hp = self.base_hp[t]
                    self.base_hp[t] = max(hp, cap if n == FULL else min(top, hp + n))
                else:
                    t.hp = max(t.hp, t.max_hp if n == FULL else min(t.max_hp, t.hp + n))
        elif act == "buff":
            a, h, mc = amounts[1], amounts[2], eff.move_cost
            turn = eff.duration == "turn"
            for u in targets:
                if a:
                    d = _shift_atk(u, a)
                    if turn:
                        u.temp_atk += d  # the applied change, so expiry restores the old value
                if h:
                    u.max_hp += h
                    u.hp += h
                    if turn:
                        u.temp_hp += h
                if mc:
                    d = _shift_move_cost(u, mc)
                    if turn:
                        u.temp_move_cost += d
        elif act == "draw":
            for p in targets:
                self._draw(p, n)
        elif act == "gain_coins":
            for p in targets:
                c = self.coins[p] + n
                self.coins[p] = c if c > 0 else 0  # a negative amount never takes coins below 0
        elif act == "increase_max_coins":
            for p in targets:
                self.coin_bonus[p] += n
        elif act == "add_trait":
            name = eff.trait[0]
            turn = eff.duration == "turn"
            for u in targets:
                if name == "armor":
                    u.armor += n
                    if turn:
                        u.temp_armor += n
                    else:
                        u.base_traits |= ARMOR_BIT
                else:
                    setattr(u, name, True)
                    bit = _BOOL_TRAIT_BITS[name]
                    if turn:
                        u.temp_traits |= bit
                    else:
                        u.base_traits |= bit
                    u.temp_removed &= ~bit
        elif act == "remove_trait":
            turn = eff.duration == "turn"
            for u in targets:
                for name in eff.trait:
                    if name == "armor":
                        if turn:
                            u.temp_armor -= u.armor  # keeps armor - temp_armor = the permanent armor
                        else:
                            u.temp_armor = 0
                            u.base_traits &= ~ARMOR_BIT
                        u.armor = 0
                    else:
                        bit = _BOOL_TRAIT_BITS[name]
                        setattr(u, name, bool(u.static_traits & bit))  # a static grant stays
                        u.temp_traits &= ~bit
                        if turn:
                            u.temp_removed |= bit
                        else:
                            u.base_traits &= ~bit
        elif act == "summon":
            Z = self.config.zone_capacity
            card = eff.card
            for p in targets:
                back = self.backline[p]
                for _ in range(n):
                    if len(back) >= Z:
                        break
                    back.append(self._new_unit(card, p))
        elif act == "return_to_hand":
            for u in targets:
                self._leave_board(u)
                self._to_hand(u.owner, u.card)
        elif act == "pin":
            for u in targets:
                u.pinned = True
                u.pin_until = self.turn + (1 if self.current != u.owner else 2)
        elif act == "discard":
            for p in targets:
                hand = self.hands[p]
                if not hand or n <= 0:
                    continue
                idx = list(range(len(hand))) if len(hand) <= n else self.rng.sample(range(len(hand)), n)
                cards = [hand[i] for i in idx]
                for i in sorted(idx, reverse=True):
                    del hand[i]
                for c in cards:
                    self._note_left_hand(p, c)
                    self.discard[p][c] += 1
        elif act == "add_card":
            card = eff.card
            for p in targets:
                for _ in range(n):
                    self._to_hand(p, card)
        elif act == "retreat":
            Z = self.config.zone_capacity
            for u in targets:
                in_front = any(x is u for x in self.frontline)
                self._leave_board(u)
                back = self.backline[u.owner]
                if in_front and len(back) < Z:
                    back.append(u)
                else:
                    self._to_hand(u.owner, u.card)
        else:  # pragma: no cover - the loader only accepts known actions
            raise ValueError(f"unknown action {act!r}")

    # ------------------------------------------------------------------ views
    def _decklist_counts(self, p: int) -> tuple:
        dl = self.decklists[p]
        cache = self.__dict__.get("_dlc") or ((None, ()), (None, ()))  # ((decklist, counts), ...) per seat
        entry = cache[p]
        if entry[0] is not dl:
            counts = [0] * len(self._card_defs)
            for c in dl:
                counts[c] += 1
            entry = (dl, tuple(counts))
            self._dlc = (entry, cache[1]) if p == 0 else (cache[0], entry)
        return entry[1]

    def _pending_view(self) -> Optional[PendingView]:
        if self.pending is None:
            return None
        inst, opts, amounts = self.pending
        eff = inst[0]
        t = eff.target
        return _new_tuple(PendingView, (inst[1], eff.index, eff.action, amounts[0], t.select, t.side, t.kind,
                                        amounts[1], amounts[2], self._choice_previews(inst, opts, amounts)))

    def _choice_previews(self, inst: tuple, opts: tuple, amounts: tuple) -> tuple:
        """PendingView.previews (SPEC §5): per CHOOSE slot, what CHOOSE(t) would do to the option itself, with
        the body that would apply to it (a target that fails `target_condition` gets the `else` body, which
        reuses that target or hits its own targets; a random own target is not previewed). Reads the state
        only: no RNG draw, no change."""
        out = [_NO_PREVIEW] * self.action_space.n_choose
        eff = inst[0]
        p = inst[2]
        tc = eff.target_condition
        alt = eff.else_
        alt_amounts = alt_hits = None
        for t in opts:
            target = self._slot_target(t, p)
            body, am = eff, amounts
            if tc is not None and type(target) is not int and not self._matches(target, tc, inst[3]):
                if alt is None:
                    continue  # a non-matching target without an else body receives nothing
                if alt.own_target:
                    if alt.target.select == "random":
                        continue
                    if alt_hits is None:
                        alt_hits = self._select_targets(alt.target, (alt,) + inst[1:])
                    if not any(x is target for x in alt_hits):
                        continue
                if alt_amounts is None:
                    alt_amounts = self._effect_amounts(alt, inst)
                body, am = alt, alt_amounts
            out[t] = self._preview(body, am, target)
        return tuple(out)

    def _preview(self, body, amounts: tuple, target) -> tuple:
        """(kills, dealt, healed, other) of one action body applied to one target (a unit, or a base as its
        player index), following `_apply` (SPEC §2.10): damage ignores armor, immune units take none; destroy
        always kills; heal never lowers hp; buff = the applied atk change + the hp change; 1 for any other
        action."""
        act = body.action
        n = amounts[0]
        if type(target) is int:  # a base
            hp = self.base_hp[target]
            if act == "damage":
                return (1 if 0 < n and hp <= n else 0, n, 0, 0)
            if act == "heal":
                cap = self.config.base_hp
                top = max(cap, UNCAPPED_BASE_HP) if body.uncapped else cap
                new = cap if n == FULL else min(top, hp + n)
                return (0, 0, new - hp if new > hp else 0, 0)
            return (0, 0, 0, 1)
        if act == "damage":
            d = n if (n > 0 and not target.immune) else 0
            return (1 if 0 < d and target.hp <= d else 0, d, 0, 0)
        if act == "destroy":
            return (1, 0, 0, 0)
        if act == "heal":
            hp = target.hp
            new = target.max_hp if n == FULL else min(target.max_hp, hp + n)
            return (0, 0, new - hp if new > hp else 0, 0)
        if act == "buff":
            da = 0
            if amounts[1]:
                c = target.copy()
                _shift_atk(c, amounts[1])
                da = c.atk - target.atk
            return (0, 0, 0, da + amounts[2])
        return (0, 0, 0, 1)

    def observe(self, player: int) -> Observation:
        if not hasattr(self, "rng"):
            raise RuntimeError("call reset(seed) before observe()")
        player = _as_index(player, "player")
        if player not in (0, 1):
            raise ValueError(f"player must be 0 or 1, got {player}")
        o = 1 - player
        fo = self.front_owner
        done = self.done
        if done:
            w = self._winner
            result = 0 if w == DRAW else (1 if w == player else -1)
        else:
            result = 0
        phase = self.phase
        hand = self.hands[player]
        marks = ()
        if phase == MULLIGAN and not done and self.current == player:
            m = self.mulligan_marks
            marks = tuple([i in m for i in range(len(hand))])
        deck_counts = [0] * len(self._card_defs)
        for c in self.deck_cards[player]:
            deck_counts[c] += 1
        turn, cur = self.turn, self.current
        ht, hg = self.history_turn, self.history_game
        return _new_tuple(Observation, (
            player, (not done) and cur == player, self.first_player == player, self.round,
            self.deck_ids[player], self.coins[player], self.coins[o], self.base_hp[player], self.base_hp[o],
            tuple(hand), len(self.hands[o]), len(self.deck_cards[player]), len(self.deck_cards[o]),
            tuple(self.played[player]), tuple(self.played[o]),
            tuple([u.view(turn, cur) for u in self.backline[player]]),
            tuple([u.view(turn, cur) for u in self.backline[o]]),
            tuple([u.view(turn, cur) for u in self.frontline]),
            0 if fo is None else (1 if fo == player else -1), done, result,
            phase, marks, self._pending_view() if self.pending is not None else None, turn,
            self.coin_bonus[player], self.coin_bonus[o], self._decklist_counts(player), tuple(deck_counts),
            tuple(self.known_hand[player]), tuple(self.known_hand[o]), tuple(self.revealed[o]),
            tuple(self.discard[player]), tuple(self.discard[o]), tuple(self.graveyard[player]),
            tuple(self.graveyard[o]), self.burned[player], self.burned[o],
            (*ht[player], *hg[player]), (*ht[o], *hg[o]),
        ))

    # ------------------------------------------------------------------ copies
    @staticmethod
    def _copy_unit_ref(u: Unit, memo: dict) -> Unit:
        c = memo.get(id(u))
        if c is None:
            c = memo[id(u)] = u.copy()
        return c

    @staticmethod
    def _copy_inst(inst: tuple, memo: dict) -> tuple:
        """An instance with its unit references mapped to the clone's units (off-board ones copied once) and
        its clause batch copied once (instances of one batch keep sharing it)."""
        cu = Game._copy_unit_ref
        src, ev, batch = inst[4], inst[6], inst[7]
        if src is not None:
            src = cu(src, memo)
        if ev is not None:
            ev = cu(ev, memo)
        if batch is not None:
            b = memo.get(id(batch))
            if b is None:
                b = memo[id(batch)] = [(tuple(t if type(t) is int else cu(t, memo) for t in rec[0]), rec[1])
                                       for rec in batch]
            batch = b
        return (inst[0], inst[1], inst[2], inst[3], src, inst[5], ev, batch, inst[8])

    def clone(self) -> "Game":
        """Independent copy (including RNG state). All attributes are first shared (scalars and per-pool
        tables), then every known mutable field is copied by hand; unknown mutable attributes are
        deep-copied, so state added later is never shared between clones."""
        cls = self.__class__
        g = cls.__new__(cls)
        src = self.__dict__
        d = g.__dict__
        d.update(src)
        memo = {}  # id(unit) -> copy, so references to units elsewhere stay aliased in the clone
        if "rng" in src:
            d["backline"] = [[memo.setdefault(id(u), u.copy()) for u in z] for z in src["backline"]]
            d["frontline"] = [memo.setdefault(id(u), u.copy()) for u in src["frontline"]]
            v = src["rng"]
            r = type(v).__new__(type(v))  # skip urandom seeding; state set below
            r.setstate(v.getstate())
            d["rng"] = r
            for k in _LIST_OF_LISTS:
                d[k] = [list(x) for x in src[k]]
            for k in _FLAT_LISTS:
                v = src[k]
                d[k] = list(v) if type(v) is list else copy.deepcopy(v, memo)
            q = src["queue"]
            d["queue"] = ([self._copy_inst(i, memo) for i in q] if q else []) if type(q) is list else (
                copy.deepcopy(q, memo))
            v = src["pending"]
            if v is not None:
                d["pending"] = (self._copy_inst(v[0], memo), v[1], v[2]) if type(v) is tuple else (
                    copy.deepcopy(v, memo))
            v = src["mulligan_marks"]
            d["mulligan_marks"] = set(v) if type(v) is set else copy.deepcopy(v, memo)
            if not _int_tuple(src["deck_ids"]):
                d["deck_ids"] = copy.deepcopy(src["deck_ids"], memo)
            v = src["decklists"]
            if not (type(v) is tuple and all(_int_tuple(x) for x in v)):
                d["decklists"] = copy.deepcopy(v, memo)
            v = src["_combat"]
            if v is not None and not _int_tuple(v):
                d["_combat"] = copy.deepcopy(v, memo)
            g._legal = g._mask = None
        else:  # never reset
            g._legal, g._mask = [], bytearray(g.num_actions)
        for k in src.keys() - _KNOWN_ATTRS:  # state added by tools/tests/later stages
            v = src[k]
            if not isinstance(v, _SCALAR_TYPES):
                d[k] = copy.deepcopy(v, memo)
        return g

    def determinize(self, player: int, rng: random.Random) -> "Game":
        """A clone in which everything hidden from `player` is resampled consistently with what `player`
        knows (SPEC §4). Reads only `player`'s information plus `rng`."""
        if not hasattr(self, "rng"):
            raise RuntimeError("call reset(seed) before determinize()")
        player = _as_index(player, "player")
        if player not in (0, 1):
            raise ValueError(f"player must be 0 or 1, got {player}")
        o = 1 - player
        cfg = self.config
        n = len(self._card_defs)
        g = self.clone()
        revealed = self.revealed[o]
        decklist = generate_deck(rng, cfg, required=revealed)
        counts = [0] * n
        for c in decklist:
            counts[c] += 1
        unknown = [c for c in range(n) for _ in range(counts[c] - revealed[c])]  # canonical order
        rng.shuffle(unknown)
        kh = self.known_hand[o]
        known = [c for c in range(n) for _ in range(kh[c])]
        hand_size, deck_size = len(self.hands[o]), len(self.deck_cards[o])
        known = known[:hand_size]
        need = hand_size - len(known) + deck_size
        while len(unknown) < need:  # only for inconsistent hand-built positions
            extra = list(generate_deck(rng, cfg))
            rng.shuffle(extra)
            unknown += extra
        n_unknown_hand = hand_size - len(known)
        g.hands[o] = sorted(known + unknown[:n_unknown_hand])
        g.deck_cards[o] = unknown[n_unknown_hand:need]
        own = sorted(self.deck_cards[player])  # contents are known, the order is not
        rng.shuffle(own)
        g.deck_cards[player] = own
        if g.phase == MULLIGAN and g.current == o:
            g.mulligan_marks = set()
        ids = list(g.deck_ids)
        ids[o] = -1
        g.deck_ids = tuple(ids)
        lists = list(g.decklists)
        lists[o] = decklist
        g.decklists = tuple(lists)
        g.rng = random.Random(rng.getrandbits(64))
        g.seed = None
        g.num_steps = 0  # it counted o's hidden MULLIGAN marks
        g.__dict__.pop("_dlc", None)  # the decklist-count cache may hold o's true decklist
        g._legal = g._mask = None
        return g

    # ------------------------------------------------------------------ debugging
    def describe(self, action: int) -> str:
        try:
            action = _as_index(action, "action")
        except TypeError:
            return f"<not an action {action!r}>"
        if 0 <= action < self.num_actions:
            return self.action_space.describe(action)
        return f"<out of range {action!r}>"

    def render(self) -> str:
        """Omniscient debug view. Never give this to an agent."""
        if not hasattr(self, "rng"):
            return "<game not started: call reset(seed)>"
        cards = self._card_defs
        tag = {TROOP: "", FAST: "^", RANGED: "~"}

        def zone(units):
            out = []
            for u in units:
                traits = (("D" if u.defense else "") + (f"A{u.armor}" if u.armor else "") + ("B" if u.blitz else "")
                          + ("S" if u.smokescreen else "") + ("F" if u.fury else "")
                          + (("Am" if u.ambush_ready else "am") if u.ambush else "") + ("Sh" if u.shock else "")
                          + ("Im" if u.immune else ""))
                flags = "".join(f for f, on in (("*", u.summoned), ("m", u.moved), ("a", u.attacks > 0),
                                                ("p", u.pinned), ("t", u.token)) if on)
                temp = f" t{u.temp_atk:+d}/{u.temp_hp:+d}" if (u.temp_atk or u.temp_hp) else ""
                stat = (f" s{u.static_atk:+d}/{u.static_hp:+d}" if (u.static_atk or u.static_hp or u.static_traits)
                        else "")
                out.append(f"{tag.get(u.nature, '?')}{cards[u.card].name}({u.atk}/{u.hp}"
                           f"{' ' + traits if traits else ''}{temp}{stat}){flags}")
            return " ".join(out) or "-"

        def deck_name(p):
            d = self.deck_ids[p]
            return self.config.deck_names[d] if d >= 0 else "random"

        owner = "-" if self.front_owner is None else f"P{self.front_owner}"
        lines = [
            f"round {self.round} turn {self.turn}  {PHASE_NAMES[self.phase]}  current P{self.current}  "
            f"first P{self.first_player}  done={self.done} winner={self._winner}  "
            f"decks {deck_name(0)} vs {deck_name(1)}",
        ]
        for p in (0, 1):
            bonus = f"{self.coin_bonus[p]:+d}" if self.coin_bonus[p] else ""
            lines.append(f"P{p}: base {self.base_hp[p]}  coins {self.coins[p]}{bonus}  deck {len(self.deck_cards[p])}  "
                         f"burned {self.burned[p]}  hand [{', '.join(cards[c].name for c in self.hands[p])}]")
        lines.append(f"P0 back : {zone(self.backline[0])}")
        lines.append(f"front({owner}): {zone(self.frontline)}")
        lines.append(f"P1 back : {zone(self.backline[1])}")
        if self.phase == MULLIGAN:
            lines.append(f"mulligan marks: {sorted(self.mulligan_marks)}  done {self.mulligan_done}")
        if self.pending is not None:
            pv = self._pending_view()
            opts = ", ".join(self.action_space.choice_slot_name(t) for t in self.pending[1])
            lines.append(f"pending: {cards[pv.card].name}#{pv.effect} {pv.action} {pv.amount} "
                         f"{pv.select}/{pv.side}/{pv.kind}  options [{opts}]")
        if self.queue:
            lines.append(f"queue: {len(self.queue)} instance(s)")
        if self.guard_trips:
            lines.append(f"loop guard trips: {self.guard_trips}")
        lines.append("legend: ^fast ~ranged D=defense A=armor B=blitz S=smokescreen F=fury Am=ambush (am=used) "
                     "Sh=shock Im=immune t=turn s=static *=deployed this round m=moved a=attacked p=pinned t=token")
        return "\n".join(lines)


# ---------------------------------------------------------------- module helpers (effects)
def _tagged(targets: list, kind: str) -> list:
    """Targets as recorded in a clause batch: units and bases as they are, players as -1 - p."""
    if kind == "player":
        return [-1 - p for p in targets]
    return list(targets)


def _merge_damaged(targets: list, first: list, second: list) -> list:
    """The damaged entries of a target_condition split's two bodies as one damage step's list: in the order
    of the clause's targets (an else body's own targets after them), a unit hit by both bodies once with the
    total damage."""
    if not first or not second:
        return first or second
    pos = {}
    for i, t in enumerate(targets):
        if type(t) is not int:
            pos.setdefault(id(t), i)
    merged = {}
    extra = len(targets)
    for u, src, n in first + second:
        e = merged.get(id(u))
        if e is None:
            merged[id(u)] = [pos.get(id(u), extra), u, src, n]
            extra += 1
        else:
            e[3] += n
    return [(u, src, n) for _, u, src, n in sorted(merged.values(), key=lambda e: e[0])]


def _record(hit: list, dead: list) -> tuple:
    """Batch record of a resolved clause: (its distinct targets in first-hit order, whether any of its unit
    targets was killed by the clause)."""
    out, seen = [], set()
    for t in hit:
        k = ("i", t) if type(t) is int else id(t)
        if k not in seen:
            seen.add(k)
            out.append(t)
    killed = bool(dead) and any(type(t) is not int and any(t is d for d in dead) for t in out)
    return (tuple(out), killed)


def _shift_atk(u: Unit, d: int) -> int:
    """Change the unit's own atk (atk - static_atk) by d, clamped at 0; static contributions stay on top and
    the effective atk is clamped at 0 too. Returns the applied change of the own value (SPEC §2.10)."""
    s = u.static_atk
    own = u.atk - s
    new = own + d
    if new < 0:
        new = 0
    eff = new + s
    if eff < 0:
        eff = 0
    u.atk = eff
    u.static_atk = eff - new
    return new - own


def _shift_move_cost(u: Unit, d: int) -> int:
    """`_shift_atk` for the move cost (never below 0)."""
    s = u.static_move_cost
    own = u.move_cost - s
    new = own + d
    if new < 0:
        new = 0
    eff = new + s
    if eff < 0:
        eff = 0
    u.move_cost = eff
    u.static_move_cost = eff - new
    return new - own


def _set_static(u: Unit, a: int, h: int, m: int, t: int) -> None:
    """Apply a unit's new static contributions (SPEC §2.12): atk and move cost on top of the own values
    (clamped at 0; the fields record the applied deltas); max_hp by the hp sum (an increase raises hp, a
    decrease caps hp at the new max_hp, so it never kills); traits = own | static."""
    old = u.static_atk
    if a or old:
        own = u.atk - old
        eff = own + a
        if eff < 0:
            eff = 0
        u.atk = eff
        u.static_atk = eff - own
    old = u.static_hp
    if h != old:
        d = h - old
        u.max_hp += d
        if d > 0:
            u.hp += d
        elif u.hp > u.max_hp:
            u.hp = u.max_hp
        u.static_hp = h
    old = u.static_move_cost
    if m or old:
        own = u.move_cost - old
        eff = own + m
        if eff < 0:
            eff = 0
        u.move_cost = eff
        u.static_move_cost = eff - own
    old = u.static_traits
    if t != old:
        changed = t ^ old
        new = (u.base_traits & ~u.temp_removed) | u.temp_traits | t
        for name, bit in _BOOL_TRAIT_LIST:
            if changed & bit:
                setattr(u, name, bool(new & bit))
        u.static_traits = t
