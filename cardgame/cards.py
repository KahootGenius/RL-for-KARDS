"""Card definitions, the effect schema and ruleset loading (SPEC §1). Cards are data; rules live in
the engine.

Cards are units, operations or tokens (`token: true`, created only by effects). Effects are parsed
into frozen dataclasses (`EffectDef`, `TargetDef`, `AmountDef`, `ConditionDef`, `FilterDef`) by a
strict loader: unknown keys, triggers, actions, traits or filters, wrong types, duplicate JSON keys
and invalid combinations raise `ValueError` naming the card. Phase 1b (SPEC §1.2b) adds card tags, the
ambush / shock / immune traits, `on_attacked` and `static` triggers, watcher `event_filter`s, `prev` and
`adjacent` targets, compare / history / prev conditions, `target_condition` + `else` (parsed into
`EffectDef.else_`), `repeat`, uncapped heals, negative coin gains and buff move costs.
"""
from __future__ import annotations

import json
import operator
import random
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_CARDS = DATA_DIR / "cards.json"
DEFAULT_DECKS = DATA_DIR / "decks.json"

# ---------------------------------------------------------------- vocabulary (SPEC §1.1-1.2)
UNIT, OPERATION = "unit", "operation"
SUPPORTED_TYPES = frozenset({UNIT, OPERATION})
NATURES = ("troop", "fast", "ranged")  # index = nature code used by the engine
TROOP, FAST, RANGED = 0, 1, 2
NO_NATURE = -1  # operations have no nature
DEFAULT_MOVE_COST = 1

TRAITS = ("defense", "armor", "blitz", "smokescreen", "fury", "ambush", "shock", "immune")
SUPPORTED_TRAITS = frozenset(TRAITS)
BOOL_TRAITS = ("defense", "blitz", "smokescreen", "fury", "ambush", "shock", "immune")
DEFENSE_BIT, ARMOR_BIT, BLITZ_BIT, SMOKESCREEN_BIT, FURY_BIT = 1, 2, 4, 8, 16
AMBUSH_BIT, SHOCK_BIT, IMMUNE_BIT = 32, 64, 128
TRAIT_BITS = {"defense": DEFENSE_BIT, "armor": ARMOR_BIT, "blitz": BLITZ_BIT, "smokescreen": SMOKESCREEN_BIT,
              "fury": FURY_BIT, "ambush": AMBUSH_BIT, "shock": SHOCK_BIT, "immune": IMMUNE_BIT}

# Stage 3 phase 1 triggers keep their indices; the §1.2b ones are appended.
TRIGGERS = ("on_play", "on_deploy", "on_death", "on_attack", "on_damaged", "on_move", "on_kill",
            "start_of_turn", "end_of_turn", "on_attacked", "static")
TRIGGER_INDEX = {t: i for i, t in enumerate(TRIGGERS)}
TURN_TRIGGERS = frozenset({"start_of_turn", "end_of_turn"})
SCOPES = ("self", "friendly", "enemy", "any")
ACTIONS = ("damage", "heal", "buff", "destroy", "draw", "gain_coins", "increase_max_coins", "add_trait",
           "remove_trait", "summon", "return_to_hand", "pin", "discard", "add_card", "retreat")
SELECTS = ("chosen", "random", "all", "self", "event", "prev", "adjacent")
SIDES = ("friendly", "enemy", "any")
KINDS = ("unit", "base", "player", "unit_or_base")
ZONES = ("board", "backline", "frontline")
DURATIONS = ("permanent", "turn")
POSITIONS = ("both", "left", "right")
AMOUNT_STATS = ("atk", "hp", "max_hp", "cost", "move_cost")
AMOUNT_COUNTS = ("units", "hand", "coins", "deck", "base_hp")
CONDITION_TYPES = ("control", "frontline", "source_zone", "base_hp", "hand_size", "turn", "prev", "compare",
                   "history")
COMPARE_OPS = (">", ">=", "<", "<=", "==")
HISTORY_EVENTS = ("operation_played", "unit_deployed", "unit_died")  # index = engine counter slot
HISTORY_WINDOWS = ("turn", "game")
FILTER_KEYS = ("nature", "trait", "not_trait", "damaged", "min_cost", "max_cost", "min_atk", "max_atk",
               "min_hp", "max_hp", "token", "other", "tag", "not_tag", "pinned")
CARD_FILTER_KEYS = ("min_cost", "max_cost", "tag", "not_tag")  # event_filter of on_play watchers (the card)
FULL = -1  # resolved value of the heal amount "full"
UNCAPPED_BASE_HP = 99  # cap of an "uncapped" base heal

# action -> target kinds it accepts, the parameters it takes and the ones it requires
ACTION_KINDS = {
    "damage": {"unit", "base", "unit_or_base"}, "heal": {"unit", "base", "unit_or_base"},
    "buff": {"unit"}, "destroy": {"unit"}, "draw": {"player"}, "gain_coins": {"player"},
    "increase_max_coins": {"player"}, "add_trait": {"unit"}, "remove_trait": {"unit"}, "summon": {"player"},
    "return_to_hand": {"unit"}, "pin": {"unit"}, "discard": {"player"}, "add_card": {"player"},
    "retreat": {"unit"},
}
ACTION_PARAMS = {
    "damage": {"amount"}, "heal": {"amount", "uncapped"}, "buff": {"atk", "hp", "move_cost", "duration"},
    "destroy": set(), "draw": {"amount"}, "gain_coins": {"amount"}, "increase_max_coins": {"amount"},
    "add_trait": {"trait", "amount", "duration"}, "remove_trait": {"trait", "duration"},
    "summon": {"card", "amount"}, "return_to_hand": set(), "pin": set(), "discard": {"amount"},
    "add_card": {"card", "amount"}, "retreat": set(),
}
REQUIRED_PARAMS = {
    "damage": {"amount"}, "heal": {"amount"}, "draw": {"amount"}, "gain_coins": {"amount"},
    "increase_max_coins": {"amount"}, "discard": {"amount"}, "add_trait": {"trait"}, "remove_trait": {"trait"},
    "summon": {"card"}, "add_card": {"card"},
}
PARAM_KEYS = frozenset({"amount", "atk", "hp", "move_cost", "duration", "trait", "card", "uncapped"})
EFFECT_KEYS = frozenset({"trigger", "scope", "condition", "target", "action", "event_filter", "target_condition",
                         "else", "repeat"}) | PARAM_KEYS
ELSE_KEYS = frozenset({"action", "target"}) | PARAM_KEYS
# triggers whose `chosen` targets are allowed (scope self only): fired by the controller's own action
CHOSEN_TRIGGERS = frozenset({"on_play", "on_deploy", "on_move", "on_attack"})
# (trigger, scope) pairs that have an event unit (SPEC §2.8, §1.2b)
_SELF_EVENT_TRIGGERS = frozenset({"on_attack", "on_damaged", "on_kill", "on_attacked"})
_WATCH_EVENT_TRIGGERS = frozenset({"on_deploy", "on_death", "on_attack", "on_damaged", "on_move", "on_kill",
                                   "on_attacked"})
EVENT_DAMAGE_TRIGGERS = frozenset({"on_damaged", "on_kill"})  # {"event": "damage"} amounts
STATIC_ACTIONS = frozenset({"buff", "add_trait"})
STATIC_SELECTS = frozenset({"self", "all", "adjacent"})
MAX_EFFECTS = 3


def has_event_unit(trigger: str, scope: str) -> bool:
    """Whether instances of (trigger, scope) carry an event unit (SPEC §2.8 table)."""
    if scope == "self":
        return trigger in _SELF_EVENT_TRIGGERS
    return trigger in _WATCH_EVENT_TRIGGERS


# ---------------------------------------------------------------- parsed effect schema
@dataclass(frozen=True)
class FilterDef:
    """Unit filter: every given key must hold. `traits` / `not_traits` are trait bitmasks; `tags` = has any of
    them, `not_tags` = has none of them (sorted tag names)."""
    natures: tuple = ()          # nature codes (empty = any)
    traits: int = 0              # must have all of these
    not_traits: int = 0          # must have none of these
    damaged: Optional[bool] = None
    min_cost: Optional[int] = None
    max_cost: Optional[int] = None
    min_atk: Optional[int] = None
    max_atk: Optional[int] = None
    min_hp: Optional[int] = None
    max_hp: Optional[int] = None
    token: Optional[bool] = None
    other: bool = False          # not the source unit
    tags: tuple = ()
    not_tags: tuple = ()
    pinned: Optional[bool] = None


@dataclass(frozen=True)
class AmountDef:
    """`kind`: "literal" (value), "full" (heal only), "stat" (stat of self/event/prev), "count" or "event"
    (the damage of the triggering event)."""
    kind: str
    value: int = 0
    stat: str = ""
    of: str = ""
    count: str = ""
    side: str = ""
    zone: str = "board"
    filter: Optional[FilterDef] = None
    times: int = 1
    plus: int = 0


@dataclass(frozen=True)
class TargetDef:
    select: str
    side: str = ""               # "" for select self/event/prev; adjacent: "" = any
    kind: str = "unit"
    zone: str = "board"
    filter: Optional[FilterDef] = None
    count: int = 1               # random only
    of: str = ""                 # adjacent: reference unit (self / event / prev)
    position: str = ""           # adjacent: both / left / right


@dataclass(frozen=True)
class ConditionDef:
    type: str
    side: str = ""
    zone: str = "board"
    filter: Optional[FilterDef] = None
    min: Optional[int] = None
    max: Optional[int] = None
    owner: str = ""              # frontline: friendly / enemy / none
    whose: str = ""              # turn: own / opponent
    killed: Optional[bool] = None    # prev
    left: Optional[AmountDef] = None  # compare
    op: str = ""                      # compare
    right: Optional[AmountDef] = None  # compare
    event: str = ""              # history: operation_played / unit_deployed / unit_died
    window: str = ""             # history: turn / game


@dataclass(frozen=True)
class EffectDef:
    index: int                   # slot within the card (0..2)
    trigger: str
    scope: str
    action: str
    target: TargetDef
    condition: tuple = ()        # ConditionDefs, all must hold
    amount: Optional[AmountDef] = None
    atk: Optional[AmountDef] = None   # buff
    hp: Optional[AmountDef] = None    # buff
    duration: str = "permanent"
    trait: tuple = ()            # add_trait: one name; remove_trait: one or more names
    card: int = -1               # summon / add_card: token card index
    card_id: str = ""
    # ---- §1.2b
    move_cost: int = 0           # buff: move-cost delta
    uncapped: bool = False       # heal: a base may exceed base_hp (up to UNCAPPED_BASE_HP)
    event_filter: Optional[FilterDef] = None      # watchers: filter on the event unit / played card
    target_condition: Optional[FilterDef] = None  # per selected unit target: match -> action, else -> else_
    else_: Optional["EffectDef"] = None           # the `else` body (same trigger/scope/index, no condition)
    repeat: int = 1
    is_else: bool = False        # this EffectDef is an `else` body
    own_target: bool = True      # else bodies: False = reuses the effect's target / failing targets


@dataclass(frozen=True)
class CardDef:
    index: int
    id: str
    name: str
    type: str
    cost: int
    attack: int
    health: int
    nature: int = TROOP
    move_cost: int = DEFAULT_MOVE_COST
    defense: bool = False
    armor: int = 0
    effects: tuple = ()
    blitz: bool = False
    smokescreen: bool = False
    fury: bool = False
    token: bool = False
    ambush: bool = False
    shock: bool = False
    immune: bool = False
    tags: tuple = ()             # KARDS unit kind / nation / family (strings, card order)

    @property
    def is_unit(self) -> bool:
        return self.type == UNIT

    @property
    def is_operation(self) -> bool:
        return self.type == OPERATION

    @property
    def nature_name(self) -> str:
        return NATURES[self.nature] if self.nature >= 0 else "none"

    @property
    def traits(self) -> dict:
        out = {}
        if self.defense:
            out["defense"] = True
        if self.armor:
            out["armor"] = self.armor
        for name in BOOL_TRAITS[1:]:
            if getattr(self, name):
                out[name] = True
        return out

    @property
    def trait_mask(self) -> int:
        """Bitmask of the card's traits (armor bit set when armor > 0)."""
        m = ARMOR_BIT if self.armor else 0
        for name in BOOL_TRAITS:
            if getattr(self, name):
                m |= TRAIT_BITS[name]
        return m


@dataclass(frozen=True)
class CardPool:
    cards: tuple

    def __len__(self) -> int:
        return len(self.cards)

    def __getitem__(self, index: int) -> CardDef:
        return self.cards[index]

    def by_id(self, card_id: str) -> CardDef:
        for c in self.cards:
            if c.id == card_id:
                return c
        raise KeyError(f"unknown card id {card_id!r}")

    @property
    def max_cost(self) -> int:
        return max(c.cost for c in self.cards)

    @property
    def max_attack(self) -> int:
        return max(c.attack for c in self.cards)

    @property
    def max_health(self) -> int:
        return max(c.health for c in self.cards)

    @property
    def max_move_cost(self) -> int:
        return max(c.move_cost for c in self.cards)

    @property
    def max_armor(self) -> int:
        return max(c.armor for c in self.cards)

    @property
    def has_effects(self) -> bool:
        return any(c.effects for c in self.cards)


@dataclass(frozen=True)
class GameConfig:
    cards: CardPool
    decks: tuple  # one sorted tuple of card indices per deck; each game uses two of them
    deck_names: tuple = ()
    deck_styles: tuple = ()
    base_hp: int = 20
    max_rounds: int = 50
    zone_capacity: int = 5
    max_hand_size: int = 10
    opening_hand: tuple = (4, 5)  # (first player, second player)
    deck_size: int = 40
    max_copies: int = 3
    coin_cap: Optional[int] = None
    mulligan: bool = True         # False skips the mulligan phase (engine tests, scenarios)
    max_effect_events: int = 256  # loop guard: effect instances resolved per action (SPEC §2.8)

    def __post_init__(self):
        for name in ("base_hp", "max_rounds", "zone_capacity", "max_hand_size", "deck_size", "max_copies",
                     "max_effect_events"):
            v = getattr(self, name)
            if type(v) is not int or v < 1:
                raise ValueError(f"{name} must be a positive integer, got {v!r}")
        if type(self.mulligan) is not bool:
            raise ValueError(f"mulligan must be True or False, got {self.mulligan!r}")
        if len(self.opening_hand) != 2 or any(type(n) is not int or n < 0 for n in self.opening_hand):
            raise ValueError(f"opening_hand must be two non-negative integers, got {self.opening_hand!r}")
        if self.coin_cap is not None and (type(self.coin_cap) is not int or self.coin_cap < 0):
            raise ValueError(f"coin_cap must be None or a non-negative integer, got {self.coin_cap!r}")
        if len(self.decks) < 1:
            raise ValueError("at least one deck is required")
        if not self.deck_names:
            object.__setattr__(self, "deck_names", tuple(f"deck{i}" for i in range(len(self.decks))))
        if not self.deck_styles:
            object.__setattr__(self, "deck_styles", ("",) * len(self.decks))
        if len(self.deck_names) != len(self.decks) or len(self.deck_styles) != len(self.decks):
            raise ValueError("deck_names/deck_styles must match the number of decks")

    @property
    def n_decks(self) -> int:
        return len(self.decks)

    def coins_for_round(self, rnd: int) -> int:
        return rnd if self.coin_cap is None else min(rnd, self.coin_cap)


# ---------------------------------------------------------------- strict parsing helpers
UNIT_KEYS = frozenset({"id", "name", "type", "nature", "cost", "attack", "health", "move_cost", "traits",
                       "effects", "token", "tags"})
OPERATION_KEYS = frozenset({"id", "name", "type", "cost", "effects", "token", "tags"})
CARD_KEYS = UNIT_KEYS  # Stage 2 name


class _Ctx:
    """Card being parsed (for error messages and context-dependent validation)."""
    __slots__ = ("card_id", "is_operation", "trigger", "scope", "action", "has_prev", "static")

    def __init__(self, card_id: str, is_operation: bool):
        self.card_id, self.is_operation = card_id, is_operation
        self.trigger = self.scope = self.action = ""
        self.has_prev = False   # an earlier clause of the card has the same trigger (prev is available)
        self.static = False     # parsing a static effect

    def fail(self, msg: str):
        raise ValueError(f"card {self.card_id!r}: {msg}")


def _is_int(v) -> bool:
    return type(v) is int  # no bools, floats or numeric strings


def _stat(card_id: str, raw: dict, key: str) -> int:
    v = raw[key]
    if not _is_int(v):
        raise ValueError(f"card {card_id!r}: {key} must be an integer, got {v!r}")
    return v


def _no_duplicate_keys(pairs: list) -> dict:
    keys = [k for k, _ in pairs]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise ValueError(f"duplicate JSON keys {dup}")
    return dict(pairs)


def _load_json(path: Path | str, top_keys: frozenset) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f, object_pairs_hook=_no_duplicate_keys)
    if not isinstance(raw, dict) or set(raw) != top_keys:
        raise ValueError(f"{path}: expected an object with exactly the keys {sorted(top_keys)}")
    return raw


def _obj(ctx: _Ctx, raw, what: str, allowed, required=()) -> dict:
    if not isinstance(raw, dict):
        ctx.fail(f"{what} must be an object, got {raw!r}")
    extra = set(raw) - set(allowed)
    if extra:
        ctx.fail(f"{what}: unknown keys {sorted(map(str, extra))}")
    missing = [k for k in required if k not in raw]
    if missing:
        ctx.fail(f"{what}: missing {missing}")
    return raw


def _choice(ctx: _Ctx, value, options, what: str) -> str:
    if not isinstance(value, str) or value not in options:
        ctx.fail(f"{what} must be one of {list(options)}, got {value!r}")
    return value


def _opt_int(ctx: _Ctx, raw: dict, key: str, what: str, minimum: Optional[int] = None) -> Optional[int]:
    if key not in raw:
        return None
    v = raw[key]
    if not _is_int(v) or (minimum is not None and v < minimum):
        ctx.fail(f"{what}.{key} must be an integer{f' >= {minimum}' if minimum is not None else ''}, got {v!r}")
    return v


def _opt_bool(ctx: _Ctx, raw: dict, key: str, what: str) -> Optional[bool]:
    if key not in raw:
        return None
    v = raw[key]
    if type(v) is not bool:
        ctx.fail(f"{what}.{key} must be true or false, got {v!r}")
    return v


def _names(ctx: _Ctx, value, options, what: str) -> tuple:
    """A name or a non-empty list of distinct names."""
    items = value if isinstance(value, list) else [value]
    if not items:
        ctx.fail(f"{what} must not be empty")
    out = tuple(_choice(ctx, v, options, what) for v in items)
    if len(set(out)) != len(out):
        ctx.fail(f"{what} lists a name twice: {value!r}")
    return out


def _tag_names(ctx: _Ctx, value, what: str) -> tuple:
    """A tag or a non-empty list of distinct tags (non-empty strings), sorted."""
    items = value if isinstance(value, list) else [value]
    if not items:
        ctx.fail(f"{what} must not be empty")
    for t in items:
        if not isinstance(t, str) or not t:
            ctx.fail(f"{what} must be a non-empty string or a list of them, got {value!r}")
    if len(set(items)) != len(items):
        ctx.fail(f"{what} lists a tag twice: {value!r}")
    return tuple(sorted(items))


def _trait_mask(names: tuple) -> int:
    m = 0
    for n in names:
        m |= TRAIT_BITS[n]
    return m


def parse_filter(ctx: _Ctx, raw, what: str = "filter", card_only: bool = False) -> FilterDef:
    """A unit filter; `card_only` (on_play event filters) allows only the card keys min/max_cost, tag, not_tag."""
    raw = _obj(ctx, raw, what, CARD_FILTER_KEYS if card_only else FILTER_KEYS)
    natures = ()
    if "nature" in raw:
        natures = tuple(NATURES.index(n) for n in _names(ctx, raw["nature"], NATURES, f"{what}.nature"))
    traits = _trait_mask(_names(ctx, raw["trait"], TRAITS, f"{what}.trait")) if "trait" in raw else 0
    not_traits = _trait_mask(_names(ctx, raw["not_trait"], TRAITS, f"{what}.not_trait")) if "not_trait" in raw else 0
    if traits & not_traits:
        ctx.fail(f"{what}: a trait is both required and excluded")
    tags = _tag_names(ctx, raw["tag"], f"{what}.tag") if "tag" in raw else ()
    not_tags = _tag_names(ctx, raw["not_tag"], f"{what}.not_tag") if "not_tag" in raw else ()
    if set(tags) & set(not_tags):
        ctx.fail(f"{what}: a tag is both required and excluded")
    vals = {k: _opt_int(ctx, raw, k, what) for k in ("min_cost", "max_cost", "min_atk", "max_atk", "min_hp", "max_hp")}
    for lo, hi in (("min_cost", "max_cost"), ("min_atk", "max_atk"), ("min_hp", "max_hp")):
        if vals[lo] is not None and vals[hi] is not None and vals[lo] > vals[hi]:
            ctx.fail(f"{what}: {lo} > {hi} never matches")
    other = _opt_bool(ctx, raw, "other", what)
    return FilterDef(natures=natures, traits=traits, not_traits=not_traits,
                     damaged=_opt_bool(ctx, raw, "damaged", what), token=_opt_bool(ctx, raw, "token", what),
                     other=bool(other), tags=tags, not_tags=not_tags, pinned=_opt_bool(ctx, raw, "pinned", what),
                     **vals)


def _check_of(ctx: _Ctx, of: str, stat: str, what: str) -> None:
    if of == "event" and not has_event_unit(ctx.trigger, ctx.scope):
        ctx.fail(f"{what}: {{'of': 'event'}} needs a trigger with an event unit, not {ctx.trigger}/{ctx.scope}")
    if of == "self" and ctx.is_operation and stat != "cost":
        ctx.fail(f"{what}: an operation has no unit stats (only {{'stat': 'cost', 'of': 'self'}})")
    if of == "prev":
        _need_prev(ctx, what)


def _need_prev(ctx: _Ctx, what: str) -> None:
    if ctx.static:
        ctx.fail(f"{what}: static effects have no previous clause")
    if not ctx.has_prev:
        ctx.fail(f"{what}: 'prev' needs an earlier clause of the card with the same trigger ({ctx.trigger})")


def parse_amount(ctx: _Ctx, raw, what: str, minimum: Optional[int] = 0, allow_full: bool = False,
                 allow_expr: bool = True) -> AmountDef:
    """An int literal (>= `minimum`; None = any int), "full" (heal) or an expression object."""
    if _is_int(raw):
        if minimum is not None and raw < minimum:
            ctx.fail(f"{what} must be >= {minimum}, got {raw!r}")
        return AmountDef(kind="literal", value=raw)
    if raw == "full":
        if not allow_full:
            ctx.fail(f"{what}: 'full' is only allowed for heal")
        return AmountDef(kind="full", value=FULL)
    if not isinstance(raw, dict):
        ctx.fail(f"{what} must be an integer or an expression object, got {raw!r}")
    if not allow_expr:
        ctx.fail(f"{what}: static effects take integer amounts only, got {raw!r}")
    if "stat" in raw:
        raw = _obj(ctx, raw, what, ("stat", "of", "times", "plus"), ("stat", "of"))
        stat = _choice(ctx, raw["stat"], AMOUNT_STATS, f"{what}.stat")
        of = _choice(ctx, raw["of"], ("self", "event", "prev"), f"{what}.of")
        _check_of(ctx, of, stat, what)
        return AmountDef(kind="stat", stat=stat, of=of, times=_times(ctx, raw, what), plus=_plus(ctx, raw, what))
    if "count" in raw:
        count = _choice(ctx, raw["count"], AMOUNT_COUNTS, f"{what}.count")
        if count == "units":
            raw = _obj(ctx, raw, what, ("count", "side", "zone", "filter", "times", "plus"), ("side",))
            flt = parse_filter(ctx, raw["filter"], f"{what}.filter") if "filter" in raw else None
            return AmountDef(kind="count", count=count, side=_choice(ctx, raw["side"], SIDES, f"{what}.side"),
                             zone=_choice(ctx, raw.get("zone", "board"), ZONES, f"{what}.zone"), filter=flt,
                             times=_times(ctx, raw, what), plus=_plus(ctx, raw, what))
        raw = _obj(ctx, raw, what, ("count", "side", "times", "plus"), ("side",))
        return AmountDef(kind="count", count=count, side=_choice(ctx, raw["side"], SIDES, f"{what}.side"),
                         times=_times(ctx, raw, what), plus=_plus(ctx, raw, what))
    if "event" in raw:
        raw = _obj(ctx, raw, what, ("event", "times", "plus"), ("event",))
        _choice(ctx, raw["event"], ("damage",), f"{what}.event")
        if ctx.trigger not in EVENT_DAMAGE_TRIGGERS:
            ctx.fail(f"{what}: {{'event': 'damage'}} needs trigger on_damaged or on_kill, not {ctx.trigger}")
        return AmountDef(kind="event", times=_times(ctx, raw, what), plus=_plus(ctx, raw, what))
    ctx.fail(f"{what}: an expression needs 'stat', 'count' or 'event', got {raw!r}")


def _times(ctx: _Ctx, raw: dict, what: str) -> int:
    v = _opt_int(ctx, raw, "times", what)
    return 1 if v is None else v


def _plus(ctx: _Ctx, raw: dict, what: str) -> int:
    v = _opt_int(ctx, raw, "plus", what)
    return 0 if v is None else v


TARGET_SHORTHANDS = {
    "self": dict(select="self"),
    "event": dict(select="event"),
    "friendly_base": dict(select="all", side="friendly", kind="base"),
    "enemy_base": dict(select="all", side="enemy", kind="base"),
    "controller": dict(select="all", side="friendly", kind="player"),
    "opponent": dict(select="all", side="enemy", kind="player"),
}


def parse_target(ctx: _Ctx, raw) -> TargetDef:
    if isinstance(raw, str):
        if raw not in TARGET_SHORTHANDS:
            ctx.fail(f"unknown target shorthand {raw!r} (known: {sorted(TARGET_SHORTHANDS)})")
        return TargetDef(**TARGET_SHORTHANDS[raw])
    raw = _obj(ctx, raw, "target", ("select", "side", "kind", "zone", "filter", "count", "of", "position"),
               ("select",))
    select = _choice(ctx, raw["select"], SELECTS, "target.select")
    if select == "prev" and "kind" not in raw:  # prev passes on whatever the action can take (§1.2b)
        kinds = ACTION_KINDS.get(ctx.action, {"unit"})
        kind = "unit_or_base" if "base" in kinds else ("player" if "player" in kinds else "unit")
    else:
        kind = _choice(ctx, raw.get("kind", "unit"), KINDS, "target.kind")
    of = position = ""
    if select in ("self", "event", "prev"):
        if select != "prev" and kind != "unit":
            ctx.fail(f"target select {select!r} only works with kind 'unit'")
        for key in ("side", "zone"):
            if key in raw:
                ctx.fail(f"target select {select!r} takes no {key!r}")
        side = ""
        if select == "prev":
            _need_prev(ctx, "target 'prev'")
    elif select == "adjacent":
        if kind != "unit":
            ctx.fail("target select 'adjacent' only works with kind 'unit'")
        if "zone" in raw:
            ctx.fail("target select 'adjacent' takes no 'zone' (the reference unit's zone)")
        side = _choice(ctx, raw.get("side", "any"), SIDES, "target.side")
        of = _choice(ctx, raw.get("of", "self"), ("self", "event", "prev"), "target.of")
        position = _choice(ctx, raw.get("position", "both"), POSITIONS, "target.position")
        if of == "event" and not has_event_unit(ctx.trigger, ctx.scope):
            ctx.fail(f"adjacent of 'event' needs a trigger with an event unit, not {ctx.trigger}/{ctx.scope}")
        if of == "self":
            if ctx.is_operation:
                ctx.fail("an operation has no source unit to be adjacent to")
            if ctx.trigger == "on_death" and ctx.scope == "self":
                ctx.fail("on_death effects may not use the dead source as the adjacent reference")
        if of == "prev":
            _need_prev(ctx, "adjacent of 'prev'")
    else:
        if "side" not in raw:
            ctx.fail(f"target select {select!r} needs a 'side'")
        side = _choice(ctx, raw["side"], SIDES, "target.side")
    if select != "adjacent":
        for key in ("of", "position"):
            if key in raw:
                ctx.fail(f"target {key!r} only works with select 'adjacent'")
    if kind == "player" and select not in ("all", "prev"):
        ctx.fail("target kind 'player' only works with select 'all' (or 'prev')")
    if "zone" in raw and kind not in ("unit", "unit_or_base"):
        ctx.fail("target zone applies to units only")
    zone = _choice(ctx, raw.get("zone", "board"), ZONES, "target.zone")
    flt = None
    if "filter" in raw:
        if kind not in ("unit", "unit_or_base"):
            ctx.fail("target filter applies to units only")
        flt = parse_filter(ctx, raw["filter"], "target.filter")
    count = 1
    if "count" in raw:
        if select != "random":
            ctx.fail("target count only works with select 'random'")
        count = raw["count"]
        if not _is_int(count) or count < 1:
            ctx.fail(f"target.count must be an integer >= 1, got {count!r}")
    return TargetDef(select=select, side=side, kind=kind, zone=zone, filter=flt, count=count, of=of,
                     position=position)


def _bounds(ctx: _Ctx, raw: dict, what: str) -> tuple:
    """(min, max) of a count condition: min defaults to 1 when neither bound is given, 0 when only max is."""
    lo, hi = _opt_int(ctx, raw, "min", what, 0), _opt_int(ctx, raw, "max", what, 0)
    if lo is None:
        lo = 0 if hi is not None else 1
    if hi is not None and lo > hi:
        ctx.fail(f"{what}: min {lo} > max {hi} never holds")
    return lo, hi


def parse_condition(ctx: _Ctx, raw) -> ConditionDef:
    if not isinstance(raw, dict) or "type" not in raw:
        ctx.fail(f"a condition must be an object with a 'type', got {raw!r}")
    ctype = _choice(ctx, raw["type"], CONDITION_TYPES, "condition.type")
    what = f"condition {ctype!r}"
    if ctype == "control":
        raw = _obj(ctx, raw, what, ("type", "side", "zone", "filter", "min", "max"), ("side",))
        lo, hi = _bounds(ctx, raw, what)
        flt = parse_filter(ctx, raw["filter"], f"{what}.filter") if "filter" in raw else None
        return ConditionDef(type=ctype, side=_choice(ctx, raw["side"], SIDES, f"{what}.side"),
                            zone=_choice(ctx, raw.get("zone", "board"), ZONES, f"{what}.zone"), filter=flt,
                            min=lo, max=hi)
    if ctype == "frontline":
        raw = _obj(ctx, raw, what, ("type", "owner"), ("owner",))
        return ConditionDef(type=ctype, owner=_choice(ctx, raw["owner"], ("friendly", "enemy", "none"),
                                                      f"{what}.owner"))
    if ctype == "source_zone":
        if ctx.is_operation:
            ctx.fail(f"{what}: an operation has no source unit")
        raw = _obj(ctx, raw, what, ("type", "zone"), ("zone",))
        return ConditionDef(type=ctype, zone=_choice(ctx, raw["zone"], ("backline", "frontline"), f"{what}.zone"))
    if ctype in ("base_hp", "hand_size"):
        raw = _obj(ctx, raw, what, ("type", "side", "min", "max"), ("side",))
        lo, hi = _opt_int(ctx, raw, "min", what), _opt_int(ctx, raw, "max", what)
        if lo is not None and hi is not None and lo > hi:
            ctx.fail(f"{what}: min > max never holds")
        return ConditionDef(type=ctype, side=_choice(ctx, raw["side"], ("friendly", "enemy"), f"{what}.side"),
                            min=lo, max=hi)
    if ctype == "turn":
        raw = _obj(ctx, raw, what, ("type", "whose"), ("whose",))
        return ConditionDef(type=ctype, whose=_choice(ctx, raw["whose"], ("own", "opponent"), f"{what}.whose"))
    if ctype == "prev":
        raw = _obj(ctx, raw, what, ("type", "killed", "filter"))
        _need_prev(ctx, what)
        flt = parse_filter(ctx, raw["filter"], f"{what}.filter") if "filter" in raw else None
        return ConditionDef(type=ctype, killed=_opt_bool(ctx, raw, "killed", what), filter=flt)
    if ctype == "compare":
        raw = _obj(ctx, raw, what, ("type", "left", "op", "right"), ("left", "op", "right"))
        return ConditionDef(type=ctype, left=parse_amount(ctx, raw["left"], f"{what}.left", minimum=None),
                            op=_choice(ctx, raw["op"], COMPARE_OPS, f"{what}.op"),
                            right=parse_amount(ctx, raw["right"], f"{what}.right", minimum=None))
    raw = _obj(ctx, raw, what, ("type", "event", "side", "window", "min", "max"), ("event", "side", "window"))
    lo, hi = _bounds(ctx, raw, what)  # history
    return ConditionDef(type=ctype, event=_choice(ctx, raw["event"], HISTORY_EVENTS, f"{what}.event"),
                        side=_choice(ctx, raw["side"], SIDES, f"{what}.side"),
                        window=_choice(ctx, raw["window"], HISTORY_WINDOWS, f"{what}.window"), min=lo, max=hi)


def _parse_conditions(ctx: _Ctx, raw: dict) -> tuple:
    cond = raw.get("condition")
    if cond is None and "condition" in raw:
        ctx.fail("condition must be an object or a list of objects")
    if cond is None:
        return ()
    if isinstance(cond, list):
        if not cond:
            ctx.fail("condition list must not be empty")
        return tuple(parse_condition(ctx, c) for c in cond)
    return (parse_condition(ctx, cond),)


def _parse_body(ctx: _Ctx, raw: dict, slot: int, parent_target: Optional[TargetDef], is_else: bool) -> EffectDef:
    """Action, parameters and target of an effect or of its `else` body (which may reuse the effect's target)."""
    trigger, scope = ctx.trigger, ctx.scope
    who = "else" if is_else else "effect"
    action = _choice(ctx, raw["action"], ACTIONS, f"{who} action")
    ctx.action = action
    if ctx.static and action not in STATIC_ACTIONS:
        ctx.fail(f"static effects allow only {sorted(STATIC_ACTIONS)}, not {action!r}")
    params = set(raw) & PARAM_KEYS
    bad = params - ACTION_PARAMS[action]
    if bad:
        ctx.fail(f"action {action!r} takes no {sorted(bad)}")
    missing = REQUIRED_PARAMS.get(action, set()) - params
    if missing:
        ctx.fail(f"action {action!r} needs {sorted(missing)}")
    if ctx.static and "duration" in raw:
        ctx.fail("static effects take no duration")
    own = "target" in raw
    target = parse_target(ctx, raw["target"]) if own else parent_target
    if target.kind not in ACTION_KINDS[action]:
        ctx.fail(f"action {action!r} cannot target kind {target.kind!r} (allowed: {sorted(ACTION_KINDS[action])})")
    if target.select == "chosen" and (trigger not in CHOSEN_TRIGGERS or scope != "self"):
        ctx.fail(f"'chosen' targets need trigger on_play/on_deploy/on_move/on_attack with scope self, "
                 f"not {trigger}/{scope}")
    if target.select == "event" and not has_event_unit(trigger, scope):
        ctx.fail(f"target 'event' needs a trigger with an event unit, not {trigger}/{scope}")
    if target.select == "self":
        if ctx.is_operation:
            ctx.fail("an operation has no source unit to target with 'self'")
        if trigger == "on_death" and scope == "self":  # the source is dead; a watcher may target itself
            ctx.fail("on_death effects may not target 'self'")
    if ctx.static:
        if target.select not in STATIC_SELECTS:
            ctx.fail(f"static effects allow selects {sorted(STATIC_SELECTS)}, not {target.select!r}")
        if target.select == "adjacent" and target.of != "self":
            ctx.fail("static adjacent targets are adjacent to 'self'")

    expr = not ctx.static
    amount = atk = hp = None
    duration = "permanent"
    trait: tuple = ()
    card_id = ""
    move_cost = 0
    uncapped = False
    if action in ("damage", "draw", "discard"):
        amount = parse_amount(ctx, raw["amount"], "amount", allow_expr=expr)
    elif action in ("gain_coins", "increase_max_coins"):
        amount = parse_amount(ctx, raw["amount"], "amount", minimum=None, allow_expr=expr)
    elif action == "heal":
        amount = parse_amount(ctx, raw["amount"], "amount", allow_full=True, allow_expr=expr)
        if "uncapped" in raw:
            uncapped = raw["uncapped"]
            if type(uncapped) is not bool:
                ctx.fail(f"heal.uncapped must be true or false, got {uncapped!r}")
            if uncapped and amount.kind == "full":
                ctx.fail("an uncapped heal needs a number, not 'full' (the cap would be meaningless)")
    elif action == "buff":
        if not {"atk", "hp", "move_cost"} & params:
            ctx.fail("buff needs 'atk', 'hp' and/or 'move_cost'")
        atk = parse_amount(ctx, raw.get("atk", 0), "atk", minimum=None, allow_expr=expr)
        hp = parse_amount(ctx, raw.get("hp", 0), "hp", allow_expr=expr)
        if "move_cost" in raw:
            move_cost = raw["move_cost"]
            if not _is_int(move_cost):
                ctx.fail(f"buff.move_cost must be an integer (a delta), got {move_cost!r}")
    elif action in ("add_trait", "remove_trait"):
        trait = _names(ctx, raw["trait"], TRAITS, "trait")
        if action == "add_trait":
            if len(trait) != 1 or isinstance(raw["trait"], list):
                ctx.fail("add_trait takes one trait name")
            if "amount" in raw and trait[0] != "armor":
                ctx.fail("add_trait takes an amount for armor only")
            if ctx.static and trait[0] == "armor":
                ctx.fail("static add_trait takes a boolean trait (armor has an amount; units carry no static armor)")
            amount = parse_amount(ctx, raw.get("amount", 1), "amount", minimum=1) if trait[0] == "armor" else None
    elif action in ("summon", "add_card"):
        card_id = raw["card"]
        if not isinstance(card_id, str) or not card_id:
            ctx.fail(f"{action}.card must be a card id, got {card_id!r}")
        amount = parse_amount(ctx, raw.get("amount", 1), "amount", minimum=1)
    if "duration" in raw:
        duration = _choice(ctx, raw["duration"], DURATIONS, "duration")
    return EffectDef(index=slot, trigger=trigger, scope=scope, action=action, target=target, amount=amount,
                     atk=atk, hp=hp, duration=duration, trait=trait, card_id=card_id, move_cost=move_cost,
                     uncapped=uncapped, is_else=is_else, own_target=own)


def parse_effect(ctx: _Ctx, slot: int, raw) -> EffectDef:
    """One effect. `ctx.has_prev` must say whether an earlier clause of the card has the same trigger."""
    if not isinstance(raw, dict):
        ctx.fail(f"effect #{slot} must be an object, got {raw!r}")
    extra = set(raw) - EFFECT_KEYS
    if extra:
        ctx.fail(f"effect #{slot}: unknown keys {sorted(map(str, extra))}")
    for key in ("trigger", "target", "action"):
        if key not in raw:
            ctx.fail(f"effect #{slot} is missing {key!r}")
    trigger = _choice(ctx, raw["trigger"], TRIGGERS, "trigger")
    default_scope = "friendly" if trigger in TURN_TRIGGERS else "self"
    scope = _choice(ctx, raw.get("scope", default_scope), SCOPES, "scope")
    if trigger in TURN_TRIGGERS and scope == "self":
        ctx.fail(f"{trigger} takes scope friendly / enemy / any, not 'self'")
    if ctx.is_operation and (trigger != "on_play" or scope != "self"):
        ctx.fail("operations may only use trigger 'on_play' with scope 'self'")
    if not ctx.is_operation and trigger == "on_play" and scope == "self":
        ctx.fail("units may not use on_play with scope 'self' (use friendly / enemy / any to watch operations)")
    static = trigger == "static"
    if static and scope != "self":
        ctx.fail("static effects take scope 'self' (they apply while the source is on the board)")
    ctx.trigger, ctx.scope, ctx.static = trigger, scope, static
    if static:
        for key in ("repeat", "event_filter"):
            if key in raw:
                ctx.fail(f"static effects take no {key!r}")
    event_filter = None
    if "event_filter" in raw:
        if scope == "self":
            ctx.fail("event_filter needs a watcher (scope friendly / enemy / any)")
        if trigger in TURN_TRIGGERS:
            ctx.fail(f"{trigger} has no event unit or card to filter")
        event_filter = parse_filter(ctx, raw["event_filter"], "event_filter", card_only=trigger == "on_play")
    conditions = _parse_conditions(ctx, raw)
    body = _parse_body(ctx, raw, slot, None, False)
    target_condition = None
    if "target_condition" in raw:
        if body.target.kind not in ("unit", "unit_or_base"):
            ctx.fail("target_condition needs a target kind with units")
        target_condition = parse_filter(ctx, raw["target_condition"], "target_condition")
    repeat = 1
    if "repeat" in raw:
        repeat = raw["repeat"]
        if not _is_int(repeat) or repeat < 1:
            ctx.fail(f"repeat must be an integer >= 1, got {repeat!r}")
        if body.target.select == "chosen":
            ctx.fail("repeat needs a target that is not 'chosen' (one choice per repetition is not supported)")
    else_ = None
    if "else" in raw:
        alt = raw["else"]
        if not isinstance(alt, dict):
            ctx.fail(f"else must be an object, got {alt!r}")
        bad = set(alt) - ELSE_KEYS
        if bad:
            ctx.fail(f"else: unknown keys {sorted(map(str, bad))} (an else body has an action, its parameters and "
                     f"an optional target; it may not nest else)")
        if "action" not in alt:
            ctx.fail("else is missing 'action'")
        else_ = _parse_body(ctx, alt, slot, body.target, True)
        if target_condition is not None and else_.own_target and else_.target.select == "chosen":
            ctx.fail("an else body under target_condition cannot choose its own target")
        ctx.action = body.action
    return replace(body, condition=conditions, event_filter=event_filter, target_condition=target_condition,
                   else_=else_, repeat=repeat)


def _card_tags(card_id: str, raw: dict) -> tuple:
    tags = raw.get("tags", [])
    if not isinstance(tags, list) or any(not isinstance(t, str) or not t for t in tags):
        raise ValueError(f"card {card_id!r}: tags must be a list of non-empty strings, got {tags!r}")
    if len(set(tags)) != len(tags):
        raise ValueError(f"card {card_id!r}: tags list a tag twice: {tags!r}")
    return tuple(tags)


def parse_card(index: int, raw: dict) -> CardDef:
    if not isinstance(raw, dict):
        raise ValueError(f"card #{index} must be an object")
    for key in ("id", "type"):
        if key not in raw:
            raise ValueError(f"card #{index} is missing {key!r}")
    card_id = raw["id"]
    if not isinstance(card_id, str) or not card_id:
        raise ValueError(f"card #{index}: id must be a non-empty string, got {card_id!r}")
    if not isinstance(raw.get("name", ""), str) or not isinstance(raw["type"], str):
        raise ValueError(f"card {card_id!r}: name and type must be strings")
    ctype = raw["type"]
    if ctype not in SUPPORTED_TYPES:
        raise ValueError(f"card {card_id!r}: unsupported type {ctype!r}")
    is_op = ctype == OPERATION
    allowed = OPERATION_KEYS if is_op else UNIT_KEYS
    extra = set(raw) - allowed
    if extra:  # e.g. a misspelled "trait", or unit stats on an operation
        raise ValueError(f"card {card_id!r}: unknown fields for a {ctype} {sorted(extra)}")
    required = ("cost", "effects") if is_op else ("nature", "cost", "attack", "health")
    for key in required:
        if key not in raw:
            raise ValueError(f"card {card_id!r} is missing {key!r}")
    token = raw.get("token", False)
    if type(token) is not bool:
        raise ValueError(f"card {card_id!r}: token must be true or false, got {token!r}")
    tags = _card_tags(card_id, raw)
    effects_raw = raw.get("effects", [])
    if not isinstance(effects_raw, list):
        raise ValueError(f"card {card_id!r}: effects must be a list")
    if len(effects_raw) > MAX_EFFECTS:
        raise ValueError(f"card {card_id!r}: at most {MAX_EFFECTS} effects, got {len(effects_raw)}")
    if is_op and not effects_raw:
        raise ValueError(f"card {card_id!r}: an operation needs at least one effect")
    ctx = _Ctx(card_id, is_op)
    effects = []
    seen = set()  # triggers of the earlier clauses (prev needs one with the same trigger)
    for i, e in enumerate(effects_raw):
        trig = e.get("trigger") if isinstance(e, dict) else None
        ctx.has_prev = isinstance(trig, str) and trig in seen
        effects.append(parse_effect(ctx, i, e))
        seen.add(trig)
    effects = tuple(effects)
    cost = _stat(card_id, raw, "cost")
    if cost < 0:
        raise ValueError(f"card {card_id!r}: invalid stats (cost < 0)")
    name = raw.get("name", card_id)
    if is_op:
        return CardDef(index=index, id=card_id, name=name, type=ctype, cost=cost, attack=0, health=0,
                       nature=NO_NATURE, move_cost=0, effects=effects, token=token, tags=tags)

    traits = raw.get("traits", {})
    if not isinstance(traits, dict):
        raise ValueError(f"card {card_id!r}: traits must be an object")
    unknown = set(traits) - SUPPORTED_TRAITS
    if unknown:
        raise ValueError(f"card {card_id!r}: unsupported traits {sorted(map(str, unknown))}")
    flags = {}
    for name_ in BOOL_TRAITS:
        v = traits.get(name_, False)
        if type(v) is not bool:
            raise ValueError(f"card {card_id!r}: trait {name_!r} must be true or false, got {v!r}")
        flags[name_] = v
    armor = traits.get("armor", 0)
    if not _is_int(armor) or armor < 0 or ("armor" in traits and armor < 1):
        raise ValueError(f"card {card_id!r}: trait 'armor' must be a positive integer, got {armor!r}")
    nature = raw["nature"]
    if nature not in NATURES:
        raise ValueError(f"card {card_id!r}: nature must be one of {NATURES}, got {nature!r}")
    card = CardDef(
        index=index, id=card_id, name=name, type=ctype, cost=cost,
        attack=_stat(card_id, raw, "attack"), health=_stat(card_id, raw, "health"),
        nature=NATURES.index(nature),
        move_cost=_stat(card_id, raw, "move_cost") if "move_cost" in raw else DEFAULT_MOVE_COST,
        armor=armor, effects=effects, token=token, tags=tags, **flags,
    )
    if card.attack < 0 or card.health <= 0 or card.move_cost < 0:
        raise ValueError(f"card {card.id!r}: invalid stats")
    return card


def _link_card(c: CardDef, e: EffectDef, by_id: dict) -> EffectDef:
    """Resolve a summon / add_card token id to its card index (also inside an else body)."""
    if e.else_ is not None:
        alt = _link_card(c, e.else_, by_id)
        if alt is not e.else_:
            e = replace(e, else_=alt)
    if not e.card_id:
        return e
    ref = by_id.get(e.card_id)
    if ref is None:
        raise ValueError(f"card {c.id!r}: {e.action} names unknown card {e.card_id!r}")
    if not ref.token:
        raise ValueError(f"card {c.id!r}: {e.action} may only name token cards, {e.card_id!r} is not a token")
    if e.action == "summon" and not ref.is_unit:
        raise ValueError(f"card {c.id!r}: summon needs a unit token, {e.card_id!r} is an operation")
    return replace(e, card=ref.index)


def build_card_pool(raw_cards) -> CardPool:
    """A validated pool from a list of card dicts (file order = card index)."""
    if not isinstance(raw_cards, list) or not raw_cards:
        raise ValueError("'cards' must be a non-empty list")
    cards = [parse_card(i, c) for i, c in enumerate(raw_cards)]
    ids = [c.id for c in cards]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate card ids {sorted({i for i in ids if ids.count(i) > 1})}")
    by_id = {c.id: c for c in cards}
    for k, c in enumerate(cards):  # second pass: summon / add_card name existing token cards
        effects = tuple(_link_card(c, e, by_id) for e in c.effects)
        if effects != c.effects:
            cards[k] = replace(c, effects=effects)
    return CardPool(tuple(cards))


def load_card_pool(path: Path | str = DEFAULT_CARDS) -> CardPool:
    raw = _load_json(path, frozenset({"cards"}))
    return build_card_pool(raw["cards"])


def sample_decks(seed: int, n_decks: int) -> tuple:
    """Deck pair for a deal seed, from its own stream (independent of the game RNG; platform-stable)."""
    r = random.Random(f"decks:{seed}")
    return (r.randrange(n_decks), r.randrange(n_decks))


def build_deck(pool: CardPool, counts: dict, max_copies: Optional[int] = None) -> tuple:
    if not isinstance(counts, dict):
        raise ValueError("deck 'cards' must be an object mapping card id -> count")
    deck = []
    for card_id, n in counts.items():
        if type(n) is not int or n < 1:
            raise ValueError(f"deck count for {card_id!r} must be a positive integer, got {n!r}")
        if max_copies is not None and n > max_copies:
            raise ValueError(f"deck has {n} copies of {card_id!r}; at most {max_copies} allowed")
        try:
            c = pool.by_id(card_id)
        except KeyError:
            raise ValueError(f"deck references unknown card id {card_id!r}") from None
        if c.token:
            raise ValueError(f"deck references token card {card_id!r}; tokens never appear in decks")
        deck.extend([c.index] * n)
    return tuple(sorted(deck))


def build_ruleset(cards, decks, **overrides) -> GameConfig:
    """GameConfig from in-memory card dicts and deck dicts ({"name"?, "style"?, "cards": {id: n}}), with
    the same validation as `load_ruleset`."""
    pool = build_card_pool(cards)
    if not isinstance(decks, list) or not decks:
        raise ValueError("'decks' must be a non-empty list")
    for d in decks:
        if not isinstance(d, dict) or "cards" not in d or set(d) - {"name", "style", "cards"}:
            raise ValueError(f"each deck must be an object with 'cards' (and optional 'name', 'style'), got {d!r}")
        if not all(isinstance(d.get(k, ""), str) for k in ("name", "style")):
            raise ValueError(f"deck name and style must be strings, got {d!r}")
    max_copies = overrides.get("max_copies", GameConfig.__dataclass_fields__["max_copies"].default)
    built = tuple(build_deck(pool, d["cards"], max_copies) for d in decks)
    names = tuple(d.get("name", f"deck{i}") for i, d in enumerate(decks))
    styles = tuple(d.get("style", "") for d in decks)
    config = GameConfig(cards=pool, decks=built, deck_names=names, deck_styles=styles, **overrides)
    for name, deck in zip(names, built):
        if len(deck) != config.deck_size:
            raise ValueError(f"deck {name!r} has {len(deck)} cards, expected {config.deck_size}")
    return config


def load_ruleset(cards_path: Path | str | None = None, decks_path: Path | str | None = None,
                 **overrides) -> GameConfig:
    cards = _load_json(cards_path or DEFAULT_CARDS, frozenset({"cards"}))["cards"]
    decks: Sequence[dict] = _load_json(decks_path or DEFAULT_DECKS, frozenset({"decks"}))["decks"]
    return build_ruleset(cards, decks, **overrides)


# ---------------------------------------------------------------- random decks and deals (SPEC §1.3)
BUCKET_TARGETS = (4, 7, 7, 6, 5, 4, 3, 4)  # cost 1, 2, ..., 7, 8+ (sums to 40)
MAX_OPERATIONS = 12


def cost_bucket(cost: int) -> int:
    """Bucket index 0..7 for costs 1..7 and 8+ (cost 0 counts as cost 1)."""
    return min(max(cost, 1), 8) - 1


def bucket_targets(deck_size: int) -> list:
    """Target count per cost bucket: the SPEC numbers for 40 cards, scaled (largest remainder) otherwise."""
    total = sum(BUCKET_TARGETS)
    if deck_size == total:
        return list(BUCKET_TARGETS)
    exact = [t * deck_size / total for t in BUCKET_TARGETS]
    out = [int(x) for x in exact]
    order = sorted(range(len(exact)), key=lambda i: (-(exact[i] - out[i]), i))
    for i in order[:deck_size - sum(out)]:
        out[i] += 1
    return out


def _deckgen_info(pool: CardPool) -> tuple:
    """(buckets of non-token card indices, is_operation flags), cached on the pool."""
    info = pool.__dict__.get("_deckgen")
    if info is None:
        buckets = [[] for _ in BUCKET_TARGETS]
        for c in pool.cards:
            if not c.token:
                buckets[cost_bucket(c.cost)].append(c.index)
        info = (tuple(tuple(b) for b in buckets), tuple(c.is_operation for c in pool.cards))
        object.__setattr__(pool, "_deckgen", info)
    return info


def _nearest(buckets_ok: list, b: int) -> Optional[int]:
    """Nearest bucket index to b satisfying the predicate list (ties: the cheaper bucket)."""
    n = len(buckets_ok)
    for dist in range(1, n):
        for cand in (b - dist, b + dist):
            if 0 <= cand < n and buckets_ok[cand]:
                return cand
    return None


def _required_counts(pool: CardPool, config: GameConfig, required) -> list:
    n = len(pool)
    counts = [0] * n
    if required is None:
        return counts
    items = required.items() if isinstance(required, Mapping) else enumerate(required)
    for c, k in items:
        if isinstance(c, bool) or isinstance(k, bool):
            raise ValueError(f"required entries must be integers, got {c!r}: {k!r}")
        try:
            c, k = operator.index(c), operator.index(k)
        except TypeError:
            raise ValueError(f"required entries must be integers, got {c!r}: {k!r}") from None
        if not 0 <= c < n:
            raise ValueError(f"required card index {c} out of range")
        if k < 0:
            raise ValueError(f"required count for card {c} is negative")
        if k and pool.cards[c].token:
            raise ValueError(f"required card {pool.cards[c].id!r} is a token; tokens never appear in decks")
        counts[c] += k
        if counts[c] > config.max_copies:
            raise ValueError(f"required {counts[c]} copies of {pool.cards[c].id!r}; at most {config.max_copies}")
    if sum(counts) > config.deck_size:
        raise ValueError(f"required cards ({sum(counts)}) exceed the deck size {config.deck_size}")
    return counts


def generate_deck(rng: random.Random, config: GameConfig, required=None) -> tuple:
    """A legal random deck (SPEC §1.3): `deck_size` cards, <= max_copies each, no tokens, sorted by card index.

    `required` (counts per card index: a sequence or a {index: count} mapping) is placed first and counts
    towards its buckets' targets; a bucket that is already over its target takes the excess from the
    nearest buckets that still have quota (ties: the cheaper bucket). The rest is drawn bucket by bucket
    in cost order, one copy at a time, uniformly among the bucket's non-token cards with copies left.
    Operations are capped at 12 (required operations count towards the cap). A bucket with no card left
    hands its remaining quota to the nearest bucket that has cards left (ties: the cheaper bucket).
    """
    pool = config.cards
    buckets, is_op = _deckgen_info(pool)
    counts = _required_counts(pool, config, required)
    maxc = config.max_copies
    quota = bucket_targets(config.deck_size)
    for c, k in enumerate(counts):
        if k:
            quota[cost_bucket(pool.cards[c].cost)] -= k
    while True:  # over-full buckets (required cards) take their excess from the nearest buckets with quota
        b = next((i for i, q in enumerate(quota) if q < 0), None)
        if b is None:
            break
        nb = _nearest([q > 0 for q in quota], b)
        take = min(-quota[b], quota[nb])
        quota[b] += take
        quota[nb] -= take
    n_ops = sum(k for c, k in enumerate(counts) if is_op[c])
    ops_open = n_ops < MAX_OPERATIONS
    elig = [[c for c in bucket if counts[c] < maxc and (ops_open or not is_op[c])] for bucket in buckets]
    while True:
        b = next((i for i, q in enumerate(quota) if q > 0), None)
        if b is None:
            break
        lst = elig[b]
        if not lst:
            nb = _nearest([bool(x) for x in elig], b)
            if nb is None:
                raise ValueError("the card pool cannot fill a legal deck")
            quota[nb] += quota[b]
            quota[b] = 0
            continue
        c = rng.choice(lst)
        counts[c] += 1
        quota[b] -= 1
        if counts[c] >= maxc:
            lst.remove(c)
        if is_op[c]:
            n_ops += 1
            if n_ops >= MAX_OPERATIONS:
                for x in elig:
                    x[:] = [k for k in x if not is_op[k]]
    return tuple(c for c, k in enumerate(counts) for _ in range(k))


def deck_rng(seed: int, seat: int) -> random.Random:
    """The per-deal random-deck stream for one seat."""
    return random.Random(f"deck:{seed}:{seat}")


def sample_deal(seed: int, config: GameConfig, random_frac: float) -> tuple:
    """(deck0, deck1) deck specs for a deal seed: each seat independently gets a random deck
    (`generate_deck(deck_rng(seed, seat))`) with probability `random_frac`, else a fixed deck index.

    The choice uses `random.Random(f"deal:{seed}")`, which draws `random()` then `randrange(n_decks)` for
    seat 0 and then for seat 1 (both always drawn, so a seat's fixed index does not depend on the other
    seat's outcome)."""
    if isinstance(seed, bool):
        raise TypeError(f"seed must be an integer, got {seed!r}")
    seed = operator.index(seed)
    if not 0.0 <= random_frac <= 1.0:
        raise ValueError(f"random_frac must be in [0, 1], got {random_frac!r}")
    r = random.Random(f"deal:{seed}")
    out = []
    for seat in (0, 1):
        u = r.random()
        idx = r.randrange(config.n_decks)
        out.append(generate_deck(deck_rng(seed, seat), config) if u < random_frac else idx)
    return tuple(out)
