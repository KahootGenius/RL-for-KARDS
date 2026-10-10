"""Black-box helpers for the independent Stage 3 rule tests.

Everything here is derived from SPEC.md only:
* card dicts follow SPEC 1.1/1.2 (plus the 1.2b extensions: `tags`, `adjacent` targets, `else`
  bodies, `static` effects) and are loaded with `cards.build_ruleset(cards, decks, ...)`;
* positions are built by editing the public state of SPEC 4 (Stage 2 fields plus the Stage 3
  ones) and calling `game.invalidate()`;
* action indices are computed from the numbers of SPEC 3 (not read from ActionSpace), so the tests
  also pin the layout down.

The module never relies on engine internals: units are created with `Unit.from_card` and their
SPEC 4 fields are set by name; the queue and pending choice are only observed through
`legal_actions()`, `observe()` and `len(game.queue)`.
"""
from __future__ import annotations

import os
import random
import sys
from collections import Counter

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
for _path in (ROOT, TESTS_DIR):  # works under any pytest import mode
    if _path not in sys.path:
        sys.path.insert(0, _path)

import cardgame.cards as cards_mod  # noqa: E402
import cardgame.engine as engine_mod  # noqa: E402
from cardgame.engine import DRAW, Game, IllegalActionError, Unit  # noqa: E402,F401

# SPEC 4: game.phase  # MULLIGAN=0, MAIN=1, CHOICE=2 (module constants). test_choices asserts that the
# constants exist; the fallback only keeps the other tests meaningful while the engine is unfinished.
MULLIGAN = getattr(engine_mod, "MULLIGAN", 0)
MAIN = getattr(engine_mod, "MAIN", 1)
CHOICE = getattr(engine_mod, "CHOICE", 2)

# ---------------------------------------------------------------- SPEC 3 action layout (H = 10, Z = 5)
H, Z = 10, 5
N_ACTIONS = 154
END = 0
MOVE0 = 1 + H                      # 11
ATTACK0 = 1 + H + Z                # 16
CHOOSE0 = ATTACK0 + 2 * Z * (2 * Z + 1)  # 126
MULLIGAN0 = CHOOSE0 + 3 * Z + 2    # 143
CONFIRM = MULLIGAN0 + H            # 153
BASE = 2 * Z                       # ATTACK target slot of the enemy base == ENEMY_BASE_CHOICE
ENEMY_BASE = 2 * Z                 # CHOOSE slot 2Z
OWN_BASE = 3 * Z + 1               # CHOOSE slot 3Z+1


def play(i: int) -> int:
    return 1 + i


def move(j: int) -> int:
    return MOVE0 + j


def attack(a: int, t: int) -> int:
    return ATTACK0 + a * (2 * Z + 1) + t


def choose(t: int) -> int:
    return CHOOSE0 + t


def mull(i: int) -> int:
    return MULLIGAN0 + i


def back(j: int) -> int:
    """Attacker slot of own backline j / ATTACK and CHOOSE slot of the enemy backline j."""
    return j


def front(j: int) -> int:
    """Attacker / target / CHOOSE slot of frontline position j (either owner)."""
    return Z + j


def own_back(j: int) -> int:
    """CHOOSE slot of the chooser's own backline position j (SPEC 2.9: 2Z+1+j)."""
    return 2 * Z + 1 + j


def chooses(game) -> list:
    """The CHOOSE slots t currently offered."""
    return [a - CHOOSE0 for a in game.legal_actions() if CHOOSE0 <= a < MULLIGAN0]


# ---------------------------------------------------------------- card dicts (SPEC 1.1 / 1.2)
def unit(cid, atk=1, hp=1, *, nature="troop", cost=1, effects=(), traits=None, move_cost=None, token=False,
         tags=None):
    d = {"id": cid, "name": cid, "type": "unit", "nature": nature, "cost": cost, "attack": atk, "health": hp}
    if move_cost is not None:
        d["move_cost"] = move_cost
    if traits:
        d["traits"] = dict(traits)
    if token:
        d["token"] = True
    if tags is not None:  # SPEC 1.2b card attribute
        d["tags"] = list(tags)
    d["effects"] = [dict(e) for e in effects]
    return d


def operation(cid, effects, *, cost=1, token=False, tags=None):
    d = {"id": cid, "name": cid, "type": "operation", "cost": cost, "effects": [dict(e) for e in effects]}
    if token:
        d["token"] = True
    if tags is not None:
        d["tags"] = list(tags)
    return d


def effect(trigger, action, target, *, scope=None, amount=None, condition=None, **params):
    e = {"trigger": trigger}
    if scope is not None:
        e["scope"] = scope
    if condition is not None:
        e["condition"] = condition
    e["target"] = target
    e["action"] = action
    if amount is not None:
        e["amount"] = amount
    e.update(params)
    return e


def tgt(select, side=None, kind=None, zone=None, filter=None, count=None):
    """Target object (SPEC 1.2 TG)."""
    t = {"select": select}
    if side is not None:
        t["side"] = side
    if kind is not None:
        t["kind"] = kind
    if zone is not None:
        t["zone"] = zone
    if filter is not None:
        t["filter"] = filter
    if count is not None:
        t["count"] = count
    return t


def adjacent(side, of="self", position=None, filter=None):
    """SPEC 1.2b `select: "adjacent"` target (side filters still apply)."""
    t = {"select": "adjacent", "side": side, "of": of}
    if position is not None:
        t["position"] = position
    if filter is not None:
        t["filter"] = filter
    return t


def with_else(e, **body):
    """SPEC 1.2b: attach an `else` body (`else` is a Python keyword, so it cannot be a kwarg)."""
    e = dict(e)
    e["else"] = dict(body)
    return e


def static(action, target, *, condition=None, **params):
    """SPEC 1.2b `static` effect (no scope, no duration)."""
    e = {"trigger": "static"}
    if condition is not None:
        e["condition"] = condition
    e["target"] = target
    e["action"] = action
    e.update(params)
    return e


# 14 vanilla filler cards: a legal 40-card deck needs >= 14 distinct cards (<= 3 copies each).
FILLER = [unit(f"fill{i:02d}", 1 + i % 3, 1 + i % 4, cost=1 + i % 8) for i in range(14)]
FILLER_IDS = [c["id"] for c in FILLER]


def filler_deck(name="filler", offset=0):
    """A legal 40-card deck of filler cards (13 x 3 + 1)."""
    ids = FILLER_IDS[offset:] + FILLER_IDS[:offset]
    counts = {cid: 3 for cid in ids[:13]}
    counts[ids[13]] = 1
    return {"name": name, "style": "", "cards": counts}


class Rules:
    """A ruleset built from in-memory card dicts (SPEC 1.5 build_ruleset); filler cards come first."""

    def __init__(self, extra=(), *, mulligan=False, decks=None, **overrides):
        self.card_dicts = list(FILLER) + list(extra)
        ids = [c["id"] for c in self.card_dicts]
        assert len(set(ids)) == len(ids), "duplicate test card ids"
        self.index = {cid: i for i, cid in enumerate(ids)}  # SPEC 1.1: indexed by list order
        self.deck_dicts = decks if decks is not None else [filler_deck("a"), filler_deck("b", 1)]
        self.config = cards_mod.build_ruleset(self.card_dicts, self.deck_dicts, mulligan=mulligan, **overrides)
        self.n_cards = len(self.card_dicts)

    def idx(self, cid) -> int:
        return cid if isinstance(cid, int) else self.index[cid]

    def card(self, cid):
        return self.config.cards[self.idx(cid)]


# ---------------------------------------------------------------- positions (SPEC 4 state)
UNIT_FIELDS = ("card", "owner", "uid", "atk", "hp", "max_hp", "armor", "defense", "nature", "move_cost",
               "summoned", "moved", "attacked", "blitz", "smokescreen", "fury", "pinned", "pin_until", "attacks",
               "temp_atk", "temp_hp", "temp_armor", "temp_traits", "temp_removed", "base_traits", "token",
               # SPEC 4 phase-1b additions
               "ambush", "shock", "immune", "ambush_used", "static_atk", "static_hp", "static_move_cost",
               "static_traits")


def set_field(u, name, value) -> None:
    setattr(u, name, value)
    if name == "attacks":  # SPEC 2.4: attacked = attacks > 0 (may be a derived property)
        try:
            u.attacked = value > 0
        except AttributeError:
            pass


def blank(rules: Rules, current=0, first=None, round_=1, coins=10, seed=0, turn=None) -> Game:
    """A running game in phase MAIN with empty hands, decks and board (SPEC 4 state edits).

    `turn` defaults to the index the turn would have in a real game (turn 1 = first player's
    round-1 turn, SPEC 2.3 `turn += 1` at every turn start).
    """
    first = current if first is None else first
    g = Game(rules.config)
    g.reset(seed, decks=(0, 1))
    assert g.phase == MAIN and g.pending is None and len(g.queue) == 0
    g.first_player, g.current, g.round = first, current, round_
    g.turn = turn if turn is not None else 2 * (round_ - 1) + (1 if current == first else 2)
    g.hands = [[], []]
    g.deck_cards = [[], []]
    g.coins = [0, 0]
    g.coins[current] = coins
    g.base_hp = [rules.config.base_hp, rules.config.base_hp]
    g.burned = [0, 0]
    g.backline = [[], []]
    g.frontline = []
    g.front_owner = None
    g.done, g._winner = False, None
    g.invalidate()
    return g


def put(game: Game, rules: Rules, owner: int, zone: str, cid, *, ready=True, **fields) -> Unit:
    """Append a unit made by `Unit.from_card` to owner's backline ("back") or the frontline ("front").

    `ready=True` clears `summoned` (the unit was deployed on an earlier turn). Other SPEC 4 unit
    fields can be overridden by keyword.
    """
    u = Unit.from_card(rules.card(cid), owner, game.next_uid)
    game.next_uid += 1
    if ready:
        u.summoned = False
    for k, v in fields.items():
        set_field(u, k, v)
    if zone == "back":
        game.backline[owner].append(u)
    elif zone == "front":
        assert game.front_owner in (None, owner)
        game.frontline.append(u)
        game.front_owner = owner
    else:
        raise ValueError(zone)
    game.invalidate()
    return u


def set_hand(game: Game, rules: Rules, p: int, cids) -> None:
    game.hands[p] = sorted(rules.idx(c) for c in cids)
    game.invalidate()


def set_deck(game: Game, rules: Rules, p: int, cids) -> None:
    """Deck contents, top = end of the list (SPEC 2.1)."""
    game.deck_cards[p] = [rules.idx(c) for c in cids]
    game.invalidate()


def hand_slot(game: Game, rules: Rules, p: int, cid) -> int:
    return list(game.hands[p]).index(rules.idx(cid))


def bases_of(game: Game) -> list:
    return [int(x) for x in game.base_hp]


def all_units(game: Game) -> list:
    return [*game.backline[0], *game.frontline, *game.backline[1]]


def find(game: Game, uid: int):
    for u in all_units(game):
        if u.uid == uid:
            return u
    return None


def zone_of(game: Game, uid: int):
    for p in (0, 1):
        if any(u.uid == uid for u in game.backline[p]):
            return ("back", p)
    if any(u.uid == uid for u in game.frontline):
        return ("front", game.front_owner)
    return None


def uids(units) -> list:
    return [u.uid for u in units]


def board_order(game: Game) -> list:
    """SPEC 2.8 board order: the turn player's backline, then frontline (if theirs), then the
    opponent's backline, then frontline (if theirs)."""
    p = game.current
    o = 1 - p
    out = list(game.backline[p])
    if game.front_owner == p:
        out += game.frontline
    out += game.backline[o]
    if game.front_owner == o:
        out += game.frontline
    return out


def cards_of(game: Game, p: int, rules: Rules, zone="back") -> list:
    """Card ids of a zone, in slot order."""
    units = game.backline[p] if zone == "back" else game.frontline
    ids = {i: cid for cid, i in rules.index.items()}
    return [ids[u.card] for u in units]


def counts(x, n: int) -> list:
    """Normalise a 'counts per card index' value (list, tuple, array or mapping) to a list of n ints."""
    if hasattr(x, "items"):
        return [int(x.get(i, 0)) for i in range(n)]
    return [int(v) for v in list(x)] + [0] * (n - len(list(x)))


def multiset(cards) -> Counter:
    return Counter(int(c) for c in cards)


def rng_copy(game: Game) -> random.Random:
    r = random.Random()
    r.setstate(game.rng.getstate())
    return r


def predict_sample(state, population, k):
    """SPEC 2.9: rng.sample(candidates, count) replayed on a copy of the RNG state."""
    r = random.Random()
    r.setstate(state)
    return r.sample(list(population), k)


def plain(x):
    """Comparable, hashable normal form of state values."""
    if hasattr(x, "tolist") and not isinstance(x, (int, float, bool, str)):
        return plain(x.tolist())
    if isinstance(x, dict):
        return tuple(sorted((k, plain(v)) for k, v in x.items()))
    if isinstance(x, (set, frozenset)):
        return tuple(sorted(x))
    if isinstance(x, (list, tuple)):
        return tuple(plain(v) for v in x)
    return x


def unit_state(u) -> tuple:
    return tuple(plain(getattr(u, f, None)) for f in UNIT_FIELDS)


GAME_FIELDS = ("first_player", "current", "round", "turn", "phase", "deck_ids", "decklists", "deck_cards", "hands",
               "coins", "coin_bonus", "base_hp", "burned", "discard", "graveyard", "known_hand", "revealed",
               "mulligan_marks", "mulligan_done", "front_owner", "next_uid", "done", "guard_trips")


def state_key(game: Game, rng=True) -> tuple:
    """Every SPEC 4 state field (hidden ones included), the board, the public views and the RNG."""
    key = [(f, plain(getattr(game, f, None))) for f in GAME_FIELDS]
    key.append(("backline", tuple(tuple(unit_state(u) for u in z) for z in game.backline)))
    key.append(("frontline", tuple(unit_state(u) for u in game.frontline)))
    key.append(("winner", game.winner()))
    key.append(("queue_len", len(game.queue)))
    key.append(("pending", game.pending is not None))
    key.append(("obs", game.observe(0), game.observe(1)))
    key.append(("legal", tuple(game.legal_actions())))
    if rng:
        key.append(("rng", game.rng.getstate()))
    return tuple(key)


def step_all(game: Game, actions) -> None:
    for a in actions:
        assert a in game.legal_actions(), (a, game.legal_actions())
        game.step(a)


def random_playout(game: Game, rng: random.Random, max_steps=4000, until=None) -> None:
    for _ in range(max_steps):
        if game.done or (until is not None and until(game)):
            return
        game.step(rng.choice(game.legal_actions()))


def pick(game: Game, rng: random.Random, hoard=False) -> int:
    """A seeded random legal action; `hoard` mostly ends the turn, so hands fill up and burn."""
    legal = game.legal_actions()
    if hoard and END in legal and rng.random() < 0.85:
        return END
    return rng.choice(legal)


# ---------------------------------------------------------------- a richer pool for hidden-info tests
def rich_cards():
    """Filler plus effect cards that move cards between hidden and public zones (draw, discard,
    add_card, return_to_hand, random damage, summon) and two tokens."""
    rnd_enemy = tgt("random", "enemy", "unit")
    return [
        unit("scout", 1, 1, cost=1, effects=[effect("on_deploy", "draw", "controller", amount=1)]),
        unit("spy", 1, 2, cost=2, effects=[effect("on_deploy", "discard", "opponent", amount=1)]),
        unit("quartermaster", 1, 2, cost=2, effects=[effect("on_deploy", "add_card", "controller", card="ration")]),
        unit("sniper", 1, 2, nature="ranged", cost=2, effects=[effect("on_deploy", "damage", rnd_enemy, amount=1)]),
        unit("bouncer", 2, 2, cost=2,
             effects=[effect("on_deploy", "return_to_hand", tgt("chosen", "any", "unit"))]),
        unit("bugler", 1, 3, cost=2, effects=[effect("on_death", "summon", "controller", card="militia")]),
        operation("intel", [effect("on_play", "draw", "controller", amount=2)], cost=1),
        operation("sabotage", [effect("on_play", "discard", "opponent", amount=1)], cost=1),
        operation("volley", [effect("on_play", "damage", tgt("random", "enemy", "unit", count=2), amount=1)], cost=1),
        operation("recall", [effect("on_play", "return_to_hand", tgt("chosen", "friendly", "unit"))], cost=1),
        operation("ration", [effect("on_play", "heal", "friendly_base", amount=2)], cost=0, token=True),
        unit("militia", 1, 1, cost=1, token=True),
    ]


RICH_DECKS = [
    {"name": "rich_a", "style": "",
     "cards": {**{FILLER_IDS[i]: 2 for i in range(10)}, "scout": 3, "spy": 2, "quartermaster": 3, "sniper": 3,
               "bouncer": 3, "bugler": 2, "intel": 2, "sabotage": 2}},
    {"name": "rich_b", "style": "",
     "cards": {**{FILLER_IDS[i]: 2 for i in range(4, 14)}, "intel": 3, "volley": 3, "recall": 3, "sabotage": 2,
               "scout": 2, "sniper": 2, "bugler": 3, "spy": 2}},
]


def rich_rules(mulligan=True, **overrides) -> Rules:
    return Rules(rich_cards(), mulligan=mulligan, decks=RICH_DECKS, **overrides)


def perturb_hidden(game: Game, observer: int, rng: random.Random, new_decklist=False) -> Game:
    """A clone that differs from `game` only in what SPEC 5 hides from `observer`.

    The opponent's unknown hand cards (hand minus `known_hand[o]`) and deck are re-dealt (same sizes)
    from the same unknown pool, or, with `new_decklist`, from a fresh legal decklist that contains
    `revealed[o]` (deck choice hidden: `deck_ids[o] = -1`). The opponent's mulligan marks (while the
    opponent decides) and the RNG are replaced. `revealed`/`known_hand` stay as they are.
    """
    g = game.clone()
    o = 1 - observer
    n = len(g.config.cards)
    known = counts(g.known_hand[o], n)
    rest = multiset(g.hands[o])
    known_cards = []
    for c in range(n):
        if known[c]:
            assert rest[c] >= known[c], "known_hand counts cards that are not in the hand"
            known_cards += [c] * known[c]
            rest[c] -= known[c]
    unknown_hand = list(rest.elements())
    n_deck = len(g.deck_cards[o])
    if new_decklist:
        revealed = counts(g.revealed[o], n)
        deck = cards_mod.generate_deck(rng, g.config, required=g.revealed[o])
        pool = multiset(deck)
        pool.subtract({c: revealed[c] for c in range(n)})
        assert min(pool.values(), default=0) >= 0
        pool = list(pool.elements())
        dl = list(g.decklists)
        dl[o] = tuple(sorted(deck))
        g.decklists = type(g.decklists)(dl) if isinstance(g.decklists, tuple) else dl
        ids = list(g.deck_ids)
        ids[o] = -1
        g.deck_ids = type(g.deck_ids)(ids) if isinstance(g.deck_ids, tuple) else ids
    else:
        pool = unknown_hand + list(g.deck_cards[o])
    rng.shuffle(pool)
    assert len(pool) >= len(unknown_hand) + n_deck
    g.hands[o] = sorted(known_cards + pool[:len(unknown_hand)])
    g.deck_cards[o] = pool[len(unknown_hand):len(unknown_hand) + n_deck]
    if g.phase == MULLIGAN and g.current == o:
        marks = {i for i in range(len(g.hands[o])) if rng.random() < 0.5}
        g.mulligan_marks = type(g.mulligan_marks)(marks)
    g.rng = random.Random(rng.getrandbits(64))
    g.invalidate()
    return g
