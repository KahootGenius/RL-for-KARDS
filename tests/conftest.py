"""Shared fixtures and helpers for the engine tests (engine-only: no `cardgame.agents`/`cardgame.rl`).

Configs:
* `CONFIG` — the shipped ruleset with `mulligan=False`, so Stage 2-style tests keep their meaning.
* `VANILLA_CONFIG` — the frozen Stage 2 content (`tests/fixtures/stage2_*.json`, effect-free) with
  `mulligan=False`; the Stage 2 reference-model differential tests run on it, so they keep working
  when the shipped decks gain effect cards.

Building positions (SPEC §4 state):
* `blank_game(current, first, round_, coins)` — a running game (phase MAIN) with empty hands, decks
  and board.
* `add_unit(game, owner, "back" | "front", card_id=None, **fields)` — appends a unit that can act
  (flags False) with a fresh uid. With a card id the stats and keywords come from the card; without
  one, pass explicit stats (`atk`, `hp`, optional `nature`, `defense`, `armor`, `move_cost`, keywords,
  flags) and a unit card of the same nature/traits is used for the `card` field. Any `Unit` field may
  be overridden (`attacked=True` sets `attacks=1`). Card stats may be retuned: tests never hard-code a
  card's numbers.
* `set_hand(game, player, cards)` — card ids or indices (looked up in the game's own pool), sorted.
Action helpers: `END`, `play(i)`, `move(j)`, `attack(a, t)` with slots `back(j)`, `front(j)`, `BASE`,
`choose(t)`, `mulligan(i)`, `CONFIRM`.
Fuzzing: `iter_states(seed, profiles, decks)` plays a seeded game with biased policies (`PROFILES`).
"""
from __future__ import annotations

import operator
import os
import random
import sys
from itertools import product
from typing import Callable, Iterator, Optional, Sequence

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
for _path in (ROOT, TESTS_DIR):  # works under any pytest import mode
    if _path not in sys.path:
        sys.path.insert(0, _path)

from cardgame.actions import ActionKind, ActionSpace  # noqa: E402
from cardgame.cards import FAST, RANGED, TROOP, GameConfig, load_ruleset  # noqa: E402
from cardgame.engine import DRAW, MAIN, Game, Unit  # noqa: E402

FIXTURES = os.path.join(TESTS_DIR, "fixtures")
CONFIG: GameConfig = load_ruleset(mulligan=False)
VANILLA_CONFIG: GameConfig = load_ruleset(os.path.join(FIXTURES, "stage2_cards.json"),
                                          os.path.join(FIXTURES, "stage2_decks.json"), mulligan=False)
SPACE = ActionSpace(CONFIG.max_hand_size, CONFIG.zone_capacity)
NUM_ACTIONS = SPACE.n
H, Z = CONFIG.max_hand_size, CONFIG.zone_capacity
N_CARDS = len(CONFIG.cards)
N_DECKS = CONFIG.n_decks
DECK_PAIRS = tuple(product(range(N_DECKS), repeat=2))
END = SPACE.END_TURN
BASE = SPACE.BASE_TARGET
CONFIRM = SPACE.CONFIRM
# Every Unit slot (SPEC §4, §1.2b). `attacked` is derived (attacks > 0), so it is not a slot.
UNIT_FIELDS = ("card", "owner", "uid", "atk", "hp", "max_hp", "armor", "defense", "nature", "move_cost",
               "summoned", "moved", "attacks", "blitz", "smokescreen", "fury", "pinned", "pin_until",
               "temp_atk", "temp_hp", "temp_armor", "temp_traits", "temp_removed", "base_traits", "token",
               "ambush", "shock", "immune", "ambush_used", "temp_move_cost", "static_atk", "static_hp",
               "static_move_cost", "static_traits")
STAGE2_UNIT_FIELDS = ("card", "owner", "uid", "atk", "hp", "max_hp", "armor", "defense", "nature", "move_cost",
                      "summoned", "moved", "attacked")


@pytest.fixture(scope="session")
def config() -> GameConfig:
    return CONFIG


# ---------------------------------------------------------------- cards and actions
def card(card_id, config: Optional[GameConfig] = None) -> int:
    """Card index for a card id in `config` (default CONFIG); an index is returned unchanged."""
    return card_id if isinstance(card_id, int) else (config or CONFIG).cards.by_id(card_id).index


def cost(card_index: int, config: Optional[GameConfig] = None) -> int:
    return (config or CONFIG).cards[card_index].cost


def cards_where(nature: Optional[int] = None, defense: Optional[bool] = None,
                armor: Optional[bool] = None, pred: Optional[Callable] = None,
                config: Optional[GameConfig] = None, units_only: bool = True) -> list:
    """Indices of pool cards with the given nature / Defense / (armor > 0) / predicate. By default only
    non-token unit cards (what a deck can deploy)."""
    return [c.index for c in (config or CONFIG).cards.cards
            if (not units_only or (c.is_unit and not c.token))
            and (nature is None or c.nature == nature) and (defense is None or c.defense == defense)
            and (armor is None or (c.armor > 0) == armor) and (pred is None or pred(c))]


def card_for(nature: int = TROOP, defense: bool = False, armor: bool = False,
             config: Optional[GameConfig] = None) -> int:
    """A unit card of the given nature/trait combination (any unit of that nature as a fallback)."""
    found = (cards_where(nature, defense, armor, config=config) or cards_where(nature, config=config)
             or cards_where(config=config, units_only=False))
    return found[0]


def play(i: int) -> int:
    return SPACE.PLAY0 + i


def move(j: int) -> int:
    return SPACE.MOVE0 + j


def back(j: int) -> int:
    """Attacker/target slot of backline position j."""
    return j


def front(j: int) -> int:
    """Attacker/target slot of frontline position j."""
    return Z + j


def attack(a: int, t: int) -> int:
    return SPACE.attack(a, t)


def choose(t: int) -> int:
    return SPACE.choose(t)


def mulligan(i: int) -> int:
    return SPACE.mulligan(i)


def act(kind: ActionKind, a: int = -1, b: int = -1) -> int:
    return SPACE.encode(kind, a, b)


# ---------------------------------------------------------------- games
def new_game(seed: int = 0, decks: Optional[Sequence[int]] = None, config: GameConfig = CONFIG) -> Game:
    g = Game(config)
    g.reset(seed, decks)
    return g


def all_units(game: Game) -> list:
    return [*game.backline[0], *game.backline[1], *game.frontline]


_unit_fields = operator.attrgetter(*UNIT_FIELDS)
_PLAIN_UNIT = set(Unit.__slots__) == set(UNIT_FIELDS)


def unit_key(u: Unit) -> tuple:
    """Every slot of a unit (including slots a subclass adds)."""
    if type(u) is Unit and _PLAIN_UNIT:
        return _unit_fields(u)
    names = [n for klass in type(u).__mro__ for n in getattr(klass, "__slots__", ())]
    return (type(u).__name__,) + tuple((n, getattr(u, n, None)) for n in names)


def _plain_units(x):
    """`x` with every Unit (also inside lists/tuples, e.g. clause batches) replaced by its full state."""
    if isinstance(x, Unit):
        return unit_key(x)
    if isinstance(x, (list, tuple)):
        return tuple(_plain_units(v) for v in x)
    return x


def _inst_key(inst) -> tuple:
    """A queued effect instance with its unit references (and its clause batch) replaced by their full state."""
    return tuple(_plain_units(x) for x in inst)


def state_key(game: Game, rng: bool = True) -> tuple:
    """Every piece of documented state, including hidden info and (optionally) the RNG state."""
    def units(zone):
        return tuple(unit_key(u) for u in zone)

    pending = game.pending
    return (
        game.first_player, game.current, game.round, tuple(game.deck_ids), tuple(game.coins),
        tuple(game.base_hp), tuple(map(tuple, game.hands)), tuple(map(tuple, game.deck_cards)),
        tuple(map(tuple, game.played)), tuple(game.burned), tuple(units(z) for z in game.backline),
        units(game.frontline), game.front_owner, game.next_uid, game.done, game.winner(),
        game.rng.getstate() if rng else None,
        # Stage 3 state (SPEC §4)
        game.phase, game.turn, tuple(sorted(game.mulligan_marks)), tuple(game.mulligan_done),
        None if pending is None else (_inst_key(pending[0]),) + tuple(pending[1:]),
        tuple(_inst_key(i) for i in game.queue), tuple(game.coin_bonus), tuple(map(tuple, game.decklists)),
        tuple(map(tuple, game.discard)), tuple(map(tuple, game.graveyard)), tuple(map(tuple, game.known_hand)),
        tuple(map(tuple, game.revealed)), game.guard_trips,
        # phase 1b (SPEC §1.2b): history counters
        tuple(map(tuple, game.history_turn)), tuple(map(tuple, game.history_game)),
    )


def snapshot(game: Game) -> tuple:
    """state_key + everything an outside caller can read through the public API."""
    return (state_key(game), game.render(), game.observe(0), game.observe(1), tuple(game.legal_actions()),
            game.legal_mask().tobytes(), game.current_player(), game.winner(), game.done, game.round)


# ---------------------------------------------------------------- seeded fuzzing policies
def _weights(end: float, play_w: Callable, move_w: Callable, attack_w: Callable) -> Callable:
    """Profile from per-kind weight functions of (game, action, card/unit)."""
    def weight(game: Game, action) -> float:
        kind = action.kind
        p = game.current
        if kind == ActionKind.END_TURN:
            return end
        if kind in (ActionKind.CHOOSE, ActionKind.MULLIGAN, ActionKind.CONFIRM):
            return 1.0
        if kind == ActionKind.PLAY:
            return play_w(game.config.cards[game.hands[p][action.a]])
        if kind == ActionKind.MOVE:
            return move_w(game.backline[p][action.a])
        z = game.config.zone_capacity
        unit = game.backline[p][action.a] if action.a < z else game.frontline[action.a - z]
        return attack_w(unit, action.b == 2 * z)
    return weight


# Per-action weights of the fuzzing policies. Skewed profiles reach the corners of the state
# space: full zones, full hands (and burns), Defense walls, ranged duels, fast rushes, long games.
PROFILES = {
    "uniform": lambda game, action: 1.0,
    "fast_rush": _weights(0.05, lambda c: 4.0 if c.nature == FAST else 0.3,
                          lambda u: 6.0 if u.nature == FAST else 1.0,
                          lambda u, base: 6.0 if base else 1.0),
    "ranged_snipe": _weights(0.1, lambda c: 4.0 if c.nature == RANGED else 0.3,
                             lambda u: 0.5 if u.nature == RANGED else 0.2,
                             lambda u, base: 5.0 if u.nature == RANGED else 1.0),
    "defense_wall": _weights(0.3, lambda c: 5.0 if c.defense else 0.5, lambda u: 0.05,
                             lambda u, base: 0.5),
    "full_frontline": _weights(0.3, lambda c: 5.0, lambda u: 6.0, lambda u, base: 0.002),
    "hoarder": _weights(1.0, lambda c: 0.02, lambda u: 1.0, lambda u, base: 1.0),
}
PROFILE_NAMES = tuple(PROFILES)
PROFILE_PAIRS = tuple(product(PROFILE_NAMES, repeat=2))


class WeightedPolicy:
    """Seeded random policy over legal actions with profile weights."""

    def __init__(self, seed: int, profile: str | Callable = "uniform"):
        self.rng = random.Random(seed)
        self.weight = PROFILES[profile] if isinstance(profile, str) else profile

    def __call__(self, game: Game) -> int:
        legal = game.legal_actions()
        decode = game.action_space.decode
        w = [self.weight(game, decode(a)) for a in legal]
        return self.rng.choices(legal, weights=w)[0]


def profiles_for(seed: int) -> tuple:
    """Deterministic ordered profile pair for a seed, cycling through all combinations."""
    return PROFILE_PAIRS[seed % len(PROFILE_PAIRS)]


def iter_states(seed: int, profiles: Optional[Sequence[str]] = None, decks: Optional[Sequence[int]] = None,
                max_steps: int = 20_000, config: GameConfig = CONFIG) -> Iterator[tuple]:
    """Play one seeded game; yield (game, plays, action) at every state including the final one.

    `plays[p]` counts PLAY actions by seat p so far; `action` is what will be played next
    (None at the final state). Consumers must not mutate `game` (use `game.clone()`).
    """
    profiles = profiles or profiles_for(seed)
    game = new_game(seed, decks, config)
    policies = [WeightedPolicy(seed * 7919 + p, profiles[p]) for p in (0, 1)]
    plays = [0, 0]
    play0, move0 = game.action_space.PLAY0, game.action_space.MOVE0
    for _ in range(max_steps):
        if game.done:
            break
        p = game.current_player()
        action = policies[p](game)
        yield game, plays, action
        if play0 <= action < move0:
            plays[p] += 1
        game.step(action)
    assert game.done, "game did not terminate"
    yield game, plays, None


def _counts(cards, n: int) -> list:
    out = [0] * n
    for c in cards:
        out[c] += 1
    return out


def check_invariants(game: Game, plays: Optional[Sequence[int]] = None, reachable: bool = True) -> None:
    """Structural invariants of every state. `reachable=False` skips the checks that only hold for
    states reached by play from reset() (card stats, card conservation, flag history, uids). Checks that
    effects legitimately break (buffs, summons, returns, coins gained on the opponent's turn) run only on
    effect-free pools."""
    from cardgame.engine import CHOICE, MULLIGAN  # noqa: PLC0415

    cfg = game.config
    n = len(cfg.cards)
    effects = cfg.cards.has_effects
    Zc, Hc = cfg.zone_capacity, cfg.max_hand_size
    assert game.current in (0, 1) and game.first_player in (0, 1)
    assert 1 <= game.round <= cfg.max_rounds
    assert game.phase in (MULLIGAN, MAIN, CHOICE)
    assert len(game.deck_ids) == 2 and all(d == -1 or 0 <= d < cfg.n_decks for d in game.deck_ids)
    for p in (0, 1):
        dl = game.decklists[p]
        assert len(dl) == cfg.deck_size and list(dl) == sorted(dl)
        if game.deck_ids[p] >= 0:
            assert tuple(dl) == cfg.decks[game.deck_ids[p]]
    assert (game.phase == CHOICE) == (game.pending is not None)
    assert not game.queue or game.pending is not None, "the queue is empty between actions"
    if game.phase == MULLIGAN:
        assert game.round == 1 and game.turn == 0 and not game.done
        assert all(0 <= i < len(game.hands[game.current]) for i in game.mulligan_marks)
    else:
        assert not game.mulligan_marks
    units = all_units(game)
    for p in (0, 1):
        assert len(game.backline[p]) <= Zc
        assert all(u.owner == p for u in game.backline[p])
        assert game.coins[p] >= 0
        assert len(game.hands[p]) <= Hc
        assert list(game.hands[p]) == sorted(game.hands[p])
        assert game.burned[p] >= 0
        assert len(game.played[p]) == n and min(game.played[p]) >= 0
        for name in ("discard", "graveyard", "known_hand", "revealed"):
            v = getattr(game, name)[p]
            assert len(v) == n and min(v) >= 0, name
        hand = _counts(game.hands[p], n)
        assert all(map(operator.le, game.known_hand[p], hand)), f"P{p} known_hand exceeds the hand"
        assert not any(game.revealed[p][c] for c in range(n) if cfg.cards[c].token), "tokens never enter revealed"
        if plays is not None:
            assert sum(game.played[p]) == plays[p]
        if reachable:  # every non-token card of the decklist is accounted for (burned identities are lost)
            deck = _counts(game.decklists[p], n)
            seen = _counts(game.deck_cards[p], n)
            for c in range(n):
                if not cfg.cards[c].token:
                    seen[c] += hand[c] + game.graveyard[p][c] + game.discard[p][c]
            for u in units:
                if u.owner == p and not u.token:
                    seen[u.card] += 1
            assert all(map(operator.le, seen, deck)), f"P{p} holds cards that are not in its deck"
            assert all(map(operator.le, game.revealed[p], deck)), f"P{p} revealed more copies than its deck has"
            missing = sum(deck) - sum(seen)
            if effects:
                assert missing <= game.burned[p]
            else:
                assert missing == game.burned[p]
                on_board = [0] * n
                for u in units:
                    if u.owner == p:
                        on_board[u.card] += 1
                assert all(map(operator.le, on_board, game.played[p])), f"P{p} has unplayed cards on the board"
    assert len(game.frontline) <= Zc
    if game.frontline:
        assert game.front_owner in (0, 1)
        assert all(u.owner == game.front_owner for u in game.frontline)
    else:
        assert game.front_owner is None
    assert len({u.uid for u in units}) == len(units), "duplicate uids"
    front_ids = {id(u) for u in game.frontline}
    for u in units:
        # a game that ends in a damage step stops before dead units are removed (SPEC §2.8 step 1)
        assert (u.hp > 0 or game.done) and u.hp <= u.max_hp
        assert u.armor >= 0 and u.atk >= 0 and 0 <= u.attacks <= 2
        if not reachable:
            continue
        c = cfg.cards[u.card]
        assert c.is_unit, u
        if not effects:
            assert (u.atk, u.max_hp, u.armor, u.defense, u.nature, u.move_cost) == (
                c.attack, c.health, c.armor, c.defense, c.nature, c.move_cost), u
        assert 0 <= u.uid < game.next_uid
        if u.summoned and not u.blitz and not effects:  # deployed this round: has not acted (effects can
            assert not u.moved and not u.attacked, u      # take away a blitz the unit acted with)
            if not effects:
                assert id(u) not in front_ids, u
        if u.moved and not effects:
            assert id(u) in front_ids, u
        if u.nature != FAST:
            assert not (u.moved and u.attacked), u
        if not game.done and u.owner != game.current and game.phase != MULLIGAN:  # refreshed at END_TURN
            assert not (u.moved or u.attacked), u
            if not effects:
                assert not u.summoned, u
    if reachable and not effects:
        assert game.next_uid == sum(map(sum, game.played))
    assert game.done == (game.winner() is not None)
    if not game.done:
        assert min(game.base_hp) > 0
        if not effects and game.phase != MULLIGAN:
            assert game.coins[1 - game.current] == 0  # unused coins are lost at END_TURN
            if reachable:
                assert game.coins[game.current] <= cfg.coins_for_round(game.round)
    elif game.winner() in (0, 1):
        assert game.base_hp[1 - game.winner()] <= 0 < game.base_hp[game.winner()]
    else:
        assert game.winner() == DRAW
        assert (game.round == cfg.max_rounds and min(game.base_hp) > 0) or max(game.base_hp) <= 0


# ---------------------------------------------------------------- building positions (SPEC §4)
def blank_game(current: int = 0, first: int = 0, round_: int = 1, coins: int = 0, seed: int = 0,
               decks: Sequence = (0, 1), config: GameConfig = CONFIG) -> Game:
    """A running game (phase MAIN, mulligan done) with empty hands, decks and board, ready for a
    hand-built position. `turn` is consistent with (round, current, first).

    Empty decks keep turn starts free of draws. `add_unit`/`set_hand` call `invalidate()`;
    call it yourself after other direct edits.
    """
    g = new_game(seed, decks, config)
    n = len(config.cards)
    g.first_player, g.current, g.round = first, current, round_
    g.turn = 2 * (round_ - 1) + (1 if current == first else 2)
    g.phase = MAIN
    g.mulligan_marks = set()
    g.mulligan_done = [True, True]
    g.pending = None
    g.queue = []
    g.hands = [[], []]
    g.deck_cards = [[], []]
    g.coins = [0, 0]
    g.coins[current] = coins
    g.coin_bonus = [0, 0]
    g.base_hp = [config.base_hp, config.base_hp]
    g.burned = [0, 0]
    g.played = [[0] * n, [0] * n]
    for name in ("discard", "graveyard", "known_hand", "revealed"):
        setattr(g, name, [[0] * n, [0] * n])
    g.backline = [[], []]
    g.frontline = []
    g.front_owner = None
    g.next_uid = 0
    g.guard_trips = 0
    g.done, g._winner = False, None
    g.invalidate()
    return g


_ADD_UNIT_FIELDS = (set(UNIT_FIELDS) | {"attacked"}) - {"owner"}


def add_unit(game: Game, owner: int, zone: str, card_id=None, **fields) -> Unit:
    """Append a unit to `owner`'s backline (zone="back") or to the frontline (zone="front").

    `fields` override Unit fields (atk, hp, max_hp, armor, defense, nature, move_cost, keywords, pins,
    temp_* fields, summoned, moved, attacked/attacks, card). Flags default to False: the unit can act
    this turn. Card ids are looked up in the game's own pool.
    """
    unknown = set(fields) - _ADD_UNIT_FIELDS
    if unknown:
        raise TypeError(f"unknown unit fields {sorted(unknown)}")
    cfg = game.config
    if card_id is not None:
        c = cfg.cards[card(card_id, cfg)]
        stats = dict(card=c.index, atk=c.attack, hp=c.health, max_hp=c.health, armor=c.armor,
                     defense=c.defense, nature=c.nature, move_cost=c.move_cost, blitz=c.blitz,
                     smokescreen=c.smokescreen, fury=c.fury, token=c.token, ambush=c.ambush, shock=c.shock,
                     immune=c.immune)
    else:
        if "atk" not in fields or "hp" not in fields:
            raise TypeError("add_unit needs a card id or explicit atk and hp")
        nature, defense = fields.get("nature", TROOP), fields.get("defense", False)
        stats = dict(card=card_for(nature, defense, fields.get("armor", 0) > 0, config=cfg), armor=0,
                     defense=False, nature=TROOP, move_cost=1)
    stats.update(summoned=False, moved=False, attacks=0, uid=game.next_uid)
    if "attacked" in fields and "attacks" not in fields:
        stats["attacks"] = 1 if fields["attacked"] else 0
    stats.update({k: v for k, v in fields.items() if k != "attacked"})
    stats.setdefault("max_hp", stats["hp"])
    if "hp" in fields and "max_hp" not in fields:
        stats["max_hp"] = max(stats["max_hp"], stats["hp"])
    unit = Unit(stats.pop("card"), owner, **stats)
    game.next_uid = max(game.next_uid, unit.uid + 1)
    if zone == "back":
        game.backline[owner].append(unit)
    elif zone == "front":
        assert game.front_owner in (None, owner), "the enemy holds the frontline"
        game.frontline.append(unit)
        game.front_owner = owner
    else:
        raise ValueError(zone)
    game.invalidate()
    return unit


def set_hand(game: Game, player: int, cards: Sequence) -> None:
    game.hands[player] = sorted(card(c, game.config) for c in cards)
    game.invalidate()


def set_deck(game: Game, player: int, cards: Sequence) -> None:
    """Deck contents, top = end of the list."""
    game.deck_cards[player] = [card(c, game.config) for c in cards]
    game.invalidate()
