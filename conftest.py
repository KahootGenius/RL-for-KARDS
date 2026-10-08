"""Shared fixtures and helpers for the engine tests (engine-only: no `cardgame.agents`)."""
from __future__ import annotations

import os
import random
import sys
from typing import Iterator, Optional, Sequence

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
for _path in (ROOT, TESTS_DIR):  # works under any pytest import mode
    if _path not in sys.path:
        sys.path.insert(0, _path)

from cardgame.actions import ActionKind, ActionSpace  # noqa: E402
from cardgame.cards import GameConfig, load_ruleset  # noqa: E402
from cardgame.engine import Game, Unit  # noqa: E402

CONFIG: GameConfig = load_ruleset()
SPACE = ActionSpace(CONFIG.max_hand_size, CONFIG.zone_capacity)
NUM_ACTIONS = SPACE.n
H, Z = CONFIG.max_hand_size, CONFIG.zone_capacity

# Per-kind action weights for the seeded fuzzing policies. Skewed profiles reach the corners
# of the state space: full zones, full hands (and burns), frontline fights, long games.
PROFILES = {
    "uniform": {k: 1.0 for k in ActionKind},
    "aggro": {ActionKind.END_TURN: 0.02, ActionKind.PLAY: 1.0, ActionKind.MOVE: 1.0,
              ActionKind.ATTACK_BASE: 1.0, ActionKind.FRONT_ATTACK: 1.0, ActionKind.BACK_ATTACK: 1.0},
    "builder": {ActionKind.END_TURN: 0.05, ActionKind.PLAY: 5.0, ActionKind.MOVE: 3.0,
                ActionKind.ATTACK_BASE: 0.1, ActionKind.FRONT_ATTACK: 0.1, ActionKind.BACK_ATTACK: 0.1},
    "hoarder": {ActionKind.END_TURN: 1.0, ActionKind.PLAY: 0.03, ActionKind.MOVE: 1.0,
                ActionKind.ATTACK_BASE: 0.5, ActionKind.FRONT_ATTACK: 1.0, ActionKind.BACK_ATTACK: 1.0},
    "turtle": {ActionKind.END_TURN: 0.2, ActionKind.PLAY: 4.0, ActionKind.MOVE: 0.02,
               ActionKind.ATTACK_BASE: 1.0, ActionKind.FRONT_ATTACK: 1.0, ActionKind.BACK_ATTACK: 2.0},
    "wall": {ActionKind.END_TURN: 0.3, ActionKind.PLAY: 5.0, ActionKind.MOVE: 0.001,
             ActionKind.ATTACK_BASE: 0.01, ActionKind.FRONT_ATTACK: 0.01, ActionKind.BACK_ATTACK: 0.01},
    "massing": {ActionKind.END_TURN: 1.0, ActionKind.PLAY: 5.0, ActionKind.MOVE: 5.0,
                ActionKind.ATTACK_BASE: 0.001, ActionKind.FRONT_ATTACK: 0.001, ActionKind.BACK_ATTACK: 0.001},
}
PROFILE_NAMES = tuple(PROFILES)
KIND_OF = tuple(SPACE.decode(i).kind for i in range(NUM_ACTIONS))


@pytest.fixture(scope="session")
def config() -> GameConfig:
    return CONFIG


def new_game(seed: int = 0) -> Game:
    g = Game(CONFIG)
    g.reset(seed)
    return g


def card(card_id: str) -> int:
    """Card index for a card id."""
    return CONFIG.cards.by_id(card_id).index


def cost(card_index: int) -> int:
    return CONFIG.cards[card_index].cost


# ---------------------------------------------------------------- full-state comparison
def state_key(game: Game, rng: bool = True) -> tuple:
    """Every piece of documented state, including hidden info and (optionally) the RNG state."""
    def units(zone):
        return tuple((u.card, u.atk, u.hp, u.owner, u.ready) for u in zone)

    return (
        game.first_player, game.current, game.round, tuple(game.coins), tuple(game.base_hp),
        tuple(map(tuple, game.hands)), tuple(map(tuple, game.decks)), tuple(game.burned),
        tuple(units(z) for z in game.backline), units(game.frontline), game.front_owner,
        game.done, game.winner(), game.rng.getstate() if rng else None,
    )


def snapshot(game: Game) -> tuple:
    """state_key + everything an outside caller can read through the public API."""
    return (state_key(game), game.render(), game.observe(0), game.observe(1),
            tuple(game.legal_actions()), game.current_player(), game.winner(), game.done, game.round)


# ---------------------------------------------------------------- seeded fuzzing policies
class WeightedPolicy:
    """Seeded random policy over legal actions with per-kind weights."""

    def __init__(self, seed: int, weights: Optional[dict] = None):
        self.rng = random.Random(seed)
        self.weights = weights or PROFILES["uniform"]

    def __call__(self, game: Game) -> int:
        legal = game.legal_actions()
        w = [self.weights[KIND_OF[a]] for a in legal]
        return self.rng.choices(legal, weights=w)[0]


def profiles_for(seed: int) -> tuple:
    """Deterministic profile pair for a seed, cycling through all combinations."""
    n = len(PROFILE_NAMES)
    return PROFILE_NAMES[seed % n], PROFILE_NAMES[(seed // n) % n]


def iter_states(seed: int, profiles: Optional[Sequence[str]] = None,
                max_steps: int = 20_000) -> Iterator[tuple]:
    """Play one seeded game; yield (game, plays, action) at every state including the final one.

    `plays[p]` counts PLAY actions by seat p so far; `action` is what will be played next
    (None at the final state). Consumers must not mutate `game` (use `game.clone()`).
    """
    profiles = profiles or profiles_for(seed)
    game = new_game(seed)
    policies = [WeightedPolicy(seed * 7919 + p, PROFILES[profiles[p]]) for p in (0, 1)]
    plays = [0, 0]
    for _ in range(max_steps):
        if game.done:
            break
        p = game.current_player()
        action = policies[p](game)
        yield game, plays, action
        if KIND_OF[action] == ActionKind.PLAY:
            plays[p] += 1
        game.step(action)
    assert game.done, "game did not terminate"
    yield game, plays, None


def check_invariants(game: Game, plays: Optional[Sequence[int]] = None) -> None:
    """Structural invariants that must hold in every reachable state."""
    cfg = game.config
    assert game.current in (0, 1) and game.first_player in (0, 1)
    assert 1 <= game.round <= cfg.max_rounds
    for p in (0, 1):
        assert len(game.backline[p]) <= cfg.zone_capacity
        assert all(u.owner == p for u in game.backline[p])
        assert game.coins[p] >= 0
        assert len(game.hands[p]) <= cfg.max_hand_size
        assert list(game.hands[p]) == sorted(game.hands[p])
        assert game.burned[p] >= 0
        if plays is not None:
            assert len(game.hands[p]) + len(game.decks[p]) + game.burned[p] + plays[p] == cfg.deck_size
    assert len(game.frontline) <= cfg.zone_capacity
    if game.frontline:
        assert game.front_owner in (0, 1)
        assert all(u.owner == game.front_owner for u in game.frontline)
    else:
        assert game.front_owner is None
    for u in [*game.backline[0], *game.backline[1], *game.frontline]:
        assert u.hp > 0
        assert u.atk == cfg.cards[u.card].attack
        assert u.hp <= cfg.cards[u.card].health
    assert game.done == (game.winner() is not None)
    if not game.done:
        assert min(game.base_hp) > 0
        assert game.coins[1 - game.current] == 0  # unused coins are lost at END_TURN
    elif game.winner() in (0, 1):
        assert game.base_hp[1 - game.winner()] <= 0 < game.base_hp[game.winner()]


# ---------------------------------------------------------------- building positions (SPEC §4)
def blank_game(current: int = 0, first: int = 0, round_: int = 1, coins: int = 0,
               seed: int = 0) -> Game:
    """A reset game with empty hands, decks and board, ready for a hand-built position.

    Empty decks keep turn starts free of draws. Call `game.invalidate()` after further edits.
    """
    g = new_game(seed)
    g.first_player, g.current, g.round = first, current, round_
    g.hands = [[], []]
    g.decks = [[], []]
    g.coins = [0, 0]
    g.coins[current] = coins
    g.backline = [[], []]
    g.frontline = []
    g.front_owner = None
    g.invalidate()
    return g


def add_unit(game: Game, owner: int, zone: str, card_id: str, ready: bool = True,
             hp: Optional[int] = None, atk: Optional[int] = None) -> Unit:
    """Append a unit to `owner`'s backline (zone="back") or to the frontline (zone="front")."""
    cdef = CONFIG.cards.by_id(card_id)
    unit = Unit(cdef.index, cdef.attack if atk is None else atk, cdef.health if hp is None else hp,
                owner, ready)
    if zone == "back":
        game.backline[owner].append(unit)
    elif zone == "front":
        assert game.front_owner in (None, owner)
        game.frontline.append(unit)
        game.front_owner = owner
    else:
        raise ValueError(zone)
    game.invalidate()
    return unit


def set_hand(game: Game, player: int, card_ids: Sequence[str]) -> None:
    game.hands[player] = sorted(card(c) for c in card_ids)
    game.invalidate()


def act(kind: ActionKind, a: int = -1, b: int = -1) -> int:
    return SPACE.encode(kind, a, b)
