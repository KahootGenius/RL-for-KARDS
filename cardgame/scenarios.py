"""Hand-built tactical positions with a known best line (SPEC 10), a DFS solver and a runner.

A `Scenario(name, tags, decks, build, goal)` builds a position (`build(config) -> Game`, the agent to
move) and judges the agent's turn with `goal(start, end) -> bool`, where `end` is the state right after
the agent's turn ended (END_TURN, the CONFIRM of a mulligan, or game over). Units are identified by
`uid`. `run_scenarios` plays an agent (spec string or instance) through all of them; `solve` searches
the turn exhaustively.

Positions are built with the ruleset forced to `mulligan=False` (`scenario_config`): mid-game positions
(`build_position`) have cards from the named decks, consistent flags, coins, turn counter and
revealed-card bookkeeping; the agent's deck holds the rest of its decklist in a fixed order (so the
policy sees a real deck, as in training) and the opponent's deck is empty. They hold no card that
draws on the hidden engine RNG (random targets, random discards; `rng_dependent_cards`), so a known
best line never depends on luck. The mulligan scenario (`build_mulligan_position`) starts in the
mulligan phase of a real game instead (`mulligan=True`, full decks).

The searches (`solve`, `end_of_turn_states`, `can_win_this_turn` behind `survives_next_turn`) know no
game rules: they explore the engine's own legal actions on clones (plays, operations, pending CHOOSE
decisions and mulligan marks included) and read the results back from the engine (a turn is over when
the game ended or another player is to move), so they stay correct when later stages change the rules.
"""
from __future__ import annotations

import dataclasses
import functools
import operator
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from .cards import FAST, GameConfig, load_ruleset
from .engine import CHOICE, MAIN, MULLIGAN, Game, IllegalActionError, Unit

MAX_ACTIONS = 64       # a run that has not ended its turn after this many actions fails
SOLVER_DEPTH = 12      # actions per line in `solve`, including the final END_TURN / CONFIRM
RANDOM_SEEDS = 20      # random agent: success averaged over this many seeds
SCENARIO_SEED = 4_000_000_000  # agent seeds for playouts (outside every deal-seed range, SPEC 8)
BASELINES = ("lookahead", "random")  # the scenario baselines reported next to an agent (SPEC 10)

Goal = Callable[[Game, Game], bool]


@dataclass(frozen=True)
class Scenario:
    name: str
    tags: Tuple[str, ...]
    decks: Tuple[str, str]                             # deck names of seat 0 and seat 1
    build: Callable[..., Game] = field(repr=False)     # build(config=None) -> Game, agent to move
    goal: Goal = field(repr=False)
    note: str = ""                                     # the known best line, in words


# ---------------------------------------------------------------------- rulesets
def scenario_config(config: Optional[GameConfig] = None) -> GameConfig:
    """`config` (default: the shipped ruleset) with the mulligan phase switched off (SPEC 10)."""
    config = config if config is not None else load_ruleset()
    return config if not config.mulligan else dataclasses.replace(config, mulligan=False)


def mulligan_config(config: Optional[GameConfig] = None) -> GameConfig:
    """`config` (default: the shipped ruleset) with the mulligan phase switched on."""
    config = config if config is not None else load_ruleset()
    return config if config.mulligan else dataclasses.replace(config, mulligan=True)


# ---------------------------------------------------------------------- positions
@dataclass(frozen=True)
class U:
    """A unit in a scenario position: card id, current hp (default full) and this-turn flags."""
    card: str
    hp: Optional[int] = None
    summoned: bool = False
    moved: bool = False
    attacked: bool = False


UnitSpec = Union[str, U]


def _turn_index(round_: int, first_player: int, player: int) -> int:
    """SPEC 2.3/2.13: turn 1 is the first player's first turn, so round r holds turns 2r-1 and 2r."""
    return 2 * round_ - (1 if player == first_player else 0)


def build_position(config: Optional[GameConfig] = None, *, decks: Tuple[str, str], seat: int = 0,
                   round: int = 5, first_player: Optional[int] = None, coins: Optional[int] = None,
                   my_base: int = 20, opp_base: int = 20, my_hand: Sequence[str] = (),
                   opp_hand: Sequence[str] = (), my_back: Sequence[UnitSpec] = (),
                   opp_back: Sequence[UnitSpec] = (), my_front: Sequence[UnitSpec] = (),
                   opp_front: Sequence[UnitSpec] = ()) -> Game:
    """A phase-MAIN position with `seat` to move, described from that player's side ("my" = the agent's).

    The ruleset is forced to `mulligan=False`. `decks` are deck names in seat order. Coins default to
    the round's coins minus what was spent this turn (summoned units' costs and moved units' move
    costs). The agent's deck is `agent_deck_rest` (its decklist minus its hand and its non-token units,
    sorted by card index); the opponent's deck is empty, so its start-of-turn draw (part of the
    survival checks) reveals no hidden card. The agent draws only after its turn, so its deck never
    changes a goal. `played` and `revealed` count the (non-token) units on the board, `known_hand` the
    token cards in hand (added by effects), and the history counters the units deployed. uids follow
    the order my_back, my_front, opp_front, opp_back. Static effects are applied by `invalidate()`.
    """
    config = scenario_config(config)
    if my_front and opp_front:
        raise ValueError("only one side can hold the frontline")
    names = list(config.deck_names)
    deck_ids = tuple(names.index(d) for d in decks)
    pool = config.cards
    me, opp = seat, 1 - seat
    game = Game(config)
    game.reset(0, deck_ids)
    game.round = round
    game.first_player = seat if first_player is None else first_player
    game.current = me
    game.turn = _turn_index(round, game.first_player, me)
    game.phase = MAIN
    game.deck_cards = [[], []]
    game.hands = [[], []]
    game.hands[me] = sorted(pool.by_id(c).index for c in my_hand)
    game.hands[opp] = sorted(pool.by_id(c).index for c in opp_hand)
    game.burned = [0, 0]
    game.base_hp = [0, 0]
    game.base_hp[me], game.base_hp[opp] = my_base, opp_base
    n = len(pool)
    game.played = [[0] * n, [0] * n]
    game.revealed = [[0] * n, [0] * n]
    game.known_hand = [[0] * n, [0] * n]
    game.discard = [[0] * n, [0] * n]
    game.graveyard = [[0] * n, [0] * n]
    game.coin_bonus = [0, 0]
    game.queue, game.pending = [], None
    for p in (0, 1):  # token cards in hand were added by effects: known to the opponent (SPEC 5)
        for c in game.hands[p]:
            if pool.cards[c].token:
                game.known_hand[p][c] += 1
    uid = 0

    def units(specs: Sequence[UnitSpec], owner: int) -> List[Unit]:
        nonlocal uid
        out = []
        for spec in specs:
            spec = U(spec) if isinstance(spec, str) else spec
            card = pool.by_id(spec.card)
            u = Unit.from_card(card, owner, uid)
            uid += 1
            u.hp = card.health if spec.hp is None else spec.hp
            u.summoned, u.moved, u.attacked = spec.summoned, spec.moved, spec.attacked
            if not card.token:  # played from hand: a card the opponent has seen come from the deck
                game.played[owner][card.index] += 1
                game.revealed[owner][card.index] += 1
            out.append(u)
        return out

    game.backline = [[], []]
    game.backline[me] = units(my_back, me)
    game.frontline = units(my_front, me) + units(opp_front, opp)
    game.backline[opp] = units(opp_back, opp)
    game.front_owner = me if my_front else (opp if opp_front else None)
    game.next_uid = uid
    game.deck_cards[me] = agent_deck_rest(game, me)
    deployed_now = sum(1 for u in _own_units(game, me) if u.summoned)
    game.history_turn = [[0, 0, 0], [0, 0, 0]]
    game.history_turn[me][1] = deployed_now
    game.history_game = [[0, sum(game.played[p]), 0] for p in (0, 1)]
    spent = _spent_this_turn(game, me)
    game.coins = [0, 0]
    game.coins[me] = config.coins_for_round(round) - spent if coins is None else coins
    game.num_steps = 0
    game.done, game._winner = False, None
    game.invalidate()
    return game


def build_mulligan_position(config: Optional[GameConfig] = None, *, decks: Tuple[str, str], seat: int = 0,
                            first_player: int = 0, my_hand: Sequence[str] = (), opp_hand: Sequence[str] = (),
                            my_deck_top: Sequence[str] = ()) -> Game:
    """A game in its mulligan phase (ruleset forced to `mulligan=True`) with `seat` to decide.

    Opening hands are given by card id (the first player holds `opening_hand[0]` cards, the second
    `opening_hand[1]`); each deck holds the rest of its 40-card list. The agent's deck is sorted by card
    index with `my_deck_top` on top (its first entry is drawn first), so the replacements are known;
    the opponent's deck is sorted. If the agent decides second, the first player has already
    confirmed (keeping its hand).
    """
    config = mulligan_config(config)
    names = list(config.deck_names)
    deck_ids = tuple(names.index(d) for d in decks)
    pool = config.cards
    me, opp = seat, 1 - seat
    game = Game(config)
    game.reset(0, deck_ids)
    game.first_player = first_player
    game.current = me
    hands = {me: sorted(pool.by_id(c).index for c in my_hand), opp: sorted(pool.by_id(c).index for c in opp_hand)}
    for p in (0, 1):
        want = config.opening_hand[0 if p == first_player else 1]
        if len(hands[p]) != want:
            raise ValueError(f"player {p} must hold {want} opening cards, got {len(hands[p])}")
    game.hands = [hands[0], hands[1]]
    top = [pool.by_id(c).index for c in my_deck_top]
    decks_out = []
    for p in (0, 1):
        rest = Counter(game.decklists[p])
        rest.subtract(hands[p])
        if p == me:
            rest.subtract(top)
        if any(k < 0 for k in rest.values()):
            raise ValueError(f"player {p}: hand/deck cards exceed the deck {names[deck_ids[p]]}")
        cards = sorted(rest.elements())
        decks_out.append(cards + (top[::-1] if p == me else []))  # top of the deck = end of the list
    game.deck_cards = decks_out
    game.mulligan_marks = set()
    game.mulligan_done = [False, False]
    if me != first_player:
        game.mulligan_done[first_player] = True
    game.num_steps = 0
    game.invalidate()
    return game


def _own_units(game: Game, p: int) -> List[Unit]:
    return list(game.backline[p]) + [u for u in game.frontline if u.owner == p]


def agent_deck_rest(game: Game, p: int) -> List[int]:
    """Player p's decklist minus the non-token cards in its hand and on its side of the board, sorted by
    card index (a fixed order): the agent's deck in a mid-game scenario position."""
    cards = game.config.cards.cards
    rest = Counter(game.decklists[p])
    rest.subtract(c for c in game.hands[p] if not cards[c].token)
    rest.subtract(u.card for u in _own_units(game, p) if not cards[u.card].token)
    return sorted(c for c, k in rest.items() for _ in range(max(0, k)))


def _effect_chain(effect):
    while effect is not None:
        yield effect
        effect = effect.else_


def rng_dependent_cards(config: GameConfig) -> frozenset:
    """Indices of the cards whose effects draw on the engine's hidden RNG (SPEC 5): a random target
    selection or a (random) discard, in an effect or its `else` body, or in a token card it summons or
    adds. A scenario holding one would be judged on one hidden roll of the dice, so check_position
    rejects them."""
    cards = config.cards.cards
    out = {c.index for c in cards
           if any(e.target.select == "random" or e.action == "discard" for top in c.effects for e in _effect_chain(top))}
    changed = True
    while changed:  # cards that bring such a token into play
        changed = False
        for c in cards:
            if c.index not in out and any(e.card in out for top in c.effects for e in _effect_chain(top)):
                out.add(c.index)
                changed = True
    return frozenset(out)


def _spent_this_turn(game: Game, p: int) -> int:
    cards = game.config.cards.cards
    own = _own_units(game, p)
    return sum(cards[u.card].cost for u in own if u.summoned) + sum(u.move_cost for u in own if u.moved)


def check_position(scenario: Scenario, game: Game) -> List[str]:
    """Consistency problems of a built scenario position (empty list = consistent)."""
    cfg = game.config
    names = list(cfg.deck_names)
    try:
        want = tuple(names.index(d) for d in scenario.decks)
    except ValueError:
        return [f"unknown deck in {scenario.decks}"]
    problems = []
    if tuple(game.deck_ids) != want:
        problems.append(f"deck_ids {game.deck_ids} != {want}")
    if game.done or game.winner() is not None:
        problems.append("game is over")
    if game.queue or game.pending is not None:
        problems.append("effects are pending")
    if game.phase == MULLIGAN:
        problems += _check_mulligan_position(game)
    elif game.phase == MAIN:
        problems += _check_main_position(game)
        rng = rng_dependent_cards(cfg)
        cards = cfg.cards.cards
        held = sorted({cards[c].id for h in game.hands for c in h if c in rng}
                      | {cards[u.card].id for z in game.backline for u in z if u.card in rng}
                      | {cards[u.card].id for u in game.frontline if u.card in rng})
        if held:
            problems.append(f"cards with random effects in play ({', '.join(held)}): the goal would be judged on "
                            f"one hidden RNG state (SPEC 5: the RNG is hidden from the agent)")
    else:
        problems.append(f"phase {game.phase}: positions start in MAIN or MULLIGAN")
    try:
        game.clone().invalidate()
    except ValueError as exc:
        problems.append(f"invariant: {exc}")
    return problems


def _check_mulligan_position(game: Game) -> List[str]:
    cfg = game.config
    problems = []
    p, first = game.current, game.first_player
    if not cfg.mulligan:
        problems.append("mulligan phase with mulligan=False")
    if (game.round, game.turn) != (1, 0) or game.coins != [0, 0] or game.base_hp != [cfg.base_hp] * 2:
        problems.append("not the start of a game (round 1, turn 0, no coins, full bases)")
    if game.backline != [[], []] or game.frontline:
        problems.append("units on the board during the mulligan")
    if game.mulligan_marks:
        problems.append("marks already set")
    if game.mulligan_done[p] or (p != first and not game.mulligan_done[first]):
        problems.append(f"mulligan_done {game.mulligan_done} inconsistent with player {p} deciding")
    for owner in (0, 1):
        size = cfg.opening_hand[0 if owner == first else 1]
        if len(game.hands[owner]) != size or game.hands[owner] != sorted(game.hands[owner]):
            problems.append(f"hand {owner} is not a sorted opening hand of {size} cards")
        if Counter(game.hands[owner]) + Counter(game.deck_cards[owner]) != Counter(game.decklists[owner]):
            problems.append(f"player {owner}: hand + deck differ from the decklist")
    return problems


def _check_main_position(game: Game) -> List[str]:
    cfg = game.config
    cards = cfg.cards.cards
    names = list(cfg.deck_names)
    problems = []
    p, o = game.current, 1 - game.current
    if cfg.mulligan:
        problems.append("mid-game position built with mulligan=True (scenarios use mulligan=False)")
    if game.deck_cards[o]:
        problems.append("the opponent's deck is not empty")
    if game.deck_cards[p] != agent_deck_rest(game, p):
        problems.append("the agent's deck is not the rest of its decklist (sorted)")
    if not 1 <= game.round <= cfg.max_rounds:
        problems.append(f"round {game.round} out of range")
    if game.turn != _turn_index(game.round, game.first_player, p):
        problems.append(f"turn {game.turn} inconsistent with round {game.round}")
    if game.coins[o] != 0:
        problems.append("the opponent has coins during the agent's turn")
    if game.coin_bonus != [0, 0]:
        problems.append("coin bonus set")
    expected = cfg.coins_for_round(game.round) - _spent_this_turn(game, p)
    if game.coins[p] != expected or game.coins[p] > game.round:
        problems.append(f"coins {game.coins[p]} != round coins minus this turn's spending ({expected})")
    if not all(1 <= hp <= cfg.base_hp for hp in game.base_hp):
        problems.append(f"base hp {game.base_hp} out of range")
    board = [(u, "back") for z in game.backline for u in z] + [(u, "front") for u in game.frontline]
    uids = [u.uid for u, _ in board]
    if len(set(uids)) != len(uids) or any(not 0 <= x < game.next_uid for x in uids):
        problems.append(f"uids {uids} not unique or >= next_uid {game.next_uid}")
    for owner in (0, 1):
        if any(u.owner != owner for u in game.backline[owner]):
            problems.append(f"backline {owner} holds a unit of the other player")
    for u, zone in board:
        c = cards[u.card]
        base = (u.atk - u.static_atk, u.max_hp - u.static_hp, u.armor, u.nature, u.move_cost - u.static_move_cost)
        if base != (c.attack, c.health, c.armor, c.nature, c.move_cost) or (
                u.defense != c.defense and not u.static_traits):
            problems.append(f"uid {u.uid}: stats differ from card {c.id}")
        if not 1 <= u.hp <= u.max_hp:
            problems.append(f"uid {u.uid}: hp {u.hp} not in 1..{u.max_hp}")
        if u.owner == o and (u.summoned or u.moved or u.attacked):
            problems.append(f"uid {u.uid}: opponent unit with this-turn flags (refreshed at its END_TURN)")
        if u.summoned and (zone != "back" or u.moved or u.attacked):
            problems.append(f"uid {u.uid}: summoned unit that acted or left the backline")
        if u.moved and zone != "front":
            problems.append(f"uid {u.uid}: moved but not in the frontline")
        if u.moved and u.attacked and c.nature != FAST:  # only fast units may move and attack
            problems.append(f"uid {u.uid}: non-fast unit both moved and attacked")
        if u.pinned or u.temp_atk or u.temp_hp or u.temp_traits or u.temp_removed:
            problems.append(f"uid {u.uid}: pins or 'turn' changes are not supported in scenario positions")
    for owner in (0, 1):
        deck = Counter(cfg.decks[game.deck_ids[owner]])
        hand = Counter(game.hands[owner])
        on_board = Counter(u.card for u, _ in board if u.owner == owner)
        if game.hands[owner] != sorted(game.hands[owner]) or len(game.hands[owner]) > cfg.max_hand_size:
            problems.append(f"hand {owner} unsorted or too large")
        for card_index in set(hand) | set(on_board):
            if cards[card_index].token:
                continue  # tokens come from effects, never from the deck
            played = game.played[owner][card_index]
            if played < on_board[card_index] or played + hand[card_index] > deck[card_index]:
                problems.append(f"player {owner}: {cards[card_index].id} x{hand[card_index]} in hand, "
                                f"{played} played, deck {names[game.deck_ids[owner]]} has {deck[card_index]}")
        for card_index, k in enumerate(game.revealed[owner]):
            if k and (cards[card_index].token or k > deck[card_index] or k < on_board[card_index]):
                problems.append(f"player {owner}: revealed {cards[card_index].id} x{k} inconsistent")
        for card_index, k in hand.items():
            if cards[card_index].token and game.known_hand[owner][card_index] < k:
                problems.append(f"player {owner}: token {cards[card_index].id} in hand but not known to the opponent")
    return problems


# ---------------------------------------------------------------------- goal helpers
# Every goal carries `kind`: "win", "survive", "material" (judged on the board) or "rule" (a decision
# rule, e.g. the mulligan); `all_of` composites carry their `parts`. `dominance_violations` reads them.
def _units(game: Game) -> Dict[int, Unit]:
    out = {u.uid: u for z in game.backline for u in z}
    out.update((u.uid, u) for u in game.frontline)
    return out


def _uids(start: Game, owner: int, card_id: str) -> List[int]:
    index = start.config.cards.by_id(card_id).index
    return [u.uid for u in _units(start).values() if u.owner == owner and u.card == index]


def _turn_over(start: Game, end: Game) -> bool:
    return end.done or end.current != start.current


def win(start: Game, end: Game) -> bool:
    """The agent destroyed the enemy base."""
    return end.winner() == start.current


win.kind = "win"


def kills(*card_ids: str) -> Goal:
    """Every enemy unit with one of these card ids (in the start position) is dead at the end."""
    def goal(start: Game, end: Game) -> bool:
        if win(start, end):
            return True
        alive = _units(end)
        return not any(uid in alive for c in card_ids for uid in _uids(start, 1 - start.current, c))
    goal.__doc__ = f"kills {', '.join(card_ids)}"
    goal.kind = "material"
    return goal


def no_losses(start: Game, end: Game) -> bool:
    """Every unit the agent had at the start is still alive."""
    alive = _units(end)
    return all(uid in alive for uid, u in _units(start).items() if u.owner == start.current)


no_losses.kind = "material"


def summons(card_id: str) -> Goal:
    """The agent ends its turn with more units of this card on the board than it started with (e.g. a
    token an on_death effect summoned)."""
    def goal(start: Game, end: Game) -> bool:
        index = start.config.cards.by_id(card_id).index
        p = start.current

        def count(g: Game) -> int:
            return sum(1 for u in _units(g).values() if u.owner == p and u.card == index)
        return count(end) > count(start)
    goal.__doc__ = f"summons {card_id}"
    goal.kind = "material"
    return goal


def survives_next_turn(start: Game, end: Game) -> bool:
    """The agent has won, or the opponent cannot destroy the agent's base on its next turn."""
    if end.done:
        return end.winner() == start.current
    if end.current == start.current:  # turn not ended yet: judge the position after END_TURN
        end = end.clone()
        end.step(end.action_space.END_TURN)
        if end.done:
            return end.winner() == start.current
    return not can_win_this_turn(end)


survives_next_turn.kind = "survive"


def mulligan_rule(keep_max_cost: int = 2, replace_min_cost: int = 5) -> Goal:
    """At CONFIRM: every opening-hand card with cost >= `replace_min_cost` was replaced and every card
    with cost <= `keep_max_cost` kept (cards in between are free). Exact when no card drawn as a
    replacement shares a cost class with those (`build_mulligan_position` stacks the deck top)."""
    def goal(start: Game, end: Game) -> bool:
        p = start.current
        if not _turn_over(start, end) or not end.mulligan_done[p]:
            return False
        cost = start.config.cards.cards
        before, after = Counter(start.hands[p]), Counter(end.hands[p])
        for c, k in before.items():
            if cost[c].cost <= keep_max_cost and after[c] < k:
                return False
            if cost[c].cost >= replace_min_cost and after[c] > 0:
                return False
        return True
    goal.__doc__ = f"mulligan: replace cost >= {replace_min_cost}, keep cost <= {keep_max_cost}"
    goal.kind = "rule"
    goal.keep_max_cost, goal.replace_min_cost = keep_max_cost, replace_min_cost
    return goal


def all_of(*goals: Goal) -> Goal:
    def goal(start: Game, end: Game) -> bool:
        return all(g(start, end) for g in goals)
    goal.__doc__ = " and ".join((g.__doc__ or g.__name__).strip().split("\n")[0] for g in goals)
    goal.kind = "all"
    goal.parts = tuple(goals)
    return goal


def goal_kinds(goal: Goal) -> frozenset:
    """The leaf kinds of a goal ("win", "survive", "material", "rule"); composites are flattened."""
    parts = getattr(goal, "parts", None)
    if parts is not None:
        return frozenset().union(*(goal_kinds(g) for g in parts))
    return frozenset({getattr(goal, "kind", "material")})


# ---------------------------------------------------------------------- search
_UNIT_KEY = operator.attrgetter(*Unit.__slots__)
_LIST_KEYS = ("hands", "deck_cards", "played", "discard", "graveyard", "known_hand", "revealed", "history_turn",
              "history_game")
_FLAT_KEYS = ("coins", "base_hp", "burned", "coin_bonus", "mulligan_done")


def state_key(game: Game) -> tuple:
    """The whole game state (RNG included) as a hashable key, for transposition tables: equal keys =
    equal futures. A state with a pending choice, queued effects or a paused combat gets a unique key
    (it is never merged with another state)."""
    if game.phase == CHOICE or game.queue or getattr(game, "_combat", None) is not None:
        return ("unique", object())

    def zone(units):
        return tuple(_UNIT_KEY(u) for u in units)
    return (game.current, game.round, game.turn, game.phase, game.done, game.winner(), game.front_owner,
            game.next_uid, game.guard_trips, tuple(sorted(game.mulligan_marks)),
            tuple(tuple(map(tuple, getattr(game, k))) for k in _LIST_KEYS),
            tuple(tuple(getattr(game, k)) for k in _FLAT_KEYS),
            zone(game.backline[0]), zone(game.backline[1]), zone(game.frontline),
            game.deck_ids, game.decklists, game.rng.getstate())


def can_win_this_turn(game: Game) -> bool:
    """True if the player to move can destroy the enemy base before its turn ends.

    Exhaustive search with the engine alone: every legal action (plays, operations and CHOOSE
    decisions included) is stepped on a clone, a line ends when the game is over or another player is
    to move, and transpositions are skipped. No rule is assumed (who can act, reach, damage, effects);
    children are tried in order of the enemy base HP they leave, so a winning line is usually found
    first. The cost grows with the number of distinct lines (hand size, coins, units), which is small
    in the scenario positions.
    """
    if game.done:
        return False
    p = game.current
    seen = {state_key(game)}

    def search(g: Game) -> bool:
        children = []
        for a in g.legal_actions():
            c = g.clone()
            c.step(a)
            if c.done:
                if c.winner() == p:
                    return True
                continue
            if c.current != p:
                continue  # the turn is over
            key = state_key(c)
            if key not in seen:
                seen.add(key)
                children.append(c)
        children.sort(key=lambda c: c.base_hp[1 - p])  # stable: ties keep the action order
        return any(search(c) for c in children)

    return search(game)


def solve(scenario: Scenario, config: Optional[GameConfig] = None, max_depth: int = SOLVER_DEPTH,
          start: Optional[Game] = None) -> Optional[List[int]]:
    """A line of at most `max_depth` actions (ending the agent's turn: END_TURN, CONFIRM or a win)
    that meets the goal, else None.

    Depth-first over the agent's turn on clones, with a transposition table keyed by state (the
    goal depends only on the start and end states).
    """
    start = start if start is not None else scenario.build(config)
    p = start.current
    best_depth: Dict[tuple, int] = {}

    def search(g: Game, depth: int) -> Optional[List[int]]:
        for a in g.legal_actions():
            c = g.clone()
            c.step(a)
            if c.done or c.current != p:  # the agent's turn is over
                if scenario.goal(start, c):
                    return [a]
                continue
            if depth + 2 > max_depth:  # no room left for the action that ends the turn
                continue
            key = state_key(c)
            if best_depth.get(key, max_depth + 1) <= depth + 1:
                continue
            best_depth[key] = depth + 1
            rest = search(c, depth + 1)
            if rest is not None:
                return [a] + rest
        return None

    return search(start.clone(), 0)


class EndState(NamedTuple):
    """Where one line of the agent's turn leaves the game (scored for the dominance check)."""
    won: bool
    enemy_base_hp: int
    enemy_board_value: int  # total card cost of the opponent's units on the board
    own_board_value: int    # total card cost of the agent's units on the board
    line: tuple             # the actions that reach it
    goal: bool
    survives: Optional[bool] = None  # the opponent cannot win next turn (only computed for survival goals)


def end_of_turn_states(scenario: Scenario, config: Optional[GameConfig] = None,
                       start: Optional[Game] = None, survival: bool = False) -> List[EndState]:
    """Every distinct state the agent's turn can end in (END_TURN / CONFIRM taken, or game over),
    exhaustively, with the engine alone (clone/step/legal_actions) and a transposition table.
    `survival` also records whether the agent survives the opponent's next turn."""
    start = start if start is not None else scenario.build(config)
    p, o = start.current, 1 - start.current
    cost = [c.cost for c in start.config.cards.cards]
    seen, out = set(), []

    def value(g: Game, owner: int) -> int:
        return sum(cost[u.card] for u in _units(g).values() if u.owner == owner)

    def search(g: Game, line: tuple) -> None:
        for a in g.legal_actions():
            c = g.clone()
            c.step(a)
            key = state_key(c)
            if key in seen:
                continue
            seen.add(key)
            if c.done or c.current != p:
                survives = None
                if survival:
                    survives = c.winner() == p if c.done else not can_win_this_turn(c)
                out.append(EndState(c.winner() == p, c.base_hp[o], value(c, o), value(c, p), line + (a,),
                                    bool(scenario.goal(start, c)), survives))
            else:
                search(c, line + (a,))

    search(start.clone(), ())
    return out


def dominance_violations(scenario: Scenario, config: Optional[GameConfig] = None) -> List[EndState]:
    """Validity of a scenario's "known best move" (empty list = valid).

    * `win` goals are valid by definition.
    * Goals that include `survives_next_turn` are invalid if the agent has lethal this turn (winning
      would beat surviving); that is reported as a violation. Pure survival goals are then valid.
    * Otherwise every end state that misses the goal must be weakly dominated by a goal-meeting one on
      (won, enemy base HP, enemy board value, own board value), so missing the goal never buys
      anything; the non-dominated misses are returned. With a survival part, an end state from which
      the opponent can win next turn counts as dominated (it loses). Rule goals (the mulligan) compare
      equal on these, so they are valid as soon as some line meets them.
    """
    start = scenario.build(config)
    kinds = goal_kinds(scenario.goal)
    if kinds == {"win"}:
        return []
    survival = "survive" in kinds
    if survival:
        if can_win_this_turn(start):
            return [EndState(True, 0, 0, 0, (), False, True)]
        if kinds == {"survive"}:
            return []
    states = end_of_turn_states(scenario, config, start, survival=survival)
    goals = [s for s in states if s.goal]

    def dominated(s: EndState) -> bool:
        if survival and not s.survives:
            return True
        return any(g.won >= s.won and g.enemy_base_hp <= s.enemy_base_hp and g.enemy_board_value <= s.enemy_board_value
                   and g.own_board_value >= s.own_board_value for g in goals)

    return [s for s in states if not s.goal and not dominated(s)]


# ---------------------------------------------------------------------- running agents
def play_scenario(scenario: Scenario, agent, config: Optional[GameConfig] = None,
                  seed: Optional[int] = None, start: Optional[Game] = None) -> Tuple[bool, List[int]]:
    """One playout of the agent's turn: (goal met, actions taken). Moves come from
    `agents.choose_action`: plain agents see only observe() + legal actions, simulating agents
    (lookahead) get the game and may only determinize it."""
    from .agents import choose_action
    game = start.clone() if start is not None else scenario.build(config)
    start = game.clone()
    p = game.current
    agent.reset(seed)
    line: List[int] = []
    while len(line) < MAX_ACTIONS:
        legal = game.legal_actions()
        a = choose_action(agent, game)
        if a not in legal:
            raise IllegalActionError(f"agent {getattr(agent, 'name', agent)!r} chose illegal action {a!r} in "
                                     f"scenario {scenario.name!r}")
        game.step(int(a))
        line.append(int(a))
        if game.done or game.current != p:  # the agent's turn is over
            return bool(scenario.goal(start, game)), line
    return False, line


@dataclass
class ScenarioResult:
    name: str
    tags: Tuple[str, ...]
    solved: Optional[bool]    # the deterministic run (None: the agent has no deterministic mode)
    success: Optional[float]  # fraction of stochastic playouts that met the goal (None: not run)
    playouts: int
    line: Tuple[str, ...]     # actions of the deterministic run (else of the first playout)

    def to_dict(self) -> dict:
        return {"name": self.name, "tags": list(self.tags), "solved": self.solved, "success": self.success,
                "playouts": self.playouts, "line": list(self.line)}


@dataclass
class ScenarioReport:
    agent: str
    results: List[ScenarioResult]

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def deterministic(self) -> bool:
        return all(r.solved is not None for r in self.results)

    @property
    def n_solved(self) -> Optional[int]:
        return sum(bool(r.solved) for r in self.results) if self.deterministic else None

    @property
    def mean_success(self) -> Optional[float]:
        rates = [r.success for r in self.results if r.success is not None]
        return sum(rates) / len(rates) if rates and len(rates) == self.n else None

    @property
    def pct_solved(self) -> float:
        """% of scenarios solved by the deterministic run; for agents without one, the mean playout success."""
        if not self.n:
            return 0.0
        if self.deterministic:
            return 100.0 * self.n_solved / self.n
        return 100.0 * (self.mean_success or 0.0)

    def to_dict(self) -> dict:
        return {"agent": self.agent, "n": self.n, "n_solved": self.n_solved, "pct_solved": self.pct_solved,
                "mean_success": self.mean_success, "results": [r.to_dict() for r in self.results]}

    def format(self) -> str:
        width = max([len("scenario")] + [len(r.name) for r in self.results])
        lines = [f"{'scenario':<{width}s} {'solved':>7s} {'playouts':>9s}  tags"]
        for r in self.results:
            solved = "-" if r.solved is None else ("yes" if r.solved else "no")
            rate = "-" if r.success is None else f"{100 * r.success:.0f}% /{r.playouts}"
            lines.append(f"{r.name:<{width}s} {solved:>7s} {rate:>9s}  {','.join(r.tags)}")
        ms = self.mean_success
        lines.append(f"{self.agent}: {self.pct_solved:.0f}% solved"
                     + (f" ({self.n_solved}/{self.n})" if self.deterministic else "")
                     + ("" if ms is None else f"; mean playout success {100 * ms:.0f}%"))
        return "\n".join(lines)


def _agent_modes(agent_or_spec, config: GameConfig, deterministic: bool, n_stochastic: int, n_random: int):
    """(label, deterministic-run agent or None, stochastic agent or None, number of playouts)."""
    from .agents import RandomAgent, make_agent
    if isinstance(agent_or_spec, str):
        spec = agent_or_spec
        if spec == "random":
            return spec, None, RandomAgent(config), n_random
        if spec in ("greedy", "lookahead"):  # one (seeded) run each
            return spec, make_agent(spec, config), None, 0
        det = make_agent(spec, config, deterministic=deterministic)
        stoch = make_agent(spec, config) if n_stochastic > 0 else None
        return spec, det, stoch, n_stochastic
    agent = agent_or_spec
    label = getattr(agent, "name", type(agent).__name__)
    if isinstance(agent, RandomAgent):
        return label, None, agent, n_random
    if hasattr(agent, "deterministic"):  # e.g. PPOAgent: toggled per run below
        return label, agent, agent if n_stochastic > 0 else None, n_stochastic
    return label, agent, None, 0


def playout_seed(seed: int, scenario: str, playout: int) -> int:
    """Agent seed of one playout, stable per (seed, scenario name, playout index) on every platform."""
    from .evaluation import agent_seed
    return agent_seed(SCENARIO_SEED + zlib.crc32(f"{scenario}:{seed}:{playout}".encode("utf-8")), 0)


def run_scenarios(agent_or_spec, config: Optional[GameConfig] = None, deterministic: bool = True,
                  n_stochastic: int = 100, scenarios: Optional[Sequence[Scenario]] = None, seed: int = 0,
                  n_random: int = RANDOM_SEEDS) -> ScenarioReport:
    """Play every scenario: one deterministic run (argmax for PPO agents, unless deterministic=False:
    then one seeded sampled run) plus `n_stochastic` sampled playouts for PPO agents; greedy and
    lookahead once (seeded); random over `n_random` seeds. Agents are reseeded per playout
    (reproducible). `config` may have the mulligan on: positions force their own setting."""
    config = config if config is not None else load_ruleset()
    scenarios = list(SCENARIOS if scenarios is None else scenarios)
    label, det_agent, stoch_agent, n_play = _agent_modes(agent_or_spec, config, deterministic, n_stochastic,
                                                         n_random)
    results = []
    for sc in scenarios:
        start = sc.build(config)
        desc = start.describe
        solved, line = None, None
        if det_agent is not None:
            saved = getattr(det_agent, "deterministic", None)
            if saved is not None:
                det_agent.deterministic = deterministic
            try:
                ok, actions = play_scenario(sc, det_agent, config, playout_seed(seed, sc.name, 0), start)
            finally:
                if saved is not None:
                    det_agent.deterministic = saved
            solved, line = ok, tuple(desc(a) for a in actions)
        success = None
        if stoch_agent is not None and n_play > 0:
            saved = getattr(stoch_agent, "deterministic", None)
            if saved is not None:
                stoch_agent.deterministic = False
            hits = 0
            try:
                for i in range(n_play):
                    ok, actions = play_scenario(sc, stoch_agent, config, playout_seed(seed, sc.name, 1 + i), start)
                    hits += ok
                    if line is None:
                        line = tuple(desc(a) for a in actions)
            finally:
                if saved is not None:
                    stoch_agent.deterministic = saved
            success = hits / n_play
        results.append(ScenarioResult(sc.name, tuple(sc.tags), solved, success, n_play if success is not None else 0,
                                      line or ()))
    return ScenarioReport(label, results)


# ---------------------------------------------------------------------- the scenarios
def _scenario(name: str, tags: Sequence[str], goal: Goal, note: str, *, mine: str, theirs: str, seat: int = 0,
              builder: Callable[..., Game] = build_position, **position) -> Scenario:
    decks = (mine, theirs) if seat == 0 else (theirs, mine)
    build = functools.partial(builder, decks=decks, seat=seat, **position)
    return Scenario(name, tuple(tags), decks, build, goal, note)


SCENARIOS: Tuple[Scenario, ...] = (
    # ---- Stage 2 (vanilla units; revalidated on the Stage 3 decks with mulligan=False)
    _scenario(
        "fast_move_attack_lethal", ("fast", "lethal", "move_cost"), win,
        "Move the Lancer (fast) to the empty frontline and hit the base for exactly 5. Playing the "
        "Swordsman first spends the coin the move needs; the Footman was deployed this round and cannot act.",
        mine="Blitz", theirs="Bulwark", round=5, opp_base=5, my_base=12,
        my_hand=("militia", "swordsman"), my_back=("lancer", U("footman", summoned=True)),
        opp_hand=("knight", "militia", "footman"), opp_back=("pikeman", "archer")),
    _scenario(
        "fast_attack_then_move", ("fast", "frontline", "threat"), survives_next_turn,
        "Outrider (fast) kills the Raider from the backline, then moves into the emptied frontline: "
        "holding it stops the Lancer + Wolf Rider (11) from both moving in next turn. Playing the "
        "Militia first leaves no coin for the move.",
        mine="Blitz", theirs="Legion", round=5, my_base=9,
        my_hand=("militia", "paladin"), my_back=("outrider", U("knight", summoned=True)),
        opp_hand=("footman", "militia"), opp_front=("raider",), opp_back=("lancer", "wolf_rider")),
    _scenario(
        "defense_backline_order", ("defense", "sequencing", "threat"), survives_next_turn,
        "Next turn the Longbowman (4) shoots the base (4 HP) for lethal. The Shieldbearer (Defense) shields it, "
        "so the Swordsman (4) must kill the Shieldbearer first; then the Knight (3) finishes the Longbowman. "
        "Opening with the Knight on the Shieldbearer leaves both alive; no line wins this turn (7 < 14).",
        mine="Legion", theirs="Volley", round=6, my_base=4, opp_base=14,
        my_hand=("archer", "footman"), my_front=("swordsman", "knight"),
        opp_hand=("slinger", "militia"), opp_back=("shieldbearer", "longbowman")),
    _scenario(
        "defense_frontline_ranged", ("defense", "ranged", "frontline", "threat"), survives_next_turn,
        "Next turn the Wolf Rider (6) and the Ballista (5) hit the base for 11 >= 8. The Shieldbearer (Defense) "
        "guards the frontline, so a melee unit kills it first; then the Crossbowman and Archer shoot the Wolf "
        "Rider down (3 + 2 >= 4, no return damage). The Ballista (armor 1, 5 HP) cannot be killed this turn. "
        "Trading the Ogre or Swordsman into the Wolf Rider also survives, but loses a unit.",
        mine="Volley", theirs="Legion", round=7, my_base=8, opp_base=16,
        my_back=("crossbowman", "archer", "swordsman", "ogre"), my_hand=("militia",),
        opp_hand=("footman",), opp_front=("shieldbearer", "wolf_rider"), opp_back=("ballista",)),
    _scenario(
        "ranged_finish_backline", ("ranged", "armor", "threat"), survives_next_turn,
        "Next turn the Pikeman (3), Ballista (5) and Longbowman (4) hit for 12 >= 5; only killing both damaged "
        "shooters (leaving 3) survives. Only the Crossbowman (3 - 1 armor = 2) finishes the 2-HP Ballista, so the "
        "Archer must take the Longbowman. Melee cannot reach the backline past the enemy-held frontline.",
        mine="Legion", theirs="Volley", round=7, my_base=5, opp_base=15,
        my_back=("crossbowman", "archer", "footman"), my_hand=("swordsman",),
        opp_hand=("militia",), opp_front=("pikeman",),
        opp_back=(U("ballista", hp=2), U("longbowman", hp=1))),
    _scenario(
        "ranged_base_lethal", ("ranged", "lethal", "seat1"), win,
        "Crossbowman (3) + Slinger (1) shoot the base for exactly 4; the enemy holds the frontline, "
        "so melee cannot reach the base.",
        mine="Volley", theirs="Bulwark", seat=1, first_player=0, round=6, opp_base=4, my_base=11,
        my_hand=("archer", "pikeman"), my_back=("crossbowman", "slinger", "swordsman"),
        opp_hand=("militia", "footman"), opp_front=("pikeman", "warden"), opp_back=("archer",)),
    _scenario(
        "armor_pierce", ("armor", "trade"), all_of(kills("outrider"), no_losses),
        "Only the Swordsman (4 - 1 armor = 3) kills the Outrider and survives; the Raider deals 2 and "
        "dies, the Militia deals 0 and dies.",
        mine="Blitz", theirs="Bulwark", round=6, my_base=15, opp_base=15,
        my_back=("raider", "militia", "footman", "swordsman"), my_hand=("scout",),
        opp_hand=("knight",), opp_front=("outrider",), opp_back=("shieldbearer", "archer")),
    _scenario(
        "armor_attacker_survives", ("armor", "trade"), all_of(kills("swordsman"), no_losses),
        "The enemy Swordsman (4/3) holds the frontline. Our Swordsman would trade itself away and the Footman "
        "cannot kill it; only the Knight's armor (4 - 1 = 3 < 4 HP) lets it kill and survive. Nothing can reach "
        "the base this turn (the frontline is enemy-held, and troops cannot attack after moving in).",
        mine="Blitz", theirs="Legion", round=6, my_base=14, opp_base=15,
        my_back=("knight", "swordsman", "footman"), my_hand=("militia",),
        opp_hand=("archer",), opp_front=("swordsman",), opp_back=("footman",)),
    _scenario(
        "move_cost_hold_frontline", ("move_cost", "frontline", "threat"), survives_next_turn,
        "Spend the 2 coins moving the Knight (move cost 2) into the empty frontline: the Lancer and "
        "Wolf Rider (11 damage) can then not both move in. Playing a card instead loses next turn.",
        mine="Bulwark", theirs="Blitz", round=6, my_base=10, opp_base=18,
        my_hand=("archer", "footman", "militia"), my_back=("knight", U("pikeman", summoned=True)),
        opp_hand=("scout", "swordsman"), opp_back=("lancer", "wolf_rider")),
    _scenario(
        "clear_frontline_then_lethal", ("frontline", "defense", "fast", "lethal"), win,
        "Troops clear the frontline (Shieldbearer first: Defense), keeping the Lancer fresh to move in "
        "and hit the base for 5. Using the Lancer to clear, or playing the Archer, loses the lethal.",
        mine="Blitz", theirs="Bulwark", round=6, opp_base=5, my_base=12,
        my_hand=("archer", "scout"), my_back=("lancer", "swordsman", "footman", U("knight", summoned=True)),
        opp_hand=("warden",), opp_front=("shieldbearer", "militia"), opp_back=("footman",)),
    _scenario(
        "stop_lethal_kill_right_unit", ("threat", "ranged"), survives_next_turn,
        "Next turn the enemy hits for 4 + 3 + 2 = 9 >= 6. Only shooting the Swordsman (4 atk) brings it "
        "below 6; the damaged Knight is the pricier card but the wrong target.",
        mine="Volley", theirs="Legion", round=5, my_base=6, opp_base=17,
        my_hand=("militia", "longbowman"), my_back=("crossbowman", U("footman", summoned=True)),
        opp_hand=("raider", "pikeman"), opp_front=("swordsman", U("knight", hp=2)), opp_back=("archer",)),
    _scenario(
        "ranged_no_return", ("ranged", "trade", "fast", "lethal"), win,
        "The Longbowman shoots the 3-HP Ogre (ranged: no return damage), emptying the frontline; the Wolf Rider "
        "(fast) then moves in and hits the base for exactly 6. If the Wolf Rider attacks the Ogre it dies to the "
        "7 return damage, and the Longbowman alone deals only 4.",
        mine="Legion", theirs="Volley", round=7, my_base=13, opp_base=6,
        my_back=("wolf_rider", "longbowman"), my_hand=("slinger",),
        opp_hand=("knight",), opp_front=(U("ogre", hp=3),), opp_back=("archer",)),
    _scenario(
        "budget_cheap_movers", ("move_cost", "fast", "lethal"), win,
        "With 2 coins, Lancer (move 1) + Raider (move 1) hit for 8 >= 8 (the free Scout adds 2 more); moving the "
        "Cataphract (move 2) first leaves only 5 + 2 = 7.",
        mine="Blitz", theirs="Bulwark", round=6, opp_base=8, my_base=14,
        my_back=("cataphract", "lancer", "raider", "scout", U("knight", summoned=True)),
        opp_hand=("bastion",), opp_back=("pikeman",)),
    _scenario(
        "base_ignores_defense", ("defense", "lethal", "seat1"), win,
        "Defense never protects the base: Swordsman + Footman + Raider hit it for exactly 9.",
        mine="Legion", theirs="Bulwark", seat=1, first_player=0, round=7, opp_base=9, my_base=10,
        my_hand=("archer",), my_front=("swordsman", "footman", "raider"),
        opp_hand=("militia",), opp_back=("shieldbearer", "pikeman", "warden")),
    _scenario(
        "sacrifice_to_clear", ("frontline", "fast", "lethal", "trade"), win,
        "The Militia trades with the Raider (mutual kill empties the frontline); the Wolf Rider then "
        "moves in and hits for 6. Using the Wolf Rider to kill the Raider leaves no attack for the base.",
        mine="Blitz", theirs="Legion", round=5, opp_base=6, my_base=11,
        my_back=("wolf_rider", "militia", U("knight", summoned=True)), my_hand=("paladin",),
        opp_hand=("archer",), opp_front=("raider",), opp_back=("archer", "footman")),
    _scenario(
        "ranged_through_defense", ("ranged", "defense", "sequencing", "threat"), survives_next_turn,
        "Next turn the Longbowman (4), Footman (2) and Shield Archer (1) hit for 7 >= 5. Defense also binds ranged "
        "attacks: Crossbowman (3) + Slinger (1) must kill the Shield Archer (4 HP) before the Archer can finish "
        "the 2-HP Longbowman, leaving 2 damage. Shooting the Footman or the Shield Archer alone still loses.",
        mine="Volley", theirs="Volley", round=6, my_base=5, opp_base=13,
        my_back=("crossbowman", "archer", "slinger"), my_hand=("militia", "footman"),
        opp_hand=("pikeman",), opp_front=("footman",), opp_back=("shield_archer", U("longbowman", hp=2))),
    # ---- Stage 3 (effects, operations, choices, the mulligan)
    _scenario(
        "operation_lethal", ("operation", "effects", "choice", "lethal", "sequencing"), win,
        "The enemy base has 1 HP but nothing can reach it (the enemy holds the frontline, the Archer was deployed "
        "this round). Deploy the Forward Observer (3): it adds a Fire Mission (1) to the hand, and playing that "
        "artillery operation (on any enemy unit) fires the Observer's 1 damage to the enemy base. Playing the "
        "Militia or the Swordsman first leaves too few coins for the combo.",
        mine="Volley", theirs="Bulwark", round=6, opp_base=1, my_base=12,
        my_hand=("militia", "swordsman", "forward_observer"), my_back=("footman", U("archer", summoned=True)),
        opp_hand=("knight", "militia"), opp_front=("pikeman",), opp_back=("archer",)),
    _scenario(
        "on_death_exploit", ("on_death", "effects", "trade", "threat", "seat1"),
        all_of(survives_next_turn, summons("recruit")),
        "Next turn the damaged Lancer (5 atk, 2 HP) hits the base (5 HP) for lethal, and only a trade kills it. "
        "Trade the Bugler: its on_death summons a Recruit, so the turn ends with Footman + Recruit instead of a "
        "lone Bugler (trading the Footman also survives, but leaves strictly less on the board).",
        mine="Legion", theirs="Blitz", seat=1, first_player=0, round=5, my_base=5, opp_base=14,
        my_back=("bugler", "footman"), my_hand=("shieldbearer",),
        opp_hand=("swordsman",), opp_front=(U("lancer", hp=2),), opp_back=("knight",)),
    _scenario(
        "effect_clears_defense", ("effects", "operation", "choice", "defense", "armor", "threat"),
        survives_next_turn,
        "Next turn the Longbowman (4) shoots the base (4 HP). The Warden (Defense, armor 1, 3 HP) shields it and our "
        "shots cannot both kill the Warden and reach the Longbowman. Armor-Piercing Shot deals 4 to an armored "
        "unit (effect damage ignores Defense and armor): it kills the Warden, then the Crossbowman (3) kills the "
        "Longbowman. Shooting the Longbowman with the operation (2 damage) or playing the Lancer first loses.",
        mine="Legion", theirs="Volley", round=5, my_base=4, opp_base=15,
        my_back=("crossbowman", "archer"), my_hand=("militia", "lancer", "armor_piercing_shot"),
        opp_hand=("militia",), opp_back=(U("warden", hp=3), "longbowman")),
    _scenario(
        "mulligan_sanity", ("mulligan", "seat1"), mulligan_rule(2, 5),
        "Opening hand of the second player: replace every card with cost >= 5 (Warden, Bastion) and keep every "
        "card with cost <= 2 (Militia, Shieldbearer); the Pikeman (4) may go either way. Judged at CONFIRM.",
        mine="Bulwark", theirs="Blitz", seat=1, builder=build_mulligan_position, first_player=0,
        my_hand=("militia", "shieldbearer", "pikeman", "warden", "bastion"),
        opp_hand=("scout", "raider", "lancer", "wolf_rider"),
        my_deck_top=("halberdier", "crossbowman", "chaplain", "knight", "counter_battery")),
)


def scenario_names() -> List[str]:
    return [s.name for s in SCENARIOS]


def get_scenario(name: str) -> Scenario:
    for s in SCENARIOS:
        if s.name == name:
            return s
    raise KeyError(f"unknown scenario {name!r}")


__all__ = ["BASELINES", "EndState", "MAX_ACTIONS", "SCENARIOS", "SCENARIO_SEED", "SOLVER_DEPTH", "Scenario",
           "ScenarioReport", "ScenarioResult", "U", "all_of", "build_mulligan_position", "build_position",
           "can_win_this_turn", "check_position", "dominance_violations", "end_of_turn_states", "get_scenario",
           "goal_kinds", "kills", "mulligan_config", "mulligan_rule", "no_losses", "play_scenario", "run_scenarios",
           "scenario_config", "scenario_names", "solve", "state_key", "summons", "survives_next_turn", "win"]
