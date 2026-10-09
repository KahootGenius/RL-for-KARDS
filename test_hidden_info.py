"""No hidden-information leaks: observe(p), its encoding and p's legal actions depend only on what p
may see (SPEC §5). Hidden: the opponent's hand contents and deck choice, deck order and content,
burned cards and the RNG."""
from __future__ import annotations

import copy
import random
from collections import Counter

import numpy as np
import pytest

from cardgame.engine import DRAW, Game, Observation, UnitView
from cardgame.features import ObservationEncoder
from conftest import (CONFIG, DECK_PAIRS, N_CARDS, N_DECKS, PROFILE_PAIRS, H, WeightedPolicy, all_units,
                      check_invariants, iter_states, new_game, snapshot)

SPEC_FIELDS = ("player", "is_my_turn", "went_first", "round", "my_deck", "my_coins", "opp_coins", "my_base_hp",
               "opp_base_hp", "hand", "opp_hand_size", "my_deck_size", "opp_deck_size", "my_played",
               "opp_played", "my_backline", "opp_backline", "frontline", "front_owner", "done", "result")
UNIT_VIEW_FIELDS = ("card", "atk", "hp", "max_hp", "armor", "defense", "nature", "move_cost", "summoned",
                    "moved", "attacked", "can_move", "can_attack")
UNIT_VIEW_TYPES = [int, int, int, int, int, bool, int, int, bool, bool, bool, bool, bool]
INT_FIELDS = ("player", "round", "my_deck", "my_coins", "opp_coins", "my_base_hp", "opp_base_hp", "opp_hand_size",
              "my_deck_size", "opp_deck_size", "front_owner", "result")
BOOL_FIELDS = ("is_my_turn", "went_first", "done")
UNIT_FIELDS = ("my_backline", "opp_backline", "frontline")
ENCODER = ObservationEncoder(CONFIG)


def perturb_hidden(game: Game, observer: int, rng: random.Random, swap_deck: bool = False) -> Game:
    """A clone of `game` that differs only in information hidden from `observer`.

    The opponent's hand is redrawn (same size, kept sorted) from its hand + deck, or, with
    `swap_deck`, its deck id is replaced by another deck and hand and deck are redrawn from that
    deck (same sizes). Both decks are reshuffled and the RNG (and seed) reseeded.
    """
    g = game.clone()
    o = 1 - observer
    n_hand, n_deck = len(g.hands[o]), len(g.deck_cards[o])
    if swap_deck:
        new_id = rng.choice([d for d in range(N_DECKS) if d != g.deck_ids[o]])
        ids = list(g.deck_ids)
        ids[o] = new_id
        g.deck_ids = tuple(ids)
        pool = list(CONFIG.decks[new_id])
    else:
        pool = g.hands[o] + g.deck_cards[o]
    rng.shuffle(pool)
    g.hands[o] = sorted(pool[:n_hand])
    g.deck_cards[o] = pool[n_hand:n_hand + n_deck]
    rng.shuffle(g.deck_cards[observer])
    g.rng = random.Random(rng.getrandbits(64))
    if hasattr(g, "seed"):
        g.seed = rng.getrandbits(32)
    g.invalidate()
    return g


def sample_states(seeds, every: int = 1):
    """(seed, game) for every `every`-th state of seeded games over all deck pairs (plus final states)."""
    for seed in seeds:
        decks = DECK_PAIRS[seed % len(DECK_PAIRS)]
        profiles = PROFILE_PAIRS[(7 * seed) % len(PROFILE_PAIRS)]
        for i, (game, _, _) in enumerate(iter_states(seed, profiles, decks)):
            if i % every == 0 or game.done:
                yield seed, game


def encode(game: Game, p: int) -> np.ndarray:
    mask = game.legal_mask() if not game.done and game.current_player() == p else None
    return ENCODER.encode(game.observe(p), mask)


@pytest.mark.parametrize("swap_deck", [False, True], ids=["same_deck", "other_deck"])
def test_observations_ignore_hidden_information(swap_deck):
    rng = random.Random(2024 + swap_deck)
    checked = opp_view_changed = legal_checked = hand_changed = 0
    for seed, game in sample_states(range(300, 332), every=3):
        for p in (0, 1):
            alt = perturb_hidden(game, p, rng, swap_deck)
            check_invariants(alt, reachable=False)
            obs, alt_obs = game.observe(p), alt.observe(p)
            assert obs == alt_obs, f"seed {seed}: P{p}'s view depends on hidden info\n{game.render()}"
            assert np.array_equal(encode(game, p), encode(alt, p)), f"seed {seed}: encoding leaks"
            if not game.done and game.current_player() == p:
                assert alt.legal_actions() == game.legal_actions()
                assert np.array_equal(alt.legal_mask(), game.legal_mask())
                legal_checked += 1
            # Teeth: the perturbation really changes what the *other* player sees.
            opp_view_changed += alt.observe(1 - p) != game.observe(1 - p)
            hand_changed += alt.hands[1 - p] != game.hands[1 - p]
            checked += 1
    assert checked >= 1000 and legal_checked >= 400, (checked, legal_checked)
    assert opp_view_changed >= (0.99 if swap_deck else 0.4) * checked, (opp_view_changed, checked)
    assert hand_changed >= 0.4 * checked, (hand_changed, checked)


def test_hidden_information_stays_hidden_through_own_turn():
    """Playing the same actions in two hidden-info variants keeps the actor's view identical,
    including the opponent's draw at the start of their turn."""
    rng = random.Random(7)
    turns = 0
    for seed, game in sample_states(range(400, 416), every=5):
        if game.done:
            continue
        p = game.current_player()
        a, b = game.clone(), perturb_hidden(game, p, rng, swap_deck=turns % 2 == 1)
        policy = WeightedPolicy(seed)
        while not a.done and a.current_player() == p:
            assert a.observe(p) == b.observe(p) and a.legal_actions() == b.legal_actions()
            action = policy(a)
            a.step(action)
            b.step(action)
        assert a.observe(p) == b.observe(p)
        assert np.array_equal(encode(a, p), encode(b, p))
        assert a.done == b.done and a.winner() == b.winner()
        turns += 1
    assert turns >= 100


def walk(value, path="obs"):
    """Yield (path, leaf) for every leaf of a nested observation."""
    if isinstance(value, tuple):
        for i, item in enumerate(value):
            yield from walk(item, f"{path}[{i}]")
    else:
        yield path, value


def test_observation_fields_and_types_cannot_carry_hidden_cards():
    assert Observation._fields == SPEC_FIELDS
    assert UnitView._fields == UNIT_VIEW_FIELDS
    n = 0
    for _, game in sample_states(range(500, 508), every=3):
        for p in (0, 1):
            o = 1 - p
            obs = game.observe(p)
            assert isinstance(obs, Observation) and isinstance(obs, tuple)
            for name in INT_FIELDS:
                assert type(getattr(obs, name)) is int, (name, type(getattr(obs, name)))
            for name in BOOL_FIELDS:
                assert type(getattr(obs, name)) is bool, name
            assert type(obs.hand) is tuple and all(type(c) is int for c in obs.hand)
            for name in ("my_played", "opp_played"):
                v = getattr(obs, name)
                assert type(v) is tuple and len(v) == N_CARDS and all(type(x) is int for x in v), name
            for name in UNIT_FIELDS:
                zone = getattr(obs, name)
                assert type(zone) is tuple and all(type(u) is UnitView for u in zone), name
                for u in zone:
                    assert [type(x) for x in u] == UNIT_VIEW_TYPES, (name, u)
            # Only plain immutable leaves; nothing that could alias engine objects.
            for path, leaf in walk(obs):
                assert type(leaf) in (int, bool), (path, type(leaf))
            hash(obs)

            # The only card identities in the view: own hand + units on the (public) board +
            # the public played counts. The opponent's deck id appears nowhere.
            board = all_units(game)
            seen = Counter(obs.hand) + Counter(u.card for u in (*obs.my_backline, *obs.opp_backline,
                                                                 *obs.frontline))
            assert seen == Counter(game.hands[p]) + Counter(u.card for u in board)
            assert obs.opp_hand_size == len(game.hands[o]) <= H
            assert obs.opp_played == tuple(game.played[o]) and obs.my_played == tuple(game.played[p])
            # every opponent unit on the board was played by the opponent
            on_board = Counter(u.card for u in board if u.owner == o)
            assert all(obs.opp_played[c] >= k for c, k in on_board.items())
            n += 1
    assert n >= 200


def test_observation_is_egocentric_and_correct():
    for _, game in sample_states(range(600, 608), every=3):
        for p in (0, 1):
            o = 1 - p
            obs = game.observe(p)
            fo = game.front_owner
            w = game.winner()
            assert obs.player == p
            assert obs.is_my_turn == (not game.done and game.current_player() == p)
            assert obs.went_first == (game.first_player == p)
            assert obs.round == game.round and obs.my_deck == game.deck_ids[p]
            assert (obs.my_coins, obs.opp_coins) == (game.coins[p], game.coins[o])
            assert (obs.my_base_hp, obs.opp_base_hp) == (game.base_hp[p], game.base_hp[o])
            assert obs.hand == tuple(game.hands[p])
            assert (obs.my_deck_size, obs.opp_deck_size) == (len(game.deck_cards[p]), len(game.deck_cards[o]))
            for zone, units in ((obs.my_backline, game.backline[p]), (obs.opp_backline, game.backline[o]),
                                (obs.frontline, game.frontline)):
                assert zone == tuple(UnitView(u.card, u.atk, u.hp, u.max_hp, u.armor, u.defense, u.nature,
                                              u.move_cost, u.summoned, u.moved, u.attacked, u.can_move(),
                                              u.can_attack()) for u in units)
            assert obs.front_owner == (0 if fo is None else (1 if fo == p else -1))
            assert obs.done == game.done
            assert obs.result == (0 if w in (None, DRAW) else (1 if w == p else -1))


def test_observation_is_immutable_and_does_not_alias_engine_state():
    game = None
    for g, _, _ in iter_states(31, ("full_frontline", "defense_wall"), (1, 2)):
        if g.backline[0] and g.backline[1] and g.frontline and not g.done:
            game = g.clone()
            break
    assert game is not None
    p = game.current_player()
    before = snapshot(game)
    obs = game.observe(p)
    other = game.observe(1 - p)
    assert snapshot(game) == before  # observing is side-effect free
    frozen, frozen_other = copy.deepcopy(obs), copy.deepcopy(other)

    with pytest.raises(AttributeError):
        obs.round = 99
    with pytest.raises(TypeError):
        obs.hand[0] = 0
    with pytest.raises(TypeError):
        obs.my_played[0] = 5
    with pytest.raises(AttributeError):
        obs.frontline[0].hp = 99

    # Mutate the game in place, then keep playing: the old observations must not move.
    for u in all_units(game):
        u.hp += 7
        u.summoned = not u.summoned
        u.armor += 1
    if game.hands[p]:
        game.hands[p].pop()
    game.deck_cards[p].clear()
    game.played[p][0] += 1
    game.coins[p] += 5
    game.base_hp[1 - p] -= 1
    game.invalidate()
    for _ in range(30):
        if game.done:
            break
        game.step(game.legal_actions()[-1])
    assert obs == frozen and other == frozen_other


def test_encoder_uses_only_the_observation_and_mask():
    # The encoding is a pure function of (Observation, mask): equal inputs, equal vectors,
    # regardless of which Game produced them.
    g1, g2 = new_game(77), new_game(77)
    rng = random.Random(1)
    p = g1.current_player()
    alt = perturb_hidden(g2, p, rng, swap_deck=True)
    assert alt.deck_ids[1 - p] != g1.deck_ids[1 - p]
    assert np.array_equal(encode(g1, p), encode(alt, p))
    assert np.array_equal(ENCODER.encode(g1.observe(p), g1.legal_mask()),
                          ENCODER.encode(alt.observe(p), alt.legal_mask()))
    batch = ENCODER.encode_batch([g1.observe(0), g1.observe(1)])
    assert np.array_equal(batch[0], ENCODER.encode(g1.observe(0)))
    assert np.array_equal(batch[1], ENCODER.encode(g1.observe(1)))
