"""No hidden-information leaks: observe(p) depends only on what p may see (SPEC §5)."""
from __future__ import annotations

import copy
import random
from collections import Counter

import numpy as np
import pytest

from cardgame.engine import DRAW, Game, Observation, UnitView
from cardgame.features import ObservationEncoder
from conftest import CONFIG, H, WeightedPolicy, check_invariants, iter_states, new_game, snapshot

SPEC_FIELDS = ("player", "is_my_turn", "went_first", "round", "my_coins", "opp_coins", "my_base_hp",
               "opp_base_hp", "hand", "opp_hand_size", "my_deck_size", "opp_deck_size", "my_backline",
               "opp_backline", "frontline", "front_owner", "done", "result")
INT_FIELDS = ("player", "round", "my_coins", "opp_coins", "my_base_hp", "opp_base_hp", "opp_hand_size",
              "my_deck_size", "opp_deck_size", "front_owner", "result")
BOOL_FIELDS = ("is_my_turn", "went_first", "done")
UNIT_FIELDS = ("my_backline", "opp_backline", "frontline")
ENCODER = ObservationEncoder(CONFIG)


def perturb_hidden(game: Game, observer: int, rng: random.Random) -> Game:
    """A clone of `game` that differs only in information hidden from `observer`:
    the opponent's hand is redrawn from its hand + deck (same size, kept sorted), both decks
    are reshuffled, and the RNG is reseeded."""
    g = game.clone()
    o = 1 - observer
    pool = g.hands[o] + g.decks[o]
    rng.shuffle(pool)
    n = len(g.hands[o])
    g.hands[o] = sorted(pool[:n])
    g.decks[o] = pool[n:]
    rng.shuffle(g.decks[observer])
    g.rng = random.Random(rng.getrandbits(64))
    if hasattr(g, "seed"):
        g.seed = rng.getrandbits(32)
    g.invalidate()
    return g


def iter_sample_states(seeds, every: int = 1):
    for seed in seeds:
        for i, (game, _, _) in enumerate(iter_states(seed)):
            if i % every == 0 or game.done:
                yield seed, game


def test_observations_ignore_hidden_information():
    rng = random.Random(2024)
    checked = opp_view_changed = legal_checked = 0
    for seed, game in iter_sample_states(range(300, 325), every=2):
        for p in (0, 1):
            alt = perturb_hidden(game, p, rng)
            check_invariants(alt)
            obs, alt_obs = game.observe(p), alt.observe(p)
            assert obs == alt_obs, f"seed {seed}: P{p}'s view depends on hidden info\n{game.render()}"
            assert np.array_equal(ENCODER.encode(obs), ENCODER.encode(alt_obs))
            if not game.done and game.current_player() == p:
                assert alt.legal_actions() == game.legal_actions()
                assert np.array_equal(alt.legal_mask(), game.legal_mask())
                legal_checked += 1
            opp_view_changed += alt.observe(1 - p) != game.observe(1 - p)
            checked += 1
    assert checked >= 1000 and legal_checked >= 400
    # Teeth: the perturbation really changes what the *other* player sees most of the time.
    assert opp_view_changed >= 0.4 * checked, (opp_view_changed, checked)


def test_hidden_information_stays_hidden_through_own_turn():
    """Playing the same actions in two hidden-info variants keeps the actor's view identical,
    including the opponent's draw at the start of their turn."""
    rng = random.Random(7)
    turns = 0
    for seed, game in iter_sample_states(range(400, 412), every=5):
        if game.done:
            continue
        p = game.current_player()
        a, b = game.clone(), perturb_hidden(game, p, rng)
        policy = WeightedPolicy(seed)
        while not a.done and a.current_player() == p:
            assert a.observe(p) == b.observe(p) and a.legal_actions() == b.legal_actions()
            action = policy(a)
            a.step(action)
            b.step(action)
        assert a.observe(p) == b.observe(p)
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
    assert UnitView._fields == ("card", "atk", "hp", "ready")
    n = 0
    for _, game in iter_sample_states(range(500, 506), every=3):
        for p in (0, 1):
            o = 1 - p
            obs = game.observe(p)
            assert isinstance(obs, Observation) and isinstance(obs, tuple)
            for name in INT_FIELDS:
                v = getattr(obs, name)
                assert type(v) is int, (name, type(v))
            for name in BOOL_FIELDS:
                assert type(getattr(obs, name)) is bool, name
            assert type(obs.hand) is tuple and all(type(c) is int for c in obs.hand)
            for name in UNIT_FIELDS:
                zone = getattr(obs, name)
                assert type(zone) is tuple and all(type(u) is UnitView for u in zone), name
                for u in zone:
                    assert [type(x) for x in u] == [int, int, int, bool], (name, u)
            # Only plain immutable leaves; nothing that could alias engine objects.
            for path, leaf in walk(obs):
                assert type(leaf) in (int, bool), (path, type(leaf))
            hash(obs)

            # The only card identities in the view: own hand + units on the (public) board.
            board = [*game.backline[0], *game.backline[1], *game.frontline]
            seen = Counter(obs.hand) + Counter(u.card for u in (*obs.my_backline, *obs.opp_backline,
                                                                 *obs.frontline))
            assert seen == Counter(game.hands[p]) + Counter(u.card for u in board)
            assert obs.opp_hand_size == len(game.hands[o]) <= H
            n += 1
    assert n >= 200


def test_observation_is_egocentric_and_correct():
    for _, game in iter_sample_states(range(600, 606), every=3):
        for p in (0, 1):
            o = 1 - p
            obs = game.observe(p)
            fo = game.front_owner
            w = game.winner()
            assert obs.player == p
            assert obs.is_my_turn == (not game.done and game.current_player() == p)
            assert obs.went_first == (game.first_player == p)
            assert obs.round == game.round
            assert (obs.my_coins, obs.opp_coins) == (game.coins[p], game.coins[o])
            assert (obs.my_base_hp, obs.opp_base_hp) == (game.base_hp[p], game.base_hp[o])
            assert obs.hand == tuple(game.hands[p])
            assert (obs.my_deck_size, obs.opp_deck_size) == (len(game.decks[p]), len(game.decks[o]))
            for zone, units in ((obs.my_backline, game.backline[p]), (obs.opp_backline, game.backline[o]),
                                (obs.frontline, game.frontline)):
                assert zone == tuple(UnitView(u.card, u.atk, u.hp, u.ready) for u in units)
            assert obs.front_owner == (0 if fo is None else (1 if fo == p else -1))
            assert obs.done == game.done
            expected = 0 if w in (None, DRAW) else (1 if w == p else -1)
            assert obs.result == expected


def test_observation_is_immutable_and_does_not_alias_engine_state():
    game = None
    for g, _, _ in iter_states(31, ("builder", "builder")):
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
    with pytest.raises(AttributeError):
        obs.frontline[0].hp = 99

    # Mutate the game in place, then keep playing: the old observations must not move.
    for zone in (*game.backline, game.frontline):
        for u in zone:
            u.hp += 7
            u.ready = not u.ready
    if game.hands[p]:
        game.hands[p].pop()
    game.decks[p].clear()
    game.coins[p] += 5
    game.base_hp[1 - p] -= 1
    game.invalidate()
    for _ in range(30):
        if game.done:
            break
        game.step(game.legal_actions()[-1])
    assert obs == frozen and other == frozen_other


def test_encoder_uses_only_the_observation():
    # The encoding is a pure function of the Observation: equal observations, equal vectors,
    # regardless of which Game produced them.
    g1, g2 = new_game(77), new_game(77)
    rng = random.Random(1)
    alt = perturb_hidden(g2, g2.current_player(), rng)
    p = g1.current_player()
    assert np.array_equal(ENCODER.encode(g1.observe(p)), ENCODER.encode(alt.observe(p)))
    batch = ENCODER.encode_batch([g1.observe(0), g1.observe(1)])
    assert np.array_equal(batch[0], ENCODER.encode(g1.observe(0)))
    assert np.array_equal(batch[1], ENCODER.encode(g1.observe(1)))
