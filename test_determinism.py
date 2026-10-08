"""Determinism (same seed + same actions => same states) and clone() independence (SPEC §4)."""
from __future__ import annotations

import random

import numpy as np
import pytest

from cardgame.engine import Game, IllegalActionError
from conftest import (CONFIG, WeightedPolicy, check_invariants, iter_states, new_game, snapshot,
                      state_key)


def all_units(game: Game) -> list:
    return [*game.backline[0], *game.backline[1], *game.frontline]


def mid_game_state(seed: int) -> Game:
    """A clone of the first state of a seeded game with units in all three zones."""
    for game, _, _ in iter_states(seed, ("builder", "turtle")):
        if game.backline[0] and game.backline[1] and game.frontline and not game.done:
            return game.clone()
    raise AssertionError(f"seed {seed}: no state with all zones occupied")


# ---------------------------------------------------------------- determinism
@pytest.mark.parametrize("seed", [0, 5, 17, 123, 2**31 - 1])
def test_same_seed_and_actions_give_identical_states(seed):
    a, b = new_game(seed), new_game(seed)
    policy = WeightedPolicy(seed)
    steps = 0
    while not a.done:
        assert snapshot(a) == snapshot(b), f"diverged after {steps} steps"
        assert np.array_equal(a.legal_mask(), b.legal_mask())
        action = policy(a)
        a.step(action)
        b.step(action)
        steps += 1
    assert snapshot(a) == snapshot(b) and a.winner() == b.winner()


def test_replaying_a_recorded_game_reproduces_it():
    game, actions, trace = new_game(99), [], []
    policy = WeightedPolicy(3)
    while not game.done:
        trace.append(snapshot(game))
        actions.append(policy(game))
        game.step(actions[-1])
    trace.append(snapshot(game))

    random.seed(12345)  # global RNG state must not matter
    np.random.seed(12345)
    replay = Game(CONFIG)
    replay.reset(99)
    for i, action in enumerate(actions):
        assert snapshot(replay) == trace[i], f"replay diverged at step {i}"
        replay.step(action)
    assert snapshot(replay) == trace[-1]


def test_engine_leaves_global_random_untouched():
    random.seed(7)
    state = random.getstate()
    np_state = np.random.get_state()
    for _ in iter_states(8):
        pass
    assert random.getstate() == state
    after = np.random.get_state()
    assert after[0] == np_state[0] and np.array_equal(after[1], np_state[1]) and after[2:] == np_state[2:]


def test_reset_reuses_the_game_object_cleanly():
    g = new_game(1)
    for _ in range(40):
        if g.done:
            break
        g.step(g.legal_actions()[-1])
    g.reset(9)
    assert snapshot(g) == snapshot(new_game(9))
    g.reset(1)
    assert snapshot(g) == snapshot(new_game(1))


def test_different_seeds_give_different_deals():
    deals = set()
    firsts = set()
    for seed in range(300):
        g = new_game(seed)
        deals.add((tuple(map(tuple, g.hands)), tuple(map(tuple, g.decks))))
        firsts.add(g.first_player)
    assert len(deals) == 300 and firsts == {0, 1}
    # The two seats are shuffled independently.
    g = new_game(0)
    assert sorted(g.hands[0] + g.decks[0]) == list(CONFIG.decks[0])
    assert sorted(g.hands[1] + g.decks[1]) == list(CONFIG.decks[1])


# ---------------------------------------------------------------- clone
def test_clone_equals_original_and_evolves_identically():
    n = 0
    for seed in range(700, 706):
        for i, (game, _, _) in enumerate(iter_states(seed)):
            if i % 4:
                continue
            c = game.clone()
            assert snapshot(c) == snapshot(game)
            assert np.array_equal(c.legal_mask(), game.legal_mask())
            assert c.rng is not game.rng and c.rng.getstate() == game.rng.getstate()
            if i % 20 == 0:  # play both to the end with the same actions
                original = game.clone()
                policy_a, policy_b = WeightedPolicy(i), WeightedPolicy(i)
                while not original.done:
                    action = policy_a(original)
                    assert policy_b(c) == action
                    original.step(action)
                    c.step(action)
                    assert snapshot(c) == snapshot(original)
            n += 1
    assert n >= 150


def test_clone_shares_no_mutable_state():
    g = mid_game_state(1)
    c = g.clone()
    for name in ("hands", "decks", "backline"):
        for p in (0, 1):
            assert getattr(c, name)[p] is not getattr(g, name)[p], name
        assert getattr(c, name) is not getattr(g, name), name
    for name in ("coins", "base_hp", "burned", "frontline"):
        assert getattr(c, name) is not getattr(g, name), name
    original_ids = {id(u) for u in all_units(g)}
    assert not original_ids & {id(u) for u in all_units(c)}


def mutate(game: Game) -> None:
    """Scribble on every documented piece of state, then keep playing."""
    for u in all_units(game):
        u.hp += 3
        u.ready = not u.ready
    p = game.current
    if game.hands[p]:
        game.hands[p].pop()
    game.decks[0].reverse()
    if game.decks[1]:
        game.decks[1].pop()
    game.coins[p] += 4
    game.base_hp[0] -= 1
    game.burned[1] += 1
    game.backline[1].pop()
    game.rng.random()
    game.rng.shuffle(game.decks[0])
    game.invalidate()
    for _ in range(25):
        if game.done:
            break
        game.step(game.legal_actions()[-1])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_mutating_or_stepping_a_clone_never_affects_the_original(seed):
    g = mid_game_state(seed)
    before = snapshot(g)
    c = g.clone()
    mutate(c)
    assert snapshot(c) != before
    assert snapshot(g) == before
    check_invariants(g)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_mutating_or_stepping_the_original_never_affects_a_clone(seed):
    g = mid_game_state(seed)
    c = g.clone()
    before = snapshot(c)
    mutate(g)
    assert snapshot(c) == before
    check_invariants(c)


def test_clone_keeps_its_own_rng_stream():
    g = new_game(5)
    c = g.clone()
    draws_g = [g.rng.random() for _ in range(5)]
    draws_c = [c.rng.random() for _ in range(5)]
    assert draws_g == draws_c  # same state ...
    g.rng.random()
    assert g.rng.getstate() != c.rng.getstate()  # ... advanced independently


def test_clone_of_a_finished_game():
    final = None
    for game, _, action in iter_states(21, ("aggro", "aggro")):
        if action is None:
            final = game
    assert final is not None and final.done
    c = final.clone()
    assert state_key(c) == state_key(final) and snapshot(c) == snapshot(final)
    assert c.legal_actions() == [] and c.winner() == final.winner()
    with pytest.raises(IllegalActionError):
        c.step(0)
