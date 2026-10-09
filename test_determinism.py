"""Determinism (same seed + same decks + same actions => same states) and clone() correctness,
independence and cheapness (SPEC §4)."""
from __future__ import annotations

import copy
import random
import timeit

import numpy as np
import pytest

from cardgame.cards import sample_decks
from cardgame.engine import Game, IllegalActionError, Unit
from conftest import (CONFIG, DECK_PAIRS, N_DECKS, PROFILE_PAIRS, UNIT_FIELDS, WeightedPolicy, all_units,
                      card, check_invariants, iter_states, new_game, snapshot, state_key, unit_key)


def mid_game_state(seed: int) -> Game:
    """A clone of the first state of a seeded game with units in all three zones."""
    for game, _, _ in iter_states(seed, ("full_frontline", "defense_wall"), DECK_PAIRS[seed % len(DECK_PAIRS)]):
        if game.backline[0] and game.backline[1] and game.frontline and not game.done:
            return game.clone()
    raise AssertionError(f"seed {seed}: no state with all zones occupied")


def play_out(game: Game, policy: WeightedPolicy, limit: int = 10_000) -> list:
    """Play to the end; return the snapshots of every state."""
    trace = [snapshot(game)]
    for _ in range(limit):
        if game.done:
            break
        game.step(policy(game))
        trace.append(snapshot(game))
    return trace


# ---------------------------------------------------------------- determinism
@pytest.mark.parametrize("seed", [0, 5, 17, 123, 2**31 - 1])
def test_same_seed_and_actions_give_identical_states(seed):
    a, b = new_game(seed), new_game(seed)
    policy = WeightedPolicy(seed, PROFILE_PAIRS[seed % len(PROFILE_PAIRS)][0])
    steps = 0
    while not a.done:
        assert snapshot(a) == snapshot(b), f"diverged after {steps} steps"
        assert np.array_equal(a.legal_mask(), b.legal_mask())
        action = policy(a)
        a.step(action)
        b.step(action)
        steps += 1
    assert snapshot(a) == snapshot(b) and a.winner() == b.winner()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 99, 4321])
def test_reset_without_decks_equals_reset_with_the_sampled_decks(seed):
    pair = sample_decks(seed, N_DECKS)
    a, b = new_game(seed), new_game(seed, pair)
    assert tuple(a.deck_ids) == pair
    assert play_out(a, WeightedPolicy(seed)) == play_out(b, WeightedPolicy(seed))


@pytest.mark.parametrize("decks", DECK_PAIRS)
def test_same_seed_and_decks_give_identical_games(decks):
    a, b = new_game(42, decks), new_game(42, list(decks))
    assert play_out(a, WeightedPolicy(1, "fast_rush")) == play_out(b, WeightedPolicy(1, "fast_rush"))


def test_same_seed_with_other_decks_shares_the_coin_flip():
    # The deck choice does not consume the game RNG: first player and RNG state depend on the seed only.
    for seed in range(20):
        games = [new_game(seed, pair) for pair in DECK_PAIRS]
        assert len({g.first_player for g in games}) == 1
        assert len({g.rng.getstate() for g in games}) == 1


def test_replaying_a_recorded_game_reproduces_it():
    game, actions = new_game(99), []
    policy = WeightedPolicy(3, "ranged_snipe")
    trace = [snapshot(game)]
    while not game.done:
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
    for _ in iter_states(8, decks=(2, 0)):
        pass
    new_game(9).clone()
    assert random.getstate() == state
    after = np.random.get_state()
    assert after[0] == np_state[0] and np.array_equal(after[1], np_state[1]) and after[2:] == np_state[2:]


def test_reset_reuses_the_game_object_cleanly():
    g = new_game(1, (0, 3))
    for _ in range(80):
        if g.done:
            break
        g.step(g.legal_actions()[-1])
    assert g.next_uid > 0 and sum(map(sum, g.played)) > 0
    g.reset(9)
    assert snapshot(g) == snapshot(new_game(9))
    g.reset(1, (0, 3))
    assert snapshot(g) == snapshot(new_game(1, (0, 3)))
    assert g.next_uid == 0 and not any(map(any, g.played)) and g.burned == [0, 0]


def test_different_seeds_give_different_deals():
    deals = set()
    firsts = set()
    for seed in range(300):
        g = new_game(seed, (1, 1))
        deals.add((tuple(map(tuple, g.hands)), tuple(map(tuple, g.deck_cards))))
        firsts.add(g.first_player)
    assert len(deals) == 300 and firsts == {0, 1}
    g = new_game(0, (1, 1))  # a mirror match still shuffles the two seats independently
    assert g.deck_cards[0] != g.deck_cards[1]


# ---------------------------------------------------------------- clone: equality and independence
def test_clone_equals_original_and_evolves_identically():
    n = 0
    for seed in range(700, 708):
        for i, (game, _, _) in enumerate(iter_states(seed, decks=DECK_PAIRS[seed % len(DECK_PAIRS)])):
            if i % 4:
                continue
            c = game.clone()
            assert type(c) is type(game)
            assert snapshot(c) == snapshot(game)  # includes RNG state, uids and every unit slot
            assert c.next_uid == game.next_uid and [u.uid for u in all_units(c)] == [u.uid for u in all_units(game)]
            assert np.array_equal(c.legal_mask(), game.legal_mask())
            assert c.rng is not game.rng and c.rng.getstate() == game.rng.getstate()
            if i % 20 == 0:  # play both to the end with the same actions
                original = game.clone()
                assert play_out(original, WeightedPolicy(i)) == play_out(c, WeightedPolicy(i))
            n += 1
    assert n >= 150


def test_clone_shares_no_mutable_state():
    g = mid_game_state(1)
    c = g.clone()
    for name in ("hands", "deck_cards", "played", "backline"):
        for p in (0, 1):
            assert getattr(c, name)[p] is not getattr(g, name)[p], name
        assert getattr(c, name) is not getattr(g, name), name
    for name in ("coins", "base_hp", "burned", "frontline"):
        assert getattr(c, name) is not getattr(g, name), name
    assert c.rng is not g.rng
    assert c.config is g.config and c.action_space is g.action_space  # immutable tables are shared
    assert not {id(u) for u in all_units(g)} & {id(u) for u in all_units(c)}
    for u, v in zip(all_units(g), all_units(c)):
        assert unit_key(u) == unit_key(v)


def mutate(game: Game) -> None:
    """Scribble on every documented piece of state, then keep playing."""
    for u in all_units(game):
        u.hp += 3
        u.atk += 1
        u.summoned, u.moved, u.attacked = not u.summoned, not u.moved, not u.attacked
    p = game.current
    if game.hands[p]:
        game.hands[p].pop()
    game.deck_cards[0].reverse()
    if game.deck_cards[1]:
        game.deck_cards[1].pop()
    game.played[0][0] += 2
    game.coins[p] += 4
    game.base_hp[0] -= 1
    game.burned[1] += 1
    game.backline[1].pop()
    game.next_uid += 5
    game.rng.random()
    game.rng.shuffle(game.deck_cards[0])
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


def test_clone_of_a_clone_and_of_a_finished_game():
    final = None
    for game, _, action in iter_states(21, ("fast_rush", "fast_rush"), (0, 0)):
        if action is None:
            final = game
    assert final is not None and final.done
    c = final.clone().clone()
    assert state_key(c) == state_key(final) and snapshot(c) == snapshot(final)
    assert c.legal_actions() == [] and c.winner() == final.winner()
    with pytest.raises(IllegalActionError):
        c.step(0)


# ---------------------------------------------------------------- clone: cheap (no deepcopy on hot fields)
def test_clone_never_deep_copies_a_reachable_state(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("clone() fell back to copy.deepcopy for a normally reachable state")

    games = [new_game(np.int64(3)), new_game(4, (np.int64(1), np.int32(2))), new_game(5, [3, 0]), Game(CONFIG)]
    states = []
    for seed in range(16):
        for i, (game, _, _) in enumerate(iter_states(800 + seed, decks=DECK_PAIRS[seed])):
            if i % 3 == 0 or game.done:
                states.append(game.clone())
    monkeypatch.setattr(copy, "deepcopy", forbidden)
    n = 0
    for g in games + states:
        c = g.clone()
        c.clone()
        n += 1
    monkeypatch.undo()
    assert n >= 300
    for g in states[::25]:  # the clones made under the patch are still correct
        assert snapshot(g.clone()) == snapshot(g)


def test_clone_is_much_cheaper_than_deepcopy():
    # A ratio, not an absolute time, so the check holds on a busy machine.
    g = mid_game_state(2)
    clone_t = min(timeit.repeat(g.clone, number=100, repeat=5))
    deep_t = min(timeit.repeat(lambda: copy.deepcopy(g), number=100, repeat=5))
    assert clone_t * 10 < deep_t, (clone_t, deep_t)


def test_clone_deep_copies_state_added_later():
    g = new_game(2)
    g.graveyard = [[card("militia")], []]  # e.g. a later stage adds a graveyard
    c = g.clone()
    c.graveyard[0].append(card("ogre"))
    assert g.graveyard == [[card("militia")], []]


def test_clone_does_not_share_tuple_held_state():
    g = new_game(4)
    g.hands = tuple(g.hands)  # a tool may store documented state as tuples
    g.deck_cards = tuple(g.deck_cards)
    g.queue = ("on_death", [1, 2])  # later-stage state: tuple holding a list
    g.deck_ids = (np.int64(g.deck_ids[0]), [g.deck_ids[1]])  # odd but must not be shared either
    c = g.clone()
    c.hands[g.current].clear()
    c.deck_cards[0].clear()
    c.queue[1].append(3)
    c.deck_ids[1].append(9)
    assert g.hands[g.current] and g.deck_cards[0] and g.queue == ("on_death", [1, 2])
    assert len(g.deck_ids[1]) == 1


def test_clone_keeps_references_to_board_units_aliased():
    g = new_game(5)
    g.backline[0].append(Unit(card("knight"), 0, atk=4, hp=3, uid=0))
    g.next_uid = 1
    g.selected = g.backline[0][-1]  # e.g. a pending-target reference in a later stage
    g.invalidate()
    c = g.clone()
    assert c.selected is c.backline[0][-1] and c.selected is not g.selected


def test_clone_preserves_subclass():
    class Variant(Game):
        pass

    g = Variant(CONFIG)
    g.reset(1)
    assert type(g.clone()) is Variant
    assert snapshot(g.clone()) == snapshot(g)


# ---------------------------------------------------------------- Unit.copy
def test_unit_copy_covers_every_slot():
    u = Unit(3, 1, atk=4, hp=2, max_hp=5, armor=1, defense=True, nature=2, move_cost=3, uid=17,
             summoned=True, moved=False, attacked=True)
    c = u.copy()
    assert c is not u and type(c) is Unit
    assert set(Unit.__slots__) == set(UNIT_FIELDS), "update Unit.copy() and the tests"
    for name in Unit.__slots__:
        assert getattr(c, name) == getattr(u, name), name
    c.hp = 9
    c.attacked = False
    assert (u.hp, u.attacked) == (2, True)


def test_unit_keyword_defaults_and_from_card():
    u = Unit(0, 1, atk=2, hp=3)
    assert (u.max_hp, u.armor, u.defense, u.nature, u.move_cost, u.uid) == (3, 0, False, 0, 1, -1)
    assert (u.summoned, u.moved, u.attacked) == (False, False, False)
    with pytest.raises(TypeError):
        Unit(0, 1, 2, 3)  # keyword-only after owner
    for c in CONFIG.cards.cards:
        v = Unit.from_card(c, 1, uid=4)
        assert (v.card, v.owner, v.uid, v.atk, v.hp, v.max_hp, v.armor, v.defense, v.nature, v.move_cost) == (
            c.index, 1, 4, c.attack, c.health, c.health, c.armor, c.defense, c.nature, c.move_cost)
        assert (v.summoned, v.moved, v.attacked) == (True, False, False)


def test_unit_subclass_slots_survive_copy():
    class TraitUnit(Unit):
        __slots__ = ("tags",)

    u = TraitUnit(1, 0, atk=2, hp=3, armor=1)
    u.tags = ["shield"]
    c = u.copy()
    assert type(c) is TraitUnit and c.tags == ["shield"] and c.tags is not u.tags
    assert (c.hp, c.armor) == (3, 1)
