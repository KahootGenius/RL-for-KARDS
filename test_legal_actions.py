"""Illegal actions are never offered: the engine's legal set equals an independent reference
derived from SPEC.md, every offered action works, and every other index is rejected cleanly."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from cardgame.actions import ActionKind, ActionSpace
from cardgame.engine import Game, IllegalActionError
from conftest import (CONFIG, KIND_OF, NUM_ACTIONS, PROFILE_NAMES, SPACE, H, Z, check_invariants,
                      iter_states, new_game, snapshot, state_key)
from reference_rules import (KIND_NAMES, action_index, decode_index, num_actions, reference_legal)

OUT_OF_RANGE = (-1, -2, -NUM_ACTIONS, NUM_ACTIONS, NUM_ACTIONS + 1, 10**9)


# ---------------------------------------------------------------- action space layout (SPEC §3)
def test_action_space_matches_spec_table():
    assert NUM_ACTIONS == num_actions(H, Z) == 71
    assert len(SPACE) == NUM_ACTIONS and new_game(0).action_space.n == NUM_ACTIONS
    seen = set()
    for idx in range(NUM_ACTIONS):
        kind, a, b = decode_index(idx, H, Z)
        action = SPACE.decode(idx)
        assert (int(action.kind), action.a, action.b) == (kind, a, b), idx
        assert action.kind.name == KIND_NAMES[kind]
        assert SPACE.encode(action.kind, action.a, action.b) == idx
        assert action_index(kind, a, b, H, Z) == idx
        seen.add(action)
    assert len(seen) == NUM_ACTIONS
    assert SPACE.encode(ActionKind.END_TURN) == 0
    assert SPACE.encode(ActionKind.BACK_ATTACK, Z - 1, Z - 1) == NUM_ACTIONS - 1


def test_action_space_size_follows_config():
    space = ActionSpace(max_hand_size=7, zone_capacity=3)
    assert space.n == num_actions(7, 3)
    for idx in range(space.n):
        kind, a, b = decode_index(idx, 7, 3)
        action = space.decode(idx)
        assert (int(action.kind), action.a, action.b) == (kind, a, b)
        assert space.encode(ActionKind(kind), a, b) == idx


# ---------------------------------------------------------------- per-state checks
class Coverage:
    """Tracks which corners of the state space the fuzz actually reached."""

    def __init__(self):
        self.states = 0
        self.offered = Counter()
        self.flags = Counter()

    def record(self, game: Game, legal: list) -> None:
        self.states += 1
        self.offered.update(legal)
        if game.done:
            self.flags["win" if game.winner() in (0, 1) else "draw"] += 1
            return
        p, o = game.current, 1 - game.current
        back, front, fo = game.backline[p], game.frontline, game.front_owner
        any_ready_back = any(u.ready for u in back)
        affordable = any(CONFIG.cards[c].cost <= game.coins[p] for c in game.hands[p])
        flags = {
            "hand_full": len(game.hands[p]) == H,
            "burned": game.burned[p] > 0,
            "back_full_blocks_play": len(back) == Z and affordable,
            "front_full_blocks_move": fo == p and len(front) == Z and any_ready_back,
            "enemy_front_blocks_move": fo == o and any_ready_back,
            "enemy_back_full": fo == p and len(game.backline[o]) == Z,
            "exhausted_front": fo == p and any(not u.ready for u in front),
            "unaffordable_card": any(CONFIG.cards[c].cost > game.coins[p] for c in game.hands[p]),
            "deck_empty": not game.decks[p],
        }
        self.flags.update(k for k, v in flags.items() if v)


def check_state(game: Game, cov: Coverage, where: str) -> None:
    before = snapshot(game)
    legal = game.legal_actions()
    ref = reference_legal(game)
    assert legal == ref, (
        f"{where}: engine/reference legal sets differ\n{game.render()}\n"
        f"engine only: {[SPACE.describe(a) for a in sorted(set(legal) - set(ref))]}\n"
        f"reference only: {[SPACE.describe(a) for a in sorted(set(ref) - set(legal))]}")
    assert all(type(a) is int for a in legal), where
    assert legal == sorted(set(legal)), where

    mask = game.legal_mask()
    assert isinstance(mask, np.ndarray) and mask.dtype == np.bool_ and mask.shape == (NUM_ACTIONS,)
    assert np.flatnonzero(mask).tolist() == legal, where
    # Returned containers are copies: scribbling on them must not change the engine.
    legal.append(-5)
    mask[:] = ~mask
    assert game.legal_actions() == ref and np.flatnonzero(game.legal_mask()).tolist() == ref, where
    legal = ref

    if game.done:
        assert legal == [] and game.winner() is not None, where
    else:
        assert legal[0] == SPACE.END_TURN, where
    cov.record(game, legal)

    for a in legal:  # every offered action is accepted
        c = game.clone()
        c.step(a)
        check_invariants(c)

    offered = set(legal)
    c = game.clone()
    c_before = snapshot(c)
    key = state_key(c, rng=False)  # the RNG state is compared once, in the final snapshot
    for a in [*range(NUM_ACTIONS), *OUT_OF_RANGE]:  # everything else is rejected, harmlessly
        if a in offered:
            continue
        with pytest.raises(IllegalActionError):
            c.step(a)
        assert state_key(c, rng=False) == key, f"{where}: rejected action {a} mutated the game"
        assert c.legal_actions() == legal, f"{where}: rejected action {a} changed the legal set"
    assert snapshot(c) == c_before, where
    assert snapshot(game) == before, f"{where}: checking a state mutated it"


def run_fuzz(seeds, profiles=None) -> Coverage:
    cov = Coverage()
    for seed in seeds:
        for step, (game, plays, _) in enumerate(iter_states(seed, profiles)):
            check_invariants(game, plays)
            check_state(game, cov, f"seed {seed} step {step}")
    return cov


def test_fuzz_uniform_random_play():
    cov = run_fuzz(range(1000, 1010), profiles=("uniform", "uniform"))
    assert cov.states >= 1000
    assert cov.flags["win"] + cov.flags["draw"] == 10 and cov.flags["win"] > 0


def test_fuzz_biased_play_reaches_full_boards():
    # One game per ordered (seat 0, seat 1) pair of policy profiles.
    cov = run_fuzz(range(len(PROFILE_NAMES) ** 2))
    assert cov.states >= 3000
    missing = [SPACE.describe(a) for a in range(NUM_ACTIONS) if not cov.offered[a]]
    assert not missing, f"fuzz never offered {missing}"
    for flag in ("hand_full", "burned", "back_full_blocks_play", "front_full_blocks_move",
                 "enemy_front_blocks_move", "enemy_back_full", "exhausted_front",
                 "unaffordable_card", "deck_empty", "win", "draw"):
        assert cov.flags[flag] > 0, f"fuzz never reached {flag}: {dict(cov.flags)}"


# ---------------------------------------------------------------- small targeted cases
def test_numpy_integer_actions_are_accepted():
    g = new_game(3)
    legal = g.legal_actions()
    g.step(np.int64(legal[-1]))
    assert g.legal_actions() == reference_legal(g)


def test_nothing_is_legal_after_the_game_ends():
    for game, _, action in iter_states(11, ("aggro", "aggro")):
        if action is None:
            assert game.done and game.legal_actions() == [] and not game.legal_mask().any()
            for a in (0, *range(NUM_ACTIONS)):
                c = game.clone()
                with pytest.raises(IllegalActionError):
                    c.step(a)
                assert state_key(c) == state_key(game)


def test_kind_table_is_consistent():
    # KIND_OF (used by the fuzz policies) agrees with the SPEC index formulas.
    for idx in range(NUM_ACTIONS):
        assert int(KIND_OF[idx]) == decode_index(idx, H, Z)[0]
