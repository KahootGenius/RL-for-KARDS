"""Illegal actions are never offered: the engine's legal set equals an independent reference derived
from SPEC.md, every offered action steps exactly as the reference says, and every other index is
rejected cleanly. States come from seeded, biased self-play over every deck pair."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from cardgame.actions import ActionKind, ActionSpace
from cardgame.cards import FAST, RANGED, load_ruleset
from cardgame.engine import Game, IllegalActionError
from conftest import (CONFIG, DECK_PAIRS, NUM_ACTIONS, PROFILE_PAIRS, SPACE, H, Z, check_invariants,
                      iter_states, new_game, snapshot, state_key)
from reference_rules import (KIND_NAMES, action_index, attackers, can_attack, can_move, comparable,
                             decode_index, defense_allows, extract_state, legal_from_state, num_actions,
                             reaches, reference_step, targets)

OUT_OF_RANGE = (-1, -2, -NUM_ACTIONS, NUM_ACTIONS, NUM_ACTIONS + 1, 10**9)
NOT_INTS = (True, False, 0.0, 1.0, np.float32(0), np.float64(3.0), "0", None, (0,))


# ---------------------------------------------------------------- action space layout (SPEC §3)
def test_action_space_matches_spec_layout():
    assert NUM_ACTIONS == num_actions(H, Z) == 126
    assert (SPACE.END_TURN, SPACE.PLAY0, SPACE.MOVE0, SPACE.ATTACK0) == (0, 1, 1 + H, 1 + H + Z)
    assert (SPACE.n_attackers, SPACE.n_targets, SPACE.BASE_TARGET) == (2 * Z, 2 * Z + 1, 2 * Z)
    assert len(SPACE) == NUM_ACTIONS and new_game(0).action_space.n == NUM_ACTIONS == new_game(0).num_actions
    assert [k.name for k in ActionKind] == list(KIND_NAMES)
    seen = set()
    for idx in range(NUM_ACTIONS):
        kind, a, b = decode_index(idx, H, Z)
        action = SPACE.decode(idx)
        assert (int(action.kind), action.a, action.b) == (kind, a, b), idx
        assert SPACE.encode(action.kind, action.a, action.b) == idx
        assert action_index(kind, a, b, H, Z) == idx
        if kind == ActionKind.ATTACK:
            assert SPACE.attack(a, b) == idx
        seen.add(action)
    assert len(seen) == NUM_ACTIONS
    assert SPACE.encode(ActionKind.ATTACK, 2 * Z - 1, 2 * Z) == NUM_ACTIONS - 1
    assert SPACE.describe(SPACE.attack(0, Z)) == "ATTACK(back0->front0)"
    assert SPACE.describe(SPACE.attack(Z + 1, 2 * Z)) == "ATTACK(front1->base)"


@pytest.mark.parametrize("hand, zone", [(7, 3), (1, 1), (10, 5), (12, 6)])
def test_action_space_size_follows_config(hand, zone):
    space = ActionSpace(max_hand_size=hand, zone_capacity=zone)
    assert space.n == num_actions(hand, zone)
    for idx in range(space.n):
        kind, a, b = decode_index(idx, hand, zone)
        action = space.decode(idx)
        assert (int(action.kind), action.a, action.b) == (kind, a, b)
        assert space.encode(ActionKind(kind), a, b) == idx


# ---------------------------------------------------------------- per-state checks
class Coverage:
    """Tracks which corners of the state space the fuzz actually reached."""

    def __init__(self):
        self.states = 0
        self.transitions = 0
        self.offered = Counter()
        self.flags = Counter()

    def record(self, game: Game, legal: list) -> None:
        self.states += 1
        self.offered.update(legal)
        if game.done:
            self.flags["win" if game.winner() in (0, 1) else "draw"] += 1
            return
        cfg, Zc, sp = game.config, game.config.zone_capacity, game.action_space
        s = extract_state(game)
        p, o = s.current, 1 - s.current
        back, front, fo, coins = s.backline[p], s.frontline, s.front_owner, s.coins[p]
        movers = [u for u in back if can_move(u)]
        legal_set = set(legal)
        f = self.flags
        f["hand_full"] += len(s.hands[p]) == cfg.max_hand_size
        f["burned"] += s.burned[p] > 0
        f["deck_empty"] += not s.deck_cards[p]
        f["unaffordable_card"] += any(cfg.cards[c].cost > coins for c in s.hands[p])
        f["back_full_blocks_play"] += len(back) == Zc and any(cfg.cards[c].cost <= coins for c in s.hands[p])
        f["enemy_back_full"] += len(s.backline[o]) == Zc
        f["front_full_blocks_move"] += fo == p and len(front) == Zc and any(u.move_cost <= coins for u in movers)
        f["enemy_front_blocks_move"] += fo == o and any(u.move_cost <= coins for u in movers)
        f["move_cost_blocks_move"] += fo != o and len(front) < Zc and any(u.move_cost > coins for u in movers)
        f["free_move_with_0_coins"] += coins == 0 and any(sp.MOVE0 <= a < sp.ATTACK0 for a in legal)
        f["summoned_unit"] += any(u.summoned for u in back)
        for a, att in attackers(s, Zc):
            if att.nature == FAST and att.moved and can_attack(att):
                f["fast_attack_after_move_ready"] += 1
            if att.nature == FAST and att.attacked and a < Zc and sp.MOVE0 + a in legal_set:
                f["fast_move_after_attack_legal"] += 1
            if not can_attack(att):
                continue
            defense_targets = 0
            for t, tgt, zone in targets(s, Zc):
                if not reaches(a, att, t, Zc):
                    continue
                kind = "ranged" if att.nature == RANGED else "melee"
                tz = "base" if tgt is None else ("back" if t < Zc else "front")
                if not defense_allows(tgt, zone):
                    f[f"defense_blocks_{kind}_{tz}"] += 1
                    continue
                f[f"attack_{kind}_{'back' if a < Zc else 'front'}_to_{tz}"] += 1
                if tgt is not None:
                    defense_targets += tgt.defense
                    f["zero_damage_attack"] += att.atk <= tgt.armor
                    f["armor_reduced_attack"] += 0 < tgt.armor < att.atk
            f["two_defense_targets"] += defense_targets >= 2

    def record_transition(self, before, after) -> None:
        """Flags for interesting transitions (reference states)."""
        self.transitions += 1
        if before.frontline and not after.frontline:
            lost = {u.owner for u in before.frontline}
            self.flags["front_emptied_" + ("own_attacker_died" if lost == {before.current} else "enemy_killed")] += 1
        n_before = sum(map(len, before.backline)) + len(before.frontline)
        n_after = sum(map(len, after.backline)) + len(after.frontline)
        self.flags["mutual_kill"] += n_before - n_after == 2
        self.flags["burn_on_draw"] += sum(after.burned) > sum(before.burned)


def check_state(game: Game, cov: Coverage, where: str, strict_rejects: bool = False) -> None:
    """Legal set == reference; mask == list; every offered action steps like the reference;
    every other index (and non-integers) is rejected without side effects."""
    before = snapshot(game)
    legal = game.legal_actions()
    ref = legal_from_state(extract_state(game), game.config)
    assert legal == ref, (
        f"{where}: engine/reference legal sets differ\n{game.render()}\n"
        f"engine only: {[game.describe(a) for a in sorted(set(legal) - set(ref))]}\n"
        f"reference only: {[game.describe(a) for a in sorted(set(ref) - set(legal))]}")
    assert all(type(a) is int for a in legal), where
    assert legal == sorted(set(legal)), where

    mask = game.legal_mask()
    assert isinstance(mask, np.ndarray) and mask.dtype == np.bool_ and mask.shape == (game.num_actions,)
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

    s = extract_state(game)
    for a in legal:  # every offered action is accepted and does what the spec says
        c = game.clone()
        c.step(a)
        expected = reference_step(s, a, game.config, check=False)  # legal set compared above
        got = extract_state(c)
        assert comparable(got) == comparable(expected), (
            f"{where}: {game.describe(a)} from\n{game.render()}\nengine:   {got}\nexpected: {expected}")
        check_invariants(c)
        cov.record_transition(s, expected)

    offered = set(legal)
    c = game.clone()
    c_before = snapshot(c)
    key = state_key(c) if strict_rejects else None
    for a in [*range(game.num_actions), *OUT_OF_RANGE, *NOT_INTS]:  # everything else is rejected
        if type(a) is int and a in offered:
            continue
        try:
            c.step(a)
        except IllegalActionError:
            pass
        else:
            raise AssertionError(f"{where}: non-offered action {a!r} was accepted\n{game.render()}")
        if strict_rejects:
            assert state_key(c) == key, f"{where}: rejected action {a!r} mutated the game"
            assert c.legal_actions() == legal, f"{where}: rejected action {a!r} changed the legal set"
    assert snapshot(c) == c_before, f"{where}: a rejected action mutated the game"
    assert snapshot(game) == before, f"{where}: checking a state mutated it"


def run_fuzz(seeds, deck_pair, profile_pair, config=CONFIG, strict_every: int = 25) -> Coverage:
    cov = Coverage()
    for seed in seeds:
        decks, profiles = deck_pair(seed), profile_pair(seed)
        for step, (game, plays, _) in enumerate(iter_states(seed, profiles, decks, config=config)):
            check_invariants(game, plays)
            check_state(game, cov, f"seed {seed} decks {decks} {profiles} step {step}",
                        strict_rejects=step % strict_every == 0)
    return cov


REQUIRED_FLAGS = (
    "hand_full", "burned", "deck_empty", "unaffordable_card", "back_full_blocks_play", "enemy_back_full",
    "front_full_blocks_move", "enemy_front_blocks_move", "move_cost_blocks_move", "free_move_with_0_coins",
    "summoned_unit", "fast_attack_after_move_ready", "fast_move_after_attack_legal",
    "attack_melee_back_to_front", "attack_melee_front_to_back", "attack_melee_front_to_base",
    "attack_ranged_back_to_back", "attack_ranged_back_to_front", "attack_ranged_back_to_base",
    "attack_ranged_front_to_back", "attack_ranged_front_to_base",
    "defense_blocks_melee_back", "defense_blocks_melee_front", "defense_blocks_ranged_back",
    "defense_blocks_ranged_front", "two_defense_targets", "zero_damage_attack", "armor_reduced_attack",
    "front_emptied_enemy_killed", "front_emptied_own_attacker_died", "mutual_kill", "burn_on_draw", "win", "draw",
)


def test_fuzz_every_deck_pair_and_profile_pair():
    # 48 games: every ordered deck pair 3 times, every ordered profile pair at least once.
    cov = run_fuzz(range(48), lambda s: DECK_PAIRS[s % len(DECK_PAIRS)],
                   lambda s: PROFILE_PAIRS[s % len(PROFILE_PAIRS)])
    assert cov.states >= 3500 and cov.transitions >= 30_000, (cov.states, cov.transitions)
    front_to_front = {SPACE.attack(Z + a, Z + t) for a in range(Z) for t in range(Z)}
    missing = [SPACE.describe(a) for a in range(NUM_ACTIONS) if not cov.offered[a] and a not in front_to_front]
    assert not missing, f"fuzz never offered {missing}"
    # A frontline attacker means we hold the frontline, so there is never an enemy frontline target.
    assert not any(cov.offered[a] for a in front_to_front)
    unreached = [f for f in REQUIRED_FLAGS if not cov.flags[f]]
    assert not unreached, f"fuzz never reached {unreached}: {dict(cov.flags)}"


def test_fuzz_small_zones_and_hands():
    # The engine follows config sizes (and the reference with it): tiny zones fill up constantly.
    cfg = load_ruleset(zone_capacity=2, max_hand_size=4)
    space = ActionSpace(4, 2)
    cov = run_fuzz(range(500, 516), lambda s: DECK_PAIRS[s % len(DECK_PAIRS)],
                   lambda s: PROFILE_PAIRS[(5 * s) % len(PROFILE_PAIRS)], config=cfg)
    assert cov.states >= 800
    front_to_front = {space.attack(2 + a, 2 + t) for a in range(2) for t in range(2)}
    assert all(cov.offered[a] for a in range(space.n) if a not in front_to_front)
    for flag in ("hand_full", "burned", "back_full_blocks_play", "front_full_blocks_move"):
        assert cov.flags[flag], flag


# ---------------------------------------------------------------- small targeted cases
def test_numpy_integer_actions_are_accepted():
    for dtype in (np.int64, np.int32, np.int16, np.uint8):
        g = new_game(3)
        legal = g.legal_actions()
        expected = reference_step(extract_state(g), legal[-1], CONFIG)
        g.step(dtype(legal[-1]))
        assert comparable(extract_state(g)) == comparable(expected)


def test_nothing_is_legal_after_the_game_ends():
    for seed in (11, 12):
        for game, _, action in iter_states(seed, ("fast_rush", "fast_rush")):
            if action is not None:
                continue
            assert game.done and game.legal_actions() == [] and not game.legal_mask().any()
            key = state_key(game)
            for a in (*range(NUM_ACTIONS), *OUT_OF_RANGE, *NOT_INTS):
                c = game.clone()
                with pytest.raises(IllegalActionError):
                    c.step(a)
                assert state_key(c) == key


def test_legal_mask_writes_into_given_array():
    g = new_game(5)
    out = np.ones(NUM_ACTIONS, dtype=bool)
    assert g.legal_mask(out=out) is out
    assert np.array_equal(out, g.legal_mask())
    assert np.flatnonzero(out).tolist() == g.legal_actions()
