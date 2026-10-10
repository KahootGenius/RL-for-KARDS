"""Independent Stage 3 hidden-information tests (SPEC 5, 2.2 hidden, 4 state).

`observe(p)` and p's legal actions must not change when the opponent's unknown hand cards, deck
contents/order, mulligan marks and the RNG are replaced by other legal values with the same sizes and
the same `revealed` / `known_hand` (SPEC 5 "No-leak tests"). Covered states: before and after the
opponent's mulligan, after the opponent's draw effects, before and after random effects, pending
choices, and a sweep over seeded random games. The `revealed` / `known_hand` bookkeeping rules and
the egocentric Observation fields are checked too.
"""
from __future__ import annotations

import random
from collections import Counter

import pytest

import cardgame.cards as cards_mod
from stage3_helpers import (CHOICE, CONFIRM, MULLIGAN, Game, Rules, blank, counts, effect, hand_slot, mull,
                            multiset, operation, perturb_hidden, pick, play, put, rich_rules, set_hand, tgt, unit)

RULES = rich_rules(mulligan=True)
CFG = RULES.config
N = RULES.n_cards
TOKENS = {i for i, c in enumerate(RULES.card_dicts) if c.get("token")}


def new_game(seed, random_frac=0.5, rules=RULES):
    g = Game(rules.config)
    g.reset(seed, cards_mod.sample_deal(seed, rules.config, random_frac))
    return g


def assert_no_leak(g, p, rng, tries=2):
    """observe(p) and (on p's decision) p's legal actions are invariant under hidden perturbations."""
    obs = g.observe(p)
    deciding = not g.done and g.current_player() == p
    legal = g.legal_actions() if deciding else None
    mask = g.legal_mask().tobytes() if deciding else None
    for k in range(tries):
        for new_decklist in (False, True):
            h = perturb_hidden(g, p, rng, new_decklist=new_decklist)
            assert h.observe(p) == obs, (p, new_decklist)
            if deciding:
                assert h.legal_actions() == legal
                assert h.legal_mask().tobytes() == mask


def find_play(seed_range, cid, rules=RULES, max_steps=600):
    """A state (clone) where the current player can PLAY card `cid` (phase MAIN), from random play."""
    idx = rules.idx(cid)
    for seed in seed_range:
        g = new_game(seed, rules=rules)
        pol = random.Random(seed + 17)
        for _ in range(max_steps):
            if g.done:
                break
            p = g.current_player()
            if g.phase not in (MULLIGAN, CHOICE) and idx in g.hands[p]:
                a = play(list(g.hands[p]).index(idx))
                if a in g.legal_actions():
                    return g.clone(), a
            g.step(pol.choice(g.legal_actions()))
    raise AssertionError(f"no state found where {cid} can be played")


# ================================================================= the observation schema (SPEC 5)
STAGE2_OBS = ("player", "is_my_turn", "went_first", "round", "my_deck", "my_coins", "opp_coins", "my_base_hp",
              "opp_base_hp", "hand", "opp_hand_size", "my_deck_size", "opp_deck_size", "my_played", "opp_played",
              "my_backline", "opp_backline", "frontline", "front_owner", "done", "result")
STAGE3_OBS = ("phase", "mulligan_marks", "pending", "turn", "my_coin_bonus", "opp_coin_bonus", "my_decklist",
              "my_deck_counts", "my_known_hand", "opp_known_hand", "opp_revealed", "my_discard", "opp_discard",
              "my_graveyard", "opp_graveyard", "my_burned", "opp_burned", "my_history", "opp_history")
UNIT_VIEW_ADDED = ("pin_turns", "temp_move_cost", "temp_traits", "temp_removed")  # SPEC 5 UnitView additions


def test_observation_schema():
    # SPEC 5: Stage 2 fields keep their positions; new fields are appended with defaults. UnitView gains
    # ambush, shock, immune, ambush_ready (phase 1b), then pin_turns, temp_move_cost, temp_traits,
    # temp_removed; PendingView = (card, effect, action, amount, select, side, kind, atk, hp, previews);
    # Observation ends with my_history / opp_history.
    from cardgame.engine import Observation, PendingView, UnitView
    assert Observation._fields[:len(STAGE2_OBS)] == STAGE2_OBS
    assert set(Observation._fields[len(STAGE2_OBS):]) == set(STAGE3_OBS)
    assert set(STAGE3_OBS) <= set(Observation._field_defaults)
    assert Observation._fields[-2:] == ("my_history", "opp_history")
    assert UnitView._fields == ("card", "atk", "hp", "max_hp", "armor", "defense", "nature", "move_cost",
                                "summoned", "moved", "attacked", "can_move", "can_attack", "blitz", "smokescreen",
                                "fury", "pinned", "attacks", "temp_atk", "temp_hp", "token", "ambush", "shock",
                                "immune", "ambush_ready") + UNIT_VIEW_ADDED
    assert set(UNIT_VIEW_ADDED) <= set(UnitView._field_defaults)
    assert PendingView._fields == ("card", "effect", "action", "amount", "select", "side", "kind", "atk", "hp",
                                   "previews")
    assert "previews" in PendingView._field_defaults
    g = new_game(0)
    obs = g.observe(g.first_player)
    for name in ("my_decklist", "my_deck_counts", "my_known_hand", "opp_known_hand", "opp_revealed", "my_discard",
                 "opp_discard", "my_graveyard", "opp_graveyard"):
        assert len(getattr(obs, name)) == N, name  # counts per card index (tuples of length n_cards)
    # (operations played, units deployed, units died) this turn, then the same three this game
    assert obs.my_history == obs.opp_history == (0, 0, 0, 0, 0, 0)


def test_pending_previews_and_history_are_public_and_shaped():
    # SPEC 5: PendingView.previews has one (kills, dealt, healed, other) tuple per CHOOSE slot (3Z+2), zeros
    # for slots that are not options; my_history / opp_history mirror each other between the two views.
    n_choose = 3 * CFG.zone_capacity + 2
    found = 0
    for seed in range(40):
        g = new_game(seed)
        pol = random.Random(seed)
        for _ in range(500):
            if g.done:
                break
            o0, o1 = g.observe(0), g.observe(1)
            assert (o0.my_history, o0.opp_history) == (o1.opp_history, o1.my_history)
            assert len(o0.my_history) == 6 and all(type(x) is int and x >= 0 for x in o0.my_history)
            if g.phase == CHOICE:
                pv = o0.pending
                assert pv == o1.pending and len(pv.previews) == n_choose
                options = {a - g.action_space.CHOOSE0 for a in g.legal_actions()}
                for t, prev in enumerate(pv.previews):
                    assert len(prev) == 4 and all(type(x) is int for x in prev)
                    if t not in options:
                        assert prev == (0, 0, 0, 0), (t, prev)
                    assert prev[0] in (0, 1)
                found += 1
            g.step(pol.choice(g.legal_actions()))
        if found >= 8:
            break
    assert found >= 8


# ================================================================= explicit states (SPEC 5 list)
def test_no_leak_before_and_after_the_opponents_mulligan():
    rng = random.Random(1)
    for seed in range(5):
        g = new_game(seed)
        f, s = g.first_player, 1 - g.first_player
        assert_no_leak(g, s, rng)       # f decides: f's hand, deck and marks are hidden from s
        g.step(mull(0))
        g.step(mull(3))
        assert_no_leak(g, s, rng)
        assert_no_leak(g, f, rng)
        g.step(CONFIRM)                 # after f's mulligan; s decides now
        assert_no_leak(g, s, rng)
        assert_no_leak(g, f, rng)
        g.step(mull(1))
        assert_no_leak(g, f, rng)       # s's marks are hidden from f
        g.step(CONFIRM)
        assert_no_leak(g, f, rng)
        assert_no_leak(g, s, rng)


@pytest.mark.parametrize("cid", ["intel", "scout"])
def test_no_leak_after_an_opponent_draw_effect(cid):
    rng = random.Random(2)
    g, a = find_play(range(60), cid)
    o = g.current_player()
    p = 1 - o
    assert_no_leak(g, p, rng)
    g.step(a)
    while g.phase == CHOICE:
        g.step(g.legal_actions()[0])
    assert_no_leak(g, p, rng)
    assert_no_leak(g, o, rng)


@pytest.mark.parametrize("cid", ["volley", "sniper", "sabotage", "spy"])
def test_no_leak_before_and_after_random_effects(cid):
    rng = random.Random(3)
    g, a = find_play(range(60), cid)
    o = g.current_player()
    for p in (0, 1):
        assert_no_leak(g, p, rng)
    g.step(a)
    for p in (0, 1):
        assert_no_leak(g, p, rng)


def test_no_leak_at_pending_choices():
    rng = random.Random(4)
    found = 0
    for seed in range(60):
        g = new_game(seed)
        pol = random.Random(seed)
        for _ in range(500):
            if g.done:
                break
            if g.phase == CHOICE:
                for p in (0, 1):
                    assert_no_leak(g, p, rng)
                assert g.observe(0).pending == g.observe(1).pending  # SPEC 5: pending is public
                found += 1
            g.step(pol.choice(g.legal_actions()))
        if found >= 6:
            break
    assert found >= 6


@pytest.mark.parametrize("seed", range(6))
def test_no_leak_sweep(seed):
    rng = random.Random(100 + seed)
    g = new_game(seed)
    pol = random.Random(seed * 31 + 3)
    for i in range(700):
        if i % 4 == 0 or g.done:
            for p in (0, 1):
                assert_no_leak(g, p, rng, tries=1)
        if g.done:
            break
        g.step(pick(g, pol, hoard=seed % 2 == 1))


# ================================================================= bookkeeping invariants (SPEC 4, 5)
def check_bookkeeping(g):
    for p in (0, 1):
        o = 1 - p
        known = counts(g.known_hand[p], N)
        rev = counts(g.revealed[p], N)
        hand = multiset(g.hands[p])
        dl = multiset(g.decklists[p])
        assert all(known[c] <= hand[c] for c in range(N)), "known_hand counts cards in the hand"
        assert all(rev[c] <= dl[c] for c in range(N)), "revealed counts copies from the decklist"
        assert not any(rev[c] for c in TOKENS), "tokens never enter revealed"
        unknown_hand = len(g.hands[p]) - sum(known)
        leftover = CFG.deck_size - sum(rev) - unknown_hand - len(g.deck_cards[p])
        assert 0 <= leftover <= g.burned[p], "unseen deck cards = unknown hand + deck + burned draws"
        obs = g.observe(p)
        assert counts(obs.my_known_hand, N) == known and counts(obs.opp_known_hand, N) == counts(g.known_hand[o], N)
        assert counts(obs.opp_revealed, N) == counts(g.revealed[o], N)
        assert counts(obs.my_discard, N) == counts(g.discard[p], N)
        assert counts(obs.opp_discard, N) == counts(g.discard[o], N)
        assert counts(obs.my_graveyard, N) == counts(g.graveyard[p], N)
        assert counts(obs.opp_graveyard, N) == counts(g.graveyard[o], N)
        assert counts(obs.my_decklist, N) == [dl[c] for c in range(N)]
        deck = multiset(g.deck_cards[p])
        assert counts(obs.my_deck_counts, N) == [deck[c] for c in range(N)]
        assert (obs.my_burned, obs.opp_burned) == (g.burned[p], g.burned[o])
        assert (obs.my_coin_bonus, obs.opp_coin_bonus) == (g.coin_bonus[p], g.coin_bonus[o])
        assert obs.turn == g.turn and obs.phase == g.phase
        if g.phase == MULLIGAN and not g.done and g.current_player() == p:
            assert tuple(obs.mulligan_marks) == tuple(i in set(g.mulligan_marks) for i in range(len(g.hands[p])))
        else:
            assert tuple(obs.mulligan_marks) == ()


@pytest.mark.parametrize("seed", range(6))
def test_bookkeeping_invariants_along_random_games(seed):
    g = new_game(seed)
    pol = random.Random(seed)
    burned = 0
    for _ in range(700):
        check_bookkeeping(g)
        burned = max(burned, g.burned[0] + g.burned[1])
        if g.done:
            break
        g.step(pick(g, pol, hoard=seed % 2 == 1))
    if seed % 2 == 1:
        assert burned > 0  # the hoarding games exercise the burn rule


# ================================================================= bookkeeping rules (SPEC 5)
def book_rules():
    return Rules([unit("ration", 0, 1, token=True),
                  unit("veteran", 2, 3, cost=1),
                  operation("flare", [effect("on_play", "damage", "enemy_base", amount=1)]),
                  operation("supply", [effect("on_play", "add_card", "controller", card="ration")]),
                  operation("recall", [effect("on_play", "return_to_hand", tgt("all", "friendly"))]),
                  operation("sabotage", [effect("on_play", "discard", "opponent", amount=1)])])


def kh(g, r, p, cid):
    return counts(g.known_hand[p], r.n_cards)[r.idx(cid)]


def rv(g, r, p, cid):
    return counts(g.revealed[p], r.n_cards)[r.idx(cid)]


def test_playing_an_unknown_card_reveals_a_new_copy():
    # SPEC 5: when p plays a non-token card c with known_hand[p][c] == 0: revealed[p][c] += 1.
    r = book_rules()
    g = blank(r)
    set_hand(g, r, 0, ["flare", "flare", "veteran"])
    g.step(play(hand_slot(g, r, 0, "flare")))
    g.step(play(hand_slot(g, r, 0, "flare")))
    g.step(play(hand_slot(g, r, 0, "veteran")))
    assert (rv(g, r, 0, "flare"), rv(g, r, 0, "veteran")) == (2, 1)
    assert counts(g.observe(1).opp_revealed, r.n_cards)[r.idx("flare")] == 2
    assert sum(counts(g.known_hand[0], r.n_cards)) == 0


def test_returned_card_is_known_until_played_again():
    # SPEC 5: return_to_hand of p's non-token unit: known_hand[p][c] += 1; playing it again
    # decrements known_hand and reveals nothing new.
    r = book_rules()
    g = blank(r)
    set_hand(g, r, 0, ["veteran"])
    g.step(play(0))
    assert rv(g, r, 0, "veteran") == 1
    set_hand(g, r, 0, ["recall"])
    g.step(play(0))
    assert list(g.hands[0]) == [r.idx("veteran")] and kh(g, r, 0, "veteran") == 1
    assert counts(g.observe(1).opp_known_hand, r.n_cards)[r.idx("veteran")] == 1
    g.step(play(0))
    assert kh(g, r, 0, "veteran") == 0 and rv(g, r, 0, "veteran") == 1


def test_added_tokens_are_known_and_never_revealed():
    # SPEC 5: added token cards: known_hand += 1; tokens never enter revealed.
    r = book_rules()
    g = blank(r)
    set_hand(g, r, 0, ["supply"])
    g.step(play(0))
    assert kh(g, r, 0, "ration") == 1 and rv(g, r, 0, "ration") == 0
    assert counts(g.observe(1).opp_known_hand, r.n_cards)[r.idx("ration")] == 1
    g.step(play(0))  # the token unit is deployed
    assert kh(g, r, 0, "ration") == 0 and rv(g, r, 0, "ration") == 0


def test_discarded_unknown_card_is_revealed():
    # SPEC 5: discarding a non-token card c with known_hand == 0 reveals it (SPEC 2.10 discard).
    r = book_rules()
    g = blank(r)
    set_hand(g, r, 1, ["veteran"])
    set_hand(g, r, 0, ["sabotage"])
    g.step(play(0))
    assert list(g.hands[1]) == [] and rv(g, r, 1, "veteran") == 1
    assert counts(g.discard[1], r.n_cards)[r.idx("veteran")] == 1
    assert counts(g.observe(0).opp_discard, r.n_cards)[r.idx("veteran")] == 1


def test_mulligan_replacements_change_no_bookkeeping():
    # SPEC 2.2 hidden: marks and replacements are private (nothing revealed or known).
    g = new_game(4)
    before = [(counts(g.revealed[p], N), counts(g.known_hand[p], N)) for p in (0, 1)]
    g.step(mull(0))
    g.step(mull(1))
    g.step(CONFIRM)
    g.step(mull(2))
    g.step(CONFIRM)
    after = [(counts(g.revealed[p], N), counts(g.known_hand[p], N)) for p in (0, 1)]
    assert before == after and all(sum(x) == 0 for pair in after for x in pair)
