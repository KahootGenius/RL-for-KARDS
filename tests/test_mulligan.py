"""Independent Stage 3 mulligan tests (SPEC 2.1, 2.2, 2.3, 5; configs built with mulligan=True)."""
from __future__ import annotations

import random

import pytest

from stage3_helpers import (CONFIRM, END, MAIN, MULLIGAN, Game, IllegalActionError, Rules, mull, multiset,
                            set_deck, set_hand)

SEEDS = range(6)


def new(seed, mulligan=True, rules=None):
    r = rules or Rules(mulligan=mulligan)
    g = Game(r.config)
    g.reset(seed)
    return r, g


def test_setup_follows_spec_2_1():
    # SPEC 2.1.3-4: rng = Random(seed); first_player = rng.randrange(2); shuffle seat 0's deck then
    # seat 1's (top = end); opening hands first 4, second 5. With the mulligan on, nothing else uses
    # the RNG before the first CONFIRM (SPEC 2.2).
    for seed in SEEDS:
        r, g = new(seed)
        ref = random.Random(seed)
        first = ref.randrange(2)
        decks = [list(g.decklists[0]), list(g.decklists[1])]
        assert all(d == sorted(d) and len(d) == 40 for d in decks)
        ref.shuffle(decks[0])
        ref.shuffle(decks[1])
        assert g.first_player == first
        for p, n in ((first, 4), (1 - first, 5)):
            assert list(g.hands[p]) == sorted(decks[p][-n:])
            assert list(g.deck_cards[p]) == decks[p][:-n]
        assert g.rng.getstate() == ref.getstate()


def test_initial_mulligan_state():
    # SPEC 2.1.5: phase MULLIGAN, first player to act; SPEC 2.2/2.6: only MULLIGAN(i) for unmarked
    # slots i < len(hand) and CONFIRM are legal; SPEC 5 mulligan_marks visible to the decider only.
    for seed in SEEDS:
        r, g = new(seed)
        f, s = g.first_player, 1 - g.first_player
        assert g.phase == MULLIGAN and g.current_player() == f
        assert g.turn == 0 and g.observe(f).turn == 0 and g.observe(s).turn == 0  # SPEC 2.13
        assert (len(g.hands[f]), len(g.hands[s])) == (4, 5)
        assert g.legal_actions() == [mull(i) for i in range(4)] + [CONFIRM]
        assert list(g.mulligan_done) == [False, False] and len(g.mulligan_marks) == 0
        assert g.pending is None and len(g.queue) == 0
        of, os_ = g.observe(f), g.observe(s)
        assert of.phase == MULLIGAN and os_.phase == MULLIGAN
        assert tuple(of.mulligan_marks) == (False,) * 4
        assert tuple(os_.mulligan_marks) == ()
        assert of.is_my_turn and not os_.is_my_turn


def test_marks_are_irreversible_and_only_for_unmarked_slots():
    # SPEC 2.2: MULLIGAN(i) legal for unmarked slots; a mark cannot be undone.
    r, g = new(3)
    f = g.first_player
    hand = list(g.hands[f])
    g.step(mull(1))
    g.step(mull(3))
    assert set(g.mulligan_marks) == {1, 3}
    assert g.legal_actions() == [mull(0), mull(2), CONFIRM]
    assert tuple(g.observe(f).mulligan_marks) == (False, True, False, True)
    assert list(g.hands[f]) == hand  # marking changes nothing else
    for bad in (mull(1), mull(4), END):
        with pytest.raises(IllegalActionError):
            g.step(bad)
    assert set(g.mulligan_marks) == {1, 3}


def test_confirm_draws_first_then_shuffles_the_marked_cards_back():
    # SPEC 2.2 CONFIRM with k >= 1: 1. draw k from the deck, 2. put the marked cards into the deck and
    # shuffle it (engine RNG), 3. re-sort the hand. A replaced card is never redrawn.
    r, g = new(5)
    f = g.first_player
    set_hand(g, r, f, ["fill00", "fill00", "fill01", "fill02"])
    rest = [f"fill{i:02d}" for i in (3, 4, 5, 6, 7, 8)] * 5 + ["fill09", "fill10", "fill11", "fill12"]
    set_deck(g, r, f, rest + ["fill13", "fill12"])  # top = fill12, then fill13
    deck_before = list(g.deck_cards[f])
    state = g.rng.getstate()
    g.step(mull(0))
    g.step(mull(1))  # both copies of fill00
    g.step(CONFIRM)
    expected_hand = sorted(r.idx(c) for c in ["fill01", "fill02", "fill12", "fill13"])
    assert list(g.hands[f]) == expected_hand
    assert r.idx("fill00") not in g.hands[f]
    marked = [r.idx("fill00")] * 2
    assert multiset(g.deck_cards[f]) == multiset(deck_before[:-2] + marked)
    assert len(g.deck_cards[f]) == len(deck_before)
    # SPEC 2.13: the marked cards are appended to the deck in slot order, then the deck is shuffled
    # (engine RNG).
    arrangement = deck_before[:-2] + marked
    ref = random.Random()
    ref.setstate(state)
    ref.shuffle(arrangement)
    assert list(g.deck_cards[f]) == arrangement and g.rng.getstate() == ref.getstate()


@pytest.mark.parametrize("marks", [(3, 0, 2), (0, 2, 3), (2, 3, 0)])
def test_marked_cards_are_appended_in_slot_order_before_the_shuffle(marks):
    # SPEC 2.2 + 2.13: draw k, append the marked cards in slot order (not marking order), shuffle.
    r, g = new(6)
    f = g.first_player
    set_hand(g, r, f, ["fill00", "fill01", "fill02", "fill03"])
    deck = [f"fill{i:02d}" for i in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13)]
    set_deck(g, r, f, deck)
    hand = list(g.hands[f])
    deck_before = list(g.deck_cards[f])
    state = g.rng.getstate()
    for i in marks:
        g.step(mull(i))
    g.step(CONFIRM)
    k = len(marks)
    drawn = deck_before[-k:]
    kept = [c for i, c in enumerate(hand) if i not in marks]
    assert list(g.hands[f]) == sorted(kept + drawn)
    arrangement = deck_before[:-k] + [hand[i] for i in sorted(marks)]
    ref = random.Random()
    ref.setstate(state)
    ref.shuffle(arrangement)
    assert list(g.deck_cards[f]) == arrangement
    assert g.rng.getstate() == ref.getstate()


def test_deck_shorter_than_k_replaces_only_the_first_marks_in_slot_order():
    # SPEC 2.13: with a deck shorter than k, only the first marks in slot order are replaced.
    r, g = new(7)
    f = g.first_player
    set_hand(g, r, f, ["fill00", "fill01", "fill02", "fill03"])
    set_deck(g, r, f, ["fill05", "fill06"])  # 2 cards, 3 marks
    hand = list(g.hands[f])
    state = g.rng.getstate()
    for i in (3, 1, 0):
        g.step(mull(i))
    g.step(CONFIRM)
    # slots 0 and 1 are replaced (fill00, fill01); slot 3 (fill03) stays in the hand
    assert list(g.hands[f]) == sorted([hand[2], hand[3], r.idx("fill05"), r.idx("fill06")])
    arrangement = [hand[0], hand[1]]
    ref = random.Random()
    ref.setstate(state)
    ref.shuffle(arrangement)
    assert list(g.deck_cards[f]) == arrangement and g.rng.getstate() == ref.getstate()


def test_full_mulligan_replaces_every_card():
    r, g = new(8)
    f = g.first_player
    set_hand(g, r, f, ["fill00", "fill01", "fill02", "fill03"])
    set_deck(g, r, f, ["fill04"] * 3 + ["fill05"] * 3 + ["fill06", "fill07", "fill08", "fill09"])
    for i in range(4):
        g.step(mull(i))
    g.step(CONFIRM)
    assert list(g.hands[f]) == sorted(r.idx(c) for c in ["fill06", "fill07", "fill08", "fill09"])
    assert multiset(g.deck_cards[f]) == multiset(
        r.idx(c) for c in ["fill04"] * 3 + ["fill05"] * 3 + ["fill00", "fill01", "fill02", "fill03"])


def test_confirm_without_marks_leaves_rng_hand_and_deck_untouched():
    # SPEC 2.2: with k = 0 nothing is drawn and the RNG is not used.
    for seed in SEEDS:
        r, g = new(seed)
        f = g.first_player
        hand, deck, state = list(g.hands[f]), list(g.deck_cards[f]), g.rng.getstate()
        g.step(CONFIRM)
        assert list(g.hands[f]) == hand and list(g.deck_cards[f]) == deck
        assert g.rng.getstate() == state


def test_order_first_then_second_then_round_one_starts():
    # SPEC 2.2: the first player decides, then the second; after the second CONFIRM round 1 starts
    # with the first player's turn (SPEC 2.3: turn += 1, coins, draw 1).
    for seed in SEEDS:
        r, g = new(seed)
        f, s = g.first_player, 1 - g.first_player
        turn0 = g.turn
        assert turn0 == 0  # SPEC 2.13: turn is 0 during the mulligan; the first turn is 1
        g.step(mull(0))
        g.step(CONFIRM)
        assert g.phase == MULLIGAN and g.current_player() == s
        assert bool(g.mulligan_done[f]) and not g.mulligan_done[s]
        assert len(g.mulligan_marks) == 0
        assert g.legal_actions() == [mull(i) for i in range(5)] + [CONFIRM]
        assert tuple(g.observe(s).mulligan_marks) == (False,) * 5
        assert tuple(g.observe(f).mulligan_marks) == ()
        assert g.turn == turn0
        n_first = len(g.hands[f])
        g.step(mull(4))
        g.step(CONFIRM)
        assert g.phase == MAIN and g.current_player() == f and g.round == 1
        assert g.turn == turn0 + 1
        assert list(g.mulligan_done) == [True, True]
        assert g.coins[f] == r.config.coins_for_round(1) + g.coin_bonus[f] and g.coins[s] == 0
        assert len(g.hands[f]) == n_first + 1 and len(g.hands[s]) == 5
        assert tuple(g.observe(f).mulligan_marks) == () and g.observe(f).phase == MAIN
        assert END in g.legal_actions()
        assert all(a < 126 for a in g.legal_actions())
        with pytest.raises(IllegalActionError):
            g.step(CONFIRM)


def test_mulligan_off_starts_round_one_at_once():
    # SPEC 2.1.5 / 1.5: mulligan=False skips the phase; the turn index matches the mulligan game.
    for seed in SEEDS:
        r_on, g_on = new(seed)
        g_on.step(CONFIRM)
        g_on.step(CONFIRM)
        r_off, g_off = new(seed, mulligan=False)
        f = g_off.first_player
        assert g_off.phase == MAIN and g_off.current_player() == f
        assert g_off.turn == 1 and g_off.observe(f).turn == 1  # SPEC 2.13: the first turn is 1
        assert len(g_off.hands[f]) == 5 and len(g_off.hands[1 - f]) == 5
        assert g_off.coins[f] == r_off.config.coins_for_round(1)
        assert g_off.turn == g_on.turn
        assert list(g_off.hands[0]) == list(g_on.hands[0]) and list(g_off.hands[1]) == list(g_on.hands[1])
        assert g_off.rng.getstate() == g_on.rng.getstate()
        assert all(a < 126 for a in g_off.legal_actions())


def test_opponent_sees_nothing_of_the_first_players_mulligan():
    # SPEC 2.2 hidden: the opponent sees nothing; hand and deck sizes do not change.
    for seed in SEEDS:
        r, g = new(seed)
        f, s = g.first_player, 1 - g.first_player
        before = g.observe(s)
        g.step(mull(0))
        g.step(mull(2))
        assert g.observe(s) == before
        g.step(CONFIRM)
        _, plain = new(seed)
        plain.step(CONFIRM)
        assert g.observe(s) == plain.observe(s)
        assert len(g.hands[f]) == 4 and len(g.deck_cards[f]) == len(plain.deck_cards[f])


def test_first_player_sees_nothing_of_the_second_players_mulligan():
    for seed in SEEDS:
        r, g = new(seed)
        f, s = g.first_player, 1 - g.first_player
        g.step(CONFIRM)
        before = g.observe(f)
        for i in (0, 1, 4):
            g.step(mull(i))
        assert g.observe(f) == before
        g.step(CONFIRM)  # round 1 starts: f draws from f's own deck
        _, plain = new(seed)
        plain.step(CONFIRM)
        plain.step(CONFIRM)
        assert g.observe(f) == plain.observe(f)


def test_second_players_confirm_draws_then_shuffles():
    r, g = new(11)
    f, s = g.first_player, 1 - g.first_player
    g.step(CONFIRM)
    set_hand(g, r, s, ["fill00", "fill01", "fill02", "fill03", "fill04"])
    deck = ["fill05"] * 3 + ["fill06"] * 3 + ["fill07", "fill08", "fill09"]
    set_deck(g, r, s, deck)
    state = g.rng.getstate()
    g.step(mull(4))  # fill04
    g.step(mull(0))  # fill00
    g.step(CONFIRM)
    assert list(g.hands[s]) == sorted(r.idx(c) for c in ["fill01", "fill02", "fill03", "fill09", "fill08"])
    assert multiset(g.deck_cards[s]) == multiset(
        r.idx(c) for c in ["fill05"] * 3 + ["fill06"] * 3 + ["fill07", "fill00", "fill04"])
    # SPEC 2.13: appended in slot order (fill00 from slot 0, then fill04 from slot 4), then shuffled
    arrangement = [r.idx(c) for c in deck[:-2] + ["fill00", "fill04"]]
    ref = random.Random()
    ref.setstate(state)
    ref.shuffle(arrangement)
    assert list(g.deck_cards[s]) == arrangement and g.rng.getstate() == ref.getstate()


def test_mulligan_marks_are_cleared_for_the_second_decider():
    r, g = new(2)
    g.step(mull(0))
    g.step(mull(1))
    g.step(CONFIRM)
    assert len(g.mulligan_marks) == 0
    assert mull(0) in g.legal_actions() and mull(1) in g.legal_actions()


def test_mulligan_is_deterministic():
    # SPEC 4 determinism, through the mulligan.
    def run(seed):
        r, g = new(seed)
        g.step(mull(0))
        g.step(mull(3))
        g.step(CONFIRM)
        g.step(mull(2))
        g.step(CONFIRM)
        return (list(map(list, g.hands)), list(map(list, g.deck_cards)), g.rng.getstate(), g.observe(0), g.observe(1))
    for seed in SEEDS:
        assert run(seed) == run(seed)
