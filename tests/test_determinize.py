"""Independent Stage 3 tests of determinism, random decks and `determinize` (SPEC 1.3, 2.1, 4).

States come from seeded random playouts of a pool whose effects move cards between hidden and
public zones (draw, discard, add_card, return_to_hand, random damage), with the mulligan on and a
mix of fixed and random decks (`sample_deal`).
"""
from __future__ import annotations

import random
from collections import Counter

import pytest

import cardgame.cards as cards_mod
from stage3_helpers import (CHOICE, CONFIRM, MULLIGAN, Game, Rules, blank, choose, counts, effect, find, hand_slot,
                            mull, multiset, perturb_hidden, pick, play, put, random_playout, rich_rules, set_deck,
                            state_key, tgt, unit)

RULES = rich_rules(mulligan=True)
CFG = RULES.config
N = RULES.n_cards
TOKENS = {i for i, c in enumerate(RULES.card_dicts) if c.get("token")}
OPERATIONS = {i for i, c in enumerate(RULES.card_dicts) if c["type"] == "operation"}


def new_game(seed, random_frac=0.5):
    g = Game(CFG)
    g.reset(seed, cards_mod.sample_deal(seed, CFG, random_frac))
    return g


def playout_states(seeds, every=5, max_steps=500):
    """Clones of every `every`-th state of seeded random games (odd seeds hoard cards, so hands fill
    up and burn)."""
    out = []
    for seed in seeds:
        g = new_game(seed)
        pol = random.Random(seed * 7919 + 1)
        for i in range(max_steps):
            if g.done:
                break
            if i % every == 0:
                out.append(g.clone())
            g.step(pick(g, pol, hoard=seed % 2 == 1))
    return out


@pytest.fixture(scope="module")
def states():
    return playout_states(range(6))


def unknown_count(g, p):
    """Cards of p's deck the opponent has not seen: decklist minus revealed (SPEC 4 determinize 2)."""
    return CFG.deck_size - sum(counts(g.revealed[p], N))


def hidden_part(g, o):
    return (tuple(g.decklists[o]), tuple(g.hands[o]), tuple(g.deck_cards[o]))


# ================================================================= random decks (SPEC 1.3)
def check_legal_deck(deck, required=None):
    deck = list(deck)
    assert len(deck) == CFG.deck_size and deck == sorted(deck)
    c = Counter(deck)
    assert max(c.values()) <= CFG.max_copies
    assert not (set(c) & TOKENS)
    assert sum(c[i] for i in OPERATIONS) <= 12
    if required is not None:
        req = counts(required, N)
        assert all(c[i] >= req[i] for i in range(N))


def test_generate_deck_is_legal_and_deterministic():
    # SPEC 1.3: a legal deck (40 cards, <= 3 copies, no tokens), sorted, operations capped at 12,
    # deterministic for a given RNG state; deck_rng(seed, seat) = Random(f"deck:{seed}:{seat}").
    for seed in range(20):
        d1 = cards_mod.generate_deck(cards_mod.deck_rng(seed, 0), CFG)
        d2 = cards_mod.generate_deck(random.Random(f"deck:{seed}:0"), CFG)
        assert tuple(d1) == tuple(d2)
        check_legal_deck(d1)


def test_generate_deck_places_required_cards():
    # SPEC 1.3: `required` (counts per card index) is placed first.
    req = [0] * N
    req[RULES.idx("bouncer")] = 3
    req[RULES.idx("intel")] = 2
    req[RULES.idx("fill07")] = 1
    for seed in range(10):
        check_legal_deck(cards_mod.generate_deck(random.Random(seed), CFG, required=req), req)


def test_sample_deal_and_tuple_decks():
    # SPEC 1.3 deals; SPEC 2.1: deck_ids[p] = fixed index or -1, decklists[p] = sorted 40-card tuple.
    kinds = set()
    for seed in range(30):
        deal = cards_mod.sample_deal(seed, CFG, 0.5)
        assert deal == cards_mod.sample_deal(seed, CFG, 0.5)
        g = Game(CFG)
        g.reset(seed, deal)
        for p in (0, 1):
            spec = deal[p]
            if isinstance(spec, int):
                kinds.add("fixed")
                assert g.deck_ids[p] == spec
                assert tuple(g.decklists[p]) == tuple(CFG.decks[spec])
            else:
                kinds.add("random")
                assert g.deck_ids[p] == -1
                assert tuple(g.decklists[p]) == tuple(sorted(spec))
                check_legal_deck(spec)
    assert kinds == {"fixed", "random"}
    assert all(isinstance(s, int) for s in cards_mod.sample_deal(3, CFG, 0.0))
    assert all(not isinstance(s, int) for s in cards_mod.sample_deal(3, CFG, 1.0))


@pytest.mark.parametrize("frac", [0.0, 0.3, 0.7, 1.0])
def test_sample_deal_draw_order(frac):
    # SPEC 1.3 + 2.13: random.Random(f"deal:{seed}") draws random() then randrange(n_decks) for seat 0,
    # then the same for seat 1, always both; a seat is random with probability random_frac.
    for seed in range(25):
        ref = random.Random(f"deal:{seed}")
        expected = []
        for seat in (0, 1):
            u = ref.random()
            k = ref.randrange(CFG.n_decks)
            expected.append(tuple(cards_mod.generate_deck(cards_mod.deck_rng(seed, seat), CFG)) if u < frac else k)
        deal = cards_mod.sample_deal(seed, CFG, frac)
        got = [s if isinstance(s, int) else tuple(s) for s in deal]
        assert got == expected, seed


def test_a_deck_spec_may_be_any_non_string_sequence():
    # SPEC 2.13: a deck spec may be any non-string sequence of 40 card indices.
    good = cards_mod.generate_deck(random.Random(4), CFG)
    a, b = Game(CFG), Game(CFG)
    a.reset(3, (tuple(good), 1))
    b.reset(3, (list(good), 1))
    assert state_key(a) == state_key(b) and a.deck_ids[0] == -1


def test_reset_rejects_illegal_tuple_decks():
    # SPEC 2.1.2: a tuple deck must be legal (40 cards, <= 3 copies, no tokens).
    good = tuple(cards_mod.generate_deck(random.Random(1), CFG))
    bad = {
        "short": good[:-1],
        "five_copies": tuple(sorted(good[:-5] + (good[0],) * 5)),
        "token": tuple(sorted(good[:-1] + (RULES.idx("militia"),))),
    }
    for deck in bad.values():
        with pytest.raises((ValueError, TypeError)):
            Game(CFG).reset(0, (deck, good))


# ================================================================= determinism (SPEC 4)
@pytest.mark.parametrize("seed", range(5))
def test_same_seed_decks_and_actions_give_identical_states(seed):
    # SPEC 4 / 2.11: same seed, decks and actions => identical states, including random effects.
    g1, g2 = new_game(seed), new_game(seed)
    pol = random.Random(seed)
    steps = 0
    while not g1.done and steps < 800:
        assert state_key(g1) == state_key(g2)
        a = pol.choice(g1.legal_actions())
        g1.step(a)
        g2.step(a)
        steps += 1
    assert state_key(g1) == state_key(g2)


def test_clone_is_independent_with_a_pending_choice_and_queued_effects():
    # SPEC 4: clone() copies the queue and the pending choice; the copies evolve independently.
    r = Rules([unit("triple", 1, 3, effects=[effect("on_deploy", "damage", tgt("chosen", "enemy"), amount=1),
                                             effect("on_deploy", "draw", "controller", amount=1),
                                             effect("on_deploy", "damage", tgt("random", "enemy"), amount=1)])])
    g = blank(r)
    for _ in range(3):
        put(g, r, 1, "back", "fill07")
    set_deck(g, r, 0, ["fill05", "fill06"])
    g.hands[0] = [r.idx("triple")]
    g.invalidate()
    g.step(play(0))
    assert g.phase == CHOICE and len(g.queue) >= 2
    key = state_key(g)
    c = g.clone()
    assert state_key(c) == key
    c.step(choose(1))
    assert c.phase != CHOICE and len(c.queue) == 0
    assert state_key(g) == key  # the original still waits with its queue
    d = g.clone()
    d.step(choose(2))
    g.step(choose(1))
    assert state_key(g) == state_key(c)
    assert state_key(d) != state_key(c)


def test_clones_replay_identically_from_mid_game(states):
    for g in states[::15]:
        a, b = g.clone(), g.clone()
        pol = random.Random(5)
        for _ in range(60):
            if a.done:
                break
            act = pol.choice(a.legal_actions())
            a.step(act)
            b.step(act)
            assert state_key(a) == state_key(b)


# ================================================================= determinize (SPEC 4)
def test_determinize_keeps_the_players_view_and_legal_actions(states):
    # SPEC 4 guarantee: observe(player) and (on player's decision) the legal actions are identical.
    checked = 0
    for i, g in enumerate(states):
        if g.done:
            continue
        before = state_key(g)
        p = g.current_player()
        for player in (p, 1 - p):
            det = g.determinize(player, random.Random(i))
            assert det.observe(player) == g.observe(player)
            if player == p:
                assert det.legal_actions() == g.legal_actions()
                assert (det.legal_mask() == g.legal_mask()).all()
                assert det.current_player() == p and det.phase == g.phase
        assert state_key(g) == before  # the original is untouched
        checked += 1
    assert checked > 100


@pytest.mark.parametrize("new_decklist", [False, True])
def test_determinize_depends_only_on_the_players_information(new_decklist, states):
    # SPEC 4 guarantee: perturbing o's hidden hand/deck/RNG gives an identical determinization for
    # the same rng state.
    rng = random.Random(99 + new_decklist)
    for i, g in enumerate(states[::2]):
        if g.done:
            continue
        p = g.current_player()
        h = perturb_hidden(g, p, rng, new_decklist=new_decklist)
        a = g.determinize(p, random.Random(1000 + i))
        b = h.determinize(p, random.Random(1000 + i))
        assert state_key(a) == state_key(b)


def test_determinizing_a_determinization_gives_the_same_result(states):
    # A determinization is an equally valid world for `player`, so the guarantee applies to it.
    for i, g in enumerate(states[::3]):
        if g.done:
            continue
        p = g.current_player()
        world = g.determinize(p, random.Random(7 + i))
        assert state_key(world.determinize(p, random.Random(i))) == state_key(g.determinize(p, random.Random(i)))


def test_determinized_decklist_and_hidden_cards_are_consistent(states):
    # SPEC 4 determinize 1-2 and guarantees: the decklist is legal and contains revealed[o]; the
    # unknown cards (decklist minus revealed) are dealt to the unknown part of o's hand (its size minus
    # the known_hand cards, which stay) and to o's deck (same sizes); burned cards are counts only.
    for i, g in enumerate(states):
        if g.done:
            continue
        p = g.current_player()
        o = 1 - p
        det = g.determinize(p, random.Random(i))
        check_legal_deck(det.decklists[o], g.revealed[o])
        assert det.deck_ids[o] == -1
        assert counts(det.revealed[o], N) == counts(g.revealed[o], N)
        assert counts(det.known_hand[o], N) == counts(g.known_hand[o], N)
        assert det.burned[o] == g.burned[o]
        assert len(det.hands[o]) == len(g.hands[o]) and len(det.deck_cards[o]) == len(g.deck_cards[o])
        assert list(det.hands[o]) == sorted(det.hands[o])
        known = Counter({c: k for c, k in enumerate(counts(g.known_hand[o], N)) if k})
        hand = multiset(det.hands[o])
        assert not (known - hand), "known_hand cards must stay in the hand"
        dealt = (hand - known) + multiset(det.deck_cards[o])
        pool = multiset(det.decklists[o]) - Counter({c: k for c, k in enumerate(counts(g.revealed[o], N)) if k})
        assert not (dealt - pool), "dealt cards must come from decklist minus revealed"
        assert not (set(dealt) & TOKENS)
        leftover_real = unknown_count(g, o) - (len(g.hands[o]) - sum(known.values())) - len(g.deck_cards[o])
        assert sum(pool.values()) - sum(dealt.values()) == leftover_real


def test_determinize_keeps_own_cards_and_replaces_the_rng(states):
    # SPEC 4 determinize 3 (own deck keeps its contents, reshuffled) and 5 (the RNG is replaced).
    for i, g in enumerate(states):
        if g.done:
            continue
        p = g.current_player()
        det = g.determinize(p, random.Random(i))
        assert list(det.hands[p]) == list(g.hands[p])
        assert multiset(det.deck_cards[p]) == multiset(g.deck_cards[p])
        assert tuple(det.decklists[p]) == tuple(g.decklists[p]) and det.deck_ids[p] == g.deck_ids[p]
        assert counts(det.known_hand[p], N) == counts(g.known_hand[p], N)
        assert det.rng.getstate() != g.rng.getstate()
        assert det.base_hp == g.base_hp and det.coins == g.coins and det.turn == g.turn


def test_determinize_does_not_depend_on_the_true_order_of_the_own_deck(states):
    # SPEC 2.13: the own deck is sorted before shuffling, so the result cannot depend on the true order
    # (which is hidden from the player too, SPEC 5); SPEC 2.13: seed is None on the result.
    rng = random.Random(8)
    checked = 0
    for i, g in enumerate(states[::3]):
        if g.done or len(g.deck_cards[g.current_player()]) < 3:
            continue
        p = g.current_player()
        h = g.clone()
        deck = list(h.deck_cards[p])
        rng.shuffle(deck)
        h.deck_cards[p] = deck
        h.invalidate()
        a = g.determinize(p, random.Random(500 + i))
        b = h.determinize(p, random.Random(500 + i))
        assert state_key(a) == state_key(b)
        assert a.seed is None and b.seed is None
        checked += 1
    assert checked > 10


def test_different_rng_states_give_different_hidden_parts(states):
    # SPEC 4 guarantee: different rng states give different hidden parts.
    checked = 0
    for g in states:
        if g.done:
            continue
        p = g.current_player()
        o = 1 - p
        unknown = len(g.hands[o]) - sum(counts(g.known_hand[o], N)) + len(g.deck_cards[o])
        if unknown < 8:
            continue
        a = g.determinize(p, random.Random(1))
        b = g.determinize(p, random.Random(2))
        assert hidden_part(a, o) != hidden_part(b, o)
        assert a.rng.getstate() != b.rng.getstate()
        checked += 1
    assert checked > 20


def test_determinize_during_the_mulligan():
    # SPEC 4 determinize 4: o's mulligan marks are cleared; the player's own marks are their info.
    for seed in range(4):
        g = new_game(seed)
        f, s = g.first_player, 1 - g.first_player
        g.step(mull(0))
        g.step(mull(2))
        det = g.determinize(s, random.Random(seed))  # the opponent (f) is deciding
        assert len(det.mulligan_marks) == 0
        assert det.observe(s) == g.observe(s)
        det = g.determinize(f, random.Random(seed))  # f's own decision
        assert set(det.mulligan_marks) == {0, 2}
        assert det.legal_actions() == g.legal_actions() and det.observe(f) == g.observe(f)
        assert det.phase == MULLIGAN
        for d in (g.determinize(f, random.Random(3)), g.determinize(s, random.Random(3))):
            random_playout(d, random.Random(seed), max_steps=3000)  # the result is a playable game


def test_determinize_at_a_pending_choice():
    found = 0
    for seed in range(40):
        g = new_game(seed)
        pol = random.Random(seed)
        for _ in range(400):
            if g.done:
                break
            if g.phase == CHOICE:
                p = g.current_player()
                det = g.determinize(p, random.Random(seed))
                assert det.phase == CHOICE and det.legal_actions() == g.legal_actions()
                assert det.observe(p) == g.observe(p)
                a = pol.choice(g.legal_actions())
                det.step(a)  # the pending effect resumes in the determinized world too
                found += 1
                break
            g.step(pol.choice(g.legal_actions()))
        if found >= 3:
            break
    assert found >= 3


def test_determinized_games_play_to_the_end(states):
    for i, g in enumerate(states[::4]):
        if g.done:
            continue
        det = g.determinize(g.current_player(), random.Random(i))
        random_playout(det, random.Random(i), max_steps=5000)
        assert det.done
