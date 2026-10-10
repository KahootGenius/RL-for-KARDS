"""Stage 2 rule scenarios (SPEC §2.11: natures, Defense, armor, move cost, frontline control, plus setup,
turns, winning and compaction), built by editing the documented engine state (SPEC §4). Every step is
also checked against the independent Stage 2 reference model, so the games run on the frozen vanilla
content (`VANILLA_CONFIG`, mulligan off); units use explicit stats so retuning cards cannot break a test."""
from __future__ import annotations

import dataclasses
import functools
from collections import Counter
from itertools import product

import pytest

import conftest
from cardgame.cards import FAST, RANGED, TROOP, sample_decks
from cardgame.engine import DRAW, IllegalActionError, Unit
from conftest import (BASE, END, NUM_ACTIONS, H, Z, add_unit, attack, back, check_invariants, front, move, play,
                      set_deck, set_hand)
from conftest import VANILLA_CONFIG as CONFIG
from reference_rules import (can_attack as ref_can_attack, can_move as ref_can_move, comparable,
                             extract_state, ref_unit, reference_legal, reference_reset, reference_step)
from reference_rules import sample_decks as ref_sample_decks

NATURES = (TROOP, FAST, RANGED)
NATURE_IDS = ("troop", "fast", "ranged")
N_CARDS = len(CONFIG.cards)
N_DECKS = CONFIG.n_decks
DECK_PAIRS = tuple(product(range(N_DECKS), repeat=2))
blank_game = functools.partial(conftest.blank_game, config=CONFIG)
new_game = functools.partial(conftest.new_game, config=CONFIG)
cards_where = functools.partial(conftest.cards_where, config=CONFIG)


def cost(card_index: int) -> int:
    return CONFIG.cards[card_index].cost


def assert_legal(game, expected) -> None:
    """Engine and reference agree, and equal the hand-written expectation."""
    legal = game.legal_actions()
    assert legal == reference_legal(game), game.render()
    assert legal == sorted(set(expected)), [game.describe(a) for a in legal]


def do(game, action: int) -> None:
    """Step `action` (it must be offered) and check the result against the reference transition."""
    assert action in game.legal_actions(), f"{game.describe(action)} not offered\n{game.render()}"
    before = extract_state(game)
    game.step(action)
    expected = reference_step(before, action, game.config)
    assert comparable(extract_state(game)) == comparable(expected), (
        f"{game.describe(action)}\nengine:   {extract_state(game)}\nexpected: {expected}")
    check_invariants(game, reachable=False)
    assert game.legal_actions() == reference_legal(game)


def unit_actions(game, slot: int) -> list:
    """Legal MOVE/ATTACK actions of the unit in attacker slot `slot`."""
    sp = game.action_space
    out = []
    for a in game.legal_actions():
        act = sp.decode(a)
        if (act.kind.name == "MOVE" and slot < Z and act.a == slot) or (act.kind.name == "ATTACK" and act.a == slot):
            out.append(a)
    return out


def moves(game) -> list:
    """Legal MOVE actions."""
    return [a for a in game.legal_actions() if game.action_space.MOVE0 <= a < game.action_space.ATTACK0]


def cheapest(nature: int, **kw) -> int:
    return min(cards_where(nature, **kw), key=lambda i: (cost(i), i))


def hp_of(units) -> list:
    return [u.hp for u in units]


# ---------------------------------------------------------------- setup
@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42, 1234, 99991, 2**31 - 1])
def test_reset_follows_spec_setup_exactly(seed):
    expected, rng_state = reference_reset(seed, CONFIG)
    g = new_game(seed)
    assert extract_state(g) == expected
    assert g.rng.getstate() == rng_state
    assert g.current_player() == g.first_player and not g.done and g.winner() is None


@pytest.mark.parametrize("decks", DECK_PAIRS)
def test_reset_with_explicit_decks_follows_spec(decks):
    for seed in (3, 4):
        expected, rng_state = reference_reset(seed, CONFIG, decks)
        g = new_game(seed, decks)
        assert extract_state(g) == expected and g.rng.getstate() == rng_state
        assert tuple(g.deck_ids) == decks
        for p in (0, 1):  # each seat's cards come from its own deck
            assert Counter(g.hands[p] + g.deck_cards[p]) == Counter(CONFIG.decks[decks[p]])


def test_default_decks_are_sampled_from_their_own_stream():
    for seed in range(200):
        pair = sample_decks(seed, N_DECKS)
        assert pair == ref_sample_decks(seed, N_DECKS)
        g = new_game(seed)
        assert tuple(g.deck_ids) == pair
        assert extract_state(g) == extract_state(new_game(seed, pair))


def test_deck_pairs_are_uniform_and_include_mirrors():
    counts = Counter(sample_decks(seed, N_DECKS) for seed in range(3200))
    assert set(counts) == set(DECK_PAIRS)  # every ordered pair, mirrors included
    expected = 3200 / len(DECK_PAIRS)
    assert all(0.6 * expected <= n <= 1.4 * expected for n in counts.values()), counts


def test_opening_hand_sizes_and_first_draws():
    for seed in range(40):
        g = new_game(seed)
        f, s = g.first_player, 1 - g.first_player
        first_n, second_n = CONFIG.opening_hand
        assert (first_n, second_n) == (4, 5)
        assert (len(g.hands[f]), len(g.deck_cards[f])) == (first_n + 1, 40 - first_n - 1)  # + turn-1 draw
        assert (len(g.hands[s]), len(g.deck_cards[s])) == (second_n, 40 - second_n)
        assert g.coins[f] == 1 and g.coins[s] == 0 and g.round == 1
        do(g, END)
        assert g.current_player() == s and g.round == 1
        assert g.coins[s] == 1 and g.coins[f] == 0
        assert (len(g.hands[s]), len(g.deck_cards[s])) == (second_n + 1, 40 - second_n - 1)


def test_coin_flip_varies_across_seeds():
    firsts = [new_game(seed).first_player for seed in range(400)]
    assert 140 <= sum(firsts) <= 260  # ~Binomial(400, 0.5)
    for seed in range(0, 400, 37):
        assert new_game(seed).current_player() == firsts[seed]


# ---------------------------------------------------------------- turn structure
@pytest.mark.parametrize("seed", [3, 8])
def test_end_turn_only_game_is_a_draw_after_50_rounds(seed):
    g = new_game(seed)
    f = g.first_player
    for n in range(1, 101):
        p = g.current_player()
        o = 1 - p
        hand, deck, burned = list(g.hands[o]), list(g.deck_cards[o]), g.burned[o]
        rnd = g.round
        assert g.coins[p] == rnd and g.coins[o] == 0
        do(g, END)
        assert g.coins[p] == 0  # unused coins are lost
        if n == 100:
            break
        assert not g.done and g.current_player() == o
        assert g.round == (rnd + 1 if o == f else rnd)  # increments only when the first player starts
        assert g.coins[o] == g.round
        if not deck:  # deck-out: no draw, no penalty
            assert (g.hands[o], g.deck_cards[o], g.burned[o]) == (hand, [], burned)
        elif len(hand) == H:  # overdraw: the top card is burned
            assert (g.hands[o], g.deck_cards[o], g.burned[o]) == (hand, deck[:-1], burned + 1)
        else:
            assert g.hands[o] == sorted(hand + [deck[-1]]) and g.deck_cards[o] == deck[:-1]
        assert g.base_hp == [20, 20]
        check_invariants(g, (0, 0))
    assert g.done and g.winner() == DRAW and g.round == 50
    assert g.legal_actions() == [] and not g.legal_mask().any()
    for p in (0, 1):
        obs = g.observe(p)
        assert obs.done and obs.result == 0 and not obs.is_my_turn and obs.round == 50
    with pytest.raises(IllegalActionError):
        g.step(END)


def test_99_end_turns_leave_the_last_turn_of_round_50():
    g = new_game(5)
    for _ in range(99):
        g.step(END)
    assert not g.done and g.round == 50 and g.current_player() == 1 - g.first_player
    assert g.coins[g.current] == 50


def test_draw_ends_the_game_after_the_second_players_end_turn():
    g = blank_game(current=1, first=0, round_=50, coins=50)
    u = add_unit(g, 1, "front", atk=2, hp=2, moved=True, attacked=True)
    do(g, END)
    assert g.done and g.winner() == DRAW and g.round == 50
    assert g.coins[1] == 0 and not (u.moved or u.attacked)  # END_TURN effects still apply


def test_unused_coins_are_lost_and_not_carried_over():
    g = blank_game(current=0, first=0, round_=3, coins=3)
    set_hand(g, 0, [cheapest(TROOP)])
    do(g, play(0))
    assert g.coins[0] == 3 - cost(cheapest(TROOP))
    do(g, END)
    assert g.coins == [0, 3] and g.round == 3
    do(g, END)
    assert g.round == 4 and g.coins == [4, 0]


def test_coin_cap_is_applied():
    cfg = dataclasses.replace(CONFIG, coin_cap=3)
    g = new_game(0, config=cfg)
    for _ in range(12):
        before = extract_state(g)
        g.step(END)
        assert comparable(extract_state(g)) == comparable(reference_step(before, END, cfg))
        assert g.coins[g.current] == min(g.round, 3)
    assert g.round >= 6 and g.coins[g.current] == 3


def test_overdraw_burns_the_top_card():
    g = blank_game(current=0, first=0)
    hand = [k % N_CARDS for k in range(H)]
    set_hand(g, 1, hand)
    set_deck(g, 1, [0, 1])  # top = end
    assert len(g.hands[1]) == H
    do(g, END)
    assert g.hands[1] == sorted(hand) and g.deck_cards[1] == [0] and g.burned == [0, 1]
    assert g.coins[1] == 1 and not g.done


def test_draw_takes_the_top_card_and_keeps_the_hand_sorted():
    g = blank_game(current=0, first=0)
    hand = [(3 * k) % N_CARDS for k in range(H - 1)]
    set_hand(g, 1, hand)
    top = len(CONFIG.cards) // 2
    set_deck(g, 1, [0, top])
    do(g, END)
    assert g.hands[1] == sorted(hand + [top]) and len(g.hands[1]) == H
    assert g.deck_cards[1] == [0] and g.burned == [0, 0]


def test_deck_out_has_no_penalty():
    g = blank_game(current=0, first=0)
    set_hand(g, 1, [0])
    assert g.deck_cards[1] == []
    do(g, END)
    assert not g.done and g.current_player() == 1
    assert g.hands[1] == [0] and g.deck_cards[1] == [] and g.burned == [0, 0]
    assert g.base_hp == [20, 20] and g.coins[1] == 1


# ---------------------------------------------------------------- deploying
def test_play_deploys_a_summoned_unit_from_the_card():
    g = blank_game(current=0, first=0, round_=8, coins=8)
    cards = sorted(cards_where(pred=lambda c: c.cost <= 4), key=lambda i: (cost(i), i))[:2]
    set_hand(g, 0, cards)
    g.next_uid = 7
    do(g, play(1))
    c = CONFIG.cards[sorted(cards)[1]]
    u = g.backline[0][0]
    assert (u.card, u.owner, u.uid, u.atk, u.hp, u.max_hp, u.armor, u.defense, u.nature, u.move_cost) == (
        c.index, 0, 7, c.attack, c.health, c.health, c.armor, c.defense, c.nature, c.move_cost)
    assert (u.summoned, u.moved, u.attacked) == (True, False, False)
    assert g.coins[0] == 8 - c.cost and g.next_uid == 8 and g.played[0][c.index] == 1
    assert g.hands[0] == [sorted(cards)[0]]
    do(g, play(0))
    assert [x.uid for x in g.backline[0]] == [7, 8] and g.next_uid == 9  # appended, fresh uid


def test_play_respects_cost_and_backline_capacity():
    by_cost = sorted(range(len(CONFIG.cards)), key=lambda i: (cost(i), i))
    cheap, dear = by_cost[0], by_cost[-1]
    assert cost(cheap) < cost(dear)
    hand = sorted([cheap, dear])
    i_cheap = hand.index(cheap)
    g = blank_game(current=0, first=0, round_=20, coins=cost(dear) - 1)
    set_hand(g, 0, hand)
    assert_legal(g, [END, play(i_cheap)])  # the dear card is unaffordable
    g.coins[0] = cost(dear)
    g.invalidate()
    assert_legal(g, [END, play(0), play(1)])  # cost == coins is affordable
    g.coins[0] = cost(dear) + cost(cheap)
    for _ in range(Z - 1):
        add_unit(g, 0, "back", atk=1, hp=1, summoned=True)
    assert_legal(g, [END, play(0), play(1)])
    do(g, play(i_cheap))
    assert len(g.backline[0]) == Z and g.coins[0] == cost(dear)
    assert_legal(g, [END])  # affordable, but the backline is full


# ---------------------------------------------------------------- 2.1.1 deploy round and flag refresh
def test_action_economy_table():
    """can_move/can_attack depend only on (nature, flags), exactly as the SPEC table says."""
    for nature in NATURES:
        for summoned in (False, True):
            for moved in (False, True):
                for attacked in (False, True):
                    u = Unit(0, 0, atk=1, hp=1, nature=nature, summoned=summoned, moved=moved, attacked=attacked)
                    r = ref_unit(u)
                    assert u.can_move() == ref_can_move(r), (nature, summoned, moved, attacked)
                    assert u.can_attack() == ref_can_attack(r), (nature, summoned, moved, attacked)
                    v = u.view()
                    assert (v.summoned, v.moved, v.attacked) == (summoned, moved, attacked)
                    assert (v.can_move, v.can_attack) == (u.can_move(), u.can_attack())


@pytest.mark.parametrize("enemy_front", [False, True], ids=["front_empty", "enemy_front"])
@pytest.mark.parametrize("nature", NATURES, ids=NATURE_IDS)
def test_no_move_or_attack_on_the_deploy_round(nature, enemy_front):
    c = cheapest(nature)
    g = blank_game(current=0, first=0, round_=9, coins=cost(c) + 9)
    set_hand(g, 0, [c])
    add_unit(g, 1, "back", atk=1, hp=9)
    if enemy_front:
        add_unit(g, 1, "front", atk=1, hp=9)
    do(g, play(0))
    u = g.backline[0][0]
    assert u.summoned and not u.can_move() and not u.can_attack()
    assert unit_actions(g, back(0)) == []
    view = g.observe(0).my_backline[0]
    assert view.summoned and not view.can_move and not view.can_attack
    do(g, END)  # flags refresh at the owner's END_TURN: the opponent already sees a ready unit
    assert not u.summoned and u.can_move() and u.can_attack()
    view = g.observe(1).opp_backline[0]
    assert not view.summoned and view.can_move and view.can_attack
    do(g, END)
    expected = [attack(back(0), front(0))] if enemy_front else [move(0)]
    if nature == RANGED:
        expected += [attack(back(0), back(0)), attack(back(0), BASE)]
    assert unit_actions(g, back(0)) == sorted(expected)


def test_flags_refresh_at_the_owners_end_turn_only():
    g = blank_game(current=0, first=0, round_=5, coins=5)
    mine = [add_unit(g, 0, "back", atk=1, hp=5, summoned=True),
            add_unit(g, 0, "front", atk=1, hp=5, nature=FAST, moved=True, attacked=True),
            add_unit(g, 0, "front", atk=1, hp=5, moved=True)]
    theirs = add_unit(g, 1, "back", atk=1, hp=5, nature=RANGED, attacked=True)
    do(g, END)
    assert all(not (u.summoned or u.moved or u.attacked) for u in mine)
    # flags only (SPEC §2): a frontline unit shows can_move although it can never move again
    assert all(v.can_move and v.can_attack for v in g.observe(1).frontline)
    assert theirs.attacked  # untouched by the opponent's END_TURN
    do(g, END)
    assert not theirs.attacked


# ---------------------------------------------------------------- 2.1.2 troop
def test_troop_cannot_attack_after_moving():
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "front", atk=2, hp=5)               # control: a fresh frontline troop
    add_unit(g, 0, "back", atk=2, hp=5, move_cost=1)
    add_unit(g, 1, "back", atk=1, hp=9)
    assert unit_actions(g, front(0)) == [attack(front(0), back(0)), attack(front(0), BASE)]
    do(g, move(0))
    moved = g.frontline[1]
    assert moved.moved and not moved.can_attack() and not moved.can_move()
    assert unit_actions(g, front(1)) == []
    assert_legal(g, [END, attack(front(0), back(0)), attack(front(0), BASE)])


def test_troop_cannot_move_after_attacking():
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "back", atk=3, hp=5)
    add_unit(g, 0, "back", atk=1, hp=5)                # control: can move once the front is free
    add_unit(g, 1, "front", atk=1, hp=2)
    do(g, attack(back(0), front(0)))                   # kills the last enemy frontline unit
    assert g.frontline == [] and g.front_owner is None
    attacker = g.backline[0][0]
    assert attacker.attacked and not attacker.can_move()
    assert_legal(g, [END, move(1)])


# ---------------------------------------------------------------- 2.1.3 fast
@pytest.mark.parametrize("target", ["base", "backline"])
def test_fast_moves_then_attacks_from_the_frontline(target):
    g = blank_game(current=0, first=0, round_=5, coins=1)
    add_unit(g, 0, "back", atk=3, hp=3, nature=FAST, move_cost=1)
    add_unit(g, 1, "back", atk=1, hp=9)
    assert_legal(g, [END, move(0)])                    # nothing in reach from the backline
    do(g, move(0))
    assert g.coins[0] == 0 and g.front_owner == 0
    u = g.frontline[0]
    assert u.moved and u.can_attack() and not u.can_move()
    assert_legal(g, [END, attack(front(0), back(0)), attack(front(0), BASE)])
    if target == "base":
        do(g, attack(front(0), BASE))
        assert g.base_hp[1] == 20 - 3
    else:
        do(g, attack(front(0), back(0)))
        assert hp_of(g.backline[1]) == [6] and u.hp == 2
    assert u.moved and u.attacked and not u.can_move() and not u.can_attack()
    assert_legal(g, [END])                             # never a second MOVE or ATTACK


def test_fast_attack_that_clears_the_frontline_then_moves():
    g = blank_game(current=0, first=0, round_=5, coins=2)
    add_unit(g, 0, "back", atk=3, hp=3, nature=FAST, move_cost=2)
    add_unit(g, 1, "front", atk=1, hp=3)
    add_unit(g, 1, "back", atk=1, hp=9)
    assert_legal(g, [END, attack(back(0), front(0))])
    do(g, attack(back(0), front(0)))
    assert g.frontline == [] and g.front_owner is None
    u = g.backline[0][0]
    assert u.attacked and u.can_move() and not u.can_attack()
    assert_legal(g, [END, move(0)])
    do(g, move(0))
    assert g.coins[0] == 0 and g.frontline == [u] and g.front_owner == 0
    assert_legal(g, [END])                             # the enemy backline/base stay out of reach


def test_fast_attack_on_a_surviving_frontline_unit_blocks_the_move():
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "back", atk=3, hp=3, nature=FAST)
    add_unit(g, 1, "front", atk=1, hp=4)
    do(g, attack(back(0), front(0)))
    assert hp_of(g.frontline) == [1] and g.front_owner == 1
    u = g.backline[0][0]
    assert u.can_move() and not u.can_attack()         # the flag allows it, the board does not
    assert_legal(g, [END])


def test_fast_unit_with_both_actions_used_is_spent():
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "back", atk=3, hp=9, nature=FAST, attacked=True, moved=False)
    add_unit(g, 0, "front", atk=3, hp=9, nature=FAST, attacked=True, moved=True)
    add_unit(g, 1, "back", atk=1, hp=9)
    assert_legal(g, [END, move(0)])
    do(g, move(0))
    assert_legal(g, [END])


# ---------------------------------------------------------------- 2.1.4 ranged
def test_ranged_reach_from_the_backline():
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "back", atk=2, hp=2, nature=RANGED)
    add_unit(g, 1, "back", atk=1, hp=9)
    add_unit(g, 1, "back", atk=1, hp=9)
    add_unit(g, 1, "front", atk=1, hp=9)
    assert_legal(g, [END, attack(back(0), back(0)), attack(back(0), back(1)), attack(back(0), front(0)),
                     attack(back(0), BASE)])


def test_ranged_reach_from_the_frontline():
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "front", atk=2, hp=2, nature=RANGED)
    add_unit(g, 1, "back", atk=1, hp=9)
    add_unit(g, 1, "back", atk=1, hp=9)
    assert_legal(g, [END, attack(front(0), back(0)), attack(front(0), back(1)), attack(front(0), BASE)])


@pytest.mark.parametrize("where", ["back_to_back", "back_to_front", "front_to_back"])
def test_ranged_takes_no_return_damage(where):
    g = blank_game(current=0, first=0, round_=5, coins=5)
    a_zone, t_zone = where.split("_to_")
    add_unit(g, 0, a_zone, atk=2, hp=1, nature=RANGED)
    add_unit(g, 1, t_zone, atk=9, hp=5)
    a_slot = back(0) if a_zone == "back" else front(0)
    t_slot = back(0) if t_zone == "back" else front(0)
    shooter = (g.backline[0] if a_zone == "back" else g.frontline)[0]
    do(g, attack(a_slot, t_slot))
    target = (g.backline[1] if t_zone == "back" else g.frontline)[0]
    assert target.hp == 3 and shooter.hp == 1 and shooter.attacked
    assert shooter in (*g.backline[0], *g.frontline)


def test_ranged_kill_from_the_backline_finishes_a_backline_unit():
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "back", atk=3, hp=1, nature=RANGED)
    add_unit(g, 1, "back", atk=9, hp=3)
    do(g, attack(back(0), back(0)))
    assert g.backline[1] == [] and hp_of(g.backline[0]) == [1]


def test_ranged_hits_back_when_attacked():
    g = blank_game(current=1, first=0, round_=5, coins=5)
    add_unit(g, 0, "back", atk=3, hp=4, nature=RANGED)
    add_unit(g, 1, "front", atk=2, hp=5, armor=1)
    do(g, attack(front(0), back(0)))
    assert hp_of(g.backline[0]) == [2] and hp_of(g.frontline) == [3]  # 3 - armor 1


def test_ranged_cannot_attack_after_moving_or_move_after_attacking():
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "back", atk=2, hp=2, nature=RANGED)
    add_unit(g, 1, "back", atk=1, hp=9)
    do(g, move(0))
    assert not g.frontline[0].can_attack()
    assert_legal(g, [END])

    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "back", atk=5, hp=2, nature=RANGED)
    add_unit(g, 1, "front", atk=1, hp=2)
    do(g, attack(back(0), front(0)))
    assert g.front_owner is None and not g.backline[0][0].can_move()
    assert_legal(g, [END])


# ---------------------------------------------------------------- 2.1.5 Defense
@pytest.mark.parametrize("nature", NATURES, ids=NATURE_IDS)
def test_defense_in_the_enemy_backline_vs_a_frontline_attacker(nature):
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "front", atk=2, hp=9, nature=nature)
    add_unit(g, 1, "back", atk=1, hp=9)
    add_unit(g, 1, "back", atk=1, hp=9, defense=True)
    add_unit(g, 1, "back", atk=1, hp=9)
    assert_legal(g, [END, attack(front(0), back(1)), attack(front(0), BASE)])  # the base is never protected


def test_defense_in_the_enemy_backline_vs_ranged_in_the_backline():
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "back", atk=2, hp=9, nature=RANGED)
    add_unit(g, 1, "back", atk=1, hp=9)
    add_unit(g, 1, "back", atk=1, hp=9, defense=True)
    add_unit(g, 1, "front", atk=1, hp=9)               # Defense in the other zone does not restrict
    assert_legal(g, [END, attack(back(0), back(1)), attack(back(0), front(0)), attack(back(0), BASE)])


@pytest.mark.parametrize("nature", NATURES, ids=NATURE_IDS)
def test_defense_in_the_enemy_frontline_vs_a_backline_attacker(nature):
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "back", atk=2, hp=9, nature=nature)
    add_unit(g, 1, "front", atk=1, hp=9)
    add_unit(g, 1, "front", atk=1, hp=9, defense=True)
    add_unit(g, 1, "back", atk=1, hp=9)                # unprotected: no Defense in its own zone
    expected = [END, attack(back(0), front(1))]
    if nature == RANGED:
        expected += [attack(back(0), back(0)), attack(back(0), BASE)]
    assert_legal(g, expected)


def test_two_defense_units_may_either_be_targeted():
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "back", atk=2, hp=9)
    add_unit(g, 1, "front", atk=1, hp=9, defense=True)
    add_unit(g, 1, "front", atk=1, hp=9)
    add_unit(g, 1, "front", atk=1, hp=9, defense=True, armor=1)
    assert_legal(g, [END, attack(back(0), front(0)), attack(back(0), front(2))])


def test_defense_in_the_other_zone_does_not_restrict():
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "back", atk=2, hp=9)
    add_unit(g, 0, "back", atk=2, hp=9, nature=RANGED)
    add_unit(g, 1, "back", atk=1, hp=9, defense=True)
    add_unit(g, 1, "back", atk=1, hp=9)
    add_unit(g, 1, "front", atk=1, hp=9)
    add_unit(g, 1, "front", atk=1, hp=9)
    assert_legal(g, [END, attack(back(0), front(0)), attack(back(0), front(1)),
                     attack(back(1), back(0)), attack(back(1), front(0)), attack(back(1), front(1)),
                     attack(back(1), BASE)])


@pytest.mark.parametrize("nature", [TROOP, RANGED], ids=["troop", "ranged"])
def test_killing_the_last_defense_unit_lifts_the_restriction_in_the_same_turn(nature):
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "front", atk=4, hp=9, nature=nature)
    add_unit(g, 0, "front", atk=4, hp=9, nature=nature)
    add_unit(g, 1, "back", atk=1, hp=9)
    add_unit(g, 1, "back", atk=1, hp=3, defense=True)
    assert unit_actions(g, front(1)) == [attack(front(1), back(1)), attack(front(1), BASE)]
    do(g, attack(front(0), back(1)))
    assert len(g.backline[1]) == 1 and not g.backline[1][0].defense
    assert unit_actions(g, front(1)) == [attack(front(1), back(0)), attack(front(1), BASE)]


def test_a_lone_defense_unit_and_ranged_attackers_follow_the_same_rule():
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "back", atk=2, hp=9, nature=RANGED)
    add_unit(g, 1, "back", atk=1, hp=9, defense=True)
    add_unit(g, 1, "front", atk=1, hp=9, defense=True)
    add_unit(g, 1, "front", atk=1, hp=9)
    assert_legal(g, [END, attack(back(0), back(0)), attack(back(0), front(0)), attack(back(0), BASE)])


# ---------------------------------------------------------------- 2.1.6 armor
def test_armor_reduces_damage_to_the_target_and_return_damage():
    g = blank_game(current=0, first=0, round_=5, coins=5)
    att = add_unit(g, 0, "back", atk=4, hp=5, armor=1)
    tgt = add_unit(g, 1, "front", atk=3, hp=6, armor=2)
    do(g, attack(back(0), front(0)))
    assert (tgt.hp, att.hp) == (6 - (4 - 2), 5 - (3 - 1))


@pytest.mark.parametrize("armor", [3, 4])
def test_armor_at_least_attack_means_zero_damage_but_the_attack_is_legal(armor):
    g = blank_game(current=0, first=0, round_=5, coins=5)
    att = add_unit(g, 0, "back", atk=3, hp=5)
    tgt = add_unit(g, 1, "front", atk=2, hp=4, armor=armor)
    assert_legal(g, [END, attack(back(0), front(0))])
    do(g, attack(back(0), front(0)))
    assert tgt.hp == 4 and att.hp == 3 and att.attacked  # uses the action and still takes return damage
    assert_legal(g, [END])


def test_zero_attack_unit_may_attack_and_armor_can_block_return_damage():
    g = blank_game(current=0, first=0, round_=5, coins=5)
    att = add_unit(g, 0, "back", atk=0, hp=2, armor=2)
    tgt = add_unit(g, 1, "front", atk=2, hp=4)
    do(g, attack(back(0), front(0)))
    assert (tgt.hp, att.hp, att.attacked) == (4, 2, True)


@pytest.mark.parametrize("nature", NATURES, ids=NATURE_IDS)
def test_base_damage_ignores_armor(nature):
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "front", atk=3, hp=2, armor=2, nature=nature)
    add_unit(g, 1, "back", atk=9, hp=9, armor=3, defense=True)
    do(g, attack(front(0), BASE))
    assert g.base_hp == [20, 17] and hp_of(g.frontline) == [2]  # the base does not hit back


# ---------------------------------------------------------------- 2.1.7 move cost
@pytest.mark.parametrize("nature", NATURES, ids=NATURE_IDS)
def test_move_cost_above_coins_is_illegal_and_equal_coins_pays_everything(nature):
    g = blank_game(current=0, first=0, round_=5, coins=1)
    add_unit(g, 0, "back", atk=1, hp=1, nature=nature, move_cost=2)
    assert moves(g) == [] and g.backline[0][0].can_move()
    g.coins[0] = 2
    g.invalidate()
    assert moves(g) == [move(0)]
    do(g, move(0))
    assert g.coins[0] == 0 and g.front_owner == 0


@pytest.mark.parametrize("nature", NATURES, ids=NATURE_IDS)
def test_move_cost_zero_is_legal_with_zero_coins(nature):
    g = blank_game(current=0, first=0, round_=5, coins=0)
    add_unit(g, 0, "back", atk=1, hp=1, nature=nature, move_cost=0)
    add_unit(g, 0, "back", atk=1, hp=1, nature=nature, move_cost=1)
    assert moves(g) == [move(0)]
    do(g, move(0))
    assert g.coins[0] == 0


def test_fast_units_pay_their_move_cost():
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "back", atk=1, hp=1, nature=FAST, move_cost=3)
    add_unit(g, 0, "back", atk=1, hp=1, nature=FAST, move_cost=3)
    do(g, move(0))
    assert g.coins[0] == 2
    assert_legal(g, [END, attack(front(0), BASE)])     # 2 coins left, the second fast unit needs 3


def test_a_shipped_card_with_move_cost_zero_moves_for_free():
    free = cards_where(pred=lambda c: c.move_cost == 0)
    assert free
    c = CONFIG.cards[free[0]]
    g = blank_game(current=0, first=0, round_=8, coins=c.cost)
    set_hand(g, 0, [c.index])
    do(g, play(0))
    do(g, END)
    do(g, END)
    g.coins[0] = 0
    g.invalidate()
    assert move(0) in g.legal_actions()
    do(g, move(0))
    assert g.frontline[0].card == c.index and g.coins[0] == 0


# ---------------------------------------------------------------- 2.1.8 frontline control
@pytest.mark.parametrize("nature", NATURES, ids=NATURE_IDS)
def test_enemy_held_frontline_blocks_every_move(nature):
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "back", atk=1, hp=9, nature=nature, move_cost=0)
    add_unit(g, 1, "front", atk=1, hp=9, defense=True)
    assert move(0) not in g.legal_actions()
    assert_legal(g, [END, attack(back(0), front(0))] + ([attack(back(0), BASE)] if nature == RANGED else []))


def test_full_own_frontline_blocks_moves():
    g = blank_game(current=0, first=0, round_=5, coins=5)
    for _ in range(Z - 1):
        add_unit(g, 0, "front", atk=1, hp=1, moved=True)
    add_unit(g, 0, "back", atk=1, hp=1)
    add_unit(g, 0, "back", atk=1, hp=1, nature=RANGED, move_cost=0)
    assert moves(g) == [move(0), move(1)]
    do(g, move(0))
    assert len(g.frontline) == Z
    assert g.backline[0][0].can_move() and g.coins[0] >= g.backline[0][0].move_cost
    assert moves(g) == []


def frontline_scenario(case: str):
    """A position where the current player's attack empties the frontline; returns (game, action)."""
    g = blank_game(current=0, first=0, round_=5, coins=5)
    if case == "enemy_killed":           # the last enemy frontline unit dies, the attacker survives
        add_unit(g, 0, "back", atk=5, hp=5)
        add_unit(g, 1, "front", atk=1, hp=2)
        a = attack(back(0), front(0))
    elif case == "ranged_kill":
        add_unit(g, 0, "back", atk=5, hp=1, nature=RANGED)
        add_unit(g, 1, "front", atk=9, hp=2)
        a = attack(back(0), front(0))
    elif case == "return_damage":        # our last frontline attacker dies to return damage
        add_unit(g, 0, "front", atk=1, hp=2)
        add_unit(g, 1, "back", atk=5, hp=9)
        a = attack(front(0), back(0))
    elif case == "mutual_kill":
        add_unit(g, 0, "back", atk=2, hp=2)
        add_unit(g, 1, "front", atk=2, hp=2)
        a = attack(back(0), front(0))
    else:
        raise ValueError(case)
    add_unit(g, 0, "back", atk=1, hp=9)                # ours, ready to move in
    add_unit(g, 1, "back", atk=1, hp=9)                # theirs, ready to move in next turn
    return g, a


@pytest.mark.parametrize("case", ["enemy_killed", "ranged_kill", "return_damage", "mutual_kill"])
def test_an_emptied_frontline_is_free_for_either_side(case):
    g, a = frontline_scenario(case)
    do(g, a)
    assert g.frontline == [] and g.front_owner is None
    mover = len(g.backline[0]) - 1
    assert move(mover) in g.legal_actions()            # we may move in this turn
    c = g.clone()
    do(c, move(mover))
    assert c.front_owner == 0
    do(g, END)
    theirs = len(g.backline[1]) - 1
    assert move(theirs) in g.legal_actions()           # or they may on theirs
    do(g, move(theirs))
    assert g.front_owner == 1 and g.observe(0).front_owner == -1 and g.observe(1).front_owner == 1


def test_enemy_cannot_move_in_while_we_hold_the_frontline():
    g = blank_game(current=0, first=0, round_=5)
    add_unit(g, 0, "front", atk=1, hp=9, moved=True)
    add_unit(g, 1, "back", atk=1, hp=9, move_cost=0)
    do(g, END)
    assert g.current_player() == 1
    assert_legal(g, [END, attack(back(0), front(0))])


# ---------------------------------------------------------------- combat, compaction, winning
def test_combat_is_simultaneous_with_pre_combat_attack_values():
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "back", atk=3, hp=2)
    add_unit(g, 1, "front", atk=3, hp=2)
    do(g, attack(back(0), front(0)))
    assert g.backline[0] == [] and g.frontline == [] and g.front_owner is None and g.base_hp == [20, 20]

    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "back", atk=1, hp=2)
    defender = add_unit(g, 1, "front", atk=3, hp=6)
    do(g, attack(back(0), front(0)))               # attacker dies, defender survives damaged
    assert g.backline[0] == [] and defender.hp == 5 and g.front_owner == 1
    assert not defender.attacked                   # being attacked does not use an action


def test_dead_units_compact_their_zone():
    g = blank_game(current=0, first=0, round_=5, coins=5)
    a0 = add_unit(g, 1, "front", atk=1, hp=2)
    add_unit(g, 1, "front", atk=2, hp=1)
    a2 = add_unit(g, 1, "front", atk=2, hp=3)
    b0 = add_unit(g, 0, "back", atk=1, hp=1)
    b1 = add_unit(g, 0, "back", atk=7, hp=8)
    b2 = add_unit(g, 0, "back", atk=4, hp=3)
    do(g, attack(back(1), front(1)))               # kills the middle frontline unit
    assert g.frontline == [a0, a2]
    do(g, attack(back(0), front(0)))               # attacker (1 hp) dies, target left with 1 hp
    assert g.backline[0] == [b1, b2] and hp_of(g.frontline) == [1, 3]
    # Slots refer to the compacted lists: the 4/3 is now backline slot 1.
    assert_legal(g, [END, attack(back(1), front(0)), attack(back(1), front(1))])

    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "front", atk=6, hp=6)
    units = [add_unit(g, 1, "back", atk=1, hp=hp) for hp in (1, 2, 3)]
    do(g, attack(front(0), back(1)))
    assert g.backline[1] == [units[0], units[2]]


@pytest.mark.parametrize("base_hp, ends", [(5, False), (4, True), (2, True)])
def test_destroying_the_base_ends_the_game_immediately(base_hp, ends):
    g = blank_game(current=0, first=0, round_=5, coins=5)
    add_unit(g, 0, "front", atk=4, hp=3)
    add_unit(g, 0, "front", atk=5, hp=4)
    add_unit(g, 1, "back", atk=1, hp=2)
    set_hand(g, 0, [cheapest(TROOP)])
    g.base_hp[1] = base_hp
    g.invalidate()
    do(g, attack(front(0), BASE))
    assert g.base_hp[1] == base_hp - 4
    if not ends:
        assert not g.done and g.winner() is None and len(g.legal_actions()) > 1
        return
    assert g.done and g.winner() == 0
    assert g.legal_actions() == [] and not g.legal_mask().any()
    for a in range(-1, NUM_ACTIONS + 1):
        with pytest.raises(IllegalActionError):
            g.step(a)
    me, opp = g.observe(0), g.observe(1)
    assert me.done and opp.done and (me.result, opp.result) == (1, -1)
    assert not me.is_my_turn and not opp.is_my_turn


@pytest.mark.parametrize("nature", NATURES, ids=NATURE_IDS)
def test_second_player_can_win_too(nature):
    g = blank_game(current=1, first=0, round_=7, coins=7)
    add_unit(g, 1, "back" if nature == RANGED else "front", atk=7, hp=1, nature=nature)
    g.base_hp[0] = 7
    g.invalidate()
    do(g, attack(back(0) if nature == RANGED else front(0), BASE))
    assert g.done and g.winner() == 1 and g.observe(1).result == 1 and g.observe(0).result == -1


def test_lethal_with_a_fast_unit_moving_in_and_attacking():
    g = blank_game(current=0, first=0, round_=6, coins=1)
    add_unit(g, 0, "back", atk=4, hp=1, nature=FAST, move_cost=1)
    g.base_hp[1] = 4
    g.invalidate()
    do(g, move(0))
    do(g, attack(front(0), BASE))
    assert g.done and g.winner() == 0


def test_hands_stay_sorted_and_card_indices_are_ints():
    g = new_game(17)
    for _ in range(60):
        if g.done:
            break
        for p in (0, 1):
            assert g.hands[p] == sorted(g.hands[p]) and all(type(c) is int for c in g.hands[p])
        do(g, g.legal_actions()[len(g.legal_actions()) // 2])
