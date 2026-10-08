"""Rule scenarios from SPEC.md §2, built by editing the documented engine state (SPEC §4)."""
from __future__ import annotations

import random

import pytest

from cardgame.actions import ActionKind as K
from cardgame.engine import DRAW, IllegalActionError
from conftest import (CONFIG, NUM_ACTIONS, H, Z, act, add_unit, blank_game, card, check_invariants,
                      cost, iter_states, new_game, set_hand)
from reference_rules import comparable, extract_state, reference_legal, reference_step

END = 0


def play(i: int) -> int:
    return act(K.PLAY, i)


def move(j: int) -> int:
    return act(K.MOVE, j)


def hit_base(j: int) -> int:
    return act(K.ATTACK_BASE, j)


def front_attack(j: int, k: int) -> int:
    return act(K.FRONT_ATTACK, j, k)


def back_attack(j: int, k: int) -> int:
    return act(K.BACK_ATTACK, j, k)


def kinds(game) -> set:
    return {game.action_space.decode(a).kind for a in game.legal_actions()}


def units(zone) -> list:
    return [(CONFIG.cards[u.card].id, u.atk, u.hp, u.ready) for u in zone]


def assert_legal(game, expected) -> None:
    """Engine and reference agree, and equal the hand-written expectation."""
    legal = game.legal_actions()
    assert legal == reference_legal(game)
    assert legal == sorted(expected), [game.action_space.describe(a) for a in legal]


# ---------------------------------------------------------------- setup
@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42, 1234, 99991])
def test_reset_follows_spec_setup_exactly(seed):
    rng = random.Random(seed)
    first = rng.randrange(2)
    decks = [list(CONFIG.decks[0]), list(CONFIG.decks[1])]
    rng.shuffle(decks[0])
    rng.shuffle(decks[1])
    hands = [[], []]
    for p, n in ((first, 4), (1 - first, 5), (first, 1)):  # opening hands, then turn-1 draw
        for _ in range(n):
            hands[p].append(decks[p].pop())

    g = new_game(seed)
    assert g.first_player == first and g.current_player() == first and g.current == first
    assert g.hands == [sorted(hands[0]), sorted(hands[1])]
    assert g.decks == decks
    assert g.rng.getstate() == rng.getstate()
    assert g.round == 1 and g.coins[first] == 1 and g.coins[1 - first] == 0
    assert g.base_hp == [20, 20] and g.burned == [0, 0]
    assert g.backline == [[], []] and g.frontline == [] and g.front_owner is None
    assert not g.done and g.winner() is None


def test_opening_hand_sizes_and_first_draws():
    for seed in range(40):
        g = new_game(seed)
        f, s = g.first_player, 1 - g.first_player
        assert (len(g.hands[f]), len(g.decks[f])) == (4 + 1, 40 - 5)  # 4 + turn-1 draw
        assert (len(g.hands[s]), len(g.decks[s])) == (5, 40 - 5)
        g.step(END)
        assert g.current_player() == s and g.round == 1
        assert (len(g.hands[s]), len(g.decks[s])) == (6, 34)
        assert (len(g.hands[f]), len(g.decks[f])) == (5, 35)


def test_coin_flip_varies_across_seeds():
    firsts = [new_game(seed).first_player for seed in range(400)]
    assert 140 <= sum(firsts) <= 260  # ~Binomial(400, 0.5)
    for seed in range(400):  # the first player always opens
        g = new_game(seed)
        assert g.current_player() == g.first_player == firsts[seed]


# ---------------------------------------------------------------- turn structure
@pytest.mark.parametrize("seed", [3, 8])
def test_end_turn_only_game_is_a_draw_after_50_rounds(seed):
    g = new_game(seed)
    f = g.first_player
    for n in range(1, 101):
        p = g.current_player()
        o = 1 - p
        hand, deck, burned = list(g.hands[o]), list(g.decks[o]), g.burned[o]
        rnd = g.round
        assert g.coins[p] == rnd and g.coins[o] == 0
        assert g.legal_actions()[0] == END
        g.step(END)
        assert g.coins[p] == 0  # unused coins are lost
        if n == 100:
            break
        assert not g.done and g.current_player() == o
        assert g.round == (rnd + 1 if o == f else rnd)  # increments only when the first player starts
        assert g.coins[o] == g.round
        if not deck:  # deck-out: no draw, no penalty
            assert (g.hands[o], g.decks[o], g.burned[o]) == (hand, [], burned)
        elif len(hand) == H:  # overdraw: the top card is burned
            assert (g.hands[o], g.decks[o], g.burned[o]) == (hand, deck[:-1], burned + 1)
        else:
            assert g.hands[o] == sorted(hand + [deck[-1]]) and g.decks[o] == deck[:-1]
        assert g.base_hp == [20, 20]
        check_invariants(g, (0, 0))
    assert g.done and g.winner() == DRAW
    assert g.round == 50
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


def test_unused_coins_are_lost_and_not_carried_over():
    g = blank_game(current=0, first=0, round_=3, coins=3)
    set_hand(g, 0, ["squire"])
    g.step(play(0))
    assert g.coins[0] == 2
    g.step(END)
    assert g.coins == [0, 3] and g.round == 3
    g.step(END)
    assert g.round == 4 and g.coins == [4, 0]


def test_overdraw_burns_the_top_card():
    g = blank_game(current=0, first=0)
    set_hand(g, 1, ["squire", "scout", "footman", "raider", "knight"] * 2)
    g.decks[1] = [card("ogre"), card("giant")]
    g.invalidate()
    hand = list(g.hands[1])
    assert len(hand) == H
    g.step(END)
    assert g.hands[1] == hand and g.decks[1] == [card("ogre")] and g.burned == [0, 1]
    assert g.coins[1] == 1 and not g.done


def test_draw_takes_the_top_card_and_keeps_the_hand_sorted():
    g = blank_game(current=0, first=0)
    set_hand(g, 1, ["squire", "giant", "scout", "raider", "knight"] + ["footman"] * 4)
    g.decks[1] = [card("champion"), card("sentinel")]  # top = end
    g.invalidate()
    expected = sorted(g.hands[1] + [card("sentinel")])
    g.step(END)
    assert g.hands[1] == expected and len(expected) == H
    assert g.decks[1] == [card("champion")] and g.burned == [0, 0]


def test_deck_out_has_no_penalty():
    g = blank_game(current=0, first=0)
    set_hand(g, 1, ["scout"])
    assert g.decks[1] == []
    g.step(END)
    assert not g.done and g.current_player() == 1
    assert g.hands[1] == [card("scout")] and g.decks[1] == [] and g.burned == [0, 0]
    assert g.base_hp == [20, 20] and g.coins[1] == 1


def test_units_ready_only_at_their_owners_turn_start():
    g = blank_game(current=0, first=0, round_=2)
    add_unit(g, 0, "back", "footman", ready=False)
    add_unit(g, 0, "front", "knight", ready=False)
    add_unit(g, 1, "back", "ogre", ready=False)
    add_unit(g, 1, "back", "scout", ready=False)
    g.step(END)  # P1's turn starts
    assert [u.ready for u in g.backline[1]] == [True, True]
    assert not g.backline[0][0].ready and not g.frontline[0].ready
    g.step(back_attack(1, 0))  # scout dies attacking the knight; the ogre stays ready
    assert units(g.backline[1]) == [("ogre", 5, 4, True)]
    g.step(END)  # P0's turn starts: both of P0's units refresh, P1's ogre is untouched
    assert g.backline[0][0].ready and g.frontline[0].ready
    assert g.backline[1][0].ready


# ---------------------------------------------------------------- deploying
def test_play_deploys_to_own_backline_with_summoning_sickness():
    g = blank_game(current=0, first=0, round_=3, coins=3)
    set_hand(g, 0, ["knight", "squire"])  # sorted: squire (slot 0), knight (slot 1)
    add_unit(g, 1, "back", "footman")
    assert_legal(g, [END, play(0), play(1)])
    g.step(play(1))
    assert g.hands[0] == [card("squire")] and g.coins[0] == 0
    assert units(g.backline[0]) == [("knight", 4, 3, False)]
    assert g.backline[0][0].owner == 0
    assert units(g.backline[1]) == [("footman", 2, 3, True)]
    assert_legal(g, [END])  # squire unaffordable, knight cannot act on its deploy turn
    g.step(END)
    assert not g.backline[0][0].ready  # still exhausted during the opponent's turn
    g.step(END)
    assert g.backline[0][0].ready
    assert move(0) in g.legal_actions()


def test_play_appends_and_respects_cost_and_capacity():
    g = blank_game(current=0, first=0, round_=12, coins=12)
    set_hand(g, 0, ["giant", "champion", "squire"])
    for _ in range(3):
        add_unit(g, 0, "back", "scout")
    assert_legal(g, [END, play(0), play(1), play(2)] + [move(j) for j in range(3)])
    g.step(play(0))  # squire
    g.step(play(1))  # giant: the hand is now (champion, giant)
    assert [CONFIG.cards[u.card].id for u in g.backline[0]] == ["scout"] * 3 + ["squire", "giant"]
    assert g.coins[0] == 12 - cost(card("squire")) - cost(card("giant"))
    assert len(g.backline[0]) == Z
    assert g.coins[0] >= cost(card("champion"))
    assert_legal(g, [END] + [move(j) for j in range(3)])  # backline full: no PLAY


def test_unaffordable_cards_are_not_playable():
    g = blank_game(current=0, first=0, round_=2, coins=2)
    set_hand(g, 0, ["knight", "giant", "footman", "scout"])
    # sorted: scout(1), footman(2), knight(3), giant(6)
    assert_legal(g, [END, play(0), play(1)])


# ---------------------------------------------------------------- moving
def test_move_into_empty_or_own_frontline_until_full():
    g = blank_game(current=0, first=0, round_=5)
    for cid in ("squire", "scout", "footman"):
        add_unit(g, 0, "back", cid)
    assert_legal(g, [END, move(0), move(1), move(2)])
    g.step(move(1))  # scout
    assert g.front_owner == 0 and units(g.frontline) == [("scout", 2, 1, False)]
    assert units(g.backline[0]) == [("squire", 1, 2, True), ("footman", 2, 3, True)]
    assert_legal(g, [END, move(0), move(1)])  # the moved scout cannot also attack
    g.step(move(1))
    assert [u.card for u in g.frontline] == [card("scout"), card("footman")]

    g = blank_game(current=0, first=0, round_=5)
    for _ in range(4):
        add_unit(g, 0, "front", "squire", ready=False)
    add_unit(g, 0, "back", "knight")
    add_unit(g, 0, "back", "ogre")
    assert_legal(g, [END, move(0), move(1)])
    g.step(move(0))
    assert len(g.frontline) == Z
    assert_legal(g, [END])  # frontline full: the ogre cannot move


def test_cannot_move_in_while_enemy_holds_frontline():
    g = blank_game(current=0, first=0, round_=5)
    add_unit(g, 1, "front", "squire", ready=False)  # 1/2
    add_unit(g, 1, "front", "scout", ready=False)   # 2/1
    add_unit(g, 0, "back", "knight")                # 4/3
    add_unit(g, 0, "back", "footman")               # 2/3
    add_unit(g, 0, "back", "raider")                # 3/2
    expected = [END] + [back_attack(j, k) for j in range(3) for k in range(2)]
    assert_legal(g, expected)  # no MOVE while the enemy holds the frontline
    g.step(back_attack(0, 1))  # knight kills scout, takes 2
    assert units(g.frontline) == [("squire", 1, 2, False)] and g.front_owner == 1
    assert units(g.backline[0])[0] == ("knight", 4, 1, False)
    assert_legal(g, [END, back_attack(1, 0), back_attack(2, 0)])  # one enemy left: still no MOVE
    g.step(back_attack(1, 0))  # footman kills squire, takes 1
    assert g.frontline == [] and g.front_owner is None
    # All enemy front units are dead: the remaining ready unit may move in this same turn,
    # the two that already attacked may not.
    assert_legal(g, [END, move(2)])
    g.step(move(2))
    assert g.front_owner == 0 and units(g.frontline) == [("raider", 3, 2, False)]


def test_enemy_cannot_move_in_while_we_hold_frontline():
    g = blank_game(current=0, first=0, round_=5)
    add_unit(g, 0, "front", "giant", ready=False)
    add_unit(g, 1, "back", "squire")
    g.step(END)
    assert g.current_player() == 1
    assert_legal(g, [END, back_attack(0, 0)])


# ---------------------------------------------------------------- attacking
def test_frontline_attacks_enemy_backline_and_base_only():
    g = blank_game(current=0, first=0, round_=5)
    add_unit(g, 0, "front", "knight")   # 4/3
    add_unit(g, 0, "front", "squire", ready=False)
    add_unit(g, 0, "back", "footman", ready=False)
    add_unit(g, 1, "back", "scout")
    add_unit(g, 1, "back", "sentinel")
    assert_legal(g, [END, hit_base(0), front_attack(0, 0), front_attack(0, 1)])
    assert kinds(g) == {K.END_TURN, K.ATTACK_BASE, K.FRONT_ATTACK}  # never BACK_ATTACK on our own front


def test_attack_base_deals_damage_and_base_does_not_hit_back():
    g = blank_game(current=0, first=0, round_=5)
    add_unit(g, 0, "front", "ogre")  # 5/4
    add_unit(g, 0, "front", "knight")
    g.step(hit_base(0))
    assert g.base_hp == [20, 15]
    assert units(g.frontline) == [("ogre", 5, 4, False), ("knight", 4, 3, True)]
    assert_legal(g, [END, hit_base(1)])  # one action per unit per round
    g.step(hit_base(1))
    assert g.base_hp == [20, 11] and not g.done


def test_backline_attacks_only_enemy_frontline():
    g = blank_game(current=0, first=0, round_=5)
    add_unit(g, 0, "back", "knight")
    add_unit(g, 0, "back", "squire", ready=False)
    add_unit(g, 1, "front", "footman", ready=False)
    add_unit(g, 1, "front", "raider", ready=False)
    add_unit(g, 1, "back", "scout")
    assert_legal(g, [END, back_attack(0, 0), back_attack(0, 1)])
    assert kinds(g) == {K.END_TURN, K.BACK_ATTACK}

    g = blank_game(current=0, first=0, round_=5)  # front empty: nothing to attack from the back
    add_unit(g, 0, "back", "knight")
    add_unit(g, 1, "back", "scout")
    assert_legal(g, [END, move(0)])


def test_combat_is_simultaneous_and_both_can_die():
    g = blank_game(current=0, first=0, round_=5)
    add_unit(g, 0, "back", "raider")   # 3/2
    add_unit(g, 1, "front", "raider")  # 3/2
    g.step(back_attack(0, 0))
    assert g.backline[0] == [] and g.frontline == [] and g.front_owner is None
    assert g.base_hp == [20, 20]
    check_invariants(g)

    g = blank_game(current=0, first=0, round_=5)
    add_unit(g, 0, "front", "knight")    # 4/3
    add_unit(g, 1, "back", "footman")    # 2/3
    g.step(front_attack(0, 0))
    assert g.backline[1] == [] and units(g.frontline) == [("knight", 4, 1, False)]
    assert g.front_owner == 0

    g = blank_game(current=0, first=0, round_=5)
    add_unit(g, 0, "back", "squire")     # 1/2
    add_unit(g, 1, "front", "sentinel")  # 3/6
    g.step(back_attack(0, 0))  # attacker dies, defender survives damaged
    assert g.backline[0] == [] and units(g.frontline) == [("sentinel", 3, 5, True)]  # defender stays ready
    assert g.front_owner == 1


def test_attacking_front_unit_that_dies_frees_the_frontline():
    g = blank_game(current=0, first=0, round_=5)
    add_unit(g, 0, "front", "scout")    # 2/1
    add_unit(g, 1, "back", "footman")   # 2/3
    g.step(front_attack(0, 0))
    assert g.frontline == [] and g.front_owner is None
    assert units(g.backline[1]) == [("footman", 2, 1, True)]
    g.step(END)
    assert move(0) in g.legal_actions()  # P1 may now move in


def test_dead_units_compact_their_zone():
    g = blank_game(current=0, first=0, round_=5)
    add_unit(g, 1, "front", "squire")    # 1/2
    add_unit(g, 1, "front", "scout")     # 2/1
    add_unit(g, 1, "front", "footman")   # 2/3
    add_unit(g, 0, "back", "squire", hp=1)
    add_unit(g, 0, "back", "giant")      # 7/8
    add_unit(g, 0, "back", "knight")
    g.step(back_attack(1, 1))  # giant kills scout
    assert units(g.frontline) == [("squire", 1, 2, True), ("footman", 2, 3, True)]
    g.step(back_attack(0, 0))  # squire (1 hp) trades into squire: attacker dies, target 1 hp
    assert [CONFIG.cards[u.card].id for u in g.backline[0]] == ["giant", "knight"]
    assert units(g.frontline) == [("squire", 1, 1, True), ("footman", 2, 3, True)]
    # Slots refer to the compacted lists: knight is now backline slot 1.
    assert_legal(g, [END, back_attack(1, 0), back_attack(1, 1)])

    g = blank_game(current=0, first=0, round_=5)
    add_unit(g, 0, "front", "champion")  # 6/6
    for cid in ("scout", "squire", "raider"):
        add_unit(g, 1, "back", cid)
    g.step(front_attack(0, 1))
    assert [CONFIG.cards[u.card].id for u in g.backline[1]] == ["scout", "raider"]
    assert units(g.frontline) == [("champion", 6, 5, False)]


# ---------------------------------------------------------------- winning
@pytest.mark.parametrize("base_hp, ends", [(5, False), (4, True), (2, True)])
def test_destroying_the_base_ends_the_game_immediately(base_hp, ends):
    g = blank_game(current=0, first=0, round_=5)
    add_unit(g, 0, "front", "knight")  # 4 atk
    add_unit(g, 0, "front", "ogre")
    add_unit(g, 1, "back", "squire")
    set_hand(g, 0, ["squire"])
    g.coins[0] = 5
    g.base_hp[1] = base_hp
    g.invalidate()
    g.step(hit_base(0))
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


def test_second_player_can_win_too():
    g = blank_game(current=1, first=0, round_=7)
    add_unit(g, 1, "front", "giant")
    g.base_hp[0] = 7
    g.invalidate()
    g.step(hit_base(0))
    assert g.done and g.winner() == 1 and g.observe(1).result == 1 and g.observe(0).result == -1


# ---------------------------------------------------------------- every transition vs the reference model
def test_every_transition_matches_reference_model():
    n = 0
    for seed in range(200, 215):
        for game, _, _ in iter_states(seed):
            before = extract_state(game)
            for a in game.legal_actions():
                c = game.clone()
                c.step(a)
                expected = reference_step(before, a, CONFIG)
                assert comparable(extract_state(c)) == comparable(expected), (
                    f"seed {seed}: {game.action_space.describe(a)} from\n{game.render()}\n"
                    f"engine:   {extract_state(c)}\nexpected: {expected}")
                n += 1
    assert n >= 5000
