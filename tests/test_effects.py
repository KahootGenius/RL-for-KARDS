"""Independent Stage 3 effect tests (SPEC 1.2, 2.8-2.10, 2.11 "Effects").

Every position is hand-built from in-memory cards (`build_ruleset`, SPEC 1.5) and public state
(SPEC 4); every expectation is derived from the SPEC clause cited next to it. Units are re-found by
uid after each step, so the tests never depend on the engine mutating Unit objects in place.
"""
from __future__ import annotations

from collections import Counter

import pytest

from stage3_helpers import (BASE, DRAW, END, MAIN, IllegalActionError, Rules, attack, back, bases_of, blank,
                            board_order, cards_of, counts, effect, find, front, hand_slot, move, multiset,
                            operation, play, predict_sample, put, set_deck, set_hand, state_key, tgt, unit,
                            uids, zone_of)


# ---------------------------------------------------------------- local helpers
def cast(g, r, cid):
    """Give the current player card `cid` and PLAY it (SPEC 2.6)."""
    p = g.current
    g.hands[p] = sorted(list(g.hands[p]) + [r.idx(cid)])
    g.invalidate()
    a = play(hand_slot(g, r, p, cid))
    assert a in g.legal_actions(), f"PLAY {cid} not legal: {g.legal_actions()}"
    g.step(a)


def hp(g, u):
    x = find(g, u.uid)
    return None if x is None else x.hp


def unit_of(g, u):
    x = find(g, u.uid)
    assert x is not None, f"uid {u.uid} left the board"
    return x


def cnt(g, field, p, r, cid):
    return counts(getattr(g, field)[p], r.n_cards)[r.idx(cid)]


def settled(g):
    """Between actions the queue is empty and nothing is pending (SPEC 4 `queue`)."""
    return g.phase == MAIN and g.pending is None and len(g.queue) == 0


# ================================================================= triggers (SPEC 1.2 table, 2.6)
def test_on_deploy_self_resolves_after_paying_and_deploying():
    # SPEC 2.6 PLAY unit: pay, deploy to backline[p], then its on_deploy triggers resolve, so the new
    # unit already counts for {"count": "units"} (SPEC 1.2 amounts).
    r = Rules([unit("bomber", 1, 2, cost=2, effects=[
        effect("on_deploy", "damage", "enemy_base", amount={"count": "units", "side": "friendly"})])])
    g = blank(r, coins=5)
    put(g, r, 0, "back", "fill00")
    cast(g, r, "bomber")
    assert bases_of(g) == [20, 18]
    assert g.coins[0] == 3
    assert cards_of(g, 0, r) == ["fill00", "bomber"]
    assert settled(g)


def test_summon_does_not_fire_on_deploy():
    # SPEC 1.2: on_deploy fires when a unit is played from hand (not when summoned); SPEC 2.10 summon.
    r = Rules([unit("tok", 1, 1, token=True, effects=[effect("on_deploy", "damage", "enemy_base", amount=3)]),
               unit("watch", 0, 5, effects=[effect("on_deploy", "damage", "enemy_base", scope="any", amount=1)]),
               operation("call", [effect("on_play", "summon", "controller", card="tok", amount=2)])])
    g = blank(r)
    put(g, r, 0, "back", "watch")
    put(g, r, 1, "back", "fill00")
    cast(g, r, "call")
    assert cards_of(g, 0, r) == ["watch", "tok", "tok"]
    assert bases_of(g) == [20, 20]
    for u in g.backline[0][1:]:
        assert u.summoned and u.token  # SPEC 2.10: summoned; SPEC 4 unit field `token`


@pytest.mark.parametrize("caster", [0, 1])
@pytest.mark.parametrize("scope", ["friendly", "enemy", "any"])
def test_on_play_watchers_on_units_watch_operations(scope, caster):
    # SPEC 1.2: on_play on units with scope friendly/enemy/any = an operation of that side is played.
    r = Rules([unit("listener", 0, 5, effects=[effect("on_play", "damage", "enemy_base", scope=scope, amount=1)]),
               operation("memo", [effect("on_play", "gain_coins", "controller", amount=1)]),
               unit("grunt", 1, 1)])
    g = blank(r, current=caster, first=0)
    put(g, r, 0, "back", "listener")
    cast(g, r, "memo")
    fires = scope == "any" or (scope == "friendly") == (caster == 0)
    expected = [20, 19 if fires else 20]
    assert bases_of(g) == expected
    cast(g, r, "grunt")  # a unit card is not an operation
    assert bases_of(g) == expected


def test_on_attack_resolves_before_combat_damage():
    # SPEC 2.6 ATTACK: 2. on_attack resolves, 4. combat damage (with the values at that time).
    r = Rules([unit("striker", 1, 5, effects=[effect("on_attack", "buff", "self", atk=2, hp=0)]),
               unit("dummy", 0, 3)])
    g = blank(r)
    a = put(g, r, 0, "front", "striker")
    d = put(g, r, 1, "back", "dummy")
    g.step(attack(front(0), back(0)))
    assert find(g, d.uid) is None  # 3 combat damage kills the 0/3 dummy
    s = unit_of(g, a)
    assert (s.atk, s.hp, s.attacks) == (3, 5, 1)


def test_on_attack_event_is_the_target_and_none_for_the_base():
    # SPEC 2.8 event units: on_attack -> the attack target; SPEC 2.6 ATTACK 2: None for the base,
    # and SPEC 2.9 `event`: no unit -> fizzle.
    r = Rules([unit("jabber", 1, 5, effects=[effect("on_attack", "damage", "event", amount=1)]),
               unit("dummy", 2, 3)])
    g = blank(r)
    a = put(g, r, 0, "front", "jabber")
    d = put(g, r, 1, "back", "dummy")
    g.step(attack(front(0), back(0)))
    assert hp(g, d) == 1   # 1 effect damage + 1 combat damage
    assert hp(g, a) == 3   # 2 return damage
    g2 = blank(r)
    put(g2, r, 0, "front", "jabber")
    g2.step(attack(front(0), BASE))
    assert bases_of(g2) == [20, 19] and settled(g2)


def test_attack_ends_when_on_attack_removes_the_target():
    # SPEC 2.6 ATTACK 3: the target left the board -> the attack ends (no combat damage).
    r = Rules([unit("assassin", 1, 5, effects=[effect("on_attack", "destroy", "event")]),
               unit("victim", 4, 4, effects=[effect("on_death", "damage", "enemy_base", amount=2)])])
    g = blank(r)
    a = put(g, r, 0, "front", "assassin")
    v = put(g, r, 1, "back", "victim")
    g.step(attack(front(0), back(0)))
    assert find(g, v.uid) is None
    s = unit_of(g, a)
    assert s.hp == 5 and s.attacks == 1
    assert bases_of(g) == [18, 20]  # SPEC 2.10 destroy: on_death fires
    assert cnt(g, "graveyard", 1, r, "victim") == 1


@pytest.mark.parametrize("target", ["unit", "base"])
def test_attack_ends_when_on_attack_removes_the_attacker(target):
    # SPEC 2.6 ATTACK 3: the attacker left the board -> the attack ends.
    r = Rules([unit("skirmisher", 3, 3, effects=[effect("on_attack", "return_to_hand", "self")]),
               unit("dummy", 1, 5)])
    g = blank(r)
    a = put(g, r, 0, "front", "skirmisher")
    d = put(g, r, 1, "back", "dummy")
    g.step(attack(front(0), back(0) if target == "unit" else BASE))
    assert find(g, a.uid) is None and g.frontline == [] and g.front_owner is None
    assert r.idx("skirmisher") in g.hands[0]
    assert hp(g, d) == 5 and bases_of(g) == [20, 20]


def test_attack_ends_when_on_attack_ends_the_game():
    # SPEC 2.6 ATTACK 3 (the game ended) and SPEC 2.8 damage step 1.
    r = Rules([unit("berserk", 3, 3, effects=[effect("on_attack", "damage", "friendly_base", amount=5)]),
               unit("dummy", 1, 5)])
    g = blank(r)
    put(g, r, 0, "front", "berserk")
    d = put(g, r, 1, "back", "dummy")
    g.base_hp[0] = 3
    g.invalidate()
    g.step(attack(front(0), back(0)))
    assert g.done and g.winner() == 1
    assert hp(g, d) == 5
    assert g.legal_actions() == []


def test_on_damaged_fires_for_survivors_with_the_combat_opponent_as_event():
    # SPEC 2.8: on_damaged for each damaged survivor, event unit = the combat opponent.
    r = Rules([unit("thorns", 1, 5, effects=[effect("on_damaged", "damage", "event", amount=2)]),
               unit("attacker", 1, 5)])
    g = blank(r)
    a = put(g, r, 0, "front", "attacker")
    t = put(g, r, 1, "back", "thorns")
    g.step(attack(front(0), back(0)))
    assert hp(g, t) == 4
    assert hp(g, a) == 2  # 1 return damage + 2 from thorns' on_damaged
    assert settled(g)


def test_on_damaged_needs_more_than_zero_damage_and_survival():
    # SPEC 1.2: on_damaged = a unit takes > 0 damage and survives.
    eff = [effect("on_damaged", "damage", "enemy_base", amount=1)]
    r = Rules([unit("thorns", 1, 3, effects=eff), unit("plated", 1, 3, traits={"armor": 2}, effects=eff),
               unit("big", 5, 9), unit("small", 1, 9)])
    g = blank(r)  # dies: no trigger
    put(g, r, 0, "front", "big")
    put(g, r, 1, "back", "thorns")
    g.step(attack(front(0), back(0)))
    assert g.backline[1] == [] and bases_of(g) == [20, 20]
    g = blank(r)  # armor absorbs everything: 0 damage, no trigger (SPEC 2.4 armor on combat damage)
    put(g, r, 0, "front", "small")
    p = put(g, r, 1, "back", "plated")
    g.step(attack(front(0), back(0)))
    assert hp(g, p) == 3 and bases_of(g) == [20, 20]
    g = blank(r)  # survives with damage: trigger (controller p1 hits p0's base)
    put(g, r, 0, "front", "small")
    t = put(g, r, 1, "back", "thorns")
    g.step(attack(front(0), back(0)))
    assert hp(g, t) == 2 and bases_of(g) == [19, 20]


def test_on_damaged_event_is_the_effect_source_unit():
    # SPEC 2.8: on_damaged event = the effect's source unit.
    r = Rules([unit("thorns", 1, 5, effects=[effect("on_damaged", "damage", "event", amount=2)]),
               unit("pinger", 1, 3, effects=[effect("on_deploy", "damage", tgt("all", "enemy"), amount=1)])])
    g = blank(r)
    t = put(g, r, 1, "back", "thorns")
    cast(g, r, "pinger")
    pinger = g.backline[0][0]
    assert hp(g, t) == 4 and pinger.hp == 1


def test_on_damaged_from_an_operation_has_no_event_unit():
    # SPEC 2.8 instances: source uid or None (operations have no source unit) -> `event` fizzles.
    r = Rules([unit("thorns", 1, 5, effects=[effect("on_damaged", "damage", "event", amount=2),
                                              effect("on_damaged", "damage", "enemy_base", amount=1)]),
               operation("zap", [effect("on_play", "damage", tgt("all", "enemy"), amount=1)])])
    g = blank(r)
    put(g, r, 0, "back", "fill00")
    t = put(g, r, 1, "back", "thorns")
    cast(g, r, "zap")
    assert hp(g, t) == 4
    assert g.backline[0][0].hp == g.backline[0][0].max_hp  # nothing hit the bystander
    assert bases_of(g) == [19, 20] and settled(g)          # the second effect still resolved


def test_on_kill_reads_the_victims_last_known_values():
    # SPEC 2.8: on_kill event = the victim; amounts read last-known values after it left the board.
    r = Rules([unit("reaper", 3, 9, effects=[
        effect("on_kill", "damage", "enemy_base", amount={"stat": "atk", "of": "event"})]), unit("brute", 4, 2)])
    g = blank(r)
    a = put(g, r, 0, "front", "reaper")
    v = put(g, r, 1, "back", "brute", atk=6)  # a modified value, not the card's
    g.step(attack(front(0), back(0)))
    assert find(g, v.uid) is None
    assert hp(g, a) == 3
    assert bases_of(g) == [20, 14]


def test_on_kill_needs_lethal_combat_damage():
    # SPEC 1.2: on_kill = destroys another unit with combat damage. A kill by the attacker's own
    # on_attack effect ends the attack (SPEC 2.6 ATTACK 3) without combat, so no on_kill.
    r = Rules([unit("reaper", 1, 9, effects=[effect("on_attack", "damage", "event", amount=2),
                                              effect("on_kill", "damage", "enemy_base", amount=5)]),
               unit("dummy", 1, 2)])
    g = blank(r)
    put(g, r, 0, "front", "reaper")
    put(g, r, 1, "back", "dummy")
    g.step(attack(front(0), back(0)))
    assert g.backline[1] == [] and bases_of(g) == [20, 20]


def test_on_kill_of_the_defender():
    # SPEC 2.8: on_kill for each unit that dealt lethal combat damage (the defender too).
    r = Rules([unit("guardian", 3, 5, effects=[effect("on_kill", "damage", "enemy_base", amount=1)]),
               unit("weakling", 1, 2)])
    g = blank(r)
    put(g, r, 0, "front", "weakling")
    gu = put(g, r, 1, "back", "guardian")
    g.step(attack(front(0), back(0)))
    assert g.frontline == [] and g.front_owner is None
    assert hp(g, gu) == 4 and bases_of(g) == [19, 20]


def test_on_kill_in_a_mutual_kill_still_fires():
    # SPEC 2.8: on_kill is enqueued for each unit that dealt lethal combat damage; the action targets
    # a base, not the (gone) source unit, so it does not fizzle (reading of "a unit action on an
    # event or source unit that has left the board fizzles").
    r = Rules([unit("reaper", 3, 2, effects=[effect("on_kill", "damage", "enemy_base", amount=2)]),
               unit("brute", 3, 3)])
    g = blank(r)
    put(g, r, 0, "front", "reaper")
    put(g, r, 1, "back", "brute")
    g.step(attack(front(0), back(0)))
    assert g.frontline == [] and g.backline[1] == []
    assert bases_of(g) == [20, 18]
    assert cnt(g, "graveyard", 0, r, "reaper") == 1 and cnt(g, "graveyard", 1, r, "brute") == 1


def test_unit_action_on_an_event_unit_that_left_the_board_fizzles():
    # SPEC 2.8: a unit action on an event unit that has left the board fizzles (on_kill event = the
    # dead victim); the next effect still resolves.
    r = Rules([unit("reaper", 3, 9, effects=[effect("on_kill", "return_to_hand", "event"),
                                              effect("on_kill", "damage", "enemy_base", amount=1)])])
    g = blank(r)
    put(g, r, 0, "front", "reaper")
    put(g, r, 1, "back", "fill00")
    g.step(attack(front(0), back(0)))
    assert list(g.hands[1]) == [] and cnt(g, "graveyard", 1, r, "fill00") == 1
    assert bases_of(g) == [20, 19] and settled(g)


def test_on_move_resolves_after_smokescreen_is_lost():
    # SPEC 2.6 MOVE: the unit loses smokescreen, then on_move triggers resolve.
    r = Rules([unit("mover", 1, 3, traits={"smokescreen": True}, effects=[
        effect("on_move", "damage", "enemy_base", amount=1),
        effect("on_move", "damage", "enemy_base",
               amount={"count": "units", "side": "friendly", "filter": {"trait": "smokescreen"}})])])
    g = blank(r)
    m = put(g, r, 0, "back", "mover")
    g.step(move(0))
    assert zone_of(g, m.uid) == ("front", 0)
    assert not unit_of(g, m).smokescreen
    assert bases_of(g) == [20, 19]


@pytest.mark.parametrize("scope, after_p1_start, after_p0_start", [
    ("friendly", 20, 17), ("enemy", 18, 18), ("any", 18, 15)])
def test_start_of_turn_scopes_resolve_after_the_draw(scope, after_p1_start, after_p0_start):
    # SPEC 2.3 turn start: draw 1, then start_of_turn triggers; scope friendly = the controller's
    # turn (default), enemy = the opponent's, any = both.
    sc = None if scope == "friendly" else scope
    r = Rules([unit("drummer", 0, 5, effects=[
        effect("start_of_turn", "damage", "enemy_base", scope=sc, amount={"count": "hand", "side": "friendly"})])])
    g = blank(r, current=0, first=0)
    put(g, r, 0, "back", "drummer")
    set_hand(g, r, 0, ["fill00", "fill01"])
    set_deck(g, r, 0, ["fill02", "fill03"])
    set_deck(g, r, 1, ["fill04"])
    g.step(END)  # p1's turn starts: p0 still holds 2 cards
    assert bases_of(g) == [20, after_p1_start]
    g.step(END)  # p0's turn starts: p0 draws to 3 cards first
    assert len(g.hands[0]) == 3
    assert bases_of(g) == [20, after_p0_start]


@pytest.mark.parametrize("scope, after_p0_end, after_p1_end", [
    ("friendly", 19, 19), ("enemy", 20, 19), ("any", 19, 18)])
def test_end_of_turn_scopes(scope, after_p0_end, after_p1_end):
    # SPEC 1.2 end_of_turn: friendly (default) = the controller's END_TURN, enemy, any.
    sc = None if scope == "friendly" else scope
    r = Rules([unit("bell", 0, 5, effects=[effect("end_of_turn", "damage", "enemy_base", scope=sc, amount=1)])])
    g = blank(r)
    put(g, r, 0, "back", "bell")
    g.step(END)
    assert bases_of(g) == [20, after_p0_end]
    g.step(END)
    assert bases_of(g) == [20, after_p1_end]


def test_end_of_turn_resolves_before_cleanup():
    # SPEC 2.3 END_TURN: 1. end_of_turn triggers, 3. coins[p] = 0.
    r = Rules([unit("miser", 0, 5, effects=[
        effect("end_of_turn", "damage", "enemy_base", amount={"count": "coins", "side": "friendly"})])])
    g = blank(r, coins=3)
    put(g, r, 0, "back", "miser")
    g.step(END)
    assert bases_of(g) == [20, 17]
    assert g.coins[0] == 0


def test_game_ending_during_end_of_turn_skips_the_remaining_steps():
    # SPEC 2.3 END_TURN: if the game ends during step 1, the remaining steps are skipped.
    r = Rules([unit("bell", 0, 5, effects=[effect("end_of_turn", "damage", "enemy_base", amount=5)]),
               operation("rally", [effect("on_play", "buff", tgt("all", "friendly"), atk=2, hp=0, duration="turn")])])
    g = blank(r, coins=6)
    b = put(g, r, 0, "back", "bell")
    cast(g, r, "rally")
    g.base_hp[1] = 3
    g.invalidate()
    turn = g.turn
    g.step(END)
    assert g.done and g.winner() == 0
    assert g.current == 0 and g.turn == turn and g.coins[0] == 5  # no cleanup, no next turn
    assert unit_of(g, b).atk == 2                                 # the "turn" buff did not expire


# ================================================================= watcher scopes (SPEC 1.2, 2.8)
WATCH_TRIGGERS = ("on_deploy", "on_death", "on_attack", "on_damaged", "on_move", "on_kill")


def watcher_scenario(trigger, scope, x, w, subject_is_watcher=False):
    """X (owned by x) performs the event; a watcher owned by w has `trigger` with `scope` and deals
    1 damage to its enemy base. With `subject_is_watcher`, X itself carries the watcher effect."""
    weff = effect(trigger, "damage", "enemy_base", scope=scope, amount=1)
    sub_effects = [weff] if subject_is_watcher else []
    if trigger in ("on_deploy", "on_move"):
        subject = unit("subject", 1, 3, effects=sub_effects)
    elif trigger in ("on_attack", "on_kill"):
        subject = unit("subject", 5, 3, nature="ranged", effects=sub_effects)
    elif trigger == "on_damaged":
        subject = unit("subject", 0, 5, nature="fast", effects=sub_effects)
    else:  # on_death
        subject = unit("subject", 0, 1, nature="fast", effects=sub_effects)
    cards = [subject, unit("watcher", 0, 9, effects=[weff]), unit("dummy", 0, 1 if trigger == "on_kill" else 9),
             operation("zap", [effect("on_play", "damage", tgt("all", "any", filter={"nature": "fast"}), amount=1)])]
    r = Rules(cards)
    g = blank(r, current=x, first=0)
    if not subject_is_watcher:
        put(g, r, w, "back", "watcher")
    if trigger == "on_deploy":
        set_hand(g, r, x, ["subject"])
        g.step(play(0))
    elif trigger == "on_move":
        put(g, r, x, "back", "subject")
        g.step(move(len(g.backline[x]) - 1))
    elif trigger in ("on_attack", "on_kill"):
        put(g, r, x, "back", "subject")
        a = len(g.backline[x]) - 1
        put(g, r, 1 - x, "back", "dummy")
        g.step(attack(back(a), back(len(g.backline[1 - x]) - 1)))
    else:
        put(g, r, x, "back", "subject")
        cast(g, r, "zap")
    return g


@pytest.mark.parametrize("x", [0, 1])
@pytest.mark.parametrize("relation", ["friendly", "enemy"])
@pytest.mark.parametrize("scope", ["friendly", "enemy", "any"])
@pytest.mark.parametrize("trigger", WATCH_TRIGGERS)
def test_watcher_scopes(trigger, scope, relation, x):
    # SPEC 1.2: scopes friendly/enemy/any watch other units relative to the watcher's controller.
    # For on_kill the watched unit is the killer ("the unit the event is about", SPEC 2.8).
    w = x if relation == "friendly" else 1 - x
    g = watcher_scenario(trigger, scope, x, w)
    fires = scope == "any" or scope == relation
    expected = [20, 20]
    if fires:
        expected[1 - w] = 19
    assert bases_of(g) == expected


@pytest.mark.parametrize("trigger", WATCH_TRIGGERS)
def test_watchers_never_include_the_source(trigger):
    # SPEC 1.2: scopes friendly/enemy/any watch **other** units (never the source itself).
    g = watcher_scenario(trigger, "any", 0, 0, subject_is_watcher=True)
    assert bases_of(g) == [20, 20]


def test_watcher_event_unit_is_the_watched_unit():
    # SPEC 2.8 event units: watchers -> the unit the event is about.
    r = Rules([unit("sentry", 0, 9, effects=[effect("on_deploy", "damage", "event", scope="enemy", amount=1)]),
               unit("plain", 1, 3)])
    g = blank(r)
    put(g, r, 1, "back", "sentry")
    cast(g, r, "plain")
    assert g.backline[0][0].hp == 2


# ================================================================= listener order (SPEC 2.8)
def token_cards(n=4):
    return [unit(f"t{k}", 0, 1, token=True) for k in range(1, n + 1)]


def summoner(k, trigger, to, scope="any"):
    sc = None if scope == "self" else scope
    return unit(f"w{k}", 0, 9, effects=[effect(trigger, "summon", to, scope=sc, card=f"t{k}")])


def test_own_effects_first_in_card_order_then_watchers():
    # SPEC 2.8 listener order: X's own matching effects (scope self, card order) first, then watchers.
    r = Rules(token_cards(3) + [
        unit("herald", 1, 3, effects=[effect("on_deploy", "summon", "opponent", card="t1"),
                                      effect("on_deploy", "summon", "opponent", card="t2")]),
        summoner(3, "on_deploy", "opponent", scope="friendly")])
    g = blank(r)
    put(g, r, 0, "back", "w3")  # earlier in board order than the herald
    cast(g, r, "herald")
    assert cards_of(g, 1, r) == ["t1", "t2", "t3"]


def test_watchers_in_board_order_turn_player_holds_the_frontline():
    # SPEC 2.8: turn player's backline, then frontline (if theirs), then the opponent's backline.
    # Uids are assigned against board order so uid order cannot pass for board order.
    r = Rules(token_cards() + [unit("plain", 1, 1), summoner(1, "on_deploy", "opponent"),
                               summoner(2, "on_deploy", "opponent"), summoner(3, "on_deploy", "opponent"),
                               summoner(4, "on_deploy", "controller")])
    g = blank(r)
    put(g, r, 0, "back", "w1", uid=13)
    put(g, r, 0, "back", "w2", uid=12)
    put(g, r, 0, "front", "w3", uid=11)
    put(g, r, 1, "back", "w4", uid=10)
    g.next_uid = 20
    cast(g, r, "plain")
    assert cards_of(g, 1, r) == ["w4", "t1", "t2", "t3", "t4"]


def test_watchers_in_board_order_opponent_holds_the_frontline():
    # SPEC 2.8: ... then the opponent's backline, then frontline.
    r = Rules(token_cards() + [unit("plain", 1, 1), summoner(1, "on_deploy", "opponent"),
                               summoner(2, "on_deploy", "opponent"), summoner(3, "on_deploy", "controller"),
                               summoner(4, "on_deploy", "controller")])
    g = blank(r)
    put(g, r, 0, "back", "w1", uid=13)
    put(g, r, 0, "back", "w2", uid=12)
    put(g, r, 1, "back", "w3", uid=11)
    put(g, r, 1, "front", "w4", uid=10)
    g.next_uid = 20
    cast(g, r, "plain")
    assert cards_of(g, 1, r) == ["w3", "t1", "t2", "t3", "t4"]


def test_watchers_board_order_starts_with_the_turn_player():
    # SPEC 2.8: board order is relative to the turn player (here p1).
    r = Rules(token_cards(3) + [unit("plain", 1, 1), summoner(1, "on_deploy", "opponent"),
                                summoner(2, "on_deploy", "controller"), summoner(3, "on_deploy", "opponent")])
    g = blank(r, current=1, first=0)
    put(g, r, 0, "back", "w1", uid=12)
    put(g, r, 1, "back", "w2", uid=11)
    put(g, r, 0, "front", "w3", uid=10)
    g.next_uid = 20
    cast(g, r, "plain")
    assert cards_of(g, 1, r) == ["w2", "plain", "t2", "t1", "t3"]


def test_start_of_turn_triggers_in_board_order():
    # SPEC 2.8: turn triggers enqueue the matching effects of all units in the same board order
    # (the turn that starts is p1's, so p1's units come first).
    r = Rules(token_cards(3) + [summoner(1, "start_of_turn", "opponent"), summoner(2, "start_of_turn", "controller"),
                                summoner(3, "start_of_turn", "opponent")])
    g = blank(r)
    put(g, r, 0, "back", "w1", uid=12)
    put(g, r, 1, "back", "w2", uid=11)
    put(g, r, 0, "front", "w3", uid=10)
    g.next_uid = 20
    g.step(END)
    assert g.current == 1
    assert cards_of(g, 1, r) == ["w2", "t2", "t1", "t3"]


def test_end_of_turn_triggers_in_board_order():
    # SPEC 2.8 board order at p0's END_TURN: p0 backline, p0 frontline, p1 backline.
    r = Rules(token_cards(3) + [summoner(1, "end_of_turn", "opponent"), summoner(2, "end_of_turn", "controller"),
                                summoner(3, "end_of_turn", "opponent")])
    g = blank(r)
    put(g, r, 1, "back", "w2", uid=12)
    put(g, r, 0, "front", "w3", uid=11)
    put(g, r, 0, "back", "w1", uid=10)
    g.next_uid = 20
    g.step(END)
    assert cards_of(g, 1, r) == ["w2", "t1", "t3", "t2"]


# ================================================================= damage step and chains (SPEC 2.8)
def test_damage_step_order_damaged_then_kill_then_death():
    # SPEC 2.8 damage step 3: on_damaged survivors, then on_kill, then on_death (removal order).
    r = Rules([unit("tok_d", 0, 1, token=True), unit("tok_k", 0, 1, token=True), unit("tok_x", 0, 1, token=True),
               unit("slayer", 3, 5, effects=[effect("on_kill", "summon", "opponent", card="tok_k"),
                                             effect("on_damaged", "summon", "opponent", card="tok_d")]),
               unit("martyr", 2, 3, effects=[effect("on_death", "summon", "controller", card="tok_x")])])
    g = blank(r)
    a = put(g, r, 0, "front", "slayer")
    put(g, r, 1, "back", "martyr")
    g.step(attack(front(0), back(0)))
    assert hp(g, a) == 3
    assert cards_of(g, 1, r) == ["tok_d", "tok_k", "tok_x"]


def test_combat_on_damaged_order_is_target_then_attacker():
    # SPEC 2.8: on_damaged for each damaged survivor (combat: target, then attacker).
    r = Rules([unit("tok_a", 0, 1, token=True), unit("tok_t", 0, 1, token=True),
               unit("hitter", 1, 5, effects=[effect("on_damaged", "summon", "opponent", card="tok_a")]),
               unit("taker", 1, 5, effects=[effect("on_damaged", "summon", "controller", card="tok_t")])])
    g = blank(r)
    put(g, r, 0, "front", "hitter")
    put(g, r, 1, "back", "taker")
    g.step(attack(front(0), back(0)))
    assert cards_of(g, 1, r) == ["taker", "tok_t", "tok_a"]


def test_chained_on_death_kills_another_unit_whose_on_death_fires():
    # SPEC 2.11 chained triggers: an on_death that kills another unit whose on_death fires.
    r = Rules([unit("powder", 0, 1, nature="fast",
                    effects=[effect("on_death", "damage", tgt("all", "friendly"), amount=1)]),
               unit("keg", 0, 1, effects=[effect("on_death", "damage", "enemy_base", amount=3)]),
               unit("crate", 0, 3),
               operation("spark", [effect("on_play", "damage", tgt("all", "enemy", filter={"nature": "fast"}),
                                          amount=1)])])
    g = blank(r)
    put(g, r, 1, "back", "powder")
    put(g, r, 1, "back", "keg")
    c = put(g, r, 1, "back", "crate")
    cast(g, r, "spark")
    assert cards_of(g, 1, r) == ["crate"]
    assert hp(g, c) == 2
    assert bases_of(g) == [17, 20]
    assert cnt(g, "graveyard", 1, r, "powder") == 1 and cnt(g, "graveyard", 1, r, "keg") == 1
    assert settled(g)


def test_simultaneous_deaths_are_removed_in_board_order():
    # SPEC 2.8 damage step 2: dead units are removed in board order; 3: on_death in removal order.
    # Board order with p0 to act: p0 backline, p1 backline, p1 frontline (SPEC 2.8 definition).
    toks = token_cards(3)
    martyrs = [unit(f"m{k}", 0, 2, effects=[effect("on_death", "summon", "controller", card=f"t{k}")])
               for k in (1, 2, 3)]
    r = Rules(toks + martyrs + [operation("purge", [effect("on_play", "destroy", tgt("all", "enemy"))])])
    g = blank(r)
    put(g, r, 1, "back", "m1", uid=12)
    put(g, r, 1, "back", "m2", uid=11)
    put(g, r, 1, "front", "m3", uid=10)
    g.next_uid = 20
    cast(g, r, "purge")
    assert g.frontline == [] and g.front_owner is None  # SPEC 2.8: an emptied frontline -> None
    assert cards_of(g, 1, r) == ["t1", "t2", "t3"]


def test_simultaneous_deaths_on_both_sides_follow_board_order():
    # SPEC 2.8: board order = turn player's backline (p0), frontline if theirs, then p1's zones.
    toks = token_cards(3)
    cards = toks + [unit("e1", 0, 2, effects=[effect("on_death", "summon", "opponent", card="t1")]),
                    unit("m2", 0, 2, effects=[effect("on_death", "summon", "controller", card="t2")]),
                    unit("m3", 0, 2, effects=[effect("on_death", "summon", "controller", card="t3")]),
                    operation("doom", [effect("on_play", "destroy", tgt("all", "any"))])]
    r = Rules(cards)
    g = blank(r)
    put(g, r, 1, "front", "m3", uid=10)
    put(g, r, 1, "back", "m2", uid=11)
    put(g, r, 0, "back", "e1", uid=12)
    g.next_uid = 20
    cast(g, r, "doom")
    assert g.backline[0] == []
    assert cards_of(g, 1, r) == ["t1", "t2", "t3"]


def test_on_death_watchers_must_still_be_on_the_board():
    # SPEC 2.8: on_death for each removed unit (watchers must still be on the board).
    eff = [effect("on_death", "damage", "enemy_base", scope="friendly", amount=1)]
    r = Rules([unit("comrade", 0, 1, effects=eff), unit("elder", 0, 5, effects=eff),
               operation("purge", [effect("on_play", "damage", tgt("all", "enemy"), amount=1)])])
    g = blank(r)
    put(g, r, 1, "back", "comrade")
    put(g, r, 1, "back", "comrade")
    e = put(g, r, 1, "back", "elder")
    cast(g, r, "purge")
    assert cards_of(g, 1, r) == ["elder"] and hp(g, e) == 4
    assert bases_of(g) == [18, 20]  # only the elder watched the two deaths


def test_game_ending_mid_chain_discards_the_queue():
    # SPEC 2.8 damage step 1: a base at <= 0 ends the game; the queue is discarded.
    r = Rules([unit("tok", 0, 1, token=True),
               unit("bomb", 0, 1, effects=[effect("on_death", "damage", "enemy_base", amount=5)]),
               unit("herald", 0, 1, effects=[effect("on_death", "summon", "controller", card="tok")]),
               operation("purge", [effect("on_play", "destroy", tgt("all", "enemy"))])])
    g = blank(r)
    put(g, r, 1, "back", "bomb")
    put(g, r, 1, "back", "herald")
    g.base_hp[0] = 3
    g.invalidate()
    cast(g, r, "purge")
    assert g.done and g.winner() == 1
    assert cards_of(g, 1, r) == []  # the herald's on_death never resolved
    assert len(g.queue) == 0 and g.pending is None
    assert g.legal_actions() == []


def test_game_ending_discards_a_later_chosen_effect():
    # SPEC 2.8 damage step 1: the queue and any pending choice are discarded.
    r = Rules([operation("finisher", [effect("on_play", "damage", "enemy_base", amount=5),
                                      effect("on_play", "damage", tgt("chosen", "enemy"), amount=1)])])
    g = blank(r)
    put(g, r, 1, "back", "fill00")
    g.base_hp[1] = 3
    g.invalidate()
    cast(g, r, "finisher")
    assert g.done and g.winner() == 0
    assert g.pending is None and len(g.queue) == 0 and g.legal_actions() == []


def test_both_bases_destroyed_by_one_effect_is_a_draw():
    # SPEC 2 / 2.8 damage step 1: both bases at <= 0 at once -> draw.
    r = Rules([operation("cataclysm", [effect("on_play", "damage", tgt("all", "any", "base"), amount=5)])])
    g = blank(r)
    g.base_hp = [5, 4]
    g.invalidate()
    cast(g, r, "cataclysm")
    assert g.done and g.winner() == DRAW
    assert bases_of(g) == [0, -1]


def test_own_base_destroyed_by_own_effect_loses():
    # SPEC 2: destroying the enemy base wins; here only the caster's base falls.
    r = Rules([operation("cataclysm", [effect("on_play", "damage", tgt("all", "any", "base"), amount=5)])])
    g = blank(r)
    g.base_hp = [5, 6]
    g.invalidate()
    cast(g, r, "cataclysm")
    assert g.done and g.winner() == 1 and bases_of(g) == [0, 1]


def echo_rules(limit):
    eff = [effect("on_damaged", "heal", "self", amount="full"),
           effect("on_damaged", "damage", tgt("all", "enemy"), amount=1)]
    return Rules([unit("echo_a", 0, 5, effects=eff), unit("echo_b", 0, 5, effects=eff),
                  operation("spark", [effect("on_play", "damage", tgt("all", "enemy"), amount=1)])],
                 max_effect_events=limit)


def run_echo(limit):
    r = echo_rules(limit)
    g = blank(r)
    a = put(g, r, 0, "back", "echo_a")
    b = put(g, r, 1, "back", "echo_b")
    cast(g, r, "spark")
    return g, a, b, r


@pytest.mark.parametrize("limit, expected", [(7, (4, 5)), (9, (5, 4)), (256, (5, 5))])
def test_loop_guard_stops_after_max_effect_events_instances(limit, expected):
    # SPEC 2.8 loop guard: at most max_effect_events instances resolve per action; the rest of the
    # queue is discarded and guard_trips += 1. Instance n: 1 = spark hits B; then B heal, B hits A,
    # A heal, A hits B, ... so after n instances A is damaged iff n % 4 == 3 and B iff n % 4 == 1.
    g, a, b, _ = run_echo(limit)
    assert (hp(g, a), hp(g, b)) == expected
    assert g.guard_trips == 1
    assert settled(g) and not g.done


def test_loop_guard_is_per_action_and_deterministic():
    # SPEC 2.8: "This is deterministic"; the limit counts instances per action.
    g1, a, b, r = run_echo(9)
    g2, _, _, _ = run_echo(9)
    assert state_key(g1) == state_key(g2)
    cast(g1, r, "spark")  # a fresh action gets a fresh budget and trips again
    assert g1.guard_trips == 2
    # 9 more instances from (A5, B4): spark hits B (B3), B heals to full, then the same 4-cycle
    assert (hp(g1, a), hp(g1, b)) == (5, 4)


# ================================================================= actions (SPEC 2.10)
def test_damage_all_enemy_units_is_simultaneous_and_ignores_armor_and_defense():
    # SPEC 2.10 damage: unit hp -= amount (no armor); SPEC 2.4: effects ignore Defense; dead units
    # removed in board order with zones compacting (SPEC 2.8).
    r = Rules([unit("plated", 1, 3, traits={"armor": 2}), unit("wall", 1, 5, traits={"defense": True}),
               operation("barrage", [effect("on_play", "damage", tgt("all", "enemy"), amount=2)])])
    g = blank(r)
    p = put(g, r, 1, "back", "plated")
    put(g, r, 1, "back", "fill01")   # 2/2
    put(g, r, 1, "back", "fill00")   # 1/1
    wl = put(g, r, 1, "front", "wall")
    cast(g, r, "barrage")
    assert cards_of(g, 1, r) == ["plated"] and hp(g, p) == 1
    assert hp(g, wl) == 3 and g.front_owner == 1
    assert cnt(g, "graveyard", 1, r, "fill00") == 1 and cnt(g, "graveyard", 1, r, "fill01") == 1


@pytest.mark.parametrize("target, expected", [("enemy_base", [20, 17]), ("friendly_base", [17, 20]),
                                              (tgt("all", "any", "base"), [17, 17])])
def test_damage_bases(target, expected):
    # SPEC 2.10 damage: base_hp -= amount; SPEC 1.2 shorthands friendly_base / enemy_base.
    r = Rules([operation("shell", [effect("on_play", "damage", target, amount=3)])])
    g = blank(r)
    cast(g, r, "shell")
    assert bases_of(g) == expected and not g.done


def test_damage_unit_or_base_all():
    # SPEC 1.2 kind unit_or_base with select all: every matching unit and base.
    r = Rules([operation("storm", [effect("on_play", "damage", tgt("all", "enemy", "unit_or_base"), amount=1)])])
    g = blank(r)
    mine = put(g, r, 0, "back", "fill03")
    e1 = put(g, r, 1, "back", "fill03")
    e2 = put(g, r, 1, "front", "fill03")
    cast(g, r, "storm")
    assert bases_of(g) == [20, 19]
    assert hp(g, e1) == 3 and hp(g, e2) == 3 and hp(g, mine) == 4


def test_heal_units_and_bases_with_caps():
    # SPEC 2.10 heal: unit hp = min(max_hp, hp + n); base min(config.base_hp, hp + n); "full" -> cap.
    r = Rules([operation("mend", [effect("on_play", "heal", tgt("all", "friendly"), amount=3)]),
               operation("restore", [effect("on_play", "heal", tgt("all", "friendly"), amount="full")]),
               operation("patch", [effect("on_play", "heal", "friendly_base", amount=3)]),
               operation("rebuild", [effect("on_play", "heal", "friendly_base", amount="full")]),
               unit("tank", 1, 8)])
    g = blank(r)
    t = put(g, r, 0, "back", "tank", hp=1)
    u = put(g, r, 0, "back", "tank", hp=6)
    g.base_hp[0] = 12
    g.invalidate()
    cast(g, r, "mend")
    assert (hp(g, t), hp(g, u)) == (4, 8)
    cast(g, r, "restore")
    assert (hp(g, t), hp(g, u)) == (8, 8)
    cast(g, r, "patch")
    assert bases_of(g) == [15, 20]
    g.base_hp[0] = 19
    g.invalidate()
    cast(g, r, "patch")
    assert bases_of(g) == [20, 20]
    g.base_hp[0] = 2
    g.invalidate()
    cast(g, r, "rebuild")
    assert bases_of(g) == [20, 20]


def test_buff_permanent():
    # SPEC 2.10 buff: atk = max(0, atk + a), max_hp += h, hp += h; permanent by default.
    r = Rules([operation("drill", [effect("on_play", "buff", tgt("all", "friendly"), atk=2, hp=3)]),
               operation("curse", [effect("on_play", "buff", tgt("all", "enemy"), atk=-5, hp=0)]),
               unit("cadet", 2, 4)])
    g = blank(r)
    c = put(g, r, 0, "back", "cadet", hp=1)
    e = put(g, r, 1, "back", "cadet")
    cast(g, r, "drill")
    x = unit_of(g, c)
    assert (x.atk, x.hp, x.max_hp) == (4, 4, 7)
    cast(g, r, "curse")
    y = unit_of(g, e)
    assert (y.atk, y.hp, y.max_hp) == (0, 4, 4)
    assert y.static_atk == 0  # SPEC 2.12: the clamp is not a static contribution
    g.step(END)
    g.step(END)
    x = unit_of(g, c)
    assert (x.atk, x.hp, x.max_hp) == (4, 4, 7)  # permanent: no expiry


def test_destroy_kills_without_damage():
    # SPEC 2.10 destroy: dies in the damage step, no damage dealt, on_death fires.
    r = Rules([unit("doomed", 1, 9, effects=[effect("on_damaged", "damage", "enemy_base", amount=7),
                                              effect("on_death", "damage", "enemy_base", amount=2)]),
               operation("execute", [effect("on_play", "destroy", tgt("all", "enemy"))])])
    g = blank(r)
    put(g, r, 1, "back", "doomed")
    cast(g, r, "execute")
    assert g.backline[1] == []
    assert bases_of(g) == [18, 20]  # on_death only (no on_damaged)
    assert cnt(g, "graveyard", 1, r, "doomed") == 1


def test_draw_takes_the_top_burns_at_ten_and_stops_on_empty_deck():
    # SPEC 2.10 draw: n cards (burn rule; empty deck: nothing); SPEC 2.1 top = end of list;
    # SPEC 2.3 hand at 10 -> burned.
    r = Rules([operation("study", [effect("on_play", "draw", "controller", amount=2)]),
               operation("cram", [effect("on_play", "draw", "controller", amount=3)])])
    g = blank(r)
    set_deck(g, r, 0, ["fill05", "fill06", "fill07"])
    cast(g, r, "study")
    assert g.hands[0] == sorted([r.idx("fill07"), r.idx("fill06")])
    assert list(g.deck_cards[0]) == [r.idx("fill05")]
    cast(g, r, "cram")  # only one card left
    assert len(g.hands[0]) == 3 and list(g.deck_cards[0]) == [] and g.burned[0] == 0
    g = blank(r)
    set_hand(g, r, 0, ["fill00"] * 3 + ["fill01"] * 3 + ["fill02"] * 3)  # 9 cards
    set_deck(g, r, 0, ["fill05", "fill06", "fill07"])
    cast(g, r, "study")  # 9 after playing: first draw -> 10, second burned
    assert len(g.hands[0]) == 10 and g.burned[0] == 1
    assert r.idx("fill07") in g.hands[0] and r.idx("fill06") not in g.hands[0]
    assert list(g.deck_cards[0]) == [r.idx("fill05")]
    assert g.observe(0).my_burned == 1 and g.observe(1).opp_burned == 1


@pytest.mark.parametrize("target, drawn", [("controller", (1, 0)), ("opponent", (0, 1)),
                                           (tgt("all", "friendly", "player"), (1, 0)),
                                           (tgt("all", "enemy", "player"), (0, 1)),
                                           (tgt("all", "any", "player"), (1, 1))])
def test_draw_target_players(target, drawn):
    # SPEC 1.2 kind player (select all) and shorthands controller / opponent; SPEC 2.9 side any = both.
    r = Rules([operation("study", [effect("on_play", "draw", target, amount=1)])])
    g = blank(r)
    set_deck(g, r, 0, ["fill05"])
    set_deck(g, r, 1, ["fill06"])
    cast(g, r, "study")
    assert (len(g.hands[0]), len(g.hands[1])) == drawn


def test_gain_coins_is_lost_at_end_turn():
    # SPEC 2.10 gain_coins: coins[p] += n; lost at END_TURN (SPEC 2.3 step 3).
    r = Rules([operation("loot", [effect("on_play", "gain_coins", "controller", amount=2)], cost=1)])
    g = blank(r, coins=3)
    cast(g, r, "loot")
    assert g.coins[0] == 4
    g.step(END)
    assert g.coins[0] == 0


def test_coins_gained_on_the_opponents_turn_are_replaced_at_turn_start():
    # SPEC 2.10: coins gained on the opponent's turn are lost at their own turn start
    # (SPEC 2.3: coins[p] = coins_for_round(round) + coin_bonus[p]).
    r = Rules([unit("broker", 0, 5, effects=[effect("on_deploy", "gain_coins", "controller", scope="enemy", amount=3)]),
               unit("plain", 1, 1)])
    g = blank(r, current=0, first=0, round_=2, coins=5)
    put(g, r, 1, "back", "broker")
    cast(g, r, "plain")
    assert g.coins[1] == 3
    g.step(END)
    assert g.current == 1 and g.coins[1] == r.config.coins_for_round(2)


def test_increase_max_coins_changes_income_from_the_next_turn_start():
    # SPEC 2.10 increase_max_coins: coin_bonus[p] += n; changes income from p's next turn start.
    r = Rules([operation("bank", [effect("on_play", "increase_max_coins", "controller", amount=2)], cost=1)])
    g = blank(r, coins=1)
    cast(g, r, "bank")
    assert g.coin_bonus[0] == 2 and g.coins[0] == 0
    assert g.observe(0).my_coin_bonus == 2 and g.observe(1).opp_coin_bonus == 2
    g.step(END)
    assert g.coins[1] == r.config.coins_for_round(1)
    g.step(END)
    assert g.round == 2 and g.coins[0] == r.config.coins_for_round(2) + 2


def test_increase_max_coins_negative_income_never_below_zero():
    # SPEC 2.10: amount may be < 0; income is never below 0 (SPEC 2.3 step 2).
    r = Rules([operation("tax", [effect("on_play", "increase_max_coins", "controller", amount=-5)], cost=0)])
    g = blank(r, coins=1)
    cast(g, r, "tax")
    assert g.coin_bonus[0] == -5
    g.step(END)
    g.step(END)
    assert g.round == 2 and g.coins[0] == 0


def test_increase_max_coins_for_the_opponent():
    r = Rules([operation("aid", [effect("on_play", "increase_max_coins", "opponent", amount=2)], cost=0)])
    g = blank(r, coins=1)
    cast(g, r, "aid")
    assert list(g.coin_bonus) == [0, 2]
    g.step(END)
    assert g.coins[1] == r.config.coins_for_round(1) + 2


TRAITS = ("defense", "blitz", "smokescreen", "fury")


@pytest.mark.parametrize("trait", TRAITS)
def test_add_trait_permanent(trait):
    # SPEC 2.10 add_trait: Defense, blitz, smokescreen and fury become true (permanent by default).
    r = Rules([operation("gift", [effect("on_play", "add_trait", tgt("all", "friendly"), trait=trait)]),
               unit("plain", 1, 3)])
    g = blank(r)
    u = put(g, r, 0, "back", "plain")
    assert not getattr(unit_of(g, u), trait)
    cast(g, r, "gift")
    assert getattr(unit_of(g, u), trait)
    g.step(END)
    g.step(END)
    assert getattr(unit_of(g, u), trait)


def test_add_trait_armor_amount():
    # SPEC 2.10: armor goes up by `amount` (SPEC 1.2: armor only, default 1).
    r = Rules([operation("plate", [effect("on_play", "add_trait", tgt("all", "friendly"), trait="armor", amount=2)]),
               operation("shim", [effect("on_play", "add_trait", tgt("all", "friendly"), trait="armor")]),
               unit("plain", 1, 3, traits={"armor": 1})])
    g = blank(r)
    u = put(g, r, 0, "back", "plain")
    cast(g, r, "plate")
    assert unit_of(g, u).armor == 3
    cast(g, r, "shim")
    assert unit_of(g, u).armor == 4


@pytest.mark.parametrize("trait", TRAITS + ("armor",))
def test_remove_trait_permanent(trait):
    # SPEC 2.10 remove_trait: Defense, blitz, smokescreen and fury become false; armor goes to 0.
    r = Rules([operation("strip", [effect("on_play", "remove_trait", tgt("all", "friendly"), trait=trait)]),
               unit("decorated", 1, 3, traits={"defense": True, "blitz": True, "smokescreen": True, "fury": True,
                                               "armor": 2})])
    g = blank(r)
    u = put(g, r, 0, "back", "decorated")
    cast(g, r, "strip")
    x = unit_of(g, u)
    for t in TRAITS:
        assert bool(getattr(x, t)) == (t != trait)
    assert x.armor == (0 if trait == "armor" else 2)
    g.step(END)
    g.step(END)
    x = unit_of(g, u)
    assert x.armor == (0 if trait == "armor" else 2)
    for t in TRAITS:
        assert bool(getattr(x, t)) == (t != trait)


def test_remove_trait_list():
    # SPEC 1.2 remove_trait: `trait` is a name or a list of names.
    r = Rules([operation("strip", [effect("on_play", "remove_trait", tgt("all", "friendly"),
                                          trait=["defense", "armor"])]),
               unit("decorated", 1, 3, traits={"defense": True, "fury": True, "armor": 2})])
    g = blank(r)
    u = put(g, r, 0, "back", "decorated")
    cast(g, r, "strip")
    x = unit_of(g, u)
    assert (bool(x.defense), x.armor, bool(x.fury)) == (False, 0, True)


def test_summon_fills_the_backline_only_while_there_is_space():
    # SPEC 2.10 summon: amount copies go into the target player's backline while there is space;
    # they are summoned (cannot act unless blitz).
    r = Rules([unit("militia", 1, 1, token=True),
               operation("muster", [effect("on_play", "summon", "controller", card="militia", amount=3)])])
    g = blank(r, coins=10)
    for _ in range(3):
        put(g, r, 0, "back", "fill00")
    before = {u.uid for u in g.backline[0]}
    cast(g, r, "muster")
    assert cards_of(g, 0, r) == ["fill00"] * 3 + ["militia"] * 2
    new = g.backline[0][3:]
    assert len({u.uid for u in g.backline[0]}) == 5 and not ({u.uid for u in new} & before)
    assert all(u.summoned and u.token and not u.can_move() and not u.can_attack() for u in new)
    assert move(3) not in g.legal_actions() and move(4) not in g.legal_actions()
    assert move(0) in g.legal_actions()
    views = g.observe(0).my_backline
    assert [v.token for v in views] == [False] * 3 + [True] * 2 and views[3].summoned


def test_summon_into_the_enemy_backline():
    r = Rules([unit("militia", 1, 1, token=True),
               operation("conscript", [effect("on_play", "summon", "opponent", card="militia")])])
    g = blank(r)
    cast(g, r, "conscript")
    assert cards_of(g, 1, r) == ["militia"] and g.backline[0] == []


def test_return_to_hand():
    # SPEC 2.10 return_to_hand: leaves the board (no on_death), card to its owner's hand, all
    # modifications lost, known to the opponent (SPEC 5: known_hand[p][c] += 1); frontline updated.
    r = Rules([unit("veteran", 2, 3, cost=3, effects=[effect("on_death", "damage", "enemy_base", amount=5)]),
               unit("mourner", 0, 5, effects=[effect("on_death", "damage", "enemy_base", scope="any", amount=1)]),
               operation("recall", [effect("on_play", "return_to_hand", tgt("all", "enemy", zone="frontline"))])])
    g = blank(r)
    put(g, r, 0, "back", "mourner")
    put(g, r, 1, "front", "veteran", atk=5, hp=1, max_hp=6)
    revealed_before = cnt(g, "revealed", 1, r, "veteran")
    cast(g, r, "recall")
    assert g.frontline == [] and g.front_owner is None
    assert list(g.hands[1]) == [r.idx("veteran")]
    assert bases_of(g) == [20, 20]
    assert cnt(g, "known_hand", 1, r, "veteran") == 1
    assert counts(g.observe(0).opp_known_hand, r.n_cards)[r.idx("veteran")] == 1
    assert counts(g.observe(1).my_known_hand, r.n_cards)[r.idx("veteran")] == 1
    assert cnt(g, "graveyard", 1, r, "veteran") == 0
    g.step(END)
    g.coins[1] = 3
    g.invalidate()
    g.step(play(0))
    v = g.backline[1][0]
    assert (v.atk, v.hp, v.max_hp) == (2, 3, 3)
    assert cnt(g, "known_hand", 1, r, "veteran") == 0             # SPEC 5: the known copy left
    assert cnt(g, "revealed", 1, r, "veteran") == revealed_before  # no newly seen copy


def test_return_to_hand_with_a_full_hand_burns_the_card():
    # SPEC 2.10: a full hand burns the card.
    r = Rules([operation("recall", [effect("on_play", "return_to_hand", tgt("all", "enemy"))])])
    g = blank(r)
    put(g, r, 1, "back", "fill05")
    set_hand(g, r, 1, ["fill00"] * 3 + ["fill01"] * 3 + ["fill02"] * 3 + ["fill03"])
    cast(g, r, "recall")
    assert g.backline[1] == [] and len(g.hands[1]) == 10 and r.idx("fill05") not in g.hands[1]
    assert g.burned[1] == 1
    assert cnt(g, "known_hand", 1, r, "fill05") == 0


def test_discard_is_random_exact_and_public():
    # SPEC 2.10 discard: n random cards from the target player's hand (rng.sample) go to that
    # player's discard pile and become public (SPEC 5 revealed bookkeeping).
    r = Rules([operation("sabotage", [effect("on_play", "discard", "opponent", amount=2)])])
    g = blank(r)
    set_hand(g, r, 1, ["fill00", "fill01", "fill02", "fill03", "fill04"])
    hand_before = list(g.hands[1])
    state = g.rng.getstate()
    cast(g, r, "sabotage")
    picked = predict_sample(state, hand_before, 2)
    assert multiset(g.hands[1]) == multiset(hand_before) - multiset(picked)
    assert list(g.hands[1]) == sorted(g.hands[1])
    d = counts(g.discard[1], r.n_cards)
    rv = counts(g.revealed[1], r.n_cards)
    for c in range(r.n_cards):
        assert d[c] == picked.count(c) and rv[c] == picked.count(c)
    assert counts(g.observe(0).opp_discard, r.n_cards) == d
    assert cnt(g, "discard", 0, r, "sabotage") == 1  # the operation itself (SPEC 2.6)


def test_discarding_a_known_card_decrements_known_hand():
    # SPEC 5: when p discards a card c with known_hand[p][c] > 0, decrement it (the known copy left);
    # tokens never enter revealed.
    r = Rules([unit("ration", 0, 1, token=True),
               operation("gift", [effect("on_play", "add_card", "opponent", card="ration")]),
               operation("sabotage", [effect("on_play", "discard", "opponent", amount=1)])])
    g = blank(r)
    cast(g, r, "gift")
    assert list(g.hands[1]) == [r.idx("ration")] and cnt(g, "known_hand", 1, r, "ration") == 1
    cast(g, r, "sabotage")
    assert list(g.hands[1]) == []
    assert cnt(g, "known_hand", 1, r, "ration") == 0
    assert cnt(g, "revealed", 1, r, "ration") == 0
    assert cnt(g, "discard", 1, r, "ration") == 1


def test_add_card_tokens_are_known_and_burn():
    # SPEC 2.10 add_card: copies of the token go into the target player's hand (burn rule) and are
    # known to the opponent; SPEC 5: added token cards known_hand += 1, never revealed.
    r = Rules([operation("ration", [effect("on_play", "heal", "friendly_base", amount=2)], cost=0, token=True),
               operation("supply", [effect("on_play", "add_card", "controller", card="ration", amount=2)])])
    g = blank(r)
    cast(g, r, "supply")
    assert list(g.hands[0]) == [r.idx("ration")] * 2
    assert cnt(g, "known_hand", 0, r, "ration") == 2 and cnt(g, "revealed", 0, r, "ration") == 0
    assert counts(g.observe(1).opp_known_hand, r.n_cards)[r.idx("ration")] == 2
    # playing a known token: the known copy leaves the hand, nothing is revealed
    g.base_hp[0] = 10
    g.invalidate()
    g.step(play(0))
    assert bases_of(g) == [12, 20]
    assert cnt(g, "known_hand", 0, r, "ration") == 1 and cnt(g, "revealed", 0, r, "ration") == 0
    assert cnt(g, "discard", 0, r, "ration") == 1
    g = blank(r)
    set_hand(g, r, 0, ["fill00"] * 3 + ["fill01"] * 3 + ["fill02"] * 3)
    cast(g, r, "supply")
    assert len(g.hands[0]) == 10 and g.burned[0] == 1
    assert cnt(g, "known_hand", 0, r, "ration") == 1


def test_retreat_frontline_to_backline():
    # SPEC 2.10 retreat: a frontline unit goes to its owner's backline; no on_death; the frontline
    # is updated as for deaths.
    r = Rules([unit("soldier", 2, 3, effects=[effect("on_death", "damage", "enemy_base", amount=5)]),
               operation("fallback", [effect("on_play", "retreat", tgt("all", "friendly", zone="frontline"))])])
    g = blank(r)
    put(g, r, 0, "back", "fill00")
    s = put(g, r, 0, "front", "soldier")
    cast(g, r, "fallback")
    assert g.frontline == [] and g.front_owner is None
    assert cards_of(g, 0, r) == ["fill00", "soldier"] and zone_of(g, s.uid) == ("back", 0)
    assert bases_of(g) == [20, 20]


def test_retreat_frontline_to_hand_when_the_backline_is_full():
    r = Rules([unit("soldier", 2, 3, effects=[effect("on_death", "damage", "enemy_base", amount=5)]),
               operation("fallback", [effect("on_play", "retreat", tgt("all", "enemy", zone="frontline"))])])
    g = blank(r)
    for _ in range(5):
        put(g, r, 1, "back", "fill00")
    put(g, r, 1, "front", "soldier")
    cast(g, r, "fallback")
    assert g.frontline == [] and g.front_owner is None
    assert len(g.backline[1]) == 5 and list(g.hands[1]) == [r.idx("soldier")]
    assert bases_of(g) == [20, 20]


def test_retreat_backline_to_hand():
    r = Rules([unit("soldier", 2, 3, effects=[effect("on_death", "damage", "enemy_base", amount=5)]),
               operation("withdraw", [effect("on_play", "retreat", tgt("all", "friendly", zone="backline"))])])
    g = blank(r)
    put(g, r, 0, "back", "soldier")
    f = put(g, r, 0, "front", "fill00")
    cast(g, r, "withdraw")
    assert g.backline[0] == [] and list(g.hands[0]) == [r.idx("soldier")]
    assert zone_of(g, f.uid) == ("front", 0)
    assert bases_of(g) == [20, 20]


# ================================================================= select / side / zone / kind (SPEC 1.2, 2.9)
@pytest.mark.parametrize("front_owner", [0, 1])
@pytest.mark.parametrize("side", ["friendly", "enemy", "any"])
@pytest.mark.parametrize("zone", ["board", "backline", "frontline"])
def test_select_all_side_and_zone(front_owner, side, zone):
    # SPEC 1.2: side relative to the controller; zone applies to units; the frontline's units belong
    # to front_owner.
    r = Rules([operation("zap", [effect("on_play", "damage", tgt("all", side, zone=zone), amount=1)])])
    g = blank(r)
    a0 = put(g, r, 0, "back", "fill03")
    b0 = put(g, r, 1, "back", "fill03")
    f0 = put(g, r, front_owner, "front", "fill03")
    cast(g, r, "zap")
    owner = {a0.uid: 0, b0.uid: 1, f0.uid: front_owner}
    zone_name = {a0.uid: "backline", b0.uid: "backline", f0.uid: "frontline"}
    for u in (a0, b0, f0):
        side_ok = side == "any" or (side == "friendly") == (owner[u.uid] == 0)
        zone_ok = zone == "board" or zone == zone_name[u.uid]
        assert hp(g, u) == (3 if side_ok and zone_ok else 4), (u.uid, side, zone)


def test_select_self():
    # SPEC 1.2 select self: the source unit.
    r = Rules([unit("proud", 1, 1, effects=[effect("on_deploy", "buff", "self", atk=1, hp=2)])])
    g = blank(r)
    put(g, r, 0, "back", "fill03")
    cast(g, r, "proud")
    p = g.backline[0][1]
    assert (p.atk, p.hp, p.max_hp) == (2, 3, 3)
    assert (g.backline[0][0].atk, g.backline[0][0].hp) == (1, 4)


def test_self_target_fizzles_when_the_source_left_the_board():
    # SPEC 2.9 self: the unit if it is still on the board (by uid), else fizzle.
    r = Rules([unit("flincher", 1, 3, effects=[effect("on_damaged", "return_to_hand", "self"),
                                                effect("on_damaged", "buff", "self", atk=5, hp=5),
                                                effect("on_damaged", "damage", "enemy_base", amount=1)]),
               operation("zap", [effect("on_play", "damage", tgt("all", "enemy"), amount=1)])])
    g = blank(r)
    put(g, r, 1, "back", "flincher")
    cast(g, r, "zap")
    assert g.backline[1] == [] and list(g.hands[1]) == [r.idx("flincher")]
    assert bases_of(g) == [19, 20] and settled(g)


def test_random_single_target_is_exact():
    # SPEC 2.9 random: rng.sample(candidates, count) over units in board order.
    r = Rules([operation("snipe", [effect("on_play", "damage", tgt("random", "enemy"), amount=1)])])
    for seed in range(6):
        g = blank(r, seed=seed)
        g.rng.seed(1000 + seed)
        for _ in range(5):
            put(g, r, 1, "back", "fill03")
        cands = uids(g.backline[1])
        state = g.rng.getstate()
        cast(g, r, "snipe")
        [pick] = predict_sample(state, cands, 1)
        for u in g.backline[1]:
            assert u.hp == (3 if u.uid == pick else 4)


def test_random_count_two_is_exact():
    r = Rules([operation("volley", [effect("on_play", "damage", tgt("random", "enemy", count=2), amount=1)])])
    for seed in range(6):
        g = blank(r, seed=seed)
        g.rng.seed(77 + seed)
        for _ in range(4):
            put(g, r, 1, "back", "fill03")
        put(g, r, 1, "front", "fill03")
        cands = uids(g.backline[1]) + uids(g.frontline)  # enemy backline, then enemy frontline
        state = g.rng.getstate()
        cast(g, r, "volley")
        picks = set(predict_sample(state, cands, 2))
        for u in g.backline[1] + g.frontline:
            assert u.hp == (3 if u.uid in picks else 4)


@pytest.mark.parametrize("n_units, count", [(2, 3), (3, 3), (1, 1)])
def test_random_with_at_most_count_candidates_takes_all_without_rng(n_units, count):
    # SPEC 2.9: if <= count candidates, all of them (no RNG draw).
    r = Rules([operation("volley", [effect("on_play", "damage", tgt("random", "enemy", count=count), amount=1)])])
    g = blank(r)
    for _ in range(n_units):
        put(g, r, 1, "back", "fill03")
    state = g.rng.getstate()
    cast(g, r, "volley")
    assert g.rng.getstate() == state
    assert all(u.hp == 3 for u in g.backline[1])


def test_random_any_side_uses_board_order_across_zones():
    # SPEC 2.9 units listed in board order (SPEC 2.8: p0 backline, p0 frontline, p1 backline).
    r = Rules([operation("chaos", [effect("on_play", "damage", tgt("random", "any", count=2), amount=1)])])
    for seed in range(8):
        g = blank(r)
        g.rng.seed(seed)
        put(g, r, 0, "back", "fill03")
        put(g, r, 0, "back", "fill03")
        put(g, r, 0, "front", "fill03")
        put(g, r, 1, "back", "fill03")
        put(g, r, 1, "back", "fill03")
        cands = uids(board_order(g))
        assert cands == uids(g.backline[0]) + uids(g.frontline) + uids(g.backline[1])
        state = g.rng.getstate()
        cast(g, r, "chaos")
        picks = set(predict_sample(state, cands, 2))
        for u in board_order(g):
            assert u.hp == (3 if u.uid in picks else 4)


def test_random_targets_respect_filters():
    # SPEC 1.2: filters narrow the candidates; SPEC 2.9 sample over the remaining list.
    r = Rules([unit("runner", 1, 4, nature="fast"),
               operation("hunt", [effect("on_play", "damage", tgt("random", "enemy", filter={"nature": "fast"}),
                                         amount=1)])])
    for seed in range(5):
        g = blank(r)
        g.rng.seed(seed)
        put(g, r, 1, "back", "fill03")
        put(g, r, 1, "back", "runner")
        put(g, r, 1, "back", "fill03")
        put(g, r, 1, "back", "runner")
        cands = [u.uid for u in g.backline[1] if u.card == r.idx("runner")]
        state = g.rng.getstate()
        cast(g, r, "hunt")
        [pick] = predict_sample(state, cands, 1)
        for u in g.backline[1]:
            assert u.hp == u.max_hp - (1 if u.uid == pick else 0)


def test_random_base_any_order_is_controller_then_opponent():
    # SPEC 2.9 bases: side any = both, in the order controller's base, opponent's base.
    r = Rules([operation("misfire", [effect("on_play", "damage", tgt("random", "any", "base"), amount=2)])])
    seen = set()
    for seed in range(12):
        for current in (0, 1):
            g = blank(r, current=current, first=0)
            g.rng.seed(seed)
            state = g.rng.getstate()
            cast(g, r, "misfire")
            [pick] = predict_sample(state, [current, 1 - current], 1)
            expected = [20, 20]
            expected[pick] = 18
            assert bases_of(g) == expected
            seen.add(pick == current)
    assert seen == {True, False}


def test_random_from_a_watcher_on_the_opponents_turn():
    # SPEC 2.9: the side is relative to the effect's controller (the watcher's owner p1).
    r = Rules([unit("ambusher", 0, 9, effects=[effect("on_deploy", "damage", tgt("random", "enemy"),
                                                      scope="enemy", amount=1)]),
               unit("plain", 1, 4)])
    for seed in range(5):
        g = blank(r)
        g.rng.seed(seed)
        put(g, r, 0, "back", "fill03")
        put(g, r, 0, "back", "fill03")
        put(g, r, 1, "back", "ambusher")
        state = g.rng.getstate()
        cast(g, r, "plain")
        cands = uids(g.backline[0])  # p0's backline incl. the plain just deployed
        [pick] = predict_sample(state, cands, 1)
        for u in g.backline[0]:
            assert u.hp == u.max_hp - (1 if u.uid == pick else 0)


def test_select_all_never_uses_the_rng():
    # SPEC 2.8: randomness only from the game RNG for random selections and discards.
    r = Rules([operation("barrage", [effect("on_play", "damage", tgt("all", "any", "unit_or_base"), amount=1)])])
    g = blank(r)
    for p in (0, 1):
        put(g, r, p, "back", "fill03")
    state = g.rng.getstate()
    cast(g, r, "barrage")
    assert g.rng.getstate() == state


@pytest.mark.parametrize("side, expected", [("friendly", [17, 20]), ("enemy", [20, 17]), ("any", [17, 17])])
def test_base_sides(side, expected):
    r = Rules([operation("shell", [effect("on_play", "damage", tgt("all", side, "base"), amount=3)])])
    g = blank(r, current=1, first=0)
    cast(g, r, "shell")
    assert bases_of(g) == expected[::-1]  # the controller is p1


# ================================================================= filters (SPEC 1.2 F)
FILTER_CARDS = [
    unit("ftroop", 1, 3, cost=1),
    unit("ffast", 3, 4, nature="fast", cost=4, traits={"defense": True}),
    unit("franged", 2, 2, nature="ranged", cost=2, traits={"armor": 1}),
    unit("ftoken", 0, 5, cost=0, token=True),
]


@pytest.mark.parametrize("flt, hit", [
    ({"nature": "fast"}, {"ffast"}),
    ({"nature": ["troop", "ranged"]}, {"ftroop", "franged", "ftoken"}),
    ({"trait": "defense"}, {"ffast"}),
    ({"trait": "armor"}, {"franged"}),
    ({"not_trait": "defense"}, {"ftroop", "franged", "ftoken"}),
    ({"damaged": True}, {"ftroop"}),
    ({"damaged": False}, {"ffast", "franged", "ftoken"}),
    ({"min_cost": 2}, {"ffast", "franged"}),
    ({"max_cost": 1}, {"ftroop", "ftoken"}),
    ({"min_atk": 2}, {"ffast", "franged"}),
    ({"max_atk": 1}, {"ftroop", "ftoken"}),
    ({"min_hp": 4}, {"ffast", "ftoken"}),
    ({"max_hp": 2}, {"ftroop", "franged"}),
    ({"token": True}, {"ftoken"}),
    ({"token": False}, {"ftroop", "ffast", "franged"}),
    ({"nature": "troop", "max_cost": 0}, {"ftoken"}),
])
def test_filters(flt, hit):
    # SPEC 1.2 filters (all given keys must hold); min_hp/max_hp compare current hp; damaged = hp < max_hp.
    r = Rules(FILTER_CARDS + [operation("probe", [effect("on_play", "damage", tgt("all", "any", filter=flt),
                                                         amount=1)])])
    g = blank(r)
    units = {"ftroop": put(g, r, 0, "back", "ftroop", hp=2), "franged": put(g, r, 0, "back", "franged"),
             "ffast": put(g, r, 1, "back", "ffast"), "ftoken": put(g, r, 1, "back", "ftoken")}
    before = {k: u.hp for k, u in units.items()}
    cast(g, r, "probe")
    for k, u in units.items():
        assert hp(g, u) == before[k] - (1 if k in hit else 0), (flt, k)


def test_filter_other_excludes_the_source():
    # SPEC 1.2 filter `other` (true: not the source unit).
    r = Rules([unit("medic", 1, 3, effects=[effect("on_deploy", "buff", tgt("all", "friendly", filter={"other": True}),
                                                   atk=1, hp=1)]),
               unit("captain", 1, 3, effects=[effect("on_deploy", "buff", tgt("all", "friendly"), atk=1, hp=1)])])
    g = blank(r)
    f = put(g, r, 0, "back", "fill03")
    cast(g, r, "medic")
    m = g.backline[0][1]
    assert (unit_of(g, f).atk, m.atk) == (2, 1)
    cast(g, r, "captain")
    c = g.backline[0][2]
    assert (unit_of(g, f).atk, find(g, m.uid).atk, c.atk) == (3, 2, 2)


# ================================================================= amounts (SPEC 1.2 AM)
@pytest.mark.parametrize("stat, expected", [("atk", 3), ("hp", 2), ("max_hp", 4), ("cost", 5)])
def test_amount_stat_of_self(stat, expected):
    # SPEC 1.2 {"stat", "of": "self"}: current values while the unit is on the board.
    r = Rules([unit("gauge", 3, 4, cost=5, effects=[
        effect("on_attack", "damage", "enemy_base", amount={"stat": stat, "of": "self"})]), unit("dummy", 0, 9)])
    g = blank(r)
    put(g, r, 0, "front", "gauge", hp=2)
    put(g, r, 1, "back", "dummy")
    g.step(attack(front(0), back(0)))
    assert bases_of(g) == [20, 20 - expected]


@pytest.mark.parametrize("stat, expected", [("atk", 2), ("hp", 7), ("max_hp", 9), ("cost", 4)])
def test_amount_stat_of_event(stat, expected):
    # SPEC 1.2 {"of": "event"}; SPEC 2.8 on_attack event = the target (still on the board).
    r = Rules([unit("gauge", 0, 9, nature="ranged", effects=[
        effect("on_attack", "damage", "enemy_base", amount={"stat": stat, "of": "event"})]),
        unit("dummy", 2, 9, cost=4)])
    g = blank(r)
    put(g, r, 0, "back", "gauge")
    put(g, r, 1, "back", "dummy", hp=7)
    g.step(attack(back(0), back(0)))
    assert bases_of(g) == [20, 20 - expected]


def test_amount_of_self_uses_last_known_values_after_death():
    # SPEC 2.8: amounts read the unit's last-known values after it left the board.
    r = Rules([unit("avenger", 2, 1, cost=3, effects=[
        effect("on_death", "damage", "enemy_base", amount={"stat": "atk", "of": "self"}),
        effect("on_death", "damage", "enemy_base", amount={"stat": "max_hp", "of": "self"}),
        effect("on_death", "damage", "enemy_base", amount={"stat": "cost", "of": "self"})]),
        operation("purge", [effect("on_play", "destroy", tgt("all", "enemy"))])])
    g = blank(r)
    put(g, r, 1, "back", "avenger", atk=6, max_hp=4, hp=4)
    cast(g, r, "purge")
    assert bases_of(g) == [20 - 6 - 4 - 3, 20]


def test_amount_of_event_uses_last_known_values_for_a_dead_watched_unit():
    # SPEC 2.8: watcher event = the dead unit; its last-known values.
    r = Rules([unit("undertaker", 0, 9, effects=[
        effect("on_death", "damage", "enemy_base", scope="enemy", amount={"stat": "atk", "of": "event"})]),
        operation("purge", [effect("on_play", "destroy", tgt("all", "enemy"))])])
    g = blank(r)
    put(g, r, 0, "back", "undertaker")
    put(g, r, 1, "back", "fill03", atk=7)
    cast(g, r, "purge")
    assert bases_of(g) == [20, 13]


def test_amount_count_units_with_side_zone_and_filter():
    # SPEC 1.2 {"count": "units", "side", "zone"?, "filter"?}: matching units on the board,
    # evaluated before the action applies (SPEC 2.8 resolve steps 3-4).
    def amt(**kw):
        return dict({"count": "units"}, **kw)
    r = Rules([operation("c1", [effect("on_play", "damage", "enemy_base", amount=amt(side="enemy"))]),
               operation("c2", [effect("on_play", "damage", "enemy_base", amount=amt(side="any", zone="backline"))]),
               operation("c3", [effect("on_play", "damage", "enemy_base",
                                       amount=amt(side="friendly", filter={"min_atk": 2}))]),
               operation("c4", [effect("on_play", "damage", tgt("all", "enemy"), amount=amt(side="enemy"))])])
    g = blank(r)
    put(g, r, 0, "back", "fill00")  # 1/1
    put(g, r, 0, "back", "fill01")  # 2/2
    put(g, r, 1, "back", "fill03")
    put(g, r, 1, "front", "fill02")  # 3/3
    put(g, r, 1, "front", "fill00")  # 1/1
    cast(g, r, "c1")
    assert bases_of(g) == [20, 17]
    cast(g, r, "c2")
    assert bases_of(g) == [20, 14]
    cast(g, r, "c3")
    assert bases_of(g) == [20, 13]
    cast(g, r, "c4")  # 3 enemy units counted before the damage: 3 damage to each
    assert [u.hp for u in g.backline[1]] == [1] and g.frontline == [] and g.front_owner is None


@pytest.mark.parametrize("what, side, expected", [("hand", "friendly", 2), ("hand", "enemy", 4),
                                                  ("coins", "friendly", 6), ("coins", "enemy", 0),
                                                  ("deck", "friendly", 3), ("deck", "enemy", 1)])
def test_amount_count_hand_coins_deck(what, side, expected):
    # SPEC 1.2 {"count": "hand" | "coins" | "deck", "side"} at evaluation time (the operation has
    # already been paid for and left the hand, SPEC 2.6).
    r = Rules([operation("gauge", [effect("on_play", "damage", "enemy_base", amount={"count": what, "side": side})],
                         cost=1)])
    g = blank(r, coins=7)
    set_hand(g, r, 0, ["fill00", "fill01"])
    set_hand(g, r, 1, ["fill00"] * 4)
    set_deck(g, r, 0, ["fill02"] * 3)
    set_deck(g, r, 1, ["fill02"])
    cast(g, r, "gauge")
    assert bases_of(g) == [20, 20 - expected]


def test_amount_times_plus_and_clamp():
    # SPEC 1.2: value = max(0, times * value + plus).
    r = Rules([operation("a1", [effect("on_play", "damage", "enemy_base",
                                       amount={"count": "hand", "side": "enemy", "times": 2, "plus": 1})]),
               operation("a2", [effect("on_play", "damage", "enemy_base",
                                       amount={"count": "hand", "side": "enemy", "plus": -10})]),
               operation("a3", [effect("on_play", "damage", "enemy_base",
                                       amount={"count": "hand", "side": "enemy", "times": 3})])])
    g = blank(r)
    set_hand(g, r, 1, ["fill00"] * 3)
    cast(g, r, "a1")
    assert bases_of(g) == [20, 13]
    cast(g, r, "a2")  # max(0, 3 - 10) = 0
    assert bases_of(g) == [20, 13]
    cast(g, r, "a3")
    assert bases_of(g) == [20, 4]


def test_buff_accepts_amount_expressions():
    # SPEC 1.2: buff.atk / buff.hp accept the same forms as amounts.
    r = Rules([unit("growth", 1, 1, cost=3, effects=[
        effect("on_deploy", "buff", "self", atk={"count": "units", "side": "enemy"},
               hp={"stat": "cost", "of": "self", "plus": 1})])])
    g = blank(r)
    put(g, r, 1, "back", "fill00")
    put(g, r, 1, "back", "fill00")
    cast(g, r, "growth")
    u = g.backline[0][0]
    assert (u.atk, u.hp, u.max_hp) == (3, 5, 5)


# ================================================================= conditions (SPEC 1.2 C)
def cond_op(cid, condition):
    return operation(cid, [effect("on_play", "damage", "enemy_base", amount=1, condition=condition)], cost=0)


@pytest.mark.parametrize("condition, holds", [
    ({"type": "control", "side": "enemy"}, True),
    ({"type": "control", "side": "enemy", "min": 3}, False),
    ({"type": "control", "side": "enemy", "max": 1}, False),
    ({"type": "control", "side": "enemy", "min": 2, "max": 2}, True),
    ({"type": "control", "side": "friendly"}, False),
    ({"type": "control", "side": "any", "zone": "frontline"}, True),
    ({"type": "control", "side": "enemy", "zone": "backline", "min": 2}, False),
    ({"type": "control", "side": "enemy", "filter": {"nature": "fast"}}, True),
    ({"type": "control", "side": "enemy", "filter": {"nature": "ranged"}}, False),
    # SPEC 1.2: `min` defaults to 1 when neither bound is given, and to 0 when only `max` is given.
    ({"type": "control", "side": "friendly", "max": 0}, True),
    ({"type": "control", "side": "friendly", "max": 3}, True),
    ({"type": "control", "side": "enemy", "max": 2}, True),
    ({"type": "control", "side": "enemy", "filter": {"nature": "ranged"}, "max": 1}, True),
    ({"type": "control", "side": "enemy", "filter": {"nature": "ranged"}, "min": 0}, True),
    ({"type": "control", "side": "enemy", "filter": {"nature": "ranged"}, "min": 1}, False),
    ({"type": "frontline", "owner": "enemy"}, True),
    ({"type": "frontline", "owner": "friendly"}, False),
    ({"type": "frontline", "owner": "none"}, False),
    ({"type": "base_hp", "side": "friendly", "max": 10}, True),
    ({"type": "base_hp", "side": "enemy", "max": 10}, False),
    ({"type": "base_hp", "side": "enemy", "min": 15}, True),
    ({"type": "hand_size", "side": "enemy", "min": 3}, True),
    ({"type": "hand_size", "side": "friendly", "min": 1}, False),
    ({"type": "hand_size", "side": "enemy", "max": 2}, False),
    ({"type": "turn", "whose": "own"}, True),
    ({"type": "turn", "whose": "opponent"}, False),
    ([{"type": "turn", "whose": "own"}, {"type": "frontline", "owner": "enemy"}], True),
    ([{"type": "turn", "whose": "own"}, {"type": "frontline", "owner": "none"}], False),
])
def test_conditions(condition, holds):
    # SPEC 1.2 conditions (checked when the effect starts to resolve; a list = all must hold).
    r = Rules([unit("runner", 1, 4, nature="fast"), cond_op("test", condition)])
    g = blank(r)
    g.base_hp[0] = 8
    g.invalidate()
    put(g, r, 1, "back", "fill03")
    put(g, r, 1, "front", "runner")
    set_hand(g, r, 1, ["fill00"] * 3)
    cast(g, r, "test")
    assert bases_of(g) == [8, 19 if holds else 20]


def test_condition_frontline_none():
    r = Rules([cond_op("test", {"type": "frontline", "owner": "none"})])
    g = blank(r)
    cast(g, r, "test")
    assert bases_of(g) == [20, 19]


@pytest.mark.parametrize("zone, holds", [("frontline", True), ("backline", False)])
def test_condition_source_zone(zone, holds):
    # SPEC 1.2 {"type": "source_zone"}: the source unit's zone.
    r = Rules([unit("spotter", 1, 9, nature="ranged", effects=[
        effect("on_attack", "damage", "enemy_base", amount=2, condition={"type": "source_zone", "zone": "frontline"})]),
        unit("dummy", 0, 9)])
    g = blank(r)
    put(g, r, 0, "front" if zone == "frontline" else "back", "spotter")
    put(g, r, 1, "back", "dummy")
    g.step(attack(front(0) if zone == "frontline" else back(0), back(0)))
    assert bases_of(g) == [20, 18 if holds else 20]


@pytest.mark.parametrize("deployer, fires", [(0, True), (1, False)])
def test_condition_turn_for_a_watcher(deployer, fires):
    # SPEC 1.2 {"type": "turn", "whose": "own"} relative to the effect's controller.
    r = Rules([unit("patriot", 0, 9, effects=[effect("on_deploy", "damage", "enemy_base", scope="any", amount=1,
                                                     condition={"type": "turn", "whose": "own"})]),
               unit("plain", 1, 1)])
    g = blank(r, current=deployer, first=0)
    put(g, r, 0, "back", "patriot")
    cast(g, r, "plain")
    assert bases_of(g) == [20, 19 if fires else 20]


def test_condition_is_checked_when_the_effect_starts_to_resolve():
    # SPEC 1.2 / 2.8 resolve step 1: the condition is checked at resolution, not when enqueued.
    r = Rules([operation("sweep", [
        effect("on_play", "destroy", tgt("all", "enemy")),
        effect("on_play", "damage", "enemy_base", amount=3, condition={"type": "control", "side": "enemy"})]),
        unit("militia", 1, 1, token=True),
        operation("rally", [
            effect("on_play", "summon", "controller", card="militia"),
            effect("on_play", "damage", "enemy_base", amount=2,
                   condition={"type": "control", "side": "friendly", "filter": {"token": True}})])])
    g = blank(r)
    put(g, r, 1, "back", "fill03")
    cast(g, r, "sweep")
    assert g.backline[1] == [] and bases_of(g) == [20, 20]
    cast(g, r, "rally")
    assert cards_of(g, 0, r) == ["militia"] and bases_of(g) == [20, 18]


# ================================================================= operations and graveyards (SPEC 2.6, 4)
def test_operations_never_occupy_the_board_and_go_to_discard():
    # SPEC 2.6 PLAY operation: pay, put the card in p's discard pile, then on_play resolves; SPEC 5
    # revealed bookkeeping for a played non-token card.
    r = Rules([operation("flare", [effect("on_play", "damage", "enemy_base", amount=2)], cost=2)])
    g = blank(r, coins=3)
    for _ in range(5):
        put(g, r, 0, "back", "fill00")  # a full backline does not block operations
    cast(g, r, "flare")
    assert len(g.backline[0]) == 5 and g.frontline == []
    assert g.coins[0] == 1 and list(g.hands[0]) == []
    assert cnt(g, "discard", 0, r, "flare") == 1
    assert cnt(g, "revealed", 0, r, "flare") == 1
    assert counts(g.observe(1).opp_discard, r.n_cards)[r.idx("flare")] == 1
    assert counts(g.observe(0).my_discard, r.n_cards)[r.idx("flare")] == 1
    assert bases_of(g) == [20, 18]


def test_unit_play_needs_backline_space_but_operations_do_not():
    r = Rules([operation("flare", [effect("on_play", "damage", "enemy_base", amount=2)]), unit("plain", 1, 1)])
    g = blank(r)
    for _ in range(5):
        put(g, r, 0, "back", "fill00")
    set_hand(g, r, 0, ["flare", "plain"])
    legal = g.legal_actions()
    assert play(hand_slot(g, r, 0, "flare")) in legal
    assert play(hand_slot(g, r, 0, "plain")) not in legal


def test_graveyards_count_dead_units_including_tokens():
    # SPEC 4: graveyard[2] = dead units per card index, tokens included; SPEC 5 my/opp_graveyard.
    r = Rules([unit("militia", 1, 1, token=True),
               operation("muster", [effect("on_play", "summon", "opponent", card="militia")]),
               operation("purge", [effect("on_play", "damage", tgt("all", "enemy"), amount=5)])])
    g = blank(r)
    put(g, r, 1, "back", "fill03")
    cast(g, r, "muster")
    cast(g, r, "purge")
    assert g.backline[1] == []
    gy = counts(g.graveyard[1], r.n_cards)
    assert gy[r.idx("militia")] == 1 and gy[r.idx("fill03")] == 1 and sum(gy) == 2
    assert counts(g.observe(0).opp_graveyard, r.n_cards) == gy
    assert counts(g.observe(1).my_graveyard, r.n_cards) == gy
    assert sum(counts(g.graveyard[0], r.n_cards)) == 0


# ================================================================= loader validation (SPEC 1.1 / 1.2)
def _bad_cards():
    ok_dmg = effect("on_deploy", "damage", "enemy_base", amount=1)
    tok = unit("tok", 1, 1, token=True)
    return {
        "chosen_on_death": [unit("x", effects=[effect("on_death", "damage", tgt("chosen", "enemy"), amount=1)])],
        "chosen_watcher": [unit("x", effects=[effect("on_deploy", "damage", tgt("chosen", "enemy"), scope="any",
                                                     amount=1)])],
        "chosen_start_of_turn": [unit("x", effects=[effect("start_of_turn", "damage", tgt("chosen", "enemy"),
                                                           amount=1)])],
        "operation_on_deploy": [operation("x", [ok_dmg])],
        "operation_scope_friendly": [operation("x", [effect("on_play", "damage", "enemy_base", scope="friendly",
                                                            amount=1)])],
        "operation_no_effects": [operation("x", [])],
        "unit_on_play_self": [unit("x", effects=[effect("on_play", "damage", "enemy_base", amount=1)])],
        "on_death_targets_self": [unit("x", effects=[effect("on_death", "buff", "self", atk=1, hp=1)])],
        "event_without_event_unit": [operation("x", [effect("on_play", "damage", "event", amount=1)])],
        "of_event_without_event_unit": [operation("x", [effect("on_play", "damage", "enemy_base",
                                                               amount={"stat": "atk", "of": "event"})])],
        "count_with_all": [operation("x", [effect("on_play", "damage", tgt("all", "enemy", count=2), amount=1)])],
        "player_with_random": [operation("x", [effect("on_play", "draw", tgt("random", "any", "player"),
                                                      amount=1)])],
        "self_with_base": [unit("x", effects=[effect("on_deploy", "damage", tgt("self", kind="base"), amount=1)])],
        "missing_side": [operation("x", [effect("on_play", "damage", tgt("all"), amount=1)])],
        "destroy_base": [operation("x", [effect("on_play", "destroy", "enemy_base")])],
        "draw_on_unit": [operation("x", [effect("on_play", "draw", tgt("all", "enemy"), amount=1)])],
        "amount_on_destroy": [operation("x", [effect("on_play", "destroy", tgt("all", "enemy"), amount=1)])],
        "amount_on_add_defense": [operation("x", [effect("on_play", "add_trait", tgt("all", "friendly"),
                                                         trait="defense", amount=1)])],
        "full_on_damage": [operation("x", [effect("on_play", "damage", "enemy_base", amount="full")])],
        "summon_non_token": [operation("x", [effect("on_play", "summon", "controller", card="fill00")])],
        "add_card_unknown": [operation("x", [effect("on_play", "add_card", "controller", card="nope")])],
        "unknown_trigger": [unit("x", effects=[effect("on_sneeze", "damage", "enemy_base", amount=1)])],
        "unknown_action": [operation("x", [effect("on_play", "explode", "enemy_base", amount=1)])],
        "unknown_filter": [operation("x", [effect("on_play", "damage", tgt("all", "enemy", filter={"colour": "red"}),
                                                  amount=1)])],
        "unknown_trait": [operation("x", [effect("on_play", "add_trait", tgt("all", "friendly"), trait="flying")])],
        "unknown_effect_key": [operation("x", [dict(effect("on_play", "damage", "enemy_base", amount=1), bogus=1)])],
        "four_effects": [unit("x", effects=[ok_dmg] * 4)],
        "unknown_condition": [operation("x", [effect("on_play", "damage", "enemy_base", amount=1,
                                                     condition={"type": "weather"})])],
        "negative_buff_hp": [operation("x", [effect("on_play", "buff", tgt("all", "friendly"), atk=0, hp=-1)])],
        "bad_duration": [operation("x", [effect("on_play", "buff", tgt("all", "friendly"), atk=1, hp=0,
                                                duration="forever")])],
        "operation_with_stats": [dict(operation("x", [effect("on_play", "damage", "enemy_base", amount=1)]),
                                      attack=1)],
        "token_in_deck": [tok],
    }


@pytest.mark.parametrize("name", sorted(_bad_cards()))
def test_loader_rejects_invalid_effects(name):
    # SPEC 1.1: the loader is strict (ValueError); SPEC 1.2 validation; tokens never appear in decks.
    extra = _bad_cards()[name]
    decks = None
    if name == "token_in_deck":
        d = {"name": "bad", "style": "", "cards": {**{f"fill{i:02d}": 3 for i in range(13)}, "tok": 1}}
        decks = [d, d]
    with pytest.raises(ValueError):
        Rules(extra, decks=decks)


def test_loader_accepts_the_valid_forms_used_here():
    # Positive control for the validation test above.
    Rules([unit("tok", 1, 1, token=True),
           unit("x", effects=[effect("on_deploy", "damage", tgt("chosen", "enemy", "unit_or_base"), amount=1),
                              effect("on_death", "summon", "controller", card="tok"),
                              effect("on_attack", "damage", "event", amount={"stat": "atk", "of": "event"})]),
           operation("y", [effect("on_play", "draw", tgt("all", "any", "player"), amount=1)])])
