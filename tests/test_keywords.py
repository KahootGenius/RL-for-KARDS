"""Independent Stage 3 keyword tests (SPEC 2.4, 2.3 END_TURN, 2.10; SPEC 2.11 "Keywords"):
blitz, fury, smokescreen, armor, Defense vs effects, pin and its expiry on both players' turns, and
"turn" buffs / traits expiring at END_TURN without killing."""
from __future__ import annotations

import pytest

from stage3_helpers import (BASE, END, Rules, attack, back, bases_of, blank, effect, find, front, hand_slot, move,
                            operation, play, predict_sample, put, set_hand, tgt, unit, uids)


def cast(g, r, cid):
    p = g.current
    g.hands[p] = sorted(list(g.hands[p]) + [r.idx(cid)])
    g.invalidate()
    a = play(hand_slot(g, r, p, cid))
    assert a in g.legal_actions(), f"PLAY {cid} not legal: {g.legal_actions()}"
    g.step(a)


def U(g, u):
    x = find(g, u.uid)
    assert x is not None, f"uid {u.uid} left the board"
    return x


def unit_attacks(g, slot):
    """Legal ATTACK target slots of attacker slot `slot`."""
    out = []
    for t in range(11):
        if attack(slot, t) in g.legal_actions():
            out.append(t)
    return out


# ================================================================= blitz (SPEC 2.4)
def test_blitz_unit_can_attack_on_its_deploy_turn():
    # SPEC 2.4: ready = not pinned and (not summoned or blitz).
    r = Rules([unit("charger", 2, 2, traits={"blitz": True}), unit("rookie", 2, 2)])
    g = blank(r)
    e = put(g, r, 1, "front", "fill03")
    set_hand(g, r, 0, ["charger", "rookie"])
    g.step(play(hand_slot(g, r, 0, "charger")))
    c = g.backline[0][0]
    assert c.summoned and c.blitz and c.can_attack()
    view = g.observe(0).my_backline[0]
    assert view.blitz and view.summoned and view.can_attack
    g.step(play(hand_slot(g, r, 0, "rookie")))
    assert unit_attacks(g, back(0)) == [front(0)]
    assert unit_attacks(g, back(1)) == []
    g.step(attack(back(0), front(0)))
    assert U(g, e).hp == 2


def test_blitz_unit_can_move_on_its_deploy_turn():
    r = Rules([unit("charger", 2, 2, traits={"blitz": True}), unit("rookie", 2, 2)])
    g = blank(r)
    set_hand(g, r, 0, ["charger", "rookie"])
    g.step(play(hand_slot(g, r, 0, "charger")))
    g.step(play(hand_slot(g, r, 0, "rookie")))
    legal = g.legal_actions()
    assert move(0) in legal and move(1) not in legal


def test_summoned_blitz_token_can_act():
    # SPEC 2.4: blitz units may act on the turn they are summoned; SPEC 2.10 summon.
    r = Rules([unit("drone", 1, 1, nature="ranged", token=True, traits={"blitz": True}),
               operation("deploy_drone", [effect("on_play", "summon", "controller", card="drone")])])
    g = blank(r)
    cast(g, r, "deploy_drone")
    assert BASE in unit_attacks(g, back(0))
    g.step(attack(back(0), BASE))
    assert bases_of(g) == [20, 19]


def test_blitz_granted_on_deploy_lets_the_unit_act():
    r = Rules([unit("inspired", 2, 2, nature="ranged", effects=[effect("on_deploy", "add_trait", "self", trait="blitz")])])
    g = blank(r)
    cast(g, r, "inspired")
    assert BASE in unit_attacks(g, back(0))


# ================================================================= fury (SPEC 2.4)
def test_fury_frontline_troop_attacks_twice():
    # SPEC 2.4: max_attacks = 2 if fury; each attack is an ATTACK action.
    r = Rules([unit("berserker", 1, 9, traits={"fury": True})])
    g = blank(r)
    b = put(g, r, 0, "front", "berserker")
    put(g, r, 1, "back", "fill03")
    put(g, r, 1, "back", "fill03")
    g.step(attack(front(0), back(0)))
    x = U(g, b)
    assert x.attacks == 1 and x.can_attack() and not x.can_move()
    assert g.observe(0).frontline[0].attacks == 1 and g.observe(0).frontline[0].fury
    assert unit_attacks(g, front(0)) == [back(0), back(1), BASE]
    g.step(attack(front(0), BASE))
    x = U(g, b)
    assert x.attacks == 2 and not x.can_attack()
    assert unit_attacks(g, front(0)) == []
    assert bases_of(g) == [20, 19]


def test_fury_troop_cannot_move_after_attacking():
    # SPEC 2.4 troop: can_move = ready and not moved and attacks == 0.
    r = Rules([unit("berserker", 3, 9, traits={"fury": True})])
    g = blank(r)
    b = put(g, r, 0, "back", "berserker")
    put(g, r, 1, "front", "fill00")
    g.step(attack(back(0), front(0)))  # kills the last enemy frontline unit
    assert g.frontline == [] and g.front_owner is None
    x = U(g, b)
    assert x.attacks == 1 and not x.can_move() and x.can_attack()
    assert move(0) not in g.legal_actions()


def test_fury_ranged_attacks_twice_and_cannot_move():
    r = Rules([unit("gunner", 1, 9, nature="ranged", traits={"fury": True})])
    g = blank(r)
    put(g, r, 0, "back", "gunner")
    g.step(attack(back(0), BASE))
    assert move(0) not in g.legal_actions() and BASE in unit_attacks(g, back(0))
    g.step(attack(back(0), BASE))
    assert unit_attacks(g, back(0)) == [] and bases_of(g) == [20, 18]


def test_fury_fast_can_move_between_attacks():
    # SPEC 2.4 fast: can_move = ready and not moved; can_attack = ready and attacks < max_attacks.
    r = Rules([unit("raider", 2, 9, nature="fast", traits={"fury": True})])
    g = blank(r)
    rd = put(g, r, 0, "back", "raider")
    put(g, r, 1, "front", "fill00")
    g.step(attack(back(0), front(0)))
    assert g.frontline == [] and move(0) in g.legal_actions()
    g.step(move(0))
    assert g.frontline and g.frontline[0].uid == rd.uid
    assert BASE in unit_attacks(g, front(0))
    g.step(attack(front(0), BASE))
    assert unit_attacks(g, front(0)) == [] and bases_of(g) == [20, 18]


def test_troop_without_fury_attacks_once():
    r = Rules([unit("grunt", 1, 9)])
    g = blank(r)
    put(g, r, 0, "front", "grunt")
    g.step(attack(front(0), BASE))
    assert unit_attacks(g, front(0)) == []
    assert g.observe(0).frontline[0].attacked  # SPEC 2.4: attacked = attacks > 0


def test_fury_attack_counter_refreshes_at_the_owners_end_turn():
    # SPEC 2.3 END_TURN 4: attacks = 0.
    r = Rules([unit("berserker", 1, 9, traits={"fury": True})])
    g = blank(r)
    b = put(g, r, 0, "front", "berserker")
    g.step(attack(front(0), BASE))
    g.step(attack(front(0), BASE))
    g.step(END)
    assert U(g, b).attacks == 0
    g.step(END)
    assert unit_attacks(g, front(0)) == [BASE]


# ================================================================= smokescreen and Defense (SPEC 2.4, 2.6)
SMOKE_CARDS = [unit("shade", 1, 3, traits={"smokescreen": True}),
               unit("veil_wall", 1, 3, traits={"smokescreen": True, "defense": True}),
               unit("wall", 1, 3, traits={"defense": True}),
               unit("sniper", 1, 9, nature="ranged"),
               unit("ghost", 1, 3, nature="ranged", traits={"smokescreen": True})]


def test_smokescreen_units_cannot_be_attacked():
    # SPEC 2.4: enemy ATTACKs cannot target it; SPEC 2.6 smokescreened targets are excluded.
    r = Rules(SMOKE_CARDS)
    g = blank(r)
    put(g, r, 0, "back", "sniper")
    put(g, r, 0, "front", "fill03")
    put(g, r, 1, "back", "shade")
    put(g, r, 1, "back", "fill03")
    assert unit_attacks(g, back(0)) == [back(1), BASE]
    assert unit_attacks(g, front(0)) == [back(1), BASE]


def test_smokescreen_only_unit_leaves_nothing_to_attack_in_its_zone():
    r = Rules(SMOKE_CARDS)
    g = blank(r)
    put(g, r, 0, "back", "fill03")
    put(g, r, 1, "front", "shade")
    assert unit_attacks(g, back(0)) == []


def test_smokescreen_is_lost_when_the_unit_attacks():
    # SPEC 2.6 ATTACK 1: the attacker loses smokescreen.
    r = Rules(SMOKE_CARDS)
    g = blank(r)
    gh = put(g, r, 0, "back", "ghost")
    g.step(attack(back(0), BASE))
    assert not U(g, gh).smokescreen
    g.step(END)  # now p1 can target it
    put(g, r, 1, "back", "sniper")
    assert back(0) in unit_attacks(g, back(0))


def test_smokescreen_is_lost_when_the_unit_moves():
    # SPEC 2.6 MOVE: the unit loses smokescreen.
    r = Rules(SMOKE_CARDS)
    g = blank(r)
    s = put(g, r, 0, "back", "shade")
    g.step(move(0))
    assert not U(g, s).smokescreen
    assert not g.observe(0).frontline[0].smokescreen


@pytest.mark.parametrize("zone, expected", [
    (["veil_wall", "fill03"], [back(1), BASE]),          # the only Defense unit is not attackable
    (["wall", "shade"], [back(0), BASE]),                # Defense restricts; the shade is excluded anyway
    (["veil_wall", "wall", "fill03"], [back(1), BASE]),  # the attackable Defense unit must be chosen
    (["veil_wall", "shade"], [BASE]),
])
def test_defense_counts_only_attackable_units(zone, expected):
    # SPEC 2.4 Defense: if the targeted zone holds an attackable (non-smokescreen) Defense unit, the
    # attack must target one.
    r = Rules(SMOKE_CARDS)
    g = blank(r)
    put(g, r, 0, "back", "sniper")
    for cid in zone:
        put(g, r, 1, "back", cid)
    assert unit_attacks(g, back(0)) == expected


def test_effects_reach_smokescreen_units():
    # SPEC 2.4: effects can still target smokescreen units.
    r = Rules(SMOKE_CARDS + [operation("barrage", [effect("on_play", "damage", tgt("all", "enemy"), amount=1)]),
                             operation("snipe", [effect("on_play", "damage", tgt("random", "enemy"), amount=1)])])
    g = blank(r)
    s = put(g, r, 1, "back", "shade")
    cast(g, r, "barrage")
    assert U(g, s).hp == 2 and U(g, s).smokescreen
    cast(g, r, "snipe")  # the only candidate
    assert U(g, s).hp == 1


# ================================================================= armor, effect damage, Defense vs effects
def test_armor_reduces_combat_damage_both_ways():
    # SPEC 2.4 Armor X: reduces combat damage taken, attacking and defending; SPEC 2.6 combat_damage.
    r = Rules([unit("knight", 3, 6, traits={"armor": 2}), unit("brute", 4, 6, traits={"armor": 1})])
    g = blank(r)
    k = put(g, r, 0, "front", "knight")
    b = put(g, r, 1, "back", "brute")
    g.step(attack(front(0), back(0)))
    assert (U(g, k).hp, U(g, b).hp) == (4, 4)


def test_effect_damage_ignores_armor():
    # SPEC 2.4 / 2.10: effect damage is not reduced by armor.
    r = Rules([unit("bastion", 1, 6, traits={"armor": 3}),
               operation("shell", [effect("on_play", "damage", tgt("all", "enemy"), amount=2)]),
               unit("lancer", 1, 9, effects=[effect("on_attack", "damage", "event", amount=2)])])
    g = blank(r)
    b = put(g, r, 1, "back", "bastion")
    cast(g, r, "shell")
    assert U(g, b).hp == 4
    put(g, r, 0, "front", "lancer")
    g.step(attack(front(0), back(0)))  # 2 effect damage, then 1 combat damage fully absorbed
    assert U(g, b).hp == 2


def test_effects_ignore_defense():
    # SPEC 2.4: effects ignore Defense (random candidates include every unit, in board order).
    r = Rules([unit("wall", 1, 4, traits={"defense": True}),
               operation("snipe", [effect("on_play", "damage", tgt("random", "enemy"), amount=1)])])
    hits = set()
    for seed in range(10):
        g = blank(r)
        g.rng.seed(seed)
        put(g, r, 1, "back", "wall")
        put(g, r, 1, "back", "fill03")
        put(g, r, 1, "back", "fill03")
        cands = uids(g.backline[1])
        state = g.rng.getstate()
        cast(g, r, "snipe")
        [pick] = predict_sample(state, cands, 1)
        for u in g.backline[1]:
            assert u.hp == (3 if u.uid == pick else 4)
        hits.add(cands.index(pick))
    assert hits - {0}  # a non-Defense unit was hit at least once


# ================================================================= pin (SPEC 2.4, 2.3 END_TURN 5)
def pin_rules():
    return Rules([unit("anchor", 1, 3, effects=[effect("on_deploy", "pin", "self")]),
                  unit("charger", 2, 2, traits={"blitz": True}),
                  operation("net", [effect("on_play", "pin", tgt("all", "enemy"))]),
                  operation("tether", [effect("on_play", "pin", tgt("all", "friendly"))])])


def test_pin_on_own_turn_lasts_through_the_owners_next_turn():
    # SPEC 2.4 pin: pin_until = turn + 2 when pinned during the owner's own turn; lifted at END_TURN
    # of turn pin_until.
    r = pin_rules()
    g = blank(r)
    t0 = g.turn
    u = put(g, r, 0, "back", "fill03")
    cast(g, r, "tether")
    x = U(g, u)
    assert x.pinned and x.pin_until == t0 + 2
    assert not x.can_move() and not x.can_attack() and move(0) not in g.legal_actions()
    g.step(END)
    g.step(END)
    assert g.turn == t0 + 2 and g.current == 0
    assert U(g, u).pinned and move(0) not in g.legal_actions()
    view = g.observe(0).my_backline[0]
    assert view.pinned and not view.can_move and not view.can_attack
    g.step(END)  # END_TURN of turn pin_until
    assert not U(g, u).pinned
    g.step(END)
    assert move(0) in g.legal_actions()


def test_pin_on_the_opponents_turn_lasts_through_the_owners_next_turn():
    # SPEC 2.4: pin_until = turn + 1 if pinned during the opponent's turn.
    r = pin_rules()
    g = blank(r, current=1, first=0)
    t0 = g.turn
    u = put(g, r, 0, "back", "fill03")
    cast(g, r, "net")
    assert U(g, u).pinned and U(g, u).pin_until == t0 + 1
    g.step(END)
    assert g.current == 0 and g.turn == t0 + 1
    assert U(g, u).pinned and move(0) not in g.legal_actions()
    g.step(END)
    assert not U(g, u).pinned
    g.step(END)
    assert move(0) in g.legal_actions()


def test_pin_on_the_enemy_during_own_turn():
    r = pin_rules()
    g = blank(r)
    t0 = g.turn
    e = put(g, r, 1, "back", "fill03")
    cast(g, r, "net")
    assert U(g, e).pin_until == t0 + 1
    g.step(END)
    assert g.current == 1 and U(g, e).pinned and move(0) not in g.legal_actions()
    g.step(END)
    assert not U(g, e).pinned


def test_repinning_refreshes_pin_until():
    # SPEC 2.4: pinning a pinned unit refreshes pin_until.
    r = pin_rules()
    g = blank(r)
    t0 = g.turn
    u = put(g, r, 0, "back", "fill03")
    cast(g, r, "tether")
    g.step(END)
    g.step(END)
    cast(g, r, "tether")  # turn t0 + 2
    assert U(g, u).pin_until == t0 + 4
    g.step(END)           # END_TURN of t0 + 2: the old expiry no longer applies
    assert U(g, u).pinned
    g.step(END)           # END_TURN of t0 + 3
    assert U(g, u).pinned and g.turn == t0 + 4
    g.step(END)           # END_TURN of t0 + 4 = pin_until
    assert not U(g, u).pinned


def test_on_deploy_pin_self():
    r = pin_rules()
    g = blank(r)
    t0 = g.turn
    cast(g, r, "anchor")
    a = g.backline[0][0]
    assert a.pinned and a.pin_until == t0 + 2


def test_pinned_blitz_unit_cannot_act():
    # SPEC 2.4: ready = not pinned and (not summoned or blitz).
    r = Rules([unit("charger", 2, 2, traits={"blitz": True},
                    effects=[effect("on_deploy", "pin", "self")])])
    g = blank(r)
    cast(g, r, "charger")
    assert move(0) not in g.legal_actions() and unit_attacks(g, back(0)) == []


# ================================================================= "turn" durations (SPEC 2.3, 2.10)
def test_turn_buff_expires_at_end_turn_and_expiry_never_kills():
    # SPEC 2.10 buff "turn": temp_atk += a, temp_hp += h; at expiry atk -= temp_atk, max_hp -= temp_hp,
    # hp = min(hp, max_hp) but at least 1.
    r = Rules([unit("cadet", 2, 2),
               operation("rally", [effect("on_play", "buff", tgt("all", "friendly"), atk=1, hp=3, duration="turn")]),
               operation("burn_back", [effect("on_play", "damage", tgt("all", "friendly", zone="backline"), amount=4)]),
               operation("burn_front", [effect("on_play", "damage", tgt("all", "friendly", zone="frontline"),
                                               amount=1)])])
    g = blank(r)
    a = put(g, r, 0, "back", "cadet")
    b = put(g, r, 0, "front", "cadet")
    cast(g, r, "rally")
    for u in (a, b):
        x = U(g, u)
        assert (x.atk, x.hp, x.max_hp, x.temp_atk, x.temp_hp) == (3, 5, 5, 1, 3)
    assert (g.observe(0).my_backline[0].temp_atk, g.observe(0).my_backline[0].temp_hp) == (1, 3)
    cast(g, r, "burn_back")
    cast(g, r, "burn_front")
    assert (U(g, a).hp, U(g, b).hp) == (1, 4)
    g.step(END)
    xa, xb = U(g, a), U(g, b)
    assert (xa.atk, xa.hp, xa.max_hp) == (2, 1, 2)  # would be dead with hp -= temp_hp
    assert (xb.atk, xb.hp, xb.max_hp) == (2, 2, 2)
    assert (xa.temp_atk, xa.temp_hp, xb.temp_atk, xb.temp_hp) == (0, 0, 0, 0)


def test_turn_buff_records_the_atk_change_actually_applied():
    # SPEC 2.10: "turn" records the change actually applied: temp_atk += (new atk - old atk) after the
    # clamp atk = max(0, atk + a); at expiry atk -= temp_atk.
    r = Rules([unit("weak", 1, 3), unit("feeble", 0, 3), unit("cadet", 2, 3),
               operation("sap", [effect("on_play", "buff", tgt("all", "friendly"), atk=-3, duration="turn")]),
               operation("rally", [effect("on_play", "buff", tgt("all", "friendly"), atk=2, duration="turn")])])
    g = blank(r)
    w = put(g, r, 0, "back", "weak")
    f = put(g, r, 0, "back", "feeble")
    c = put(g, r, 0, "back", "cadet")
    cast(g, r, "sap")
    assert [(U(g, u).atk, U(g, u).temp_atk) for u in (w, f, c)] == [(0, -1), (0, 0), (0, -2)]
    assert [v.temp_atk for v in g.observe(0).my_backline] == [-1, 0, -2]
    cast(g, r, "rally")  # applied +2 each: (2, 1), (2, 2), (2, 0)
    assert [(U(g, u).atk, U(g, u).temp_atk) for u in (w, f, c)] == [(2, 1), (2, 2), (2, 0)]
    g.step(END)
    assert [(U(g, u).atk, U(g, u).temp_atk) for u in (w, f, c)] == [(1, 0), (0, 0), (2, 0)]


def test_turn_buff_clamped_then_expired_restores_the_printed_attack():
    # SPEC 2.10 clamp case alone: atk 1, "turn" -3 -> atk 0 with temp_atk -1; expiry -> atk 1 (not 3).
    r = Rules([operation("sap", [effect("on_play", "buff", tgt("all", "enemy"), atk=-3, duration="turn")])])
    g = blank(r)
    e = put(g, r, 1, "back", "fill00")  # 1/1
    cast(g, r, "sap")
    assert (U(g, e).atk, U(g, e).temp_atk) == (0, -1)
    g.step(END)
    assert (U(g, e).atk, U(g, e).temp_atk) == (1, 0)


def test_turn_buff_lasts_for_the_rest_of_the_turn():
    r = Rules([operation("rally", [effect("on_play", "buff", tgt("all", "friendly"), atk=2, hp=0, duration="turn")])])
    g = blank(r)
    u = put(g, r, 0, "front", "fill03")
    cast(g, r, "rally")
    g.step(attack(front(0), BASE))
    assert bases_of(g) == [20, 17]


def test_turn_buff_on_enemy_units_expires_at_the_casters_end_turn():
    # SPEC 2.3 END_TURN 2: every "turn" duration on every unit expires.
    r = Rules([operation("weaken", [effect("on_play", "buff", tgt("all", "enemy"), atk=-2, hp=0, duration="turn")])])
    g = blank(r)
    e = put(g, r, 1, "back", "fill02")  # 3/3
    cast(g, r, "weaken")
    assert U(g, e).atk == 1
    g.step(END)
    assert U(g, e).atk == 3


def test_end_of_turn_triggers_see_turn_values_before_expiry():
    # SPEC 2.3 END_TURN: 1. end_of_turn triggers, 2. "turn" durations expire.
    r = Rules([unit("herald", 2, 2, effects=[effect("end_of_turn", "damage", "enemy_base",
                                                    amount={"stat": "atk", "of": "self"})]),
               operation("rally", [effect("on_play", "buff", tgt("all", "friendly"), atk=3, hp=0, duration="turn")])])
    g = blank(r)
    h = put(g, r, 0, "back", "herald")
    cast(g, r, "rally")
    g.step(END)
    assert bases_of(g) == [20, 15] and U(g, h).atk == 2


@pytest.mark.parametrize("trait", ["defense", "blitz", "smokescreen", "fury"])
def test_turn_trait_grant_expires(trait):
    # SPEC 2.10 add_trait "turn": undone at expiry.
    r = Rules([operation("gift", [effect("on_play", "add_trait", tgt("all", "friendly"), trait=trait,
                                         duration="turn")])])
    g = blank(r)
    u = put(g, r, 0, "back", "fill03")
    cast(g, r, "gift")
    assert getattr(U(g, u), trait)
    g.step(END)
    assert not getattr(U(g, u), trait)


def test_turn_armor_grant_expires():
    r = Rules([unit("plated", 1, 4, traits={"armor": 1}),
               operation("plate", [effect("on_play", "add_trait", tgt("all", "friendly"), trait="armor", amount=2,
                                          duration="turn")])])
    g = blank(r)
    u = put(g, r, 0, "back", "plated")
    cast(g, r, "plate")
    assert U(g, u).armor == 3
    g.step(END)
    assert U(g, u).armor == 1


def test_turn_grant_of_a_trait_the_card_already_has_keeps_it():
    # SPEC 2.10: undoing a "turn" grant restores what the unit had (a printed trait stays).
    r = Rules([unit("wall", 1, 4, traits={"defense": True}),
               operation("gift", [effect("on_play", "add_trait", tgt("all", "friendly"), trait="defense",
                                         duration="turn")])])
    g = blank(r)
    u = put(g, r, 0, "back", "wall")
    cast(g, r, "gift")
    g.step(END)
    assert U(g, u).defense


def test_turn_grant_then_permanent_grant_keeps_the_trait():
    # SPEC 2.10: "turn" grants are undone at expiry, unless the trait was also granted permanently.
    r = Rules([operation("gift_turn", [effect("on_play", "add_trait", tgt("all", "friendly"), trait="defense",
                                              duration="turn")]),
               operation("gift_perm", [effect("on_play", "add_trait", tgt("all", "friendly"), trait="defense")])])
    g = blank(r)
    u = put(g, r, 0, "back", "fill03")
    cast(g, r, "gift_turn")
    cast(g, r, "gift_perm")
    g.step(END)
    assert U(g, u).defense


@pytest.mark.parametrize("trait", ["defense", "blitz", "smokescreen", "fury", "armor"])
def test_turn_trait_removal_is_restored(trait):
    # SPEC 2.10 remove_trait "turn": restored at expiry.
    r = Rules([unit("decorated", 1, 4, traits={"defense": True, "blitz": True, "smokescreen": True, "fury": True,
                                                "armor": 2}),
               operation("strip", [effect("on_play", "remove_trait", tgt("all", "friendly"), trait=trait,
                                          duration="turn")])])
    g = blank(r)
    u = put(g, r, 0, "back", "decorated")
    cast(g, r, "strip")
    x = U(g, u)
    assert (x.armor == 0) if trait == "armor" else not getattr(x, trait)
    g.step(END)
    x = U(g, u)
    assert (x.armor == 2) if trait == "armor" else getattr(x, trait)


def test_turn_removal_then_permanent_grant_keeps_the_trait():
    # SPEC 2.10: at expiry the unit has the traits it would have had.
    r = Rules([unit("wall", 1, 4, traits={"defense": True}),
               operation("strip", [effect("on_play", "remove_trait", tgt("all", "friendly"), trait="defense",
                                          duration="turn")]),
               operation("gift", [effect("on_play", "add_trait", tgt("all", "friendly"), trait="defense")])])
    g = blank(r)
    u = put(g, r, 0, "back", "wall")
    cast(g, r, "strip")
    cast(g, r, "gift")
    assert U(g, u).defense
    g.step(END)
    assert U(g, u).defense


def test_turn_removal_of_smokescreen_is_not_restored_after_attacking():
    # SPEC 2.10: "the unit then has the traits it would have had" -- it would have lost smokescreen
    # by attacking (SPEC 2.6 ATTACK 1), so expiry does not bring it back.
    r = Rules(SMOKE_CARDS + [operation("strip", [effect("on_play", "remove_trait", tgt("all", "friendly"),
                                                        trait="smokescreen", duration="turn")])])
    g = blank(r)
    gh = put(g, r, 0, "back", "ghost")
    cast(g, r, "strip")
    g.step(attack(back(0), BASE))
    g.step(END)
    assert not U(g, gh).smokescreen


def test_turn_fury_allows_two_attacks_this_turn_only():
    r = Rules([operation("frenzy", [effect("on_play", "add_trait", tgt("all", "friendly"), trait="fury",
                                           duration="turn")])])
    g = blank(r)
    put(g, r, 0, "front", "fill03")
    cast(g, r, "frenzy")
    g.step(attack(front(0), BASE))
    g.step(attack(front(0), BASE))
    assert bases_of(g) == [20, 18]
    g.step(END)
    g.step(END)
    g.step(attack(front(0), BASE))
    assert unit_attacks(g, front(0)) == []
