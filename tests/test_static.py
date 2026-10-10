"""Independent tests of static (continuous) effects (SPEC 1.2b `static`, SPEC 2.12; SPEC 2.11 "Static
effects").

SPEC 2.12: after every action, every resolved instance and every damage step the engine recomputes
all static contributions from scratch (static effects of units on the board whose condition holds) and
applies the deltas: atk; max_hp; hp (an increase raises hp, a decrease caps hp at the new max_hp,
never below 1); move_cost; traits = base | temp | static. Static smokescreen is not lost by moving or
attacking while the source still grants it.

Hand-built positions do not trigger a recompute by themselves (SPEC 2.12 lists the moments), so every
check here follows an action: the source is deployed by PLAY, or a no-op operation ("tick") is played.
"""
from __future__ import annotations

import pytest

from stage3_helpers import (BASE, END, MAIN, Rules, adjacent, attack, back, bases_of, blank, choose, effect, find,
                            front, hand_slot, move, operation, own_back, play, put, set_hand, state_key, static, tgt, unit,
                            zone_of)


# ---------------------------------------------------------------- local helpers
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


def stats(g, u):
    x = U(g, u)
    return (x.atk, x.hp, x.max_hp)


def contrib(g, u):
    """SPEC 2.12 / SPEC 4 unit fields static_atk, static_hp, static_move_cost."""
    x = U(g, u)
    return (x.static_atk, x.static_hp, x.static_move_cost)


def view(g, viewer, u):
    obs = g.observe(viewer)
    owner = U(g, u).owner
    zone, _ = zone_of(g, u.uid)
    if zone == "front":
        return obs.frontline[[x.uid for x in g.frontline].index(u.uid)]
    idx = [x.uid for x in g.backline[owner]].index(u.uid)
    return (obs.my_backline if owner == viewer else obs.opp_backline)[idx]


def unit_attacks(g, slot):
    return [t for t in range(11) if attack(slot, t) in g.legal_actions()]


def newest(g, p):
    return g.backline[p][-1]


OTHERS = {"other": True}
LEADER = {"tag": "leader"}
TICK = operation("tick", [effect("on_play", "gain_coins", "controller", amount=0)], cost=0)
PURGE = operation("purge", [effect("on_play", "destroy", tgt("all", "any", filter=LEADER))], cost=0)
RECALL = operation("recall", [effect("on_play", "return_to_hand", tgt("all", "any", filter=LEADER))], cost=0)
WITHDRAW = operation("withdraw", [effect("on_play", "retreat", tgt("all", "any", zone="backline", filter=LEADER))],
                     cost=0)


def banner(cid="banner", atk=1, hp=1, *, flt=OTHERS, side="friendly", condition=None, body=(0, 3), token=False,
           tags=("leader",)):
    """A unit whose static effect buffs `side` units matching `flt` (default: the other friendly units)."""
    params = {}
    if atk:
        params["atk"] = atk
    if hp:
        params["hp"] = hp
    return unit(cid, body[0], body[1], token=token, tags=list(tags),
                effects=[static("buff", tgt("all", side, filter=flt), condition=condition, **params)])


# ================================================================= arrive / leave (SPEC 2.12)
@pytest.mark.parametrize("leave", ["purge", "recall", "withdraw"])
def test_aura_applies_on_arrival_and_is_removed_when_the_source_leaves(leave):
    # SPEC 1.2b static: active while the source is on the board; SPEC 2.12 recompute after every action /
    # damage step (the fast path must still clear the contributions of the last source that left).
    r = Rules([banner(), unit("grunt", 2, 2), PURGE, RECALL, WITHDRAW])
    g = blank(r)
    a = put(g, r, 0, "back", "grunt")
    e = put(g, r, 1, "back", "grunt")
    cast(g, r, "banner")
    b = newest(g, 0)
    assert stats(g, a) == (3, 3, 3) and contrib(g, a) == (1, 1, 0)
    assert view(g, 0, a)[1:4] == (3, 3, 3) and view(g, 1, a)[1:4] == (3, 3, 3)
    assert stats(g, b) == (0, 3, 3) and contrib(g, b) == (0, 0, 0)   # "other": not the source
    assert stats(g, e) == (2, 2, 2) and contrib(g, e) == (0, 0, 0)   # friendly only
    cast(g, r, leave)
    assert find(g, b.uid) is None
    assert stats(g, a) == (2, 2, 2) and contrib(g, a) == (0, 0, 0)
    assert view(g, 1, a)[1:4] == (2, 2, 2)


def test_aura_from_a_summoned_source_and_on_units_arriving_later():
    # SPEC 2.12: recompute after every resolved instance (summon) and after every action (PLAY): units
    # that arrive while the aura is active get it at once (an hp increase raises hp).
    r = Rules([banner("banner_tok", token=True), unit("grunt", 2, 2), unit("militia", 1, 1, token=True),
               operation("raise_banner", [effect("on_play", "summon", "controller", card="banner_tok")]),
               operation("muster", [effect("on_play", "summon", "controller", card="militia")])])
    g = blank(r)
    cast(g, r, "raise_banner")
    cast(g, r, "grunt")
    gr = newest(g, 0)
    assert stats(g, gr) == (3, 3, 3)
    cast(g, r, "muster")
    assert stats(g, newest(g, 0)) == (2, 2, 2)


def test_aura_source_killed_in_combat():
    # SPEC 2.12: recompute after each damage step (here the combat's).
    r = Rules([banner(), unit("grunt", 2, 2), unit("ogre", 5, 9)])
    g = blank(r, current=1, first=0)
    gr = put(g, r, 1, "back", "grunt")
    cast(g, r, "banner")
    assert stats(g, gr) == (3, 3, 3)
    g.step(END)
    put(g, r, 0, "front", "ogre")
    g.step(attack(front(0), back(1)))
    assert len(g.backline[1]) == 1 and stats(g, gr) == (2, 2, 2)


def test_a_pinned_source_still_grants_its_aura():
    # SPEC 1.2b: active while the source is on the board (pinned units are on the board).
    r = Rules([banner(), unit("grunt", 2, 2), operation("net", [effect("on_play", "pin", tgt("all", "any"))])])
    g = blank(r)
    gr = put(g, r, 0, "back", "grunt")
    cast(g, r, "banner")
    cast(g, r, "net")
    assert U(g, newest(g, 0)).pinned and stats(g, gr) == (3, 3, 3)


# ================================================================= hp auras (SPEC 2.12)
def test_hp_aura_raises_hp_and_losing_it_caps_hp_without_killing():
    # SPEC 2.12: an increase raises hp; a decrease caps hp at the new max_hp, never below 1.
    r = Rules([banner(atk=0, hp=2), unit("wisp", 1, 1, tags=["w"]), unit("ogre", 2, 3), unit("brute", 2, 3, tags=["b"]),
               operation("zap_w", [effect("on_play", "damage", tgt("all", "any", filter={"tag": "w"}), amount=2)]),
               operation("zap_b", [effect("on_play", "damage", tgt("all", "any", filter={"tag": "b"}), amount=3)]),
               PURGE])
    g = blank(r)
    w = put(g, r, 0, "back", "wisp")
    o = put(g, r, 0, "back", "ogre")
    bu = put(g, r, 0, "back", "brute")
    cast(g, r, "banner")
    assert [stats(g, u) for u in (w, o, bu)] == [(1, 3, 3), (2, 5, 5), (2, 5, 5)]
    cast(g, r, "zap_w")
    cast(g, r, "zap_b")
    assert [stats(g, u) for u in (w, o, bu)] == [(1, 1, 3), (2, 5, 5), (2, 2, 5)]
    cast(g, r, "purge")
    assert [stats(g, u) for u in (w, o, bu)] == [(1, 1, 1), (2, 3, 3), (2, 2, 3)]
    assert [contrib(g, u) for u in (w, o, bu)] == [(0, 0, 0)] * 3


def test_hp_aura_lost_in_the_same_damage_step_does_not_kill():
    # SPEC 2.8 damage step: dead units (hp <= 0) are removed, then (SPEC 2.12) the recompute caps the
    # survivors' hp at the new max_hp but never below 1.
    r = Rules([banner(atk=0, hp=2, body=(0, 2)), unit("wisp", 1, 1), unit("ogre", 2, 3),
               operation("quake", [effect("on_play", "damage", tgt("all", "friendly"), amount=2)])])
    g = blank(r)
    w = put(g, r, 0, "back", "wisp")
    o = put(g, r, 0, "back", "ogre")
    cast(g, r, "banner")
    cast(g, r, "quake")
    assert len(g.backline[0]) == 2
    assert [stats(g, u) for u in (w, o)] == [(1, 1, 1), (2, 3, 3)]


# ================================================================= conditions toggling (SPEC 1.2b, 2.12)
def test_while_damaged_aura_via_a_self_filter():
    # SPEC 1.2: self targets may take a filter (no match -> nothing); SPEC 2.12: re-evaluated each recompute.
    r = Rules([unit("berserker", 1, 4, effects=[static("buff", tgt("self", filter={"damaged": True}), atk=2)]),
               operation("jab", [effect("on_play", "damage", tgt("all", "friendly"), amount=1)]),
               operation("mend", [effect("on_play", "heal", tgt("all", "friendly"), amount="full")]), TICK])
    g = blank(r)
    b = put(g, r, 0, "back", "berserker")
    cast(g, r, "tick")
    assert stats(g, b) == (1, 4, 4) and contrib(g, b) == (0, 0, 0)
    cast(g, r, "jab")
    assert stats(g, b) == (3, 3, 4) and contrib(g, b) == (2, 0, 0)
    cast(g, r, "mend")
    assert stats(g, b) == (1, 4, 4)


def test_during_your_turn_aura_toggles_with_the_turn():
    # SPEC 1.2b static condition ({"type": "turn", "whose": "own"}); SPEC 2.12 recompute after every action.
    r = Rules([unit("sentry", 1, 4, effects=[static("buff", "self", atk=2, condition={"type": "turn", "whose": "own"})]),
               TICK])
    g = blank(r)
    s = put(g, r, 0, "back", "sentry")
    cast(g, r, "tick")
    assert U(g, s).atk == 3
    g.step(END)
    assert U(g, s).atk == 1 and view(g, 1, s).atk == 1
    g.step(END)
    assert U(g, s).atk == 3


def test_control_condition_toggles_the_aura():
    r = Rules([unit("ace", 1, 4, effects=[static("buff", "self", atk=2, condition={
        "type": "control", "side": "friendly", "filter": {"tag": "tank", "other": True}})]),
        unit("panzer", 2, 4, tags=["tank", "leader"]), PURGE])
    g = blank(r)
    cast(g, r, "ace")
    ace = newest(g, 0)
    assert U(g, ace).atk == 1  # the ace itself is not "other"
    cast(g, r, "panzer")
    assert U(g, ace).atk == 3
    cast(g, r, "purge")
    assert U(g, ace).atk == 1


def test_frontline_condition_toggles_the_aura():
    r = Rules([banner("drummer", atk=1, hp=0, body=(0, 4), condition={"type": "frontline", "owner": "friendly"}),
               unit("grunt", 2, 2), TICK])
    g = blank(r)
    put(g, r, 0, "back", "drummer")
    gr = put(g, r, 0, "back", "grunt")
    cast(g, r, "tick")
    assert stats(g, gr) == (2, 2, 2)
    g.step(move(1))
    assert zone_of(g, gr.uid) == ("front", 0) and stats(g, gr) == (3, 2, 2)


def test_aura_condition_is_rechecked_after_every_action():
    # SPEC 2.12: recompute after every action (here a PLAY that empties the hand below the threshold).
    r = Rules([unit("scholar", 1, 4, effects=[static("buff", "self", atk=2, condition={
        "type": "hand_size", "side": "friendly", "min": 2})]), TICK])
    g = blank(r)
    s = put(g, r, 0, "back", "scholar")
    set_hand(g, r, 0, ["fill00", "fill01"])
    cast(g, r, "tick")
    assert U(g, s).atk == 3
    g.step(play(0))
    assert len(g.hands[0]) == 1 and U(g, s).atk == 1


# ================================================================= stacking (SPEC 2.12)
def test_two_auras_stack_and_losing_one_keeps_the_other():
    # SPEC 2.12: every static effect of every unit adds to its targets.
    r = Rules([banner(), unit("wisp", 1, 1),
               operation("snipe", [effect("on_play", "destroy", tgt("chosen", "friendly"))], cost=0)])
    g = blank(r)
    w = put(g, r, 0, "back", "wisp")
    cast(g, r, "banner")
    b1 = newest(g, 0)
    cast(g, r, "banner")
    b2 = newest(g, 0)
    assert stats(g, w) == (3, 3, 3) and contrib(g, w) == (2, 2, 0)
    assert stats(g, b1) == (1, 4, 4) and stats(g, b2) == (1, 4, 4)
    cast(g, r, "snipe")
    g.step(choose(own_back(1)))  # b1
    assert find(g, b1.uid) is None
    assert stats(g, w) == (2, 2, 2) and stats(g, b2) == (0, 3, 3)


def test_self_static_and_aura_stack():
    r = Rules([banner(), unit("proud", 1, 2, effects=[static("buff", "self", atk=1)])])
    g = blank(r)
    cast(g, r, "proud")
    p = newest(g, 0)
    assert stats(g, p) == (2, 2, 2)
    cast(g, r, "banner")
    assert stats(g, p) == (3, 3, 3) and contrib(g, p) == (2, 1, 0)


def test_static_debuff_on_enemy_units():
    # SPEC 1.2: side is relative to the effect's controller (the source's owner).
    r = Rules([banner("gloom", atk=-1, hp=0, flt=None, side="enemy"), unit("brute", 3, 3), PURGE])
    g = blank(r)
    e = put(g, r, 1, "back", "brute")
    m = put(g, r, 0, "back", "brute")
    cast(g, r, "gloom")
    assert (U(g, e).atk, U(g, m).atk) == (2, 3) and contrib(g, e) == (-1, 0, 0)
    cast(g, r, "purge")
    assert U(g, e).atk == 3


# ================================================================= static traits (SPEC 2.12)
def test_static_defense_restricts_enemy_attacks():
    # SPEC 2.12 traits = base | temp | static; SPEC 2.4 Defense restricts ATTACK targets.
    r = Rules([unit("shieldwall", 0, 3, tags=["leader"],
                    effects=[static("add_trait", tgt("all", "friendly", filter={"tag": "guard"}), trait="defense")]),
               unit("guard", 1, 4, tags=["guard"]), unit("ogre", 1, 9), PURGE])
    g = blank(r, current=1, first=0)
    gd = put(g, r, 1, "back", "guard")
    put(g, r, 1, "back", "fill03")
    cast(g, r, "shieldwall")
    assert U(g, gd).defense and view(g, 0, gd).defense and U(g, gd).static_traits
    g.step(END)
    put(g, r, 0, "front", "ogre")
    assert unit_attacks(g, front(0)) == [back(0), BASE]
    cast(g, r, "purge")
    assert not U(g, gd).defense and not U(g, gd).static_traits
    assert unit_attacks(g, front(0)) == [back(0), back(1), BASE]


def test_static_smokescreen_is_not_lost_by_attacking_or_moving():
    # SPEC 2.12 smokescreen exception (granted by a static effect, the source still grants it).
    r = Rules([unit("phantom", 1, 4, nature="ranged", effects=[static("add_trait", "self", trait="smokescreen")]),
               unit("sniper", 1, 9, nature="ranged"), TICK])
    g = blank(r)
    ph = put(g, r, 0, "back", "phantom")
    cast(g, r, "tick")
    assert U(g, ph).smokescreen
    g.step(attack(back(0), BASE))
    assert U(g, ph).smokescreen and view(g, 1, ph).smokescreen
    g.step(END)
    put(g, r, 1, "back", "sniper")
    assert unit_attacks(g, back(0)) == [BASE]
    g = blank(r)
    ph = put(g, r, 0, "back", "phantom")
    cast(g, r, "tick")
    g.step(move(0))
    assert zone_of(g, ph.uid) == ("front", 0) and U(g, ph).smokescreen


def test_printed_smokescreen_is_lost_by_attacking_but_the_aura_grant_stays_while_granted():
    # SPEC 2.4: smokescreen is lost when the unit attacks; SPEC 2.12: not the static grant. When the
    # source leaves, nothing is left.
    r = Rules([unit("veil", 0, 3, tags=["leader"],
                    effects=[static("add_trait", tgt("all", "friendly", filter=OTHERS), trait="smokescreen")]),
               unit("ghost", 1, 3, nature="ranged", traits={"smokescreen": True}), PURGE])
    g = blank(r)
    gh = put(g, r, 0, "back", "ghost")
    cast(g, r, "veil")
    g.step(attack(back(0), BASE))
    assert U(g, gh).smokescreen
    cast(g, r, "purge")
    assert not U(g, gh).smokescreen


def test_turn_trait_grant_expires_into_the_static_grant():
    # SPEC 2.12 traits = base | temp | static; SPEC 2.10 "turn" grants expire at END_TURN.
    r = Rules([unit("warden", 0, 3, tags=["leader"],
                    effects=[static("add_trait", tgt("all", "friendly", filter=OTHERS), trait="defense")]),
               unit("grunt", 2, 2),
               operation("brace", [effect("on_play", "add_trait", tgt("all", "friendly"), trait="defense",
                                          duration="turn")]), PURGE])
    g = blank(r)
    gr = put(g, r, 0, "back", "grunt")
    cast(g, r, "warden")
    wd = newest(g, 0)
    cast(g, r, "brace")
    assert U(g, gr).defense and U(g, wd).defense
    g.step(END)
    assert U(g, gr).defense and not U(g, wd).defense
    cast(g, r, "purge")  # p1 destroys the warden
    assert not U(g, gr).defense


def test_turn_trait_grant_outlasts_a_static_grant_lost_mid_turn():
    # SPEC 2.12 traits = base | temp | static: losing the static grant keeps the "turn" grant until END_TURN.
    r = Rules([unit("warden", 0, 3, tags=["leader"],
                    effects=[static("add_trait", tgt("all", "friendly", filter=OTHERS), trait="defense")]),
               unit("grunt", 2, 2),
               operation("brace", [effect("on_play", "add_trait", tgt("all", "friendly"), trait="defense",
                                          duration="turn")]), PURGE])
    g = blank(r)
    gr = put(g, r, 0, "back", "grunt")
    cast(g, r, "warden")
    cast(g, r, "brace")
    cast(g, r, "purge")
    assert U(g, gr).defense and not U(g, gr).static_traits
    g.step(END)
    assert not U(g, gr).defense


def test_static_blitz_lets_a_newly_deployed_unit_act():
    r = Rules([unit("bugle", 0, 3, effects=[static("add_trait", tgt("all", "friendly", filter=OTHERS), trait="blitz")]),
               unit("rookie", 2, 2, nature="ranged")])
    g = blank(r)
    cast(g, r, "rookie")
    assert unit_attacks(g, back(0)) == []
    cast(g, r, "bugle")
    cast(g, r, "rookie")
    assert U(g, newest(g, 0)).blitz
    assert unit_attacks(g, back(2)) == [BASE] and unit_attacks(g, back(0)) == [BASE]
    assert unit_attacks(g, back(1)) == []  # the bugle itself is not "other"


def test_static_immune_aura_blocks_effect_damage():
    r = Rules([unit("aegis", 0, 3, effects=[static("add_trait", tgt("all", "friendly", filter=OTHERS), trait="immune")]),
               operation("barrage", [effect("on_play", "damage", tgt("all", "enemy"), amount=2)])])
    g = blank(r, current=1, first=0)
    gr = put(g, r, 1, "back", "fill03")  # 1/4
    cast(g, r, "aegis")
    g.step(END)
    cast(g, r, "barrage")
    assert U(g, gr).hp == 4 and U(g, gr).immune and view(g, 0, gr).immune
    assert U(g, newest(g, 1)).hp == 1  # the aegis itself (0/3) took 2


# ================================================================= adjacency auras (SPEC 1.2b adjacent, 2.12)
def test_adjacency_aura_follows_slot_changes():
    # SPEC 1.2b adjacent (slot +-1 in the source's zone) recomputed from scratch (SPEC 2.12) as zones
    # compact and the source moves.
    r = Rules([unit("sergeant", 0, 5, effects=[static("buff", adjacent("friendly", "self"), atk=1)]),
               unit("ua", 1, 3, tags=["a"]), unit("ub", 1, 3, tags=["b"]), unit("uc", 1, 3), unit("uf", 1, 3),
               operation("kill_a", [effect("on_play", "destroy", tgt("all", "friendly", filter={"tag": "a"}))], cost=0),
               operation("kill_b", [effect("on_play", "destroy", tgt("all", "friendly", filter={"tag": "b"}))], cost=0),
               TICK])
    g = blank(r)
    a = put(g, r, 0, "back", "ua")
    s = put(g, r, 0, "back", "sergeant")
    b = put(g, r, 0, "back", "ub")
    c = put(g, r, 0, "back", "uc")
    cast(g, r, "tick")
    assert [U(g, u).atk for u in (a, s, b, c)] == [2, 0, 2, 1]
    cast(g, r, "kill_a")
    assert [U(g, u).atk for u in (s, b, c)] == [0, 2, 1]
    cast(g, r, "kill_b")
    assert [U(g, u).atk for u in (s, c)] == [0, 2]
    f = put(g, r, 0, "front", "uf")
    g.step(move(0))
    assert [x.uid for x in g.frontline] == [f.uid, s.uid]
    assert (U(g, c).atk, U(g, f).atk) == (1, 2)


@pytest.mark.parametrize("position, expected", [("left", [2, 1]), ("right", [1, 2]), ("both", [2, 2])])
def test_adjacency_aura_positions(position, expected):
    r = Rules([unit("sergeant", 0, 5, effects=[static("buff", adjacent("friendly", "self", position), atk=1)]),
               TICK])
    g = blank(r)
    a = put(g, r, 0, "back", "fill03")
    put(g, r, 0, "back", "sergeant")
    b = put(g, r, 0, "back", "fill03")
    cast(g, r, "tick")
    assert [U(g, a).atk, U(g, b).atk] == expected


# ================================================================= recompute mid-chain (SPEC 2.12)
def test_aura_removed_mid_chain_is_gone_for_the_next_instance():
    # SPEC 2.12: recompute after every resolved instance and after each damage step.
    r = Rules([banner(atk=2, hp=0), unit("grunt", 1, 3),
               operation("decapitate", [
                   effect("on_play", "destroy", tgt("all", "enemy", filter=LEADER)),
                   effect("on_play", "damage", "enemy_base",
                          amount={"count": "units", "side": "enemy", "filter": {"min_atk": 3}})]), TICK])
    g = blank(r)
    g1 = put(g, r, 1, "back", "grunt")
    put(g, r, 1, "back", "grunt")
    put(g, r, 1, "back", "banner")
    cast(g, r, "tick")
    assert U(g, g1).atk == 3
    cast(g, r, "decapitate")
    assert len(g.backline[1]) == 2 and U(g, g1).atk == 1
    assert bases_of(g) == [20, 20]


def test_aura_lost_between_repeat_iterations():
    # SPEC 2.12: recompute after each damage step, and SPEC 1.2b repeat runs selection + action + damage
    # step per iteration, so the second iteration selects against the values without the aura.
    r = Rules([banner(atk=2, hp=0, body=(3, 1)), unit("grunt", 1, 5),
               operation("volley", [effect("on_play", "damage", tgt("all", "enemy", filter={"min_atk": 3}), amount=1,
                                           repeat=2)]), TICK])
    g = blank(r)
    gr = put(g, r, 1, "back", "grunt")
    put(g, r, 1, "back", "banner")
    cast(g, r, "tick")
    assert U(g, gr).atk == 3
    cast(g, r, "volley")
    assert len(g.backline[1]) == 1 and stats(g, gr) == (1, 4, 5)


def test_aura_added_mid_chain_counts_for_the_next_instance():
    r = Rules([banner("banner_tok", atk=2, hp=0, token=True), unit("grunt", 1, 3),
               operation("rally_cry", [
                   effect("on_play", "summon", "controller", card="banner_tok"),
                   effect("on_play", "damage", "enemy_base",
                          amount={"count": "units", "side": "friendly", "filter": {"min_atk": 3}})])])
    g = blank(r)
    put(g, r, 0, "back", "grunt")
    put(g, r, 0, "back", "grunt")
    cast(g, r, "rally_cry")
    assert bases_of(g) == [20, 18]


def test_aura_condition_toggled_mid_chain():
    r = Rules([unit("berserker", 1, 4, tags=["zerk"],
                    effects=[static("buff", tgt("self", filter={"damaged": True}), atk=2)]),
               operation("goad", [
                   effect("on_play", "damage", tgt("all", "friendly", filter={"tag": "zerk"}), amount=1),
                   effect("on_play", "damage", "enemy_base",
                          amount={"count": "units", "side": "friendly", "filter": {"min_atk": 3}})])])
    g = blank(r)
    put(g, r, 0, "back", "berserker")
    cast(g, r, "goad")
    assert bases_of(g) == [20, 19]


# ================================================================= with "turn" and permanent buffs (SPEC 2.10, 2.12)
def test_aura_with_permanent_and_turn_atk_buffs():
    r = Rules([banner(), unit("cadet", 2, 2, tags=["cadet"]),
               operation("drill", [effect("on_play", "buff", tgt("all", "friendly", filter={"tag": "cadet"}), atk=1,
                                          hp=1)]),
               operation("rally", [effect("on_play", "buff", tgt("all", "friendly", filter={"tag": "cadet"}), atk=2,
                                          duration="turn")]), PURGE])
    g = blank(r)
    c = put(g, r, 0, "back", "cadet")
    cast(g, r, "banner")
    assert stats(g, c) == (3, 3, 3)
    cast(g, r, "drill")
    assert stats(g, c) == (4, 4, 4)
    cast(g, r, "rally")
    assert stats(g, c) == (6, 4, 4) and U(g, c).temp_atk == 2
    cast(g, r, "purge")
    assert stats(g, c) == (5, 3, 3) and contrib(g, c) == (0, 0, 0)
    g.step(END)
    assert stats(g, c) == (3, 3, 3) and U(g, c).temp_atk == 0


def test_aura_with_a_turn_hp_buff_and_damage():
    r = Rules([banner(), unit("cadet", 2, 2, tags=["cadet"]),
               operation("brace", [effect("on_play", "buff", tgt("all", "friendly", filter={"tag": "cadet"}), hp=2,
                                          duration="turn")]),
               operation("jab", [effect("on_play", "damage", tgt("all", "friendly", filter={"tag": "cadet"}),
                                        amount=4)]), PURGE])
    g = blank(r)
    c = put(g, r, 0, "back", "cadet")
    cast(g, r, "banner")
    cast(g, r, "brace")
    assert stats(g, c) == (3, 5, 5) and U(g, c).temp_hp == 2
    cast(g, r, "jab")
    assert stats(g, c) == (3, 1, 5)
    cast(g, r, "purge")
    assert stats(g, c) == (2, 1, 4)
    g.step(END)
    assert stats(g, c) == (2, 1, 2)


# ================================================================= move cost auras (SPEC 1.2b, 2.12)
def test_static_move_cost_aura():
    # SPEC 1.2b static buff move_cost; SPEC 2.6 MOVE needs move_cost <= coins.
    r = Rules([unit("quartermaster", 0, 3, tags=["leader"],
                    effects=[static("buff", tgt("all", "friendly", filter=OTHERS), atk=0, move_cost=-1)]),
               unit("trudger", 1, 3, move_cost=2), PURGE])
    g = blank(r, coins=2)
    t = put(g, r, 0, "back", "trudger")
    cast(g, r, "quartermaster")
    assert U(g, t).move_cost == 1 and contrib(g, t) == (0, 0, -1) and view(g, 1, t).move_cost == 1
    g.coins[0] = 1
    g.invalidate()
    assert move(0) in g.legal_actions()
    cast(g, r, "purge")
    assert U(g, t).move_cost == 2 and move(0) not in g.legal_actions()


# ================================================================= clone / determinism with statics (SPEC 4)
def test_clone_keeps_static_contributions():
    r = Rules([banner(), unit("grunt", 2, 2), PURGE])
    g = blank(r)
    put(g, r, 0, "back", "grunt")
    cast(g, r, "banner")
    c = g.clone()
    assert state_key(c) == state_key(g)
    cast(c, r, "purge")
    cast(g, r, "purge")
    assert state_key(c) == state_key(g) and g.phase == MAIN


# ================================================================= loader validation (SPEC 1.2b static)
def _bad_static():
    tok = unit("tok", 1, 1, token=True)
    return {
        "damage": [unit("x", effects=[static("damage", tgt("all", "enemy"), amount=1)])],
        "pin": [unit("x", effects=[static("pin", tgt("all", "enemy"))])],
        "remove_trait": [unit("x", effects=[static("remove_trait", tgt("all", "enemy"), trait="defense")])],
        "heal_base": [unit("x", effects=[static("heal", "friendly_base", amount=1)])],
        "summon": [tok, unit("x", effects=[static("summon", "controller", card="tok")])],
        "random_select": [unit("x", effects=[static("buff", tgt("random", "friendly"), atk=1)])],
        "chosen_select": [unit("x", effects=[static("buff", tgt("chosen", "friendly"), atk=1)])],
        "event_select": [unit("x", effects=[static("buff", "event", atk=1)])],
        "prev_select": [unit("x", effects=[static("buff", "self", atk=1), static("buff", tgt("prev"), atk=1)])],
        "duration": [unit("x", effects=[static("buff", tgt("all", "friendly"), atk=1, duration="turn")])],
        "amount_expression": [unit("x", effects=[static("buff", tgt("all", "friendly"),
                                                        atk={"count": "units", "side": "friendly"})])],
        "repeat": [unit("x", effects=[static("buff", "self", atk=1, repeat=2)])],
        "on_an_operation": [operation("x", [static("buff", tgt("all", "friendly"), atk=1)])],
    }


@pytest.mark.parametrize("name", sorted(_bad_static()))
def test_loader_rejects_invalid_static_effects(name):
    # SPEC 1.2b static: actions buff (atk, hp, move_cost) and add_trait; selects self / all / adjacent; no
    # duration, amount expressions or repeat; operations only use on_play (SPEC 1.2).
    with pytest.raises(ValueError):
        Rules(_bad_static()[name])


def test_loader_accepts_valid_static_effects():
    Rules([unit("x", effects=[static("buff", "self", atk=1, hp=1),
                              static("buff", tgt("all", "friendly", zone="frontline", filter={"tag": "tank"}),
                                     atk=-1, move_cost=1, condition={"type": "turn", "whose": "own"}),
                              static("add_trait", adjacent("friendly", "self", "right"), trait="smokescreen")]),
           unit("y", effects=[static("add_trait", tgt("all", "enemy"), trait="ambush"),
                              static("buff", adjacent("friendly", "self"), hp=2)])])
