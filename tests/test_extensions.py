"""Independent tests of the Stage 3 phase-1b extensions (SPEC 1.2b, 2.8b, 2.13; SPEC 2.11 "The 1.2b
extensions").

Black-box, like the phase-1 files: positions are built from in-memory cards (`build_ruleset`, SPEC 1.5)
and the public state of SPEC 4; expectations come from the SPEC clause cited next to each test. Random
selections are replayed on a copy of the game RNG (SPEC 2.9: `rng.sample(candidates, count)`, no draw
with <= count candidates; `repeat` redraws each iteration). Static effects have their own file
(test_static.py). The last section runs determinism / clone / no-leak / determinize sweeps (SPEC 4, 5)
over random games of a pool that uses every extension.
"""
from __future__ import annotations

import random

import pytest

import cardgame.cards as cards_mod
from stage3_helpers import (BASE, END, MAIN, Game, Rules, adjacent, attack, back, bases_of, blank, board_order,
                            cards_of, choose, chooses, counts, effect, find, front, hand_slot, move, operation,
                            own_back, perturb_hidden, pick, play, predict_sample, put, set_deck, set_hand, state_key,
                            static, tgt, unit, uids, with_else, zone_of)


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


def U(g, u):
    x = find(g, u.uid)
    assert x is not None, f"uid {u.uid} left the board"
    return x


def cnt(g, field, p, r, cid):
    return counts(getattr(g, field)[p], r.n_cards)[r.idx(cid)]


def settled(g):
    return g.phase == MAIN and g.pending is None and len(g.queue) == 0


def view(g, viewer, u):
    """The UnitView of unit u as seen by `viewer` (SPEC 5)."""
    obs = g.observe(viewer)
    owner = U(g, u).owner
    zone, _ = zone_of(g, u.uid)
    if zone == "front":
        idx = [x.uid for x in g.frontline].index(u.uid)
        return obs.frontline[idx]
    idx = [x.uid for x in g.backline[owner]].index(u.uid)
    return (obs.my_backline if owner == viewer else obs.opp_backline)[idx]


TICK = operation("tick", [effect("on_play", "gain_coins", "controller", amount=0)], cost=0)
PREV = tgt("prev")


# ================================================================= tags (SPEC 1.2b card attributes)
TAGGED = [unit("panzer", 2, 4, tags=["tank", "germany"]), unit("rifle", 1, 4, tags=["infantry"]),
          unit("plain", 1, 4)]


@pytest.mark.parametrize("flt, hit", [
    ({"tag": "tank"}, {"panzer"}),
    ({"tag": "germany"}, {"panzer"}),
    ({"tag": ["tank", "infantry"]}, {"panzer", "rifle"}),
    ({"tag": ["navy"]}, set()),
    ({"not_tag": "tank"}, {"rifle", "plain"}),
    ({"not_tag": ["tank", "infantry"]}, {"plain"}),
    ({"tag": "tank", "not_tag": "germany"}, set()),
    ({"tag": "infantry", "nature": "troop"}, {"rifle"}),
])
def test_tag_filters(flt, hit):
    # SPEC 1.2b: filter `tag` = has any of the given tags (string or list); `not_tag` = has none of them;
    # all given filter keys must hold (SPEC 1.2).
    r = Rules(TAGGED + [operation("probe", [effect("on_play", "damage", tgt("all", "any", filter=flt), amount=1)])])
    g = blank(r)
    units = {"panzer": put(g, r, 0, "back", "panzer"), "rifle": put(g, r, 1, "back", "rifle"),
             "plain": put(g, r, 1, "front", "plain")}
    cast(g, r, "probe")
    for k, u in units.items():
        assert hp(g, u) == (3 if k in hit else 4), (flt, k)


def test_tags_in_conditions_amounts_and_on_operations():
    # SPEC 1.2b: tags on any card (operations too); the tag filter works in condition and count filters.
    r = Rules(TAGGED + [operation("salvo", [effect(
        "on_play", "damage", "enemy_base", amount={"count": "units", "side": "friendly", "filter": {"tag": "tank"}},
        condition={"type": "control", "side": "friendly", "filter": {"tag": "germany"}})], tags=["artillery"])])
    g = blank(r)
    put(g, r, 0, "back", "rifle")
    cast(g, r, "salvo")
    assert bases_of(g) == [20, 20]  # no germany unit: skipped
    put(g, r, 0, "back", "panzer")
    put(g, r, 0, "back", "panzer")
    cast(g, r, "salvo")
    assert bases_of(g) == [20, 18]


@pytest.mark.parametrize("pinned, hit", [(True, (3, 4)), (False, (4, 3))])
def test_pinned_filter(pinned, hit):
    # SPEC 1.2b filter key `pinned` (bool).
    r = Rules([unit("slowpoke", 1, 4, tags=["slow"]),
               operation("net", [effect("on_play", "pin", tgt("all", "enemy", filter={"tag": "slow"}))]),
               operation("strafe", [effect("on_play", "damage", tgt("all", "enemy", filter={"pinned": pinned}),
                                           amount=1)])])
    g = blank(r)
    a = put(g, r, 1, "back", "slowpoke")
    b = put(g, r, 1, "back", "fill03")
    cast(g, r, "net")
    assert (U(g, a).pinned, U(g, b).pinned) == (True, False)
    cast(g, r, "strafe")
    assert (hp(g, a), hp(g, b)) == hit


# ================================================================= ambush (SPEC 1.2b traits)
@pytest.mark.parametrize("has_ambush", [True, False])
def test_ambush_strikes_first_and_takes_no_damage_when_it_kills(has_ambush):
    # SPEC 1.2b ambush: attacked by an attacker that would take return damage, it strikes first; if that
    # strike kills the attacker, the ambusher takes no damage. Without ambush the trade is simultaneous and
    # both die (SPEC 2.6). SPEC 5 UnitView ambush / ambush_ready (public).
    r = Rules([unit("lurker", 3, 2, traits={"ambush": True} if has_ambush else None), unit("grunt", 2, 3)])
    g = blank(r)
    a = put(g, r, 0, "front", "grunt")
    lk = put(g, r, 1, "back", "lurker")
    v = view(g, 1, lk)
    assert (bool(v.ambush), bool(v.ambush_ready)) == (has_ambush, has_ambush)
    assert view(g, 0, lk) == v
    g.step(attack(front(0), back(0)))
    assert find(g, a.uid) is None and g.frontline == [] and g.front_owner is None
    assert cnt(g, "graveyard", 0, r, "grunt") == 1
    if has_ambush:
        assert hp(g, lk) == 2 and cnt(g, "graveyard", 1, r, "lurker") == 0
        v = view(g, 1, lk)
        assert v.ambush and not v.ambush_ready
    else:
        assert g.backline[1] == [] and cnt(g, "graveyard", 1, r, "lurker") == 1
    assert settled(g)


def test_ambush_strike_that_does_not_kill_is_followed_by_the_attackers_damage():
    # SPEC 1.2b: otherwise the attacker's damage follows (armor applies to both combat hits, SPEC 2.4);
    # the ambusher hits once.
    r = Rules([unit("lurker", 3, 4, traits={"ambush": True}), unit("knight", 3, 5, traits={"armor": 1})])
    g = blank(r)
    k = put(g, r, 0, "front", "knight")
    lk = put(g, r, 1, "back", "lurker")
    g.step(attack(front(0), back(0)))
    assert (hp(g, k), hp(g, lk)) == (3, 1)
    assert not view(g, 0, lk).ambush_ready


def test_ambush_fires_only_the_first_time_each_turn():
    # SPEC 1.2b: "the first time each turn"; the second attack is ordinary simultaneous combat.
    r = Rules([unit("lurker", 3, 6, traits={"ambush": True}), unit("grunt", 2, 2), unit("brute", 3, 3)])
    g = blank(r)
    put(g, r, 0, "front", "grunt")
    put(g, r, 0, "front", "brute")
    lk = put(g, r, 1, "back", "lurker")
    g.step(attack(front(0), back(0)))
    assert hp(g, lk) == 6 and cards_of(g, 0, r, "front") == ["brute"]
    g.step(attack(front(0), back(0)))
    assert hp(g, lk) == 3 and g.frontline == []


def test_ambush_refreshes_at_every_turn_start():
    # SPEC 1.2b: its per-turn use refreshes at every turn start (also the ambusher's own turn).
    r = Rules([unit("lurker", 3, 6, traits={"ambush": True}), unit("grunt", 2, 2)])
    g = blank(r)
    put(g, r, 0, "front", "grunt")
    lk = put(g, r, 1, "back", "lurker")
    g.step(attack(front(0), back(0)))
    assert not view(g, 1, lk).ambush_ready
    g.step(END)  # p1's turn starts
    assert view(g, 1, lk).ambush_ready and view(g, 0, lk).ambush_ready
    g.step(END)  # p0's turn starts
    put(g, r, 0, "front", "grunt")
    g.step(attack(front(0), back(0)))
    assert hp(g, lk) == 6 and g.frontline == []


@pytest.mark.parametrize("attacker", ["archer", "striker"])
def test_ambush_ignores_ranged_and_shock_attackers_and_keeps_its_use(attacker):
    # SPEC 1.2b: ambush needs an attacker that would take return damage; ranged attackers take none
    # (SPEC 2.6) and shock attackers take none ("Ambush does not fire against it"). The use is kept.
    r = Rules([unit("lurker", 3, 6, traits={"ambush": True}), unit("archer", 2, 2, nature="ranged"),
               unit("striker", 2, 2, traits={"shock": True}), unit("grunt", 2, 2)])
    g = blank(r)
    if attacker == "archer":
        a = put(g, r, 0, "back", "archer")
        slot_a = back(0)
    else:
        a = put(g, r, 0, "front", "striker")
        slot_a = front(0)
    gr = put(g, r, 0, "front", "grunt")
    lk = put(g, r, 1, "back", "lurker")
    g.step(attack(slot_a, back(0)))
    assert (hp(g, a), hp(g, lk)) == (2, 4)
    assert view(g, 1, lk).ambush_ready
    slot_g = front([u.uid for u in g.frontline].index(gr.uid))
    g.step(attack(slot_g, back(0)))
    assert find(g, gr.uid) is None and hp(g, lk) == 4


def test_ambush_kill_is_a_combat_kill():
    # SPEC 1.2 on_kill (lethal combat damage) for the ambusher, event = the attacker (SPEC 2.8); the
    # attacker's on_death fires.
    r = Rules([unit("lurker", 3, 4, traits={"ambush": True}, effects=[
        effect("on_kill", "damage", "enemy_base", amount={"stat": "atk", "of": "event"}),
        effect("on_damaged", "damage", "enemy_base", amount=5)]),
        unit("grunt", 2, 2, effects=[effect("on_death", "damage", "enemy_base", amount=1)])])
    g = blank(r)
    put(g, r, 0, "front", "grunt")
    put(g, r, 1, "back", "lurker")
    g.step(attack(front(0), back(0)))
    assert bases_of(g) == [18, 19]  # on_kill 2 to p0's base, grunt's on_death 1 to p1's; no on_damaged
    assert settled(g)


# ================================================================= shock and immune (SPEC 1.2b traits)
def test_shock_attacker_takes_no_return_damage():
    # SPEC 1.2b shock: when this unit attacks, it takes no return damage.
    r = Rules([unit("striker", 2, 3, traits={"shock": True}), unit("ogre", 5, 5)])
    g = blank(r)
    s = put(g, r, 0, "front", "striker")
    o = put(g, r, 1, "back", "ogre")
    assert view(g, 0, s).shock and not view(g, 0, o).shock
    g.step(attack(front(0), back(0)))
    assert (hp(g, s), hp(g, o)) == (3, 3)


def test_shock_does_not_protect_a_defender():
    # SPEC 1.2b: shock only applies "when this unit attacks".
    r = Rules([unit("striker", 2, 3, traits={"shock": True}), unit("poker", 1, 5)])
    g = blank(r, current=1, first=0)
    s = put(g, r, 0, "front", "striker")
    p = put(g, r, 1, "back", "poker")
    g.step(attack(back(0), front(0)))
    assert (hp(g, s), hp(g, p)) == (2, 3)


def test_immune_takes_no_effect_damage_and_does_not_trigger_on_damaged():
    # SPEC 1.2b immune: no damage from effects; SPEC 2.13: 0 effect damage does not fire on_damaged.
    r = Rules([unit("saint", 1, 3, traits={"immune": True},
                    effects=[effect("on_damaged", "damage", "enemy_base", amount=1)]),
               unit("mob", 1, 4),
               operation("barrage", [effect("on_play", "damage", tgt("all", "enemy"), amount=2)])])
    g = blank(r)
    s = put(g, r, 1, "back", "saint")
    m = put(g, r, 1, "back", "mob")
    assert view(g, 0, s).immune
    cast(g, r, "barrage")
    assert (hp(g, s), hp(g, m)) == (3, 2)
    assert bases_of(g) == [20, 20]


def test_immune_takes_no_combat_damage_attacking_or_defending():
    # SPEC 1.2b immune: no damage from combat; the other unit still takes its damage.
    r = Rules([unit("saint", 2, 3, traits={"immune": True}), unit("grunt", 3, 3), unit("ogre", 5, 5)])
    g = blank(r)
    a = put(g, r, 0, "front", "grunt")
    s = put(g, r, 1, "back", "saint")
    g.step(attack(front(0), back(0)))
    assert (hp(g, a), hp(g, s)) == (1, 3)
    g = blank(r)
    s = put(g, r, 0, "front", "saint")
    o = put(g, r, 1, "back", "ogre")
    g.step(attack(front(0), back(0)))
    assert (hp(g, s), hp(g, o)) == (3, 3)


def test_destroy_still_kills_an_immune_unit():
    # SPEC 1.2b: `destroy` still kills it (SPEC 2.10: on_death fires).
    r = Rules([unit("saint", 1, 3, traits={"immune": True},
                    effects=[effect("on_death", "damage", "enemy_base", amount=2)]),
               operation("execute", [effect("on_play", "destroy", tgt("all", "enemy"))])])
    g = blank(r)
    put(g, r, 1, "back", "saint")
    cast(g, r, "execute")
    assert g.backline[1] == [] and cnt(g, "graveyard", 1, r, "saint") == 1
    assert bases_of(g) == [18, 20]


@pytest.mark.parametrize("trait", ["ambush", "shock", "immune"])
def test_new_traits_can_be_granted_for_a_turn_and_removed(trait):
    # SPEC 1.2b traits are also for add_trait / remove_trait; SPEC 2.10 "turn" grants expire at END_TURN.
    r = Rules([unit("decorated", 1, 4, traits={trait: True}),
               operation("give", [effect("on_play", "add_trait", tgt("all", "friendly"), trait=trait,
                                         duration="turn")]),
               operation("strip", [effect("on_play", "remove_trait", tgt("all", "friendly"), trait=trait)])])
    g = blank(r)
    plain = put(g, r, 0, "back", "fill03")
    deco = put(g, r, 0, "back", "decorated")
    cast(g, r, "give")
    assert getattr(U(g, plain), trait) and getattr(view(g, 1, plain), trait)
    cast(g, r, "strip")
    assert not getattr(U(g, plain), trait) and not getattr(U(g, deco), trait)
    g.step(END)
    assert not getattr(U(g, plain), trait) and not getattr(U(g, deco), trait)


def test_granted_shock_and_immune_work_in_combat():
    r = Rules([unit("ogre", 4, 9),
               operation("charge", [effect("on_play", "add_trait", tgt("all", "friendly"), trait="shock",
                                           duration="turn")]),
               operation("aegis", [effect("on_play", "add_trait", tgt("all", "friendly"), trait="immune",
                                          duration="turn")])])
    g = blank(r)
    a = put(g, r, 0, "front", "fill02")  # 3/3
    o = put(g, r, 1, "back", "ogre")
    cast(g, r, "charge")
    g.step(attack(front(0), back(0)))
    assert (hp(g, a), hp(g, o)) == (3, 6)
    g = blank(r, current=1, first=0)
    d = put(g, r, 1, "back", "fill02")
    put(g, r, 0, "front", "ogre")
    cast(g, r, "aegis")
    g.step(attack(back(0), front(0)))
    assert hp(g, d) == 3
    g.step(END)  # the "turn" grant expires
    assert not U(g, d).immune


# ================================================================= on_attacked (SPEC 1.2b triggers)
def test_on_attacked_resolves_before_combat_with_the_attacker_as_event():
    # SPEC 1.2b: on_attacked after on_attack and before combat damage; the event unit is the attacker.
    r = Rules([unit("sentinel", 1, 5, effects=[effect("on_attacked", "damage", "event", amount=2)]),
               unit("grunt", 3, 4)])
    g = blank(r)
    a = put(g, r, 0, "front", "grunt")
    s = put(g, r, 1, "back", "sentinel")
    g.step(attack(front(0), back(0)))
    assert (hp(g, a), hp(g, s)) == (1, 2)
    assert settled(g)


def test_on_attacked_killing_the_attacker_ends_the_attack():
    # SPEC 2.6 ATTACK 3: the attacker left the board -> the attack ends (no combat damage).
    r = Rules([unit("sentinel", 1, 5, effects=[effect("on_attacked", "damage", "event", amount=4)]),
               unit("grunt", 3, 3)])
    g = blank(r)
    a = put(g, r, 0, "front", "grunt")
    s = put(g, r, 1, "back", "sentinel")
    g.step(attack(front(0), back(0)))
    assert find(g, a.uid) is None and g.front_owner is None and hp(g, s) == 5


def test_on_attack_effects_resolve_before_on_attacked():
    # SPEC 1.2b: on_attacked comes after on_attack (the attacker's own effect and on_attack watchers).
    r = Rules([unit("tA", 0, 1, token=True), unit("tW", 0, 1, token=True), unit("tD", 0, 1, token=True),
               unit("herald", 1, 9, effects=[effect("on_attack", "summon", "opponent", card="tA")]),
               unit("lookout", 0, 9, effects=[effect("on_attack", "summon", "controller", scope="enemy", card="tW")]),
               unit("sentry", 0, 9, effects=[effect("on_attacked", "summon", "controller", card="tD")])])
    g = blank(r)
    put(g, r, 0, "front", "herald")
    put(g, r, 1, "back", "sentry")
    put(g, r, 1, "back", "lookout")
    g.step(attack(front(0), back(0)))
    assert cards_of(g, 1, r) == ["sentry", "lookout", "tA", "tW", "tD"]


def test_on_attacked_does_not_fire_for_base_attacks():
    r = Rules([unit("sentry", 0, 9, effects=[effect("on_attacked", "damage", "enemy_base", amount=3)]),
               unit("lookout", 0, 9, effects=[effect("on_attacked", "damage", "enemy_base", scope="any", amount=3)]),
               unit("grunt", 2, 4)])
    g = blank(r)
    put(g, r, 0, "front", "grunt")
    put(g, r, 1, "back", "sentry")
    put(g, r, 1, "back", "lookout")
    g.step(attack(front(0), BASE))
    assert bases_of(g) == [20, 18]


@pytest.mark.parametrize("w", [0, 1])
@pytest.mark.parametrize("scope", ["friendly", "enemy", "any"])
def test_on_attacked_watcher_scopes(scope, w):
    # SPEC 1.2b: scopes as for the other unit events, matched against the attacked unit's side relative
    # to the watcher; watchers never include the source (the attacked unit's own `any` effect is silent).
    weff = effect("on_attacked", "damage", "enemy_base", scope=scope, amount=1)
    r = Rules([unit("watcher", 0, 9, effects=[weff]), unit("dummy", 0, 9, effects=[weff]), unit("grunt", 1, 9)])
    g = blank(r)
    put(g, r, 0, "front", "grunt")
    put(g, r, 1, "back", "dummy")
    put(g, r, w, "back", "watcher")
    g.step(attack(front(0), back(0)))
    relation = "friendly" if w == 1 else "enemy"
    expected = [20, 20]
    if scope == "any" or scope == relation:
        expected[1 - w] -= 1
    assert bases_of(g) == expected


def test_on_attacked_watcher_event_is_the_attacker():
    # SPEC 1.2b: "The event unit is the attacker" (for watchers too).
    r = Rules([unit("guard", 0, 9, effects=[effect("on_attacked", "damage", "event", scope="friendly", amount=2)]),
               unit("dummy", 0, 9), unit("grunt", 1, 5)])
    g = blank(r)
    a = put(g, r, 0, "front", "grunt")
    d = put(g, r, 1, "back", "dummy")
    put(g, r, 1, "back", "guard")
    g.step(attack(front(0), back(0)))
    assert (hp(g, a), hp(g, d)) == (3, 8)


# ================================================================= event_filter (SPEC 1.2b watcher filters)
def test_event_filter_on_a_deploy_watcher_applies_to_the_deployed_unit():
    r = Rules([unit("recruiter", 0, 9, effects=[effect("on_deploy", "damage", "enemy_base", scope="friendly",
                                                       amount=1, event_filter={"nature": "fast"})]),
               unit("plain", 1, 1), unit("runner", 1, 1, nature="fast")])
    g = blank(r)
    put(g, r, 0, "back", "recruiter")
    cast(g, r, "plain")
    assert bases_of(g) == [20, 20]
    cast(g, r, "runner")
    assert bases_of(g) == [20, 19]


@pytest.mark.parametrize("flt, fires", [
    ({"max_cost": 2}, (True, False, True)),
    ({"min_cost": 3}, (False, True, False)),
    ({"tag": "artillery"}, (False, True, True)),
    ({"tag": ["navy", "spy"]}, (False, False, True)),
    ({"not_tag": "artillery"}, (True, False, False)),
    ({"tag": "artillery", "max_cost": 2}, (False, False, True)),
])
def test_event_filter_on_an_on_play_watcher_applies_to_the_played_card(flt, fires):
    # SPEC 1.2b: for on_play watchers F applies to the played card (max_cost / min_cost / tag / not_tag).
    r = Rules([unit("listener", 0, 9, effects=[effect("on_play", "damage", "enemy_base", scope="any", amount=1,
                                                      event_filter=flt)]),
               operation("memo", [effect("on_play", "gain_coins", "controller", amount=0)], cost=1),
               operation("bombard", [effect("on_play", "gain_coins", "controller", amount=0)], cost=3,
                         tags=["artillery"]),
               operation("shell", [effect("on_play", "gain_coins", "controller", amount=0)], cost=2,
                         tags=["artillery", "navy"])])
    g = blank(r)
    put(g, r, 0, "back", "listener")
    base = 20
    for cid, f in zip(("memo", "bombard", "shell"), fires):
        cast(g, r, cid)
        base -= f
        assert bases_of(g) == [20, base], cid


def test_event_filter_on_a_death_watcher_applies_to_the_dead_unit():
    r = Rules([unit("ghoul", 0, 9, effects=[effect("on_death", "damage", "enemy_base", scope="any", amount=1,
                                                   event_filter={"token": True})]),
               unit("militia", 0, 1, token=True),
               operation("purge", [effect("on_play", "damage", tgt("all", "enemy"), amount=1)])])
    g = blank(r)
    put(g, r, 0, "back", "ghoul")
    put(g, r, 1, "back", "militia")
    put(g, r, 1, "back", "fill00")
    put(g, r, 1, "back", "militia")
    cast(g, r, "purge")
    assert g.backline[1] == [] and bases_of(g) == [20, 18]


def test_event_filter_on_a_kill_watcher_applies_to_the_victim():
    # SPEC 1.2b: for unit events F applies to the event unit; SPEC 2.13: on_kill's event unit is the victim
    # (scopes are matched against the killer's side).
    r = Rules([unit("tally", 0, 9, effects=[effect("on_kill", "damage", "enemy_base", scope="friendly", amount=1,
                                                   event_filter={"token": True})]),
               unit("militia", 1, 1, token=True), unit("reaper", 3, 9)])
    g = blank(r)
    put(g, r, 0, "back", "tally")
    put(g, r, 0, "front", "reaper")
    put(g, r, 0, "front", "reaper")
    put(g, r, 1, "back", "fill00")    # 1/1, not a token
    put(g, r, 1, "back", "militia")
    g.step(attack(front(0), back(0)))  # kills fill00: no
    assert bases_of(g) == [20, 20]
    g.step(attack(front(1), back(0)))  # kills the token: yes
    assert g.backline[1] == [] and bases_of(g) == [20, 19]


def test_event_filter_on_an_attack_watcher_applies_to_the_attacker():
    r = Rules([unit("spotter", 0, 9, effects=[effect("on_attack", "damage", "enemy_base", scope="friendly", amount=1,
                                                     event_filter={"nature": "ranged"})]),
               unit("archer", 1, 9, nature="ranged"), unit("grunt", 1, 9)])
    g = blank(r)
    put(g, r, 0, "back", "spotter")
    put(g, r, 0, "back", "archer")
    put(g, r, 0, "front", "grunt")
    g.step(attack(front(0), BASE))
    assert bases_of(g) == [20, 19]
    g.step(attack(back(1), BASE))
    assert bases_of(g) == [20, 17]


# ================================================================= prev select and clause batches (SPEC 1.2b, 2.8b)
def test_prev_targets_the_chosen_unit_of_the_previous_clause():
    r = Rules([operation("pincer", [effect("on_play", "damage", tgt("chosen", "enemy"), amount=1),
                                    effect("on_play", "pin", PREV)])])
    g = blank(r)
    a = put(g, r, 1, "back", "fill03")
    b = put(g, r, 1, "back", "fill03")
    cast(g, r, "pincer")
    g.step(choose(1))
    assert (hp(g, a), hp(g, b)) == (4, 3)
    assert (U(g, a).pinned, U(g, b).pinned) == (False, True)
    assert settled(g)


def test_prev_in_a_unit_batch():
    r = Rules([unit("raider", 1, 3, effects=[effect("on_deploy", "damage", tgt("chosen", "enemy"), amount=1),
                                             effect("on_deploy", "buff", PREV, atk=-1)])])
    g = blank(r)
    a = put(g, r, 1, "back", "fill02")  # 3/3
    b = put(g, r, 1, "back", "fill02")
    cast(g, r, "raider")
    g.step(choose(0))
    assert (U(g, a).atk, U(g, a).hp, U(g, b).atk, U(g, b).hp) == (2, 2, 3, 3)


def test_prev_after_a_random_clause_hits_the_same_unit():
    # SPEC 1.2b prev + SPEC 2.9 random (one draw only).
    r = Rules([operation("followup", [effect("on_play", "damage", tgt("random", "enemy"), amount=1),
                                      effect("on_play", "damage", PREV, amount=2)])])
    for seed in range(6):
        g = blank(r)
        g.rng.seed(seed)
        for _ in range(4):
            put(g, r, 1, "back", "fill03")
        cands = uids(g.backline[1])
        state = g.rng.getstate()
        cast(g, r, "followup")
        ref = random.Random()
        ref.setstate(state)
        [pick] = ref.sample(cands, 1)
        assert {u.uid: u.hp for u in g.backline[1]} == {u: (1 if u == pick else 4) for u in cands}
        assert g.rng.getstate() == ref.getstate()


def test_prev_skips_unit_targets_that_left_the_board():
    # SPEC 1.2b: prev = unit targets still on the board (zone compaction does not redirect it).
    r = Rules([operation("followup", [effect("on_play", "damage", tgt("chosen", "enemy"), amount=2),
                                      effect("on_play", "damage", PREV, amount=3)])])
    g = blank(r)
    put(g, r, 1, "back", "fill00")  # 1/1
    b = put(g, r, 1, "back", "fill03")
    cast(g, r, "followup")
    g.step(choose(0))
    assert cards_of(g, 1, r) == ["fill03"] and hp(g, b) == 4 and settled(g)


@pytest.mark.parametrize("pick, returned", [(0, False), (1, True)])
def test_prev_never_reaches_a_unit_that_died(pick, returned):
    # SPEC 1.2b: prev = unit targets still on the board (a dead unit's card does not come back to hand).
    r = Rules([operation("bounce", [effect("on_play", "damage", tgt("chosen", "enemy"), amount=2),
                                    effect("on_play", "return_to_hand", PREV)])])
    g = blank(r)
    put(g, r, 1, "back", "fill00")  # 1/1: dies
    put(g, r, 1, "back", "fill03")  # 1/4: survives
    cast(g, r, "bounce")
    g.step(choose(pick))
    cid = ("fill00", "fill03")[pick]
    assert list(g.hands[1]) == ([r.idx(cid)] if returned else [])
    assert cnt(g, "graveyard", 1, r, "fill00") == (0 if returned else 1)
    assert cards_of(g, 1, r) == (["fill00"] if returned else ["fill03"])


def test_prev_reads_the_most_recent_earlier_clause():
    # SPEC 2.8b: prev reads the most recent earlier clause of the batch.
    r = Rules([operation("triad", [effect("on_play", "pin", tgt("all", "enemy")),
                                   effect("on_play", "add_trait", tgt("all", "friendly"), trait="defense"),
                                   effect("on_play", "damage", PREV, amount=1)])])
    g = blank(r)
    m = put(g, r, 0, "back", "fill03")
    e = put(g, r, 1, "back", "fill03")
    cast(g, r, "triad")
    assert (hp(g, m), hp(g, e)) == (3, 4)
    assert U(g, e).pinned and U(g, m).defense and not U(g, m).pinned


def test_a_fizzled_clause_is_recorded_as_an_empty_target_list():
    # SPEC 2.8b: fizzled clauses are recorded as an empty target list, so a later prev finds nothing.
    r = Rules([operation("triad", [effect("on_play", "pin", tgt("all", "enemy")),
                                   effect("on_play", "damage", tgt("all", "enemy", filter={"nature": "fast"}), amount=1),
                                   effect("on_play", "damage", PREV, amount=3)])])
    g = blank(r)
    e = put(g, r, 1, "back", "fill03")
    cast(g, r, "triad")
    assert hp(g, e) == 4 and U(g, e).pinned and settled(g)


def test_prev_with_unit_and_base_targets():
    # SPEC 1.2b: prev holds the previous clause's unit targets still on the board and its bases. The prev
    # target declares kind unit_or_base: SPEC 1.2 TG gives `kind` the default "unit", and the SPEC does
    # not say whether a bare {"select": "prev"} also takes bases (reported as an ambiguity).
    r = Rules([operation("double_tap", [effect("on_play", "damage", tgt("all", "enemy", "unit_or_base"), amount=2),
                                        effect("on_play", "damage", tgt("prev", kind="unit_or_base"), amount=1)])])
    g = blank(r)
    e = put(g, r, 1, "back", "fill03")
    m = put(g, r, 0, "back", "fill03")
    cast(g, r, "double_tap")
    assert bases_of(g) == [20, 17] and (hp(g, e), hp(g, m)) == (1, 4)


def test_prev_with_player_targets():
    # SPEC 1.2b: prev also holds players (declared with kind player; see the report on `kind player only
    # with select all`).
    r = Rules([operation("study", [effect("on_play", "draw", "opponent", amount=1),
                                   effect("on_play", "draw", tgt("prev", kind="player"), amount=1)])])
    g = blank(r)
    set_deck(g, r, 0, ["fill01", "fill02"])
    set_deck(g, r, 1, ["fill03", "fill04", "fill05"])
    cast(g, r, "study")
    assert (len(g.hands[0]), len(g.hands[1])) == (0, 2)


def test_prev_after_target_condition_holds_the_matching_targets():
    # SPEC 2.8b: the recorded targets are taken after the target_condition split (no else: the
    # non-matching targets receive nothing and are not the clause's targets).
    r = Rules([unit("runner", 1, 4, nature="fast"),
               operation("sweep", [effect("on_play", "damage", tgt("all", "enemy"), amount=1,
                                          target_condition={"nature": "troop"}),
                                   effect("on_play", "pin", PREV)])])
    g = blank(r)
    t = put(g, r, 1, "back", "fill03")
    f = put(g, r, 1, "back", "runner")
    cast(g, r, "sweep")
    assert (hp(g, t), hp(g, f)) == (3, 4)
    assert (U(g, t).pinned, U(g, f).pinned) == (True, False)


# ================================================================= adjacent select (SPEC 1.2b targets)
@pytest.mark.parametrize("position, hit", [(None, {0, 2}), ("both", {0, 2}), ("left", {0}), ("right", {2})])
def test_adjacent_of_the_event_unit(position, hit):
    # SPEC 1.2b adjacent: slot +-1 in the reference unit's zone; position both (default) / left / right.
    r = Rules([unit("splasher", 1, 9, effects=[effect("on_attack", "damage", adjacent("enemy", "event", position),
                                                      amount=1)]),
               unit("dummy", 0, 5)])
    g = blank(r)
    put(g, r, 0, "front", "splasher")
    ds = [put(g, r, 1, "back", "dummy") for _ in range(4)]
    g.step(attack(front(0), back(1)))
    assert [hp(g, d) for d in ds] == [4 if i in hit else (4 if i == 1 else 5) for i in range(4)]


def test_adjacent_at_the_edge_of_a_zone():
    r = Rules([unit("splasher", 1, 9, effects=[effect("on_attack", "damage", adjacent("enemy", "event"), amount=1)]),
               unit("dummy", 0, 5)])
    g = blank(r)
    put(g, r, 0, "front", "splasher")
    ds = [put(g, r, 1, "back", "dummy") for _ in range(3)]
    g.step(attack(front(0), back(0)))
    assert [hp(g, d) for d in ds] == [4, 4, 5]


def test_adjacent_of_self_in_the_frontline():
    r = Rules([unit("captain", 1, 9, effects=[effect("on_attack", "buff", adjacent("friendly", "self"), atk=1)])])
    g = blank(r)
    side = put(g, r, 0, "back", "fill03")
    x = put(g, r, 0, "front", "fill03")
    c = put(g, r, 0, "front", "captain")
    y = put(g, r, 0, "front", "fill03")
    z = put(g, r, 0, "front", "fill03")
    g.step(attack(front(1), BASE))
    assert [U(g, u).atk for u in (side, x, c, y, z)] == [1, 2, 1, 2, 1]


@pytest.mark.parametrize("pick, buffed", [(own_back(1), {0, 2}), (1, set())])
def test_adjacent_of_prev_and_side_filters(pick, buffed):
    # SPEC 1.2b: "of": "prev"; side filters still apply (an enemy's neighbours are not friendly).
    r = Rules([operation("rally", [effect("on_play", "pin", tgt("chosen", "any")),
                                   effect("on_play", "buff", adjacent("friendly", "prev"), atk=2)])])
    g = blank(r)
    mine = [put(g, r, 0, "back", "fill03") for _ in range(3)]
    theirs = [put(g, r, 1, "back", "fill03") for _ in range(3)]
    cast(g, r, "rally")
    g.step(choose(pick))
    assert [U(g, u).atk for u in mine] == [3 if i in buffed else 1 for i in range(3)]
    assert [U(g, u).atk for u in theirs] == [1, 1, 1]


# ================================================================= amounts (SPEC 1.2b)
def test_amount_stat_move_cost_of_self_and_event():
    r = Rules([unit("trekker", 1, 3, move_cost=2, effects=[
        effect("on_move", "damage", "enemy_base", amount={"stat": "move_cost", "of": "self"})]),
        unit("spotter", 0, 9, nature="ranged", effects=[
            effect("on_attack", "damage", "enemy_base", amount={"stat": "move_cost", "of": "event"})]),
        unit("hauler", 0, 9, move_cost=3)])
    g = blank(r)
    put(g, r, 0, "back", "trekker")
    g.step(move(0))
    assert bases_of(g) == [20, 18] and g.coins[0] == 8
    g = blank(r)
    put(g, r, 0, "back", "spotter")
    put(g, r, 1, "back", "hauler")
    g.step(attack(back(0), back(0)))
    assert bases_of(g) == [20, 17]


def test_amount_of_prev_uses_last_known_values():
    # SPEC 1.2b: "of": "prev" = last-known values of the first prev target.
    r = Rules([operation("execute", [effect("on_play", "destroy", tgt("chosen", "enemy")),
                                     effect("on_play", "damage", "enemy_base", amount={"stat": "atk", "of": "prev"})])])
    g = blank(r)
    put(g, r, 1, "back", "fill03")
    put(g, r, 1, "back", "fill03", atk=6)
    cast(g, r, "execute")
    g.step(choose(1))
    assert len(g.backline[1]) == 1 and bases_of(g) == [20, 14]


def test_amount_of_prev_reads_the_first_target_in_turn_relative_board_order():
    # SPEC 1.2b "first prev target" + SPEC 2.13: all-candidate lists follow board order starting with the
    # turn player (p1 here): p1 backline, p0 backline, p0 frontline.
    r = Rules([unit("u2", 2, 5), unit("u3", 3, 5), unit("u4", 4, 5),
               operation("census", [effect("on_play", "pin", tgt("all", "any")),
                                    effect("on_play", "damage", "enemy_base", amount={"stat": "atk", "of": "prev"})])])
    g = blank(r, current=1, first=0)
    put(g, r, 0, "back", "u2")
    put(g, r, 0, "front", "u3")
    put(g, r, 1, "back", "u4")
    cast(g, r, "census")
    assert bases_of(g) == [16, 20]


def test_amount_count_base_hp():
    r = Rules([operation("retaliate", [effect("on_play", "damage", "enemy_base",
                                              amount={"count": "base_hp", "side": "friendly"})]),
               operation("mirror", [effect("on_play", "damage", "friendly_base",
                                           amount={"count": "base_hp", "side": "enemy", "plus": -10})])])
    g = blank(r)
    g.base_hp[0] = 7
    g.invalidate()
    cast(g, r, "retaliate")
    assert bases_of(g) == [7, 13]
    cast(g, r, "mirror")
    assert bases_of(g) == [4, 13]


def test_amount_event_damage_on_damaged_self_after_armor():
    # SPEC 1.2b {"event": "damage"}: the damage just taken (combat damage after armor, SPEC 2.4).
    r = Rules([unit("brooder", 1, 9, traits={"armor": 1}, effects=[
        effect("on_damaged", "damage", "enemy_base", amount={"event": "damage"})]), unit("ogre", 3, 9)])
    g = blank(r)
    put(g, r, 0, "front", "ogre")
    b = put(g, r, 1, "back", "brooder")
    g.step(attack(front(0), back(0)))
    assert hp(g, b) == 7 and bases_of(g) == [18, 20]


def test_amount_event_damage_on_damaged_watcher():
    r = Rules([unit("medic", 0, 9, effects=[effect("on_damaged", "damage", "enemy_base", scope="friendly",
                                                   amount={"event": "damage", "plus": 1})]),
               operation("blast", [effect("on_play", "damage", tgt("all", "enemy", filter={"max_hp": 6}), amount=3)])])
    g = blank(r)
    put(g, r, 1, "back", "medic")
    put(g, r, 1, "back", "fill03")  # 1/4
    cast(g, r, "blast")
    assert bases_of(g) == [16, 20]


def test_amount_event_damage_on_kill_is_the_lethal_hit():
    # SPEC 1.2b: on_kill -> the damage dealt by the lethal hit (all 5, the victim's hp goes to -3).
    r = Rules([unit("reaper", 5, 9, effects=[effect("on_kill", "damage", "enemy_base", amount={"event": "damage"})]),
               unit("victim", 1, 2)])
    g = blank(r)
    put(g, r, 0, "front", "reaper")
    put(g, r, 1, "back", "victim")
    g.step(attack(front(0), back(0)))
    assert g.backline[1] == [] and bases_of(g) == [20, 15]


# ================================================================= conditions (SPEC 1.2b)
@pytest.mark.parametrize("pick, killed", [(0, True), (1, False)])
@pytest.mark.parametrize("want", [True, False])
def test_prev_condition_killed(pick, killed, want):
    # SPEC 1.2b {"type": "prev", "killed": bool}: about the previous clause's (single) unit target.
    r = Rules([operation("strike", [effect("on_play", "damage", tgt("chosen", "enemy"), amount=2),
                                    effect("on_play", "damage", "enemy_base", amount=3,
                                           condition={"type": "prev", "killed": want})])])
    g = blank(r)
    put(g, r, 1, "back", "fill01")  # 2/2: dies
    put(g, r, 1, "back", "fill03")  # 1/4: survives
    cast(g, r, "strike")
    g.step(choose(pick))
    assert bases_of(g) == [20, 17 if killed == want else 20]


def test_prev_condition_killed_with_several_targets():
    # SPEC 1.2b: "any prev unit target killed".
    r = Rules([operation("sweep", [effect("on_play", "damage", tgt("all", "enemy"), amount=1),
                                   effect("on_play", "draw", "controller", amount=1,
                                          condition={"type": "prev", "killed": True})])])
    g = blank(r)
    put(g, r, 1, "back", "fill03")
    put(g, r, 1, "back", "fill00")  # dies
    set_deck(g, r, 0, ["fill05"])
    cast(g, r, "sweep")
    assert list(g.hands[0]) == [r.idx("fill05")]


@pytest.mark.parametrize("pick, holds", [(0, True), (1, False)])
def test_prev_condition_filter(pick, holds):
    # SPEC 1.2b {"type": "prev", "filter": F}: any prev unit target matching F.
    r = Rules([unit("runner", 1, 4, nature="fast"),
               operation("probe", [effect("on_play", "damage", tgt("chosen", "enemy"), amount=1),
                                   effect("on_play", "damage", "enemy_base", amount=2,
                                          condition={"type": "prev", "filter": {"nature": "fast"}})])])
    g = blank(r)
    put(g, r, 1, "back", "runner")
    put(g, r, 1, "back", "fill03")
    cast(g, r, "probe")
    g.step(choose(pick))
    assert bases_of(g) == [20, 18 if holds else 20]


HAND_F = {"count": "hand", "side": "friendly"}   # 2 cards
HAND_E = {"count": "hand", "side": "enemy"}      # 3 cards
UNITS_E = {"count": "units", "side": "enemy"}    # 2 units


@pytest.mark.parametrize("left, op_, right, holds", [
    (HAND_F, ">", HAND_E, False), (HAND_F, ">=", HAND_E, False), (HAND_F, "<", HAND_E, True),
    (HAND_F, "<=", HAND_E, True), (HAND_F, "==", HAND_E, False),
    (UNITS_E, "==", 2, True), (UNITS_E, ">=", 2, True), (UNITS_E, ">", 2, False),
    (2, "<", dict(UNITS_E, plus=1), True), (dict(HAND_E, times=2), "==", 6, True),
    (dict(HAND_F, plus=1), "<=", HAND_E, True), (dict(HAND_F, plus=1), "<", HAND_E, False),
])
def test_compare_condition(left, op_, right, holds):
    # SPEC 1.2b {"type": "compare", "left": AM, "op", "right": AM}.
    r = Rules([operation("gauge", [effect("on_play", "damage", "enemy_base", amount=1,
                                          condition={"type": "compare", "left": left, "op": op_, "right": right})],
                         cost=0)])
    g = blank(r)
    set_hand(g, r, 0, ["fill00", "fill01"])
    set_hand(g, r, 1, ["fill00"] * 3)
    put(g, r, 1, "back", "fill03")
    put(g, r, 1, "back", "fill03")
    cast(g, r, "gauge")
    assert bases_of(g) == [20, 19 if holds else 20]


def chronicler(cid, condition, amount=1, scope=None):
    """A unit whose end_of_turn effect checks a history condition (so the checking card never counts
    itself)."""
    return unit(cid, 0, 9, effects=[effect("end_of_turn", "damage", "enemy_base", scope=scope, amount=amount,
                                           condition=condition)])


def hist(event, side, window, **bounds):
    return dict({"type": "history", "event": event, "side": side, "window": window}, **bounds)


@pytest.mark.parametrize("n_ops, holds", [(0, False), (1, False), (2, True), (3, True)])
def test_history_operations_played_this_turn(n_ops, holds):
    # SPEC 1.2b history: counted from the engine's per-turn counters.
    r = Rules([chronicler("chron", hist("operation_played", "friendly", "turn", min=2)), TICK])
    g = blank(r)
    put(g, r, 0, "back", "chron")
    for _ in range(n_ops):
        cast(g, r, "tick")
    g.step(END)
    assert bases_of(g) == [20, 19 if holds else 20]


def test_history_max_bound():
    r = Rules([chronicler("chron", hist("operation_played", "friendly", "turn", max=0)), TICK])
    g = blank(r)
    put(g, r, 0, "back", "chron")
    g.step(END)
    assert bases_of(g) == [20, 19]
    g.step(END)
    cast(g, r, "tick")
    g.step(END)
    assert bases_of(g) == [20, 19]


def test_history_window_turn_versus_game():
    # SPEC 1.2b windows: "turn" (this turn only) vs "game" (whole game).
    r = Rules([chronicler("chron_turn", hist("operation_played", "friendly", "turn", min=2), amount=1),
               chronicler("chron_game", hist("operation_played", "friendly", "game", min=2), amount=2), TICK])
    g = blank(r)
    put(g, r, 0, "back", "chron_turn")
    put(g, r, 0, "back", "chron_game")
    cast(g, r, "tick")
    g.step(END)
    assert bases_of(g) == [20, 20]
    g.step(END)
    cast(g, r, "tick")
    g.step(END)
    assert bases_of(g) == [20, 18]


def test_history_side_enemy_counts_the_opponents_operations():
    r = Rules([chronicler("chron", hist("operation_played", "enemy", "turn", min=1), scope="any"), TICK])
    g = blank(r)
    put(g, r, 0, "back", "chron")
    cast(g, r, "tick")  # p0's own operation does not count
    g.step(END)
    assert bases_of(g) == [20, 20]
    cast(g, r, "tick")  # p1 plays one on its turn
    g.step(END)
    assert bases_of(g) == [20, 19]


def test_history_units_deployed_this_turn():
    r = Rules([chronicler("chron", hist("unit_deployed", "friendly", "turn", min=2)), unit("plain", 1, 1)])
    g = blank(r)
    put(g, r, 0, "back", "chron")
    cast(g, r, "plain")
    g.step(END)
    assert bases_of(g) == [20, 20]
    g.step(END)
    cast(g, r, "plain")
    cast(g, r, "plain")
    g.step(END)
    assert bases_of(g) == [20, 19]


def test_history_units_died():
    r = Rules([chronicler("chron_e", hist("unit_died", "enemy", "turn", min=2), amount=1),
               chronicler("chron_f", hist("unit_died", "friendly", "game", min=1), amount=4),
               operation("purge", [effect("on_play", "destroy", tgt("all", "enemy"))])])
    g = blank(r)
    put(g, r, 0, "back", "chron_e")
    put(g, r, 0, "back", "chron_f")
    put(g, r, 1, "back", "fill03")
    put(g, r, 1, "back", "fill03")
    cast(g, r, "purge")
    g.step(END)
    assert bases_of(g) == [20, 19]


# ================================================================= target_condition and else (SPEC 1.2b)
def test_target_condition_with_an_else_body():
    # SPEC 1.2b: matching targets receive the action, the others the else body (same targets).
    r = Rules([unit("runner", 1, 4, nature="fast"), unit("gunner", 1, 4, nature="ranged"),
               operation("barrage", [with_else(effect("on_play", "damage", tgt("all", "enemy"), amount=2,
                                                      target_condition={"nature": "troop"}), action="pin")])])
    g = blank(r)
    t = put(g, r, 1, "back", "fill03")
    f = put(g, r, 1, "back", "runner")
    k = put(g, r, 1, "front", "gunner")
    cast(g, r, "barrage")
    assert [(hp(g, u), U(g, u).pinned) for u in (t, f, k)] == [(2, False), (4, True), (4, True)]


def test_target_condition_without_else_gives_nothing_to_the_rest():
    r = Rules([unit("runner", 1, 4, nature="fast"),
               operation("barrage", [effect("on_play", "damage", tgt("all", "enemy"), amount=2,
                                            target_condition={"nature": "troop"})])])
    g = blank(r)
    t = put(g, r, 1, "back", "fill03")
    f = put(g, r, 1, "back", "runner")
    cast(g, r, "barrage")
    assert (hp(g, t), hp(g, f), U(g, f).pinned) == (2, 4, False)


@pytest.mark.parametrize("has_artillery, expected", [(True, [20, 16]), (False, [20, 19])])
def test_else_replaces_the_effect_when_the_condition_fails(has_artillery, expected):
    # SPEC 1.2b: else is resolved instead of the effect when the condition fails; without its own target
    # it uses the effect's targets.
    r = Rules([unit("howitzer", 0, 3, tags=["artillery"]),
               operation("salvo", [with_else(effect("on_play", "damage", "enemy_base", amount=4, condition={
                   "type": "control", "side": "friendly", "filter": {"tag": "artillery"}}),
                   action="damage", amount=1)])])
    g = blank(r)
    if has_artillery:
        put(g, r, 0, "back", "howitzer")
    cast(g, r, "salvo")
    assert bases_of(g) == expected


def test_else_with_its_own_target():
    r = Rules([operation("gamble", [with_else(effect("on_play", "damage", "enemy_base", amount=4,
                                                     condition={"type": "hand_size", "side": "enemy", "min": 5}),
                                              action="draw", amount=1, target="controller")])])
    g = blank(r)
    set_deck(g, r, 0, ["fill05"])
    cast(g, r, "gamble")
    assert bases_of(g) == [20, 20] and list(g.hands[0]) == [r.idx("fill05")]
    set_hand(g, r, 1, ["fill00"] * 5)
    cast(g, r, "gamble")
    assert bases_of(g) == [20, 16] and len(g.hands[0]) == 1


@pytest.mark.parametrize("pick, lost", [(0, 4), (1, 2)])
def test_target_condition_on_a_chosen_target(pick, lost):
    # SPEC 1.2b: target_condition is evaluated after selection, so it does not restrict the options.
    r = Rules([unit("panzer", 2, 6, tags=["tank"]), unit("rifle", 1, 6),
               operation("at_gun", [with_else(effect("on_play", "damage", tgt("chosen", "enemy"), amount=4,
                                                     target_condition={"tag": "tank"}), action="damage", amount=2)])])
    g = blank(r)
    a = put(g, r, 1, "back", "panzer")
    b = put(g, r, 1, "back", "rifle")
    cast(g, r, "at_gun")
    assert chooses(g) == [0, 1]
    g.step(choose(pick))
    assert [hp(g, u) for u in (a, b)] == [6 - (lost if i == pick else 0) for i in range(2)]


def test_else_buff_on_non_matching_targets():
    r = Rules([unit("runner", 1, 4, nature="fast"),
               operation("drill", [with_else(effect("on_play", "buff", tgt("all", "friendly"), atk=2,
                                                    target_condition={"nature": "fast"}), action="buff", hp=1)])])
    g = blank(r)
    f = put(g, r, 0, "back", "runner")
    t = put(g, r, 0, "back", "fill03")
    cast(g, r, "drill")
    assert [(U(g, u).atk, U(g, u).hp, U(g, u).max_hp) for u in (f, t)] == [(3, 4, 4), (1, 5, 5)]


# ================================================================= repeat (SPEC 1.2b)
def simulate_random_repeat(state, units, bases, count, times, amount):
    """SPEC 1.2b repeat with SPEC 2.9 random: each iteration lists the live candidates (units in the given
    order, then bases), draws rng.sample(candidates, count) unless there are <= count, applies the damage
    and removes dead units (the damage step) before the next iteration."""
    rng = random.Random()
    rng.setstate(state)
    units = [list(u) for u in units]
    bases = dict(bases)
    for _ in range(times):
        cands = [("u", u[0]) for u in units] + [("b", p) for p in bases]
        picks = cands if len(cands) <= count else rng.sample(cands, count)
        for kind, key in picks:
            if kind == "u":
                next(u for u in units if u[0] == key)[1] -= amount
            else:
                bases[key] -= amount
        units = [u for u in units if u[1] > 0]
    return {u[0]: u[1] for u in units}, bases, rng.getstate()


def test_repeat_redraws_random_targets_each_time():
    # SPEC 1.2b repeat: the whole clause (selection + action + damage step) runs `repeat` times; random
    # targets are redrawn each time (dead units are no longer candidates).
    r = Rules([unit("w1", 0, 1), unit("w2", 0, 2), unit("w3", 0, 3),
               operation("split", [effect("on_play", "damage", tgt("random", "enemy"), amount=1, repeat=3)])])
    deaths_before_last = 0
    for seed in range(12):
        g = blank(r)
        g.rng.seed(seed)
        for cid in ("w1", "w2", "w3"):
            put(g, r, 1, "back", cid)
        before = [(u.uid, u.hp) for u in g.backline[1]]
        state = g.rng.getstate()
        cast(g, r, "split")
        exp_units, _, exp_state = simulate_random_repeat(state, before, {}, 1, 3, 1)
        assert {u.uid: u.hp for u in g.backline[1]} == exp_units, seed
        assert uids(g.backline[1]) == [u for u, _ in before if u in exp_units]
        assert g.rng.getstate() == exp_state, seed
        partial, _, _ = simulate_random_repeat(state, before, {}, 1, 2, 1)
        deaths_before_last += len(partial) < 3
    assert deaths_before_last > 0  # some runs changed the candidate list mid-repeat


def test_repeat_random_unit_or_base_lists_units_then_the_base():
    # SPEC 1.2b repeat + SPEC 2.13: unit_or_base candidates are units first, then bases.
    r = Rules([unit("w1", 0, 1), unit("w2", 0, 2),
               operation("strafe", [effect("on_play", "damage", tgt("random", "enemy", "unit_or_base"), amount=1,
                                           repeat=3)])])
    base_hit = 0
    for seed in range(12):
        g = blank(r)
        g.rng.seed(100 + seed)
        put(g, r, 1, "back", "w1")
        put(g, r, 1, "front", "w2")
        before = [(u.uid, u.hp) for u in g.backline[1] + g.frontline]
        state = g.rng.getstate()
        cast(g, r, "strafe")
        exp_units, exp_bases, exp_state = simulate_random_repeat(state, before, {1: 20}, 1, 3, 1)
        assert {u.uid: u.hp for u in g.backline[1] + g.frontline} == exp_units, seed
        assert bases_of(g) == [20, exp_bases[1]] and g.rng.getstate() == exp_state, seed
        base_hit += exp_bases[1] < 20
    assert base_hit > 0


def test_repeat_with_select_all():
    r = Rules([unit("w1", 0, 1), unit("w2", 0, 2), unit("w3", 0, 3),
               operation("double", [effect("on_play", "damage", tgt("all", "enemy"), amount=1, repeat=2)])])
    g = blank(r)
    w3 = None
    for cid in ("w1", "w2", "w3"):
        w3 = put(g, r, 1, "back", cid)
    state = g.rng.getstate()
    cast(g, r, "double")
    assert cards_of(g, 1, r) == ["w3"] and hp(g, w3) == 1 and g.rng.getstate() == state
    assert cnt(g, "graveyard", 1, r, "w1") == 1 and cnt(g, "graveyard", 1, r, "w2") == 1


def test_repeat_stops_when_the_game_ends():
    # SPEC 2.8 damage step 1: a base at <= 0 ends the game; nothing more resolves.
    r = Rules([operation("bombard", [effect("on_play", "damage", "enemy_base", amount=3, repeat=3)])])
    g = blank(r)
    g.base_hp[1] = 5
    g.invalidate()
    cast(g, r, "bombard")
    assert g.done and g.winner() == 0 and bases_of(g) == [20, -1]


def test_repeat_on_a_triggered_effect():
    r = Rules([unit("gunner", 1, 3, effects=[effect("on_deploy", "damage", "enemy_base", amount=2, repeat=2)])])
    g = blank(r)
    cast(g, r, "gunner")
    assert bases_of(g) == [20, 16]


# ================================================================= action parameters (SPEC 1.2b)
def test_uncapped_base_heal_and_healing_never_lowers_hp():
    # SPEC 1.2b heal "uncapped": a base may exceed base_hp (cap 99); SPEC 2.13 healing never lowers hp.
    r = Rules([operation("fortify", [effect("on_play", "heal", "friendly_base", amount=5, uncapped=True)]),
               operation("patch", [effect("on_play", "heal", "friendly_base", amount=3)])])
    g = blank(r)
    g.base_hp[0] = 18
    g.invalidate()
    cast(g, r, "fortify")
    assert bases_of(g) == [23, 20] and g.observe(0).my_base_hp == 23 and g.observe(1).opp_base_hp == 23
    cast(g, r, "patch")
    assert bases_of(g) == [23, 20]
    g.base_hp[0] = 97
    g.invalidate()
    cast(g, r, "fortify")
    assert bases_of(g) == [99, 20]


def test_uncapped_heal_still_caps_units_at_max_hp():
    r = Rules([operation("mend", [effect("on_play", "heal", tgt("all", "friendly", "unit_or_base"), amount=4,
                                         uncapped=True)])])
    g = blank(r)
    u = put(g, r, 0, "back", "fill03", hp=2)  # 1/4
    cast(g, r, "mend")
    assert (hp(g, u), U(g, u).max_hp) == (4, 4) and bases_of(g) == [24, 20]


@pytest.mark.parametrize("coins, after", [(5, 2), (2, 0), (1, 0)])
def test_negative_gain_coins_never_below_zero(coins, after):
    # SPEC 1.2b: gain_coins accepts a negative amount; coins never below 0.
    r = Rules([operation("tax", [effect("on_play", "gain_coins", "controller", amount=-2)], cost=1)])
    g = blank(r, coins=coins)
    cast(g, r, "tax")
    assert g.coins[0] == after and g.observe(0).my_coins == after


def test_negative_gain_coins_on_the_opponents_turn():
    r = Rules([unit("taxman", 0, 9, effects=[effect("on_deploy", "gain_coins", "opponent", scope="enemy",
                                                    amount=-3)]),
               unit("plain", 1, 1, cost=1)])
    g = blank(r, coins=5)
    put(g, r, 1, "back", "taxman")
    cast(g, r, "plain")
    assert g.coins[0] == 1
    cast(g, r, "plain")
    assert g.coins[0] == 0


def test_buff_move_cost_delta_and_clamp():
    # SPEC 1.2b buff "move_cost": delta, move cost never below 0; MOVE needs move_cost <= coins (SPEC 2.6).
    r = Rules([unit("trudger", 1, 3, move_cost=2),
               operation("lighten", [effect("on_play", "buff", tgt("all", "friendly"), atk=0, move_cost=-1)], cost=0),
               operation("strip", [effect("on_play", "buff", tgt("all", "friendly"), atk=0, move_cost=-5)], cost=0),
               operation("mire", [effect("on_play", "buff", tgt("all", "enemy"), atk=0, move_cost=2)], cost=0),
               operation("slow", [effect("on_play", "buff", tgt("all", "friendly"), atk=0, move_cost=1)], cost=0)])
    g = blank(r, coins=1)
    t = put(g, r, 0, "back", "trudger")
    e = put(g, r, 1, "back", "trudger")
    assert move(0) not in g.legal_actions()
    cast(g, r, "lighten")
    assert U(g, t).move_cost == 1 and view(g, 0, t).move_cost == 1 and move(0) in g.legal_actions()
    cast(g, r, "strip")
    assert U(g, t).move_cost == 0
    assert U(g, t).static_move_cost == 0  # SPEC 2.12: no static source, so no static contribution
    g.coins[0] = 0
    g.invalidate()
    assert move(0) in g.legal_actions()
    g.step(move(0))
    assert g.coins[0] == 0 and zone_of(g, t.uid) == ("front", 0)
    cast(g, r, "mire")
    assert U(g, e).move_cost == 4 and view(g, 0, e).move_cost == 4
    cast(g, r, "slow")  # the delta applies to the clamped value: 0 + 1
    assert U(g, t).move_cost == 1


def test_buff_move_cost_for_a_turn():
    r = Rules([unit("trudger", 1, 3, move_cost=1),
               operation("bog", [effect("on_play", "buff", tgt("all", "friendly"), atk=0, move_cost=2,
                                        duration="turn")], cost=0)])
    g = blank(r, coins=2)
    t = put(g, r, 0, "back", "trudger")
    cast(g, r, "bog")
    assert U(g, t).move_cost == 3 and move(0) not in g.legal_actions()
    g.step(END)
    assert U(g, t).move_cost == 1


# ================================================================= resolved details (SPEC 2.13)
def test_random_candidates_follow_the_turn_players_board_order():
    # SPEC 2.13: board order = the turn player's backline, frontline if theirs, the opponent's backline,
    # frontline if theirs; it applies to random candidate lists (p1 to act here).
    r = Rules([operation("chaos", [effect("on_play", "damage", tgt("random", "any", count=2), amount=1)])])
    for seed in range(10):
        g = blank(r, current=1, first=0)
        g.rng.seed(seed)
        a0 = put(g, r, 0, "back", "fill03")
        a1 = put(g, r, 0, "back", "fill03")
        f0 = put(g, r, 0, "front", "fill03")
        b0 = put(g, r, 1, "back", "fill03")
        b1 = put(g, r, 1, "back", "fill03")
        cands = [b0.uid, b1.uid, a0.uid, a1.uid, f0.uid]
        assert uids(board_order(g)) == cands
        state = g.rng.getstate()
        cast(g, r, "chaos")
        picks = set(predict_sample(state, cands, 2))
        for u in (a0, a1, f0, b0, b1):
            assert hp(g, u) == (3 if u.uid in picks else 4)


def test_unit_or_base_candidates_are_units_then_controller_base_then_opponent_base():
    # SPEC 2.13: unit_or_base candidates: units first, then bases (controller's, then opponent's).
    r = Rules([operation("scatter", [effect("on_play", "damage", tgt("random", "any", "unit_or_base", count=2),
                                            amount=1)])])
    seen_bases = set()
    for seed in range(16):
        g = blank(r, current=1, first=0)
        g.rng.seed(seed)
        a0 = put(g, r, 0, "back", "fill03")
        b0 = put(g, r, 1, "back", "fill03")
        cands = [("u", b0.uid), ("u", a0.uid), ("b", 1), ("b", 0)]
        state = g.rng.getstate()
        cast(g, r, "scatter")
        picks = predict_sample(state, cands, 2)
        assert hp(g, b0) == (3 if ("u", b0.uid) in picks else 4)
        assert hp(g, a0) == (3 if ("u", a0.uid) in picks else 4)
        assert bases_of(g) == [19 if ("b", 0) in picks else 20, 19 if ("b", 1) in picks else 20]
        seen_bases.update(k for kind, k in picks if kind == "b")
    assert seen_bases == {0, 1}


@pytest.mark.parametrize("what, expected", [("hand", 2 + 4), ("coins", 6 + 2), ("deck", 3 + 1)])
def test_count_with_side_any_sums_both_players(what, expected):
    # SPEC 2.13: {"count": "hand" | "coins" | "deck", "side": "any"} sums both players.
    r = Rules([operation("gauge", [effect("on_play", "damage", "enemy_base", amount={"count": what, "side": "any"})],
                         cost=1)])
    g = blank(r, coins=7)
    g.coins[1] = 2
    set_hand(g, r, 0, ["fill00", "fill01"])
    set_hand(g, r, 1, ["fill00"] * 4)
    set_deck(g, r, 0, ["fill02"] * 3)
    set_deck(g, r, 1, ["fill02"])
    cast(g, r, "gauge")
    assert bases_of(g) == [20, 20 - expected]


def test_simultaneous_deaths_follow_the_turn_players_board_order():
    # SPEC 2.13: death removal follows the board order of the turn player (p1): p1 backline, then p0's
    # backline, then p0's frontline; on_death fires in removal order (SPEC 2.8).
    toks = [unit(f"t{k}", 0, 1, token=True) for k in (1, 2, 3)]
    r = Rules(toks + [unit("m1", 0, 2, effects=[effect("on_death", "summon", "opponent", card="t1")]),
                      unit("m2", 0, 2, effects=[effect("on_death", "summon", "opponent", card="t2")]),
                      unit("m3", 0, 2, effects=[effect("on_death", "summon", "controller", card="t3")]),
                      operation("doom", [effect("on_play", "destroy", tgt("all", "any"))])])
    g = blank(r, current=1, first=0)
    put(g, r, 0, "front", "m2", uid=10)
    put(g, r, 0, "back", "m1", uid=11)
    put(g, r, 1, "back", "m3", uid=12)
    g.next_uid = 20
    cast(g, r, "doom")
    assert g.frontline == [] and g.backline[0] == []
    assert cards_of(g, 1, r) == ["t3", "t1", "t2"]


def test_on_kill_watchers_match_the_killers_side_and_get_the_victim_as_event():
    # SPEC 2.13: on_kill scopes are matched against the killer's side; the event unit is the victim.
    r = Rules([unit("scribe", 0, 9, effects=[effect("on_kill", "damage", "enemy_base", scope="friendly",
                                                    amount={"stat": "atk", "of": "event"})]),
               unit("rival", 0, 9, effects=[effect("on_kill", "damage", "enemy_base", scope="enemy", amount=1)]),
               unit("mourner", 0, 9, effects=[effect("on_kill", "damage", "enemy_base", scope="friendly", amount=5)]),
               unit("reaper", 2, 9), unit("victim", 6, 1)])
    g = blank(r)
    put(g, r, 0, "back", "scribe")
    re = put(g, r, 0, "front", "reaper")
    put(g, r, 1, "back", "victim")
    put(g, r, 1, "back", "rival")
    put(g, r, 1, "back", "mourner")
    g.step(attack(front(0), back(0)))
    assert hp(g, re) == 3
    assert bases_of(g) == [19, 14]


def test_the_attackers_kill_fires_before_the_defenders():
    # SPEC 2.13: the attacker's on_kill fires before the defender's (both died: SPEC 2.13 self effects of a
    # killer that also died still fire).
    r = Rules([unit("tA", 0, 1, token=True), unit("tB", 0, 1, token=True),
               unit("duel_a", 3, 2, effects=[effect("on_kill", "summon", "opponent", card="tA")]),
               unit("duel_b", 3, 3, effects=[effect("on_kill", "summon", "controller", card="tB")])])
    g = blank(r)
    put(g, r, 0, "front", "duel_a")
    put(g, r, 1, "back", "duel_b")
    g.step(attack(front(0), back(0)))
    assert g.frontline == [] and cards_of(g, 1, r) == ["tA", "tB"]


def test_combat_waits_until_every_on_attack_chain_has_resolved():
    # SPEC 2.13: the on_attack chain (here an on_death it causes) resolves before combat damage.
    r = Rules([unit("decoy", 0, 1, token=True, effects=[
        effect("on_death", "buff", tgt("all", "friendly"), atk=2)]),
        unit("target", 1, 5),
        unit("raider", 1, 9, effects=[effect("on_attack", "destroy", tgt("all", "enemy", filter={"token": True}))])])
    g = blank(r)
    a = put(g, r, 0, "front", "raider")
    t = put(g, r, 1, "back", "target")
    put(g, r, 1, "back", "decoy")
    g.step(attack(front(0), back(0)))
    assert (hp(g, a), hp(g, t), U(g, t).atk) == (6, 4, 3)


def test_a_loop_guard_trip_during_on_attack_does_not_cancel_the_combat():
    # SPEC 2.13: a loop-guard trip does not cancel the combat.
    loop = [effect("on_damaged", "heal", "self", amount="full"),
            effect("on_damaged", "damage", tgt("all", "enemy", filter={"token": True}), amount=1)]
    r = Rules([unit("echo_a", 0, 5, token=True, effects=loop), unit("echo_b", 0, 5, token=True, effects=loop),
               unit("target", 0, 3),
               unit("raider", 2, 5, effects=[effect("on_attack", "damage", tgt("all", "enemy", filter={"token": True}),
                                                    amount=1)])],
              max_effect_events=9)
    g = blank(r)
    a = put(g, r, 0, "front", "raider")
    put(g, r, 0, "back", "echo_a")
    put(g, r, 1, "back", "echo_b")
    t = put(g, r, 1, "back", "target")
    g.step(attack(front(0), back(1)))
    assert g.guard_trips == 1
    assert (hp(g, a), hp(g, t)) == (5, 1)
    assert settled(g)


def test_last_known_hp_of_a_dead_unit_is_its_value_at_removal():
    # SPEC 2.13: a dead unit's last-known hp is its value at removal (<= 0), for self and event amounts.
    r = Rules([unit("martyr", 1, 2, effects=[effect("on_death", "damage", "enemy_base",
                                                    amount={"stat": "hp", "of": "self", "plus": 5})]),
               unit("mourner", 0, 9, effects=[effect("on_death", "damage", "enemy_base", scope="enemy",
                                                     amount={"stat": "hp", "of": "event", "plus": 4})]),
               operation("blast", [effect("on_play", "damage", tgt("all", "enemy"), amount=4)])])
    g = blank(r)
    put(g, r, 0, "back", "mourner")
    put(g, r, 1, "back", "martyr")
    cast(g, r, "blast")
    assert g.backline[1] == []
    assert bases_of(g) == [17, 18]  # martyr: -2 + 5 = 3 to p0; mourner: -2 + 4 = 2 to p1


def test_source_zone_is_false_when_the_source_is_not_on_the_board():
    # SPEC 2.13: source_zone is false when the source is not on the board.
    r = Rules([unit("relic", 0, 1, effects=[
        effect("on_death", "damage", "enemy_base", amount=2, condition={"type": "source_zone", "zone": "backline"}),
        effect("on_death", "damage", "enemy_base", amount=1, condition={"type": "source_zone", "zone": "frontline"}),
        effect("on_death", "damage", "enemy_base", amount=4)]),
        operation("execute", [effect("on_play", "destroy", tgt("all", "enemy"))])])
    g = blank(r)
    put(g, r, 1, "back", "relic")
    cast(g, r, "execute")
    assert bases_of(g) == [16, 20]


@pytest.mark.parametrize("limit, p0_base, trips", [(2, 20, 1), (4, 19, 1), (5, 18, 0), (256, 18, 0)])
def test_loop_guard_counts_skips_and_fizzles_and_end_turn_shares_the_budget(limit, p0_base, trips):
    # SPEC 2.13 loop guard: every instance taken off the queue counts (skips and fizzles too); END_TURN and
    # the next turn's start share one budget; it trips at most once per action.
    r = Rules([unit("clock", 0, 9, effects=[
        effect("end_of_turn", "damage", "enemy_base", amount=1, condition={"type": "turn", "whose": "opponent"}),
        effect("end_of_turn", "damage", tgt("all", "enemy", filter={"nature": "fast"}), amount=1),
        effect("end_of_turn", "damage", "enemy_base", amount=1, condition={"type": "turn", "whose": "opponent"})]),
        unit("bell", 0, 9, effects=[effect("start_of_turn", "damage", "enemy_base", amount=1),
                                    effect("start_of_turn", "damage", "enemy_base", amount=1)])],
        max_effect_events=limit)
    g = blank(r)
    put(g, r, 0, "back", "clock")
    put(g, r, 1, "back", "bell")
    g.step(END)
    assert g.current == 1
    assert bases_of(g) == [p0_base, 20] and g.guard_trips == trips


def test_a_game_that_ends_in_a_damage_step_leaves_its_dead_units_on_the_board():
    # SPEC 2.13: dead units stay on the board in the final state (and their on_death never resolves).
    r = Rules([unit("tok", 0, 1, token=True),
               unit("herald", 0, 1, effects=[effect("on_death", "summon", "controller", card="tok")]),
               operation("doomsday", [effect("on_play", "damage", tgt("all", "any", "unit_or_base"), amount=3)])])
    g = blank(r)
    g.base_hp[1] = 3
    g.invalidate()
    h = put(g, r, 1, "back", "herald")
    m = put(g, r, 0, "back", "fill01")  # 2/2
    cast(g, r, "doomsday")
    assert g.done and g.winner() == 0 and bases_of(g) == [17, 0]
    assert (hp(g, h), hp(g, m)) == (-2, -1)
    assert cards_of(g, 1, r) == ["herald"] and cnt(g, "graveyard", 1, r, "herald") == 0


def test_zero_effect_damage_does_not_fire_on_damaged():
    # SPEC 2.13: effect damage of 0 does not fire on_damaged.
    r = Rules([unit("thorn", 1, 5, effects=[effect("on_damaged", "damage", "enemy_base", amount=1)]),
               operation("graze", [effect("on_play", "damage", tgt("all", "enemy"),
                                          amount={"count": "hand", "side": "friendly"})]),
               operation("tap", [effect("on_play", "damage", tgt("all", "enemy"), amount=0)])])
    g = blank(r)
    t = put(g, r, 1, "back", "thorn")
    cast(g, r, "graze")
    cast(g, r, "tap")
    assert hp(g, t) == 5 and bases_of(g) == [20, 20]


@pytest.mark.parametrize("hand_n, amount", [(2, 2), (1, 3), (3, 3), (0, 1)])
def test_discard_takes_the_whole_hand_without_an_rng_draw(hand_n, amount):
    # SPEC 2.13: with n or fewer cards in hand, discard takes all of them without an RNG draw.
    r = Rules([operation("sabotage", [effect("on_play", "discard", "opponent", amount=amount)])])
    g = blank(r)
    cards = ["fill00", "fill01", "fill02"][:hand_n]
    set_hand(g, r, 1, cards)
    state = g.rng.getstate()
    cast(g, r, "sabotage")
    assert list(g.hands[1]) == [] and g.rng.getstate() == state
    for c in cards:
        assert cnt(g, "discard", 1, r, c) == 1 and cnt(g, "revealed", 1, r, c) == 1


def test_retreat_to_hand_is_bookkept_like_return_to_hand():
    # SPEC 2.13: retreat to hand is bookkept like return_to_hand (SPEC 5: known_hand += 1); playing the card
    # again decrements known_hand and reveals nothing new.
    r = Rules([unit("soldier", 2, 3, cost=1),
               operation("withdraw", [effect("on_play", "retreat", tgt("all", "enemy", zone="backline"))])])
    g = blank(r)
    put(g, r, 1, "back", "soldier")
    rev = cnt(g, "revealed", 1, r, "soldier")
    cast(g, r, "withdraw")
    assert list(g.hands[1]) == [r.idx("soldier")] and g.backline[1] == []
    assert cnt(g, "known_hand", 1, r, "soldier") == 1
    assert counts(g.observe(0).opp_known_hand, r.n_cards)[r.idx("soldier")] == 1
    assert cnt(g, "graveyard", 1, r, "soldier") == 0
    g.step(END)
    g.step(play(0))
    assert cnt(g, "known_hand", 1, r, "soldier") == 0 and cnt(g, "revealed", 1, r, "soldier") == rev


def test_retreat_to_a_full_hand_burns_with_no_known_card_change():
    # SPEC 2.13: a card retreated to a full hand burns, with no known-card change.
    r = Rules([unit("soldier", 2, 3),
               operation("fallback", [effect("on_play", "retreat", tgt("all", "enemy", zone="frontline"))])])
    g = blank(r)
    for _ in range(5):
        put(g, r, 1, "back", "fill00")
    put(g, r, 1, "front", "soldier")
    set_hand(g, r, 1, ["fill01"] * 3 + ["fill02"] * 3 + ["fill03"] * 3 + ["fill04"])
    cast(g, r, "fallback")
    assert g.frontline == [] and g.front_owner is None and len(g.hands[1]) == 10
    assert g.burned[1] == 1 and r.idx("soldier") not in g.hands[1]
    assert sum(counts(g.known_hand[1], r.n_cards)) == 0


# ================================================================= loader validation (SPEC 1.2b additions)
def _bad_cards():
    tok = unit("tok", 1, 1, token=True)
    dmg = effect("on_play", "damage", "enemy_base", amount=1)
    return {
        "tags_string": [dict(unit("x"), tags="tank")],
        "tags_not_strings": [dict(unit("x"), tags=[3])],
        "chosen_on_attacked": [unit("x", effects=[effect("on_attacked", "damage", tgt("chosen", "enemy"), amount=1)])],
        "event_filter_on_self": [unit("x", effects=[effect("on_deploy", "damage", "enemy_base", amount=1,
                                                           event_filter={"nature": "troop"})])],
        "event_filter_on_play_nature": [unit("x", effects=[effect("on_play", "damage", "enemy_base", scope="any",
                                                                  amount=1, event_filter={"nature": "troop"})])],
        "event_filter_unknown_key": [unit("x", effects=[effect("on_deploy", "damage", "enemy_base", scope="any",
                                                               amount=1, event_filter={"colour": "red"})])],
        "prev_in_first_clause": [operation("x", [effect("on_play", "damage", PREV, amount=1)])],
        "prev_condition_in_first_clause": [operation("x", [effect("on_play", "damage", "enemy_base", amount=1,
                                                                  condition={"type": "prev", "killed": True})])],
        "prev_other_trigger": [unit("x", effects=[effect("on_deploy", "damage", tgt("all", "enemy"), amount=1),
                                                  effect("on_attack", "pin", PREV)])],
        "else_nested": [operation("x", [with_else(dmg, action="damage", amount=1,
                                                  **{"else": {"action": "damage", "amount": 1}})])],
        "else_kind_mismatch": [operation("x", [with_else(effect("on_play", "damage", tgt("all", "enemy"), amount=1),
                                                         action="draw", amount=1)])],
        "else_unknown_action": [operation("x", [with_else(dmg, action="explode", amount=1)])],
        "else_missing_amount": [operation("x", [with_else(dmg, action="damage")])],
        "repeat_zero": [operation("x", [dict(dmg, repeat=0)])],
        "repeat_string": [operation("x", [dict(dmg, repeat="2")])],
        "uncapped_on_damage": [operation("x", [dict(dmg, uncapped=True)])],
        "negative_heal": [operation("x", [effect("on_play", "heal", "friendly_base", amount=-1)])],
        "adjacent_bad_position": [unit("x", effects=[effect("on_deploy", "pin", adjacent("friendly", "self",
                                                                                         "middle"))])],
        "adjacent_bad_of": [unit("x", effects=[effect("on_deploy", "pin", dict(adjacent("friendly"), of="random"))])],
        "compare_bad_op": [operation("x", [dict(dmg, condition={"type": "compare", "left": 1, "op": "!=",
                                                                "right": 2})])],
        "history_bad_event": [operation("x", [dict(dmg, condition=hist("card_drawn", "friendly", "turn", min=1))])],
        "history_bad_window": [operation("x", [dict(dmg, condition=hist("unit_died", "friendly", "round", min=1))])],
        "event_amount_unknown": [unit("x", effects=[effect("on_damaged", "damage", "enemy_base",
                                                           amount={"event": "heal"})])],
        "stat_unknown": [unit("x", effects=[effect("on_deploy", "damage", "enemy_base",
                                                   amount={"stat": "speed", "of": "self"})])],
        "pinned_filter_not_bool": [operation("x", [effect("on_play", "damage", tgt("all", "enemy",
                                                                                  filter={"pinned": "yes"}),
                                                         amount=1)])],
        "summon_prev_in_first_clause": [tok, operation("x", [effect("on_play", "summon", PREV, card="tok")])],
    }


@pytest.mark.parametrize("name", sorted(_bad_cards()))
def test_loader_rejects_invalid_extensions(name):
    # SPEC 1.1: the loader is strict (ValueError); SPEC 1.2b validation additions.
    with pytest.raises(ValueError):
        Rules(_bad_cards()[name])


def test_loader_accepts_the_valid_extension_forms():
    # Positive control for the validation test above.
    Rules([unit("tok", 1, 1, token=True, tags=["infantry"]),
           unit("x", tags=["tank", "germany"], traits={"ambush": True, "shock": True, "immune": True}, effects=[
               effect("on_attacked", "damage", "event", amount={"stat": "move_cost", "of": "event"}),
               effect("on_attacked", "pin", PREV, condition={"type": "prev", "killed": False}),
               effect("on_attacked", "damage", adjacent("enemy", "prev", "left"), amount=1, repeat=2)]),
           unit("y", effects=[
               effect("on_play", "damage", "enemy_base", scope="enemy", amount=1,
                      event_filter={"max_cost": 3, "tag": "artillery"}),
               with_else(effect("on_death", "damage", tgt("all", "enemy"), scope="friendly", amount=1,
                                event_filter={"pinned": True, "not_tag": ["tank"]},
                                target_condition={"damaged": True}), action="pin"),
               effect("on_damaged", "heal", "friendly_base", amount={"event": "damage"}, uncapped=True,
                      condition={"type": "compare", "left": {"count": "base_hp", "side": "enemy"}, "op": ">=",
                                 "right": 2})]),
           operation("w", [effect("on_play", "gain_coins", "controller", amount=-2,
                                  condition=hist("operation_played", "any", "game", max=3))]),
           operation("z", [effect("on_play", "damage", tgt("all", "enemy"), amount=1),
                           effect("on_play", "buff", PREV, atk=0, move_cost=-1, duration="turn"),
                           with_else(effect("on_play", "damage", "enemy_base", amount=2,
                                            condition={"type": "control", "side": "friendly", "max": 1}),
                                     action="draw", amount=1, target="controller")], tags=["artillery"])])


# ================================================================= sweeps over a pool that uses every extension
def ext_pool():
    rnd = tgt("random", "enemy", "unit_or_base")
    return [
        unit("lurker", 3, 2, cost=2, traits={"ambush": True}),
        unit("striker", 2, 2, nature="fast", cost=2, traits={"shock": True}),
        unit("saint", 1, 3, cost=3, traits={"immune": True}, tags=["medic"]),
        unit("banner", 0, 3, cost=3, tags=["leader"],
             effects=[static("buff", tgt("all", "friendly", filter={"other": True}), atk=1, hp=1)]),
        unit("sergeant", 1, 3, cost=2, effects=[static("buff", adjacent("friendly", "self"), atk=1),
                                                 static("add_trait", "self", trait="smokescreen",
                                                        condition={"type": "turn", "whose": "opponent"})]),
        unit("splasher", 2, 3, cost=3, effects=[effect("on_attack", "damage", adjacent("enemy", "event"), amount=1)]),
        unit("sentinel", 1, 4, cost=2, effects=[effect("on_attacked", "damage", "event", amount=1)]),
        unit("raider", 2, 2, nature="fast", cost=3, effects=[
            effect("on_deploy", "damage", tgt("random", "enemy"), amount=1, repeat=2),
            effect("on_deploy", "pin", tgt("prev"), condition={"type": "prev", "killed": False})]),
        unit("tank", 3, 4, cost=4, tags=["tank"], effects=[
            effect("on_kill", "heal", "friendly_base", amount={"event": "damage"}, uncapped=True)]),
        unit("chronicler", 1, 3, cost=2, effects=[
            effect("end_of_turn", "draw", "controller", amount=1,
                   condition={"type": "history", "event": "operation_played", "side": "friendly", "window": "turn",
                              "min": 2})]),
        operation("volley", [effect("on_play", "damage", rnd, amount=1, repeat=3)], cost=2, tags=["artillery"]),
        operation("at_gun", [with_else(effect("on_play", "damage", tgt("chosen", "enemy"), amount=4,
                                              target_condition={"tag": "tank"}), action="damage", amount=2)], cost=2),
        operation("fortify", [effect("on_play", "heal", "friendly_base", amount=3, uncapped=True),
                              effect("on_play", "gain_coins", "opponent", amount=-1)], cost=1),
        operation("bog", [effect("on_play", "buff", tgt("all", "enemy"), atk=0, move_cost=1, duration="turn"),
                          effect("on_play", "damage", tgt("prev"), amount=1,
                                 condition={"type": "compare", "left": {"count": "units", "side": "enemy"}, "op": ">=",
                                            "right": 3})], cost=1),
        unit("listener", 1, 3, cost=1, effects=[effect("on_play", "damage", "enemy_base", scope="any", amount=1,
                                                       event_filter={"tag": "artillery"})]),
    ]


def ext_rules():
    ext = ext_pool()
    ids = [c["id"] for c in ext]
    deck_a = {"name": "ext_a", "style": "", "cards": {**{f"fill{i:02d}": 2 for i in range(10)},
                                                      **{cid: 2 for cid in ids[:10]}}}
    deck_b = {"name": "ext_b", "style": "", "cards": {**{f"fill{i:02d}": 2 for i in range(4, 14)},
                                                      **{cid: 2 for cid in ids[5:]}}}
    return Rules(ext, mulligan=True, decks=[deck_a, deck_b])


EXT = None


def ext():
    global EXT
    if EXT is None:
        EXT = ext_rules()
    return EXT


def ext_game(seed):
    r = ext()
    g = Game(r.config)
    g.reset(seed, cards_mod.sample_deal(seed, r.config, 0.5))
    return g


@pytest.mark.parametrize("seed", range(4))
def test_extension_pool_is_deterministic_and_clones_replay(seed):
    # SPEC 4: same seed, decks and actions => identical states (random repeats, statics, ambush included);
    # clone() is independent and replays identically.
    g1, g2 = ext_game(seed), ext_game(seed)
    pol = random.Random(seed)
    for i in range(600):
        assert state_key(g1) == state_key(g2)
        if g1.done:
            break
        if i == 150:
            c = g1.clone()
            pol_c = random.Random(seed + 99)
            a, b = c.clone(), c.clone()
            for _ in range(80):
                if a.done:
                    break
                act = pol_c.choice(a.legal_actions())
                a.step(act)
                b.step(act)
                assert state_key(a) == state_key(b)
        act = pol.choice(g1.legal_actions())
        g1.step(act)
        g2.step(act)


@pytest.mark.parametrize("seed", range(4))
def test_extension_pool_no_leak_and_determinize(seed):
    # SPEC 5 no-leak and SPEC 4 determinize guarantees on a pool with the phase-1b mechanics.
    rng = random.Random(1000 + seed)
    g = ext_game(seed)
    pol = random.Random(seed * 7 + 1)
    for i in range(500):
        if g.done:
            break
        if i % 6 == 0:
            for p in (0, 1):
                obs = g.observe(p)
                deciding = g.current_player() == p
                for nd in (False, True):
                    h = perturb_hidden(g, p, rng, new_decklist=nd)
                    assert h.observe(p) == obs
                    if deciding:
                        assert h.legal_actions() == g.legal_actions()
                d = g.determinize(p, random.Random(i))
                assert d.observe(p) == obs
                if deciding:
                    assert d.legal_actions() == g.legal_actions()
        g.step(pick(g, pol))
