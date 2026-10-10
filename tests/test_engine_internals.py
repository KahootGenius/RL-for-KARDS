"""Focused tests of the Stage 3 engine internals (SPEC §1-§5): the strict effect loader, random decks
and deals, deck specs in reset(), per-pool trigger tables, clone() with a pending choice and a queue,
the loop guard, determinize(), and a fuzz over an effect-heavy pool (every trigger, action, select,
side, kind, zone, amount form, condition and duration) with the mulligan on.

Phase 1b (SPEC §1.2b, §2.8b, §2.12): the loader for every schema extension (tags, ambush / shock /
immune, on_attacked, static, event_filter, prev / adjacent, new filter keys, amounts and conditions,
target_condition + else, repeat, uncapped heals, negative coins, buff move_cost), clause batches in
clone(), the static recompute (fast path, layering with buffs and expiry), combat_damage with the new
traits, history counters, and the same fuzz with every new primitive in the pool.

The rule-by-rule Stage 3 tests live in test_effects.py, test_choices.py, test_mulligan.py,
test_keywords.py, test_determinize.py and test_no_leak_stage3.py.
"""
from __future__ import annotations

import copy
import random
from collections import Counter

import numpy as np
import pytest

from cardgame.cards import (AMBUSH_BIT, DEFENSE_BIT, FULL, IMMUNE_BIT, MAX_OPERATIONS, SHOCK_BIT, SMOKESCREEN_BIT,
                            AmountDef, ConditionDef, EffectDef, FilterDef, TargetDef, build_ruleset, cost_bucket,
                            deck_rng, generate_deck, load_ruleset, sample_deal)
from cardgame.engine import (CHOICE, MAIN, MULLIGAN, T_ATTACKED, T_DAMAGED, T_DEATH, T_DEPLOY, T_KILL, T_PLAY,
                             T_STATIC, Game, IllegalActionError, PendingView, Unit, UnitView, ambush_fires,
                             combat_damage)
from conftest import (CONFIG, FIXTURES, VANILLA_CONFIG, add_unit, attack, blank_game, check_invariants, choose,
                      play, set_hand, state_key)


# ---------------------------------------------------------------- an effect-heavy pool
def U(cid, nature, cost, atk, hp, *effects, traits=None, token=False, move_cost=None, tags=None):
    d = {"id": cid, "name": cid.replace("_", " ").title(), "type": "unit", "nature": nature, "cost": cost,
         "attack": atk, "health": hp, "traits": traits or {}, "effects": list(effects)}
    if token:
        d["token"] = True
    if move_cost is not None:
        d["move_cost"] = move_cost
    if tags is not None:
        d["tags"] = list(tags)
    return d


def O(cid, cost, *effects, token=False, tags=None):
    d = {"id": cid, "name": cid.replace("_", " ").title(), "type": "operation", "cost": cost, "effects": list(effects)}
    if token:
        d["token"] = True
    if tags is not None:
        d["tags"] = list(tags)
    return d


def E(trigger, action, target, **kw):
    return {"trigger": trigger, "action": action, "target": target, **kw}


def T(select, side=None, kind=None, **kw):
    t = {"select": select, **kw}
    if side is not None:
        t["side"] = side
    if kind is not None:
        t["kind"] = kind
    return t


ENEMY_UNIT = T("chosen", "enemy", "unit")
FRIENDLY_UNIT = T("chosen", "friendly", "unit")
ALL_ENEMY = T("all", "enemy", "unit")
ALL_FRIENDLY = T("all", "friendly", "unit")

EFFECT_CARDS = [
    # tokens
    U("conscript", "troop", 1, 1, 1, token=True),
    U("jeep", "fast", 1, 1, 1, traits={"blitz": True}, token=True),
    O("supply", 1, E("on_play", "gain_coins", "controller", amount=1), token=True),
    # units
    U("grenadier", "troop", 3, 2, 3, E("on_deploy", "damage", ENEMY_UNIT, amount=1), traits={"defense": True}),
    U("medic", "troop", 2, 1, 3, E("on_deploy", "heal", T("chosen", "friendly", "unit_or_base"), amount=2)),
    U("bomber", "ranged", 4, 3, 2, E("on_death", "damage", T("random", "enemy", "unit"), amount=2),
      E("on_death", "damage", "enemy_base", amount=1)),
    U("sapper", "fast", 3, 2, 2, E("on_attack", "buff", "self", atk=1, duration="turn"), traits={"blitz": True}),
    U("sentinel", "troop", 4, 2, 5, E("on_damaged", "damage", "event", amount=1),
      traits={"defense": True, "armor": 1}),
    U("spotter", "fast", 2, 1, 1, E("on_move", "draw", "controller", amount=1), traits={"smokescreen": True}),
    U("berserker", "troop", 4, 3, 3, E("on_kill", "gain_coins", "controller", amount=1), traits={"fury": True}),
    U("commander", "troop", 5, 3, 4, E("start_of_turn", "buff", T("all", "friendly", "unit", zone="frontline"),
                                       atk=1, duration="turn")),
    U("quartermaster", "troop", 3, 2, 3, E("end_of_turn", "increase_max_coins", "controller", amount=1,
                                           condition={"type": "control", "side": "friendly", "min": 2})),
    U("saboteur", "fast", 3, 2, 2, E("on_deploy", "discard", "opponent", amount=1)),
    U("recruiter", "troop", 3, 1, 2, E("on_deploy", "summon", "controller", card="conscript", amount=2)),
    U("bouncer", "ranged", 3, 1, 2, E("on_deploy", "return_to_hand",
                                      T("chosen", "enemy", "unit", filter={"max_cost": 3}))),
    U("pinner", "ranged", 3, 2, 2, E("on_attack", "pin", "event")),
    U("mechanic", "troop", 2, 1, 2, E("on_deploy", "add_trait", T("chosen", "friendly", "unit",
                                                                  filter={"other": True}), trait="armor", amount=1)),
    U("jammer", "troop", 3, 2, 2, E("on_deploy", "remove_trait", ALL_ENEMY, trait=["defense", "smokescreen"],
                                    duration="turn")),
    U("mourner", "troop", 3, 2, 3, E("on_death", "buff", "self", scope="friendly", atk=1, hp=1)),
    U("avenger", "troop", 4, 3, 3, E("on_death", "gain_coins", "controller", scope="enemy", amount=1)),
    U("tripwire", "troop", 2, 1, 3, E("on_deploy", "damage", "event", scope="any", amount=1,
                                      condition={"type": "source_zone", "zone": "backline"})),
    U("lookout", "troop", 3, 2, 3, E("on_attack", "add_trait", "self", scope="enemy", trait="smokescreen",
                                     duration="turn")),
    U("nurse", "troop", 3, 2, 4, E("on_damaged", "heal", "event", scope="friendly", amount=1)),
    U("ambusher", "troop", 2, 2, 2, E("on_move", "damage", "event", scope="enemy", amount=1)),
    U("scholar", "troop", 3, 2, 2, E("on_kill", "draw", "controller", scope="any", amount=1,
                                     condition={"type": "hand_size", "side": "friendly", "max": 7})),
    U("flak", "ranged", 3, 2, 2, E("on_play", "damage", T("random", "enemy", "unit_or_base"), scope="enemy",
                                   amount=1)),
    U("marshal", "troop", 2, 2, 2, E("on_deploy", "retreat", T("chosen", "any", "unit", zone="frontline"))),
    U("clerk", "troop", 2, 1, 2, E("on_deploy", "add_card", "controller", card="supply")),
    U("executioner", "troop", 6, 4, 4, E("on_deploy", "destroy", T("chosen", "enemy", "unit",
                                                                   filter={"damaged": True}))),
    U("drillmaster", "troop", 4, 2, 4, E("on_deploy", "damage", T("all", "enemy", "unit", zone="backline"),
                                         amount={"count": "units", "side": "friendly",
                                                 "filter": {"nature": "troop"}, "plus": -1})),
    U("sniper", "ranged", 5, 3, 3, E("on_deploy", "damage", "enemy_base", amount={"stat": "atk", "of": "self"},
                                     condition={"type": "base_hp", "side": "enemy", "max": 12})),
    U("vanguard", "troop", 3, 2, 3, E("on_deploy", "draw", "controller", amount=1,
                                      condition={"type": "frontline", "owner": "friendly"})),
    U("vampire", "troop", 4, 3, 3, E("on_kill", "heal", "self", amount={"stat": "max_hp", "of": "event"})),
    U("banker", "troop", 2, 1, 2, E("on_deploy", "gain_coins", "controller",
                                    amount={"count": "hand", "side": "enemy", "times": 0, "plus": 1},
                                    condition=[{"type": "turn", "whose": "own"}])),
    U("echo", "troop", 5, 1, 9, E("on_damaged", "damage", T("random", "enemy", "unit"), amount=1),
      E("on_damaged", "heal", "self", amount=1)),
    U("thief", "fast", 4, 2, 3, E("on_attack", "damage", T("chosen", "enemy", "unit_or_base"), amount=1),
      E("on_deploy", "draw", "controller", amount={"count": "deck", "side": "friendly", "times": 0, "plus": 1})),
    U("warlord", "troop", 6, 4, 5, E("on_deploy", "add_trait", FRIENDLY_UNIT, trait="fury", duration="turn"),
      E("on_deploy", "add_trait", FRIENDLY_UNIT, trait="blitz", duration="turn")),
    U("hexer", "troop", 4, 2, 3, E("on_deploy", "buff", ALL_ENEMY, atk=-1, duration="turn")),
    U("coiner", "troop", 3, 2, 2, E("on_deploy", "damage", T("all", "any", "unit", filter={
        "trait": "armor", "min_hp": 2, "token": False, "not_trait": ["fury"], "nature": ["troop", "ranged"],
        "min_cost": 1, "max_atk": 9}), amount=1),
      E("on_deploy", "gain_coins", "controller", amount={"count": "coins", "side": "any", "times": 0, "plus": 1})),
    U("heavy", "troop", 8, 6, 7, traits={"armor": 2}),
    U("brute", "troop", 7, 6, 6),
    U("militia", "troop", 1, 1, 2),
    U("runner", "fast", 1, 2, 1),
    U("slinger", "ranged", 1, 1, 1, move_cost=0),
    U("guard", "troop", 2, 1, 4, traits={"defense": True}),
    U("jumper", "fast", 5, 4, 3, traits={"blitz": True}),
    # operations
    O("barrage", 3, E("on_play", "damage", ALL_ENEMY, amount=1)),
    O("strike", 2, E("on_play", "damage", T("chosen", "enemy", "unit_or_base"), amount=3)),
    O("reinforce", 2, E("on_play", "buff", FRIENDLY_UNIT, atk=2, hp=2), E("on_play", "draw", "controller", amount=1)),
    O("purge", 4, E("on_play", "destroy", T("random", "enemy", "unit", count=2, filter={"not_trait": "defense"}))),
    O("triage", 3, E("on_play", "heal", T("all", "friendly", "unit_or_base"), amount="full")),
    O("snare", 1, E("on_play", "pin", T("chosen", "enemy", "unit", zone="frontline"))),
    O("loot", 2, E("on_play", "draw", "controller", amount={"count": "deck", "side": "friendly", "times": 0,
                                                             "plus": 2})),
    O("raid", 2, E("on_play", "discard", "opponent", amount=2)),
    O("frenzy", 3, E("on_play", "add_trait", ALL_FRIENDLY, trait="fury", duration="turn"),
      E("on_play", "add_trait", ALL_FRIENDLY, trait="blitz", duration="turn")),
    O("recall", 1, E("on_play", "return_to_hand", FRIENDLY_UNIT)),
    O("volley", 2, E("on_play", "damage", ENEMY_UNIT, amount=1), E("on_play", "damage", ENEMY_UNIT, amount=1)),
    O("inflation", 3, E("on_play", "increase_max_coins", "opponent", amount=-1),
      E("on_play", "increase_max_coins", "controller", amount=1)),
    O("conscription", 4, E("on_play", "summon", "controller", card="conscript", amount=3),
      E("on_play", "summon", "opponent", card="jeep")),
    O("fortify", 2, E("on_play", "buff", ALL_FRIENDLY, hp={"stat": "cost", "of": "self"})),
    O("airstrike", 5, E("on_play", "damage", T("random", "any", "unit_or_base", count=3), amount=2)),
    O("fallback", 1, E("on_play", "retreat", T("all", "friendly", "unit", zone="frontline"))),
    O("sharpen", 2, E("on_play", "remove_trait", T("chosen", "enemy", "unit"), trait="armor"),
      E("on_play", "add_trait", ALL_FRIENDLY, trait="defense")),
    # ---- phase 1b (SPEC §1.2b): every new primitive
    U("scout_car", "fast", 1, 1, 1, traits={"shock": True}, token=True, tags=["tank"]),
    U("bushwhacker", "troop", 3, 2, 3, traits={"ambush": True}, tags=["infantry"]),
    U("stormer", "fast", 3, 2, 2, traits={"shock": True, "blitz": True}, tags=["tank"]),
    U("ghost", "troop", 5, 1, 4, traits={"immune": True}, tags=["spirit"]),
    U("sentry", "troop", 2, 1, 4, E("on_attacked", "damage", "event", amount=1), tags=["infantry"]),
    U("signal", "troop", 3, 1, 3, E("on_attacked", "buff", "self", scope="friendly", atk=1,
                                    event_filter={"nature": ["troop", "fast"]})),
    U("banner", "troop", 4, 1, 3, E("static", "buff", T("all", "friendly", filter={"other": True}), atk=1, hp=1),
      tags=["leader"]),
    U("smokepot", "troop", 3, 1, 3, E("static", "add_trait", T("adjacent", "friendly", of="self"),
                                      trait="smokescreen")),
    U("depot", "troop", 3, 1, 4, E("static", "buff", T("all", "friendly", filter={"other": True}), move_cost=-1,
                                   condition={"type": "control", "side": "friendly", "min": 3},
                                   **{"else": {"action": "buff", "target": "self", "atk": 1}})),
    U("warden", "troop", 4, 2, 4, E("static", "add_trait", "self", trait="defense",
                                    condition={"type": "compare", "left": {"stat": "hp", "of": "self"}, "op": "<",
                                               "right": {"stat": "max_hp", "of": "self"}})),
    U("gloom", "ranged", 4, 1, 3, E("static", "buff", T("all", "enemy"), atk=-1, target_condition={"not_tag": "tank"},
                                    **{"else": {"action": "add_trait", "trait": "ambush"}})),
    O("grenade", 3, E("on_play", "damage", ENEMY_UNIT, amount=2),
      E("on_play", "damage", T("adjacent", "enemy", of="prev"), amount=1),
      E("on_play", "draw", "controller", amount=1, condition={"type": "prev", "killed": True})),
    U("marksman", "ranged", 4, 2, 2, E("on_deploy", "damage", T("random", "enemy", "unit"), amount=1, repeat=3),
      E("on_deploy", "pin", T("prev", filter={"damaged": True}))),
    U("interrogator", "troop", 3, 2, 2, E("on_deploy", "damage", ENEMY_UNIT, amount=1, target_condition={"tag": "tank"},
                                          **{"else": {"action": "pin"}}),
      E("on_deploy", "gain_coins", "controller", amount={"stat": "move_cost", "of": "prev"},
        condition={"type": "prev", "filter": {"pinned": True}})),
    U("flanker", "fast", 4, 3, 2, E("on_attack", "damage", T("adjacent", of="event", position="both"), amount=1)),
    O("double_tap", 2, E("on_play", "damage", "enemy_base", amount=1), E("on_play", "damage", T("prev"), amount=1)),
    U("pragmatist", "troop", 3, 2, 3, E("on_deploy", "damage", "enemy_base", amount=2,
                                        condition={"type": "frontline", "owner": "friendly"},
                                        **{"else": {"action": "draw", "target": "controller", "amount": 1}})),
    U("veteran", "troop", 3, 2, 2, E("on_deploy", "buff", "self", atk=2, hp=2,
                                     condition={"type": "history", "event": "unit_died", "side": "any",
                                                "window": "turn"})),
    U("strategist", "troop", 3, 1, 3, E("end_of_turn", "draw", "controller", amount=1,
                                        condition={"type": "history", "event": "operation_played",
                                                   "side": "friendly", "window": "game", "min": 2})),
    U("duelist", "fast", 3, 2, 2, E("on_attack", "buff", "self", atk=2, duration="turn",
                                    condition={"type": "compare", "left": {"stat": "atk", "of": "event"}, "op": ">=",
                                               "right": {"stat": "atk", "of": "self"}})),
    U("avenger2", "troop", 4, 2, 5, E("on_damaged", "damage", T("random", "enemy", "unit_or_base"),
                                      amount={"event": "damage"})),
    U("reaper", "troop", 4, 3, 3, E("on_kill", "heal", "friendly_base", amount={"event": "damage"}, uncapped=True),
      E("on_kill", "summon", "controller", scope="friendly", card="scout_car", event_filter={"tag": "infantry"})),
    U("deployer", "troop", 2, 1, 2, E("on_deploy", "add_trait", "event", scope="friendly", trait="ambush",
                                      event_filter={"max_cost": 2, "token": False}),
      E("on_play", "gain_coins", "controller", scope="any", amount=1, event_filter={"tag": "artillery"})),
    O("tax", 2, E("on_play", "gain_coins", "opponent", amount=-2),
      E("on_play", "heal", "friendly_base", amount={"count": "base_hp", "side": "enemy", "times": 0, "plus": 2}),
      tags=["artillery"]),
    O("convoy", 1, E("on_play", "buff", ALL_FRIENDLY, move_cost=-1, duration="turn"),
      E("on_play", "add_trait", T("chosen", "friendly", "unit", filter={"pinned": False}), trait="immune",
        duration="turn")),
    O("bombard", 4, E("on_play", "damage", T("all", "enemy", "unit", filter={"tag": ["tank", "infantry"]}), amount=2),
      tags=["artillery"]),
    O("entrench", 2, E("on_play", "add_trait", FRIENDLY_UNIT, trait="ambush"),
      E("on_play", "remove_trait", T("all", "enemy", "unit"), trait=["shock", "immune"], duration="turn")),
]
NON_TOKEN_IDS = [c["id"] for c in EFFECT_CARDS if not c.get("token")]


def _deck_dict(ids, size=40, copies=3) -> dict:
    counts: dict = {}
    k = 0
    while sum(counts.values()) < size:
        cid = ids[k % len(ids)]
        if counts.get(cid, 0) < copies:
            counts[cid] = counts.get(cid, 0) + 1
        k += 1
    return counts


EFFECT_DECKS = [{"name": "mix", "style": "test", "cards": _deck_dict(NON_TOKEN_IDS)},
                {"name": "ops", "style": "test", "cards": _deck_dict(NON_TOKEN_IDS[::-1])}]
EFFECT_CONFIG = build_ruleset(EFFECT_CARDS, EFFECT_DECKS)            # mulligan on (the default)
EFFECT_CONFIG_NM = build_ruleset(EFFECT_CARDS, EFFECT_DECKS, mulligan=False)
IDX = {c.id: c.index for c in EFFECT_CONFIG.cards.cards}


def eblank(**kw) -> Game:
    """A hand-built position on the effect pool (mulligan done, phase MAIN)."""
    kw.setdefault("decks", (0, 1))
    return blank_game(config=EFFECT_CONFIG, **kw)


# ---------------------------------------------------------------- loader (SPEC §1.1-1.2)
def test_effect_pool_parses_into_frozen_dataclasses():
    pool = EFFECT_CONFIG.cards
    g = pool.by_id("grenadier")
    assert g.effects == (EffectDef(index=0, trigger="on_deploy", scope="self", action="damage",
                                   target=TargetDef(select="chosen", side="enemy", kind="unit"),
                                   amount=AmountDef(kind="literal", value=1)),)
    assert g.defense and not g.token and g.is_unit
    with pytest.raises(Exception):
        g.effects[0].action = "heal"  # frozen
    rec = pool.by_id("recruiter").effects[0]
    assert (rec.card, rec.card_id, rec.amount.value) == (IDX["conscript"], "conscript", 2)
    assert rec.target == TargetDef(select="all", side="friendly", kind="player")  # "controller"
    assert pool.by_id("bomber").effects[1].target == TargetDef(select="all", side="enemy", kind="base")
    assert pool.by_id("commander").effects[0].scope == "friendly"  # turn triggers default to friendly
    assert pool.by_id("commander").effects[0].duration == "turn"
    assert pool.by_id("quartermaster").effects[0].condition == (ConditionDef(type="control", side="friendly", min=1 + 1),)
    assert pool.by_id("banker").effects[0].condition == (ConditionDef(type="turn", whose="own"),)
    assert pool.by_id("triage").effects[0].amount == AmountDef(kind="full", value=FULL)
    coiner = pool.by_id("coiner").effects[0].target.filter
    assert coiner == FilterDef(natures=(0, 2), traits=2, not_traits=16, min_hp=2, token=False, min_cost=1, max_atk=9)
    drill = pool.by_id("drillmaster").effects[0].amount
    assert (drill.kind, drill.count, drill.side, drill.plus, drill.filter.natures) == ("count", "units", "friendly",
                                                                                       -1, (0,))
    hexer = pool.by_id("hexer").effects[0]
    assert (hexer.atk.value, hexer.hp.value) == (-1, 0)
    assert pool.by_id("jammer").effects[0].trait == ("defense", "smokescreen")
    sup = pool.by_id("supply")
    assert sup.is_operation and sup.token and sup.nature == -1 and sup.attack == sup.health == 0
    assert pool.by_id("jeep").blitz and pool.by_id("spotter").smokescreen and pool.by_id("berserker").fury
    assert hash(EFFECT_CONFIG) == hash(build_ruleset(EFFECT_CARDS, EFFECT_DECKS))
    assert EFFECT_CONFIG == build_ruleset(EFFECT_CARDS, EFFECT_DECKS)


def test_effect_pool_covers_the_whole_schema():
    effs = [e for c in EFFECT_CONFIG.cards.cards for e in c.effects]
    from cardgame.cards import (ACTIONS, AMOUNT_COUNTS, BOOL_TRAITS, CONDITION_TYPES, HISTORY_EVENTS, KINDS,
                                SELECTS, SIDES, TRIGGERS, ZONES)
    assert {e.trigger for e in effs} == set(TRIGGERS)
    assert {e.action for e in effs} == set(ACTIONS)
    assert {e.target.select for e in effs} == set(SELECTS)
    assert {e.target.side for e in effs} - {""} == set(SIDES)
    assert {e.target.kind for e in effs} == set(KINDS)
    assert {e.target.zone for e in effs} == set(ZONES)
    assert {c.type for e in effs for c in e.condition} == set(CONDITION_TYPES)
    assert {e.scope for e in effs} == {"self", "friendly", "enemy", "any"}
    assert {e.duration for e in effs} == {"permanent", "turn"}
    amounts = [a for e in effs for a in (e.amount, e.atk, e.hp) if a is not None]
    amounts += [a for e in effs for c in e.condition if c.type == "compare" for a in (c.left, c.right)]
    assert {a.kind for a in amounts} == {"literal", "full", "stat", "count", "event"}
    assert {a.count for a in amounts if a.kind == "count"} == set(AMOUNT_COUNTS)
    assert {(a.stat, a.of) for a in amounts if a.kind == "stat"} >= {("atk", "self"), ("max_hp", "event"),
                                                                     ("move_cost", "prev"), ("atk", "event")}
    # phase 1b extras
    cards = EFFECT_CONFIG.cards.cards
    assert {t for c in cards for t in c.tags} >= {"tank", "infantry", "artillery"}
    assert all(any(getattr(c, t) for c in cards) for t in BOOL_TRAITS)
    assert {e.target.of for e in effs if e.target.select == "adjacent"} == {"self", "event", "prev"}
    assert any(e.event_filter is not None for e in effs if e.trigger == "on_play")
    assert any(e.event_filter is not None for e in effs if e.trigger != "on_play")
    assert any(e.target_condition is not None and e.else_ is not None for e in effs if e.trigger != "static")
    assert any(e.target_condition is not None and e.else_ is not None for e in effs if e.trigger == "static")
    assert any(e.else_ is not None and e.else_.own_target for e in effs)
    assert any(e.repeat > 1 for e in effs) and any(e.uncapped for e in effs) and any(e.move_cost for e in effs)
    assert any(e.action == "gain_coins" and e.amount.kind == "literal" and e.amount.value < 0 for e in effs)
    assert {c.event for e in effs for c in e.condition if c.type == "history"} < set(HISTORY_EVENTS)
    assert {(e.action, e.trait) for e in effs if e.trigger == "static"} >= {("add_trait", ("smokescreen",)),
                                                                           ("add_trait", ("defense",))}
    filters = [f for e in effs for f in (e.target.filter, e.event_filter, e.target_condition) if f is not None]
    assert any(f.tags for f in filters) and any(f.not_tags for f in filters) and any(f.pinned is not None
                                                                                     for f in filters)


GOOD = U("x", "troop", 2, 1, 2)
TOKEN = U("tok", "troop", 1, 1, 1, token=True)
OPTOKEN = O("optok", 1, E("on_play", "draw", "controller", amount=1), token=True)


def _pool(*cards):
    """build_ruleset with the given cards plus 14 filler units (for a legal deck)."""
    filler = [U(f"f{i}", "troop", 1 + i % 8, 1, 1) for i in range(14)]
    deck = {"cards": {f"f{i}": 3 for i in range(13)} | {"f13": 1}}
    return build_ruleset([*filler, *cards], [deck])


def _unit_with(*effects, **kw):
    return U("x", "troop", 2, 1, 2, *effects, **kw)


@pytest.mark.parametrize("card", [
    _unit_with(E("on_deploy", "damage", ENEMY_UNIT, amount=1, typo=1)),                  # unknown effect key
    _unit_with(E("on_spawn", "damage", ENEMY_UNIT, amount=1)),                           # unknown trigger
    _unit_with(E("on_deploy", "explode", ENEMY_UNIT, amount=1)),                         # unknown action
    _unit_with({"trigger": "on_deploy", "action": "damage", "amount": 1}),               # no target
    O("op", 1, E("on_deploy", "draw", "controller", amount=1)),                          # operation, not on_play
    O("op", 1, E("on_play", "draw", "controller", amount=1, scope="friendly")),          # operation scope
    O("op", 1),                                                                          # operation without effects
    {**O("op", 1, E("on_play", "draw", "controller", amount=1)), "attack": 1},          # unit stat on an operation
    _unit_with(E("on_play", "draw", "controller", amount=1)),                            # unit on_play + self
    _unit_with(E("start_of_turn", "draw", "controller", amount=1, scope="self")),        # turn trigger scope self
    _unit_with(E("on_death", "damage", ENEMY_UNIT, amount=1)),                           # chosen on a death
    _unit_with(E("on_deploy", "damage", ENEMY_UNIT, amount=1, scope="friendly")),        # chosen, watcher scope
    _unit_with(E("end_of_turn", "damage", ENEMY_UNIT, amount=1)),                        # chosen at turn end
    _unit_with(E("on_deploy", "damage", "event", amount=1)),                             # no event unit
    _unit_with(E("on_move", "damage", "event", amount=1)),
    _unit_with(E("on_play", "damage", "event", amount=1, scope="enemy")),                # operations: no event unit
    _unit_with(E("on_deploy", "draw", "controller", amount={"stat": "atk", "of": "event"})),
    _unit_with(E("on_death", "buff", "self", atk=1)),                                    # on_death targets self
    O("op", 1, E("on_play", "buff", "self", atk=1)),                                     # operation has no self
    O("op", 1, E("on_play", "damage", "enemy_base", amount={"stat": "atk", "of": "self"})),
    _unit_with(E("on_deploy", "damage", "controller", amount=1)),                        # damage a player
    _unit_with(E("on_deploy", "draw", ENEMY_UNIT, amount=1)),                            # draw a unit
    _unit_with(E("on_deploy", "draw", T("random", "friendly", "player"), amount=1)),     # player needs select all
    _unit_with(E("on_deploy", "damage", T("all", "enemy", "unit", count=2), amount=1)),  # count without random
    _unit_with(E("on_deploy", "damage", T("random", "enemy", "unit", count=0), amount=1)),
    _unit_with(E("on_deploy", "damage", T("all", None, "unit"), amount=1)),              # side required
    _unit_with(E("on_deploy", "buff", T("self", "friendly"), atk=1)),                    # side with self
    _unit_with(E("on_deploy", "damage", T("all", "enemy", "base", zone="backline"), amount=1)),  # zone on bases
    _unit_with(E("on_deploy", "draw", T("all", "enemy", "player", filter={"token": True}), amount=1)),
    _unit_with(E("on_deploy", "damage", T("all", "enemy", "hero"), amount=1)),           # unknown kind
    _unit_with(E("on_deploy", "damage", "my_base", amount=1)),                           # unknown shorthand
    _unit_with(E("on_deploy", "destroy", ENEMY_UNIT, amount=1)),                         # amount not taken
    _unit_with(E("on_deploy", "damage", ENEMY_UNIT)),                                    # amount missing
    _unit_with(E("on_deploy", "damage", ENEMY_UNIT, amount="full")),                     # full only for heal
    _unit_with(E("on_deploy", "damage", ENEMY_UNIT, amount=-1)),
    _unit_with(E("on_deploy", "damage", ENEMY_UNIT, amount=True)),
    _unit_with(E("on_deploy", "damage", ENEMY_UNIT, amount=1.5)),
    _unit_with(E("on_deploy", "damage", ENEMY_UNIT, amount={"stat": "atk"})),             # missing "of"
    _unit_with(E("on_deploy", "damage", ENEMY_UNIT, amount={"count": "units"})),          # missing side
    _unit_with(E("on_deploy", "damage", ENEMY_UNIT, amount={"count": "hand", "side": "enemy", "zone": "board"})),
    _unit_with(E("on_deploy", "damage", ENEMY_UNIT, amount={"stat": "speed", "of": "self"})),
    _unit_with(E("on_deploy", "damage", ENEMY_UNIT, amount={"count": "units", "side": "enemy", "times": 1.0})),
    _unit_with(E("on_deploy", "damage", ENEMY_UNIT, amount=1, duration="turn")),          # duration not taken
    _unit_with(E("on_deploy", "buff", FRIENDLY_UNIT, duration="turn")),                  # buff needs atk/hp
    _unit_with(E("on_deploy", "buff", FRIENDLY_UNIT, hp=-1)),
    _unit_with(E("on_deploy", "buff", FRIENDLY_UNIT, atk=1, duration="forever")),
    _unit_with(E("on_deploy", "add_trait", FRIENDLY_UNIT, trait="taunt")),
    _unit_with(E("on_deploy", "add_trait", FRIENDLY_UNIT, trait=["fury", "blitz"])),
    _unit_with(E("on_deploy", "add_trait", FRIENDLY_UNIT, trait="fury", amount=1)),       # amount: armor only
    _unit_with(E("on_deploy", "add_trait", FRIENDLY_UNIT, trait="armor", amount=0)),
    _unit_with(E("on_deploy", "remove_trait", FRIENDLY_UNIT, trait=[])),
    _unit_with(E("on_deploy", "remove_trait", FRIENDLY_UNIT, trait=["fury", "fury"])),
    _unit_with(E("on_deploy", "add_trait", FRIENDLY_UNIT)),                              # trait missing
    _unit_with(E("on_deploy", "summon", "controller", card="f0")),                       # not a token
    _unit_with(E("on_deploy", "summon", "controller", card="optok")),                    # operation token
    _unit_with(E("on_deploy", "add_card", "controller", card="nope")),                   # unknown card
    _unit_with(E("on_deploy", "summon", "controller")),                                  # card missing
    _unit_with(E("on_deploy", "summon", "controller", card="tok", amount=0)),
    _unit_with(*[E("on_deploy", "draw", "controller", amount=1)] * 4),                   # > 3 effects
    _unit_with(E("on_deploy", "draw", "controller", amount=1, condition=[])),
    _unit_with(E("on_deploy", "draw", "controller", amount=1, condition={"type": "weather"})),
    _unit_with(E("on_deploy", "draw", "controller", amount=1, condition={"type": "control", "side": "enemy",
                                                                         "min": 1, "max": 0})),  # min > max
    _unit_with(E("on_deploy", "draw", "controller", amount=1, condition={"type": "control"})),
    _unit_with(E("on_deploy", "draw", "controller", amount=1, condition={"type": "base_hp", "side": "any",
                                                                         "max": 3})),
    _unit_with(E("on_deploy", "draw", "controller", amount=1, condition={"type": "turn", "whose": "mine"})),
    _unit_with(E("on_deploy", "draw", "controller", amount=1, condition={"type": "frontline", "owner": "me"})),
    _unit_with(E("on_deploy", "draw", "controller", amount=1, condition={"type": "source_zone", "zone": "board"})),
    O("op", 1, E("on_play", "draw", "controller", amount=1, condition={"type": "source_zone", "zone": "backline"})),
    _unit_with(E("on_deploy", "damage", T("all", "enemy", "unit", filter={"colour": "red"}), amount=1)),
    _unit_with(E("on_deploy", "damage", T("all", "enemy", "unit", filter={"nature": "flying"}), amount=1)),
    _unit_with(E("on_deploy", "damage", T("all", "enemy", "unit", filter={"damaged": 1}), amount=1)),
    _unit_with(E("on_deploy", "damage", T("all", "enemy", "unit", filter={"min_hp": 3, "max_hp": 2}), amount=1)),
    _unit_with(E("on_deploy", "damage", T("all", "enemy", "unit", filter={"trait": "fury",
                                                                           "not_trait": "fury"}), amount=1)),
    {**_unit_with(E("on_deploy", "draw", "controller", amount=1)), "token": "yes"},
    _unit_with("draw a card"),                                                           # string effect
])
def test_bad_effect_cards_are_rejected(card):
    with pytest.raises(ValueError):
        _pool(TOKEN, OPTOKEN, card)


def test_good_edge_cards_load():
    cfg = _pool(TOKEN, OPTOKEN,
                _unit_with(E("on_death", "buff", "self", scope="friendly", atk=1)),  # a watcher may buff itself
                U("y", "troop", 2, 1, 2, E("on_deploy", "draw", "controller", amount=1,
                                           condition={"type": "hand_size", "side": "friendly"})),
                U("z", "ranged", 2, 1, 2, E("on_attack", "damage", "event", amount={"stat": "hp", "of": "event"}),
                  E("on_damaged", "buff", "event", scope="any", atk=1, hp=1)),
                O("op", 0, E("on_play", "damage", "enemy_base", amount={"stat": "cost", "of": "self", "times": 2})),
                U("w", "troop", 2, 1, 2, E("on_deploy", "draw", "controller", amount=1,
                                           condition={"type": "control", "side": "enemy", "max": 0})))
    assert cfg.cards.by_id("op").effects[0].amount == AmountDef(kind="stat", stat="cost", of="self", times=2)
    assert cfg.cards.by_id("y").effects[0].condition[0] == ConditionDef(type="hand_size", side="friendly")
    # SPEC 1.2: control's min defaults to 0 when only max is given (1 when neither is)
    assert cfg.cards.by_id("w").effects[0].condition[0] == ConditionDef(type="control", side="enemy", min=0, max=0)


def test_decks_reject_tokens():
    cards = [TOKEN] + [U(f"f{i}", "troop", 1, 1, 1) for i in range(14)]
    with pytest.raises(ValueError, match="token"):
        build_ruleset(cards, [{"cards": {f"f{i}": 3 for i in range(13)} | {"tok": 1}}])


def test_build_ruleset_matches_load_ruleset():
    import json
    with open(f"{FIXTURES}/stage2_cards.json", encoding="utf-8") as f:
        cards = json.load(f)["cards"]
    with open(f"{FIXTURES}/stage2_decks.json", encoding="utf-8") as f:
        decks = json.load(f)["decks"]
    assert build_ruleset(cards, decks, mulligan=False) == VANILLA_CONFIG
    with pytest.raises(ValueError):
        build_ruleset(cards, [])
    with pytest.raises(ValueError):
        build_ruleset([], decks)


# ---------------------------------------------------------------- loader: phase-1b extensions (SPEC §1.2b)
def test_1b_schema_parses_into_dataclasses():
    pool = EFFECT_CONFIG.cards
    lit = lambda v: AmountDef(kind="literal", value=v)  # noqa: E731
    # tags and the new traits
    bw, st, gh = pool.by_id("bushwhacker"), pool.by_id("stormer"), pool.by_id("ghost")
    assert bw.tags == ("infantry",) and pool.by_id("tax").tags == ("artillery",) and pool.by_id("ghost").tags == (
        "spirit",)
    assert pool.by_id("militia").tags == ()
    assert (bw.ambush, bw.shock, bw.immune, st.shock, st.blitz, gh.immune) == (True, False, False, True, True, True)
    assert bw.trait_mask == AMBUSH_BIT and gh.trait_mask == IMMUNE_BIT and st.traits == {"blitz": True, "shock": True}
    # on_attacked (self scope: the event unit is the attacker)
    assert pool.by_id("sentry").effects[0] == EffectDef(index=0, trigger="on_attacked", scope="self", action="damage",
                                                        target=TargetDef(select="event"), amount=lit(1))
    sig = pool.by_id("signal").effects[0]
    assert (sig.scope, sig.event_filter) == ("friendly", FilterDef(natures=(0, 1)))
    # static
    ban = pool.by_id("banner").effects[0]
    assert ban == EffectDef(index=0, trigger="static", scope="self", action="buff",
                            target=TargetDef(select="all", side="friendly", filter=FilterDef(other=True)),
                            atk=lit(1), hp=lit(1))
    assert pool.by_id("smokepot").effects[0].target == TargetDef(select="adjacent", side="friendly", of="self",
                                                                 position="both")
    depot = pool.by_id("depot").effects[0]
    assert (depot.move_cost, depot.atk, depot.hp) == (-1, lit(0), lit(0))
    assert depot.else_ == EffectDef(index=0, trigger="static", scope="self", action="buff", target=TargetDef("self"),
                                    atk=lit(1), hp=lit(0), is_else=True, own_target=True)
    warden = pool.by_id("warden").effects[0].condition[0]
    assert warden == ConditionDef(type="compare", left=AmountDef(kind="stat", stat="hp", of="self"), op="<",
                                  right=AmountDef(kind="stat", stat="max_hp", of="self"))
    gloom = pool.by_id("gloom").effects[0]
    assert gloom.target_condition == FilterDef(not_tags=("tank",)) and gloom.atk == lit(-1)
    assert (gloom.else_.action, gloom.else_.trait, gloom.else_.target, gloom.else_.own_target) == (
        "add_trait", ("ambush",), gloom.target, False)
    # prev / adjacent targets, prev conditions, repeat
    gren = pool.by_id("grenade").effects
    assert gren[1].target == TargetDef(select="adjacent", side="enemy", of="prev", position="both")
    assert gren[2].condition == (ConditionDef(type="prev", killed=True),)
    dt = pool.by_id("double_tap").effects[1]
    assert dt.target == TargetDef(select="prev", kind="unit_or_base")  # prev passes on what damage can take
    mk = pool.by_id("marksman").effects
    assert mk[0].repeat == 3 and mk[1].target == TargetDef(select="prev", filter=FilterDef(damaged=True))
    assert pool.by_id("flanker").effects[0].target == TargetDef(select="adjacent", side="any", of="event",
                                                                position="both")
    # target_condition + else reusing the effect's target
    inter = pool.by_id("interrogator").effects
    assert inter[0].target_condition == FilterDef(tags=("tank",))
    assert inter[0].else_ == EffectDef(index=0, trigger="on_deploy", scope="self", action="pin",
                                       target=ENEMY_TARGET, is_else=True, own_target=False)
    assert inter[1].amount == AmountDef(kind="stat", stat="move_cost", of="prev")
    assert inter[1].condition == (ConditionDef(type="prev", filter=FilterDef(pinned=True)),)
    # else with its own target after a failed condition
    prag = pool.by_id("pragmatist").effects[0]
    assert prag.else_ == EffectDef(index=0, trigger="on_deploy", scope="self", action="draw",
                                   target=TargetDef(select="all", side="friendly", kind="player"), amount=lit(1),
                                   is_else=True)
    # history (min defaults to 1), event damage, uncapped heals, event filters, negative coins, base_hp counts
    assert pool.by_id("veteran").effects[0].condition == (ConditionDef(type="history", event="unit_died", side="any",
                                                                       window="turn", min=1),)
    assert pool.by_id("strategist").effects[0].condition[0].min == 2
    rp = pool.by_id("reaper").effects
    assert rp[0].amount == AmountDef(kind="event") and rp[0].uncapped
    assert rp[1].event_filter == FilterDef(tags=("infantry",)) and rp[1].card == IDX["scout_car"]
    dep = pool.by_id("deployer").effects
    assert dep[0].event_filter == FilterDef(max_cost=2, token=False)
    assert dep[1].event_filter == FilterDef(tags=("artillery",))
    tax = pool.by_id("tax").effects
    assert tax[0].amount == lit(-2)
    assert tax[1].amount == AmountDef(kind="count", count="base_hp", side="enemy", times=0, plus=2)
    conv = pool.by_id("convoy").effects
    assert (conv[0].move_cost, conv[0].duration) == (-1, "turn")
    assert conv[1].target.filter == FilterDef(pinned=False) and conv[1].trait == ("immune",)
    assert pool.by_id("bombard").effects[0].target.filter == FilterDef(tags=("infantry", "tank"))  # sorted
    assert pool.by_id("entrench").effects[1].trait == ("shock", "immune")


ENEMY_TARGET = TargetDef(select="chosen", side="enemy", kind="unit")
MOVER = E("on_deploy", "draw", "controller", amount=1)  # a first clause for prev to read


def _static_unit(*effects, **kw):
    return U("x", "troop", 2, 1, 2, *effects, **kw)


@pytest.mark.parametrize("card", [
    # tags and traits
    {**GOOD, "tags": "tank"}, {**GOOD, "tags": [""]}, {**GOOD, "tags": ["tank", "tank"]}, {**GOOD, "tags": [3]},
    {**O("op", 1, E("on_play", "draw", "controller", amount=1)), "tags": "artillery"},
    U("x", "troop", 2, 1, 2, traits={"ambush": 1}), U("x", "troop", 2, 1, 2, traits={"stealth": True}),
    _unit_with(E("on_deploy", "add_trait", FRIENDLY_UNIT, trait="stealth")),
    # on_attacked
    _unit_with(E("on_attacked", "damage", ENEMY_UNIT, amount=1)),                        # chosen off-turn
    _unit_with(E("on_attacked", "damage", "event", scope="friendly", amount=1, event_filter={"colour": "red"})),
    # static restrictions
    _static_unit(E("static", "buff", "self", scope="friendly", atk=1)),
    O("op", 1, E("static", "buff", ALL_FRIENDLY, atk=1)),
    _static_unit(E("static", "damage", ALL_ENEMY, amount=1)),
    _static_unit(E("static", "buff", T("random", "friendly", "unit"), atk=1)),
    _static_unit(E("static", "buff", T("chosen", "friendly", "unit"), atk=1)),
    _static_unit(E("static", "buff", ALL_FRIENDLY, atk=1, duration="turn")),
    _static_unit(E("static", "buff", ALL_FRIENDLY, atk={"count": "units", "side": "friendly"})),
    _static_unit(E("static", "buff", ALL_FRIENDLY, atk=1, repeat=2)),
    _static_unit(E("static", "buff", ALL_FRIENDLY, atk=1, event_filter={"token": True})),
    _static_unit(E("static", "add_trait", ALL_FRIENDLY, trait="armor")),
    _static_unit(E("static", "add_trait", ALL_FRIENDLY, trait="armor", amount=2)),
    _static_unit(E("static", "buff", T("adjacent", "friendly", of="event"), atk=1)),
    _static_unit(E("static", "buff", "self", atk=1), E("static", "buff", "self", atk=1,
                                                       condition={"type": "prev", "killed": True})),
    _static_unit(E("static", "buff", "self", atk=1, **{"else": {"action": "draw", "target": "controller",
                                                               "amount": 1}})),
    _static_unit(E("static", "buff", "self", atk=1, **{"else": {"action": "buff", "atk": {"stat": "atk",
                                                                                         "of": "self"}}})),
    # event_filter
    _unit_with(E("on_deploy", "draw", "controller", amount=1, event_filter={"token": True})),  # scope self
    _unit_with(E("end_of_turn", "draw", "controller", amount=1, event_filter={"token": True})),
    _unit_with(E("on_play", "draw", "controller", scope="any", amount=1, event_filter={"nature": "troop"})),
    _unit_with(E("on_play", "draw", "controller", scope="any", amount=1, event_filter={"pinned": True})),
    # prev
    _unit_with(E("on_deploy", "damage", T("prev"), amount=1)),                            # no earlier clause
    _unit_with(E("on_move", "draw", "controller", amount=1), E("on_deploy", "damage", T("prev"), amount=1)),
    _unit_with(MOVER, E("on_deploy", "damage", T("prev", "enemy"), amount=1)),            # side on prev
    _unit_with(MOVER, E("on_deploy", "damage", T("prev", zone="backline"), amount=1)),
    _unit_with(E("on_deploy", "draw", "controller", amount=1, condition={"type": "prev"})),
    _unit_with(MOVER, E("on_deploy", "draw", "controller", amount=1, condition={"type": "prev", "dead": True})),
    _unit_with(MOVER, E("on_deploy", "draw", "controller", amount=1, condition={"type": "prev", "killed": 1})),
    _unit_with(E("on_deploy", "draw", "controller", amount={"stat": "atk", "of": "prev"})),
    _unit_with(MOVER, E("on_deploy", "draw", T("prev", kind="unit"), amount=1)),          # draw needs players
    # adjacent
    _unit_with(E("on_deploy", "damage", T("adjacent", "enemy", zone="backline"), amount=1)),
    _unit_with(E("on_deploy", "damage", T("adjacent", "enemy", "base"), amount=1)),
    _unit_with(E("on_deploy", "damage", T("adjacent", of="event"), amount=1)),            # no event unit
    O("op", 1, E("on_play", "damage", T("adjacent", "enemy"), amount=1)),                 # no source unit
    _unit_with(E("on_death", "damage", T("adjacent", "friendly"), amount=1)),             # dead source
    _unit_with(E("on_deploy", "damage", T("adjacent", position="up"), amount=1)),
    _unit_with(E("on_deploy", "damage", T("adjacent", of="target"), amount=1)),
    _unit_with(E("on_deploy", "damage", T("adjacent", of="prev"), amount=1)),             # no earlier clause
    _unit_with(E("on_deploy", "damage", T("all", "enemy", "unit", of="self"), amount=1)),
    _unit_with(E("on_deploy", "damage", T("all", "enemy", "unit", position="left"), amount=1)),
    _unit_with(E("on_deploy", "damage", T("adjacent", count=2), amount=1)),
    # filter keys
    _unit_with(E("on_deploy", "damage", T("all", "enemy", "unit", filter={"tag": []}), amount=1)),
    _unit_with(E("on_deploy", "damage", T("all", "enemy", "unit", filter={"tag": ""}), amount=1)),
    _unit_with(E("on_deploy", "damage", T("all", "enemy", "unit", filter={"tag": ["a", "a"]}), amount=1)),
    _unit_with(E("on_deploy", "damage", T("all", "enemy", "unit", filter={"tag": "a", "not_tag": ["a"]}), amount=1)),
    _unit_with(E("on_deploy", "damage", T("all", "enemy", "unit", filter={"pinned": "yes"}), amount=1)),
    # amounts
    _unit_with(E("on_deploy", "damage", ENEMY_UNIT, amount={"stat": "atk", "of": "victim"})),
    _unit_with(E("on_deploy", "damage", ENEMY_UNIT, amount={"count": "base_hp", "side": "enemy", "zone": "board"})),
    _unit_with(E("on_deploy", "damage", ENEMY_UNIT, amount={"count": "base_hp"})),
    _unit_with(E("on_deploy", "damage", ENEMY_UNIT, amount={"event": "damage"})),         # no event damage
    _unit_with(E("on_damaged", "damage", "event", amount={"event": "heal"})),
    _unit_with(E("on_damaged", "damage", "event", amount={"event": "damage", "of": "self"})),
    _unit_with(E("on_damaged", "damage", "event", amount={"event": "damage", "stat": "atk"})),
    # conditions
    _unit_with(E("on_deploy", "draw", "controller", amount=1, condition={"type": "compare", "left": 1, "right": 2})),
    _unit_with(E("on_deploy", "draw", "controller", amount=1, condition={"type": "compare", "left": 1, "op": "!=",
                                                                         "right": 2})),
    _unit_with(E("on_deploy", "draw", "controller", amount=1, condition={"type": "compare", "left": "full",
                                                                         "op": ">", "right": 2})),
    _unit_with(E("on_deploy", "draw", "controller", amount=1, condition={"type": "compare", "left": 1.5,
                                                                         "op": ">", "right": 2})),
    _unit_with(E("on_deploy", "draw", "controller", amount=1, condition={
        "type": "history", "event": "unit_summoned", "side": "any", "window": "turn"})),
    _unit_with(E("on_deploy", "draw", "controller", amount=1, condition={
        "type": "history", "event": "unit_died", "side": "any", "window": "round"})),
    _unit_with(E("on_deploy", "draw", "controller", amount=1, condition={
        "type": "history", "event": "unit_died", "side": "any"})),
    _unit_with(E("on_deploy", "draw", "controller", amount=1, condition={
        "type": "history", "event": "unit_died", "side": "self", "window": "game"})),
    _unit_with(E("on_deploy", "draw", "controller", amount=1, condition={
        "type": "history", "event": "unit_died", "side": "any", "window": "game", "min": 3, "max": 2})),
    _unit_with(E("on_deploy", "draw", "controller", amount=1, condition={
        "type": "history", "event": "unit_died", "side": "any", "window": "game", "min": -1})),
    # target_condition / else / repeat
    _unit_with(E("on_deploy", "draw", "controller", amount=1, target_condition={"token": True})),
    _unit_with(E("on_deploy", "damage", ALL_ENEMY, amount=1, target_condition={"colour": "red"})),
    _unit_with(E("on_deploy", "damage", ALL_ENEMY, amount=1, **{"else": "pin"})),
    _unit_with(E("on_deploy", "damage", ALL_ENEMY, amount=1, **{"else": {"action": "pin", "else": {"action": "pin"}}})),
    _unit_with(E("on_deploy", "damage", ALL_ENEMY, amount=1, **{"else": {"action": "pin", "condition": {
        "type": "turn", "whose": "own"}}})),
    _unit_with(E("on_deploy", "damage", ALL_ENEMY, amount=1, **{"else": {"target": ALL_ENEMY}})),
    _unit_with(E("on_deploy", "damage", ALL_ENEMY, amount=1, **{"else": {"action": "draw", "amount": 1}})),
    _unit_with(E("on_deploy", "damage", ALL_ENEMY, amount=1, **{"else": {"action": "pin", "amount": 1}})),
    _unit_with(E("on_deploy", "damage", ALL_ENEMY, amount=1, target_condition={"token": True},
                 **{"else": {"action": "pin", "target": ENEMY_UNIT}})),
    _unit_with(E("on_death", "draw", "controller", scope="friendly", amount=1, **{"else": {
        "action": "pin", "target": ENEMY_UNIT}})),                                         # chosen off-turn
    _unit_with(E("on_deploy", "damage", ALL_ENEMY, amount=1, **{"else": {"action": "summon", "target": "controller",
                                                                       "card": "f0"}})),  # not a token
    _unit_with(E("on_deploy", "damage", ALL_ENEMY, amount=1, repeat=0)),
    _unit_with(E("on_deploy", "damage", ALL_ENEMY, amount=1, repeat=True)),
    _unit_with(E("on_deploy", "damage", ENEMY_UNIT, amount=1, repeat=2)),                 # repeat + chosen
    # action parameters
    _unit_with(E("on_deploy", "damage", ENEMY_UNIT, amount=1, uncapped=True)),
    _unit_with(E("on_deploy", "heal", "friendly_base", amount=1, uncapped="yes")),
    _unit_with(E("on_deploy", "heal", "friendly_base", amount="full", uncapped=True)),
    _unit_with(E("on_deploy", "buff", FRIENDLY_UNIT, move_cost=1.0)),
    _unit_with(E("on_deploy", "buff", FRIENDLY_UNIT, move_cost={"stat": "cost", "of": "self"})),
    _unit_with(E("on_deploy", "draw", "controller", amount=-1)),                         # only coins may be < 0
    _unit_with(E("on_deploy", "gain_coins", "controller", amount=1.5)),
])
def test_bad_1b_cards_are_rejected(card):
    with pytest.raises(ValueError):
        _pool(TOKEN, OPTOKEN, card)


def test_good_1b_edge_cards_load():
    cfg = _pool(TOKEN, OPTOKEN,
                # prev with an explicit kind, a watcher batch, an else body that summons a token
                U("a", "troop", 2, 1, 2, MOVER, E("on_deploy", "gain_coins", T("prev", kind="player"), amount=1),
                  E("on_deploy", "damage", ALL_ENEMY, amount=1, target_condition={"token": True},
                    **{"else": {"action": "summon", "target": "controller", "card": "tok"}})),
                U("b", "troop", 2, 1, 2, E("on_death", "draw", "controller", scope="friendly", amount=1),
                  E("on_death", "damage", T("prev"), amount=1, scope="any")),
                U("c", "troop", 2, 1, 2, E("static", "add_trait", ALL_ENEMY, trait="ambush",
                                           target_condition={"pinned": False}),
                  E("static", "buff", T("adjacent", position="left"), hp=1),
                  E("on_attack", "damage", T("adjacent", of="event", position="right"), amount=1,
                    condition={"type": "compare", "left": -3, "op": "<", "right": 0})),
                O("op", 1, E("on_play", "damage", ALL_ENEMY, amount=1), E("on_play", "heal", T("prev"), amount=1),
                  tags=[]))
    a = cfg.cards.by_id("a").effects
    assert a[1].target == TargetDef(select="prev", kind="player")
    assert a[2].else_.card == cfg.cards.by_id("tok").index and a[2].else_.target == TargetDef(
        select="all", side="friendly", kind="player")
    assert cfg.cards.by_id("b").effects[1].target == TargetDef(select="prev", kind="unit_or_base")
    c = cfg.cards.by_id("c").effects
    assert c[1].target == TargetDef(select="adjacent", side="any", of="self", position="left")
    assert c[2].condition[0].left == AmountDef(kind="literal", value=-3)
    assert cfg.cards.by_id("op").effects[1].target == TargetDef(select="prev", kind="unit_or_base")
    assert cfg.cards.by_id("op").tags == ()


# ---------------------------------------------------------------- random decks and deals (SPEC §1.3)
def assert_legal_deck(deck, cfg) -> None:
    assert type(deck) is tuple and len(deck) == cfg.deck_size and list(deck) == sorted(deck)
    counts = Counter(deck)
    assert max(counts.values()) <= cfg.max_copies
    assert not any(cfg.cards[c].token for c in counts)
    assert sum(k for c, k in counts.items() if cfg.cards[c].is_operation) <= MAX_OPERATIONS


@pytest.mark.parametrize("cfg", [EFFECT_CONFIG, VANILLA_CONFIG, CONFIG], ids=["effects", "vanilla", "shipped"])
def test_generated_decks_are_legal_and_deterministic(cfg):
    decks = set()
    for seed in range(200):
        d = generate_deck(random.Random(seed), cfg)
        assert_legal_deck(d, cfg)
        assert d == generate_deck(random.Random(seed), cfg)
        decks.add(d)
    assert len(decks) == 200


def test_generated_decks_follow_the_cost_curve():
    # 6 cards per cost 1..8 (3 copies each): every bucket can meet its target exactly.
    cards = [U(f"c{cost}_{k}", "troop", cost, 1, 1) for cost in range(1, 9) for k in range(6)]
    cfg = build_ruleset(cards, [{"cards": {f"c{c}_0": 3 for c in range(1, 9)} | {"c1_1": 3, "c2_1": 3,
                                                                                  "c3_1": 3, "c4_1": 3,
                                                                                  "c5_1": 3, "c6_1": 1}}])
    for seed in range(50):
        d = generate_deck(random.Random(seed), cfg)
        per_bucket = Counter(cost_bucket(cfg.cards[c].cost) for c in d)
        assert [per_bucket[b] for b in range(8)] == [4, 7, 7, 6, 5, 4, 3, 4]


def test_empty_buckets_hand_their_quota_to_the_nearest_bucket():
    # no cost-1 cards (quota 4 -> cost 2); costs 7 and 8+ missing (3 + 4 -> cost 6, the nearest with cards)
    cards = [U(f"c{cost}_{k}", "troop", cost, 1, 1) for cost in range(2, 7) for k in range(8)]
    cfg = build_ruleset(cards, [{"cards": {f"c{c}_{k}": 2 for c in range(2, 7) for k in range(4)}}])
    for seed in range(30):
        d = generate_deck(random.Random(seed), cfg)
        assert_legal_deck(d, cfg)
        per_cost = Counter(cfg.cards[c].cost for c in d)
        assert per_cost == {2: 11, 3: 7, 4: 6, 5: 5, 6: 11}


def test_operations_are_capped_and_required_cards_are_placed_first():
    ops = [O(f"op{i}", 1 + i % 8, E("on_play", "draw", "controller", amount=1)) for i in range(30)]
    units = [U(f"u{i}", "troop", 1 + i % 8, 1, 1) for i in range(16)]
    cfg = build_ruleset(ops + units, [{"cards": {f"u{i}": 2 for i in range(16)} | {"op0": 3, "op1": 3,
                                                                                    "op2": 2}}])
    for seed in range(30):
        d = generate_deck(random.Random(seed), cfg)
        assert sum(cfg.cards[c].is_operation for c in d) == MAX_OPERATIONS
    req = [0] * len(cfg.cards)
    req[0], req[35] = 3, 2
    for seed in range(30):
        d = Counter(generate_deck(random.Random(seed), cfg, required=req))
        assert d[0] == 3 and d[35] >= 2
    assert generate_deck(random.Random(1), cfg, required={0: 3, 35: 2}) == generate_deck(random.Random(1), cfg,
                                                                                          required=req)
    for bad in ({0: 4}, {0: -1}, {999: 1}, {0: 1.0}, [3] * len(cfg.cards)):
        with pytest.raises(ValueError):
            generate_deck(random.Random(0), cfg, required=bad)
    with pytest.raises(ValueError):
        generate_deck(random.Random(0), EFFECT_CONFIG, required={IDX["conscript"]: 1})  # tokens never required


def test_deck_rng_and_sample_deal():
    assert deck_rng(5, 1).random() == random.Random("deck:5:1").random()
    for seed in range(40):
        d0, d1 = sample_deal(seed, EFFECT_CONFIG, 0.7)
        assert (d0, d1) == sample_deal(seed, EFFECT_CONFIG, 0.7)
        for seat, d in enumerate((d0, d1)):
            if isinstance(d, tuple):
                assert d == generate_deck(deck_rng(seed, seat), EFFECT_CONFIG)
            else:
                assert 0 <= d < EFFECT_CONFIG.n_decks
        assert all(type(d) is int for d in sample_deal(seed, EFFECT_CONFIG, 0.0))
        assert all(type(d) is tuple for d in sample_deal(seed, EFFECT_CONFIG, 1.0))
    kinds = Counter(type(d).__name__ for s in range(1000) for d in sample_deal(s, EFFECT_CONFIG, 0.7))
    assert 1250 <= kinds["tuple"] <= 1550
    with pytest.raises(ValueError):
        sample_deal(0, EFFECT_CONFIG, 1.5)


# ---------------------------------------------------------------- reset with deck specs (SPEC §2.1)
def test_reset_accepts_fixed_and_tuple_decks():
    deck = generate_deck(random.Random(3), EFFECT_CONFIG)
    g = Game(EFFECT_CONFIG_NM)
    g.reset(4, (list(reversed(deck)), 1))
    assert g.deck_ids == (-1, 1) and g.decklists == (deck, EFFECT_CONFIG.decks[1])
    for p in (0, 1):
        assert sorted(g.hands[p] + g.deck_cards[p]) == list(g.decklists[p])
    h = Game(EFFECT_CONFIG_NM)
    h.reset(4, (deck, np.int64(1)))
    assert state_key(g) == state_key(h)
    check_invariants(g)
    assert g.observe(0).my_deck == -1 and sum(g.observe(0).my_decklist) == 40


@pytest.mark.parametrize("decks", [
    (tuple(range(40)), 0),                                    # 1 copy each but indices include tokens/operations
    ((IDX["militia"],) * 40, 0),                              # 40 copies
    (generate_deck(random.Random(0), EFFECT_CONFIG)[:39], 0),  # 39 cards
    (generate_deck(random.Random(0), EFFECT_CONFIG)[:39] + (IDX["conscript"],), 0),  # a token
    (generate_deck(random.Random(0), EFFECT_CONFIG)[:39] + (999,), 0),               # out of range
    (generate_deck(random.Random(0), EFFECT_CONFIG)[:39] + (1.0,), 0),
    ("x" * 40, 0),
])
def test_reset_rejects_illegal_tuple_decks(decks):
    with pytest.raises((TypeError, ValueError)):
        Game(EFFECT_CONFIG).reset(0, decks)


# ---------------------------------------------------------------- tables and the vanilla fast path
def test_trigger_tables_and_effect_free_fast_path():
    assert not Game(VANILLA_CONFIG)._has_effects and Game(EFFECT_CONFIG)._has_effects
    g = Game(EFFECT_CONFIG)
    assert g._self_eff[IDX["grenadier"]][T_DEPLOY] == EFFECT_CONFIG.cards.by_id("grenadier").effects
    assert g._watch_eff[IDX["grenadier"]][T_DEPLOY] == ()
    mourner = EFFECT_CONFIG.cards.by_id("mourner").effects[0]
    assert g._watch_eff[IDX["mourner"]][T_DEATH] == ((mourner, 1),) and g._self_eff[IDX["mourner"]][T_DEATH] == ()
    assert g._t is Game(EFFECT_CONFIG)._t  # computed once per pool
    assert g.clone()._t is g._t


# ---------------------------------------------------------------- choices, clone and the queue
def test_two_choices_in_one_chain_and_clone_maps_unit_references(monkeypatch):
    g = eblank(coins=10, round_=10)
    a = add_unit(g, 0, "back", "militia")
    add_unit(g, 1, "back", "guard")
    set_hand(g, 0, ["warlord"])
    g.step(play(0))
    assert g.phase == CHOICE and g.pending is not None and len(g.queue) == 1
    w = g.backline[0][-1]
    assert g.pending[0][4] is w and g.queue[0][4] is w
    assert g.legal_actions() == [choose(t) for t in (11, 12)]  # own backline slots 0 and 1
    pv = g.observe(1).pending
    assert (pv.card, pv.effect, pv.action, pv.select, pv.side, pv.kind) == (IDX["warlord"], 0, "add_trait",
                                                                            "chosen", "friendly", "unit")

    def forbidden(*args, **kwargs):
        raise AssertionError("clone() deep-copied a reachable state")

    monkeypatch.setattr(copy, "deepcopy", forbidden)
    c = g.clone()
    monkeypatch.undo()
    assert state_key(c) == state_key(g)
    cw = c.backline[0][-1]
    assert c.pending[0][4] is cw and c.queue[0][4] is cw and cw is not w
    c.step(choose(11))
    assert c.phase == CHOICE and c.backline[0][0].fury and not c.queue
    c.step(choose(12))
    assert c.phase == MAIN and c.backline[0][1].blitz and c.pending is None
    assert g.phase == CHOICE and not a.fury and not w.blitz  # the original is untouched
    with pytest.raises(IllegalActionError):
        g.step(0)  # END_TURN is not legal during a choice


def test_operation_needs_an_option_and_pending_amount_is_resolved():
    g = eblank(coins=10, round_=10)
    set_hand(g, 0, ["volley", "strike"])
    i_volley, i_strike = g.hands[0].index(IDX["volley"]), g.hands[0].index(IDX["strike"])
    assert g.legal_actions() == [0, play(i_strike)]  # volley has no enemy unit; strike can hit the base
    add_unit(g, 1, "front", "echo", hp=4)
    assert g.legal_actions() == [0, play(0), play(1)]
    g.step(play(i_strike))
    pv = g.observe(0).pending
    assert (pv.action, pv.amount, pv.kind) == ("damage", 3, "unit_or_base")
    assert g.legal_actions() == [choose(5), choose(10)]  # frontline 0, enemy base
    g.step(choose(10))
    assert g.base_hp[1] == 17 and g.phase == MAIN and g.discard[0][IDX["strike"]] == 1


def test_loop_guard_discards_the_rest_after_256_instances(monkeypatch):
    g = eblank(coins=10, round_=10)
    add_unit(g, 0, "back", "echo")
    add_unit(g, 1, "back", "echo")
    set_hand(g, 0, ["strike"])
    g.step(play(0))
    resolved = []
    original = Game._resolve
    monkeypatch.setattr(Game, "_resolve", lambda self, inst: (resolved.append(inst[0].action), original(self, inst)))
    g.step(choose(0))
    assert len(resolved) == 256 == g.config.max_effect_events
    assert g.guard_trips == 1 and g.queue == [] and g.phase == MAIN and not g.done
    check_invariants(g, reachable=False)


def test_turn_buffs_record_the_applied_change():
    g = eblank(coins=10, round_=10)
    weak = add_unit(g, 1, "back", "militia", atk=0)
    strong = add_unit(g, 1, "back", "militia", atk=3)
    set_hand(g, 0, ["hexer"])
    g.step(play(0))
    assert (weak.atk, weak.temp_atk, strong.atk, strong.temp_atk) == (0, 0, 2, -1)
    g.step(0)  # END_TURN: "turn" durations expire
    assert (weak.atk, strong.atk, weak.temp_atk, strong.temp_atk) == (0, 3, 0, 0)


def test_turn_removal_of_armor_and_traits_restores_what_the_unit_would_have():
    g = eblank(coins=10, round_=10)
    s = add_unit(g, 1, "back", "sentinel")
    assert (s.armor, s.defense) == (1, True)
    set_hand(g, 0, ["jammer", "sharpen"])
    g.step(play(0))  # jammer: defense removed for the turn
    assert not s.defense and s.temp_removed
    g.step(play(0))  # sharpen: armor removed permanently (chosen), own units gain defense
    assert g.legal_actions() == [choose(0)]
    g.step(choose(0))
    assert s.armor == 0 and g.backline[0][0].defense and g.phase == MAIN
    g.step(0)
    assert s.defense and s.armor == 0 and s.temp_removed == 0


def test_mulligan_with_marks_draws_then_shuffles():
    g = Game(EFFECT_CONFIG)
    g.reset(11)
    f = g.first_player
    assert g.phase == MULLIGAN and g.current == f and g.turn == 0
    hand, deck = list(g.hands[f]), list(g.deck_cards[f])
    g.step(g.action_space.mulligan(0))
    g.step(g.action_space.mulligan(2))
    assert g.legal_actions() == [g.action_space.mulligan(i) for i in (1, 3)] + [g.action_space.CONFIRM]
    assert g.observe(f).mulligan_marks == (True, False, True, False) and g.observe(1 - f).mulligan_marks == ()
    g.step(g.action_space.CONFIRM)
    drawn = deck[-2:]
    assert g.hands[f] == sorted([hand[1], hand[3]] + drawn)
    assert sorted(g.deck_cards[f]) == sorted(deck[:-2] + [hand[0], hand[2]])
    assert g.current == 1 - f and g.phase == MULLIGAN
    state = g.rng.getstate()
    g.step(g.action_space.CONFIRM)  # k = 0: nothing drawn, RNG untouched
    assert g.phase == MAIN and g.current == f and g.turn == 1 and g.round == 1
    assert len(g.hands[f]) == 5 and g.coins[f] == 1
    assert g.rng.getstate() == state


# ---------------------------------------------------------------- phase-1b internals (SPEC §1.2b, §2.8b, §2.12)
def _forbid_deepcopy(*args, **kwargs):
    raise AssertionError("clone() deep-copied a reachable state")


def slot(g: Game, p: int, cid: str) -> int:
    return g.hands[p].index(IDX[cid])


def test_1b_tables_and_vanilla_flags():
    v = Game(VANILLA_CONFIG)
    assert not (v._any_static or v._uses_ambush or v._uses_history or v._attacked_any)
    assert not any(v._static_eff) and not any(map(any, v._self_batch)) and v._tags == (frozenset(),) * len(v._tags)
    g = Game(EFFECT_CONFIG)
    assert g._any_static and g._uses_ambush and g._uses_history and g._attacked_any
    assert g._static_eff[IDX["banner"]] == EFFECT_CONFIG.cards.by_id("banner").effects
    assert g._static_eff[IDX["militia"]] == () and g._self_eff[IDX["banner"]][T_STATIC] == ()  # never enqueued
    assert g._self_batch[IDX["grenade"]][T_PLAY] and not g._self_batch[IDX["volley"]][T_PLAY]
    assert g._self_batch[IDX["marksman"]][T_DEPLOY] and not g._self_batch[IDX["warlord"]][T_DEPLOY]
    assert g._tags[IDX["bombard"]] == frozenset({"artillery"})
    w = _pool(U("w", "troop", 2, 1, 2, E("on_death", "draw", "controller", scope="friendly", amount=1),
                E("on_death", "damage", T("prev"), scope="any", amount=1)),
              U("amb", "troop", 1, 1, 1, E("on_deploy", "add_trait", "self", trait="ambush")))
    gw = Game(w)
    k = w.cards.by_id("w").index
    assert gw._watch_batch[k][T_DEATH] and not gw._self_batch[k][T_DEATH]
    assert gw._uses_ambush and not gw._any_static and not gw._uses_history and not gw._attacked_any


def test_clause_batch_is_shared_by_its_instances_and_cloned_once(monkeypatch):
    g = eblank(coins=10, round_=10)
    e0 = add_unit(g, 1, "back", "militia", hp=1)
    brute = add_unit(g, 1, "back", "brute")          # 6/6
    e2 = add_unit(g, 1, "back", "militia")           # 1/2
    set_hand(g, 0, ["grenade"])
    g.deck_cards[0] = [IDX["militia"]]
    g.step(play(0))
    assert g.phase == CHOICE and len(g.queue) == 2 and g.legal_actions() == [choose(t) for t in (0, 1, 2)]
    batch = g.pending[0][7]
    assert batch == [] and g.queue[0][7] is batch and g.queue[1][7] is batch
    monkeypatch.setattr(copy, "deepcopy", _forbid_deepcopy)
    c = g.clone()
    monkeypatch.undo()
    cb = c.pending[0][7]
    assert cb == [] and cb is not batch and c.queue[0][7] is cb and c.queue[1][7] is cb
    assert state_key(c) == state_key(g)
    seen = []
    original = Game._resolve
    monkeypatch.setattr(Game, "_resolve", lambda self, inst: (seen.append((inst[0].index, list(inst[7]))),
                                                              original(self, inst)))
    g.step(choose(1))  # 2 damage to the brute; 1 to its neighbours (e0 dies); e0 was killed -> draw
    assert seen[0] == (1, [((brute,), False)])
    assert seen[1] == (2, [((brute,), False), ((e0, e2), True)])
    assert (brute.hp, e0.hp, e2.hp) == (4, 0, 1) and g.backline[1] == [brute, e2] and g.hands[0] == [IDX["militia"]]
    c.step(choose(1))
    assert state_key(c) == state_key(g)


def test_batch_records_and_tags():
    from cardgame.engine import _record, _tagged
    u, v = Unit(0, 0, atk=1, hp=1, uid=1), Unit(0, 1, atk=1, hp=1, uid=2)
    assert _tagged([0, 1], "player") == [-1, -2] and _tagged([u, 1], "unit_or_base") == [u, 1]
    assert _record([u, v, u, 1, 1, -2], [v]) == ((u, v, 1, -2), True)
    assert _record([u, 0], [Unit(0, 0, atk=1, hp=1, uid=1)]) == ((u, 0), False)  # killed means this very unit
    assert _record([], []) == ((), False)


def test_prev_and_adjacent_selection_internals():
    g = eblank()
    a = add_unit(g, 0, "back", "militia")
    b = add_unit(g, 0, "back", "militia", hp=1)
    gone = Unit(IDX["militia"], 1, atk=1, hp=0, uid=99)  # left the board
    inst = (None, 0, 0, -1, None, -1, None, [((gone, a, 1, -1), False)], 0)

    def prev(**kw):
        return g._prev_targets(TargetDef(select="prev", **kw), inst)

    assert prev() == [a] and prev(kind="unit_or_base") == [a, 1] and prev(kind="base") == [1]
    assert prev(kind="player") == [0] and prev(filter=FilterDef(damaged=True)) == []
    assert g._prev_targets(TargetDef(select="prev"), inst[:7] + (None, 0)) == []
    adj = TargetDef(select="adjacent", side="any", of="prev", position="both")
    assert g._adjacent(adj, inst) == [b]  # the first prev unit still on the board is the reference
    assert g._adjacent(TargetDef(select="adjacent", side="enemy", of="prev", position="both"), inst) == []
    assert g._adjacent(TargetDef(select="adjacent", side="friendly", of="prev", position="left"), inst) == []
    assert g._adjacent(TargetDef(select="adjacent", side="any", of="prev", filter=FilterDef(max_hp=1)), inst) == [b]
    assert g._first_prev_unit(inst) is gone  # amounts read the first prev unit, on the board or not
    assert g._amount(AmountDef(kind="stat", stat="hp", of="prev"), inst) == 0  # last-known hp 0
    assert g._amount(AmountDef(kind="count", count="base_hp", side="any"), inst) == 40


def test_static_recompute_fast_path(monkeypatch):
    import cardgame.engine as eng
    calls = Counter()
    original = eng._set_static
    monkeypatch.setattr(eng, "_set_static", lambda u, *c: (calls.update(["set"]), original(u, *c)))
    g = eblank(coins=10, round_=10)
    m = add_unit(g, 0, "back", "militia")
    set_hand(g, 0, ["militia", "banner"])
    g.step(play(slot(g, 0, "militia")))
    assert not calls and not g._static_on  # no static source on the board: skipped
    g.step(play(slot(g, 0, "banner")))
    ban = g.backline[0][-1]
    assert calls and g._static_on
    assert (m.atk, m.hp, m.max_hp, m.static_atk, m.static_hp) == (2, 3, 3, 1, 1)
    assert (ban.atk, ban.static_atk) == (1, 0)  # "other" friendly units only
    m.hp = 1  # damaged under the aura
    g.backline[0].remove(ban)
    g.invalidate()  # the source is gone: contributions are cleared once
    assert (m.atk, m.hp, m.max_hp, m.static_atk, m.static_hp) == (1, 1, 2, 0, 0) and not g._static_on
    calls.clear()
    g.step(0)
    g.step(0)
    assert not calls


LAYER_CONFIG = _pool(
    U("hexer2", "troop", 1, 0, 5, E("static", "buff", T("all", "enemy"), atk=-2), tags=["leader"]),
    U("drummer", "troop", 1, 0, 5, E("static", "buff", T("all", "friendly", filter={"other": True}), atk=2,
                                     move_cost=1), tags=["leader"]),
    U("brute3", "troop", 1, 3, 9, move_cost=1),
    O("weaken", 0, E("on_play", "buff", T("all", "enemy", "unit"), atk=-2, duration="turn")),
    O("sap", 0, E("on_play", "buff", T("all", "enemy", "unit"), atk=-5)),
    O("purge", 0, E("on_play", "destroy", T("all", "any", "unit", filter={"tag": "leader"}))),
    O("rally", 0, E("on_play", "buff", T("all", "friendly", "unit"), atk=1, move_cost=-2, duration="turn")))


def _lblank():
    return blank_game(config=LAYER_CONFIG, coins=10, round_=5, decks=(0, 0))


def _cast(g, cid):
    set_hand(g, g.current, [cid])
    g.step(play(0))


def test_static_layering_with_turn_buffs_and_expiry():
    # Buffs and expiry act on the unit's own value (field - static_*), clamped at 0; statics stay on top and
    # the effective value is clamped at 0 too (the static_* fields record the applied deltas).
    g = _lblank()
    add_unit(g, 0, "back", "hexer2")
    b = add_unit(g, 1, "back", "brute3")
    assert (b.atk, b.static_atk) == (1, -2)
    _cast(g, "weaken")  # own 3 -> 1 (temp -2); effective max(0, 1 - 2) = 0
    assert (b.atk, b.static_atk, b.temp_atk) == (0, -1, -2)
    _cast(g, "purge")  # the aura leaves: the own value shows
    assert (b.atk, b.static_atk, b.temp_atk) == (1, 0, -2)
    g.step(0)  # END_TURN: the "turn" debuff expires
    assert (b.atk, b.static_atk, b.temp_atk) == (3, 0, 0)


def test_static_layering_with_permanent_debuffs_and_move_cost():
    g = _lblank()
    add_unit(g, 1, "back", "drummer")
    b = add_unit(g, 1, "back", "brute3")
    assert (b.atk, b.static_atk, b.move_cost, b.static_move_cost) == (5, 2, 2, 1)
    _cast(g, "sap")  # -5 on the own atk (3 -> 0); the aura stays on top
    assert (b.atk, b.static_atk) == (2, 2)
    g.step(0)
    _cast(g, "rally")  # P1: +1 atk and -2 move cost for the turn, on the own values (move cost 1 -> 0)
    assert (b.atk, b.temp_atk, b.move_cost, b.temp_move_cost, b.static_move_cost) == (3, 1, 1, -1, 1)
    set_hand(g, 1, ["purge"])
    g.step(play(0))
    assert (b.atk, b.move_cost, b.static_atk, b.static_move_cost) == (1, 0, 0, 0)
    g.step(0)
    assert (b.atk, b.move_cost, b.temp_atk, b.temp_move_cost) == (0, 1, 0, 0)


def test_unit_default_base_traits_exclude_turn_and_static_grants():
    assert Unit(0, 0, atk=1, hp=1, defense=True, static_traits=DEFENSE_BIT).base_traits == 0
    assert Unit(0, 0, atk=1, hp=1, smokescreen=True, temp_traits=SMOKESCREEN_BIT).base_traits == 0
    assert Unit(0, 0, atk=1, hp=1, temp_removed=DEFENSE_BIT).base_traits == DEFENSE_BIT
    assert Unit(0, 0, atk=1, hp=1, ambush=True, shock=True, immune=True).base_traits == (AMBUSH_BIT | SHOCK_BIT
                                                                                         | IMMUNE_BIT)
    c = EFFECT_CONFIG.cards.by_id("stormer")
    u = Unit.from_card(c, 1)
    assert u.base_traits == u.trait_mask() == c.trait_mask and u.shock and u.blitz and not u.ambush_used
    assert not u.ambush_ready and Unit.from_card(EFFECT_CONFIG.cards.by_id("bushwhacker"), 0).ambush_ready


def test_combat_damage_with_ambush_shock_and_immune():
    from cardgame.cards import RANGED

    def unit(atk, hp, nature=0, armor=0, **kw):
        return Unit(0, 0, atk=atk, hp=hp, nature=nature, armor=armor, **kw)

    att = unit(3, 3)
    assert combat_damage(att, unit(2, 4)) == (3, 2)
    assert combat_damage(att, unit(2, 4, immune=True)) == (0, 2)
    assert combat_damage(unit(3, 3, immune=True), unit(2, 4)) == (3, 0)
    assert combat_damage(unit(3, 3, shock=True), unit(2, 4)) == (3, 0)
    amb = unit(3, 4, ambush=True)
    assert ambush_fires(att, amb) and combat_damage(att, amb) == (0, 3)        # the strike kills first
    assert combat_damage(unit(3, 5), amb) == (3, 3)                            # it does not: both hit
    assert combat_damage(unit(3, 3, armor=1), amb) == (3, 2)                   # armor saves the attacker
    assert ambush_fires(unit(3, 3, armor=1), amb)
    # no return damage, no ambush (it stays ready): ranged, shock, immune, or armor that absorbs the hit
    for no_return in (unit(3, 3, shock=True), unit(3, 3, nature=RANGED), unit(3, 3, immune=True),
                      unit(3, 3, armor=3)):
        assert not ambush_fires(no_return, amb) and combat_damage(no_return, amb) == (3, 0)
    used = unit(3, 4, ambush=True, ambush_used=True)
    assert not ambush_fires(att, used) and combat_damage(att, used) == (3, 3)
    # UnitViews carry ambush_ready; 13-field Stage 2 style views default to no new traits
    assert combat_damage(att.view(), amb.view()) == (0, 3) and amb.view().ambush_ready
    assert not used.view().ambush_ready and used.view().ambush
    old = UnitView(0, 3, 3, 3, 0, False, 0, 1, False, False, False, True, True)
    assert combat_damage(old, amb.view()) == (0, 3) and combat_damage(amb.view(), old) == (3, 3)


def test_ambush_is_spent_per_turn_and_refreshes_at_every_turn_start():
    g = eblank(coins=10, round_=10)
    amb = add_unit(g, 1, "back", "bushwhacker")  # 2/3 ambush
    add_unit(g, 0, "back", "slinger")              # ranged 1/1
    t = add_unit(g, 0, "front", atk=1, hp=5)
    g.step(attack(0, 0))  # a ranged attacker takes no return damage: Ambush does not fire
    assert (amb.hp, amb.ambush_used) == (2, False)
    g.step(attack(5, 0))  # the ambush strikes first (2), then the attacker's damage follows (1)
    assert (t.hp, amb.hp, amb.ambush_used) == (3, 1, True)
    assert not g.observe(0).opp_backline[0].ambush_ready and g.observe(1).my_backline[0].ambush
    g.step(0)  # P1's turn start refreshes every Ambush
    assert not amb.ambush_used and g.observe(0).opp_backline[0].ambush_ready
    # a hand-set ambush on a pool without the trait still refreshes (invalidate() turns the scan on)
    v = blank_game(config=VANILLA_CONFIG, round_=3)
    x = add_unit(v, 1, "back", atk=1, hp=1, ambush=True, ambush_used=True)
    assert not v._uses_ambush
    v.step(0)
    assert not x.ambush_used


def test_history_counters_per_turn_and_per_game():
    g = eblank(coins=10, round_=10)
    add_unit(g, 1, "back", "militia", hp=1)
    set_hand(g, 0, ["barrage", "veteran"])
    g.step(play(slot(g, 0, "barrage")))  # 1 damage to every enemy unit: the militia dies
    assert g.history_turn == [[1, 0, 0], [0, 0, 1]] and g.history_game == [[1, 0, 0], [0, 0, 1]]
    g.step(play(slot(g, 0, "veteran")))  # a unit died this turn: +2/+2
    v = g.backline[0][-1]
    assert (v.atk, v.hp) == (4, 4) and g.history_turn[0] == [1, 1, 0]
    c = g.clone()
    c.history_game[0][0] = 99
    c.history_turn[1][2] = 99
    assert g.history_game[0][0] == 1 and g.history_turn[1][2] == 1
    g.step(0)  # the next turn starts: per-turn counters reset, per-game counters stay
    assert g.history_turn == [[0, 0, 0], [0, 0, 0]] and g.history_game == [[1, 1, 0], [0, 0, 1]]
    v2 = Game(VANILLA_CONFIG)
    v2.reset(1)
    assert not v2._uses_history and v2.history_game == [[0, 0, 0], [0, 0, 0]]  # not maintained without users


def test_pending_view_reports_buff_atk_and_hp():
    g = eblank(coins=10, round_=10)
    add_unit(g, 0, "back", "militia")
    set_hand(g, 0, ["reinforce"])
    g.step(play(0))
    pending = g.observe(1).pending
    assert pending._replace(previews=()) == PendingView(IDX["reinforce"], 0, "buff", 0, "chosen", "friendly", "unit", 2, 2)


def test_on_attacked_resolves_after_the_on_attack_chain_and_combat_waits(monkeypatch):
    events = []
    resolve, combat = Game._resolve, Game._do_combat
    monkeypatch.setattr(Game, "_resolve", lambda self, inst: (events.append(inst[0].trigger), resolve(self, inst)))
    monkeypatch.setattr(Game, "_do_combat", lambda self, a, t: (events.append("combat"), combat(self, a, t)))
    g = eblank(coins=10, round_=10)
    th = add_unit(g, 0, "front", "thief")   # fast 2/3; on_attack: 1 damage to a chosen enemy unit or base
    se = add_unit(g, 1, "back", "sentry")   # 1/4; on_attacked: 1 damage to the attacker
    g.step(attack(5, 0))
    assert g.phase == CHOICE and g._combat == (th.uid, se.uid, 0) and g.legal_actions() == [choose(0), choose(10)]
    assert events == ["on_attack"]  # pending: on_attacked and combat wait for the on_attack chain
    c = g.clone()
    assert c._combat == g._combat
    g.step(choose(10))
    assert events == ["on_attack", "on_attacked", "combat"]
    assert g.base_hp[1] == 19 and (th.hp, se.hp) == (1, 2) and g._combat is None
    c.step(choose(10))
    assert state_key(c) == state_key(g)


def test_event_units_and_event_damage_of_instances(monkeypatch):
    seen = []
    original = Game._resolve
    monkeypatch.setattr(Game, "_resolve", lambda self, inst: (
        seen.append((inst[0].trigger, inst[1], None if inst[6] is None else inst[6].uid, inst[8])),
        original(self, inst)))
    g = eblank(coins=10, round_=10)
    add_unit(g, 0, "back", "reaper")           # on_kill watcher (friendly; event filter: the victim is infantry)
    k = add_unit(g, 0, "front", atk=3, hp=9)
    se = add_unit(g, 1, "back", "sentry", hp=3)  # infantry; on_attacked: 1 damage to the attacker
    sig = add_unit(g, 1, "back", "signal")      # on_attacked watcher: the event unit is the attacker
    g.step(attack(5, 0))
    assert seen == [("on_attacked", IDX["sentry"], k.uid, 0), ("on_attacked", IDX["signal"], k.uid, 0),
                    ("on_kill", IDX["reaper"], se.uid, 3)]
    assert k.hp == 7 and sig.atk == 2 and g.backline[0][-1].card == IDX["scout_car"]


def test_invalidate_recomputes_statics_and_legal_actions_follow():
    g = eblank(coins=10, round_=10)
    u = add_unit(g, 0, "back", "militia")      # 1/2
    add_unit(g, 0, "back", "banner")           # other friendly units +1/+1
    assert (u.atk, u.hp, u.max_hp, u.static_atk, u.static_hp) == (2, 3, 3, 1, 1) and g._static_on
    add_unit(g, 1, "back", "smokepot")         # adjacent friendly units have smokescreen
    hidden = add_unit(g, 1, "back", "militia")
    assert hidden.smokescreen and hidden.static_traits == SMOKESCREEN_BIT and not hidden.base_traits
    add_unit(g, 0, "back", "slinger")          # ranged attacker in slot 2
    assert [a for a in g.legal_actions() if attack(2, 0) <= a <= attack(2, 10)] == [attack(2, 0), attack(2, 10)]


# ---------------------------------------------------------------- fuzz: random games on the effect pool
class Probe(Game):
    """Records which triggers resolve, which actions apply and which phase-1b mechanics run (fuzz coverage)."""
    seen: Counter = Counter()

    def _resolve(self, inst):
        Probe.seen["trigger:" + inst[0].trigger] += 1
        super()._resolve(inst)

    def _apply(self, eff, inst, targets, amounts):
        Probe.seen["action:" + eff.action] += 1
        Probe.seen["select:" + eff.target.select] += 1
        Probe.seen["kind:" + eff.target.kind] += 1
        Probe.seen["else"] += eff.is_else
        Probe.seen["repeat"] += eff.repeat > 1
        Probe.seen["prev_read"] += bool(inst[7])
        super()._apply(eff, inst, targets, amounts)

    def _add_static(self, contrib, body, targets):
        Probe.seen["trigger:static"] += 1
        Probe.seen["static:" + body.action + ("" if body.target.select != "adjacent" else ":adjacent")] += 1
        Probe.seen["static:else"] += body.is_else
        super()._add_static(contrib, body, targets)

    def _do_combat(self, attacker, target):
        if target is not None:
            Probe.seen["ambush"] += ambush_fires(attacker, target)
            Probe.seen["shock"] += bool(attacker.shock)
            Probe.seen["immune"] += bool(target.immune or attacker.immune)
        super()._do_combat(attacker, target)


def perturb_hidden(game: Game, observer: int, rng: random.Random) -> Game:
    """A clone that differs only in what `observer` cannot see: the opponent's unknown hand cards and deck
    (redealt from the same multiset), both decks' order, the opponent's mulligan marks and the RNG."""
    g = game.clone()
    o = 1 - observer
    known = Counter({c: k for c, k in enumerate(g.known_hand[o]) if k})
    hand = Counter(g.hands[o])
    unknown = list((hand - known).elements()) + g.deck_cards[o]
    rng.shuffle(unknown)
    n_unknown = len(g.hands[o]) - sum(known.values())
    g.hands[o] = sorted(list(known.elements()) + unknown[:n_unknown])
    g.deck_cards[o] = unknown[n_unknown:]
    rng.shuffle(g.deck_cards[observer])
    if g.phase == MULLIGAN and g.current == o:
        g.mulligan_marks = set(rng.sample(range(len(g.hands[o])), rng.randrange(len(g.hands[o]) + 1)))
    g.rng = random.Random(rng.getrandbits(64))
    g.invalidate()
    return g


def fuzz_game(seed: int, check_every: int = 5) -> tuple:
    cfg = EFFECT_CONFIG
    decks = sample_deal(seed, cfg, 0.7)
    g = Probe(cfg)
    g.reset(seed, decks)
    rng = random.Random(seed)
    actions = []
    stats = Counter()
    while not g.done:
        legal = g.legal_actions()
        assert legal and legal == sorted(set(legal))
        assert np.flatnonzero(g.legal_mask()).tolist() == legal
        check_invariants(g)
        step = len(actions)
        if step % check_every == 0:
            p = g.current
            o = 1 - p
            for who in (p, o):
                d = g.determinize(who, random.Random(seed * 7919 + step))
                assert d.observe(who) == g.observe(who)
                if who == p:
                    assert d.legal_actions() == legal
                check_invariants(d)  # card conservation holds for the resampled decklist too
                rev = Counter({c: k for c, k in enumerate(g.revealed[1 - who]) if k})
                assert not rev - Counter(d.decklists[1 - who]) and d.deck_ids[1 - who] == -1
                alt = perturb_hidden(g, who, random.Random(step))
                assert alt.observe(who) == g.observe(who)
                d2 = alt.determinize(who, random.Random(seed * 7919 + step))
                assert state_key(d2) == state_key(d), "determinize read hidden information"
                d3 = g.determinize(who, random.Random(seed * 7919 + step + 1))
                stats["differs"] += (d3.hands[1 - who], d3.deck_cards[1 - who]) != (d.hands[1 - who],
                                                                                    d.deck_cards[1 - who])
                stats["determinized"] += 1
            c = g.clone()
            assert state_key(c) == state_key(g)
            if g._any_static:  # the recompute from scratch is idempotent at rest (SPEC §2.12)
                r = g.clone()
                r._refresh_statics()
                assert state_key(r) == state_key(g), "a second static recompute changed the state"
                stats["static_on"] += g._static_on
        weights = [0.25 if a == 0 else 1.0 for a in legal]
        a = rng.choices(legal, weights)[0]
        if step % check_every == 0:  # a clone steps identically
            c.step(a)
        g.step(a)
        if step % check_every == 0:
            assert state_key(c) == state_key(g)
        actions.append(a)
        stats["choice"] += g.phase == CHOICE
        assert len(actions) < 5000
    check_invariants(g)
    replay = Probe(cfg)
    replay.reset(seed, decks)
    for a in actions:
        replay.step(a)
    assert state_key(replay) == state_key(g)
    stats["guard"] += g.guard_trips
    stats["draw" if g.winner() == -1 else "win"] += 1
    return stats


def test_fuzz_effect_pool_with_mulligan():
    Probe.seen.clear()
    total = Counter()
    for seed in range(24):
        total += fuzz_game(seed)
    from cardgame.cards import ACTIONS, KINDS, SELECTS, TRIGGERS
    missing = ([f"trigger:{t}" for t in TRIGGERS] + [f"action:{a}" for a in ACTIONS]
               + [f"select:{s}" for s in SELECTS] + [f"kind:{k}" for k in KINDS]
               + ["else", "repeat", "prev_read", "static:buff", "static:add_trait", "static:add_trait:adjacent",
                  "static:else", "ambush", "shock", "immune"])
    missing = [m for m in missing if not Probe.seen[m]]
    assert not missing, f"the fuzz never exercised {missing}"
    assert total["choice"] > 50 and total["win"] >= 20 and total["static_on"] > 0
    assert total["differs"] >= 0.8 * total["determinized"], total
