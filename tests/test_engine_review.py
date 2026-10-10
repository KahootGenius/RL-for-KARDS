"""Regression tests for the Stage 3 engine review (one block per confirmed finding).

1. Event filters see static contributions at fire time (SPEC 1.2b `static`, SPEC 2.13 "event_filter is
   checked when the event fires, against the event unit (current ... values)").
2. A `target_condition` split (the action plus the `else` body) is one damage step (SPEC 2.8: "Damage
   from one effect ... is applied simultaneously"; resolution order select, amount, apply, damage step).
3. A unit created for the non-turn player can act on its owner's next turn (SPEC 2.4 Blitz: only the
   turn of deployment/summoning is lost; SPEC 2.3 END_TURN step 4).
4. determinize() keeps no state that depends on the opponent's hidden information (SPEC 4).
5. The static recompute schedule does not depend on unrelated cards in the pool (SPEC 2.12 / 2.13 "When").
"""
from __future__ import annotations

import random

from stage3_helpers import (CONFIRM, END, Game, Rules, adjacent, attack, back, blank, cards_mod, effect, front,
                            hand_slot, move, mull, operation, play, put, rich_rules, set_hand, state_key, static, tgt,
                            unit, with_else)


def stats(units):
    return [(u.atk, u.hp, u.max_hp) for u in units]


# ================================================================= 1. event filters vs static contributions
def _watcher(trigger, flt):
    """A unit whose friendly `trigger` watcher deals 3 to the enemy base when the event unit matches `flt`."""
    return unit("watch", 0, 5, effects=[effect(trigger, "damage", "enemy_base", scope="friendly", amount=3,
                                               event_filter=flt)])


def test_on_deploy_watcher_sees_a_trait_granted_by_an_aura():
    aura = unit("aura", 0, 5, effects=[static("add_trait", tgt("all", "friendly", "unit"), trait="defense")])
    r = Rules([aura, _watcher("on_deploy", {"trait": "defense"}), unit("recruit", 1, 1)])
    g = blank(r)
    put(g, r, 0, "back", "aura")
    put(g, r, 0, "back", "watch")
    set_hand(g, r, 0, ["recruit"])
    g.step(play(0))
    assert g.backline[0][-1].defense
    assert g.base_hp[1] == 17


def test_on_deploy_watcher_sees_the_deployed_units_own_static():
    r = Rules([_watcher("on_deploy", {"min_atk": 3}),
               unit("hothead", 1, 3, effects=[static("buff", "self", atk=2)])])
    g = blank(r)
    put(g, r, 0, "back", "watch")
    set_hand(g, r, 0, ["hothead"])
    g.step(play(0))
    assert g.backline[0][-1].atk == 3
    assert g.base_hp[1] == 17


def test_on_move_watcher_sees_a_frontline_aura():
    aura = unit("aura", 0, 5, effects=[static("buff", tgt("all", "friendly", "unit", zone="frontline"), atk=2)])
    r = Rules([aura, _watcher("on_move", {"min_atk": 3}), unit("mover", 1, 3)])
    g = blank(r)
    put(g, r, 0, "back", "aura")
    put(g, r, 0, "back", "watch")
    put(g, r, 0, "back", "mover")
    g.step(move(2))
    assert g.frontline[0].atk == 3
    assert g.base_hp[1] == 17


def test_on_move_watcher_no_longer_sees_a_backline_aura():
    # the symmetric case: the unit left the aura's zone, so a trait filter must not match any more
    aura = unit("aura", 0, 5, effects=[static("add_trait", tgt("all", "friendly", "unit", zone="backline"),
                                              trait="defense")])
    r = Rules([aura, _watcher("on_move", {"trait": "defense"}), unit("mover", 1, 3)])
    g = blank(r)
    put(g, r, 0, "back", "aura")
    put(g, r, 0, "back", "watch")
    m = put(g, r, 0, "back", "mover")
    assert m.defense
    g.step(move(2))
    assert not g.frontline[0].defense
    assert g.base_hp[1] == 20


def test_on_attack_watcher_sees_an_aura_lost_by_attacking():
    # "friendly units with smokescreen have +1 atk": attacking loses smokescreen, so the attacker is a 1-atk
    # unit when on_attack fires
    aura = unit("aura", 0, 5, effects=[static("buff", tgt("all", "friendly", "unit", filter={"trait": "smokescreen"}),
                                              atk=1)])
    r = Rules([aura, _watcher("on_attack", {"max_atk": 1}),
               unit("scout", 1, 3, nature="ranged", traits={"smokescreen": True})])
    g = blank(r)
    put(g, r, 0, "back", "aura")
    put(g, r, 0, "back", "watch")
    s = put(g, r, 0, "back", "scout")
    assert s.atk == 2
    g.step(attack(back(2), 10))
    assert s.atk == 1 and not s.smokescreen
    assert g.base_hp[1] == 20 - 3 - 1


def test_fire_time_recompute_keeps_the_recompute_count():
    # moving the first recompute of a deploy to the moment the event fires must not add a recompute: a
    # self-referential aura (one pass, SPEC 2.13) ends where it ended before, with and without listeners
    hot = unit("hothead", 1, 5, effects=[static("buff", "self", atk=2, condition={
        "type": "compare", "left": {"stat": "atk", "of": "self"}, "op": "<", "right": 3})])
    ping = unit("ping", 0, 5, effects=[effect("on_deploy", "gain_coins", "controller", scope="friendly", amount=0)])
    for listeners in ([], ["ping"]):
        r = Rules([hot, ping])
        g = blank(r)
        for cid in listeners:
            put(g, r, 0, "back", cid)
        set_hand(g, r, 0, ["hothead"])
        g.step(play(0))
        # deploy: recompute (1 -> 3); with a listener the instance's own recompute replaced the after-action one
        assert g.backline[0][-1].atk == (3 if not listeners else 1), listeners


# ================================================================= 2. target_condition + else = one damage step
def test_target_condition_split_is_one_damage_step_for_an_hp_aura():
    src = unit("src", 0, 1, effects=[static("buff", tgt("all", "friendly", "unit", filter={"other": True}), hp=2)])
    purge = operation("purge", [with_else(effect("on_play", "destroy", tgt("all", "enemy", "unit"),
                                                 target_condition={"nature": "troop"}), action="damage", amount=2)])
    r = Rules([src, unit("bee", 1, 1, nature="fast"), purge])
    g = blank(r)
    put(g, r, 1, "back", "src")
    bee = put(g, r, 1, "back", "bee")
    assert stats([bee]) == [(1, 3, 3)]
    set_hand(g, r, 0, ["purge"])
    g.step(play(0))
    # the aura is lost in the same damage step as the 2 damage, so it caps hp at 1 and never kills
    assert g.backline[1] == [bee] and stats([bee]) == [(1, 1, 1)]


def test_target_condition_split_enqueues_like_one_damage_step():
    # SPEC 2.8: one damage step enqueues on_damaged (survivors) before on_death (removed units)
    trooper = unit("trooper", 1, 1, effects=[effect("on_death", "buff", tgt("all", "friendly", "unit"), atk=2)])
    bee = unit("bee", 1, 3, nature="fast",
               effects=[effect("on_damaged", "damage", "enemy_base", amount={"stat": "atk", "of": "self"})])
    split = operation("split", [with_else(effect("on_play", "damage", tgt("all", "enemy", "unit"), amount=1,
                                                 target_condition={"nature": "troop"}), action="damage", amount=1)])
    plain = operation("plain", [effect("on_play", "damage", tgt("all", "enemy", "unit"), amount=1)])
    out = {}
    for name in ("split", "plain"):
        r = Rules([trooper, bee, split, plain])
        g = blank(r)
        put(g, r, 1, "back", "trooper")
        put(g, r, 1, "back", "bee")
        set_hand(g, r, 0, [name])
        g.step(play(0))
        out[name] = (g.base_hp[0], stats(g.backline[1]))
    assert out["split"] == out["plain"] == (19, [(3, 2, 3)])


def test_target_condition_else_amount_is_evaluated_before_the_damage_step():
    # SPEC 2.8 resolution: select, evaluate the amount, apply, then the damage step
    purge = operation("purge", [with_else(effect("on_play", "destroy", tgt("all", "enemy", "unit"),
                                                 target_condition={"nature": "troop"}),
                                          action="damage", amount={"count": "units", "side": "enemy"})])
    r = Rules([unit("trooper", 1, 1), unit("bee", 1, 5, nature="fast"), purge])
    g = blank(r)
    put(g, r, 1, "back", "trooper")
    put(g, r, 1, "back", "trooper")
    bee = put(g, r, 1, "back", "bee")
    set_hand(g, r, 0, ["purge"])
    g.step(play(0))
    assert g.backline[1] == [bee] and bee.hp == 5 - 3


def test_target_condition_else_with_its_own_target_shares_the_damage_step():
    # the else body's own targets overlap the main targets: one damage step, so the twice-hit troop takes 2
    # and fires on_damaged once (SPEC 2.8: on_damaged for each damaged survivor)
    sting = operation("sting", [with_else(
        effect("on_play", "damage", tgt("all", "enemy", "unit"), amount=1, target_condition={"nature": "troop"}),
        action="damage", amount=1, target=tgt("all", "enemy", "unit"))])
    r = Rules([unit("trooper", 1, 5, effects=[effect("on_damaged", "damage", "enemy_base", amount=1)]),
               unit("bee", 1, 5, nature="fast"), sting])
    g = blank(r)
    tr = put(g, r, 1, "back", "trooper")
    bee = put(g, r, 1, "back", "bee")
    set_hand(g, r, 0, ["sting"])
    g.step(play(0))
    assert (tr.hp, bee.hp) == (3, 4)
    assert g.base_hp[0] == 19


def test_target_condition_split_records_both_parts_for_prev():
    # SPEC 2.8b: the clause records the targets after the split (main + else), killed = any of them died
    split = operation("split", [
        with_else(effect("on_play", "destroy", tgt("all", "enemy", "unit"), target_condition={"nature": "troop"}),
                  action="damage", amount=1),
        effect("on_play", "damage", tgt("prev", kind="unit"), amount=1, condition={"type": "prev", "killed": True})])
    r = Rules([unit("trooper", 1, 1), unit("bee", 1, 5, nature="fast"), split])
    g = blank(r)
    put(g, r, 1, "back", "trooper")
    bee = put(g, r, 1, "back", "bee")
    set_hand(g, r, 0, ["split"])
    g.step(play(0))
    assert g.backline[1] == [bee] and bee.hp == 3  # 1 from the else body, 1 from the prev clause


# ================================================================= 3. summoned flag of units created on the other turn
def test_unit_summoned_for_the_opponent_acts_on_its_owners_next_turn():
    r = Rules([unit("bugler", 1, 1, effects=[effect("on_death", "summon", "controller", card="militia")]),
               unit("militia", 2, 2, token=True), unit("archer", 3, 3, nature="ranged")])
    g = blank(r, current=0, first=0)
    put(g, r, 0, "back", "archer")
    put(g, r, 1, "back", "bugler")
    put(g, r, 1, "back", "archer")
    g.step(attack(back(0), back(0)))  # P0 kills the bugler: P1 gets a militia during P0's turn
    tok = g.backline[1][-1]
    assert tok.token and tok.summoned
    g.step(END)
    assert g.current == 1
    assert not tok.summoned
    view = g.observe(1).my_backline[1]
    assert view.token and view.can_move and view.can_attack
    assert move(1) in g.legal_actions()


def test_unit_summoned_on_its_owners_turn_still_waits():
    r = Rules([operation("call", [effect("on_play", "summon", "controller", card="militia")]),
               unit("militia", 2, 2, token=True)])
    g = blank(r, current=0, first=0)
    set_hand(g, r, 0, ["call"])
    g.step(play(0))
    tok = g.backline[0][-1]
    assert tok.summoned and move(0) not in g.legal_actions()
    g.step(END)
    assert not tok.summoned  # refreshed at its owner's END_TURN, as before


def test_enemy_summon_on_the_opponents_turn_is_ready_after_end_turn():
    # a summon targeting the opponent (side enemy) during p's turn
    r = Rules([operation("gift", [effect("on_play", "summon", "opponent", card="militia")]),
               unit("militia", 2, 2, token=True)])
    g = blank(r, current=0, first=0)
    set_hand(g, r, 0, ["gift"])
    g.step(play(0))
    tok = g.backline[1][-1]
    assert tok.summoned
    g.step(END)
    assert not tok.summoned and move(0) in g.legal_actions()


# ================================================================= 4. determinize keeps no hidden-dependent state
def _after_first_mulligan(marks):
    r = rich_rules(mulligan=True)
    g = Game(r.config)
    g.reset(5, cards_mod.sample_deal(5, r.config, 0.5))
    for i in marks:
        g.step(mull(i))
    g.step(CONFIRM)  # the first player's mulligan is hidden from the second player (SPEC 2.2)
    return g, 1 - g.first_player


def test_determinize_does_not_depend_on_the_opponents_mulligan_marks():
    a, s = _after_first_mulligan([])
    b, _ = _after_first_mulligan([0, 1, 2])
    assert a.observe(s) == b.observe(s) and a.legal_actions() == b.legal_actions()
    da, db = a.determinize(s, random.Random(7)), b.determinize(s, random.Random(7))
    assert da.num_steps == db.num_steps
    assert state_key(da) == state_key(db)


def test_determinize_drops_the_decklist_cache():
    g, s = _after_first_mulligan([1])
    o = 1 - s
    g.observe(o)  # fills the cache with o's true decklist
    d = g.determinize(s, random.Random(1))
    assert "_dlc" not in d.__dict__
    assert d.observe(o).my_decklist == tuple(d.decklists[o].count(c) for c in range(len(d.config.cards)))
    assert d.observe(s) == g.observe(s)


# ================================================================= 5. the recompute schedule is pool-independent
def test_combat_does_not_depend_on_an_unused_on_attacked_card():
    # chained auras (one pass, SPEC 2.13): the result follows the recompute points, which must not change
    # because the pool happens to contain an on_attacked effect that is not on the board
    drummer = unit("drummer", 0, 5, effects=[static("buff", adjacent("friendly"), atk=1)])
    b_aura = unit("b_aura", 0, 5, effects=[static("buff", tgt("all", "friendly", filter={"min_atk": 2, "other": True}),
                                                  atk=1)])
    d_aura = unit("d_aura", 0, 5, effects=[static("buff", tgt("all", "friendly", filter={"min_atk": 3, "other": True}),
                                                  atk=1)])
    bystander = unit("bystander", 0, 1, effects=[effect("on_attacked", "damage", tgt("event"), amount=1)])
    dealt = []
    for extra in ([], [bystander]):
        r = Rules([drummer, b_aura, d_aura, unit("grunt", 1, 5), unit("wall", 0, 30)] + extra)
        g = blank(r)
        put(g, r, 0, "back", "b_aura")
        put(g, r, 0, "back", "d_aura")
        put(g, r, 0, "back", "grunt")
        wall = put(g, r, 1, "front", "wall")
        set_hand(g, r, 0, ["drummer"])
        g.step(play(hand_slot(g, r, 0, "drummer")))
        g.step(attack(back(2), front(0)))
        dealt.append(30 - wall.hp)
    assert dealt[0] == dealt[1]


def test_on_attacked_listener_still_resolves_before_combat():
    # the on_attacked stage still runs (and is followed by its own pre-combat recompute) when it has listeners
    r = Rules([unit("thorn", 0, 5, effects=[effect("on_attacked", "damage", tgt("event"), amount=1)]),
               unit("grunt", 2, 1)])
    g = blank(r)
    a = put(g, r, 0, "front", "grunt")
    t = put(g, r, 1, "back", "thorn")
    g.step(attack(front(0), back(0)))
    assert a not in g.frontline and t.hp == 5  # the attacker died before combat
