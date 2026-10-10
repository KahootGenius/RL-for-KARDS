"""Independent Stage 3 tests of pending choices and the action space (SPEC 2.6, 2.8, 2.9, 3).

Choice slots are computed from SPEC 2.9 (enemy backline t < Z, frontline Z <= t < 2Z of either
owner, enemy base 2Z, own backline 2Z+1.., own base 3Z+1); the action indices from SPEC 3.
"""
from __future__ import annotations

import pytest

import cardgame.engine as engine_mod
from cardgame.actions import ActionKind, ActionSpace
from stage3_helpers import (BASE, CHOICE, CHOOSE0, CONFIRM, END, ENEMY_BASE, MAIN, MULLIGAN0, N_ACTIONS, OWN_BASE,
                            IllegalActionError, Rules, attack, back, bases_of, blank, choose, chooses, counts,
                            effect, find, front, hand_slot, move, operation, own_back, play, put, set_deck,
                            set_hand, tgt, unit, zone_of)


def cast(g, r, cid):
    p = g.current
    g.hands[p] = sorted(list(g.hands[p]) + [r.idx(cid)])
    g.invalidate()
    a = play(hand_slot(g, r, p, cid))
    assert a in g.legal_actions(), f"PLAY {cid} not legal: {g.legal_actions()}"
    g.step(a)


def hp(g, u):
    x = find(g, u.uid)
    return None if x is None else x.hp


def op(cid, target, action="damage", amount=1, condition=None, cost=1, **params):
    if action in ("damage", "heal", "draw") and "amount" not in params:
        params["amount"] = amount
    return operation(cid, [effect("on_play", action, target, condition=condition, **params)], cost=cost)


# ================================================================= constants and action space (SPEC 3, 4)
def test_phase_constants():
    # SPEC 4: game.phase  # MULLIGAN=0, MAIN=1, CHOICE=2 (module constants)
    assert (engine_mod.MULLIGAN, engine_mod.MAIN, engine_mod.CHOICE) == (0, 1, 2)


def test_action_space_layout():
    # SPEC 3: Stage 2 indices unchanged; CHOOSE0 = 126, MULLIGAN0 = 143, CONFIRM = 153, N = 154.
    sp = ActionSpace(10, 5)
    assert sp.n == N_ACTIONS == 154
    assert (sp.END_TURN, sp.MOVE0, sp.ATTACK0) == (0, 11, 16)
    assert (sp.CHOOSE0, sp.MULLIGAN0, sp.CONFIRM) == (126, 143, 153)
    assert (sp.n_choose, sp.ENEMY_BASE_CHOICE, sp.OWN_BASE_CHOICE) == (17, 10, 16)
    assert all(sp.attack(a, t) == 16 + a * 11 + t for a in range(10) for t in range(11))
    for name in ("END_TURN", "PLAY", "MOVE", "ATTACK", "CHOOSE", "MULLIGAN", "CONFIRM"):
        assert hasattr(ActionKind, name)


def test_decode_encode_describe():
    # SPEC 3: decode, encode, describe cover the new kinds.
    sp = ActionSpace(10, 5)
    kinds = [sp.decode(i).kind for i in range(154)]
    assert kinds[0] == ActionKind.END_TURN
    assert kinds[1:11] == [ActionKind.PLAY] * 10
    assert kinds[11:16] == [ActionKind.MOVE] * 5
    assert kinds[16:126] == [ActionKind.ATTACK] * 110
    assert kinds[126:143] == [ActionKind.CHOOSE] * 17
    assert kinds[143:153] == [ActionKind.MULLIGAN] * 10
    assert kinds[153] == ActionKind.CONFIRM
    for i in range(154):
        assert sp.encode(*sp.decode(i)) == i
    assert [sp.encode(ActionKind.CHOOSE, t) for t in range(17)] == list(range(126, 143))
    assert [sp.encode(ActionKind.MULLIGAN, i) for i in range(10)] == list(range(143, 153))
    assert sp.encode(ActionKind.CONFIRM) == 153
    descs = [sp.describe(i) for i in range(154)]
    assert all(isinstance(d, str) and d for d in descs)
    assert len(set(descs)) == 154


def test_game_mask_has_154_entries():
    r = Rules()
    g = blank(r)
    assert g.legal_mask().shape == (154,)


# ================================================================= the pending state (SPEC 2.6, 2.9, 5)
def grenadier_rules(**kw):
    return Rules([unit("grenadier", 2, 3, cost=2, effects=[
        effect("on_deploy", "damage", tgt("chosen", "enemy"), amount=1)])], **kw)


def test_choice_pending_state_and_resolution():
    # SPEC 2.9 chosen: phase CHOICE with `pending` set (even with a single option); SPEC 2.6 only
    # CHOOSE is legal and the player to act is the turn player; SPEC 5 pending is public.
    r = grenadier_rules()
    g = blank(r, coins=5)
    e0 = put(g, r, 1, "back", "fill03")
    e1 = put(g, r, 1, "back", "fill03")
    cast(g, r, "grenadier")
    assert g.phase == CHOICE and g.pending is not None
    assert g.current_player() == 0
    assert g.legal_actions() == [choose(0), choose(1)]
    mask = g.legal_mask()
    assert [i for i in range(154) if mask[i]] == [choose(0), choose(1)]
    assert [r.card_dicts[u.card]["id"] for u in g.backline[0]] == ["grenadier"] and g.coins[0] == 3
    obs = g.observe(0)
    assert obs.phase == CHOICE
    assert obs.pending.card == r.idx("grenadier") and obs.pending.effect == 0 and obs.pending.amount == 1
    assert g.observe(1).pending == obs.pending and g.observe(1).phase == CHOICE
    for bad in (END, play(0), choose(2), choose(ENEMY_BASE)):
        with pytest.raises(IllegalActionError):
            g.step(bad)
    g.step(choose(1))
    assert (hp(g, e0), hp(g, e1)) == (4, 3)
    assert g.phase == MAIN and g.pending is None and len(g.queue) == 0
    assert g.observe(0).pending is None
    assert END in g.legal_actions()


def test_single_option_still_asks():
    # SPEC 2.9: phase becomes CHOICE "even with a single option".
    r = grenadier_rules()
    g = blank(r)
    put(g, r, 1, "back", "fill03")
    cast(g, r, "grenadier")
    assert g.phase == CHOICE and g.legal_actions() == [choose(0)]


def test_choice_on_player_ones_turn():
    r = grenadier_rules()
    g = blank(r, current=1, first=0)
    e = put(g, r, 0, "back", "fill03")
    cast(g, r, "grenadier")
    assert g.phase == CHOICE and g.current_player() == 1
    assert g.legal_actions() == [choose(0)]
    g.step(choose(0))
    assert hp(g, e) == 3 and g.current_player() == 1 and g.phase == MAIN


# ================================================================= option slots (SPEC 2.9 / 3)
def apply_and_check(g, t, units_by_slot, bases_expected):
    """Clone, CHOOSE t, and check that exactly the unit / base mapped to t took 1 damage."""
    c = g.clone()
    c.step(choose(t))
    for slot, u in units_by_slot.items():
        assert hp(c, u) == u.max_hp - (1 if slot == t else 0), (t, slot)
    assert bases_of(c) == bases_expected.get(t, [20, 20]), t


def test_option_slots_enemy_front():
    # SPEC 2.9 canonical slots, frontline held by the enemy.
    r = Rules([op("strike", tgt("chosen", "any", "unit_or_base"))])
    g = blank(r)
    a0 = put(g, r, 0, "back", "fill03")
    a1 = put(g, r, 0, "back", "fill07")
    b0 = put(g, r, 1, "back", "fill03")
    b1 = put(g, r, 1, "back", "fill07")
    b2 = put(g, r, 1, "back", "fill11")
    f0 = put(g, r, 1, "front", "fill03")
    f1 = put(g, r, 1, "front", "fill07")
    cast(g, r, "strike")
    assert chooses(g) == [0, 1, 2, 5, 6, 10, 11, 12, 16]
    slots = {0: b0, 1: b1, 2: b2, front(0): f0, front(1): f1, own_back(0): a0, own_back(1): a1}
    for t in chooses(g):
        apply_and_check(g, t, slots, {ENEMY_BASE: [20, 19], OWN_BASE: [19, 20]})


def test_option_slots_own_front():
    # SPEC 2.9: Z <= t < 2Z is the frontline of either owner.
    r = Rules([op("strike", tgt("chosen", "any", "unit_or_base"))])
    g = blank(r)
    a0 = put(g, r, 0, "back", "fill03")
    f0 = put(g, r, 0, "front", "fill03")
    f1 = put(g, r, 0, "front", "fill07")
    b0 = put(g, r, 1, "back", "fill03")
    cast(g, r, "strike")
    assert chooses(g) == [0, 5, 6, 10, 11, 16]
    slots = {0: b0, front(0): f0, front(1): f1, own_back(0): a0}
    for t in chooses(g):
        apply_and_check(g, t, slots, {ENEMY_BASE: [20, 19], OWN_BASE: [19, 20]})


def test_option_slots_are_relative_to_the_chooser():
    # SPEC 2.9 (p1 chooses): enemy = p0's backline, own = p1's backline, own base = p1's base.
    r = Rules([op("strike", tgt("chosen", "any", "unit_or_base"))])
    g = blank(r, current=1, first=0)
    a0 = put(g, r, 0, "back", "fill03")
    a1 = put(g, r, 0, "back", "fill07")
    f0 = put(g, r, 0, "front", "fill03")
    b0 = put(g, r, 1, "back", "fill03")
    cast(g, r, "strike")
    assert chooses(g) == [0, 1, 5, 10, 11, 16]
    slots = {0: a0, 1: a1, front(0): f0, own_back(0): b0}
    for t in chooses(g):
        apply_and_check(g, t, slots, {ENEMY_BASE: [19, 20], OWN_BASE: [20, 19]})


@pytest.mark.parametrize("target, expected", [
    (tgt("chosen", "enemy"), [0, 5, 6]),
    (tgt("chosen", "enemy", zone="backline"), [0]),
    (tgt("chosen", "enemy", zone="frontline"), [5, 6]),
    (tgt("chosen", "friendly"), [11, 12, 13]),   # the caller itself is on the board (own backline 2)
    (tgt("chosen", "friendly", zone="frontline"), []),
    (tgt("chosen", "any"), [0, 5, 6, 11, 12, 13]),
    (tgt("chosen", "enemy", "base"), [10]),
    (tgt("chosen", "friendly", "base"), [16]),
    (tgt("chosen", "any", "base"), [10, 16]),
    (tgt("chosen", "enemy", "unit_or_base"), [0, 5, 6, 10]),
    (tgt("chosen", "friendly", "unit_or_base"), [11, 12, 13, 16]),
    (tgt("chosen", "enemy", filter={"min_hp": 4}), [0, 6]),
    (tgt("chosen", "any", filter={"nature": "fast"}), [6, 12]),
])
def test_options_follow_side_kind_zone_and_filter(target, expected):
    # SPEC 1.2 side/kind/zone/filter applied to the SPEC 2.9 slots; no options -> fizzle.
    r = Rules([unit("runner", 1, 4, nature="fast"),
               unit("caller", 1, 1, effects=[effect("on_deploy", "damage", target, amount=1)])])
    g = blank(r)
    put(g, r, 0, "back", "fill03")
    put(g, r, 0, "back", "runner")
    put(g, r, 1, "back", "fill03")
    put(g, r, 1, "front", "fill00")
    put(g, r, 1, "front", "runner")
    cast(g, r, "caller")
    if expected:
        assert g.phase == CHOICE and chooses(g) == expected
    else:
        assert g.phase == MAIN and g.pending is None  # fizzled


def test_friendly_frontline_slots():
    r = Rules([unit("caller", 1, 1, effects=[effect("on_deploy", "damage", tgt("chosen", "friendly"), amount=1)])])
    g = blank(r)
    put(g, r, 0, "front", "fill03")
    put(g, r, 0, "front", "fill03")
    cast(g, r, "caller")
    assert chooses(g) == [5, 6, 11]  # own frontline 0, 1 and the caller itself in own backline 0


# ================================================================= fizzle, chains, playability
def test_chosen_without_options_fizzles_and_the_unit_still_deploys():
    # SPEC 2.9: no options -> fizzle (operations need options to be played, units do not: SPEC 2.6).
    r = grenadier_rules()
    g = blank(r)
    set_hand(g, r, 0, ["grenadier"])
    assert play(0) in g.legal_actions()
    g.step(play(0))
    assert g.phase == MAIN and g.pending is None and len(g.queue) == 0
    assert len(g.backline[0]) == 1


def test_two_choices_in_one_chain():
    # SPEC 2.11: two choices in one chain; options are recomputed when each effect resolves and
    # zones compact after deaths (SPEC 2.8).
    r = Rules([unit("double", 1, 1, effects=[effect("on_deploy", "damage", tgt("chosen", "enemy"), amount=1),
                                              effect("on_deploy", "damage", tgt("chosen", "enemy"), amount=1)])])
    g = blank(r)
    e0 = put(g, r, 1, "back", "fill00")  # 1/1
    e1 = put(g, r, 1, "back", "fill03")  # 1/4
    cast(g, r, "double")
    assert g.phase == CHOICE and chooses(g) == [0, 1] and g.observe(0).pending.effect == 0
    g.step(choose(0))
    assert find(g, e0.uid) is None
    assert g.phase == CHOICE and chooses(g) == [0] and g.observe(0).pending.effect == 1
    g.step(choose(0))
    assert hp(g, e1) == 3
    assert g.phase == MAIN and g.pending is None and len(g.queue) == 0


def test_second_choice_fizzles_when_no_option_is_left():
    r = Rules([unit("double", 1, 1, effects=[effect("on_deploy", "damage", tgt("chosen", "enemy"), amount=1),
                                              effect("on_deploy", "damage", tgt("chosen", "enemy"), amount=1)])])
    g = blank(r)
    put(g, r, 1, "back", "fill00")
    cast(g, r, "double")
    g.step(choose(0))
    assert g.phase == MAIN and g.backline[1] == [] and g.pending is None


def test_queue_continues_after_choose():
    # SPEC 2.6 CHOOSE: the pending effect resolves on t and the queue continues; SPEC 4 the queue
    # is not empty while a choice is pending.
    r = Rules([unit("triple", 1, 1, effects=[effect("on_deploy", "damage", tgt("chosen", "enemy"), amount=1),
                                              effect("on_deploy", "draw", "controller", amount=1),
                                              effect("on_deploy", "damage", "enemy_base", amount=2)])])
    g = blank(r)
    e = put(g, r, 1, "back", "fill03")
    set_deck(g, r, 0, ["fill05"])
    cast(g, r, "triple")
    assert g.phase == CHOICE and len(g.queue) >= 2
    assert list(g.hands[0]) == [] and bases_of(g) == [20, 20]  # later effects wait
    g.step(choose(0))
    assert hp(g, e) == 3 and list(g.hands[0]) == [r.idx("fill05")] and bases_of(g) == [20, 18]
    assert g.phase == MAIN and len(g.queue) == 0


def test_pending_amount_is_resolved():
    # SPEC 5: PendingView amount is already resolved.
    r = Rules([op("focus", tgt("chosen", "enemy"), amount={"count": "units", "side": "enemy", "plus": 1})])
    g = blank(r)
    for _ in range(3):
        put(g, r, 1, "back", "fill07")
    cast(g, r, "focus")
    assert g.observe(0).pending.amount == 4 and g.observe(0).pending.card == r.idx("focus")
    g.step(choose(2))
    assert [u.hp for u in g.backline[1]] == [4, 4]  # 4 damage killed the third 2/4


@pytest.mark.parametrize("eff, expected", [
    # (action, amount, select, side, kind, atk, hp); SPEC 5: amount/atk/hp are resolved when the choice
    # opens; atk/hp are the buff values (0 for other actions); amount is -1 for heal "full" and 0 where the
    # action has no amount; action/select/side/kind are the JSON names.
    (effect("on_play", "damage", tgt("chosen", "enemy", "unit_or_base"), amount=3),
     ("damage", 3, "chosen", "enemy", "unit_or_base", 0, 0)),
    (effect("on_play", "heal", tgt("chosen", "any"), amount="full"), ("heal", -1, "chosen", "any", "unit", 0, 0)),
    (effect("on_play", "heal", tgt("chosen", "friendly", "base"), amount={"count": "units", "side": "enemy"}),
     ("heal", 2, "chosen", "friendly", "base", 0, 0)),
    (effect("on_play", "buff", tgt("chosen", "any"), atk={"count": "units", "side": "enemy", "plus": 1}, hp=2),
     ("buff", 0, "chosen", "any", "unit", 3, 2)),
    (effect("on_play", "buff", tgt("chosen", "enemy"), atk=-2), ("buff", 0, "chosen", "enemy", "unit", -2, 0)),
    (effect("on_play", "destroy", tgt("chosen", "enemy")), ("destroy", 0, "chosen", "enemy", "unit", 0, 0)),
    (effect("on_play", "pin", tgt("chosen", "enemy")), ("pin", 0, "chosen", "enemy", "unit", 0, 0)),
])
def test_pending_view_fields(eff, expected):
    r = Rules([operation("order", [effect("on_play", "draw", "controller", amount=0), eff])])
    g = blank(r)
    put(g, r, 0, "back", "fill03")
    put(g, r, 1, "back", "fill03")
    put(g, r, 1, "back", "fill07")
    cast(g, r, "order")
    pv = g.observe(0).pending
    assert pv is not None and g.observe(1).pending == pv
    assert (pv.card, pv.effect) == (r.idx("order"), 1)
    assert tuple(pv)[2:9] == expected  # SPEC 5: previews (field 9) are checked in test_features.py


def test_operation_playability_requires_options():
    # SPEC 2.6 PLAY operation: legal only if each chosen effect whose condition holds now has an option.
    r = Rules([op("snipe", tgt("chosen", "enemy")),
               op("lastditch", tgt("chosen", "enemy"), condition={"type": "base_hp", "side": "friendly", "max": 10}),
               operation("combo", [effect("on_play", "damage", tgt("chosen", "enemy"), amount=1),
                                   effect("on_play", "buff", tgt("chosen", "friendly"), atk=1, hp=1)]),
               op("selfrepair", tgt("chosen", "friendly", "base"), action="heal", amount=2)])
    g = blank(r)
    set_hand(g, r, 0, ["snipe", "lastditch", "combo", "selfrepair"])

    def playable(cid):
        return play(hand_slot(g, r, 0, cid)) in g.legal_actions()

    assert not playable("snipe")
    assert playable("lastditch")        # condition false now -> no requirement
    assert not playable("combo")
    assert playable("selfrepair")       # a base is always an option
    g.base_hp[0] = 10
    g.invalidate()
    assert not playable("lastditch")    # condition holds now and there is no option
    put(g, r, 1, "back", "fill03")
    assert playable("snipe") and playable("lastditch")
    assert not playable("combo")        # the second chosen effect still has no option
    put(g, r, 0, "back", "fill03")
    assert playable("combo")


def test_operation_with_false_condition_is_played_and_skipped():
    r = Rules([op("lastditch", tgt("chosen", "enemy"), condition={"type": "base_hp", "side": "friendly", "max": 10})])
    g = blank(r)
    cast(g, r, "lastditch")
    assert g.phase == MAIN and counts(g.discard[0], r.n_cards)[r.idx("lastditch")] == 1


def test_smokescreen_units_are_choosable_and_defense_does_not_restrict_effects():
    # SPEC 2.4: effects (including chosen ones) can target smokescreen units; effects ignore Defense.
    r = Rules([unit("wall", 1, 4, traits={"defense": True}), unit("shade", 1, 4, traits={"smokescreen": True}),
               op("snipe", tgt("chosen", "enemy"))])
    g = blank(r)
    put(g, r, 1, "back", "wall")
    s = put(g, r, 1, "back", "shade")
    put(g, r, 1, "back", "fill03")
    cast(g, r, "snipe")
    assert chooses(g) == [0, 1, 2]
    g.step(choose(1))
    assert hp(g, s) == 3


def test_chosen_on_attack_resumes_the_attack_after_choose():
    # SPEC 2.6 ATTACK: 2. on_attack triggers resolve (here a choice), then 3./4. combat.
    r = Rules([unit("duelist", 2, 5, effects=[effect("on_attack", "damage", tgt("chosen", "enemy"), amount=1)])])
    g = blank(r)
    a = put(g, r, 0, "front", "duelist")
    t = put(g, r, 1, "back", "fill02")  # 3/3
    e = put(g, r, 1, "back", "fill01")  # 2/2
    g.step(attack(front(0), back(0)))
    assert g.phase == CHOICE and chooses(g) == [0, 1]
    assert find(g, a.uid).attacks == 1
    g.step(choose(1))
    assert hp(g, e) == 1
    assert hp(g, t) == 1 and hp(g, a) == 2  # the combat happened after the choice
    assert g.phase == MAIN


def test_chosen_on_attack_killing_the_target_ends_the_attack():
    r = Rules([unit("duelist", 2, 5, effects=[effect("on_attack", "damage", tgt("chosen", "enemy"), amount=1)])])
    g = blank(r)
    a = put(g, r, 0, "front", "duelist")
    t = put(g, r, 1, "back", "fill00")  # 1/1
    put(g, r, 1, "back", "fill01")
    g.step(attack(front(0), back(0)))
    g.step(choose(0))
    assert find(g, t.uid) is None and hp(g, a) == 5 and g.phase == MAIN  # SPEC 2.6 ATTACK 3


def test_chosen_on_move():
    # SPEC 1.2: chosen is allowed for on_move (self); the unit is already in the frontline.
    r = Rules([unit("pathfinder", 1, 3, effects=[effect("on_move", "damage", tgt("chosen", "enemy"), amount=2)])])
    g = blank(r)
    m = put(g, r, 0, "back", "pathfinder")
    e = put(g, r, 1, "back", "fill03")
    g.step(move(0))
    assert g.phase == CHOICE and zone_of(g, m.uid) == ("front", 0)
    assert chooses(g) == [0]
    g.step(choose(0))
    assert hp(g, e) == 2 and g.phase == MAIN


def test_chosen_heal_on_own_units_and_base():
    r = Rules([op("triage", tgt("chosen", "friendly", "unit_or_base"), action="heal", amount=3)])
    g = blank(r)
    u = put(g, r, 0, "back", "fill07", hp=1)
    g.base_hp[0] = 10
    g.invalidate()
    cast(g, r, "triage")
    assert chooses(g) == [own_back(0), OWN_BASE]
    c = g.clone()
    c.step(choose(OWN_BASE))
    assert bases_of(c) == [13, 20]
    g.step(choose(own_back(0)))
    assert hp(g, u) == 4 and bases_of(g) == [10, 20]


def test_no_mulligan_or_confirm_outside_the_mulligan_phase():
    r = grenadier_rules()
    g = blank(r)
    put(g, r, 1, "back", "fill03")
    assert all(a < CHOOSE0 for a in g.legal_actions())
    cast(g, r, "grenadier")
    assert all(CHOOSE0 <= a < MULLIGAN0 for a in g.legal_actions())
    with pytest.raises(IllegalActionError):
        g.step(CONFIRM)
