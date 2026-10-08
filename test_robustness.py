"""API robustness: pre-reset queries, input validation, clone generality, invariant checks, strict loading."""
from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import CONFIG, NUM_ACTIONS, add_unit, blank_game, card, new_game, set_hand
from cardgame.cards import load_card_pool, load_ruleset
from cardgame.engine import Game, IllegalActionError, Unit


# ---------------------------------------------------------------- before reset()
def test_never_reset_game_is_safely_over():
    g = Game(CONFIG)
    assert g.done and g.winner() is None
    assert g.legal_actions() == []
    assert not g.legal_mask().any()
    assert not g.is_legal(0)
    with pytest.raises(IllegalActionError):
        g.step(0)
    with pytest.raises(RuntimeError):
        g.observe(0)
    assert "reset" in g.render()


def test_clone_of_never_reset_game_behaves_the_same_and_can_start():
    c = Game(CONFIG).clone()
    assert c.done and c.legal_actions() == []
    c.reset(3)
    assert c.legal_actions() == new_game(3).legal_actions()


# ---------------------------------------------------------------- seeds and action inputs
def test_numpy_integer_seed_is_the_same_deal():
    a, b = Game(CONFIG), Game(CONFIG)
    a.reset(7)
    b.reset(np.int64(7))
    assert a.render() == b.render()


@pytest.mark.parametrize("bad", [None, 1.0, "3"])
def test_non_integer_seed_is_rejected(bad):
    with pytest.raises(TypeError):
        Game(CONFIG).reset(bad)


def test_negative_seed_is_rejected():
    # random.Random(-s) == random.Random(s), so allowing it would silently replay deals
    with pytest.raises(ValueError):
        Game(CONFIG).reset(-1)


@pytest.mark.parametrize("bad", [True, False, 0.0, 1.0, "0", None, np.float32(0)])
def test_non_integer_actions_raise_illegal_action(bad):
    g = new_game(0)
    before = g.render()
    with pytest.raises(IllegalActionError):
        g.step(bad)
    assert g.render() == before


def test_legal_mask_writes_into_given_array():
    g = new_game(5)
    out = np.ones(NUM_ACTIONS, dtype=bool)
    assert g.legal_mask(out=out) is out
    assert np.array_equal(out, g.legal_mask())
    assert sorted(np.flatnonzero(out).tolist()) == g.legal_actions()


# ---------------------------------------------------------------- clone generality
def test_clone_deep_copies_state_added_later():
    g = new_game(2)
    g.graveyard = [[card("squire")], []]  # e.g. a later stage adds a graveyard
    c = g.clone()
    c.graveyard[0].append(card("giant"))
    assert g.graveyard == [[card("squire")], []]


def test_clone_preserves_subclass():
    class Variant(Game):
        pass

    g = Variant(CONFIG)
    g.reset(1)
    assert type(g.clone()) is Variant


def test_unit_copy_covers_every_slot():
    u = Unit(3, 4, 5, 1, True)
    c = u.copy()
    for name in Unit.__slots__:
        assert getattr(c, name) == getattr(u, name), name
    assert set(Unit.__slots__) == {"card", "atk", "hp", "owner", "ready"}, "update Unit.copy()"


# ---------------------------------------------------------------- invariant checks on direct edits
def test_invalidate_rejects_oversized_hand():
    g = blank_game()
    g.hands[0] = [card("squire")] * (CONFIG.max_hand_size + 1)
    with pytest.raises(ValueError):
        g.invalidate()


def test_invalidate_rejects_overfull_zone():
    g = blank_game()
    for _ in range(CONFIG.zone_capacity):
        add_unit(g, 0, "back", "squire")
    g.backline[0].append(Unit(card("squire"), 1, 2, 0, True))
    with pytest.raises(ValueError):
        g.invalidate()


def test_invalidate_rejects_inconsistent_frontline():
    g = blank_game()
    add_unit(g, 0, "front", "squire")
    g.frontline.append(Unit(card("scout"), 2, 1, 1, True))  # enemy unit in a held frontline
    with pytest.raises(ValueError):
        g.invalidate()
    g = blank_game()
    g.front_owner = 1  # owner set with an empty frontline
    with pytest.raises(ValueError):
        g.invalidate()


# ---------------------------------------------------------------- strict card / deck loading
GOOD_CARD = {"id": "c", "name": "C", "type": "unit", "cost": 1, "attack": 1, "health": 1, "traits": [], "effects": []}


def write_cards(tmp_path, *cards):
    path = tmp_path / "cards.json"
    path.write_text(json.dumps({"cards": list(cards)}))
    return path


def test_good_card_loads(tmp_path):
    pool = load_card_pool(write_cards(tmp_path, GOOD_CARD))
    assert pool[0].cost == 1 and pool[0].index == 0


@pytest.mark.parametrize("change", [
    {"on_death": {"type": "draw"}},       # field the engine cannot run yet
    {"trait": ["taunt"]},                 # misspelled field
    {"traits": ["taunt"]},                # unsupported trait
    {"effects": [{"type": "on_play_damage"}]},
    {"type": "spell"},
    {"cost": 1.5}, {"attack": True}, {"health": "3"},
    {"health": 0}, {"cost": -1},
    {"traits": "taunt"},
])
def test_bad_card_is_rejected(tmp_path, change):
    with pytest.raises(ValueError):
        load_card_pool(write_cards(tmp_path, {**GOOD_CARD, **change}))


def test_missing_field_and_duplicate_ids_are_rejected(tmp_path):
    no_cost = {k: v for k, v in GOOD_CARD.items() if k != "cost"}
    with pytest.raises(ValueError):
        load_card_pool(write_cards(tmp_path, no_cost))
    with pytest.raises(ValueError):
        load_card_pool(write_cards(tmp_path, GOOD_CARD, GOOD_CARD))


@pytest.mark.parametrize("counts", [{"c": 41}, {"c": 39}, {"c": 40, "d": 0}, {"c": 42, "d": -2}, {"c": 40.0}])
def test_bad_decks_are_rejected(tmp_path, counts):
    cards = write_cards(tmp_path, GOOD_CARD, {**GOOD_CARD, "id": "d"})
    decks = tmp_path / "decks.json"
    decks.write_text(json.dumps({"decks": [{"name": "a", "cards": counts}, {"name": "b", "cards": {"c": 40}}]}))
    with pytest.raises(ValueError):
        load_ruleset(cards, decks)


def test_unknown_card_in_deck_is_rejected(tmp_path):
    cards = write_cards(tmp_path, GOOD_CARD)
    decks = tmp_path / "decks.json"
    decks.write_text(json.dumps({"decks": [{"cards": {"zzz": 40}}, {"cards": {"c": 40}}]}))
    with pytest.raises(KeyError):
        load_ruleset(cards, decks)


def test_bundled_ruleset_is_valid():
    cfg = load_ruleset()
    assert len(cfg.cards) == 10 and all(len(d) == 40 for d in cfg.decks)
    assert all(c.type == "unit" and not c.traits and not c.effects for c in cfg.cards.cards)


# ---------------------------------------------------------------- final-review regressions
def test_bool_seed_is_rejected():
    with pytest.raises(TypeError):
        Game(CONFIG).reset(True)


def test_is_legal_and_describe_handle_odd_inputs():
    g = new_game(0)
    assert g.is_legal(True) is False and g.is_legal(1.0) is False and g.is_legal("0") is False
    assert g.describe(np.int64(0)) == "END_TURN"
    with pytest.raises(ValueError):
        g.observe(2)
    with pytest.raises(TypeError):
        g.observe(True)


def test_invalidate_drops_stale_legal_actions():
    g = blank_game(coins=5)
    set_hand(g, 0, ["knight"])
    assert g.legal_actions() == [0, 1]  # END_TURN, PLAY(0)
    g.hands[0] = []
    g.invalidate()
    assert g.legal_actions() == [0] and not g.is_legal(1)


def test_rejected_edit_does_not_leave_a_stale_cache():
    g = blank_game(coins=5)
    set_hand(g, 0, ["knight"])
    assert g.legal_actions() == [0, 1]
    g.hands[0] = [card("squire")] * (CONFIG.max_hand_size + 1)
    with pytest.raises(ValueError):
        g.invalidate()
    g.hands[0] = []
    assert g.legal_actions() == [0]  # recomputed, not the pre-edit cache


def test_reset_clears_burned_on_a_reused_game():
    g = new_game(3)
    while not g.done and sum(g.burned) == 0:
        g.step(0)  # END_TURN only: hands fill up and cards burn
    assert sum(g.burned) > 0
    g.reset(3)
    assert g.burned == [0, 0] and g.render() == new_game(3).render()


def test_coin_cap_is_applied():
    import dataclasses
    g = Game(dataclasses.replace(CONFIG, coin_cap=3))
    g.reset(0)
    for _ in range(12):
        g.step(0)
    assert g.round >= 6 and g.coins[g.current] == 3


def test_clone_does_not_share_tuple_held_state():
    g = new_game(4)
    g.hands = tuple(g.hands)  # a tool may store documented state as tuples
    g.queue = ("on_death", [1, 2])  # later-stage state: tuple holding a list
    c = g.clone()
    c.hands[g.current].clear()
    c.queue[1].append(3)
    assert g.hands[g.current] and g.queue == ("on_death", [1, 2])


def test_clone_keeps_references_to_board_units_aliased():
    g = new_game(5)
    g.backline[0].append(Unit(card("knight"), 4, 3, 0, True))
    g.selected = g.backline[0][-1]  # e.g. a pending-target reference in a later stage
    g.invalidate()
    c = g.clone()
    assert c.selected is c.backline[0][-1] and c.selected is not g.selected


def test_unit_subclass_slots_survive_copy():
    class TraitUnit(Unit):
        __slots__ = ("traits",)

    u = TraitUnit(1, 2, 3, 0, True)
    u.traits = ["taunt"]
    c = u.copy()
    assert type(c) is TraitUnit and c.traits == ["taunt"] and c.traits is not u.traits and c.hp == 3


@pytest.mark.parametrize("override", [{"base_hp": 0}, {"max_rounds": 0}, {"zone_capacity": 0},
                                      {"max_hand_size": -1}, {"opening_hand": (-3, 5)}, {"coin_cap": -2}])
def test_invalid_game_config_is_rejected(override):
    with pytest.raises(ValueError):
        load_ruleset(**override)


@pytest.mark.parametrize("change", [{"id": None}, {"id": 5}, {"id": ""}, {"name": None}, {"traits": [[1]]},
                                    {"effects": [1]}])
def test_more_bad_cards_are_rejected(tmp_path, change):
    with pytest.raises(ValueError):
        load_card_pool(write_cards(tmp_path, {**GOOD_CARD, **change}))


def test_duplicate_json_keys_and_unknown_top_level_keys_are_rejected(tmp_path):
    path = tmp_path / "cards.json"
    path.write_text('{"cards": [{"id": "c", "type": "unit", "cost": 1, "cost": 9, "attack": 1, "health": 1}]}')
    with pytest.raises(ValueError):
        load_card_pool(path)
    path.write_text(json.dumps({"cards": [GOOD_CARD], "spells": []}))
    with pytest.raises(ValueError):
        load_card_pool(path)


@pytest.mark.parametrize("decks", [
    {"decks": [{"cards": ["c"] * 40}, {"cards": {"c": 40}}]},           # cards as a list
    {"decks": [{"card": {"c": 40}}, {"cards": {"c": 40}}]},             # misspelled key
    {"decks": [{"cards": {"c": 40}, "hero": "x"}, {"cards": {"c": 40}}]},  # unknown deck key
    {"decks": {"a": {"cards": {"c": 40}}}},                             # decks as an object
])
def test_malformed_decks_are_rejected(tmp_path, decks):
    cards = write_cards(tmp_path, GOOD_CARD)
    path = tmp_path / "decks.json"
    path.write_text(json.dumps(decks))
    with pytest.raises(ValueError):
        load_ruleset(cards, path)
