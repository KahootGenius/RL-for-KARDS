"""API robustness: pre-reset queries, input validation, invariant checks on direct edits and strict
loading of the Stage 2 data format (natures, traits object, move_cost, max_copies, N decks)."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from cardgame.cards import (DEFAULT_MOVE_COST, FAST, RANGED, TROOP, GameConfig, load_card_pool, load_ruleset,
                            sample_decks)
from cardgame.engine import Game, IllegalActionError, Unit
from conftest import CONFIG, END, NUM_ACTIONS, add_unit, blank_game, check_invariants, new_game, set_hand


# ---------------------------------------------------------------- before reset()
def test_never_reset_game_is_safely_over():
    g = Game(CONFIG)
    assert g.done and g.winner() is None
    assert g.legal_actions() == []
    assert not g.legal_mask().any() and g.legal_mask().shape == (NUM_ACTIONS,)
    assert not g.is_legal(0)
    with pytest.raises(IllegalActionError):
        g.step(0)
    with pytest.raises(RuntimeError):
        g.observe(0)
    assert "reset" in g.render()


def test_clone_of_never_reset_game_behaves_the_same_and_can_start():
    c = Game(CONFIG).clone()
    assert c.done and c.legal_actions() == [] and c.winner() is None
    c.reset(3)
    assert c.legal_actions() == new_game(3).legal_actions()


def test_default_config_is_the_bundled_ruleset():
    g = Game()
    g.reset(0)
    assert g.config == load_ruleset() and g.num_actions == NUM_ACTIONS
    assert g.config.mulligan and dataclasses.replace(g.config, mulligan=False) == CONFIG


# ---------------------------------------------------------------- seeds, decks and action inputs
def test_numpy_integer_seed_is_the_same_deal():
    a, b = Game(CONFIG), Game(CONFIG)
    a.reset(7)
    b.reset(np.int64(7))
    assert a.render() == b.render() and a.rng.getstate() == b.rng.getstate()
    assert tuple(a.deck_ids) == tuple(b.deck_ids) == sample_decks(7, CONFIG.n_decks)


@pytest.mark.parametrize("bad", [None, 1.0, "3", True, False, np.float64(2.0), np.bool_(True)])
def test_non_integer_seed_is_rejected(bad):
    with pytest.raises(TypeError):
        Game(CONFIG).reset(bad)


def test_negative_seed_is_rejected():
    # random.Random(-s) == random.Random(s), so allowing it would silently replay deals
    with pytest.raises(ValueError):
        Game(CONFIG).reset(-1)


@pytest.mark.parametrize("decks", [(0, CONFIG.n_decks), (-1, 0), (0,), (0, 1, 2), (True, 0), (0, 1.0),
                                   ("0", 1), 3])
def test_bad_deck_pairs_are_rejected(decks):
    with pytest.raises((TypeError, ValueError)):
        Game(CONFIG).reset(0, decks)


def test_numpy_deck_indices_are_accepted():
    a, b = new_game(4, (1, 2)), new_game(4, (np.int64(1), np.int8(2)))
    assert a.render() == b.render() and all(type(d) is int for d in b.deck_ids)


@pytest.mark.parametrize("bad", [True, False, 0.0, 1.0, "0", None, np.float32(0), np.bool_(False), [0], (0,)])
def test_non_integer_actions_raise_illegal_action(bad):
    g = new_game(0)
    before = g.render()
    with pytest.raises(IllegalActionError):
        g.step(bad)
    assert g.render() == before


def test_is_legal_describe_and_observe_handle_odd_inputs():
    g = new_game(0)
    assert g.is_legal(True) is False and g.is_legal(1.0) is False and g.is_legal("0") is False
    assert g.is_legal(-1) is False and g.is_legal(NUM_ACTIONS) is False and g.is_legal(np.int64(0)) is True
    assert g.describe(np.int64(0)) == "END_TURN"
    assert "range" in g.describe(NUM_ACTIONS) and "not an action" in g.describe("x")
    with pytest.raises(ValueError):
        g.observe(2)
    with pytest.raises(TypeError):
        g.observe(True)
    assert g.observe(np.int64(1)) == g.observe(1)


def test_illegal_action_error_is_a_value_error():
    assert issubclass(IllegalActionError, ValueError)


# ---------------------------------------------------------------- invariant checks on direct edits
def test_invalidate_rejects_oversized_hand():
    g = blank_game()
    g.hands[0] = [0] * (CONFIG.max_hand_size + 1)
    with pytest.raises(ValueError):
        g.invalidate()


@pytest.mark.parametrize("zone", ["back", "front"])
def test_invalidate_rejects_overfull_zones(zone):
    g = blank_game()
    for _ in range(CONFIG.zone_capacity):
        add_unit(g, 0, zone, atk=1, hp=1)
    extra = Unit(0, 0, atk=1, hp=1, uid=99)
    (g.backline[0] if zone == "back" else g.frontline).append(extra)
    with pytest.raises(ValueError):
        g.invalidate()


def test_invalidate_rejects_inconsistent_zones():
    g = blank_game()
    add_unit(g, 0, "front", atk=1, hp=1)
    g.frontline.append(Unit(0, 1, atk=1, hp=1, uid=50))  # enemy unit in a held frontline
    with pytest.raises(ValueError):
        g.invalidate()
    g = blank_game()
    g.front_owner = 1  # owner set with an empty frontline
    with pytest.raises(ValueError):
        g.invalidate()
    g = blank_game()
    g.frontline.append(Unit(0, 1, atk=1, hp=1))  # units but no owner
    with pytest.raises(ValueError):
        g.invalidate()
    g = blank_game()
    g.backline[0].append(Unit(0, 1, atk=1, hp=1))  # enemy unit in my backline
    with pytest.raises(ValueError):
        g.invalidate()


def test_invalidate_drops_stale_legal_actions():
    g = blank_game(coins=5)
    set_hand(g, 0, [0])
    assert g.legal_actions() == [END, 1]  # END_TURN, PLAY(0)
    g.hands[0] = []
    g.invalidate()
    assert g.legal_actions() == [END] and not g.is_legal(1)


def test_rejected_edit_does_not_leave_a_stale_cache():
    g = blank_game(coins=5)
    set_hand(g, 0, [0])
    assert g.legal_actions() == [END, 1]
    g.hands[0] = [0] * (CONFIG.max_hand_size + 1)
    with pytest.raises(ValueError):
        g.invalidate()
    g.hands[0] = []
    assert g.legal_actions() == [END]  # recomputed, not the pre-edit cache


def test_reset_clears_burned_played_and_uids_on_a_reused_game():
    g = new_game(3)
    while not g.done and sum(g.burned) == 0:
        g.step(END)  # END_TURN only: hands fill up and cards burn
    assert sum(g.burned) > 0
    g.reset(3)
    assert g.burned == [0, 0] and g.render() == new_game(3).render()
    g = blank_game(coins=9)
    set_hand(g, 0, [0])
    g.step(1)
    assert g.next_uid == 1 and sum(g.played[0]) == 1
    g.reset(3)
    assert g.next_uid == 0 and not any(g.played[0]) and g.backline == [[], []]
    check_invariants(g, (0, 0))


# ---------------------------------------------------------------- strict card / deck loading
GOOD_CARD = {"id": "c", "name": "C", "type": "unit", "nature": "troop", "cost": 1, "attack": 1, "health": 1,
             "move_cost": 1, "traits": {}, "effects": []}


def write_cards(tmp_path, *cards):
    path = tmp_path / "cards.json"
    path.write_text(json.dumps({"cards": list(cards)}), encoding="utf-8")
    return path


def write_decks(tmp_path, *decks, raw=None):
    path = tmp_path / "decks.json"
    path.write_text(json.dumps(raw if raw is not None else {"decks": list(decks)}), encoding="utf-8")
    return path


def deck_of(counts: dict, **extra) -> dict:
    return {"name": "d", "style": "s", "cards": counts, **extra}


def pool14(tmp_path, **overrides):
    """A pool with 14 cards (c0..c13): enough for legal 40-card decks with max 3 copies."""
    return write_cards(tmp_path, *[{**GOOD_CARD, "id": f"c{i}", **overrides} for i in range(14)])


LEGAL_COUNTS = {f"c{i}": 3 for i in range(13)} | {"c13": 1}


@pytest.mark.parametrize("change, expected", [
    ({}, dict(nature=TROOP, move_cost=1, defense=False, armor=0)),
    ({"nature": "fast"}, dict(nature=FAST)),
    ({"nature": "ranged", "move_cost": 0}, dict(nature=RANGED, move_cost=0)),
    ({"move_cost": 3}, dict(move_cost=3)),
    ({"traits": {"defense": True}}, dict(defense=True, armor=0)),
    ({"traits": {"defense": False}}, dict(defense=False)),
    ({"traits": {"armor": 2}}, dict(armor=2, defense=False)),
    ({"traits": {"defense": True, "armor": 1}}, dict(defense=True, armor=1)),
    ({"cost": 0, "attack": 0}, dict(cost=0, attack=0)),
])
def test_good_cards_load(tmp_path, change, expected):
    pool = load_card_pool(write_cards(tmp_path, {**GOOD_CARD, **change}))
    c = pool[0]
    assert c.index == 0 and c.id == "c" and c.effects == ()
    for k, v in expected.items():
        assert getattr(c, k) == v and type(getattr(c, k)) is type(v), k


def test_move_cost_defaults_to_one(tmp_path):
    raw = {k: v for k, v in GOOD_CARD.items() if k != "move_cost"}
    assert load_card_pool(write_cards(tmp_path, raw))[0].move_cost == DEFAULT_MOVE_COST == 1


@pytest.mark.parametrize("change", [
    {"on_death": {"type": "draw"}},            # field the engine cannot run yet
    {"trait": {"defense": True}},              # misspelled field
    {"nature": "flying"}, {"nature": "Troop"}, {"nature": 0}, {"nature": None},
    {"traits": {"taunt": True}},               # unsupported trait
    {"traits": {"defense": 1}}, {"traits": {"defense": "yes"}}, {"traits": {"defense": None}},
    {"traits": {"armor": 0}}, {"traits": {"armor": -1}}, {"traits": {"armor": True}},
    {"traits": {"armor": 1.5}}, {"traits": {"armor": "2"}},
    {"traits": ["defense"]}, {"traits": "defense"}, {"traits": None},
    {"effects": [{"type": "on_play_damage"}]}, {"effects": ["charge"]}, {"effects": {}}, {"effects": [1]},
    {"type": "spell"}, {"type": "operation"},      # an operation has no unit stats
    {"traits": {"blitz": 1}}, {"traits": {"fury": "yes"}}, {"traits": {"smokescreen": None}},
    {"token": 1}, {"token": "yes"},
    {"cost": 1.5}, {"attack": True}, {"health": "3"}, {"move_cost": 1.0}, {"move_cost": True},
    {"move_cost": -1}, {"move_cost": None},
    {"health": 0}, {"cost": -1}, {"attack": -2},
    {"id": None}, {"id": 5}, {"id": ""}, {"name": None},
])
def test_bad_card_is_rejected(tmp_path, change):
    with pytest.raises(ValueError):
        load_card_pool(write_cards(tmp_path, {**GOOD_CARD, **change}))


@pytest.mark.parametrize("missing", ["id", "type", "nature", "cost", "attack", "health"])
def test_missing_required_field_is_rejected(tmp_path, missing):
    raw = {k: v for k, v in GOOD_CARD.items() if k != missing}
    with pytest.raises(ValueError):
        load_card_pool(write_cards(tmp_path, raw))


def test_duplicate_ids_empty_pools_and_bad_top_level_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_card_pool(write_cards(tmp_path, GOOD_CARD, GOOD_CARD))
    with pytest.raises(ValueError):
        load_card_pool(write_cards(tmp_path))
    path = tmp_path / "cards.json"
    path.write_text(json.dumps({"cards": [GOOD_CARD], "spells": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_card_pool(path)
    path.write_text(json.dumps([GOOD_CARD]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_card_pool(path)


@pytest.mark.parametrize("text", [
    '{"cards": [{"id": "c", "type": "unit", "nature": "troop", "cost": 1, "cost": 9, "attack": 1, "health": 1}]}',
    '{"cards": [{"id": "c", "type": "unit", "nature": "troop", "cost": 1, "attack": 1, "health": 1,'
    ' "traits": {"armor": 1, "armor": 2}}]}',
    '{"cards": [], "cards": []}',
])
def test_duplicate_json_keys_are_rejected(tmp_path, text):
    path = tmp_path / "cards.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_card_pool(path)


def test_ruleset_with_n_decks_loads(tmp_path):
    cards = pool14(tmp_path)
    for n in (1, 2, 5):
        decks = write_decks(tmp_path, *[deck_of(LEGAL_COUNTS, name=f"d{i}", style=f"s{i}") for i in range(n)])
        cfg = load_ruleset(cards, decks, mulligan=False)
        assert cfg.n_decks == n == len(cfg.decks) == len(cfg.deck_names) == len(cfg.deck_styles)
        assert cfg.deck_names == tuple(f"d{i}" for i in range(n))
        assert all(d == tuple(sorted(d)) and len(d) == 40 for d in cfg.decks)
        g = Game(cfg)
        for seed in range(5):
            g.reset(seed)
            assert all(0 <= d < n for d in g.deck_ids)
            check_invariants(g, (0, 0))
        if n == 1:
            assert tuple(g.deck_ids) == (0, 0)  # the single deck plays the mirror


def test_deck_name_and_style_are_optional(tmp_path):
    cards = pool14(tmp_path)
    cfg = load_ruleset(cards, write_decks(tmp_path, {"cards": LEGAL_COUNTS}, {"cards": LEGAL_COUNTS}))
    assert cfg.deck_names == ("deck0", "deck1") and len(cfg.deck_styles) == 2


@pytest.mark.parametrize("counts", [
    {f"c{i}": 3 for i in range(13)},                       # 39 cards
    LEGAL_COUNTS | {"c13": 2},                             # 41 cards
    {f"c{i}": 3 for i in range(12)} | {"c12": 4},          # 4 copies > max_copies
    LEGAL_COUNTS | {"c13": 0},
    LEGAL_COUNTS | {"c13": -1},
    LEGAL_COUNTS | {"c13": 1.0},
    LEGAL_COUNTS | {"c13": True},
    LEGAL_COUNTS | {"c13": "1"},
])
def test_bad_deck_counts_are_rejected(tmp_path, counts):
    cards = pool14(tmp_path)
    with pytest.raises(ValueError):
        load_ruleset(cards, write_decks(tmp_path, deck_of(counts), deck_of(LEGAL_COUNTS)))


def test_max_copies_override(tmp_path):
    cards = pool14(tmp_path)
    counts = {f"c{i}": 4 for i in range(10)}
    with pytest.raises(ValueError):
        load_ruleset(cards, write_decks(tmp_path, deck_of(counts)))
    cfg = load_ruleset(cards, write_decks(tmp_path, deck_of(counts)), max_copies=4)
    assert cfg.max_copies == 4 and len(cfg.decks[0]) == 40


def test_unknown_card_in_deck_is_rejected(tmp_path):
    cards = pool14(tmp_path)
    with pytest.raises((KeyError, ValueError)):
        load_ruleset(cards, write_decks(tmp_path, deck_of(LEGAL_COUNTS | {"zzz": 1}), deck_of(LEGAL_COUNTS)))


@pytest.mark.parametrize("raw", [
    {"decks": [{"cards": [f"c{i}" for i in range(13)] * 3 + ["c13"]}]},  # cards as a list
    {"decks": [{"card": LEGAL_COUNTS}]},                                  # misspelled key
    {"decks": [{"cards": LEGAL_COUNTS, "hero": "x"}]},                    # unknown deck key
    {"decks": {"a": {"cards": LEGAL_COUNTS}}},                            # decks as an object
    {"decks": []},                                                        # no decks
    {"decks": [LEGAL_COUNTS]},                                            # counts without "cards"
    {"decks": [{"cards": LEGAL_COUNTS}], "extra": 1},                     # unknown top-level key
])
def test_malformed_decks_are_rejected(tmp_path, raw):
    cards = pool14(tmp_path)
    with pytest.raises(ValueError):
        load_ruleset(cards, write_decks(tmp_path, raw=raw))


@pytest.mark.parametrize("override", [{"base_hp": 0}, {"max_rounds": 0}, {"zone_capacity": 0},
                                      {"max_hand_size": -1}, {"opening_hand": (-3, 5)}, {"opening_hand": (4,)},
                                      {"coin_cap": -2}, {"coin_cap": 2.5}, {"max_copies": 0}, {"deck_size": 0},
                                      {"base_hp": True}, {"max_rounds": 10.0}, {"mulligan": 1},
                                      {"mulligan": None}, {"max_effect_events": 0}, {"max_effect_events": 2.0},
                                      {"max_effect_events": True}])
def test_invalid_game_config_is_rejected(override):
    with pytest.raises(ValueError):
        load_ruleset(**override)


def test_game_config_needs_decks_and_matching_names():
    with pytest.raises(ValueError):
        GameConfig(cards=CONFIG.cards, decks=())
    with pytest.raises(ValueError):
        dataclasses.replace(CONFIG, deck_names=("only one",))


def test_config_overrides_reach_the_engine():
    cfg = load_ruleset(base_hp=7, coin_cap=2, max_rounds=3, mulligan=False)
    g = Game(cfg)
    g.reset(0)
    assert g.base_hp == [7, 7]
    for _ in range(5):
        g.step(END)
    assert not g.done and g.round == 3 and g.coins[g.current] == 2
    g.step(END)
    assert g.done and g.winner() == -1 and g.round == 3
