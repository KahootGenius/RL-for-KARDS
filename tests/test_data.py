"""Shipped content (SPEC §1): the bundled card pool and the four decks have the promised shape.
Thresholds come from the SPEC; card numbers are read from the data, never hard-coded."""
from __future__ import annotations

import json
from collections import Counter
from itertools import product

import pytest

from cardgame.cards import DEFAULT_CARDS, DEFAULT_DECKS, FAST, NATURES, RANGED, TROOP, load_ruleset, sample_decks
from conftest import CONFIG

POOL = CONFIG.cards.cards


def raw_json(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


RAW_CARDS = raw_json(DEFAULT_CARDS)["cards"]
RAW_DECKS = raw_json(DEFAULT_DECKS)["decks"]


def trait_set(c) -> str:
    return {(False, False): "none", (True, False): "defense", (False, True): "armor",
            (True, True): "defense+armor"}[(c.defense, c.armor > 0)]


def deck_cards(i: int) -> list:
    return [POOL[c] for c in CONFIG.decks[i]]


def fraction(i: int, pred) -> float:
    cards = deck_cards(i)
    return sum(map(pred, cards)) / len(cards)


# ---------------------------------------------------------------- loader defaults (SPEC §1)
def test_default_ruleset_numbers():
    cfg = load_ruleset()
    assert (cfg.base_hp, cfg.max_rounds, cfg.zone_capacity, cfg.max_hand_size) == (20, 50, 5, 10)
    assert (tuple(cfg.opening_hand), cfg.deck_size, cfg.max_copies, cfg.coin_cap) == ((4, 5), 40, 3, None)
    assert cfg == CONFIG


# ---------------------------------------------------------------- cards
def test_card_file_is_in_the_documented_format():
    assert [c.id for c in POOL] == [raw["id"] for raw in RAW_CARDS]  # indexed by file order
    for i, (c, raw) in enumerate(zip(POOL, RAW_CARDS)):
        assert c.index == i
        assert raw["type"] == "unit" and raw["nature"] in NATURES and raw["effects"] == []
        assert isinstance(raw["traits"], dict) and set(raw["traits"]) <= {"defense", "armor"}
        assert all(type(raw[k]) is int for k in ("cost", "attack", "health"))
        assert c.health >= 1 and c.attack >= 0 and c.move_cost >= 0
    assert len({c.id for c in POOL}) == len(POOL)


def test_pool_size_and_costs():
    assert 22 <= len(POOL) <= 28
    costs = {c.cost for c in POOL}
    assert set(range(1, 9)) <= costs, f"missing costs {set(range(1, 9)) - costs}"
    assert costs <= set(range(1, 9)), "costs stay within 1..8 (brief)"


def test_every_nature_and_trait_combination_occurs():
    seen = Counter((c.nature, trait_set(c)) for c in POOL)
    combos = list(product((TROOP, FAST, RANGED), ("none", "defense", "armor", "defense+armor")))
    missing = [(NATURES[n], t) for n, t in combos if not seen[(n, t)]]
    assert not missing, f"missing nature/trait combinations {missing}"
    assert len(combos) == 12


def test_move_costs_vary():
    assert any(c.move_cost == 0 for c in POOL)
    assert any(c.move_cost >= 2 for c in POOL)


# ---------------------------------------------------------------- decks
def test_four_legal_decks():
    assert CONFIG.n_decks == len(RAW_DECKS) == 4
    for i, raw in enumerate(RAW_DECKS):
        assert set(raw) == {"name", "style", "cards"}
        assert all(type(n) is int and 1 <= n <= CONFIG.max_copies for n in raw["cards"].values()), raw["name"]
        assert sum(raw["cards"].values()) == CONFIG.deck_size == len(CONFIG.decks[i]) == 40
        assert CONFIG.decks[i] == tuple(sorted(CONFIG.decks[i]))
        assert max(Counter(CONFIG.decks[i]).values()) <= 3
        assert CONFIG.deck_names[i] == raw["name"] and CONFIG.deck_styles[i] == raw["style"]
    assert len(set(CONFIG.decks)) == 4 and len(set(CONFIG.deck_names)) == 4


@pytest.mark.parametrize("i, keyword", [(0, "aggro"), (1, "defens"), (2, "ranged"), (3, "balanced")])
def test_deck_order_and_styles(i, keyword):
    assert keyword in CONFIG.deck_styles[i].lower()


def test_aggro_deck_is_mostly_fast():
    assert fraction(0, lambda c: c.nature == FAST) >= 0.5


def test_defensive_deck_has_many_defense_units():
    assert fraction(1, lambda c: c.defense) >= 0.4


def test_ranged_deck_is_ranged_heavy():
    assert fraction(2, lambda c: c.nature == RANGED) >= 0.4


def test_balanced_deck_mixes_every_nature():
    for nature in (TROOP, FAST, RANGED):
        assert fraction(3, lambda c: c.nature == nature) >= 0.2, NATURES[nature]


def test_each_game_samples_two_decks():
    pairs = Counter(sample_decks(seed, CONFIG.n_decks) for seed in range(1600))
    assert len(pairs) == CONFIG.n_decks ** 2
