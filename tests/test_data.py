"""Shipped content (SPEC §1.1, §1.4): the bundled card pool and the four decks have the promised shape.
Thresholds come from the SPEC; card numbers are read from the data, never hard-coded."""
from __future__ import annotations

import dataclasses
import json
from collections import Counter
from itertools import product

import pytest

from cardgame.cards import (ACTIONS, DEFAULT_CARDS, DEFAULT_DECKS, FAST, KINDS, NATURES, RANGED, SELECTS, SIDES,
                            TRIGGERS, TROOP, TURN_TRIGGERS, load_ruleset, sample_decks)
from conftest import CONFIG, FIXTURES, VANILLA_CONFIG

POOL = CONFIG.cards.cards
N_STAGE2 = 25
COMBAT_KEYWORDS = ("blitz", "smokescreen", "fury", "ambush", "shock", "immune")


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


def unit_fraction(i: int, pred) -> float:
    """Fraction of deck i's unit cards satisfying `pred` (SPEC §1.4: archetype rules count units)."""
    units = [c for c in deck_cards(i) if c.is_unit]
    return sum(map(pred, units)) / len(units)


def bodies(card):
    """Every parsed effect body of a card: the effects and their `else` bodies."""
    for e in card.effects:
        yield e
        if e.else_ is not None:
            yield e.else_


def amounts(e):
    return [a for a in (e.amount, e.atk, e.hp) if a is not None]


def filters(e):
    """Every unit filter an effect body uses (target, target_condition, conditions, amounts)."""
    out = [e.target.filter, e.target_condition]
    out += [c.filter for c in e.condition]
    out += [a.filter for a in amounts(e)]
    return [f for f in out if f is not None]


EFFECT_CARDS = [c for c in POOL if c.effects]
ALL_BODIES = [(c, e) for c in POOL for e in bodies(c)]


# ---------------------------------------------------------------- loader defaults (SPEC §1)
def test_default_ruleset_numbers():
    cfg = load_ruleset()
    assert (cfg.base_hp, cfg.max_rounds, cfg.zone_capacity, cfg.max_hand_size) == (20, 50, 5, 10)
    assert (tuple(cfg.opening_hand), cfg.deck_size, cfg.max_copies, cfg.coin_cap) == ((4, 5), 40, 3, None)
    assert (cfg.mulligan, cfg.max_effect_events) == (True, 256)  # SPEC §1.4 loader defaults
    assert dataclasses.replace(cfg, mulligan=False) == CONFIG and not CONFIG.mulligan


def test_stage2_fixture_is_the_vanilla_content():
    """tests/fixtures/stage2_*.json freeze the Stage 2 content for the reference-model tests: 25 effect-free
    units, 4 decks; the shipped pool keeps those 25 units unchanged (SPEC §1.4)."""
    raw = raw_json(f"{FIXTURES}/stage2_cards.json")["cards"]
    assert len(raw) == len(VANILLA_CONFIG.cards) == 25 and VANILLA_CONFIG.n_decks == 4
    assert not VANILLA_CONFIG.mulligan and not VANILLA_CONFIG.cards.has_effects
    assert all(c.is_unit and not c.token for c in VANILLA_CONFIG.cards.cards)
    for c in VANILLA_CONFIG.cards.cards:
        shipped = CONFIG.cards.by_id(c.id)
        assert dataclasses.replace(shipped, index=c.index) == c, c.id


# ---------------------------------------------------------------- cards: format and pool shape
def test_stage2_units_come_first_unchanged():
    """The 25 Stage 2 units keep their ids, stats, file order and card indices (so Stage 2 positions and the
    reference model keep their meaning)."""
    fixture = raw_json(f"{FIXTURES}/stage2_cards.json")["cards"]
    assert len(fixture) == N_STAGE2
    assert RAW_CARDS[:N_STAGE2] == fixture
    assert [c.id for c in POOL[:N_STAGE2]] == [c.id for c in VANILLA_CONFIG.cards.cards]


def test_card_file_is_in_the_documented_format():
    assert [c.id for c in POOL] == [raw["id"] for raw in RAW_CARDS]  # indexed by file order
    assert len({c.id for c in POOL}) == len(POOL)
    for i, (c, raw) in enumerate(zip(POOL, RAW_CARDS)):
        assert c.index == i
        assert raw["type"] in ("unit", "operation") and isinstance(raw["effects"], list), c.id
        assert type(raw["cost"]) is int and isinstance(raw["name"], str) and raw["name"], c.id
        assert len(raw["effects"]) <= 3 and (c.is_unit or raw["effects"]), c.id
        if c.is_unit:
            assert raw["nature"] in NATURES and isinstance(raw["traits"], dict), c.id
            assert all(type(raw[k]) is int for k in ("attack", "health")), c.id
            assert c.health >= 1 and c.attack >= 0 and c.move_cost >= 0, c.id
        else:
            assert not {"nature", "attack", "health", "move_cost", "traits"} & set(raw), c.id
            assert all(e.trigger == "on_play" and e.scope == "self" for e in c.effects), c.id


def test_pool_size_operations_and_tokens():
    """SPEC §1.4: 44-56 cards, >= 8 operations, >= 1 token; costs stay within 1..8."""
    assert 44 <= len(POOL) <= 56
    assert sum(c.is_operation for c in POOL) >= 8
    tokens = [c for c in POOL if c.token]
    assert tokens
    costs = {c.cost for c in POOL}
    assert set(range(1, 9)) <= costs, f"missing costs {set(range(1, 9)) - costs}"
    assert costs <= set(range(1, 9)), "costs stay within 1..8 (brief)"


def test_tokens_are_created_by_effects_and_never_dealt():
    """Every token is named by some summon / add_card (a unit token for summon, any token for add_card); every
    summon / add_card names a token; no deck holds a token."""
    named = Counter()
    for c, e in ALL_BODIES:
        if e.action in ("summon", "add_card"):
            ref = POOL[e.card]
            assert ref.token and ref.id == e.card_id, (c.id, e.card_id)
            assert ref.is_unit or e.action == "add_card", (c.id, e.card_id)
            named[(e.action, ref.index)] += 1
    assert any(a == "summon" and POOL[i].is_unit for a, i in named), "a unit token for summon"
    assert any(a == "add_card" for a, _ in named), "a token for add_card"
    for t in (c for c in POOL if c.token):
        assert any(i == t.index for _, i in named), f"token {t.id} is never created"
    for d in CONFIG.decks:
        assert not any(POOL[c].token for c in d)


def test_every_nature_and_trait_combination_occurs():
    units = [c for c in POOL if c.is_unit and not c.token]
    seen = Counter((c.nature, trait_set(c)) for c in units)
    combos = list(product((TROOP, FAST, RANGED), ("none", "defense", "armor", "defense+armor")))
    missing = [(NATURES[n], t) for n, t in combos if not seen[(n, t)]]
    assert not missing, f"missing nature/trait combinations {missing}"
    assert len(combos) == 12


@pytest.mark.parametrize("keyword", COMBAT_KEYWORDS)
def test_every_combat_keyword_is_on_some_card(keyword):
    """SPEC §1.4: some card has each of blitz, smokescreen, fury, ambush, shock, immune (as a card trait)."""
    assert any(c.is_unit and getattr(c, keyword) for c in POOL), keyword


def test_move_costs_vary():
    assert any(c.move_cost == 0 for c in POOL if c.is_unit)
    assert any(c.move_cost >= 2 for c in POOL if c.is_unit)


# ---------------------------------------------------------------- effect coverage (SPEC §1.4)
@pytest.mark.parametrize("trigger", TRIGGERS)
def test_every_trigger_occurs(trigger):
    assert any(e.trigger == trigger for _, e in ALL_BODIES), trigger


@pytest.mark.parametrize("action", ACTIONS)
def test_every_action_occurs(action):
    assert any(e.action == action for _, e in ALL_BODIES), action


@pytest.mark.parametrize("select", SELECTS)
def test_every_select_occurs(select):
    assert any(e.target.select == select for _, e in ALL_BODIES), select


@pytest.mark.parametrize("side", SIDES)
def test_every_target_side_occurs(side):
    assert any(e.target.side == side for _, e in ALL_BODIES), side


@pytest.mark.parametrize("kind", KINDS)
def test_every_target_kind_occurs(kind):
    assert any(e.target.kind == kind for _, e in ALL_BODIES), kind


def test_effect_feature_coverage():
    """A filter, an amount expression, a condition, a "turn" duration, a non-default scope, an event filter,
    else + target_condition, repeat and tags (with tag filters) all occur in the shipped pool."""
    assert any(e.target.filter is not None for _, e in ALL_BODIES), "a target filter"
    assert any(a.kind in ("stat", "count", "event") for _, e in ALL_BODIES for a in amounts(e)), "an amount expression"
    assert any(e.condition for _, e in ALL_BODIES), "a condition"
    assert any(e.duration == "turn" for _, e in ALL_BODIES), 'a "turn" duration'
    assert any(e.scope != ("friendly" if e.trigger in TURN_TRIGGERS else "self") for _, e in ALL_BODIES), \
        "a non-default scope"
    assert any(e.event_filter is not None for _, e in ALL_BODIES), "an event_filter"
    assert any(e.else_ is not None for _, e in ALL_BODIES), "an else body"
    assert any(e.target_condition is not None for _, e in ALL_BODIES), "a target_condition"
    assert any(e.repeat > 1 for _, e in ALL_BODIES), "repeat"
    assert sum(bool(c.tags) for c in POOL) >= len(POOL) - N_STAGE2, "every new card carries tags"
    tag_users = {c.id for c, e in ALL_BODIES
                 for f in filters(e) + ([e.event_filter] if e.event_filter is not None else [])
                 if f.tags or f.not_tags}
    assert len(tag_users) >= 2, f"tag filters on a few cards, got {sorted(tag_users)}"
    vocab = {t for c in POOL for t in c.tags}
    for c, e in ALL_BODIES:  # tag filters name tags that exist in the pool
        for f in filters(e) + ([e.event_filter] if e.event_filter is not None else []):
            assert set(f.tags) | set(f.not_tags) <= vocab, (c.id, f)


def test_new_cards_have_effects_or_new_keywords():
    """Every card added in Stage 3 brings an effect or a Stage 3 keyword (no new vanilla filler)."""
    for c in POOL[N_STAGE2:]:
        assert c.effects or any(getattr(c, k, False) for k in COMBAT_KEYWORDS) or c.token, c.id


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
        assert not any(POOL[c].token for c in CONFIG.decks[i])
    assert len(set(CONFIG.decks)) == 4 and len(set(CONFIG.deck_names)) == 4
    assert CONFIG.deck_names == ("Blitz", "Bulwark", "Volley", "Legion")


@pytest.mark.parametrize("i, keyword", [(0, "aggro"), (1, "defens"), (2, "ranged"), (3, "balanced")])
def test_deck_order_and_styles(i, keyword):
    assert keyword in CONFIG.deck_styles[i].lower()


def test_aggro_deck_is_mostly_fast():
    assert unit_fraction(0, lambda c: c.nature == FAST) >= 0.5


def test_defensive_deck_has_many_defense_units():
    assert unit_fraction(1, lambda c: c.defense) >= 0.4


def test_ranged_deck_is_ranged_heavy():
    assert unit_fraction(2, lambda c: c.nature == RANGED) >= 0.4


def test_balanced_deck_mixes_every_nature():
    for nature in (TROOP, FAST, RANGED):
        assert unit_fraction(3, lambda c: c.nature == nature) >= 0.2, NATURES[nature]


@pytest.mark.parametrize("i", range(4))
def test_each_deck_has_effect_cards(i):
    """SPEC §1.4: >= 6 effect cards (operations or units with effects) per fixed deck, counted as distinct
    cards; every deck also plays operations."""
    distinct = {c.id for c in deck_cards(i) if c.effects}
    assert len(distinct) >= 6, (CONFIG.deck_names[i], sorted(distinct))
    assert any(c.is_operation for c in deck_cards(i)), CONFIG.deck_names[i]


def test_every_new_non_token_card_is_in_a_fixed_deck():
    """The fixed decks show off the whole Stage 3 pool (random decks draw from it anyway)."""
    in_decks = {c for d in CONFIG.decks for c in d}
    missing = [c.id for c in POOL[N_STAGE2:] if not c.token and c.index not in in_decks]
    assert not missing, missing


def test_each_game_samples_two_decks():
    pairs = Counter(sample_decks(seed, CONFIG.n_decks) for seed in range(1600))
    assert len(pairs) == CONFIG.n_decks ** 2
