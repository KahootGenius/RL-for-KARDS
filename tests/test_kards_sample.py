"""KARDS sample (SPEC 1.5): research/kards_sample.json holds real KARDS cards, each encoded in the Stage 3 card
schema or marked unsupported with the missing primitives. Every encoding must parse with the strict loader,
follow the documented KARDS -> schema mapping and run in the engine; the stored coverage summary must match a
recomputation."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from cardgame.agents import RandomAgent, choose_action
from cardgame.cards import build_card_pool, build_ruleset, generate_deck
from cardgame.engine import Game

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "research" / "kards_sample.json"
with open(SAMPLE_PATH, encoding="utf-8") as _f:
    SAMPLE = json.load(_f)
ENTRIES = SAMPLE["cards"]
TOKENS = SAMPLE["tokens"]
VOCAB = SAMPLE["unsupported_vocabulary"]
SUPPORTED = [e for e in ENTRIES if "encoding" in e]
UNSUPPORTED = [e for e in ENTRIES if "unsupported" in e]
ENCODINGS = [e["encoding"] for e in SUPPORTED]

KINDS = ("infantry", "tank", "artillery", "fighter", "bomber", "order", "countermeasure")
KIND_NATURE = {"infantry": "troop", "tank": "fast", "artillery": "ranged", "fighter": "ranged", "bomber": "ranged"}
NATION_TAG = {"Britain": "britain", "USA": "usa", "Germany": "germany", "Soviet Union": "soviet", "Japan": "japan",
              "France": "france", "Poland": "poland", "Finland": "finland", "Italy": "italy", "ANZAC": "anzac"}
BOOL_KEYWORDS = {"Guard": "defense", "Blitz": "blitz", "Smokescreen": "smokescreen", "Ambush": "ambush",
                 "Fury": "fury", "Shock": "shock"}
# SPEC "Deferred to Stage 4": the unsupported vocabulary names these primitives
SPEC_DEFERRED = {"countermeasure", "cost_modifier", "delayed_or_granted_effect", "copy_or_random_card",
                 "transform_veteran", "deck_manipulation", "damage_modifier", "take_control_or_remove", "choose_one",
                 "set_stat", "rule_modifier", "intel_covert", "develop", "forecast"}


def recompute_coverage(entries) -> dict:
    """The coverage summary, recomputed from the entries (same definitions as the stored one)."""
    sup = [e for e in entries if "encoding" in e]
    uns = [e for e in entries if "unsupported" in e]

    def order(c: Counter) -> dict:
        return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))

    pats = Counter(p for e in uns for p in e["unsupported"])
    trig = Counter(t for e in sup for t in {f["trigger"] for f in e["encoding"]["effects"]})
    acts = Counter(a for e in sup for a in {f["action"] for f in e["encoding"]["effects"]})
    return {
        "entries": len(entries), "supported": len(sup), "unsupported": len(uns),
        "fraction_supported": round(len(sup) / len(entries), 3),
        "by_kind": {k: {"supported": sum(e["kind"] == k for e in sup), "unsupported": sum(e["kind"] == k for e in uns)}
                    for k in sorted({e["kind"] for e in entries})},
        "unsupported_patterns": order(pats),
        "unsupported_spec_deferred": sum(1 for e in uns if any(VOCAB[p]["spec_deferred"] for p in e["unsupported"])),
        "supported_triggers": order(trig),
        "supported_actions": order(acts),
        "supported_without_effects": sum(1 for e in sup if not e["encoding"]["effects"]),
    }


def sample_pool_config(**overrides):
    """A ruleset over every supported encoding plus the tokens; one filler deck (40 non-token cards)."""
    cards = ENCODINGS + TOKENS
    ids = [c["id"] for c in ENCODINGS]
    filler = {cid: 3 for cid in ids[:13]}
    filler[ids[13]] = 1
    return build_ruleset(cards, [{"name": "filler", "cards": filler}], **overrides)


# ---------------------------------------------------------------- shape of the sample
def test_sample_size_and_fields():
    assert len(ENTRIES) >= 60
    names = [e["name"] for e in ENTRIES]
    ids = [e["card_id"] for e in ENTRIES]
    assert len(set(names)) == len(names) and len(set(ids)) == len(ids)
    for e in ENTRIES:
        assert set(e) >= {"name", "card_id", "nation", "kind", "kredits", "operation_cost", "attack", "defense",
                          "keywords", "paraphrase", "source"}, e["name"]
        assert ("encoding" in e) != ("unsupported" in e), e["name"]
        assert e["kind"] in KINDS and e["nation"] in NATION_TAG, e["name"]
        assert type(e["kredits"]) is int and e["kredits"] >= 0 and isinstance(e["keywords"], list), e["name"]
        unit = e["kind"] not in ("order", "countermeasure")
        for k in ("operation_cost", "attack", "defense"):
            assert (type(e[k]) is int) if unit else (e[k] is None), (e["name"], k)
        assert isinstance(e["paraphrase"], str) and len(e["paraphrase"]) >= 15, e["name"]
        assert isinstance(e["source"], str) and e["source"].startswith("https://"), e["name"]


def test_sample_is_spread_out():
    """A spread of kinds, nations, costs and effect patterns, including cards the schema cannot express."""
    kinds = Counter(e["kind"] for e in ENTRIES)
    assert set(kinds) == set(KINDS) and min(kinds.values()) >= 3, kinds
    assert len({e["nation"] for e in ENTRIES}) >= 8
    assert len({e["kredits"] for e in ENTRIES}) >= 8
    assert len(UNSUPPORTED) >= 15 and len(SUPPORTED) >= 40
    triggers = {f["trigger"] for c in ENCODINGS for f in c["effects"]}
    assert len(triggers) >= 8, triggers


def test_unsupported_entries_name_known_primitives():
    assert SPEC_DEFERRED <= set(VOCAB)
    assert all(VOCAB[k]["spec_deferred"] == (k in SPEC_DEFERRED) for k in VOCAB)
    assert all(isinstance(v["description"], str) and v["description"] for v in VOCAB.values())
    for e in UNSUPPORTED:
        pats = e["unsupported"]
        assert isinstance(pats, list) and pats and len(set(pats)) == len(pats), e["name"]
        assert set(pats) <= set(VOCAB), (e["name"], pats)
    assert all("unsupported" in e and "countermeasure" in e["unsupported"]
               for e in ENTRIES if e["kind"] == "countermeasure")
    used = {p for e in UNSUPPORTED for p in e["unsupported"]}
    assert used == set(VOCAB), f"vocabulary entries no card uses: {sorted(set(VOCAB) - used)}"


# ---------------------------------------------------------------- encodings
@pytest.mark.parametrize("entry", SUPPORTED, ids=[e["card_id"] for e in SUPPORTED])
def test_each_encoding_parses(entry):
    """Each encoding alone (with the tokens) passes the strict card loader."""
    pool = build_card_pool([entry["encoding"]] + TOKENS)
    assert pool[0].id == entry["card_id"]


def test_the_whole_sample_builds_one_ruleset():
    cfg = sample_pool_config()
    assert len(cfg.cards) == len(ENCODINGS) + len(TOKENS)
    assert sum(c.token for c in cfg.cards.cards) == len(TOKENS)
    assert cfg.cards.has_effects


def test_encodings_follow_the_mapping():
    """kredits -> cost, attack, defense -> health, operation cost -> move_cost; KARDS kind -> nature (+ tags);
    Guard -> defense, Heavy Armor X -> armor X, combat keywords -> traits."""
    for e in SUPPORTED:
        c = e["encoding"]
        assert c["id"] == e["card_id"] and c["name"] == e["name"] and c["cost"] == e["kredits"], e["name"]
        tags = c["tags"]
        assert e["kind"] in tags and NATION_TAG[e["nation"]] in tags, e["name"]
        assert ("air" in tags) == (e["kind"] in ("fighter", "bomber")), e["name"]
        if e["kind"] == "order":
            assert c["type"] == "operation", e["name"]
            continue
        assert c["type"] == "unit", e["name"]
        assert c["nature"] == KIND_NATURE[e["kind"]] or (
            e["kind"] == "infantry" and c["nature"] == "fast" and "notes" in e), e["name"]
        assert (c["attack"], c["health"], c["move_cost"]) == (e["attack"], e["defense"], e["operation_cost"]), e["name"]
        want = {}
        for k in e["keywords"]:
            if k in BOOL_KEYWORDS:
                want[BOOL_KEYWORDS[k]] = True
            elif k.startswith("Heavy Armor "):
                want["armor"] = int(k.split()[-1])
            else:
                assert k.startswith("Exile") and "exile" in tags, (e["name"], k)
        assert c["traits"] == want, e["name"]


def test_tokens_are_listed_and_referenced():
    token_ids = {t["id"] for t in TOKENS}
    assert token_ids and all(t.get("token") is True for t in TOKENS)
    assert set(SAMPLE["token_sources"]) == token_ids
    named = set()
    for c in ENCODINGS:
        for f in c["effects"]:
            for body in (f, f.get("else", {})):
                if body.get("action") in ("summon", "add_card"):
                    assert body["card"] in token_ids, c["id"]
                    named.add(body["card"])
    assert named == token_ids


def test_coverage_summary_matches_a_recomputation():
    assert SAMPLE["coverage"] == recompute_coverage(ENTRIES)
    cov = SAMPLE["coverage"]
    assert cov["supported"] + cov["unsupported"] == cov["entries"] == len(ENTRIES)


def test_sample_encodings_run_in_the_engine():
    """Random self-play on random decks drawn from the sample pool: the real-card encodings resolve without
    errors and without tripping the loop guard."""
    cfg = sample_pool_config()
    game = Game(cfg)
    agents = (RandomAgent(cfg, seed=1), RandomAgent(cfg, seed=2))
    played = Counter()
    for k in range(24):
        decks = (generate_deck(random.Random(k), cfg), generate_deck(random.Random(10_000 + k), cfg))
        game.reset(k, decks=decks)
        while not game.done:
            game.step(choose_action(agents[game.current_player()], game))
        assert game.guard_trips == 0
        for p in (0, 1):
            played.update({c: n for c, n in enumerate(game.played[p]) if n})
    assert len(played) >= len(ENCODINGS) // 2
