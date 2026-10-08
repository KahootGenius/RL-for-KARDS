"""Card definitions and ruleset loading. Cards are data; rules live in the engine."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_CARDS = DATA_DIR / "cards.json"
DEFAULT_DECKS = DATA_DIR / "decks.json"

# Card features the Stage 1 engine knows how to run. Later stages extend these sets
# together with the engine code that implements them.
SUPPORTED_TYPES = frozenset({"unit"})
SUPPORTED_TRAITS: frozenset = frozenset()
SUPPORTED_EFFECTS: frozenset = frozenset()


@dataclass(frozen=True)
class CardDef:
    index: int
    id: str
    name: str
    type: str
    cost: int
    attack: int
    health: int
    traits: tuple = ()
    effects: tuple = ()


@dataclass(frozen=True)
class CardPool:
    cards: tuple

    def __len__(self) -> int:
        return len(self.cards)

    def __getitem__(self, index: int) -> CardDef:
        return self.cards[index]

    def by_id(self, card_id: str) -> CardDef:
        for c in self.cards:
            if c.id == card_id:
                return c
        raise KeyError(f"unknown card id {card_id!r}")

    @property
    def max_cost(self) -> int:
        return max(c.cost for c in self.cards)

    @property
    def max_attack(self) -> int:
        return max(c.attack for c in self.cards)

    @property
    def max_health(self) -> int:
        return max(c.health for c in self.cards)


@dataclass(frozen=True)
class GameConfig:
    cards: CardPool
    decks: tuple  # (seat-0 card indices, seat-1 card indices)
    deck_names: tuple = ("deck0", "deck1")
    base_hp: int = 20
    max_rounds: int = 50
    zone_capacity: int = 5
    max_hand_size: int = 10
    opening_hand: tuple = (4, 5)  # (first player, second player)
    deck_size: int = 40
    coin_cap: Optional[int] = None

    def __post_init__(self):
        for name in ("base_hp", "max_rounds", "zone_capacity", "max_hand_size", "deck_size"):
            v = getattr(self, name)
            if type(v) is not int or v < 1:
                raise ValueError(f"{name} must be a positive integer, got {v!r}")
        if len(self.opening_hand) != 2 or any(type(n) is not int or n < 0 for n in self.opening_hand):
            raise ValueError(f"opening_hand must be two non-negative integers, got {self.opening_hand!r}")
        if self.coin_cap is not None and (type(self.coin_cap) is not int or self.coin_cap < 0):
            raise ValueError(f"coin_cap must be None or a non-negative integer, got {self.coin_cap!r}")
        if len(self.decks) != 2:
            raise ValueError("exactly two decks are required")

    def coins_for_round(self, rnd: int) -> int:
        return rnd if self.coin_cap is None else min(rnd, self.coin_cap)


CARD_KEYS = frozenset({"id", "name", "type", "cost", "attack", "health", "traits", "effects"})


def _stat(card_id: str, raw: dict, key: str) -> int:
    v = raw[key]
    if type(v) is not int:  # no bools, floats or numeric strings
        raise ValueError(f"card {card_id!r}: {key} must be an integer, got {v!r}")
    return v


def _no_duplicate_keys(pairs: list) -> dict:
    keys = [k for k, _ in pairs]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise ValueError(f"duplicate JSON keys {dup}")
    return dict(pairs)


def _load_json(path: Path | str, top_keys: frozenset) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f, object_pairs_hook=_no_duplicate_keys)
    if not isinstance(raw, dict) or set(raw) != top_keys:
        raise ValueError(f"{path}: expected an object with exactly the keys {sorted(top_keys)}")
    return raw


def parse_card(index: int, raw: dict) -> CardDef:
    if not isinstance(raw, dict):
        raise ValueError(f"card #{index} must be an object")
    for key in ("id", "type", "cost", "attack", "health"):
        if key not in raw:
            raise ValueError(f"card #{index} is missing {key!r}")
    card_id = raw["id"]
    if not isinstance(card_id, str) or not card_id:
        raise ValueError(f"card #{index}: id must be a non-empty string, got {card_id!r}")
    if not isinstance(raw.get("name", ""), str) or not isinstance(raw["type"], str):
        raise ValueError(f"card {card_id!r}: name and type must be strings")
    extra = set(raw) - CARD_KEYS
    if extra:  # e.g. a misspelled "trait" or an "on_death" the engine cannot run yet
        raise ValueError(f"card {card_id!r}: unknown fields {sorted(extra)}")
    traits, effects = raw.get("traits", []), raw.get("effects", [])
    if not isinstance(traits, list) or not isinstance(effects, list):
        raise ValueError(f"card {card_id!r}: traits and effects must be lists")
    if not all(isinstance(t, str) for t in traits) or not all(isinstance(e, (str, dict)) for e in effects):
        raise ValueError(f"card {card_id!r}: traits must be strings and effects strings or objects")
    card = CardDef(
        index=index,
        id=card_id,
        name=raw.get("name", card_id),
        type=raw["type"],
        cost=_stat(card_id, raw, "cost"),
        attack=_stat(card_id, raw, "attack"),
        health=_stat(card_id, raw, "health"),
        traits=tuple(traits),
        effects=tuple(effects),
    )
    if card.type not in SUPPORTED_TYPES:
        raise ValueError(f"card {card.id!r}: unsupported type {card.type!r}")
    unknown = (set(card.traits) - SUPPORTED_TRAITS) | (
        {e.get("type") if isinstance(e, dict) else e for e in card.effects} - SUPPORTED_EFFECTS)
    if unknown:
        raise ValueError(f"card {card.id!r}: unsupported traits/effects {sorted(map(str, unknown))}")
    if card.cost < 0 or card.attack < 0 or card.health <= 0:
        raise ValueError(f"card {card.id!r}: invalid stats")
    return card


def load_card_pool(path: Path | str = DEFAULT_CARDS) -> CardPool:
    raw = _load_json(path, frozenset({"cards"}))
    if not isinstance(raw["cards"], list) or not raw["cards"]:
        raise ValueError(f"{path}: 'cards' must be a non-empty list")
    cards = tuple(parse_card(i, c) for i, c in enumerate(raw["cards"]))
    ids = [c.id for c in cards]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate card ids")
    return CardPool(cards)


def build_deck(pool: CardPool, counts: dict) -> tuple:
    if not isinstance(counts, dict):
        raise ValueError("deck 'cards' must be an object mapping card id -> count")
    deck = []
    for card_id, n in counts.items():
        if type(n) is not int or n < 1:
            raise ValueError(f"deck count for {card_id!r} must be a positive integer, got {n!r}")
        deck.extend([pool.by_id(card_id).index] * n)
    return tuple(sorted(deck))


def load_ruleset(cards_path: Path | str | None = None, decks_path: Path | str | None = None,
                 **overrides) -> GameConfig:
    pool = load_card_pool(cards_path or DEFAULT_CARDS)
    raw = _load_json(decks_path or DEFAULT_DECKS, frozenset({"decks"}))
    decks_raw: Sequence[dict] = raw["decks"]
    if not isinstance(decks_raw, list) or len(decks_raw) != 2:
        raise ValueError("'decks' must be a list of exactly two decks")
    for d in decks_raw:
        if not isinstance(d, dict) or "cards" not in d or set(d) - {"name", "cards"}:
            raise ValueError(f"each deck must be an object with 'cards' (and optional 'name'), got {d!r}")
    decks = tuple(build_deck(pool, d["cards"]) for d in decks_raw)
    names = tuple(d.get("name", f"deck{i}") for i, d in enumerate(decks_raw))
    config = GameConfig(cards=pool, decks=decks, deck_names=names, **overrides)
    for name, deck in zip(names, decks):
        if len(deck) != config.deck_size:
            raise ValueError(f"deck {name!r} has {len(deck)} cards, expected {config.deck_size}")
    return config
