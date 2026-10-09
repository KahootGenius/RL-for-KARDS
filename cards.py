"""Card definitions and ruleset loading. Cards are data; rules live in the engine."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_CARDS = DATA_DIR / "cards.json"
DEFAULT_DECKS = DATA_DIR / "decks.json"

# Card features the engine knows how to run. Later stages extend these together with the
# engine code that implements them.
SUPPORTED_TYPES = frozenset({"unit"})
NATURES = ("troop", "fast", "ranged")  # index = nature code used by the engine
TROOP, FAST, RANGED = 0, 1, 2
SUPPORTED_TRAITS = frozenset({"defense", "armor"})
SUPPORTED_EFFECTS: frozenset = frozenset()
DEFAULT_MOVE_COST = 1


@dataclass(frozen=True)
class CardDef:
    index: int
    id: str
    name: str
    type: str
    cost: int
    attack: int
    health: int
    nature: int = TROOP
    move_cost: int = DEFAULT_MOVE_COST
    defense: bool = False
    armor: int = 0
    effects: tuple = ()

    @property
    def nature_name(self) -> str:
        return NATURES[self.nature]

    @property
    def traits(self) -> dict:
        out = {}
        if self.defense:
            out["defense"] = True
        if self.armor:
            out["armor"] = self.armor
        return out


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

    @property
    def max_move_cost(self) -> int:
        return max(c.move_cost for c in self.cards)

    @property
    def max_armor(self) -> int:
        return max(c.armor for c in self.cards)


@dataclass(frozen=True)
class GameConfig:
    cards: CardPool
    decks: tuple  # one sorted tuple of card indices per deck; each game uses two of them
    deck_names: tuple = ()
    deck_styles: tuple = ()
    base_hp: int = 20
    max_rounds: int = 50
    zone_capacity: int = 5
    max_hand_size: int = 10
    opening_hand: tuple = (4, 5)  # (first player, second player)
    deck_size: int = 40
    max_copies: int = 3
    coin_cap: Optional[int] = None

    def __post_init__(self):
        for name in ("base_hp", "max_rounds", "zone_capacity", "max_hand_size", "deck_size", "max_copies"):
            v = getattr(self, name)
            if type(v) is not int or v < 1:
                raise ValueError(f"{name} must be a positive integer, got {v!r}")
        if len(self.opening_hand) != 2 or any(type(n) is not int or n < 0 for n in self.opening_hand):
            raise ValueError(f"opening_hand must be two non-negative integers, got {self.opening_hand!r}")
        if self.coin_cap is not None and (type(self.coin_cap) is not int or self.coin_cap < 0):
            raise ValueError(f"coin_cap must be None or a non-negative integer, got {self.coin_cap!r}")
        if len(self.decks) < 1:
            raise ValueError("at least one deck is required")
        if not self.deck_names:
            object.__setattr__(self, "deck_names", tuple(f"deck{i}" for i in range(len(self.decks))))
        if not self.deck_styles:
            object.__setattr__(self, "deck_styles", ("",) * len(self.decks))
        if len(self.deck_names) != len(self.decks) or len(self.deck_styles) != len(self.decks):
            raise ValueError("deck_names/deck_styles must match the number of decks")

    @property
    def n_decks(self) -> int:
        return len(self.decks)

    def coins_for_round(self, rnd: int) -> int:
        return rnd if self.coin_cap is None else min(rnd, self.coin_cap)


CARD_KEYS = frozenset({"id", "name", "type", "nature", "cost", "attack", "health", "move_cost", "traits",
                       "effects"})


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
    for key in ("id", "type", "nature", "cost", "attack", "health"):
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
    traits, effects = raw.get("traits", {}), raw.get("effects", [])
    if not isinstance(traits, dict) or not isinstance(effects, list):
        raise ValueError(f"card {card_id!r}: traits must be an object and effects a list")
    if not all(isinstance(e, (str, dict)) for e in effects):
        raise ValueError(f"card {card_id!r}: effects must be strings or objects")
    unknown = (set(traits) - SUPPORTED_TRAITS) | (
        {e.get("type") if isinstance(e, dict) else e for e in effects} - SUPPORTED_EFFECTS)
    if unknown:
        raise ValueError(f"card {card_id!r}: unsupported traits/effects {sorted(map(str, unknown))}")
    defense = traits.get("defense", False)
    if type(defense) is not bool:
        raise ValueError(f"card {card_id!r}: trait 'defense' must be true or false, got {defense!r}")
    armor = traits.get("armor", 0)
    if type(armor) is not int or armor < 0 or ("armor" in traits and armor < 1):
        raise ValueError(f"card {card_id!r}: trait 'armor' must be a positive integer, got {armor!r}")
    nature = raw["nature"]
    if nature not in NATURES:
        raise ValueError(f"card {card_id!r}: nature must be one of {NATURES}, got {nature!r}")
    card = CardDef(
        index=index,
        id=card_id,
        name=raw.get("name", card_id),
        type=raw["type"],
        cost=_stat(card_id, raw, "cost"),
        attack=_stat(card_id, raw, "attack"),
        health=_stat(card_id, raw, "health"),
        nature=NATURES.index(nature),
        move_cost=_stat(card_id, raw, "move_cost") if "move_cost" in raw else DEFAULT_MOVE_COST,
        defense=defense,
        armor=armor,
        effects=tuple(effects),
    )
    if card.type not in SUPPORTED_TYPES:
        raise ValueError(f"card {card.id!r}: unsupported type {card.type!r}")
    if card.cost < 0 or card.attack < 0 or card.health <= 0 or card.move_cost < 0:
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


def sample_decks(seed: int, n_decks: int) -> tuple:
    """Deck pair for a deal seed, from its own stream (independent of the game RNG; platform-stable)."""
    r = random.Random(f"decks:{seed}")
    return (r.randrange(n_decks), r.randrange(n_decks))


def build_deck(pool: CardPool, counts: dict, max_copies: Optional[int] = None) -> tuple:
    if not isinstance(counts, dict):
        raise ValueError("deck 'cards' must be an object mapping card id -> count")
    deck = []
    for card_id, n in counts.items():
        if type(n) is not int or n < 1:
            raise ValueError(f"deck count for {card_id!r} must be a positive integer, got {n!r}")
        if max_copies is not None and n > max_copies:
            raise ValueError(f"deck has {n} copies of {card_id!r}; at most {max_copies} allowed")
        try:
            deck.extend([pool.by_id(card_id).index] * n)
        except KeyError:
            raise ValueError(f"deck references unknown card id {card_id!r}") from None
    return tuple(sorted(deck))


def load_ruleset(cards_path: Path | str | None = None, decks_path: Path | str | None = None,
                 **overrides) -> GameConfig:
    pool = load_card_pool(cards_path or DEFAULT_CARDS)
    raw = _load_json(decks_path or DEFAULT_DECKS, frozenset({"decks"}))
    decks_raw: Sequence[dict] = raw["decks"]
    if not isinstance(decks_raw, list) or not decks_raw:
        raise ValueError("'decks' must be a non-empty list")
    for d in decks_raw:
        if not isinstance(d, dict) or "cards" not in d or set(d) - {"name", "style", "cards"}:
            raise ValueError(f"each deck must be an object with 'cards' (and optional 'name', 'style'), got {d!r}")
        if not all(isinstance(d.get(k, ""), str) for k in ("name", "style")):
            raise ValueError(f"deck name and style must be strings, got {d!r}")
    max_copies = overrides.get("max_copies", GameConfig.__dataclass_fields__["max_copies"].default)
    decks = tuple(build_deck(pool, d["cards"], max_copies) for d in decks_raw)
    names = tuple(d.get("name", f"deck{i}") for i, d in enumerate(decks_raw))
    styles = tuple(d.get("style", "") for d in decks_raw)
    config = GameConfig(cards=pool, decks=decks, deck_names=names, deck_styles=styles, **overrides)
    for name, deck in zip(names, decks):
        if len(deck) != config.deck_size:
            raise ValueError(f"deck {name!r} has {len(deck)} cards, expected {config.deck_size}")
    return config
