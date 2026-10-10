"""ObservationEncoder v5 (SPEC §6): token layout, hand-built positions read back by feature name,
mask hints and previews against the mask, engine.combat_damage and PendingView.previews, the §5
observation fields (history, pin_turns, "turn" changes), no-leak, strictness, fingerprint.

Expected values are written from SPEC §5/§6 (fixed scales, token order, mask-derived hints), not
read from features.py; the feature *names* come from its schema tuples.
"""
from __future__ import annotations

import dataclasses
import json
import random
from collections import Counter, namedtuple

import numpy as np
import pytest

import cardgame.features as fmod
from cardgame.actions import ActionSpace
from cardgame.cards import FAST, RANGED, TRAIT_BITS, TROOP, CardDef, FilterDef, build_ruleset, load_ruleset
from cardgame.engine import CHOICE, MAIN, MULLIGAN, Game, Observation, PendingView, Unit, UnitView, combat_damage
from cardgame.features import (ATTACK_PREVIEW_FEATURES, CARD_FEATURES, CARD_TABLE_FEATURES, CHOOSE_PREVIEW_FEATURES,
                               EFFECT_FEATURES, ENCODER_VERSION, GLOBAL_FEATURES, TOKEN_FEATURES, ObservationEncoder,
                               card_table, pool_fingerprint)
from conftest import CONFIG, add_unit, blank_game, set_hand

CONFIG_M = load_ruleset()  # shipped content with the mulligan (the training config)
ENC = ObservationEncoder(CONFIG)
LAY = ENC.layout()
H, Z = CONFIG.max_hand_size, CONFIG.zone_capacity
SP = ActionSpace(H, Z)
N = SP.n
N_CARDS = len(CONFIG.cards)
TF = {k: i for i, k in enumerate(TOKEN_FEATURES)}
GF = {k: i for i, k in enumerate(GLOBAL_FEATURES)}
NEW_UNIT_FIELDS = ("ambush", "shock", "immune", "ambush_ready")  # SPEC §5 UnitView additions (§1.2b)
TRAIT_NAMES = ("defense", "armor", "blitz", "smokescreen", "fury", "ambush", "shock", "immune")  # SPEC §2.4, §1.2b
HISTORY_NAMES = tuple(f"{who}_{e}_{w}" for who in ("my", "opp") for w in ("turn", "game")
                      for e in ("ops", "deployed", "died"))  # my_history / opp_history order (SPEC §5)

# SPEC §6 token order (T = 50 for H = 10, Z = 5, R = 20)
G_TOK, HAND0, MYB0, FRONT0, OPPB0, MYBASE, OPPBASE, PEND, REV0, DECK = (0, 1, 11, 16, 21, 26, 27, 28, 29, 49)
R = 20


def need(cond: bool, what: str):
    if not cond:
        pytest.skip(f"engine does not provide {what} yet (SPEC §1.2b / §5)")


class Read:
    """Read an encoding back by name."""

    def __init__(self, x, enc: ObservationEncoder = ENC):
        self.x = x
        self.p = enc.split(x)

    def f(self, tok: int, name: str) -> float:
        return float(self.p.features[tok, TF[name]])

    def g(self, name: str) -> float:
        return float(self.p.globals[GF[name]])

    def row_is(self, tok: int, expected: dict, ident: int = 0, present: bool = True) -> None:
        want = np.zeros(len(TOKEN_FEATURES))
        for k, v in expected.items():
            want[TF[k]] = float(v)
        got = self.p.features[tok]
        bad = {TOKEN_FEATURES[i]: (float(got[i]), want[i]) for i in np.flatnonzero(np.abs(got - want) > 1e-6)}
        assert not bad, f"token {tok}: (got, want) {bad}"
        assert int(self.p.ids[tok]) == ident, (tok, int(self.p.ids[tok]), ident)
        assert bool(self.p.present[tok]) == present, tok


def unit_expect(u: UnitView) -> dict:
    """SPEC §6 unit-state features of a UnitView (fixed scales)."""
    d = {"atk": u.atk / 10, "hp": u.hp / 10, "max_hp": u.max_hp / 10, "armor": u.armor / 4, "defense": u.defense,
         "blitz": u.blitz, "smokescreen": u.smokescreen, "fury": u.fury, "pinned": u.pinned, "summoned": u.summoned,
         "moved": u.moved, "attacks": u.attacks / 2, "can_move": u.can_move, "can_attack": u.can_attack,
         "temp_atk": u.temp_atk / 10, "temp_hp": u.temp_hp / 10, "move_cost": u.move_cost / 4,
         "damaged": u.hp < u.max_hp, "token": u.token, "pin_turns": u.pin_turns / 2,
         "temp_move_cost": u.temp_move_cost / 4}
    for name in NEW_UNIT_FIELDS:
        d[name] = getattr(u, name, False)
    for name in TRAIT_NAMES:  # signed "lapses this turn" flag: +1 granted, -1 removed for the turn
        bit = TRAIT_BITS[name]
        d[f"lapse_{name}"] = float(bool(u.temp_traits & bit)) - float(bool(u.temp_removed & bit))
    return d


def attack_grid(mask):
    return mask[SP.ATTACK0:SP.CHOOSE0].reshape(2 * Z, 2 * Z + 1)


def attackers_of(obs) -> list:
    """Unit in each attacker slot (SPEC §3): own backline 0..Z-1, frontline Z..2Z-1 when held."""
    slots = list(obs.my_backline) + [None] * (Z - len(obs.my_backline))
    slots += list(obs.frontline) if obs.front_owner > 0 else []
    return slots + [None] * (2 * Z - len(slots))


def targets_of(obs) -> list:
    """Unit in each attack target slot (enemy backline, enemy frontline), then None for the base."""
    slots = list(obs.opp_backline) + [None] * (Z - len(obs.opp_backline))
    slots += list(obs.frontline) if obs.front_owner < 0 else []
    return slots + [None] * (2 * Z + 1 - len(slots))


def incoming(obs, mask) -> np.ndarray:
    att = attack_grid(mask)
    atk = [0 if u is None else u.atk for u in attackers_of(obs)]
    return np.array([sum(atk[a] for a in range(2 * Z) if att[a, t]) for t in range(2 * Z + 1)], dtype=float)


# ---------------------------------------------------------------- custom effect rulesets
def vunit(cid, atk, hp, nature="troop", cost=1, traits=None, effects=(), token=False):
    d = {"id": cid, "name": cid, "type": "unit", "nature": nature, "cost": cost, "attack": atk, "health": hp,
         "effects": list(effects)}
    if traits:
        d["traits"] = traits
    if token:
        d["token"] = True
    return d


def op(cid, effects, cost=1):
    return {"id": cid, "name": cid, "type": "operation", "cost": cost, "effects": list(effects)}


def chosen(action, side, kind="unit", **params):
    e = {"trigger": "on_play", "action": action, "target": {"select": "chosen", "side": side, "kind": kind}}
    e.update(params)
    return e


FILLER = [vunit(f"f{i:02d}", 1 + i % 3, 1 + i % 4, cost=1 + i % 8) for i in range(14)]
OPS = [op("zap", [chosen("damage", "any", "unit_or_base", amount=3)]),
       op("mend", [chosen("heal", "friendly", "unit_or_base", amount=2)]),
       op("mend_full", [chosen("heal", "any", "unit", amount="full")]),
       op("doom", [chosen("destroy", "enemy")]),
       op("rally", [chosen("buff", "friendly", atk=2, hp=1)]),
       op("weaken", [chosen("buff", "enemy", atk=-3)]),
       op("snare", [chosen("pin", "enemy")]),
       op("double", [{"trigger": "on_play", "action": "draw", "amount": 1, "target": "controller"},
                     chosen("damage", "enemy", amount=2)])]
CHOICE_UNITS = [vunit("sniper", 2, 2, cost=2, effects=[
    {"trigger": "on_deploy", "action": "damage", "amount": 1,
     "target": {"select": "chosen", "side": "enemy", "kind": "unit_or_base"}}])]


def effect_rules(extra=(), mulligan=False):
    cards = FILLER + OPS + CHOICE_UNITS + list(extra)
    ids = [c["id"] for c in FILLER]
    deck_a = {cid: 2 for cid in ids[:10]}
    deck_a.update({"zap": 2, "mend": 2, "mend_full": 2, "doom": 2, "rally": 2, "sniper": 3, ids[10]: 3, ids[11]: 3,
                   ids[12]: 1})
    deck_b = {cid: 2 for cid in ids}
    deck_b.update({"weaken": 3, "snare": 3, "doom": 2, "mend_full": 2, "sniper": 2})
    assert sum(deck_a.values()) == sum(deck_b.values()) == 40
    return build_ruleset(cards, [{"name": "a", "cards": deck_a}, {"name": "b", "cards": deck_b}], mulligan=mulligan)


RULES = effect_rules()
ENC_R = ObservationEncoder(RULES)


def cid(config, name) -> int:
    return config.cards.by_id(name).index


# ---------------------------------------------------------------- state sampling
def sample_states(config, n_games, seed=0, every=1):
    rng = random.Random(seed)
    g = Game(config)
    for s in range(n_games):
        g.reset(seed * 1000 + s)
        step = 0
        while not g.done:
            if step % every == 0:
                yield g
            la = g.legal_actions()
            g.step(la[-1] if rng.random() < 0.3 else la[rng.randrange(len(la))])
            step += 1


def own_mask(g, p):
    return g.legal_mask() if (not g.done and g.current_player() == p) else None


# ================================================================ layout
def test_layout_sizes_offsets_and_names():
    assert ENCODER_VERSION == 5
    L = LAY
    assert (L["T"], L["H"], L["Z"], L["R"]) == (50, H, Z, R)
    assert L["F"] == len(TOKEN_FEATURES) == len(set(TOKEN_FEATURES))
    assert L["G"] == len(GLOBAL_FEATURES) == len(set(GLOBAL_FEATURES))
    assert L["P"] == len(ATTACK_PREVIEW_FEATURES) == 4 and L["PC"] == len(CHOOSE_PREVIEW_FEATURES) == 4
    assert L["token_features"] == list(TOKEN_FEATURES) and L["global_features"] == list(GLOBAL_FEATURES)
    assert L["n_cards"] == N_CARDS and L["n_actions"] == N
    T, F, G = L["T"], L["F"], L["G"]
    sizes = [("globals", G), ("features", T * F), ("ids", T), ("present", T), ("deck_counts", N_CARDS),
             ("attack_preview", 2 * Z * (2 * Z + 1) * 4), ("choose_preview", (3 * Z + 2) * 4)]
    off = 0
    for name, size in sizes:
        assert L["offsets"][name] == off, name
        off += size
    assert L["dim"] == ENC.dim == off
    # SPEC §6 token order: groups tile [0, T) in order
    order = ["global", "hand", "my_back", "front", "opp_back", "my_base", "opp_base", "pending", "revealed", "deck"]
    start = 0
    for name, size in zip(order, [1, H, Z, Z, Z, 1, 1, 1, R, 1]):
        assert L["groups"][name] == [start, start + size], name
        start += size
    assert start == T and L["groups"]["bases"] == [MYBASE, OPPBASE + 1]
    assert (L["groups"]["hand"][0], L["groups"]["my_back"][0], L["groups"]["revealed"][0], L["groups"]["deck"][0]) \
        == (HAND0, MYB0, REV0, DECK)
    # action blocks (SPEC §3)
    for k in ("END_TURN", "PLAY0", "MOVE0", "ATTACK0", "CHOOSE0", "MULLIGAN0", "CONFIRM", "n_choose",
              "n_attackers", "n_targets", "BASE_TARGET", "ENEMY_BASE_CHOICE", "OWN_BASE_CHOICE"):
        assert L[k] == getattr(SP, k), k
    assert (L["CHOOSE0"], L["MULLIGAN0"], L["CONFIRM"], L["n_actions"]) == (126, 143, 153, 154)
    # action slot -> token maps
    assert L["attacker_tokens"] == list(range(MYB0, MYB0 + Z)) + list(range(FRONT0, FRONT0 + Z))
    tgt = list(range(OPPB0, OPPB0 + Z)) + list(range(FRONT0, FRONT0 + Z)) + [OPPBASE]
    assert L["attack_target_tokens"] == tgt
    assert L["choose_tokens"] == tgt + list(range(MYB0, MYB0 + Z)) + [MYBASE]
    # card table
    assert L["S"] == len(CARD_TABLE_FEATURES) + len(L["tags"]) == len(L["card_table_features"])
    assert np.asarray(L["card_table"]).shape == (N_CARDS, L["S"])
    assert L["card_table_features"][:len(CARD_TABLE_FEATURES)] == list(CARD_TABLE_FEATURES)
    assert len(CARD_TABLE_FEATURES) == len(CARD_FEATURES) + 3 * len(EFFECT_FEATURES)
    assert L["fingerprint"] == pool_fingerprint(CONFIG) and L["version"] == 5
    assert L["card_ids"] == [c.id for c in CONFIG.cards.cards]
    json.loads(json.dumps(L))  # plain Python types (checkpoint- and JSON-safe)
    assert ObservationEncoder(CONFIG).layout() == L


def test_split_round_trip_numpy_and_torch():
    g = Game(CONFIG_M)
    g.reset(3)
    enc = ObservationEncoder(CONFIG_M)
    x = enc.encode(g.observe(g.current_player()), g.legal_mask())
    p = enc.split(x)
    T, F = enc.T, enc.F
    assert p.globals.shape == (enc.G,) and p.features.shape == (T, F) and p.ids.shape == (T,)
    assert p.present.shape == (T,) and p.present.dtype == np.bool_ and p.ids.dtype == np.int64
    assert p.deck_counts.shape == (enc.n_cards,)
    assert p.attack_preview.shape == (2 * Z, 2 * Z + 1, 4) and p.choose_preview.shape == (3 * Z + 2, 4)
    back = np.concatenate([p.globals, p.features.ravel(), p.ids.astype(np.float32), p.present.astype(np.float32),
                           p.deck_counts, p.attack_preview.ravel(), p.choose_preview.ravel()])
    assert np.array_equal(back, x)
    # absent tokens carry nothing; present tokens have a type
    assert not p.features[~p.present].any() and not p.ids[~p.present].any()
    n_types = len(fmod.TOKEN_TYPES)
    assert (p.features[p.present, :n_types].sum(axis=1) == 1).all()
    # batch shapes and torch tensors
    batch = np.stack([x, x * 0, x])
    pb = enc.split(batch.reshape(1, 3, -1))
    assert pb.features.shape == (1, 3, T, F) and pb.choose_preview.shape == (1, 3, 3 * Z + 2, 4)
    torch = pytest.importorskip("torch")
    pt = enc.split(torch.from_numpy(batch))
    assert pt.ids.dtype == torch.int64 and pt.present.dtype == torch.bool
    assert pt.attack_preview.shape == (3, 2 * Z, 2 * Z + 1, 4)
    assert np.array_equal(pt.features.numpy(), np.stack([p.features, p.features * 0, p.features]))


# ================================================================ hand-built positions
def _cards_by_cost(config):
    units = sorted((c for c in config.cards.cards if c.is_unit and not c.token), key=lambda c: (c.cost, c.index))
    return units[0].index, units[len(units) // 2].index, units[-1].index


def test_every_token_group_of_a_hand_built_position():
    cheap, mid, dear = _cards_by_cost(CONFIG)
    assert CONFIG.cards[cheap].cost <= 4 < CONFIG.cards[dear].cost
    g = blank_game(current=0, first=1, round_=4, coins=4)
    set_hand(g, 0, [dear, cheap, cheap])
    set_hand(g, 1, [mid, mid, cheap])
    g.deck_cards[0] = [mid, cheap, dear, mid]
    g.deck_cards[1] = [cheap] * 5
    add_unit(g, 0, "back", CONFIG.cards[cheap].id)                                       # ready
    add_unit(g, 0, "back", atk=3, hp=2, max_hp=5, armor=1, pinned=True, pin_until=g.turn + 1)  # damaged, pinned
    add_unit(g, 0, "back", atk=2, hp=1, nature=FAST, summoned=True)                     # just deployed
    front_card = add_unit(g, 0, "front", atk=4, hp=3, max_hp=3, nature=FAST, moved=True, fury=True, attacks=1,
                          temp_atk=1, temp_hp=2, move_cost=2).card
    add_unit(g, 1, "back", atk=2, hp=4, defense=True)
    add_unit(g, 1, "back", atk=5, hp=2, smokescreen=True)
    add_unit(g, 1, "back", atk=1, hp=1, blitz=True, token=True)
    g.coin_bonus = [1, -2]
    g.burned = [2, 1]
    g.base_hp = [17, 9]
    g.known_hand[0][cheap] = 1          # one of my two cheap cards is known to the opponent
    g.known_hand[1][mid] = 1            # I know one opponent hand card
    revealed_card, grave_card, disc_card = (c.index for c in CONFIG.cards.cards[-3:])
    g.revealed[1][revealed_card] = 2
    g.graveyard[1][grave_card] = 1
    g.discard[1][disc_card] = 1
    g.revealed[0][mid] = 3              # my own public cards never become opponent-revealed tokens
    g.revealed[1][front_card] += 1      # the opponent revealed a card I hold on the board: not on *their* board
    g.history_turn = [[1, 2, 0], [0, 1, 3]]  # public counters (SPEC §5): ops, deployed, died
    g.history_game = [[4, 5, 6], [7, 8, 9]]
    g.invalidate()
    obs, mask = g.observe(0), g.legal_mask()
    r = Read(ENC.encode(obs, mask))
    att = attack_grid(mask)
    inc = incoming(obs, mask)
    assert att.any(), "the position should allow an attack"
    assert (obs.my_history, obs.opp_history) == ((1, 2, 0, 4, 5, 6), (0, 1, 3, 7, 8, 9))

    # globals
    want = {"is_my_turn": 1, "went_first": 0, "round": 4 / 50, "turn": g.turn / 100, "phase_mulligan": 0,
            "phase_main": 1, "phase_choice": 0, "my_coins": 0.4, "opp_coins": 0, "my_coin_bonus": 0.1,
            "opp_coin_bonus": -0.2, "my_base_hp": 17 / 20, "opp_base_hp": 9 / 20, "my_hand_size": 0.3,
            "opp_hand_size": 0.3, "my_deck_size": 4 / 40, "opp_deck_size": 5 / 40, "my_burned": 0.2,
            "opp_burned": 0.1, "front_mine": 1, "front_empty": 0, "front_opp": 0, "n_legal": mask.sum() / N,
            "base_damage_ready": inc[2 * Z] / 20}
    # history: this turn / 5, this game / 20
    for name, v in zip(HISTORY_NAMES, (1, 2, 0, 4, 5, 6, 0, 1, 3, 7, 8, 9)):
        want[name] = v / (5 if name.endswith("_turn") else 20)
    assert {k: pytest.approx(v, abs=1e-6) for k, v in want.items()} == {k: r.g(k) for k in GLOBAL_FEATURES}
    r.row_is(G_TOK, {"type_global": 1})

    # hand: sorted by card index; the first copy of the known card is marked
    hand = obs.hand
    assert hand == tuple(sorted([dear, cheap, cheap]))
    first_cheap = hand.index(cheap)
    for i in range(H):
        if i < len(hand):
            r.row_is(HAND0 + i, {"type_hand": 1, "playable": mask[SP.PLAY0 + i], "known_to_opp": i == first_cheap},
                     ident=hand[i] + 1)
        else:
            r.row_is(HAND0 + i, {}, present=False)
    assert r.f(HAND0 + hand.index(cheap), "playable") == 1 and r.f(HAND0 + hand.index(dear), "playable") == 0

    # units: unit state + mask hints
    for j in range(Z):
        tok = MYB0 + j
        if j < len(obs.my_backline):
            u = obs.my_backline[j]
            r.row_is(tok, {"type_my_back": 1, **unit_expect(u), "move_legal": mask[SP.MOVE0 + j],
                           "attack_ready": att[j].any()}, ident=u.card + 1)
        else:
            r.row_is(tok, {}, present=False)
    assert r.f(MYB0 + 1, "damaged") == 1 and r.f(MYB0 + 1, "pinned") == 1 and r.f(MYB0 + 2, "summoned") == 1
    assert r.f(MYB0 + 1, "armor") == pytest.approx(0.25) and r.f(MYB0 + 1, "max_hp") == pytest.approx(0.5)
    u = obs.frontline[0]
    r.row_is(FRONT0, {"type_front_mine": 1, **unit_expect(u), "attack_ready": att[Z].any()}, ident=u.card + 1)
    assert (r.f(FRONT0, "temp_atk"), r.f(FRONT0, "temp_hp"), r.f(FRONT0, "attacks"), r.f(FRONT0, "move_cost")) \
        == pytest.approx((0.1, 0.2, 0.5, 0.5))
    assert r.f(FRONT0, "attack_ready") == 1  # fury: a second attack after moving (fast)
    for j in range(1, Z):
        r.row_is(FRONT0 + j, {}, present=False)
    for j in range(Z):
        tok = OPPB0 + j
        if j < len(obs.opp_backline):
            u = obs.opp_backline[j]
            r.row_is(tok, {"type_opp_back": 1, **unit_expect(u), "attack_targetable": att[:, j].any(),
                           "incoming_atk": inc[j] / 10}, ident=u.card + 1)
        else:
            r.row_is(tok, {}, present=False)
    assert r.f(OPPB0, "attack_targetable") == 1 and r.f(OPPB0 + 1, "attack_targetable") == 0  # Defense, smokescreen
    assert r.f(OPPB0, "incoming_atk") == pytest.approx(0.4)

    # bases
    r.row_is(MYBASE, {"type_my_base": 1, "base_hp": 17 / 20})
    r.row_is(OPPBASE, {"type_opp_base": 1, "base_hp": 9 / 20, "attack_targetable": att[:, 2 * Z].any(),
                       "incoming_atk": inc[2 * Z] / 20})
    # no pending choice in phase MAIN
    r.row_is(PEND, {}, present=False)

    # opponent-revealed cards: unique indices with any public presence, ascending
    pub = sorted({c for c in range(N_CARDS) if g.revealed[1][c] or g.known_hand[1][c] or g.graveyard[1][c]
                  or g.discard[1][c]})
    assert pub == sorted({mid, revealed_card, grave_card, disc_card, front_card})
    board1 = Counter(u.card for u in g.backline[1])
    assert board1[front_card] == 0
    for k in range(R):
        if k < len(pub):
            c = pub[k]
            r.row_is(REV0 + k, {"type_revealed": 1, "revealed": g.revealed[1][c] / 3,
                                "known_in_hand": g.known_hand[1][c] / 3, "graveyard": g.graveyard[1][c] / 3,
                                "discard": g.discard[1][c] / 3, "on_board": board1[c] / 3}, ident=c + 1)
        else:
            r.row_is(REV0 + k, {}, present=False)

    # own deck summary token + deck counts block (raw remaining counts)
    r.row_is(DECK, {"type_deck": 1})
    want_counts = Counter(g.deck_cards[0])
    assert r.p.deck_counts.tolist() == [float(want_counts[c]) for c in range(N_CARDS)]
    # attack previews from combat_damage on the legal pairs only
    assert att[Z, 2 * Z] and att[Z, 0]
    for a, t in zip(*np.nonzero(att)):
        attacker, target = attackers_of(obs)[a], targets_of(obs)[t]
        if target is None:  # the enemy base
            want = [float(attacker.atk >= obs.opp_base_hp), 0.0, attacker.atk / 10, 0.0]
        else:
            d, k = combat_damage(attacker, target)
            want = [float(d >= target.hp), float(k >= attacker.hp), d / 10, k / 10]
        assert r.p.attack_preview[a, t].tolist() == pytest.approx(want, abs=1e-6)
    assert not r.p.attack_preview[~att].any() and not r.p.choose_preview.any()


def test_opponent_frontline_and_ranged_base_attack():
    g = blank_game(current=1, first=0, round_=6, coins=6)
    add_unit(g, 1, "back", atk=3, hp=3, nature=RANGED)
    add_unit(g, 1, "back", atk=2, hp=5, nature=TROOP)
    add_unit(g, 0, "front", atk=4, hp=2, armor=1)
    add_unit(g, 0, "front", atk=1, hp=6, defense=True)
    add_unit(g, 0, "back", atk=2, hp=2)
    seen_card = g.frontline[0].card
    g.revealed[0][seen_card] = 1
    g.invalidate()
    obs, mask = g.observe(1), g.legal_mask()
    r = Read(ENC.encode(obs, mask))
    on_board = sum(u.card == seen_card for u in (*g.backline[0], *g.frontline))
    assert on_board >= 1 and int(r.p.ids[REV0]) == seen_card + 1 and not r.p.present[REV0 + 1]
    assert r.f(REV0, "on_board") == pytest.approx(on_board / 3) and r.f(REV0, "revealed") == pytest.approx(1 / 3)
    att = attack_grid(mask)
    inc = incoming(obs, mask)
    assert r.g("front_opp") == 1 and r.g("front_mine") == 0 and r.g("is_my_turn") == 1 and r.g("went_first") == 0
    for j, u in enumerate(obs.frontline):
        r.row_is(FRONT0 + j, {"type_front_opp": 1, **unit_expect(u), "attack_targetable": att[:, Z + j].any(),
                              "incoming_atk": inc[Z + j] / 10}, ident=u.card + 1)
    # Defense in the frontline: both of my backline units must hit the Defense unit there
    assert r.f(FRONT0 + 1, "attack_targetable") == 1 and r.f(FRONT0, "attack_targetable") == 0
    assert r.f(FRONT0 + 1, "incoming_atk") == pytest.approx(0.5)
    # only the ranged unit reaches the base
    assert r.f(OPPBASE, "attack_targetable") == 1 and r.f(OPPBASE, "incoming_atk") == pytest.approx(3 / 20)
    assert r.g("base_damage_ready") == pytest.approx(3 / 20)
    assert r.p.attack_preview[0, 2 * Z].tolist() == pytest.approx([0, 0, 0.3, 0])
    # the same position seen by the waiting player: no hints, no previews, frontline is theirs
    r0 = Read(ENC.encode(g.observe(0)))
    assert r0.g("front_mine") == 1 and r0.g("n_legal") == 0
    assert r0.f(FRONT0, "type_front_mine") == 1 and r0.f(FRONT0, "attack_targetable") == 0
    assert not r0.p.attack_preview.any()


def test_mulligan_tokens_and_hidden_marks():
    g = Game(CONFIG_M)
    g.reset(11)
    enc = ObservationEncoder(CONFIG_M)
    p = g.current_player()
    assert g.phase == MULLIGAN
    obs, mask = g.observe(p), g.legal_mask()
    r = Read(enc.encode(obs, mask), enc)
    assert r.g("phase_mulligan") == 1 and r.g("phase_main") == 0 and r.g("turn") == 0
    for i in range(len(obs.hand)):
        assert r.f(HAND0 + i, "mulligan_legal") == 1 and r.f(HAND0 + i, "mulligan_marked") == 0
        assert r.f(HAND0 + i, "playable") == 0
    other_before = enc.encode(g.observe(1 - p))
    g.step(SP.mulligan(1))
    obs, mask = g.observe(p), g.legal_mask()
    r = Read(enc.encode(obs, mask), enc)
    assert r.f(HAND0 + 1, "mulligan_marked") == 1 and r.f(HAND0 + 1, "mulligan_legal") == 0
    assert r.f(HAND0, "mulligan_marked") == 0 and r.f(HAND0, "mulligan_legal") == 1
    assert r.g("n_legal") == pytest.approx(len(obs.hand) / N)  # the unmarked slots + CONFIRM
    # the decider's marks are hidden from the waiting player
    assert np.array_equal(enc.encode(g.observe(1 - p)), other_before)
    assert not Read(other_before, enc).p.features[:, TF["mulligan_marked"]].any()


# ================================================================ choices
def _choice_position(op_name, rules=None, immune_unit=False):
    g = blank_game(current=0, first=0, round_=5, coins=10, config=rules or RULES)
    set_hand(g, 0, [op_name])
    add_unit(g, 0, "back", "f00", atk=2, hp=2, max_hp=5)        # damaged
    add_unit(g, 0, "back", "f01", atk=1, hp=1, max_hp=1)
    add_unit(g, 1, "back", "f02", atk=2, hp=3, max_hp=3)
    add_unit(g, 1, "back", "f03", atk=5, hp=2, max_hp=4)
    add_unit(g, 1, "front", "f04", atk=4, hp=4, max_hp=4)
    if immune_unit:  # Unit.from_card carries every card trait (SPEC §2.4)
        u = Unit.from_card(g.config.cards.by_id("warded"), 1, g.next_uid)
        g.next_uid += 1
        u.summoned = False
        g.backline[1].append(u)
    g.base_hp = [12, 3]
    g.invalidate()
    g.step(SP.PLAY0)
    assert g.phase == CHOICE
    return g


def _tc_match(u, f) -> bool:
    """A target_condition filter on a UnitView, for the filter keys the pools under test use (SPEC §1.2)."""
    extra = [k for k in ("min_cost", "max_cost", "min_atk", "max_atk", "min_hp", "max_hp", "token", "pinned")
             if getattr(f, k) is not None] + [k for k in ("tags", "not_tags", "other") if getattr(f, k)]
    if extra:
        raise NotImplementedError(f"test helper: target_condition keys {extra}")
    has = sum(TRAIT_BITS[n] for n in TRAIT_NAMES if (u.armor > 0 if n == "armor" else getattr(u, n)))
    return ((f.traits & has) == f.traits and not (f.not_traits & has) and (not f.natures or u.nature in f.natures)
            and (f.damaged is None or (u.hp < u.max_hp) == f.damaged))


def _choose_expect(obs, t, base_cap=20, config=None):
    """SPEC §2.10 / §5 outcome of CHOOSE(t) on the option itself under obs.pending: (kills, dealt, healed,
    other), scaled /10. With `config`, an option that fails the effect's target_condition gets the else body
    (one that reuses the effect's target, literal amount) or nothing."""
    pend = obs.pending
    slots = (list(obs.opp_backline) + [None] * (Z - len(obs.opp_backline)) + list(obs.frontline)
             + [None] * (Z - len(obs.frontline)) + ["opp_base"] + list(obs.my_backline)
             + [None] * (Z - len(obs.my_backline)) + ["my_base"])
    u = slots[t]
    base = isinstance(u, str)
    hp = (obs.opp_base_hp if u == "opp_base" else obs.my_base_hp) if base else u.hp
    action, amount, b_atk, b_hp = pend.action, pend.amount, getattr(pend, "atk", 0), getattr(pend, "hp", 0)
    if config is not None and not base:
        eff = config.cards[pend.card].effects[pend.effect]
        if eff.target_condition is not None:
            assert not eff.condition, "test helper: a condition and a target_condition on one effect"
            if not _tc_match(u, eff.target_condition):
                alt = eff.else_
                if alt is None:
                    return [0, 0, 0, 0]
                if alt.own_target or alt.action == "buff" or (alt.amount is not None and alt.amount.kind != "literal"):
                    raise NotImplementedError("test helper: else body with its own target, a buff or an expression")
                action, amount = alt.action, (0 if alt.amount is None else alt.amount.value)
    if action == "damage":
        dealt = 0 if (not base and getattr(u, "immune", False)) else amount
        return [float(dealt > 0 and dealt >= hp), dealt / 10, 0, 0]
    if action == "destroy":
        return [1, 0, 0, 0]
    if action == "heal":
        cap = base_cap if base else u.max_hp
        new = cap if amount == -1 else min(cap, hp + amount)
        return [0, 0, max(0, new - hp) / 10, 0]
    if action == "buff":
        return [0, 0, 0, (max(0, u.atk + b_atk) - u.atk + b_hp) / 10]
    return [0, 0, 0, 0.1]  # any other action that applies to the option: other = 1


def _check_choice(g, enc=None):
    enc = enc or (ENC_R if g.config is RULES else ObservationEncoder(g.config))
    p = g.current_player()
    obs, mask = g.observe(p), g.legal_mask()
    r = Read(enc.encode(obs, mask), enc)
    pend = obs.pending
    choose = mask[SP.CHOOSE0:SP.CHOOSE0 + SP.n_choose]
    assert choose.any() and r.g("phase_choice") == 1
    want = {"type_pending": 1, f"effect{pend.effect}": 1}
    if pend.action == "heal" and pend.amount == -1:
        want["pending_full"] = 1
    else:
        want["pending_amount"] = pend.amount / 10
    want["pending_atk"] = getattr(pend, "atk", 0) / 10
    want["pending_hp"] = getattr(pend, "hp", 0) / 10
    r.row_is(PEND, want, ident=pend.card + 1)
    previews = np.asarray(pend.previews, dtype=float)
    assert previews.shape == (SP.n_choose, 4) and not previews[~choose].any()  # zeros for non-options
    for t, tok in enumerate(LAY["choose_tokens"]):
        assert r.f(tok, "choose_legal") == choose[t], t
        eff = g.config.cards[pend.card].effects[pend.effect]
        cap = 99 if getattr(eff, "uncapped", False) else g.config.base_hp  # SPEC §1.2b uncapped heal
        exp = _choose_expect(obs, t, cap, g.config) if choose[t] else [0, 0, 0, 0]
        assert r.p.choose_preview[t].tolist() == pytest.approx(exp, abs=1e-6), (pend.action, t)
        # the encoder copies the engine's preview (kills as is, amounts /10)
        assert r.p.choose_preview[t].tolist() == pytest.approx(previews[t] * [1, .1, .1, .1], abs=1e-6), t
    # the waiting player sees the (public) pending token but gets no hints or previews
    r1 = Read(enc.encode(g.observe(1 - p)), enc)
    assert r1.p.features[PEND].tolist() == r.p.features[PEND].tolist() and int(r1.p.ids[PEND]) == pend.card + 1
    assert not r1.p.choose_preview.any() and not r1.p.features[:, TF["choose_legal"]].any()
    return r, obs, choose


def test_choose_previews_damage():
    g = _choice_position("zap")
    r, obs, choose = _check_choice(g)
    assert obs.pending.action == "damage" and obs.pending.amount == 3
    assert choose.sum() == 2 + 1 + 1 + 2 + 1  # 2 enemy back, enemy front, enemy base, 2 own back, own base
    assert r.p.choose_preview[0].tolist() == pytest.approx([1, 0.3, 0, 0])      # 3/3 dies
    assert r.p.choose_preview[Z].tolist() == pytest.approx([0, 0.3, 0, 0])      # 4/4 survives
    assert r.p.choose_preview[2 * Z].tolist() == pytest.approx([1, 0.3, 0, 0])  # enemy base at 3: lethal
    assert r.p.choose_preview[3 * Z + 1].tolist() == pytest.approx([0, 0.3, 0, 0])  # own base at 12
    assert r.f(OPPBASE, "choose_legal") == 1 and r.f(MYBASE, "choose_legal") == 1


def test_choose_previews_heal_destroy_and_other_actions():
    r, obs, choose = _check_choice(_choice_position("mend"))
    assert r.p.choose_preview[2 * Z + 1].tolist() == pytest.approx([0, 0, 0.2, 0])  # 2/5 -> 4/5
    assert r.p.choose_preview[2 * Z + 2].tolist() == pytest.approx([0, 0, 0, 0])    # full hp
    assert r.p.choose_preview[3 * Z + 1].tolist() == pytest.approx([0, 0, 0.2, 0])  # base 12 -> 14
    assert not choose[:2 * Z + 1].any()  # friendly only
    r, obs, choose = _check_choice(_choice_position("mend_full"))
    assert r.f(PEND, "pending_full") == 1 and r.f(PEND, "pending_amount") == 0
    assert r.p.choose_preview[2 * Z + 1].tolist() == pytest.approx([0, 0, 0.3, 0])  # 2/5 -> 5/5
    assert r.p.choose_preview[1].tolist() == pytest.approx([0, 0, 0.2, 0])          # 2/4 -> 4/4
    r, obs, choose = _check_choice(_choice_position("doom"))
    assert [r.p.choose_preview[t][0] for t in np.flatnonzero(choose)] == [1.0] * int(choose.sum())
    r, obs, choose = _check_choice(_choice_position("snare"))  # pin: other = 1 on every option
    assert choose.any() and r.p.choose_preview[choose].tolist() == [pytest.approx([0, 0, 0, 0.1])] * choose.sum()
    assert not r.p.choose_preview[~choose].any()
    r, obs, choose = _check_choice(_choice_position("double"))  # the chosen effect is the card's second
    assert obs.pending.effect == 1 and r.f(PEND, "effect1") == 1 and r.f(PEND, "effect0") == 0
    assert r.p.choose_preview[1].tolist() == pytest.approx([1, 0.2, 0, 0])  # 2 damage on a 5/2


def test_choose_previews_are_copied_from_the_pending_view():
    """The encoder copies PendingView.previews (SPEC §5, §6): no rule is re-derived from the unit views, so
    editing a view (immune, hp) leaves the previews alone, while edited previews are copied on legal slots."""
    g = _choice_position("zap")
    obs, mask = g.observe(0), g.legal_mask()
    choose = mask[SP.CHOOSE0:SP.CHOOSE0 + SP.n_choose]
    base = Read(ENC_R.encode(obs, mask), ENC_R).p.choose_preview.copy()
    assert base[0].tolist() == pytest.approx([1, 0.3, 0, 0])  # 3 damage on a 3/3
    zones = {name: tuple(u._replace(immune=True, hp=9) for u in getattr(obs, name))
             for name in ("my_backline", "opp_backline", "frontline")}
    assert np.array_equal(Read(ENC_R.encode(obs._replace(**zones), mask), ENC_R).p.choose_preview, base)
    fake = tuple((t % 2, t, 2 * t, -t) for t in range(SP.n_choose))
    r = Read(ENC_R.encode(obs._replace(pending=obs.pending._replace(previews=fake)), mask), ENC_R)
    want = [[t % 2, t / 10, 2 * t / 10, -t / 10] if choose[t] else [0, 0, 0, 0] for t in range(SP.n_choose)]
    assert np.allclose(r.p.choose_preview, want, atol=1e-6)
    # a PendingView without previews encodes none; a wrong slot count is rejected
    r = Read(ENC_R.encode(obs._replace(pending=obs.pending._replace(previews=())), mask), ENC_R)
    assert not r.p.choose_preview.any() and r.p.features[LAY["choose_tokens"], TF["choose_legal"]].any()
    with pytest.raises(ValueError, match="slots"):
        ENC_R.encode(obs._replace(pending=obs.pending._replace(previews=fake[:-1])), mask)


def test_malformed_masks_never_mark_absent_tokens_or_misread_slots():
    cheap, _, _ = _cards_by_cost(CONFIG)
    g = blank_game(current=0, first=0, round_=3, coins=3)
    set_hand(g, 0, [cheap])
    add_unit(g, 0, "back", atk=2, hp=2)
    add_unit(g, 1, "back", atk=1, hp=1)
    obs, mask = g.observe(0), g.legal_mask()
    bad = mask.copy()
    bad[SP.PLAY0 + 3] = bad[SP.MOVE0 + 4] = bad[SP.MULLIGAN0 + 5] = True
    r = Read(ENC.encode(obs, bad))
    r.row_is(HAND0 + 3, {}, present=False)
    r.row_is(HAND0 + 5, {}, present=False)
    r.row_is(MYB0 + 4, {}, present=False)
    for a, t in ((3, 0), (Z, 2 * Z), (0, 4), (0, Z + 1)):  # empty attacker / front not mine / empty targets
        bad = mask.copy()
        bad[SP.attack(a, t)] = True
        with pytest.raises(ValueError, match="does not belong"):
            ENC.encode(obs, bad)
    g = _choice_position("zap")
    bad = g.legal_mask()
    bad[SP.choose(4)] = True  # enemy backline slot 4 is empty
    with pytest.raises(ValueError, match="does not belong"):
        ENC_R.encode(g.observe(0), bad)


def test_choose_previews_buff():
    need("atk" in PendingView._fields and "hp" in PendingView._fields, "PendingView.atk/hp")
    r, obs, choose = _check_choice(_choice_position("rally"))
    assert (obs.pending.atk, obs.pending.hp) == (2, 1)
    assert r.f(PEND, "pending_atk") == pytest.approx(0.2) and r.f(PEND, "pending_hp") == pytest.approx(0.1)
    assert r.p.choose_preview[2 * Z + 1].tolist() == pytest.approx([0, 0, 0, 0.3])
    r, obs, choose = _check_choice(_choice_position("weaken"))
    assert r.p.choose_preview[0].tolist() == pytest.approx([0, 0, 0, -0.2])   # 2 atk -> 0 (clamped)
    assert r.p.choose_preview[Z].tolist() == pytest.approx([0, 0, 0, -0.3])   # 4 atk -> 1


def test_choose_previews_immune_target_takes_no_damage():
    need("immune" in UnitView._fields, "UnitView.immune")
    try:
        rules = effect_rules([vunit("warded", 1, 1, cost=2, traits={"immune": True})])
    except ValueError:
        pytest.skip("card loader does not accept the immune trait yet")
    g = _choice_position("zap", rules=rules, immune_unit=True)
    r, obs, choose = _check_choice(g, ObservationEncoder(rules))
    assert obs.opp_backline[2].immune
    assert choose[2] and r.p.choose_preview[2].tolist() == pytest.approx([0, 0, 0, 0])
    assert r.f(OPPB0 + 2, "immune") == 1


def _slot_unit(g, t, p=0):
    """The unit behind CHOOSE slot t of chooser p (SPEC §2.9 canonical slots)."""
    if t < Z:
        return g.backline[1 - p][t]
    if t < 2 * Z:
        return g.frontline[t - Z]
    return g.backline[p][t - 2 * Z - 1]


def _realized(g, t):
    """What CHOOSE(t) does to the option itself, played on a clone: (killed, hp lost, hp gained)."""
    c = g.clone()
    u = _slot_unit(c, t)
    before = u.hp
    c.step(SP.choose(t))
    alive = any(x is u for x in (*c.backline[0], *c.backline[1], *c.frontline))
    return float(not alive), max(0, before - u.hp), max(0, u.hp - before)


def test_choose_previews_follow_target_condition_and_else():
    """Review finding 3: an option that fails target_condition is previewed with the body that applies to it
    (the else body on the same target, an else body with its own target only if it reaches the option, or
    nothing); a failed condition previews the else body. Every preview matches what CHOOSE really does."""
    enemy = {"select": "chosen", "side": "enemy", "kind": "unit"}
    extra = [op("pierce", [{"trigger": "on_play", "action": "damage", "amount": 4, "target": enemy,
                            "target_condition": {"trait": "armor"}, "else": {"action": "damage", "amount": 2}}]),
             op("flak", [{"trigger": "on_play", "action": "damage", "amount": 3, "target": enemy,
                          "target_condition": {"nature": "fast"},
                          "else": {"action": "damage", "amount": 1, "target": {
                              "select": "all", "side": "enemy", "kind": "unit", "filter": {"damaged": True}}}}]),
             op("hush", [{"trigger": "on_play", "action": "destroy", "target": enemy,
                          "target_condition": {"damaged": True}}]),
             op("stand", [{"trigger": "on_play", "condition": {"type": "frontline", "owner": "friendly"},
                           "action": "damage", "amount": 5, "else": {"action": "heal", "amount": 2},
                           "target": {"select": "chosen", "side": "any", "kind": "unit"}}])]
    rules = effect_rules(extra)
    enc = ObservationEncoder(rules)
    # enemy back: A armored 2/5, B 2/3, C damaged 2/2 (max 4); enemy front: D fast 3/3, E fast damaged 1/1 (max
    # 2); own back: F damaged 2/2 (max 5)
    want = {"pierce": {0: (0, 4, 0, 0), 1: (0, 2, 0, 0), 2: (1, 2, 0, 0), Z: (0, 2, 0, 0), Z + 1: (1, 2, 0, 0)},
            "flak": {0: (0, 0, 0, 0), 1: (0, 0, 0, 0), 2: (0, 1, 0, 0), Z: (1, 3, 0, 0), Z + 1: (1, 3, 0, 0)},
            "hush": {0: (0, 0, 0, 0), 1: (0, 0, 0, 0), 2: (1, 0, 0, 0), Z: (0, 0, 0, 0), Z + 1: (1, 0, 0, 0)},
            "stand": {0: (0, 0, 0, 0), 1: (0, 0, 0, 0), 2: (0, 0, 2, 0), Z: (0, 0, 0, 0), Z + 1: (0, 0, 1, 0),
                      2 * Z + 1: (0, 0, 2, 0)}}
    for name, expected in want.items():
        g = blank_game(current=0, first=0, round_=5, coins=10, config=rules)
        set_hand(g, 0, [name])
        add_unit(g, 1, "back", "f00", atk=2, hp=5, max_hp=5, armor=1)
        add_unit(g, 1, "back", "f01", atk=2, hp=3, max_hp=3)
        add_unit(g, 1, "back", "f02", atk=2, hp=2, max_hp=4)
        add_unit(g, 1, "front", "f03", atk=3, hp=3, max_hp=3, nature=FAST)
        add_unit(g, 1, "front", "f04", atk=1, hp=1, max_hp=2, nature=FAST)
        add_unit(g, 0, "back", "f05", atk=2, hp=2, max_hp=5)
        g.step(SP.PLAY0)
        assert g.phase == CHOICE, name
        obs, mask = g.observe(0), g.legal_mask()
        pend = obs.pending
        choose = mask[SP.CHOOSE0:SP.CHOOSE0 + SP.n_choose]
        assert sorted(np.flatnonzero(choose).tolist()) == sorted(expected), name
        # the pending body: the effect, or its else body when the condition fails (no friendly frontline)
        assert (pend.action, pend.amount) == {"pierce": ("damage", 4), "flak": ("damage", 3), "hush": ("destroy", 0),
                                              "stand": ("heal", 2)}[name]
        r = Read(enc.encode(obs, mask), enc)
        for t in range(SP.n_choose):
            exp = expected.get(t, (0, 0, 0, 0))
            assert pend.previews[t] == exp, (name, t, pend.previews[t])
            assert r.p.choose_preview[t].tolist() == pytest.approx([exp[0], exp[1] / 10, exp[2] / 10, exp[3] / 10])
        for t in expected:  # the preview is what happens to the option
            killed, lost, gained = _realized(g, t)
            kills, dealt, healed, _ = pend.previews[t]
            assert (killed, lost, gained) == (kills, dealt, healed), (name, t)
        assert g.observe(1).pending == pend  # public, identical for the waiting player


def test_choose_previews_of_other_actions_and_buffs_match_the_engine():
    """`other` = the applied atk change + hp change for a buff (atk clamped at 0), 1 for any other action that
    applies (SPEC §5); a move-cost-only buff changes neither."""
    extra = [op("mud", [chosen("buff", "enemy", move_cost=1, duration="turn")]),
             op("oust", [chosen("return_to_hand", "enemy")]),
             op("rouse", [chosen("add_trait", "friendly", trait="fury")])]
    rules = effect_rules(extra)
    cases = (("weaken", 0, (0, 0, 0, -2)), ("weaken", Z, (0, 0, 0, -3)), ("rally", 2 * Z + 1, (0, 0, 0, 3)),
             ("mud", 0, (0, 0, 0, 0)), ("oust", 1, (0, 0, 0, 1)), ("rouse", 2 * Z + 2, (0, 0, 0, 1)),
             ("doom", Z, (1, 0, 0, 0)))
    for name, slot, want in cases:
        g = _choice_position(name, rules)
        pend = g.observe(0).pending
        assert pend.previews[slot] == want, (name, slot, pend.previews[slot])
        if pend.action == "buff":  # the applied change, read back from the engine after CHOOSE
            c = g.clone()
            u = _slot_unit(c, slot)
            atk, hp = u.atk, u.hp
            c.step(SP.choose(slot))
            assert (u.atk - atk) + (u.hp - hp) == want[3], name


# ================================================================ sweeps over real games
def _sweep_configs():
    return [(CONFIG, ENC, 30), (CONFIG_M, ObservationEncoder(CONFIG_M), 20), (RULES, ENC_R, 40)]


def test_mask_hints_and_previews_match_mask_and_combat_damage():
    seen = Counter()
    for config, enc, n_games in _sweep_configs():
        lay = enc.layout()
        for g in sample_states(config, n_games, seed=5, every=2):
            p = g.current_player()
            obs, mask = g.observe(p), g.legal_mask()
            r = Read(enc.encode(obs, mask), enc)
            f = r.p.features
            att = attack_grid(mask)
            choose = mask[SP.CHOOSE0:SP.CHOOSE0 + SP.n_choose]
            nh = len(obs.hand)
            assert np.array_equal(f[HAND0:HAND0 + nh, TF["playable"]], mask[SP.PLAY0:SP.PLAY0 + nh])
            assert np.array_equal(f[HAND0:HAND0 + nh, TF["mulligan_legal"]], mask[SP.MULLIGAN0:SP.MULLIGAN0 + nh])
            assert np.array_equal(f[MYB0:MYB0 + Z, TF["move_legal"]], mask[SP.MOVE0:SP.MOVE0 + Z])
            assert np.array_equal(f[lay["attacker_tokens"], TF["attack_ready"]], att.any(axis=1))
            assert np.array_equal(f[lay["attack_target_tokens"], TF["attack_targetable"]], att.any(axis=0))
            assert np.array_equal(f[lay["choose_tokens"], TF["choose_legal"]], choose)
            scale = np.array([10.0] * (2 * Z) + [20.0])
            assert np.allclose(f[lay["attack_target_tokens"], TF["incoming_atk"]], incoming(obs, mask) / scale)
            assert r.g("n_legal") == pytest.approx(mask.sum() / N)
            # attack previews: combat_damage on legal pairs, zero elsewhere
            want = np.zeros((2 * Z, 2 * Z + 1, 4))
            for a, t in zip(*np.nonzero(att)):
                u, v = attackers_of(obs)[a], targets_of(obs)[t]
                if t == 2 * Z:
                    want[a, t] = [u.atk >= obs.opp_base_hp, 0, u.atk / 10, 0]
                else:
                    d, k = combat_damage(u, v)
                    want[a, t] = [d >= v.hp, k >= u.hp, d / 10, k / 10]
            assert np.allclose(r.p.attack_preview, want, atol=1e-6)
            if choose.any():
                act = obs.pending.action
                if act != "buff" or "atk" in PendingView._fields:
                    wantc = np.zeros((SP.n_choose, 4))
                    for t in np.flatnonzero(choose):
                        wantc[t] = _choose_expect(obs, t, config=config)
                    assert np.allclose(r.p.choose_preview, wantc, atol=1e-6), obs.pending
                    tc = config.cards[obs.pending.card].effects[obs.pending.effect].target_condition
                    seen["target_condition"] += tc is not None
                seen[act] += 1
            else:
                assert not r.p.choose_preview.any()
            # the waiting player's encoding has no hints at all
            r1 = Read(enc.encode(g.observe(1 - p)), enc)
            hint_cols = [TF[k] for k in fmod.HINT_FEATURES + ("playable", "mulligan_legal")]
            assert not r1.p.features[:, hint_cols].any() and not r1.p.attack_preview.any()
            assert not r1.p.choose_preview.any() and r1.g("n_legal") == 0
            seen["phase", obs.phase] += 1
            seen["attacks"] += int(att.any())
    assert seen["phase", MULLIGAN] > 20 and seen["phase", MAIN] > 500 and seen["phase", CHOICE] > 20, seen
    assert seen["damage"] > 5 and seen["attacks"] > 200, seen
    assert seen["target_condition"] > 3 and seen["pin"] > 5, seen  # shipped Armor-Piercing Shot, else bodies


def test_hint_free_encoding_without_mask():
    for g in sample_states(CONFIG, 3, seed=9, every=3):
        p = g.current_player()
        obs = g.observe(p)
        r = Read(ENC.encode(obs), ENC)
        cols = [TF[k] for k in fmod.HINT_FEATURES + ("playable", "mulligan_legal")]
        assert not r.p.features[:, cols].any() and not r.p.attack_preview.any() and not r.p.choose_preview.any()
        with_mask = ENC.encode(obs, g.legal_mask())
        p_mask = ENC.split(with_mask)
        # identical apart from the mask-derived entries
        same = np.ones(len(TOKEN_FEATURES), bool)
        same[cols] = False
        assert np.array_equal(p_mask.features[:, same], r.p.features[:, same])
        g_same = [i for i, k in enumerate(GLOBAL_FEATURES) if k not in ("n_legal", "base_damage_ready")]
        assert np.array_equal(p_mask.globals[g_same], r.p.globals[g_same])


# ================================================================ revealed tokens and deck counts
def test_revealed_tokens_are_ascending_unique_and_truncated():
    g = blank_game(current=0, first=0, round_=3)
    b = next(c.index for c in CONFIG.cards.cards if c.is_unit and not c.token)
    add_unit(g, 1, "back", CONFIG.cards[b].id)  # on the board only: no public-presence record of its own
    g.invalidate()
    base = g.observe(0)
    n = N_CARDS
    assert n > R + 2
    rev, known, grave, disc = [0] * n, [0] * n, [0] * n, [0] * n
    present = []
    for c in range(n):
        if c != b:
            (rev, known, grave, disc)[c % 4][c] = 1 + c % 3
            present.append(c)
    obs = base._replace(opp_revealed=tuple(rev), opp_known_hand=tuple(known), opp_graveyard=tuple(grave),
                        opp_discard=tuple(disc))
    r = Read(ENC.encode(obs))
    kept = present[:R]
    for k, c in enumerate(kept):
        r.row_is(REV0 + k, {"type_revealed": 1, "revealed": rev[c] / 3, "known_in_hand": known[c] / 3,
                            "graveyard": grave[c] / 3, "discard": disc[c] / 3}, ident=c + 1)
    assert r.p.ids[REV0:REV0 + R].tolist() == [c + 1 for c in kept]
    # card b is on the opponent's board but has no public-presence record: not a revealed token;
    # once it is in the graveyard too, it takes its ascending place and carries its on-board count
    grave[b] = 1
    r = Read(ENC.encode(obs._replace(opp_graveyard=tuple(grave))))
    kept = sorted(present + [b])[:R]
    assert r.p.ids[REV0:REV0 + R].tolist() == [c + 1 for c in kept]
    k = kept.index(b)
    assert r.f(REV0 + k, "on_board") == pytest.approx(1 / 3) and r.f(REV0 + k, "graveyard") == pytest.approx(1 / 3)
    assert not r.p.features[REV0:REV0 + R, TF["on_board"]].sum() > 1 / 3 + 1e-6
    # few cards: the rest of the group is absent
    r = Read(ENC.encode(base._replace(opp_discard=tuple(1 if c == 5 else 0 for c in range(n)))))
    assert r.p.present[REV0:REV0 + R].tolist() == [True] + [False] * (R - 1) and int(r.p.ids[REV0]) == 6
    assert not r.p.features[REV0 + 1:REV0 + R].any()


def test_deck_counts_track_the_remaining_deck():
    g = Game(CONFIG)
    g.reset(4)
    for _ in range(12):
        p = g.current_player()
        obs = g.observe(p)
        x = ENC.encode(obs, g.legal_mask())
        assert ENC.split(x).deck_counts.tolist() == [float(k) for k in obs.my_deck_counts]
        assert ENC.split(x).deck_counts.sum() == obs.my_deck_size == len(g.deck_cards[p])
        g.step(SP.END_TURN)


# ================================================================ hidden information
def _perturb_hidden(game: Game, observer: int, rng: random.Random) -> Game:
    """Clone differing only in what `observer` cannot see: the opponent's unknown hand cards and deck
    (contents drawn from another fixed deck's cards, same sizes), both deck orders, the opponent's
    mulligan marks while they decide, and the RNG."""
    g = game.clone()
    o = 1 - observer
    known = Counter({c: k for c, k in enumerate(g.known_hand[o]) if k})
    hand = Counter(g.hands[o])
    unknown_n = len(g.hands[o]) - sum(known.values())
    assert (hand & known) == known
    pool = [c for d in g.config.decks for c in d if c not in known]
    rng.shuffle(pool)
    g.hands[o] = sorted(list(known.elements()) + pool[:unknown_n])
    g.deck_cards[o] = pool[unknown_n:unknown_n + len(g.deck_cards[o])]
    rng.shuffle(g.deck_cards[observer])
    if g.phase == MULLIGAN and g.current == o:
        g.mulligan_marks = {i for i in range(len(g.hands[o])) if rng.random() < 0.5}
    g.rng = random.Random(rng.getrandbits(64))
    g.invalidate()
    return g


@pytest.mark.parametrize("config", [CONFIG_M, RULES], ids=["shipped_mulligan", "effects"])
def test_encoding_ignores_hidden_information(config):
    enc = ObservationEncoder(config)
    rng = random.Random(17)
    checked = changed = 0
    for g in sample_states(config, 25, seed=21, every=2):
        for p in (0, 1):
            alt = _perturb_hidden(g, p, rng)
            assert alt.observe(p) == g.observe(p)
            m, m_alt = own_mask(g, p), own_mask(alt, p)
            if m is not None:
                assert np.array_equal(m, m_alt)
            assert np.array_equal(enc.encode(g.observe(p), m), enc.encode(alt.observe(p), m_alt))
            changed += alt.hands[1 - p] != g.hands[1 - p]
            checked += 1
    assert checked > 500 and changed > 0.3 * checked, (checked, changed)


# ================================================================ strictness, fingerprint, determinism
def test_mask_argument_is_strict():
    g = Game(CONFIG)
    g.reset(1)
    p = g.current_player()
    obs = g.observe(p)
    mask = g.legal_mask()
    for bad in (g.legal_actions(), mask.astype(np.int8), mask.astype(np.float32), mask[:-1], mask[None, :],
                list(mask), np.zeros(N + 1, bool)):
        with pytest.raises(TypeError):
            ENC.encode(obs, bad)
    with pytest.raises(ValueError, match="not the observer's turn"):
        ENC.encode(g.observe(1 - p), mask)
    # an all-False mask carries nothing, so it is accepted for the waiting player
    assert np.array_equal(ENC.encode(g.observe(1 - p), np.zeros(N, bool)), ENC.encode(g.observe(1 - p)))
    with pytest.raises(ValueError):
        ENC.encode_into(obs, np.zeros(ENC.dim - 1, np.float32), mask)


def test_mask_on_the_opponents_turn_is_rejected_and_would_leak():
    """The mover's mask reflects their hidden hand: encoding it for the waiting player would leak it."""
    cheap, _, dear = _cards_by_cost(CONFIG)
    encodings = []
    for hidden in ([cheap, cheap], [dear, dear]):
        g = blank_game(current=0, first=0, round_=3, coins=3)
        set_hand(g, 0, hidden)
        assert g.legal_mask().any()
        with pytest.raises(ValueError, match="not the observer's turn"):
            ENC.encode(g.observe(1), g.legal_mask())
        encodings.append(ENC.encode(g.observe(1)))
    assert np.array_equal(encodings[0], encodings[1])


def _replace_card(config, index, **changes):
    cards = list(config.cards.cards)
    cards[index] = dataclasses.replace(cards[index], **changes)
    return dataclasses.replace(config, cards=dataclasses.replace(config.cards, cards=tuple(cards)))


def test_fingerprint_sensitivity(monkeypatch):
    fp = pool_fingerprint(CONFIG)
    assert fp == ENC.fingerprint == pool_fingerprint(load_ruleset(mulligan=False))
    c0 = CONFIG.cards[0]
    changed = [
        _replace_card(CONFIG, 0, attack=c0.attack + 1),
        _replace_card(CONFIG, 0, move_cost=c0.move_cost + 1),
        _replace_card(CONFIG, 0, defense=not c0.defense),
        dataclasses.replace(CONFIG, decks=(CONFIG.decks[1],) + CONFIG.decks[1:]),
        dataclasses.replace(CONFIG, base_hp=CONFIG.base_hp + 5),
        dataclasses.replace(CONFIG, max_rounds=CONFIG.max_rounds + 1),
    ]
    fps = [pool_fingerprint(c) for c in changed]
    assert fp not in fps and len(set(fps)) == len(fps)
    # effect definitions count: same card ids and stats, a different amount
    zap = RULES.cards.by_id("zap")
    eff = dataclasses.replace(zap.effects[0], amount=dataclasses.replace(zap.effects[0].amount, value=4))
    assert pool_fingerprint(_replace_card(RULES, zap.index, effects=(eff,))) != pool_fingerprint(RULES)
    # ... including details the card table does not show (a filter's bounds)
    eff = zap.effects[0]
    variants = [_replace_card(RULES, zap.index, effects=(dataclasses.replace(
        eff, target=dataclasses.replace(eff.target, filter=FilterDef(min_cost=k))),)) for k in (3, 4)]
    assert np.array_equal(card_table(variants[0])[0], card_table(variants[1])[0])
    assert pool_fingerprint(variants[0]) != pool_fingerprint(variants[1])
    # gameplay switches that do not change a feature's meaning are left out
    assert pool_fingerprint(dataclasses.replace(CONFIG, mulligan=True)) == fp
    assert pool_fingerprint(dataclasses.replace(CONFIG, max_effect_events=9)) == fp
    # encoder version, schema and scales
    monkeypatch.setattr(fmod, "ENCODER_VERSION", 6)
    assert pool_fingerprint(CONFIG) != fp
    monkeypatch.setattr(fmod, "ENCODER_VERSION", 5)
    monkeypatch.setattr(fmod, "GLOBAL_FEATURES", fmod.GLOBAL_FEATURES + ("extra",))
    assert pool_fingerprint(CONFIG) != fp
    monkeypatch.setattr(fmod, "GLOBAL_FEATURES", GLOBAL_FEATURES)
    monkeypatch.setattr(fmod, "SCALES", {**fmod.SCALES, "atk": 12.0})
    assert pool_fingerprint(CONFIG) != fp


def test_fingerprint_tracks_tags():
    need("tags" in CardDef.__dataclass_fields__, "CardDef.tags")
    tagged = _replace_card(CONFIG, 0, tags=("tank",))
    assert pool_fingerprint(tagged) != pool_fingerprint(CONFIG)


def test_determinism_batch_and_rows():
    states = [(g.observe(g.current_player()), g.legal_mask()) for g in sample_states(RULES, 4, seed=2, every=4)]
    rows = np.stack([ENC_R.encode(o, m) for o, m in states])
    assert np.array_equal(rows, np.stack([ENC_R.encode(o, m) for o, m in states]))
    assert np.array_equal(ENC_R.encode_batch([o for o, _ in states], masks=[m for _, m in states]), rows)
    out = np.full((len(states) + 2, ENC_R.dim), 7.0, dtype=np.float32)
    ENC_R.encode_batch([o for o, _ in states], out=out, masks=[m for _, m in states])
    assert np.array_equal(out[:len(states)], rows)
    # a non-contiguous row (column of a Fortran-ordered buffer) gets the same values
    col = np.zeros((ENC_R.dim, 2), dtype=np.float32)[:, 1]
    assert not col.flags.c_contiguous
    ENC_R.encode_into(states[0][0], col, states[0][1])
    assert np.array_equal(col, rows[0])
    # an encoder built twice encodes identically
    assert np.array_equal(ObservationEncoder(RULES).encode(*states[-1]), rows[-1])


# ================================================================ static card table
def _table_row(enc, name):
    lay = enc.layout()
    names = lay["card_table_features"]
    row = lay["card_table"][enc.config.cards.by_id(name).index]
    return {k: v for k, v in zip(names, row) if v != 0.0}


def _approx(d):
    return {k: pytest.approx(v, abs=1e-6) for k, v in d.items()}


def test_card_table_vanilla_and_operation_rows():
    warden = next(c for c in CONFIG.cards.cards if c.defense and c.armor and c.nature == TROOP)
    want = {"is_unit": 1, "cost": warden.cost / 8, "atk": warden.attack / 10, "hp": warden.health / 10,
            "move_cost": warden.move_cost / 4, "troop": 1, "defense": 1, "armor": warden.armor / 4}
    assert _table_row(ENC, warden.id) == _approx(want)
    assert _table_row(ENC_R, "zap") == _approx({
        "is_operation": 1, "cost": 1 / 8, "e0_present": 1, "e0_trigger_on_play": 1, "e0_scope_self": 1,
        "e0_action_damage": 1, "e0_select_chosen": 1, "e0_side_any": 1, "e0_kind_unit_or_base": 1,
        "e0_zone_board": 1, "e0_amount": 0.3, "e0_repeat": 1 / 3})


def test_card_table_effect_slots():
    medic = vunit("medic", 2, 3, cost=3, traits={"defense": True, "armor": 1}, effects=[
        {"trigger": "on_deploy", "action": "heal", "amount": "full",
         "target": {"select": "chosen", "side": "friendly", "kind": "unit", "filter": {"damaged": True}}},
        {"trigger": "end_of_turn", "action": "buff", "atk": {"count": "units", "side": "friendly"}, "hp": 1,
         "duration": "turn", "condition": {"type": "control", "side": "enemy"},
         "target": {"select": "random", "side": "friendly", "kind": "unit", "zone": "backline", "count": 2}},
        {"trigger": "on_attack", "action": "remove_trait", "trait": ["defense", "armor"], "target": "event"}])
    conscript = vunit("conscript", 1, 1, token=True)
    caller = vunit("caller", 1, 2, nature="ranged", cost=4, effects=[
        {"trigger": "on_death", "action": "summon", "card": "conscript", "amount": 2, "target": "controller"},
        {"trigger": "on_move", "action": "add_trait", "trait": "armor", "amount": 2, "target": "self"}])
    enc = ObservationEncoder(effect_rules([medic, conscript, caller]))
    assert _table_row(enc, "medic") == _approx({
        "is_unit": 1, "cost": 3 / 8, "atk": 0.2, "hp": 0.3, "move_cost": 0.25, "troop": 1, "defense": 1,
        "armor": 0.25,
        "e0_present": 1, "e0_trigger_on_deploy": 1, "e0_scope_self": 1, "e0_action_heal": 1, "e0_select_chosen": 1,
        "e0_side_friendly": 1, "e0_kind_unit": 1, "e0_zone_board": 1, "e0_amount_full": 1, "e0_has_filter": 1,
        "e0_repeat": 1 / 3,
        "e1_present": 1, "e1_trigger_end_of_turn": 1, "e1_scope_friendly": 1, "e1_action_buff": 1,
        "e1_select_random": 1, "e1_side_friendly": 1, "e1_kind_unit": 1, "e1_zone_backline": 1,
        "e1_amount_is_expr": 1, "e1_buff_hp": 0.1, "e1_duration_turn": 1, "e1_has_condition": 1,
        "e1_count": 2 / 3, "e1_repeat": 1 / 3,
        "e2_present": 1, "e2_trigger_on_attack": 1, "e2_scope_self": 1, "e2_action_remove_trait": 1,
        "e2_select_event": 1, "e2_kind_unit": 1, "e2_zone_board": 1, "e2_trait_defense": 1, "e2_trait_armor": 1,
        "e2_repeat": 1 / 3})
    assert _table_row(enc, "conscript") == _approx({"is_unit": 1, "token": 1, "cost": 1 / 8, "atk": 0.1, "hp": 0.1,
                                                    "move_cost": 0.25, "troop": 1})
    assert _table_row(enc, "caller") == _approx({
        "is_unit": 1, "cost": 0.5, "atk": 0.1, "hp": 0.2, "move_cost": 0.25, "ranged": 1,
        "e0_present": 1, "e0_trigger_on_death": 1, "e0_scope_self": 1, "e0_action_summon": 1, "e0_select_all": 1,
        "e0_side_friendly": 1, "e0_kind_player": 1, "e0_zone_board": 1, "e0_amount": 0.2, "e0_repeat": 1 / 3,
        "e1_present": 1, "e1_trigger_on_move": 1, "e1_scope_self": 1, "e1_action_add_trait": 1, "e1_select_self": 1,
        "e1_kind_unit": 1, "e1_zone_board": 1, "e1_amount": 0.2, "e1_trait_armor": 1, "e1_repeat": 1 / 3})


def test_card_table_schema_extensions():
    """SPEC §1.2b effect fields: static buff with move_cost, target_condition + else, event_filter, repeat,
    uncapped heal."""
    extra = [
        op("surge", [{"trigger": "on_play", "action": "heal", "amount": 5, "uncapped": True,
                      "target": {"select": "chosen", "side": "friendly", "kind": "base"}}]),
        op("volley", [{"trigger": "on_play", "action": "damage", "amount": 1, "repeat": 3,
                       "target": {"select": "random", "side": "enemy", "kind": "unit"}}]),
        vunit("marshal", 2, 3, cost=4, effects=[
            {"trigger": "static", "action": "buff", "atk": 1, "move_cost": -1,
             "target": {"select": "all", "side": "friendly", "kind": "unit"}},
            {"trigger": "on_deploy", "action": "damage", "amount": 2, "target_condition": {"damaged": True},
             "else": {"action": "pin"}, "target": {"select": "chosen", "side": "enemy", "kind": "unit"}},
            {"trigger": "on_deploy", "scope": "friendly", "event_filter": {"nature": "troop"}, "action": "buff",
             "atk": 1, "target": "event"}])]
    rules = effect_rules(extra)
    enc = ObservationEncoder(rules)
    common = {"e0_present": 1, "e0_trigger_on_play": 1, "e0_scope_self": 1, "e0_zone_board": 1, "is_operation": 1,
              "cost": 1 / 8}
    assert _table_row(enc, "surge") == _approx({**common, "e0_action_heal": 1, "e0_select_chosen": 1,
                                                "e0_side_friendly": 1, "e0_kind_base": 1, "e0_amount": 0.5,
                                                "e0_uncapped": 1, "e0_repeat": 1 / 3})
    assert _table_row(enc, "volley") == _approx({**common, "e0_action_damage": 1, "e0_select_random": 1,
                                                 "e0_side_enemy": 1, "e0_kind_unit": 1, "e0_amount": 0.1,
                                                 "e0_repeat": 1.0, "e0_count": 1 / 3})
    assert _table_row(enc, "marshal") == _approx({
        "is_unit": 1, "cost": 0.5, "atk": 0.2, "hp": 0.3, "move_cost": 0.25, "troop": 1,
        "e0_present": 1, "e0_trigger_static": 1, "e0_scope_self": 1, "e0_action_buff": 1, "e0_select_all": 1,
        "e0_side_friendly": 1, "e0_kind_unit": 1, "e0_zone_board": 1, "e0_buff_atk": 0.1, "e0_buff_move_cost": -0.25,
        "e0_repeat": 1 / 3,
        "e1_present": 1, "e1_trigger_on_deploy": 1, "e1_scope_self": 1, "e1_action_damage": 1, "e1_select_chosen": 1,
        "e1_side_enemy": 1, "e1_kind_unit": 1, "e1_zone_board": 1, "e1_amount": 0.2, "e1_has_filter": 1,
        "e1_has_else": 1, "e1_repeat": 1 / 3,
        "e2_present": 1, "e2_trigger_on_deploy": 1, "e2_scope_friendly": 1, "e2_action_buff": 1, "e2_select_event": 1,
        "e2_kind_unit": 1, "e2_zone_board": 1, "e2_buff_atk": 0.1, "e2_has_event_filter": 1, "e2_repeat": 1 / 3})
    # an uncapped heal previews past base_hp; the capped one stops at it
    for name, healed in (("surge", 0.5), ("mend", 0.2)):
        g = blank_game(current=0, first=0, round_=5, coins=10, config=rules)
        set_hand(g, 0, [name])
        g.base_hp = [18, 20]
        g.invalidate()
        g.step(SP.PLAY0)
        r, obs, choose = _check_choice(g, enc)
        assert r.p.choose_preview[3 * Z + 1].tolist() == pytest.approx([0, 0, healed, 0]), name


def test_card_table_new_traits_and_tags():
    need("tags" in CardDef.__dataclass_fields__, "CardDef.tags")
    try:
        rules = effect_rules([dict(vunit("panzer", 4, 4, cost=5, traits={"ambush": True, "shock": True,
                                                                          "immune": True}), tags=["tank", "germany"]),
                              dict(vunit("rifle", 1, 2), tags=["infantry", "germany"])])
    except ValueError as e:
        pytest.skip(f"card loader does not accept tags / new traits yet: {e}")
    enc = ObservationEncoder(rules)
    assert enc.layout()["tags"] == ["germany", "infantry", "tank"]
    row = _table_row(enc, "panzer")
    assert row["ambush"] == row["shock"] == row["immune"] == 1
    assert row["tag_tank"] == row["tag_germany"] == 1 and "tag_infantry" not in row
    assert _table_row(enc, "rifle")["tag_infantry"] == 1


def test_unknown_effect_vocabulary_is_rejected():
    zap = RULES.cards.by_id("zap")
    bogus = dataclasses.replace(zap.effects[0], trigger="on_full_moon")
    with pytest.raises(ValueError, match="on_full_moon"):
        ObservationEncoder(_replace_card(RULES, zap.index, effects=(bogus,)))


def test_new_unit_view_fields_are_encoded():
    """ambush / shock / immune / ambush_ready (SPEC §5) land in the unit columns of the same name."""
    need(all(k in UnitView._fields for k in NEW_UNIT_FIELDS), "UnitView ambush/shock/immune/ambush_ready")
    g = blank_game(current=0, first=0, round_=2)
    add_unit(g, 1, "back", atk=1, hp=1)
    obs = g.observe(0)
    u = obs.opp_backline[0]._replace(ambush=True, shock=True, immune=True, ambush_ready=True)
    r = Read(ENC.encode(obs._replace(opp_backline=(u,))))
    assert [r.f(OPPB0, k) for k in NEW_UNIT_FIELDS] == [1, 1, 1, 1]


# ================================================================ SPEC §5 observation fields (review finding 5)
def test_history_globals_count_this_turn_and_this_game():
    """my_history / opp_history = (operations played, units deployed, units died) this turn, then this game
    (public, SPEC §5); globals scale the turn counters /5 and the game counters /20."""
    g = blank_game(current=0, first=0, round_=5, coins=10, config=RULES)
    set_hand(g, 0, ["zap", "f00"])
    add_unit(g, 1, "back", "f02", atk=2, hp=3, max_hp=3)
    g.step(SP.PLAY0 + g.hands[0].index(cid(RULES, "zap")))
    g.step(SP.choose(0))  # 3 damage kills the 2/3
    g.step(SP.PLAY0)      # deploy f00
    assert (g.observe(0).my_history, g.observe(0).opp_history) == ((1, 1, 0, 1, 1, 0), (0, 0, 1, 0, 0, 1))
    assert (g.observe(1).my_history, g.observe(1).opp_history) == ((0, 0, 1, 0, 0, 1), (1, 1, 0, 1, 1, 0))

    def globals_of(p):
        r = Read(ENC_R.encode(g.observe(p), own_mask(g, p)), ENC_R)
        return [round(r.g(k) * (5 if k.endswith("_turn") else 20), 6) for k in HISTORY_NAMES]

    assert globals_of(0) == [1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1]
    assert globals_of(1) == [0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0]
    g.step(SP.END_TURN)  # the turn counters reset at the next turn start, the game counters stay
    assert g.observe(0).my_history == (0, 0, 0, 1, 1, 0) and g.observe(0).opp_history == (0, 0, 0, 0, 0, 1)
    assert globals_of(1) == [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0]
    # an observation without the counters (default ()) encodes zeros there
    r = Read(ENC_R.encode(g.observe(0)._replace(my_history=(), opp_history=())), ENC_R)
    assert all(r.g(k) == 0 for k in HISTORY_NAMES)


def test_history_counters_are_kept_without_history_cards():
    """The counters are public observation fields, so the engine keeps them on pools whose cards never read
    them (the shipped pool has no history condition)."""
    g = Game(CONFIG)
    assert not any(cd.type == "history" for c in CONFIG.cards.cards for e in c.effects for cd in e.condition)
    g.reset(5)
    rng = random.Random(5)
    played, died = [0, 0], [0, 0]
    while not g.done and g.round < 12:
        p = g.current_player()
        a = rng.choice(g.legal_actions())
        before = [sum(gy) for gy in g.graveyard]
        if SP.PLAY0 <= a < SP.MOVE0:
            played[p] += 1
        g.step(a)
        for q in (0, 1):
            died[q] += sum(g.graveyard[q]) - before[q]
    for q in (0, 1):
        h = g.observe(q).my_history
        assert h[3] + h[4] == played[q] == sum(g.played[q]) and h[5] == died[q], (q, h)
        assert h[3] == sum(k for c, k in enumerate(g.played[q]) if CONFIG.cards[c].is_operation)
    assert sum(played) > 10 and sum(died) > 3


def test_pin_turns_and_turn_changes_are_observed_and_encoded():
    """UnitView.pin_turns, temp_move_cost, temp_traits / temp_removed (SPEC §5) along a pin's life and a turn
    of "turn" effects; the encoder turns them into pin_turns/2, temp_move_cost/4 and the signed per-trait
    lapse flags. Everything lapses at END_TURN."""
    def chosen_turn(action, side, **kw):
        return chosen(action, side, duration="turn", **kw)

    extra = [op("drill", [chosen_turn("add_trait", "friendly", trait="smokescreen")]),
             op("strip", [chosen_turn("remove_trait", "enemy", trait="defense")]),
             op("mud", [chosen_turn("buff", "enemy", move_cost=2)]),
             op("plate", [chosen_turn("add_trait", "friendly", trait="armor", amount=2)]),
             op("rust", [chosen_turn("remove_trait", "enemy", trait="armor")])]
    rules = effect_rules(extra)
    enc = ObservationEncoder(rules)
    g = blank_game(current=0, first=0, round_=5, coins=10, config=rules)
    T = g.turn
    add_unit(g, 0, "back", "f00", atk=2, hp=2)
    add_unit(g, 0, "back", "f01", atk=1, hp=1, pinned=True, pin_until=T + 2)  # as if pinned on my turn
    add_unit(g, 1, "back", "f02", atk=2, hp=3, defense=True)
    tank = add_unit(g, 1, "back", "f03", atk=2, hp=4, armor=1)
    for name, slot in (("drill", 2 * Z + 1), ("strip", 0), ("mud", 0), ("plate", 2 * Z + 1), ("rust", 1),
                       ("snare", 1)):
        set_hand(g, 0, [name])
        g.step(SP.PLAY0)
        g.step(SP.choose(slot))
    assert tank.pinned and tank.pin_until == T + 1  # pinned on its owner's opponent's turn
    S, A, D = TRAIT_BITS["smokescreen"], TRAIT_BITS["armor"], TRAIT_BITS["defense"]

    def views(p):
        obs = g.observe(p)
        mine, theirs = (obs.my_backline, obs.opp_backline)
        return (mine, theirs) if p == 0 else (theirs, mine)

    for p in (0, 1):  # public: both players see the same unit fields
        (v_own, v_held), (v_wall, v_tank) = views(p)
        assert (v_own.temp_traits, v_own.temp_removed, v_own.armor) == (S | A, 0, 2)
        assert (v_wall.temp_traits, v_wall.temp_removed, v_wall.temp_move_cost, v_wall.defense) == (0, D, 2, False)
        assert (v_tank.temp_removed, v_tank.armor, v_tank.pinned, v_tank.pin_turns) == (A, 0, True, 1)
        assert (v_held.pinned, v_held.pin_turns) == (True, 2)
        obs = g.observe(p)
        r = Read(enc.encode(obs, own_mask(g, p)), enc)
        toks = (MYB0, MYB0 + 1, OPPB0, OPPB0 + 1) if p == 0 else (OPPB0, OPPB0 + 1, MYB0, MYB0 + 1)
        t_own, t_held, t_wall, t_tank = toks
        assert (r.f(t_own, "lapse_smokescreen"), r.f(t_own, "lapse_armor"), r.f(t_own, "lapse_defense")) == (1, 1, 0)
        assert r.f(t_wall, "lapse_defense") == -1 and r.f(t_wall, "temp_move_cost") == pytest.approx(0.5)
        assert r.f(t_tank, "lapse_armor") == -1 and r.f(t_tank, "pin_turns") == pytest.approx(0.5)
        assert r.f(t_held, "pin_turns") == pytest.approx(1.0) and r.f(t_held, "lapse_armor") == 0
        units = (*obs.my_backline, *obs.opp_backline) if p == 0 else (*obs.opp_backline, *obs.my_backline)
        for tok, u in zip(toks, units):  # every unit-state column, from the view (SPEC §6 scales)
            assert {k: r.f(tok, k) for k in unit_expect(u)} == {k: pytest.approx(float(v), abs=1e-6)
                                                                  for k, v in unit_expect(u).items()}, tok
    g.step(SP.END_TURN)  # T + 1 (P1): every "turn" change lapses; the pins still cover one owner turn each
    for p in (0, 1):
        (v_own, v_held), (v_wall, v_tank) = views(p)
        assert all((v.temp_traits, v.temp_removed, v.temp_move_cost) == (0, 0, 0) for v in (v_own, v_wall, v_tank))
        assert (v_own.smokescreen, v_own.armor, v_wall.defense, v_tank.armor) == (False, 0, True, 1)
        assert (v_tank.pin_turns, v_held.pin_turns) == (1, 1)
        r = Read(enc.encode(g.observe(p), own_mask(g, p)), enc)
        assert not r.p.features[:, [TF[f"lapse_{t}"] for t in TRAIT_NAMES] + [TF["temp_move_cost"]]].any()
    g.step(SP.END_TURN)  # T + 2 (P0): the tank's pin is lifted at the end of T + 1; mine covers this turn only
    (v_own, v_held), (v_wall, v_tank) = views(0)
    assert (v_tank.pinned, v_tank.pin_turns, v_held.pinned, v_held.pin_turns) == (False, 0, True, 1)
    g.step(SP.END_TURN)
    assert not g.observe(0).my_backline[1].pinned and g.observe(0).my_backline[1].pin_turns == 0


def test_new_observation_unit_fields_are_encoded():
    """pin_turns /2, temp_move_cost /4 and the signed lapse_<trait> flags read UnitView fields only."""
    g = blank_game(current=0, first=0, round_=2)
    add_unit(g, 1, "back", atk=1, hp=1)
    obs = g.observe(0)
    u = obs.opp_backline[0]._replace(pin_turns=2, temp_move_cost=-1,
                                     temp_traits=TRAIT_BITS["defense"] | TRAIT_BITS["armor"],
                                     temp_removed=TRAIT_BITS["fury"] | TRAIT_BITS["immune"])
    r = Read(ENC.encode(obs._replace(opp_backline=(u,))))
    assert r.f(OPPB0, "pin_turns") == 1.0 and r.f(OPPB0, "temp_move_cost") == pytest.approx(-0.25)
    lapse = {t: r.f(OPPB0, f"lapse_{t}") for t in TRAIT_NAMES}
    assert lapse == {"defense": 1, "armor": 1, "blitz": 0, "smokescreen": 0, "fury": -1, "ambush": 0, "shock": 0,
                     "immune": -1}
    r.row_is(OPPB0, {"type_opp_back": 1, **unit_expect(u)}, ident=u.card + 1)


def test_fingerprint_covers_the_observation_schema(monkeypatch):
    """ENCODER_VERSION stays 5 while unreleased, so the fingerprint must move with every schema change,
    including the engine views the encoder reads."""
    fp = pool_fingerprint(CONFIG)
    for name, nt in (("Observation", Observation), ("UnitView", UnitView), ("PendingView", PendingView)):
        monkeypatch.setattr(fmod, name, namedtuple(name, nt._fields + ("extra",)))
        assert pool_fingerprint(CONFIG) != fp, name
        monkeypatch.setattr(fmod, name, nt)
    assert pool_fingerprint(CONFIG) == fp
    monkeypatch.setattr(fmod, "TOKEN_FEATURES", fmod.TOKEN_FEATURES[:-1])
    assert pool_fingerprint(CONFIG) != fp
