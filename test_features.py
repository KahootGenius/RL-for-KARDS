"""ObservationEncoder (Stage 2) against an independent re-implementation of SPEC §6."""
from __future__ import annotations

import dataclasses
import random

import numpy as np
import pytest

from cardgame.actions import ActionSpace
from cardgame.cards import FAST, RANGED, TROOP, load_ruleset
from cardgame.engine import Game
from cardgame.features import ObservationEncoder, pool_fingerprint
from conftest import blank_game, set_hand

CONFIG = load_ruleset()
ENC = ObservationEncoder(CONFIG)
SP = ActionSpace(CONFIG.max_hand_size, CONFIG.zone_capacity)
H, Z = CONFIG.max_hand_size, CONFIG.zone_capacity


def ref_encode(obs, mask=None) -> np.ndarray:
    """Written from SPEC §6 (fixed scales, slot order, mask-derived hints), not from features.py."""
    cards = CONFIG.cards.cards
    n = len(cards)
    att = None if mask is None else mask[SP.ATTACK0:].reshape(2 * Z, 2 * Z + 1)
    fo = obs.front_owner
    attackers = list(obs.my_backline) + [None] * (Z - len(obs.my_backline))
    attackers += list(obs.frontline) if fo > 0 else []
    atk = [u.atk if u is not None else 0 for u in attackers] + [0] * (2 * Z - len(attackers))

    def incoming(t):  # combined attack of my units with a legal attack on target slot t
        return sum(atk[a] for a in range(2 * Z) if att[a, t])

    g = [obs.is_my_turn, obs.went_first, obs.round / 50, obs.my_coins / 10, obs.opp_coins / 10,
         obs.my_base_hp / 20, obs.opp_base_hp / 20, len(obs.hand) / 10, obs.opp_hand_size / 10,
         obs.my_deck_size / 40, obs.opp_deck_size / 40, fo > 0, fo == 0, fo < 0,
         0.0 if att is None else att[:, 2 * Z].any(), 0.0 if mask is None else mask.sum() / SP.n,
         0.0 if att is None else incoming(2 * Z) / 20]
    deck = [cards[i] for i in CONFIG.decks[obs.my_deck]]
    k = len(deck)
    hist = [0.0] * 8
    for c in deck:
        hist[min(c.cost, 8) - 1] += 1 / k
    g += [sum(c.cost for c in deck) / k / 8, sum(c.attack for c in deck) / k / 10,
          sum(c.health for c in deck) / k / 10, sum(c.move_cost for c in deck) / k / 4,
          sum(c.nature == TROOP for c in deck) / k, sum(c.nature == FAST for c in deck) / k,
          sum(c.nature == RANGED for c in deck) / k, sum(c.defense for c in deck) / k,
          sum(c.armor > 0 for c in deck) / k, sum(c.armor for c in deck) / k / 4] + hist
    g += [x / 3 for x in obs.my_played] + [x / 3 for x in obs.opp_played]
    E, F = H + 3 * Z, 27
    feats = np.zeros((E, F), dtype=np.float64)
    ids = np.zeros(E)
    present = np.zeros(E)

    def static(c):
        return [1, c.cost / 8, c.attack / 10, c.health / 10, c.health / 10, c.move_cost / 4, c.nature == TROOP,
                c.nature == FAST, c.nature == RANGED, c.defense, c.armor > 0, c.armor / 4]

    for i, card in enumerate(obs.hand):
        row = static(cards[card]) + [0] * 5 + [mask is not None and mask[SP.PLAY0 + i], 0, 0, 0, 0, 1, 0, 0, 0, 1]
        feats[i], ids[i], present[i] = row, card + 1, 1
    zones = [(obs.my_backline, H, 0, True), (obs.frontline, H + Z, 1, fo > 0), (obs.opp_backline, H + 2 * Z, 2, False)]
    for units, first, zi, mine in zones:
        for j, u in enumerate(units):
            c = cards[u.card]
            row = [1, c.cost / 8, u.atk / 10, u.max_hp / 10, u.hp / 10, u.move_cost / 4, u.nature == TROOP,
                   u.nature == FAST, u.nature == RANGED, u.defense, u.armor > 0, u.armor / 4,
                   u.summoned, u.moved, u.attacked, u.can_move, u.can_attack, 0,
                   mask is not None and zi == 0 and mask[SP.MOVE0 + j],
                   att is not None and ((zi == 0 and att[j].any()) or (zi == 1 and mine and att[Z + j].any())),
                   att is not None and ((zi == 2 and att[:, j].any()) or (zi == 1 and not mine and att[:, Z + j].any())),
                   0 if att is None else (incoming(j) / 10 if zi == 2 else incoming(Z + j) / 10 if zi == 1 and not mine else 0),
                   0, zi == 0, zi == 1, zi == 2, mine]
            feats[first + j], ids[first + j], present[first + j] = row, u.card + 1, 1
    # attack previews, from SPEC §2 combat: armor reduces each hit, ranged takes no return damage
    previews = np.zeros((2 * Z, 2 * Z + 1, 4))
    targets = list(obs.opp_backline) + [None] * (Z - len(obs.opp_backline))
    targets += list(obs.frontline) if fo < 0 else []
    att_units = list(obs.my_backline) + [None] * (Z - len(obs.my_backline))
    att_units += list(obs.frontline) if fo > 0 else []
    if att is not None:
        for a, t in zip(*np.nonzero(att)):
            u = att_units[a]
            if t == 2 * Z:
                previews[a, t] = [u.atk >= obs.opp_base_hp, 0, u.atk / 10, 0]
            else:
                v = targets[t]
                dealt = max(0, u.atk - v.armor)
                taken = 0 if u.nature == RANGED else max(0, v.atk - u.armor)
                previews[a, t] = [dealt >= v.hp, taken >= u.hp, dealt / 10, taken / 10]
    return np.concatenate([np.asarray(g, float), feats.ravel(), previews.ravel(), ids, present]).astype(np.float32)


def sample_states(n_games=25, seed=0):
    rng = random.Random(seed)
    g = Game(CONFIG)
    for s in range(n_games):
        g.reset(s)
        step = 0
        while not g.done:
            if step % 3 == 0:
                yield g
            la = g.legal_actions()
            # bias towards board-building so zones fill up
            g.step(la[-1] if rng.random() < 0.3 else la[rng.randrange(len(la))])
            step += 1


def test_dimension_and_layout():
    obs = Game(CONFIG)
    obs.reset(0)
    assert ENC.dim == len(ref_encode(obs.observe(0)))
    lay = ENC.layout()
    assert lay["dim"] == ENC.dim and lay["n_actions"] == SP.n and lay["fingerprint"] == pool_fingerprint(CONFIG)


def test_encoder_matches_spec_reference():
    n = 0
    for g in sample_states():
        for p in (0, 1):
            obs = g.observe(p)
            mask = g.legal_mask() if p == g.current_player() else None
            assert np.allclose(ENC.encode(obs, mask), ref_encode(obs, mask), atol=1e-6), (g.seed, p)
            n += 1
    assert n > 300


def test_split_recovers_parts_and_batch_matches_rows():
    states = [(g.observe(g.current_player()), g.legal_mask()) for g in sample_states(5)]
    batch = ENC.encode_batch([o for o, _ in states], masks=[m for _, m in states])
    assert np.array_equal(batch, np.stack([ENC.encode(o, m) for o, m in states]))
    parts = ENC.split(batch)
    assert parts.features.shape == (len(states), ENC.E, ENC.F)
    assert (parts.ids[~parts.mask] == 0).all() and (parts.ids[parts.mask] >= 1).all()


def test_mask_argument_is_strict():
    g = Game(CONFIG)
    g.reset(1)
    obs = g.observe(g.current_player())
    for bad in (g.legal_actions(), g.legal_mask().astype(np.int8), g.legal_mask()[:-1]):
        with pytest.raises(TypeError):
            ENC.encode(obs, bad)


def test_mask_on_the_opponents_turn_is_rejected():
    """The to-move player's mask reflects their hidden hand: encoding it for the waiting player would leak it."""
    encodings = []
    for hidden in (["militia", "footman"], ["paladin", "bastion"]):  # same size; only the first is affordable
        g = blank_game(current=0, first=0, round_=5, coins=5)
        set_hand(g, 0, hidden)
        assert g.legal_mask().any()
        with pytest.raises(ValueError, match="not the observer's turn"):
            ENC.encode(g.observe(1), g.legal_mask())
        encodings.append(ENC.encode(g.observe(1)))
    assert np.array_equal(encodings[0], encodings[1])


def test_fingerprint_tracks_cards_and_decks():
    other_card = dataclasses.replace(CONFIG.cards.cards[0], attack=CONFIG.cards.cards[0].attack + 1)
    cards = dataclasses.replace(CONFIG.cards, cards=(other_card,) + CONFIG.cards.cards[1:])
    assert pool_fingerprint(dataclasses.replace(CONFIG, cards=cards)) != ENC.fingerprint
    decks = (CONFIG.decks[1],) + CONFIG.decks[1:]
    assert pool_fingerprint(dataclasses.replace(CONFIG, decks=decks)) != ENC.fingerprint
    assert pool_fingerprint(load_ruleset()) == ENC.fingerprint
