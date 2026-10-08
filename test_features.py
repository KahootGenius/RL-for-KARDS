"""ObservationEncoder values checked against an independent re-implementation of its documented layout."""
from __future__ import annotations

import numpy as np

from conftest import CONFIG, add_unit, blank_game, card, iter_states, set_hand
from cardgame.actions import ActionSpace
from cardgame.features import ObservationEncoder

ENC = ObservationEncoder(CONFIG)
PLAY0 = ActionSpace(CONFIG.max_hand_size, CONFIG.zone_capacity).PLAY0


def ref_encode(obs, legal=None) -> np.ndarray:
    """Written from the ObservationEncoder docstring, not from its code."""
    cards = CONFIG.cards.cards
    n, H, Z = len(cards), CONFIG.max_hand_size, CONFIG.zone_capacity
    ia, ih, ic = 1 / max(c.attack for c in cards), 1 / max(c.health for c in cards), 1 / max(c.cost for c in cards)
    mine = list(obs.my_backline) + (list(obs.frontline) if obs.front_owner > 0 else [])
    theirs = list(obs.opp_backline) + (list(obs.frontline) if obs.front_owner < 0 else [])
    row = [obs.is_my_turn, obs.went_first, obs.player == 0, obs.player == 1, obs.round / CONFIG.max_rounds,
           obs.my_coins / 10, obs.opp_coins / 10, obs.my_base_hp / CONFIG.base_hp, obs.opp_base_hp / CONFIG.base_hp,
           len(obs.hand) / H, obs.opp_hand_size / H, obs.my_deck_size / CONFIG.deck_size,
           obs.opp_deck_size / CONFIG.deck_size, obs.front_owner > 0, obs.front_owner == 0, obs.front_owner < 0,
           len(obs.my_backline) / Z, len(obs.frontline) / Z, len(obs.opp_backline) / Z,
           sum(u.atk for u in mine) * ia * 0.2, sum(u.hp for u in mine) * ih * 0.2,
           sum(u.atk for u in theirs) * ia * 0.2, sum(u.hp for u in theirs) * ih * 0.2]
    for i in range(H):
        slot = [0.0] * (5 + n)
        if i < len(obs.hand):
            c = cards[obs.hand[i]]
            playable = legal is not None and bool(legal[PLAY0 + i])
            slot[:5] = [1, c.cost * ic, c.attack * ia, c.health * ih, playable]
            slot[5 + c.index] = 1
        row += slot
    counts = [0.0] * n
    for c in obs.hand:
        counts[c] += 0.25
    row += counts
    for zone in (obs.my_backline, obs.frontline, obs.opp_backline):
        for j in range(Z):
            slot = [0.0] * (4 + n)
            if j < len(zone):
                u = zone[j]
                slot[:4] = [1, u.atk * ia, u.hp * ih, u.ready]
                slot[4 + u.card] = 1
            row += slot
    return np.asarray(row, dtype=np.float32)


def test_dimension_matches_layout():
    assert ENC.dim == len(ref_encode(blank_game().observe(0)))


def test_encoder_matches_documented_layout():
    n = 0
    for seed in range(20):
        for i, (game, _, _) in enumerate(iter_states(seed)):
            if i % 7:
                continue
            for p in (0, 1):
                obs = game.observe(p)
                legal = game.legal_mask() if p == game.current_player() and not game.done else None
                assert np.allclose(ENC.encode(obs, legal), ref_encode(obs, legal), atol=1e-6), (seed, i, p)
                n += 1
    assert n > 500


def test_batch_encoding_matches_rows():
    states = [g.clone() for i, (g, _, _) in enumerate(iter_states(3)) if i % 5 == 0 and not g.done]
    obs = [g.observe(g.current_player()) for g in states]
    legals = [g.legal_mask() for g in states]
    batch = ENC.encode_batch(obs, legals=legals)
    assert np.array_equal(batch, np.stack([ENC.encode(o, m) for o, m in zip(obs, legals)]))


def test_playable_bit_follows_the_engine_mask():
    g = blank_game(coins=10)
    for _ in range(CONFIG.zone_capacity):
        add_unit(g, 0, "back", "squire")
    set_hand(g, 0, ["squire"])
    x = ENC.encode(g.observe(0), g.legal_mask())
    assert x[ENC.off_hand] == 1.0 and x[ENC.off_hand + 4] == 0.0  # present, but the backline is full
    g.backline[0].pop()
    g.invalidate()
    x = ENC.encode(g.observe(0), g.legal_mask())
    assert x[ENC.off_hand + 4] == 1.0
    assert ENC.encode(g.observe(0))[ENC.off_hand + 4] == 0.0  # no mask given: never claims playability


def test_hand_counts_and_ready_flags():
    g = blank_game(coins=0)
    set_hand(g, 0, ["squire", "squire", "giant"])
    add_unit(g, 0, "back", "knight", ready=True)
    add_unit(g, 0, "back", "ogre", ready=False)
    x = ENC.encode(g.observe(0))
    assert x[ENC.off_counts + card("squire")] == 0.5 and x[ENC.off_counts + card("giant")] == 0.25
    uw = ENC.unit_width
    assert x[ENC.off_my_back + 3] == 1.0 and x[ENC.off_my_back + uw + 3] == 0.0
    assert x[ENC.off_my_back + 4 + card("knight")] == 1.0
