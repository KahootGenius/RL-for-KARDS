"""Deck balance gate (SPEC section 1) and the tools/deck_balance.py bookkeeping it relies on."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import deck_balance as db  # noqa: E402

from cardgame.agents import GreedyAgent, RandomAgent  # noqa: E402
from cardgame.cards import FAST, RANGED, load_ruleset  # noqa: E402
from cardgame.engine import DRAW, Game  # noqa: E402

CONFIG = load_ruleset()
GATE_DEALS = 4096       # greedy vs greedy: 512 games per off-diagonal P cell (SE ~0.022)
GVR_DEALS = 2048        # greedy vs random: 4096 games, 512 per unordered cross cell (256 per mirror)
RUN_SLOW = os.environ.get("CARDGAME_SLOW") == "1"


@pytest.fixture(scope="module")
def gate_results():
    gvg = db.run_match(CONFIG, "greedy", "greedy", GATE_DEALS)
    gvr = db.run_match(CONFIG, "greedy", "random", GVR_DEALS)
    return gvg, gvr


def test_deck_balance_gate(gate_results):
    gvg, gvr = gate_results
    print("\n" + gvg.format() + "\n" + gvr.format())
    assert db.gate_failures(gvg, gvr) == []


def test_deck_balance_has_margin(gate_results):
    """Shipped decks aim at P in [0.40, 0.60] so that the gate is not passed by luck."""
    gvg, _ = gate_results
    n = gvg.n_decks
    worst = max(abs(gvg.P[a][b].win_rate - 0.5) for a in range(n) for b in range(n) if a != b)
    assert worst <= 0.10, gvg.format()


def test_shipped_decks_keep_their_styles():
    """SPEC section 1: aggro >= 50% fast, defensive >= 40% Defense, ranged-heavy >= 40% ranged,
    balanced >= 20% of each nature; 40 cards, at most 3 copies."""
    cards = CONFIG.cards
    assert CONFIG.n_decks == 4
    for deck in CONFIG.decks:
        assert len(deck) == CONFIG.deck_size == 40
        assert max(deck.count(c) for c in set(deck)) <= 3
    frac = [{"fast": sum(cards[c].nature == FAST for c in d) / 40,
             "ranged": sum(cards[c].nature == RANGED for c in d) / 40,
             "troop": sum(cards[c].nature not in (FAST, RANGED) for c in d) / 40,
             "defense": sum(cards[c].defense for c in d) / 40} for d in CONFIG.decks]
    assert frac[0]["fast"] >= 0.5
    assert frac[1]["defense"] >= 0.4
    assert frac[2]["ranged"] >= 0.4
    assert min(frac[3]["fast"], frac[3]["ranged"], frac[3]["troop"]) >= 0.2


def test_deal_decks_cover_every_ordered_pair():
    n = CONFIG.n_decks
    assert sorted(db.deal_decks(k, n) for k in range(n * n)) == [(i, j) for i in range(n) for j in range(n)]
    assert db.deal_decks(n * n + 3, n) == db.deal_decks(3, n)


def test_greedy_mirror_replica_matches_a_real_replay():
    """The tool does not replay greedy vs greedy with seats swapped: the swapped game is identical."""
    game = Game(CONFIG)
    a, b = GreedyAgent(CONFIG), GreedyAgent(CONFIG)
    for k in range(16):
        decks = db.deal_decks(k, CONFIG.n_decks)
        assert db.play(game, (a, b), k, decks) == db.play(game, (b, a), k, decks)


def test_matrix_bookkeeping():
    n = CONFIG.n_decks
    gvg = db.run_match(CONFIG, "greedy", "greedy", 3)  # rounded up to n^2 deals
    assert len(gvg.games) == 2 * n * n and gvg.total.games == 2 * n * n
    for a in range(n):
        for b in range(n):
            c, rev = gvg.P[a][b], gvg.P[b][a]
            assert c.games == 2
            if a != b:  # deterministic mirror: deck a's wins are deck b's losses
                assert c.wins + rev.wins + c.draws == 2
    assert all(c.games == (2 if i == j else 4) for (i, j), c in gvg.C.items())
    assert gvg.C[(0, 0)].wins == 1 or gvg.C[(0, 0)].draws == 2

    gvr = db.run_match(CONFIG, "greedy", "random", 16)
    game = Game(CONFIG)
    greedy, rnd = GreedyAgent(CONFIG), RandomAgent(CONFIG)
    for g in gvr.games[:8]:  # every record can be replayed from its deal
        seats = (greedy, rnd) if g.a_seat == 0 else (rnd, greedy)
        decks = (g.a_deck, g.b_deck) if g.a_seat == 0 else (g.b_deck, g.a_deck)
        winner, rounds = db.play(game, seats, g.deal, decks)
        outcome = 0 if winner == DRAW else (1 if winner == g.a_seat else -1)
        assert (outcome, rounds) == (g.outcome, g.rounds)
    assert sum(c.games for c in gvr.C.values()) == len(gvr.games) == 2 * n * n


def test_workers_give_identical_results():
    serial = db.run_match(CONFIG, "greedy", "random", 32)
    parallel = db.run_match(CONFIG, "greedy", "random", 32, workers=2)
    assert serial.games == parallel.games


def test_cli_prints_matrices(capsys):
    code = db.main(["--deals", "16", "--workers", "1"])
    out = capsys.readouterr().out
    assert "P[a][b]" in out and "C{i,j}" in out and "deck balance gate" in out
    assert code in (0, 1)
    for name in CONFIG.deck_names:
        assert name in out


@pytest.mark.slow
@pytest.mark.skipif(not RUN_SLOW, reason="set CARDGAME_SLOW=1 to run")
def test_deck_balance_gate_large():
    workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    gvg = db.run_match(CONFIG, "greedy", "greedy", 16384, start_seed=100_000, workers=workers)
    gvr = db.run_match(CONFIG, "greedy", "random", 8192, start_seed=100_000, workers=workers)
    print("\n" + gvg.format() + "\n" + gvr.format())
    assert db.gate_failures(gvg, gvr) == []
