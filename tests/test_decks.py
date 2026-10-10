"""Deck balance gate (SPEC section 1.4), the tools/deck_balance.py bookkeeping it relies on, and the health of
the shipped content in real games (random and lookahead agents, mulligan on, fixed and random decks).

The full gate (lookahead vs lookahead and lookahead vs random over thousands of deals) runs under
CARDGAME_SLOW=1; the always-on tests are a smoke run of the same code plus the bookkeeping."""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import deck_balance as db  # noqa: E402

from cardgame.agents import LookaheadAgent, RandomAgent, choose_action, make_agent  # noqa: E402
from cardgame.cards import FAST, RANGED, TROOP, load_ruleset, sample_deal  # noqa: E402
from cardgame.engine import DRAW, Game  # noqa: E402

CONFIG = load_ruleset()  # the shipped ruleset, mulligan on (the gate's setting)
RUN_SLOW = os.environ.get("CARDGAME_SLOW") == "1"
SMOKE_DEALS = 128         # lookahead vs lookahead smoke: 16 games per off-diagonal P cell
SMOKE_LVR_DEALS = 64      # lookahead vs random smoke: 128 games
GATE_DEALS = 8192         # full gate: 1024 games per off-diagonal P cell (SE ~0.016)
GATE_LVR_DEALS = 4096     # 8192 games, 1024 per unordered cross cell
HEALTH_GAMES = 160        # per agent kind


def _workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


# ---------------------------------------------------------------- full gate (slow)
@pytest.fixture(scope="module")
def gate_results():
    workers = _workers()
    lvl = db.run_match(CONFIG, "lookahead", "lookahead", GATE_DEALS, start_seed=100_000, workers=workers)
    lvr = db.run_match(CONFIG, "lookahead", "random", GATE_LVR_DEALS, start_seed=100_000, workers=workers)
    return lvl, lvr


@pytest.mark.slow
@pytest.mark.skipif(not RUN_SLOW, reason="set CARDGAME_SLOW=1 to run")
def test_deck_balance_gate_large(gate_results):
    lvl, lvr = gate_results
    print("\n" + lvl.format() + "\n" + lvr.format())
    assert db.gate_failures(lvl, lvr) == []


@pytest.mark.slow
@pytest.mark.skipif(not RUN_SLOW, reason="set CARDGAME_SLOW=1 to run")
def test_deck_balance_has_margin(gate_results):
    """Shipped decks aim at P in [0.40, 0.60], so that the [0.30, 0.70] gate is not passed by luck."""
    lvl, _ = gate_results
    n = lvl.n_decks
    worst = max(abs(lvl.P[a][b].win_rate - 0.5) for a in range(n) for b in range(n) if a != b)
    assert worst <= 0.10, lvl.format()


# ---------------------------------------------------------------- smoke (always)
def test_deck_balance_smoke():
    """A small run of the gate's code path: matrices fill, every game ends, and loose versions of the gate
    hold (the real thresholds need the large sample of the slow test)."""
    lvl = db.run_match(CONFIG, "lookahead", "lookahead", SMOKE_DEALS, start_seed=200_000)
    lvr = db.run_match(CONFIG, "lookahead", "random", SMOKE_LVR_DEALS, start_seed=200_000)
    n = lvl.n_decks
    assert lvl.simulated == SMOKE_DEALS and len(lvl.games) == 2 * SMOKE_DEALS
    assert lvr.simulated == len(lvr.games) == 2 * SMOKE_LVR_DEALS
    for a in range(n):
        for b in range(n):
            if a != b:
                assert 0.10 <= lvl.P[a][b].win_rate <= 0.90, lvl.format()
    assert lvl.total.draw_rate <= db.MAX_DRAW_RATE and lvl.total.mean_rounds <= db.MAX_MEAN_ROUNDS
    assert lvr.total.win_rate >= 0.9, lvr.format()
    db.gate_failures(lvl, lvr)  # runs on any result


def test_shipped_decks_keep_their_styles():
    """SPEC section 1.4, counted over units: aggro >= 50% fast, defensive >= 40% Defense, ranged-heavy >= 40%
    ranged, balanced >= 20% of each nature; 40 cards, at most 3 copies."""
    cards = CONFIG.cards
    assert CONFIG.n_decks == 4
    frac = []
    for deck in CONFIG.decks:
        assert len(deck) == CONFIG.deck_size == 40
        assert max(deck.count(c) for c in set(deck)) <= 3
        units = [cards[c] for c in deck if cards[c].is_unit]
        frac.append({k: sum(map(pred, units)) / len(units) for k, pred in (
            ("fast", lambda c: c.nature == FAST), ("ranged", lambda c: c.nature == RANGED),
            ("troop", lambda c: c.nature == TROOP), ("defense", lambda c: c.defense))})
    assert frac[0]["fast"] >= 0.5
    assert frac[1]["defense"] >= 0.4
    assert frac[2]["ranged"] >= 0.4
    assert min(frac[3]["fast"], frac[3]["ranged"], frac[3]["troop"]) >= 0.2


def test_deal_decks_cover_every_ordered_pair():
    n = CONFIG.n_decks
    assert sorted(db.deal_decks(k, n) for k in range(n * n)) == [(i, j) for i in range(n) for j in range(n)]
    assert db.deal_decks(n * n + 3, n) == db.deal_decks(3, n)


def test_lookahead_mirror_replica_matches_a_real_replay():
    """The tool does not replay a mirror of one seeded agent spec with seats swapped: agent seeds come from
    the seat, so the swapped game is identical."""
    assert db.replicates("lookahead", "lookahead") and not db.replicates("lookahead", "random")
    game = Game(CONFIG)
    a, b = LookaheadAgent(CONFIG), LookaheadAgent(CONFIG)
    for k in range(8):
        decks = db.deal_decks(k, CONFIG.n_decks)
        assert db.play(game, (a, b), k, decks) == db.play(game, (b, a), k, decks)


def test_play_uses_choose_action_and_the_mulligan():
    """db.play drives agents through choose_action from the mulligan on (lookahead needs the game)."""
    game = Game(CONFIG)
    agents = (LookaheadAgent(CONFIG), RandomAgent(CONFIG))
    winner, rounds = db.play(game, agents, 5, (0, 1))
    assert game.done and winner in (0, 1, DRAW) and rounds == game.round >= 1
    assert all(game.mulligan_done) and game.guard_trips == 0


def test_matrix_bookkeeping():
    n = CONFIG.n_decks
    lvl = db.run_match(CONFIG, "lookahead", "lookahead", 3)  # rounded up to n^2 deals
    assert len(lvl.games) == 2 * n * n and lvl.total.games == 2 * n * n and lvl.simulated == n * n
    for a in range(n):
        for b in range(n):
            c, rev = lvl.P[a][b], lvl.P[b][a]
            assert c.games == 2
            if a != b:  # replicated mirror: deck a's wins are deck b's losses
                assert c.wins + rev.wins + c.draws == 2
    assert all(c.games == (2 if i == j else 4) for (i, j), c in lvl.C.items())

    lvr = db.run_match(CONFIG, "lookahead", "random", 16)
    game = Game(CONFIG)
    la, rnd = LookaheadAgent(CONFIG), RandomAgent(CONFIG)
    for g in lvr.games[:6]:  # every record can be replayed from its deal
        seats = (la, rnd) if g.a_seat == 0 else (rnd, la)
        decks = (g.a_deck, g.b_deck) if g.a_seat == 0 else (g.b_deck, g.a_deck)
        winner, rounds = db.play(game, seats, g.deal, decks)
        outcome = 0 if winner == DRAW else (1 if winner == g.a_seat else -1)
        assert (outcome, rounds) == (g.outcome, g.rounds)
    assert sum(c.games for c in lvr.C.values()) == len(lvr.games) == lvr.simulated == 2 * n * n


def test_workers_give_identical_results():
    serial = db.run_match(CONFIG, "lookahead", "random", 32)
    parallel = db.run_match(CONFIG, "lookahead", "random", 32, workers=2)
    assert serial.games == parallel.games


def test_gate_failures_report_each_violation():
    lvl = db.run_match(CONFIG, "lookahead", "lookahead", 16)
    lvr = db.run_match(CONFIG, "lookahead", "random", 16)
    for c in lvl.P[0][1], lvl.P[1][0]:
        c.wins, c.draws = 0, 0
    lvr.C[(0, 1)].wins = 0
    lvl.total.rounds = 31 * lvl.total.games
    lvl.P[2][2].draws = lvl.P[2][2].games
    text = "\n".join(db.gate_failures(lvl, lvr))
    assert "P[Blitz][Bulwark]" in text and "P[Bulwark][Blitz]" in text and "C{Blitz,Bulwark}" in text
    assert "mean game length" in text and "draws Volley vs Volley" in text


def test_cli_prints_matrices(capsys):
    code = db.main(["--deals", "16", "--lvr-deals", "16", "--workers", "1"])
    out = capsys.readouterr().out
    assert "P[a][b]" in out and "C{i,j}" in out and "deck balance gate" in out
    assert "lookahead vs lookahead" in out and "lookahead vs random" in out
    assert code in (0, 1)
    for name in CONFIG.deck_names:
        assert name in out


# ---------------------------------------------------------------- shipped-content health (always)
def _health_games(spec: str, seed0: int) -> tuple:
    """HEALTH_GAMES games of `spec` vs `spec` on sample_deal(seed, 0.7) decks (mulligan on). Returns (cards seen
    in play: on the board or played from hand, guard trips, rounds per game, winners, fixed/random seats)."""
    seen, trips, rounds, winners, kinds = Counter(), 0, [], [], Counter()
    game = Game(CONFIG)
    for k in range(HEALTH_GAMES):
        seed = seed0 + k
        decks = sample_deal(seed, CONFIG, 0.7)
        kinds.update("fixed" if isinstance(d, int) else "random" for d in decks)
        game.reset(seed, decks=decks)
        agents = (make_agent(spec, CONFIG, seed=2 * seed), make_agent(spec, CONFIG, seed=2 * seed + 1))
        steps = 0
        while not game.done:
            game.step(choose_action(agents[game.current_player()], game))
            steps += 1
            assert steps < 5000, f"{spec} game {seed} does not end"
            for zone in (game.backline[0], game.backline[1], game.frontline):
                for u in zone:
                    seen[u.card] += 1
        for p in (0, 1):
            for c, n in enumerate(game.played[p]):
                seen[c] += n
        trips += game.guard_trips
        rounds.append(game.round)
        winners.append(game.winner())
    return seen, trips, rounds, winners, kinds


@pytest.fixture(scope="module")
def health():
    return {spec: _health_games(spec, seed0) for spec, seed0 in (("random", 300_000), ("lookahead", 310_000))}


@pytest.mark.parametrize("spec", ["random", "lookahead"])
def test_shipped_content_plays_cleanly(health, spec):
    """Random and lookahead self-play with the mulligan on, on fixed and random decks: no exception, the loop
    guard never trips, games end well before the round limit."""
    seen, trips, rounds, winners, kinds = health[spec]
    assert kinds["fixed"] > 0 and kinds["random"] > kinds["fixed"]
    assert trips == 0
    assert sum(rounds) / len(rounds) <= 20 and max(rounds) < CONFIG.max_rounds
    assert winners.count(DRAW) <= 0.02 * len(winners)


def test_every_card_appears_in_play(health):
    """Every card of the pool (tokens included) reaches the board or is played in these games."""
    seen = health["random"][0] + health["lookahead"][0]
    missing = [c.id for c in CONFIG.cards.cards if not seen[c.index]]
    assert not missing, missing
