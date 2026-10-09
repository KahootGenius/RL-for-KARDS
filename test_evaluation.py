"""Evaluation tests (SPEC §10/§11): duplicate pairing over deck matchups, per-seat / cell / P-matrix
bookkeeping recomputed from the game records, reproducibility across worker counts, a dying worker
raising instead of hanging, the decks="all" schedule, Wilson intervals, PPO checkpoints (Stage 1 ones
rejected), the "Done when" verdict, and eval.py / bench.py end to end."""
from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from typing import List, Optional, Sequence

import pytest

from cardgame.agents import GreedyAgent, RandomAgent, make_agent
from cardgame.cards import load_ruleset, sample_decks
from cardgame.engine import DRAW, Game, IllegalActionError, Observation
from cardgame.evaluation import (BrokenProcessPool, Evaluator, GameRecord, IllegalAgentActionError, MatchResult,
                                 agent_seed, check_decks, deal_schedule, deals_for_games, duplicate_match,
                                 outcome_for, play_game, round_deals, short_deck_names, wilson_interval)

CONFIG = load_ruleset()
N_DECKS = CONFIG.n_decks
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SpyAgent:
    """Wraps an agent and records the seeds and observations it is given (one list per game)."""

    def __init__(self, inner):
        self.inner = inner
        self.name = f"spy({inner.name})"
        self.seeds: List[Optional[int]] = []
        self.games: List[List[Observation]] = []

    def reset(self, seed: Optional[int] = None) -> None:
        self.seeds.append(seed)
        self.games.append([])
        self.inner.reset(seed)

    def act(self, obs: Observation, legal_actions: Sequence[int]) -> int:
        self.games[-1].append(obs)
        return self.inner.act(obs, legal_actions)


class FixedActionAgent:
    name = "fixed"

    def __init__(self, action):
        self.action = action

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def act(self, obs: Observation, legal_actions: Sequence[int]):
        return self.action


def first_player_of(seed: int) -> int:
    g = Game(CONFIG)
    g.reset(seed)
    return g.first_player


def counts_without_timing(r: MatchResult) -> dict:
    d = r.to_dict(include_games=True)
    for k in ("elapsed_s", "games_per_s"):
        d.pop(k)
    return d


def synthetic_result(outcome, n_deals: int = 1008, decks="all") -> MatchResult:
    """A MatchResult from made-up records: outcome(i, j, a_seat, seed) = A's +1/0/-1 with A holding
    deck i = decks[a_seat] against deck j."""
    deals = []
    for seed, pair in deal_schedule(0, n_deals, decks, N_DECKS):
        pair = pair or sample_decks(seed, N_DECKS)
        games = []
        for a_seat in (0, 1):
            out = outcome(pair[a_seat], pair[1 - a_seat], a_seat, seed)
            winner = DRAW if out == 0 else (a_seat if out > 0 else 1 - a_seat)
            games.append(GameRecord(seed, winner, seed % 2, 10, 60, tuple(pair)))
        deals.append(tuple(games))
    return MatchResult("ppo", "greedy", 0, deals, 1.0, {}, {}, decks, CONFIG.deck_names)


# ---------------------------------------------------------------------- play_game
def test_play_game_record_and_agent_view():
    seed = 11
    a, b = SpyAgent(GreedyAgent(CONFIG)), SpyAgent(RandomAgent(CONFIG))
    rec = play_game(CONFIG, (a, b), seed)
    assert isinstance(rec, GameRecord)
    assert rec.seed == seed and rec.first_player == first_player_of(seed)
    assert rec.decks == sample_decks(seed, N_DECKS)  # default: the deal seed's sampled decks
    assert rec.winner in (0, 1, DRAW) and 1 <= rec.rounds <= CONFIG.max_rounds
    assert rec.steps == len(a.games[-1]) + len(b.games[-1])
    assert a.seeds == [agent_seed(seed, 0)] and b.seeds == [agent_seed(seed, 1)]
    for seat, spy in enumerate((a, b)):
        assert spy.games[-1], "each seat acts at least once"
        for obs in spy.games[-1]:
            assert obs.player == seat and obs.is_my_turn and not obs.done
            assert obs.my_deck == rec.decks[seat]
    rec2 = play_game(CONFIG, (a, b), seed, decks=(3, 1))
    assert rec2.decks == (3, 1)
    assert a.games[-1][0].my_deck == 3 and b.games[-1][0].my_deck == 1


def test_play_game_reuses_game_and_is_deterministic():
    game = Game(CONFIG)
    agents = (RandomAgent(CONFIG), RandomAgent(CONFIG))
    first = [play_game(CONFIG, agents, s, game=game, decks=(s % 4, 3 - s % 4)) for s in range(6)]
    again = [play_game(CONFIG, agents, s, decks=(s % 4, 3 - s % 4)) for s in range(6)]
    assert first == again
    custom = play_game(CONFIG, agents, 3, agent_seeds=(1, 2))
    assert custom == play_game(CONFIG, agents, 3, agent_seeds=(1, 2))


@pytest.mark.parametrize("bad", [999, -1, Game(CONFIG).num_actions, None, "0"])
def test_illegal_agent_action_raises_clear_error(bad):
    with pytest.raises(IllegalAgentActionError, match="fixed") as exc:
        play_game(CONFIG, (FixedActionAgent(bad), GreedyAgent(CONFIG)), 0)
    assert isinstance(exc.value, IllegalActionError)
    assert "seat" in str(exc.value) and "legal" in str(exc.value) and "decks" in str(exc.value)


def test_agent_seed_is_deterministic_and_distinct():
    seeds = {agent_seed(s, p) for s in range(500) for p in (0, 1)}
    assert len(seeds) == 1000
    assert all(0 <= x < 2 ** 63 for x in seeds)
    assert agent_seed(7, 1) == agent_seed(7, 1) and agent_seed(-3, 0) >= 0


# ---------------------------------------------------------------------- deck schedules
def test_decks_all_schedule_and_rounding():
    n2 = N_DECKS * N_DECKS
    assert round_deals(1, "all", N_DECKS) == n2 and round_deals(n2, "all", N_DECKS) == n2
    assert round_deals(n2 + 1, "all", N_DECKS) == 2 * n2 and round_deals(7, "sampled", N_DECKS) == 7
    assert deals_for_games(2000, "all", 4) == 1008  # 2,000 games -> 2,016
    assert deals_for_games(2000, "sampled", 4) == 1000 and deals_for_games(3, "sampled", 4) == 2
    sched = deal_schedule(5, 3 * n2 - 2, "all", N_DECKS)
    assert len(sched) == 3 * n2
    for k, (seed, pair) in enumerate(sched):
        assert seed == 5 + k and pair == divmod(k % n2, N_DECKS)
    counts = Counter(pair for _, pair in sched)
    assert set(counts) == {(i, j) for i in range(N_DECKS) for j in range(N_DECKS)}
    assert set(counts.values()) == {3}  # every ordered pair (mirrors included) equally often
    assert deal_schedule(0, 4, "sampled", N_DECKS) == [(k, None) for k in range(4)]
    assert deal_schedule(2, 2, (1, 2), N_DECKS) == [(2, (1, 2)), (3, (1, 2))]
    for bad in ("both", (0, N_DECKS), (0,), (-1, 0)):
        with pytest.raises(ValueError):
            check_decks(bad, N_DECKS)


def test_decks_all_match_covers_every_cell_equally():
    res = duplicate_match("greedy", "random", 2 * N_DECKS ** 2, start_seed=3)
    assert res.n_deals == 2 * N_DECKS ** 2 and res.games == 4 * N_DECKS ** 2
    for (i, j), w in res.cells.items():
        assert w.games == (4 if i == j else 8)  # 2 deals per ordered pair x 2 games
    for a in range(N_DECKS):
        for b in range(N_DECKS):
            assert res.matrix[a][b].games == 4  # deals (a, b) seat 0 + deals (b, a) seat 1
    assert len(res.cells) == N_DECKS * (N_DECKS + 1) // 2


# ---------------------------------------------------------------------- duplicate pairing
@pytest.mark.parametrize("decks", ["all", "sampled"])
def test_duplicate_pairs_use_identical_deals_with_swapped_seats(decks):
    start, n = 5, 16
    res = duplicate_match("greedy", "random", n, start_seed=start, decks=decks)
    assert res.n_deals == n and res.games == 2 * n and res.decks == decks
    for k, (g0, g1) in enumerate(res.deals):
        s = start + k
        want = divmod(k % N_DECKS ** 2, N_DECKS) if decks == "all" else sample_decks(s, N_DECKS)
        assert g0.seed == g1.seed == s and g0.decks == g1.decks == want
        assert g0.first_player == g1.first_player == first_player_of(s)
        # Game 1: A (greedy) in seat 0; game 2: B (random) in seat 0. Replays must match exactly.
        assert g0 == play_game(CONFIG, (GreedyAgent(CONFIG), RandomAgent(CONFIG)), s, decks=want)
        assert g1 == play_game(CONFIG, (RandomAgent(CONFIG), GreedyAgent(CONFIG)), s, decks=want)


def test_swapped_games_deal_the_same_hands():
    """Each seat is dealt the same deck and cards in both games of a deal, whoever sits there."""
    for seed in range(6):
        decks = (seed % N_DECKS, (seed + 1) % N_DECKS)
        g, r = SpyAgent(GreedyAgent(CONFIG)), SpyAgent(RandomAgent(CONFIG))
        play_game(CONFIG, (g, r), seed, decks=decks)
        play_game(CONFIG, (r, g), seed, decks=decks)
        for seat in (0, 1):
            game1 = (g, r)[seat].games[0][0]
            game2 = (r, g)[seat].games[1][0]
            assert game1.player == game2.player == seat and game1.my_deck == game2.my_deck == decks[seat]
            assert game1.hand == game2.hand and game1.went_first == game2.went_first
            assert game1.my_deck_size == game2.my_deck_size


@pytest.mark.parametrize("spec", ["greedy", "random"])
def test_mirror_match_is_exactly_balanced(spec):
    """Same agent on both sides + per-seat agent seeds => both games of a deal are identical."""
    res = duplicate_match(spec, spec, N_DECKS ** 2, start_seed=100)
    for g0, g1 in res.deals:
        assert g0 == g1
    assert res.wins == res.losses
    ds = res.deal_summary()
    assert ds["a_won_both"] == ds["b_won_both"] == 0
    assert ds["split"] + ds["with_draw"] == res.n_deals
    assert res.by_seat[0].wins == res.by_seat[1].losses
    for w in res.cells.values():
        assert w.wins == w.losses
    for a in range(N_DECKS):
        for b in range(N_DECKS):
            assert res.matrix[a][b].wins == res.matrix[b][a].losses


# ---------------------------------------------------------------------- bookkeeping
@pytest.mark.parametrize("spec_a,spec_b,decks", [("greedy", "random", "all"), ("random", "greedy", "sampled"),
                                                  ("random", "random", "all")])
def test_bookkeeping_matches_the_game_records(spec_a, spec_b, decks):
    res = duplicate_match(spec_a, spec_b, 32, start_seed=40, decks=decks)
    n = res.n_deals
    records = [(a_seat, rec) for pair in res.deals for a_seat, rec in enumerate(pair)]

    def wdl(rows):
        outs = [outcome_for(rec, a_seat) for a_seat, rec in rows]
        return outs.count(1), outs.count(0), outs.count(-1)

    def same(w, rows):
        assert (w.wins, w.draws, w.losses) == wdl(rows)

    same(res.overall, records)
    for seat in (0, 1):
        same(res.by_seat[seat], [(s, r) for s, r in records if s == seat])
    same(res.first, [(s, r) for s, r in records if r.first_player == s])
    same(res.second, [(s, r) for s, r in records if r.first_player != s])
    for (i, j), w in res.cells.items():
        same(w, [(s, r) for s, r in records if sorted(r.decks) == [i, j]])
    for a in range(N_DECKS):
        for b in range(N_DECKS):
            same(res.matrix[a][b], [(s, r) for s, r in records if (r.decks[s], r.decks[1 - s]) == (a, b)])
    assert res.by_seat[0].games == res.by_seat[1].games == n  # A sits in each seat once per deal
    assert res.first.games == res.second.games == n  # same coin flip, swapped seats
    assert sum(w.games for w in res.cells.values()) == res.games == 2 * n
    assert sum(w.games for row in res.matrix for w in row) == res.games
    assert res.win_rate + res.draw_rate + res.loss_rate == pytest.approx(1.0)
    assert sum(res.deal_summary().values()) == n

    d = res.to_dict(include_games=True)
    json.dumps(d)  # serialisable
    assert d["games"] == 2 * n and d["wins"] == res.wins and d["agent_a"] == spec_a and d["decks"] == decks
    assert d["by_seat"]["0"]["games"] == n and d["by_turn_order"]["first"]["games"] == n
    assert d["win_rate_ci95"] == list(wilson_interval(res.wins, res.games))
    assert len(d["matchups"]) == len(res.cells) and d["deck_names"] == list(CONFIG.deck_names)
    for cell in d["matchups"]:
        assert cell["games"] == res.cell(*cell["decks"]).games
        assert cell["names"] == [CONFIG.deck_names[k] for k in cell["decks"]]
    assert d["deck_matrix"]["games"] == [[w.games for w in row] for row in res.matrix]
    assert len(d["games_played"]) == 2 * n and all(len(g["decks"]) == 2 for g in d["games_played"])
    text = res.format()
    for word in ("overall", "seat 0", "seat 1", "first", "second", "95% CI", spec_b, "C{i,j}", "P[a][b]",
                 *CONFIG.deck_names):
        assert word in text


def test_cells_and_matrix_from_hand_made_records():
    """Deal (0, 1): A wins with deck 0 (seat 0) and loses with deck 1 (seat 1); deal (2, 2): a draw and a win."""
    deals = [(GameRecord(0, 0, 0, 9, 50, (0, 1)), GameRecord(0, 0, 0, 9, 50, (0, 1))),
             (GameRecord(1, DRAW, 1, 50, 300, (2, 2)), GameRecord(1, 1, 1, 7, 40, (2, 2)))]
    res = MatchResult("a", "b", 0, deals, decks="sampled", deck_names=CONFIG.deck_names)
    assert (res.cell(1, 0).wins, res.cell(0, 1).losses, res.cell(0, 1).games) == (1, 1, 2)
    assert (res.cell(2, 2).wins, res.cell(2, 2).draws) == (1, 1)
    assert res.matrix[0][1].wins == 1 and res.matrix[1][0].losses == 1 and res.matrix[0][1].games == 1
    assert res.matrix[2][2].games == 2 and res.cell(0, 3).games == 0
    (key, worst) = res.min_cell()
    assert key == (0, 1) and worst.win_rate == 0.5
    assert res.cell_name(key) == f"{CONFIG.deck_names[0]}-{CONFIG.deck_names[1]}"
    (key, worst) = res.min_ordered_cell()  # P[1][0]: A held deck 1 and lost (P[0][1] = 100%, P[2][2] = 50%)
    assert key == (1, 0) and worst.win_rate == 0.0 and worst.games == 1
    assert res.ordered_cell_name(key) == f"P[{CONFIG.deck_names[1]}][{CONFIG.deck_names[0]}]"
    d = res.to_dict()["min_ordered_cell"]
    assert d["decks"] == [1, 0] and d["diagnostic"] is True and d["games"] == 1
    assert res.deal_summary() == {"a_won_both": 0, "split": 1, "b_won_both": 0, "with_draw": 1}
    assert "-" in res.format_cells()  # unplayed cells shown as "-"


def test_outcome_for():
    rec = GameRecord(0, 1, 0, 5, 40, (0, 0))
    assert outcome_for(rec, 1) == 1 and outcome_for(rec, 0) == -1
    assert outcome_for(rec._replace(winner=DRAW), 0) == 0


# ---------------------------------------------------------------------- reproducibility
def test_reproducible_and_independent_of_worker_count():
    args = ("random", "greedy", 2 * N_DECKS ** 2)
    r1 = duplicate_match(*args, start_seed=7, workers=1)
    r1b = duplicate_match(*args, start_seed=7, workers=1)
    with Evaluator(workers=2, config=CONFIG) as ev:
        r2 = ev.match(*args, start_seed=7, chunk_size=3)
        s2 = ev.match("greedy", "random", 9, start_seed=3, decks="sampled")
    assert r1.deals == r1b.deals == r2.deals
    assert counts_without_timing(r1) == counts_without_timing(r2)
    s1 = duplicate_match("greedy", "random", 9, start_seed=3, decks="sampled")
    assert s1.deals == s2.deals
    r3 = duplicate_match(*args, start_seed=8, workers=1)
    assert r3.deals != r1.deals
    # A vs B and B vs A over the same deals are the same games with the roles relabelled.
    flipped = duplicate_match("greedy", "random", 2 * N_DECKS ** 2, start_seed=7)
    assert r1.wins == flipped.losses and r1.draws == flipped.draws


def test_bad_arguments():
    with pytest.raises(ValueError):
        duplicate_match("greedy", "random", 0)
    with pytest.raises(ValueError):
        duplicate_match("greedy", "random", 2, start_seed=-1)
    with pytest.raises(FileNotFoundError):
        duplicate_match("greedy", "no/such/checkpoint.pt", 2)
    with pytest.raises(ValueError):
        duplicate_match("greedy", "not-an-agent", 2, decks="sampled")
    with pytest.raises(ValueError):
        duplicate_match("greedy", "random", 2, decks="everything")


def test_worker_death_raises_instead_of_hanging():
    """A worker killed mid-match (crash, OOM killer) makes match() raise BrokenProcessPool, never wait forever."""
    before = set(multiprocessing.active_children())
    ev = Evaluator(workers=2, config=CONFIG)
    try:
        ev.warmup("greedy", "greedy")  # starts both workers
        workers = [p for p in multiprocessing.active_children() if p not in before]
        assert len(workers) == 2
        outcome = {}

        def run() -> None:
            try:
                outcome["result"] = ev.match("greedy", "greedy", 8192, chunk_size=2, decks="sampled")
            except BaseException as exc:  # noqa: BLE001 - inspected below
                outcome["error"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        time.sleep(0.3)  # the match (16,384 games, seconds of work) is running
        workers[0].kill()
        thread.join(60)
        assert not thread.is_alive(), "match() hung after a worker died"
        assert isinstance(outcome.get("error"), BrokenProcessPool), outcome
        assert "died" in str(outcome["error"])
        with pytest.raises(RuntimeError, match="closed"):  # the broken pool was shut down
            ev.match("greedy", "random", 2, decks="sampled")
        assert not any(p.is_alive() for p in workers)
    finally:
        ev.__exit__(RuntimeError, None, None)


def test_mirror_match_uses_two_agent_instances():
    from cardgame.evaluation import _DealRunner
    runner = _DealRunner(CONFIG)
    runner.run(("random", {}, "random", {}, [(0, (0, 1))]))
    assert len({id(a) for a in runner.agents.values()}) == 2


# ---------------------------------------------------------------------- Wilson interval
def test_wilson_interval():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.403832, abs=1e-6) and hi == pytest.approx(0.596168, abs=1e-6)
    assert wilson_interval(0, 10) == (0.0, pytest.approx(0.277533, abs=1e-6))
    assert wilson_interval(10, 10) == (pytest.approx(0.722467, abs=1e-6), 1.0)
    for k, n in [(1, 3), (7, 20), (1400, 2000), (1999, 2000)]:
        lo, hi = wilson_interval(k, n)
        assert 0.0 <= lo < k / n < hi <= 1.0
        mlo, mhi = wilson_interval(n - k, n)  # symmetric under success <-> failure
        assert mlo == pytest.approx(1 - hi) and mhi == pytest.approx(1 - lo)
        wlo, whi = wilson_interval(k, n, z=2.576)  # wider at 99%
        assert wlo < lo and whi > hi
    lo, hi = wilson_interval(1400, 2000)  # the target: 70% of 2000 games
    assert lo == pytest.approx(0.6795, abs=1e-3) and hi == pytest.approx(0.7197, abs=1e-3)


# ---------------------------------------------------------------------- verdict (eval.py)
def verdict_args(**kw) -> argparse.Namespace:
    base = dict(target=0.70, cell_target=0.60, scenario_target=0.50, decks="all", seed=0, deterministic=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_done_when_verdict():
    eval_cli = importlib.import_module("eval")
    # loses its seat-0 game in every other block of 16 deals: 94/126 = 74.6% in every cell
    strong = synthetic_result(lambda i, j, seat, seed: -1 if seat == 0 and (seed // 16) % 2 == 0 else 1)
    weak_cell = synthetic_result(lambda i, j, seat, seed: -1 if {i, j} == {0, 2} else 1)  # 0% in one cell
    assert strong.games == 2016 and strong.win_rate == pytest.approx(94 / 126)
    assert all(w.win_rate == pytest.approx(94 / 126) for w in strong.cells.values())

    # a conclusive verdict needs all three parts: without --scenarios it is INDICATIVE, never PASS
    check = eval_cli.target_check([("greedy", strong)], verdict_args(), True, None)
    assert check["checks"]["overall"] and check["checks"]["cells"] and "scenarios" not in check["checks"]
    assert not check["conclusive"] and not check["passed"]
    assert check["indicative_reasons"] == ["scenarios not run (--scenarios)"]
    assert eval_cli.verdict_text(check).startswith("INDICATIVE (scenarios not run (--scenarios))")
    check = eval_cli.target_check([("greedy", strong)], verdict_args(), True, 56.25)
    assert check["passed"] and check["conclusive"] and check["indicative_reasons"] == []
    assert eval_cli.verdict_text(check).startswith("PASS")
    check = eval_cli.target_check([("greedy", strong)], verdict_args(), True, 50.0)
    assert check["conclusive"] and not check["passed"] and not check["checks"]["scenarios"]  # needs > 50%
    assert eval_cli.verdict_text(check).startswith("FAIL")

    check = eval_cli.target_check([("greedy", weak_cell)], verdict_args(), True, 100.0)
    assert weak_cell.win_rate > 0.7 and check["checks"]["overall"] and not check["checks"]["cells"]
    names = CONFIG.deck_names
    assert check["worst_cell"]["name"] == f"{names[0]}-{names[2]}" and not check["passed"]
    assert check["conclusive"] and eval_cli.verdict_text(check).startswith("FAIL")

    for kw, reason in [({"deterministic": True}, "deterministic"), ({"seed": 5}, "seeds"),
                       ({"decks": "sampled"}, "decks"), ({"target": 0.69}, "--target 0.69 below"),
                       ({"cell_target": 0.5}, "--cell-target 0.5 below"),
                       ({"scenario_target": 0.25}, "--scenario-target 0.25 below")]:
        check = eval_cli.target_check([("greedy", strong)], verdict_args(**kw), True, 56.25)
        assert not check["conclusive"] and not check["passed"]
        assert all(check["checks"].values())  # the numbers pass; the settings make it indicative
        assert reason in " ".join(check["indicative_reasons"])
        assert eval_cli.verdict_text(check).startswith("INDICATIVE")
    stricter = eval_cli.target_check([("greedy", strong)], verdict_args(target=0.72, cell_target=0.7,
                                                                         scenario_target=0.55), True, 56.25)
    assert stricter["conclusive"] and stricter["passed"]  # raising the bar keeps the verdict conclusive
    small = synthetic_result(lambda i, j, seat, seed: 1, n_deals=16)
    check = eval_cli.target_check([("greedy", small)], verdict_args(), True, 100.0)
    assert not check["conclusive"] and "32 games" in check["indicative_reasons"][0]
    assert eval_cli.target_check([("random", strong)], verdict_args(), True, None) is None


def test_verdict_reports_the_worst_ordered_cell_as_a_diagnostic():
    """SPEC §10: the worst ordered cell P[a][b] is shown next to the verdict but never decides it."""
    eval_cli = importlib.import_module("eval")
    names = CONFIG.deck_names
    # loses only holding deck 1 in seat 0 against deck 3: P[1][3] = 50%, while C{1,3} = 75% (worst cell)
    res = synthetic_result(lambda i, j, seat, seed: -1 if (i, j) == (1, 3) and seat == 0 else 1)
    assert res.matrix[1][3].win_rate == 0.5 and res.min_cell()[1].win_rate == pytest.approx(0.75)
    check = eval_cli.target_check([("greedy", res)], verdict_args(), True, 75.0)
    assert check["passed"] and check["conclusive"]  # an ordered cell below 60% does not fail the test
    wo = check["worst_ordered_cell"]
    assert wo["diagnostic"] is True and wo["decks"] == [1, 3] and wo["win_rate"] == 0.5
    assert wo["name"] == f"P[{names[1]}][{names[3]}]" and wo["games"] == res.matrix[1][3].games == 126
    assert (wo["agent_deck"], wo["opponent_deck"]) == (names[1], names[3])
    text = eval_cli.verdict_text(check)
    first, second = text.split("\n")
    assert first.startswith("PASS") and "P[" not in first
    assert "diagnostic" in second and f"P[{names[1]}][{names[3]}] = 50.0% over 126 games" in second
    json.dumps(check)


def test_cells_table_and_short_deck_names():
    eval_cli = importlib.import_module("eval")
    assert short_deck_names(("Blitz", "Bulwark", "Volley", "Legion")) == ("Bl", "Bu", "Vo", "Le")
    assert short_deck_names(("aaa1", "aaa2")) == ("aaa1", "aaa2") and short_deck_names(("x", "y")) == ("x", "y")
    a = synthetic_result(lambda i, j, seat, seed: 1)
    b = synthetic_result(lambda i, j, seat, seed: -1 if {i, j} == {0, 2} else 1)
    lines = eval_cli.cells_table([("greedy", a), ("ckpt_00001.pt", b)], 14).splitlines()
    short = short_deck_names(CONFIG.deck_names)
    titles = lines[0].split()
    assert titles[0] == "opponent" and len(titles) == 1 + len(a.cells)
    assert titles[1:] == [f"{short[i]}-{short[j]}" for i, j in a.cells]
    assert lines[1].split() == ["greedy"] + ["100.0"] * len(a.cells)
    row = dict(zip(titles[1:], lines[2].split()[1:]))
    assert row[f"{short[0]}-{short[2]}"] == "0.0" and row[f"{short[0]}-{short[1]}"] == "100.0"
    assert lines[3].split() == ["(games)"] + [str(w.games) for w in a.cells.values()]


# ---------------------------------------------------------------------- PPO checkpoints
@pytest.fixture(scope="module")
def ckpt_dir(tmp_path_factory):
    """Two small randomly initialised Stage 2 checkpoints (SPEC §7/§8 format) and one Stage 1 checkpoint."""
    torch = pytest.importorskip("torch")
    from cardgame.features import ObservationEncoder
    from cardgame.rl.network import EntityPolicyNet, PolicyValueNet

    d = tmp_path_factory.mktemp("ckpts")
    layout = ObservationEncoder(CONFIG).layout()
    for update in (1, 2):
        torch.manual_seed(update)
        net = EntityPolicyNet(layout, d_model=32, id_dim=8, ctx_dim=64, pair_dim=32)
        torch.save({"model": net.state_dict(), "net": net.spec(), "update": update, "env_steps": 0,
                    "args": {}}, d / f"ckpt_{update:05d}.pt")
    stage1 = PolicyValueNet(393, 71, (16,))
    torch.save({"model": stage1.state_dict(), "net": {"obs_dim": 393, "n_actions": 71, "hidden": [16]},
                "update": 3, "env_steps": 0, "args": {}}, d / "stage1.pt")
    return d


def test_ppo_checkpoint_loads_and_plays_legal_games(ckpt_dir):
    path = str(ckpt_dir / "ckpt_00001.pt")
    for spec in (path, "ppo:" + path):
        for kwargs in ({}, {"deterministic": True}):
            agent = make_agent(spec, CONFIG, **kwargs)
            for seed in range(2):  # play_game raises IllegalAgentActionError on any illegal action
                for seats in ((agent, GreedyAgent(CONFIG)), (RandomAgent(CONFIG), agent)):
                    rec = play_game(CONFIG, seats, seed, decks=(seed, 3 - seed))
                    assert rec.winner in (0, 1, DRAW)


def test_stage1_checkpoint_is_rejected(ckpt_dir):
    with pytest.raises(ValueError, match="Stage 1"):
        make_agent(str(ckpt_dir / "stage1.pt"), CONFIG)
    shipped = os.path.join(ROOT, "models", "stage1", "ppo_final.pt")  # the Stage 1 model, if still shipped
    if os.path.isfile(shipped):
        with pytest.raises(ValueError, match="Stage 1"):
            make_agent(shipped, CONFIG)


def test_ppo_duplicate_match_is_reproducible_across_workers(ckpt_dir):
    path = str(ckpt_dir / "ckpt_00002.pt")
    r1 = duplicate_match(path, "greedy", 6, start_seed=21, workers=1, decks="sampled")
    r2 = duplicate_match(path, "greedy", 6, start_seed=21, workers=2, decks="sampled")
    assert r1.deals == r2.deals  # stochastic policy, but seeded per game and seat
    det = duplicate_match(path, "ppo:" + str(ckpt_dir / "ckpt_00001.pt"), 2, workers=1, decks=(2, 0),
                          agent_kwargs_a={"deterministic": True})
    assert det.kwargs_a == {"deterministic": True} and det.games == 4
    assert all(g.decks == (2, 0) for pair in det.deals for g in pair)


# ---------------------------------------------------------------------- command lines
def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=600)


def test_find_checkpoints(tmp_path):
    eval_cli = importlib.import_module("eval")
    for name in ("ckpt_00020.pt", "ckpt_00005.pt", "ckpt_00100.pt", "ckpt_00010.pt", "best.pt", "ckpt_x.pt"):
        (tmp_path / name).write_bytes(b"")
    names = lambda paths: [os.path.basename(p) for p in paths]  # noqa: E731
    assert names(eval_cli.find_checkpoints(str(tmp_path))) == [
        "ckpt_00005.pt", "ckpt_00010.pt", "ckpt_00020.pt", "ckpt_00100.pt"]
    assert names(eval_cli.find_checkpoints(str(tmp_path), 2)) == ["ckpt_00010.pt", "ckpt_00100.pt"]
    assert names(eval_cli.find_checkpoints(str(tmp_path), 3)) == ["ckpt_00005.pt", "ckpt_00100.pt"]


def test_eval_cli_scripted_agents(tmp_path):
    out = tmp_path / "res.json"
    p = run_cli("eval.py", "--agent", "greedy", "--opponents", "random", "--games", "20", "--workers", "1",
                "--json", str(out))
    assert p.returncode == 0, p.stderr
    assert "rounded up" in p.stdout and "C{i,j}" in p.stdout and "P[a][b]" in p.stdout
    assert "greedy vs greedy (reference)" in p.stdout
    assert "PASS" not in p.stdout and "FAIL" not in p.stdout  # no greedy opponent: no verdict
    data = json.loads(out.read_text())
    assert data["target"] is None and len(data["results"]) == 1 and data["games_per_opponent"] == 32
    row = data["results"][0]
    assert row["opponent"] == "random" and row["games"] == 32 and row["n_deals"] == 16 and row["decks"] == "all"
    assert len(row["matchups"]) == 10 and len(row["deck_matrix"]["win_rate"]) == N_DECKS
    assert data["reference"]["agent_a"] == data["reference"]["agent_b"] == "greedy"

    # fewer games than the brief's 2,000: no PASS/FAIL verdict, and --strict refuses to pass
    p = run_cli("eval.py", "--agent", "random", "--games", "10", "--workers", "1", "--strict", "--no-reference")
    assert p.returncode == 1 and "INDICATIVE" in p.stdout and "PASS" not in p.stdout
    p = run_cli("eval.py", "--agent", "random", "--games", "12", "--decks", "sampled", "--workers", "1",
                "--no-reference")
    assert p.returncode == 0 and "INDICATIVE" in p.stdout and "12 games per opponent" in p.stdout
    p = run_cli("eval.py", "--agent", "random", "--games", "11", "--decks", "sampled")
    assert p.returncode == 2 and "even" in p.stderr


def test_eval_cli_checkpoints_and_scenarios(ckpt_dir, tmp_path):
    out = tmp_path / "res.json"
    agent = str(ckpt_dir / "ckpt_00002.pt")
    p = run_cli("eval.py", "--agent", agent, "--deterministic", "--opponents", "greedy", "random",
                "--checkpoints-dir", str(ckpt_dir), "--games", "32", "--workers", "2", "--target", "0.0",
                "--cell-target", "0.0", "--scenarios", "--playouts", "2", "--json", str(out))
    assert p.returncode == 0, p.stderr
    assert "INDICATIVE" in p.stdout and "ckpt_00001.pt" in p.stdout and "== scenarios" in p.stdout
    assert "skipping stage1.pt" not in p.stdout  # only ckpt_*.pt files are picked up
    assert "--target 0 below the brief's 0.7" in p.stdout and "diagnostic only" in p.stdout
    # compact per-opponent cell table: one row per opponent (checkpoints included), one column per cell
    lines = p.stdout.splitlines()
    head = lines.index(next(x for x in lines if x.startswith("== matchup cells C{i,j} per opponent")))
    assert lines[head + 1].split()[0] == "opponent" and len(lines[head + 1].split()) == 11
    rows = {x.split()[0]: x.split()[1:] for x in lines[head + 2:head + 5]}
    assert set(rows) == {"greedy", "random", "ckpt_00001.pt"} and all(len(v) == 10 for v in rows.values())
    assert lines[head + 5].split()[0] == "(games)"
    data = json.loads(out.read_text())
    assert data["agent_kwargs"] == {"deterministic": True}
    labels = [r["label"] for r in data["results"]]
    assert labels == ["greedy", "random", "ckpt_00001.pt"]  # the agent's own file is skipped
    target = data["target"]
    assert target["opponent"] == "greedy" and target["conclusive"] is False and target["passed"] is False
    assert any("deterministic" in r for r in target["indicative_reasons"])
    assert not any("scenarios not run" in r for r in target["indicative_reasons"])
    assert target["worst_ordered_cell"]["diagnostic"] is True and target["worst_ordered_cell"]["games"] > 0
    cells = {tuple(c["decks"]): c for c in data["results"][2]["matchups"]}  # ckpt_00001.pt
    for key, value in zip(cells, rows["ckpt_00001.pt"]):
        assert value == f"{100 * cells[key]['win_rate']:.1f}"
    assert target["scenarios_pct_solved"] == data["scenarios"]["agent"]["pct_solved"]
    for r in data["results"]:
        assert r["games"] == 32 and r["by_seat"]["0"]["games"] == 16 and r["by_turn_order"]["second"]["games"] == 16
    scen = data["scenarios"]
    assert set(scen) == {"agent", "greedy", "random"}
    assert all(x["playouts"] == 2 and x["solved"] in (True, False) for x in scen["agent"]["results"])
    assert all(x["playouts"] == 20 and x["solved"] is None for x in scen["random"]["results"])


def test_eval_cli_skips_stage1_and_incompatible_checkpoints(ckpt_dir, tmp_path):
    """Checkpoints that cannot play this ruleset are reported and skipped, not crashed on."""
    import shutil
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    shutil.copy(ckpt_dir / "ckpt_00001.pt", mixed / "ckpt_00001.pt")
    shutil.copy(ckpt_dir / "stage1.pt", mixed / "ckpt_00009.pt")
    p = run_cli("eval.py", "--agent", "greedy", "--opponents", "random", "--checkpoints-dir", str(mixed),
                "--games", "4", "--decks", "sampled", "--workers", "1", "--no-reference")
    assert p.returncode == 0, p.stderr
    assert "skipping ckpt_00009.pt" in p.stdout and "Stage 1" in p.stdout and "ckpt_00001.pt" in p.stdout
    p = run_cli("eval.py", "--agent", str(mixed / "ckpt_00009.pt"), "--games", "4", "--workers", "1")
    assert p.returncode == 2 and "Stage 1" in p.stderr


STAGE1_BENCH = """python 3.13.5 on arm64 (8 CPUs); 2000 games per row, seeds from 0

benchmark                      games  seconds   games/s    steps/s steps/game rounds/game
engine only, random actions     2000     0.66    3011.5     507067      168.4        23.9
random vs random (play_game)    2000     1.18    1688.2     287969      170.6        24.4
greedy vs greedy (play_game)    2000     1.11    1803.3     251825      139.6        17.6
random vs random, 8 workers     2000     0.25    8073.7    1377202      170.6        24.4
ppo vs greedy (play_game)       2000     3.10     644.5      53795       83.5        11.1

per-call cost (2000 sampled positions)   us/call
Game.observe(player)                        1.34
ObservationEncoder.encode(obs)              5.84
Game.legal_actions() (uncached)             1.04
Game.legal_mask() (uncached)                1.37
Game.clone()                               14.58
"""


def test_parse_bench_output():
    bench = importlib.import_module("bench")
    rows, calls = bench.parse_bench(STAGE1_BENCH)
    assert rows["random vs random, K workers"]["games_per_s"] == 8073.7
    assert rows["engine only, random actions"]["steps_per_s"] == 507067
    assert calls[bench.canonical("ObservationEncoder.encode(obs, mask)")] == 5.84
    assert calls["Game.clone()"] == 14.58 and len(rows) == 5 and len(calls) == 5
    new_header = "python 3.13.5 on x86_64 (32 CPUs; load average 1.0); 100 games per row, seeds from 0"
    text = bench.format_compare(STAGE1_BENCH, [("random vs random, 2 workers", 100, 0.5, 10000, 1500),
                                               ("random vs random, 8 workers", 100, 0.5, 10000, 1500),
                                               ("engine only, random actions", 100, 0.05, 10000, 1500)],
                                [("Game.clone()", 29.16)], "stage1.txt", new_header)
    lines = text.splitlines()
    assert lines[1] == "  old: " + STAGE1_BENCH.splitlines()[0] and lines[2] == "  new: " + new_header
    assert "different setups (machine arm64 vs x86_64, cpus 8 vs 32)" in text
    row = {ln.split("  ")[0]: ln.split() for ln in lines}
    two, eight = row["random vs random, 2 workers"], row["random vs random, 8 workers"]
    assert two[-6:] == ["8073.7", "200.0", "-", "1377202", "20000", "-"]  # worker counts differ: no ratio
    assert eight[-4] == "0.02x" and row["engine only, random actions"][-4] == "0.66x"
    assert "2.00x" in text and "different worker counts" in text
    same = bench.format_compare(STAGE1_BENCH, [], [("Game.clone()", 14.58)], "stage1.txt",
                                "python 3.13.5 on arm64 (8 CPUs; load average 2.0); 2000 games per row")
    assert "different setups" not in same and "warning" not in same and "1.00x" in same


def test_bench_machine_line_and_load_warning():
    bench = importlib.import_module("bench")
    m = bench.parse_machine(bench.machine_line(10, 0, 4))
    assert m is not None and m["cpus"] == os.cpu_count() and m["python"] == sys.version.split()[0]
    assert bench.parse_machine(STAGE1_BENCH.splitlines()[0]) == {"python": "3.13.5", "machine": "arm64", "cpus": 8,
                                                                 "load": None}
    assert bench.load_warning(4.0, 8) is None and bench.load_warning(None, 8) is None
    warn = bench.load_warning(4.1, 8)
    assert warn.startswith("warning: 1-minute load average 4.1 > 0.5 x 8 CPUs")
    busy = STAGE1_BENCH.replace("(8 CPUs)", "(8 CPUs; load average 7.5)")
    assert "load average 7.5 > 0.5 x 8 CPUs on the old run" in bench.format_compare(busy, [], [], "old.txt")


def test_bench_cli(ckpt_dir, tmp_path):
    """--compare and --out without values: Stage 1's results/stage1/bench.txt and results/bench_stage2.txt."""
    stage1 = tmp_path / "results" / "stage1" / "bench.txt"
    stage1.parent.mkdir(parents=True)
    stage1.write_text(STAGE1_BENCH, encoding="utf-8")
    p = subprocess.run([sys.executable, os.path.join(ROOT, "bench.py"), "--games", "8", "--workers", "2",
                        "--states", "30", "--checkpoint", str(ckpt_dir / "ckpt_00001.pt"), "--compare", "--out"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr
    for text in ("games/s", "steps/s", "engine only", "greedy vs greedy", "2 workers", "ppo vs greedy",
                 "Game.observe", "ObservationEncoder.encode(obs, mask)", "Game.clone()",
                 f"comparison with {os.path.join('results', 'stage1', 'bench.txt')}", "ratio",
                 "  old: python 3.13.5 on arm64 (8 CPUs)", "  new: python "):
        assert text in p.stdout
    two = next(ln for ln in p.stdout.splitlines() if ln.startswith("random vs random, 2 workers") and "-" in ln)
    assert two.split()[-1] == "-"  # Stage 1 used 8 workers: no ratio
    written = (tmp_path / "results" / "bench_stage2.txt").read_text(encoding="utf-8")
    assert "comparison with" in written and "ppo vs greedy" in written and "  old: python" in written
    p = subprocess.run([sys.executable, os.path.join(ROOT, "bench.py"), "--compare"], cwd=tmp_path / "results",
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 2 and "results/stage1/bench.txt".replace("/", os.sep) in p.stderr
