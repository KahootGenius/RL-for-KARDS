"""Evaluation tests (SPEC 10/11): duplicate pairing over deck matchups and random decks (each agent plays
both decks of a deal), moves through `choose_action` (the lookahead agent spec), per-seat / turn-order /
cell / P-matrix bookkeeping recomputed from the game records, reproducibility across worker counts, a dying
worker raising instead of hanging, the deck schedules, Wilson intervals, PPO checkpoints (Stage 1 and
Stage 2 ones rejected, the moved Stage 2 artifacts), the four-part "Done when" verdict, and eval.py /
bench.py end to end."""
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

from cardgame.agents import GreedyAgent, LookaheadAgent, RandomAgent, make_agent
from cardgame.cards import deck_rng, generate_deck, load_ruleset, sample_decks
from cardgame.engine import DRAW, Game, IllegalActionError, Observation
from cardgame.evaluation import (DECK_MODES, RANDOM_DECKS, BrokenProcessPool, Evaluator, GameRecord,
                                 IllegalAgentActionError, MatchResult, agent_seed, check_decks, deal_decks,
                                 deal_schedule, deals_for_games, duplicate_match, outcome_for, play_game,
                                 random_deal, round_deals, short_deck_names, wilson_interval)

CONFIG = load_ruleset()  # the shipped ruleset: the mulligan is on, as in training and evaluation
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


class SpyGameAgent(SpyAgent):
    """A simulating spy (`needs_game`): records the observation of the player it is asked to move."""
    needs_game = True

    def act_game(self, game: Game, player: int) -> int:
        self.games[-1].append(game.observe(player))
        return self.inner.act_game(game, player)


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


def synthetic_result(outcome, n_deals: int = 1008, decks="all", agent_b: str = "lookahead") -> MatchResult:
    """A MatchResult from made-up records: outcome(i, j, a_seat, seed) = A's +1/0/-1 with A holding
    deck i = decks[a_seat] against deck j (i = j = -1 for random decks)."""
    deals = []
    for seed, pair in deal_schedule(0, n_deals, decks, N_DECKS):
        pair = (-1, -1) if pair == RANDOM_DECKS else (pair or sample_decks(seed, N_DECKS))
        games = []
        for a_seat in (0, 1):
            out = outcome(pair[a_seat], pair[1 - a_seat], a_seat, seed)
            winner = DRAW if out == 0 else (a_seat if out > 0 else 1 - a_seat)
            games.append(GameRecord(seed, winner, seed % 2, 10, 60, tuple(pair)))
        deals.append(tuple(games))
    return MatchResult("ppo", agent_b, 0, deals, 1.0, {}, {}, decks, CONFIG.deck_names)


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


def test_play_game_asks_simulating_agents_through_choose_action():
    """The lookahead agent (needs_game) gets the game; plain agents get observe() + legal actions."""
    look, rnd = SpyGameAgent(LookaheadAgent(CONFIG)), SpyAgent(RandomAgent(CONFIG))
    rec = play_game(CONFIG, (look, rnd), 5, decks=random_deal(5, CONFIG))
    assert rec.decks == (-1, -1) and rec.winner in (0, 1, DRAW)
    assert rec.steps == len(look.games[-1]) + len(rnd.games[-1]) and look.games[-1] and rnd.games[-1]
    assert all(o.player == 0 for o in look.games[-1]) and all(o.player == 1 for o in rnd.games[-1])
    assert look.seeds == [agent_seed(5, 0)]
    with pytest.raises(TypeError):  # a simulating agent cannot be asked with act() alone
        LookaheadAgent(CONFIG).act(look.games[-1][0], [0])


def test_play_game_reuses_game_and_is_deterministic():
    game = Game(CONFIG)
    agents = (RandomAgent(CONFIG), LookaheadAgent(CONFIG))
    first = [play_game(CONFIG, agents, s, game=game, decks=(s % 4, 3 - s % 4)) for s in range(4)]
    again = [play_game(CONFIG, agents, s, decks=(s % 4, 3 - s % 4)) for s in range(4)]
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
def test_deck_modes():
    assert DECK_MODES == ("random", "all", "sampled") and RANDOM_DECKS == "random"
    assert check_decks("random", N_DECKS) == "random"


def test_decks_all_schedule_and_rounding():
    n2 = N_DECKS * N_DECKS
    assert round_deals(1, "all", N_DECKS) == n2 and round_deals(n2, "all", N_DECKS) == n2
    assert round_deals(n2 + 1, "all", N_DECKS) == 2 * n2 and round_deals(7, "sampled", N_DECKS) == 7
    assert round_deals(7, "random", N_DECKS) == 7
    assert deals_for_games(2000, "all", 4) == 1008  # 2,000 games -> 2,016
    assert deals_for_games(2000, "sampled", 4) == 1000 and deals_for_games(3, "sampled", 4) == 2
    assert deals_for_games(2000, "random", 4) == 1000
    sched = deal_schedule(5, 3 * n2 - 2, "all", N_DECKS)
    assert len(sched) == 3 * n2
    for k, (seed, pair) in enumerate(sched):
        assert seed == 5 + k and pair == divmod(k % n2, N_DECKS)
    counts = Counter(pair for _, pair in sched)
    assert set(counts) == {(i, j) for i in range(N_DECKS) for j in range(N_DECKS)}
    assert set(counts.values()) == {3}  # every ordered pair (mirrors included) equally often
    assert deal_schedule(0, 4, "sampled", N_DECKS) == [(k, None) for k in range(4)]
    assert deal_schedule(2, 2, (1, 2), N_DECKS) == [(2, (1, 2)), (3, (1, 2))]
    assert deal_schedule(7, 3, "random", N_DECKS) == [(7, "random"), (8, "random"), (9, "random")]
    for bad in ("both", (0, N_DECKS), (0,), (-1, 0)):
        with pytest.raises(ValueError):
            check_decks(bad, N_DECKS)


def test_random_deal_is_the_per_seat_deck_stream():
    """SPEC 10: deal k gives seat 0 / seat 1 generate_deck(deck_rng(k, 0 / 1))."""
    for seed in (0, 1, 123456):
        d0, d1 = random_deal(seed, CONFIG)
        assert d0 == generate_deck(deck_rng(seed, 0), CONFIG) and d1 == generate_deck(deck_rng(seed, 1), CONFIG)
        assert len(d0) == len(d1) == CONFIG.deck_size and d0 != d1
        assert deal_decks((seed, RANDOM_DECKS), CONFIG) == (d0, d1)
    assert deal_decks((3, None), CONFIG) is None and deal_decks((3, (1, 2)), CONFIG) == (1, 2)
    assert random_deal(0, CONFIG) != random_deal(1, CONFIG)


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
@pytest.mark.parametrize("decks", ["all", "sampled", "random"])
def test_duplicate_pairs_use_identical_deals_with_swapped_seats(decks):
    start, n = 5, 16
    res = duplicate_match("greedy", "random", n, start_seed=start, decks=decks)
    assert res.n_deals == n and res.games == 2 * n and res.decks == decks
    for k, (g0, g1) in enumerate(res.deals):
        s = start + k
        if decks == "all":
            want, ids = divmod(k % N_DECKS ** 2, N_DECKS), divmod(k % N_DECKS ** 2, N_DECKS)
        elif decks == "sampled":
            want = ids = sample_decks(s, N_DECKS)
        else:
            want, ids = random_deal(s, CONFIG), (-1, -1)
        assert g0.seed == g1.seed == s and g0.decks == g1.decks == ids
        assert g0.first_player == g1.first_player == first_player_of(s)
        # Game 1: A (greedy) in seat 0; game 2: B (random) in seat 0. Replays must match exactly.
        assert g0 == play_game(CONFIG, (GreedyAgent(CONFIG), RandomAgent(CONFIG)), s, decks=want)
        assert g1 == play_game(CONFIG, (RandomAgent(CONFIG), GreedyAgent(CONFIG)), s, decks=want)


def test_random_decks_each_agent_plays_both_decks_of_a_deal():
    """The decks stay with the seats: A holds seat 0's random deck in game 1 and seat 1's in game 2."""
    for seed in range(3):
        decks = random_deal(seed, CONFIG)
        g, r = SpyAgent(GreedyAgent(CONFIG)), SpyAgent(RandomAgent(CONFIG))
        play_game(CONFIG, (g, r), seed, decks=decks)
        play_game(CONFIG, (r, g), seed, decks=decks)
        counts = [tuple(Counter(d).get(c, 0) for c in range(len(CONFIG.cards))) for d in decks]
        assert g.games[0][0].my_decklist == counts[0] and g.games[1][0].my_decklist == counts[1]
        assert r.games[0][0].my_decklist == counts[1] and r.games[1][0].my_decklist == counts[0]
        assert g.games[0][0].my_deck == -1  # a random deck has no fixed index
        assert g.games[0][0].hand == r.games[1][0].hand  # same seat, same shuffle, whoever sits there


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


@pytest.mark.parametrize("spec,decks", [("greedy", "all"), ("random", "all"), ("lookahead", "random"),
                                        ("lookahead", "all")])
def test_mirror_match_is_exactly_balanced(spec, decks):
    """Same agent on both sides + per-seat agent seeds => both games of a deal are identical."""
    res = duplicate_match(spec, spec, N_DECKS ** 2, start_seed=100, decks=decks)
    for g0, g1 in res.deals:
        assert g0 == g1
    assert res.wins == res.losses
    ds = res.deal_summary()
    assert ds["a_won_both"] == ds["b_won_both"] == 0
    assert ds["split"] + ds["with_draw"] == res.n_deals
    assert res.by_seat[0].wins == res.by_seat[1].losses
    for w in res.cells.values():
        assert w.wins == w.losses
    for a in range(len(res.matrix)):
        for b in range(N_DECKS):
            assert res.matrix[a][b].wins == res.matrix[b][a].losses


# ---------------------------------------------------------------------- bookkeeping
@pytest.mark.parametrize("spec_a,spec_b,decks", [("greedy", "random", "all"), ("random", "greedy", "sampled"),
                                                  ("random", "random", "all"), ("lookahead", "random", "random")])
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
    assert res.by_seat[0].games == res.by_seat[1].games == n  # A sits in each seat once per deal
    assert res.first.games == res.second.games == n  # same coin flip, swapped seats
    assert res.win_rate + res.draw_rate + res.loss_rate == pytest.approx(1.0)
    assert sum(res.deal_summary().values()) == n
    d = res.to_dict(include_games=True)
    json.dumps(d)  # serialisable
    assert d["games"] == 2 * n and d["wins"] == res.wins and d["agent_a"] == spec_a and d["decks"] == decks
    assert d["by_seat"]["0"]["games"] == n and d["by_turn_order"]["first"]["games"] == n
    assert d["win_rate_ci95"] == list(wilson_interval(res.wins, res.games))
    assert len(d["games_played"]) == 2 * n and all(len(g["decks"]) == 2 for g in d["games_played"])
    text = res.format()
    for word in ("overall", "seat 0", "seat 1", "first", "second", "95% CI", spec_b):
        assert word in text

    if decks == "random":  # no fixed decks: overall / per-seat / turn order only
        assert not res.fixed_decks and res.cells == {} and res.matrix == []
        assert res.min_cell() is None and res.min_ordered_cell() is None
        assert d["matchups"] == [] and d["min_cell"] is None and d["deck_matrix"] is None and not d["fixed_decks"]
        assert all(g["decks"] == [-1, -1] for g in d["games_played"])
        assert "no matchup cells" in text and "decks=random" in text and "C{i,j}" not in text
        assert "no matchup cells" in res.format_cells() and "no deck matrix" in res.format_matrix()
        return
    for (i, j), w in res.cells.items():
        same(w, [(s, r) for s, r in records if sorted(r.decks) == [i, j]])
    for a in range(N_DECKS):
        for b in range(N_DECKS):
            same(res.matrix[a][b], [(s, r) for s, r in records if (r.decks[s], r.decks[1 - s]) == (a, b)])
    assert sum(w.games for w in res.cells.values()) == res.games == 2 * n
    assert sum(w.games for row in res.matrix for w in row) == res.games
    assert len(d["matchups"]) == len(res.cells) and d["deck_names"] == list(CONFIG.deck_names) and d["fixed_decks"]
    for cell in d["matchups"]:
        assert cell["games"] == res.cell(*cell["decks"]).games
        assert cell["names"] == [CONFIG.deck_names[k] for k in cell["decks"]]
    assert d["deck_matrix"]["games"] == [[w.games for w in row] for row in res.matrix]
    for word in ("C{i,j}", "P[a][b]", *CONFIG.deck_names):
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
    rnd = MatchResult("a", "b", 0, [(GameRecord(0, 0, 0, 9, 50, (-1, -1)), GameRecord(0, 1, 0, 9, 50, (-1, -1)))],
                      decks="random", deck_names=CONFIG.deck_names)
    assert rnd.wins == 2 and rnd.cells == {} and rnd.by_seat[0].wins == rnd.by_seat[1].wins == 1


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
        l2 = ev.match("lookahead", "random", 6, start_seed=11, decks="random", chunk_size=2)
    assert r1.deals == r1b.deals == r2.deals
    assert counts_without_timing(r1) == counts_without_timing(r2)
    s1 = duplicate_match("greedy", "random", 9, start_seed=3, decks="sampled")
    assert s1.deals == s2.deals
    l1 = duplicate_match("lookahead", "random", 6, start_seed=11, decks="random")
    assert l1.deals == l2.deals  # lookahead determinizes with its per-game, per-seat seed
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
    runner.run(("lookahead", {}, "lookahead", {}, [(0, (0, 1)), (1, RANDOM_DECKS)]))
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
    base = dict(target=0.70, cell_target=0.60, baseline_target=0.50, scenario_target=0.50, seed=0,
                deterministic=False)
    base.update(kw)
    return argparse.Namespace(**base)


def loses_seat0_every_other_block(i, j, seat, seed):
    """Loses its seat-0 game in every other block of 16 deals: 1496/2000 = 74.8% overall on random decks
    (1000 deals), 94/126 = 74.6% in every fixed-deck cell (1008 deals)."""
    return -1 if seat == 0 and (seed // 16) % 2 == 0 else 1


STRONG_RANDOM = synthetic_result(loses_seat0_every_other_block, n_deals=1000, decks="random")
STRONG_ALL = synthetic_result(loses_seat0_every_other_block)
H2H = synthetic_result(lambda i, j, seat, seed: 1 if (seed + seat) % 20 < 11 else -1, n_deals=1000, decks="random",
                       agent_b="pooled.pt")  # 55%
FULL = [("lookahead", "random", STRONG_RANDOM), ("lookahead", "all", STRONG_ALL)]


def verdict(results=FULL, baseline=H2H, args=None, is_ppo=True, scenarios=56.25, agent_kind="transformer",
            baseline_kind="pooled"):
    eval_cli = importlib.import_module("eval")
    return eval_cli.stage3_verdict(results, baseline, args or verdict_args(), is_ppo, scenarios, agent_kind,
                                   baseline_kind)


def test_done_when_verdict_four_parts():
    eval_cli = importlib.import_module("eval")
    assert STRONG_RANDOM.games == 2000 and STRONG_RANDOM.win_rate == pytest.approx(1496 / 2000)
    assert STRONG_ALL.games == 2016 and all(w.win_rate == pytest.approx(94 / 126) for w in STRONG_ALL.cells.values())
    assert H2H.games == 2000 and H2H.win_rate == pytest.approx(0.55)

    check = verdict()
    assert check["conclusive"] and check["passed"] and check["indicative_reasons"] == []
    assert check["measured"] == ["random_decks", "cells", "baseline", "scenarios"] and all(check["checks"].values())
    text = eval_cli.verdict_text(check)
    assert text.startswith("verdict: PASS")
    for part in ("(1) vs lookahead, random decks: win rate 74.8% (>= 70%) over 2000",
                 "(2) vs lookahead, fixed decks (all): worst matchup cell", "(3) head-to-head vs pooled.pt: win rate "
                 "55.0% (> 50%", "(4) scenarios: 56% solved, argmax (> 50%)", "diagnostic only"):
        assert part in text
    json.dumps(check)

    # each part can fail on its own; with all four measured under the brief's settings that is a FAIL
    weak_random = synthetic_result(lambda i, j, seat, seed: 1 if seed % 10 < 6 else -1, n_deals=1000, decks="random")
    weak_cell = synthetic_result(lambda i, j, seat, seed: -1 if {i, j} == {0, 2} else 1)  # 0% in one cell
    even = synthetic_result(lambda i, j, seat, seed: 1 if seat == 0 else -1, n_deals=1000, decks="random")  # 50%
    assert weak_cell.win_rate > 0.7
    for kw, failed in [({"results": [("lookahead", "random", weak_random), FULL[1]]}, "random_decks"),
                       ({"results": [FULL[0], ("lookahead", "all", weak_cell)]}, "cells"),
                       ({"baseline": even}, "baseline"),       # the baseline needs > 50%
                       ({"scenarios": 50.0}, "scenarios")]:    # scenarios need > 50%
        check = verdict(**kw)
        assert check["conclusive"] and not check["passed"], failed
        assert [k for k, v in check["checks"].items() if not v] == [failed]
        assert eval_cli.verdict_text(check).startswith("verdict: FAIL") and "[MISSED]" in eval_cli.verdict_text(check)
    names = CONFIG.deck_names
    check = verdict(results=[FULL[0], ("lookahead", "all", weak_cell)])
    assert check["parts"]["cells"]["worst_cell"]["name"] == f"{names[0]}-{names[2]}"


def test_verdict_is_indicative_unless_measured_with_the_briefs_settings():
    eval_cli = importlib.import_module("eval")
    small = synthetic_result(lambda i, j, seat, seed: 1, n_deals=16, decks="random")
    fixed_h2h = synthetic_result(lambda i, j, seat, seed: 1, n_deals=1008, agent_b="pooled.pt")
    cases = [
        ({"scenarios": None}, "(4) not measured: scenarios not run (--scenarios)"),
        ({"baseline": None}, "(3) not measured: no baseline (--baseline <pooled.pt>)"),
        ({"results": FULL[1:]}, "(1) not measured: no random-deck match vs lookahead (--decks random)"),
        ({"results": FULL[:1]}, "(2) not measured: no fixed-deck match vs lookahead (--decks all)"),
        ({"results": [("greedy", m, r) for _, m, r in FULL]}, "(1) not measured"),  # another opponent
        ({"results": [("lookahead", "random", small), FULL[1]]}, "(1) 32 games < 2000"),
        ({"baseline": small}, "(3) 32 games < 2000"),
        ({"baseline": fixed_h2h}, "(3) decks=all (the test uses random decks)"),
        ({"baseline_kind": "transformer"}, "(3) the baseline is a 'transformer' network, not the pooled network"),
        ({"agent_kind": "pooled"}, "(3) the agent is a 'pooled' network, not a Transformer network"),
        ({"agent_kind": None}, "(3) the agent is a scripted agent"),
        ({"args": verdict_args(deterministic=True)}, "--deterministic (the test samples from the policy)"),
        ({"args": verdict_args(seed=5)}, "seeds start at 5"),
        ({"args": verdict_args(target=0.69)}, "--target 0.69 below the brief's 0.7"),
        ({"args": verdict_args(cell_target=0.5)}, "--cell-target 0.5 below"),
        ({"args": verdict_args(baseline_target=0.4)}, "--baseline-target 0.4 below"),
        ({"args": verdict_args(scenario_target=0.25)}, "--scenario-target 0.25 below"),
    ]
    for kw, reason in cases:
        check = verdict(**kw)
        assert not check["conclusive"] and not check["passed"], reason
        assert reason in " | ".join(check["indicative_reasons"]), (reason, check["indicative_reasons"])
        assert all(check["checks"].values())  # what was measured passes; the settings make it indicative
        assert eval_cli.verdict_text(check).startswith("verdict: INDICATIVE (")
    check = verdict(scenarios=None, baseline=None)
    assert "not measured" in eval_cli.verdict_text(check) and check["measured"] == ["random_decks", "cells"]
    # a scripted agent: --deterministic does not apply
    assert verdict(args=verdict_args(deterministic=True), is_ppo=False, agent_kind="transformer")["conclusive"]
    stricter = verdict(args=verdict_args(target=0.72, cell_target=0.7, baseline_target=0.52, scenario_target=0.55))
    assert stricter["conclusive"] and stricter["passed"]  # raising the bar keeps the verdict conclusive
    assert verdict(results=[], baseline=None, scenarios=None) is None  # nothing measured: no verdict
    assert verdict(results=[("lookahead", "random", STRONG_RANDOM)], baseline=None, scenarios=None) is not None


def test_verdict_reports_the_worst_ordered_cell_as_a_diagnostic():
    """SPEC 10: the worst ordered cell P[a][b] is shown next to the verdict but never decides it."""
    eval_cli = importlib.import_module("eval")
    names = CONFIG.deck_names
    # loses only holding deck 1 in seat 0 against deck 3: P[1][3] = 50%, while C{1,3} = 75% (worst cell)
    res = synthetic_result(lambda i, j, seat, seed: -1 if (i, j) == (1, 3) and seat == 0 else 1)
    assert res.matrix[1][3].win_rate == 0.5 and res.min_cell()[1].win_rate == pytest.approx(0.75)
    check = verdict(results=[FULL[0], ("lookahead", "all", res)])
    assert check["passed"] and check["conclusive"]  # an ordered cell below 60% does not fail the test
    wo = check["parts"]["cells"]["worst_ordered_cell"]
    assert wo["diagnostic"] is True and wo["decks"] == [1, 3] and wo["win_rate"] == 0.5
    assert wo["name"] == f"P[{names[1]}][{names[3]}]" and wo["games"] == res.matrix[1][3].games == 126
    assert (wo["agent_deck"], wo["opponent_deck"]) == (names[1], names[3])
    lines = eval_cli.verdict_text(check).split("\n")
    assert lines[0] == "verdict: PASS" and all("P[" not in ln for ln in lines[:-1])
    assert "diagnostic" in lines[-1] and f"P[{names[1]}][{names[3]}] = 50.0% over 126 games" in lines[-1]
    json.dumps(check)


def test_cells_table_and_short_deck_names():
    eval_cli = importlib.import_module("eval")
    assert short_deck_names(("Blitz", "Bulwark", "Volley", "Legion")) == ("Bl", "Bu", "Vo", "Le")
    assert short_deck_names(("aaa1", "aaa2")) == ("aaa1", "aaa2") and short_deck_names(("x", "y")) == ("x", "y")
    a = synthetic_result(lambda i, j, seat, seed: 1)
    b = synthetic_result(lambda i, j, seat, seed: -1 if {i, j} == {0, 2} else 1)
    lines = eval_cli.cells_table([("lookahead", a), ("ckpt_00001.pt", b)], 14).splitlines()
    short = short_deck_names(CONFIG.deck_names)
    titles = lines[0].split()
    assert titles[0] == "opponent" and len(titles) == 1 + len(a.cells)
    assert titles[1:] == [f"{short[i]}-{short[j]}" for i, j in a.cells]
    assert lines[1].split() == ["lookahead"] + ["100.0"] * len(a.cells)
    row = dict(zip(titles[1:], lines[2].split()[1:]))
    assert row[f"{short[0]}-{short[2]}"] == "0.0" and row[f"{short[0]}-{short[1]}"] == "100.0"
    assert lines[3].split() == ["(games)"] + [str(w.games) for w in a.cells.values()]
    row = eval_cli.table_row("lookahead", STRONG_RANDOM, 10).split()
    assert row[:3] == ["lookahead", "random", "2000"] and row[-2] == "-"  # no worst cell on random decks


# ---------------------------------------------------------------------- PPO checkpoints
@pytest.fixture(scope="module")
def ckpt_dir(tmp_path_factory):
    """Tiny randomly initialised Stage 3 checkpoints (SPEC 7/8 format: two Transformer ckpt_*.pt files and a
    pooled baseline), plus a Stage 1 and a Stage 2 checkpoint (rejected). Built with cardgame.rl.network only."""
    torch = pytest.importorskip("torch")
    from cardgame.features import ObservationEncoder
    from cardgame.rl.network import PolicyValueNet, PooledPolicyNet, TransformerPolicyNet

    d = tmp_path_factory.mktemp("ckpts")
    layout = ObservationEncoder(CONFIG).layout()
    for update in (1, 2):
        torch.manual_seed(update)
        net = TransformerPolicyNet(layout, d_model=16, layers=1, heads=2, ff=32, id_dim=4)
        torch.save({"model": net.state_dict(), "net": net.spec(), "update": update, "env_steps": 0,
                    "args": {}}, d / f"ckpt_{update:05d}.pt")
    torch.manual_seed(3)
    pooled = PooledPolicyNet(layout, d_model=16, ctx_dim=32, id_dim=4)
    torch.save({"model": pooled.state_dict(), "net": pooled.spec(), "update": 3}, d / "pooled.pt")
    stage1 = PolicyValueNet(393, 71, (16,))
    torch.save({"model": stage1.state_dict(), "net": {"obs_dim": 393, "n_actions": 71, "hidden": [16]},
                "update": 3, "env_steps": 0, "args": {}}, d / "stage1.pt")
    torch.save({"model": {}, "net": {"kind": "entity", "obs_dim": 900}, "update": 4}, d / "stage2.pt")
    return d


def test_ppo_checkpoint_loads_and_plays_legal_games(ckpt_dir):
    for name in ("ckpt_00001.pt", "pooled.pt"):
        path = str(ckpt_dir / name)
        for spec in (path, "ppo:" + path):
            for kwargs in ({}, {"deterministic": True}):
                agent = make_agent(spec, CONFIG, **kwargs)
                for seed in range(2):  # play_game raises IllegalAgentActionError on any illegal action
                    for seats in ((agent, LookaheadAgent(CONFIG)), (RandomAgent(CONFIG), agent)):
                        decks = random_deal(seed, CONFIG) if seed else (seed, 3 - seed)
                        rec = play_game(CONFIG, seats, seed, decks=decks)
                        assert rec.winner in (0, 1, DRAW)
    eval_cli = importlib.import_module("eval")
    assert eval_cli.checkpoint_check(str(ckpt_dir / "ckpt_00001.pt"), CONFIG) == (None, "transformer")
    assert eval_cli.checkpoint_check(str(ckpt_dir / "pooled.pt"), CONFIG) == (None, "pooled")
    assert eval_cli.checkpoint_check("lookahead", CONFIG) == (None, None)


def test_stage1_and_stage2_checkpoints_are_rejected(ckpt_dir):
    with pytest.raises(ValueError, match="Stage 1"):
        make_agent(str(ckpt_dir / "stage1.pt"), CONFIG)
    with pytest.raises(ValueError, match="Stage 2"):
        make_agent(str(ckpt_dir / "stage2.pt"), CONFIG)
    shipped = os.path.join(ROOT, "models", "stage1", "ppo_final.pt")  # the Stage 1 model, if still shipped
    if os.path.isfile(shipped):
        with pytest.raises(ValueError, match="Stage 1"):
            make_agent(shipped, CONFIG)
    for name in ("ppo_s2b.pt", "ppo_laptop.pt"):  # the Stage 2 models, moved like Stage 1's
        shipped = os.path.join(ROOT, "models", "stage2", name)
        if os.path.isfile(shipped):
            with pytest.raises(ValueError, match="Stage 2"):
                make_agent(shipped, CONFIG)


def test_stage2_artifacts_live_in_their_stage_folders():
    """Stage 2 outputs moved to models/stage2/ and results/stage2/ (as Stage 1's to */stage1/)."""
    for old in ("models/stage2.pt", "models/stage2_mac.pt", "results/stage2_eval.json", "results/stage2_h2h.json",
                "results/stage2_mac_eval.json", "results/stage2_mac_matches.json", "results/bench_stage2.txt"):
        assert not os.path.exists(os.path.join(ROOT, *old.split("/"))), old
    bench_txt = os.path.join(ROOT, "results", "stage2", "bench.txt")
    if os.path.isfile(bench_txt):  # the Stage 2 report bench.py --compare reads by default
        bench = importlib.import_module("bench")
        with open(bench_txt, encoding="utf-8") as f:
            rows, calls = bench.parse_bench(f.read())
        assert rows and "Game.clone()" in calls
        assert bench.DEFAULT_COMPARE == os.path.join("results", "stage2", "bench.txt")
    for name in ("stage2_eval.json", "stage2_h2h.json"):
        path = os.path.join(ROOT, "results", "stage2", name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                json.load(f)


def test_ppo_duplicate_match_is_reproducible_across_workers(ckpt_dir):
    path = str(ckpt_dir / "ckpt_00002.pt")
    r1 = duplicate_match(path, "lookahead", 4, start_seed=21, workers=1, decks="random")
    r2 = duplicate_match(path, "lookahead", 4, start_seed=21, workers=2, decks="random")
    assert r1.deals == r2.deals  # stochastic policy, but seeded per game and seat
    det = duplicate_match(path, "ppo:" + str(ckpt_dir / "pooled.pt"), 2, workers=1, decks=(2, 0),
                          agent_kwargs_a={"deterministic": True})
    assert det.kwargs_a == {"deterministic": True} and det.games == 4
    assert all(g.decks == (2, 0) for pair in det.deals for g in pair)


# ---------------------------------------------------------------------- command lines
def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=900)


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
    assert "lookahead vs lookahead (reference)" in p.stdout
    assert "verdict" not in p.stdout  # no lookahead opponent, baseline or scenarios: nothing to judge
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["verdict"] is None and data["decks"] == ["random", "all"] and len(data["results"]) == 2
    assert data["games_per_opponent"] == {"random": 20, "all": 32}
    rnd, fixed = data["results"]
    assert rnd["opponent"] == fixed["opponent"] == "random" and rnd["decks"] == "random" and fixed["decks"] == "all"
    assert rnd["games"] == 20 and rnd["matchups"] == [] and rnd["deck_matrix"] is None
    assert fixed["games"] == 32 and fixed["n_deals"] == 16 and len(fixed["matchups"]) == 10
    assert len(fixed["deck_matrix"]["win_rate"]) == N_DECKS
    assert data["reference"]["all"]["agent_a"] == data["reference"]["all"]["agent_b"] == "lookahead"
    assert data["baseline"] is None and data["scenarios"] is None

    # fewer games than the brief's 2,000: no PASS/FAIL verdict, and --strict refuses to pass
    p = run_cli("eval.py", "--agent", "random", "--games", "10", "--workers", "1", "--strict", "--no-reference")
    assert p.returncode == 1 and "INDICATIVE" in p.stdout and "verdict: PASS" not in p.stdout
    assert "(1) 10 games < 2000" in p.stdout and "(3) not measured" in p.stdout
    p = run_cli("eval.py", "--agent", "random", "--games", "12", "--decks", "sampled", "--workers", "1",
                "--no-reference")  # sampled decks measure no part of the verdict
    assert p.returncode == 0 and "verdict" not in p.stdout and "decks=sampled: 12 games per opponent" in p.stdout
    p = run_cli("eval.py", "--agent", "random", "--games", "11", "--decks", "random")
    assert p.returncode == 2 and "even" in p.stderr
    p = run_cli("eval.py", "--agent", "random", "--games", "11", "--decks", "all", "--no-reference",
                "--workers", "1", "--opponents", "random")
    assert p.returncode == 0 and "22 games" not in p.stdout and "32 games per opponent" in p.stdout


def test_eval_cli_full_verdict_command(ckpt_dir, tmp_path):
    """The one command of the brief (small): agent + --baseline + --scenarios + --checkpoints-dir + --json."""
    out = tmp_path / "res.json"
    agent, pooled = str(ckpt_dir / "ckpt_00002.pt"), str(ckpt_dir / "pooled.pt")
    p = run_cli("eval.py", "--agent", agent, "--baseline", pooled, "--scenarios", "--playouts", "2",
                "--opponents", "lookahead", "random", "--checkpoints-dir", str(ckpt_dir), "--games", "32",
                "--workers", "2", "--json", str(out))
    assert p.returncode == 0, p.stderr
    assert "[transformer network]" in p.stdout and "[pooled network]" in p.stdout
    assert "verdict: INDICATIVE" in p.stdout and "(1) 32 games < 2000" in p.stdout and "== scenarios" in p.stdout
    assert "pooled.pt (baseline)" in p.stdout and "skipping stage" not in p.stdout  # only ckpt_*.pt files
    assert "diagnostic only" in p.stdout and "lookahead" in p.stdout
    lines = p.stdout.splitlines()
    head = lines.index(next(x for x in lines if x.startswith("== decks=all: matchup cells C{i,j} per opponent")))
    assert lines[head + 1].split()[0] == "opponent" and len(lines[head + 1].split()) == 11
    rows = {x.split()[0]: x.split()[1:] for x in lines[head + 2:head + 5]}
    assert set(rows) == {"lookahead", "random", "ckpt_00001.pt"} and all(len(v) == 10 for v in rows.values())
    assert lines[head + 5].split()[0] == "(games)"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["agent_kind"] == "transformer" and data["agent_kwargs"] == {}
    labels = [(r["label"], r["decks"]) for r in data["results"]]
    assert labels == [(x, m) for m in ("random", "all") for x in ("lookahead", "random", "ckpt_00001.pt")]
    for r in data["results"]:
        assert r["games"] == 32 and r["by_seat"]["0"]["games"] == 16
    base = data["baseline"]
    assert base["kind"] == "pooled" and base["decks"] == "random" and base["games"] == 32 and base["agent_b"] == pooled
    v = data["verdict"]
    assert v["conclusive"] is False and v["passed"] is False and v["measured"] == list(
        ("random_decks", "cells", "baseline", "scenarios"))
    assert all(f"({k}) 32 games < 2000" in " ".join(v["indicative_reasons"]) for k in (1, 2, 3))
    assert not any("kind" in r or "not the pooled" in r or "Transformer" in r for r in v["indicative_reasons"])
    assert v["parts"]["baseline"]["games"] == 32 and v["parts"]["baseline"]["baseline_kind"] == "pooled"
    assert v["parts"]["scenarios"]["pct_solved"] == data["scenarios"]["agent"]["pct_solved"]
    assert v["parts"]["cells"]["worst_ordered_cell"]["diagnostic"] is True
    scen = data["scenarios"]
    assert set(scen) == {"agent", "lookahead", "random", "greedy"} and scen["agent"]["n"] == 20  # SPEC 10
    assert all(x["playouts"] == 2 and x["solved"] in (True, False) for x in scen["agent"]["results"])
    for ref in ("lookahead", "greedy"):  # one deterministic run each; greedy is the reference
        assert all(x["playouts"] == 0 and x["solved"] in (True, False) for x in scen[ref]["results"])
    assert all(x["playouts"] == 20 and x["solved"] is None for x in scen["random"]["results"])
    table = lines[lines.index(next(x for x in lines if x.startswith("== scenarios"))) + 1].split()
    assert table == ["scenario", "agent", "sampled", "lookahead", "random", "greedy", "tags"]
    # --deterministic applies to both networks and makes the verdict indicative
    p = run_cli("eval.py", "--agent", agent, "--baseline", pooled, "--deterministic", "--decks", "random",
                "--games", "4", "--workers", "1", "--json", str(out))
    assert p.returncode == 0, p.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["agent_kwargs"] == {"deterministic": True} and data["baseline"]["kwargs"] == {"deterministic": True}
    assert any("deterministic" in r for r in data["verdict"]["indicative_reasons"])
    assert data["verdict"]["parts"]["cells"] is None


def test_eval_cli_skips_stage1_and_incompatible_checkpoints(ckpt_dir, tmp_path):
    """Checkpoints that cannot play this ruleset are reported and skipped, not crashed on."""
    import shutil
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    shutil.copy(ckpt_dir / "ckpt_00001.pt", mixed / "ckpt_00001.pt")
    shutil.copy(ckpt_dir / "stage1.pt", mixed / "ckpt_00009.pt")
    shutil.copy(ckpt_dir / "stage2.pt", mixed / "ckpt_00010.pt")
    p = run_cli("eval.py", "--agent", "greedy", "--opponents", "random", "--checkpoints-dir", str(mixed),
                "--games", "4", "--decks", "sampled", "--workers", "1", "--no-reference")
    assert p.returncode == 0, p.stderr
    assert "skipping ckpt_00009.pt" in p.stdout and "Stage 1" in p.stdout and "ckpt_00001.pt" in p.stdout
    assert "skipping ckpt_00010.pt" in p.stdout and "Stage 2" in p.stdout
    p = run_cli("eval.py", "--agent", str(mixed / "ckpt_00009.pt"), "--games", "4", "--workers", "1")
    assert p.returncode == 2 and "Stage 1" in p.stderr
    p = run_cli("eval.py", "--agent", "random", "--baseline", str(mixed / "ckpt_00010.pt"), "--games", "4")
    assert p.returncode == 2 and "baseline" in p.stderr and "Stage 2" in p.stderr
    p = run_cli("eval.py", "--agent", "random", "--baseline", str(tmp_path / "missing.pt"), "--games", "4")
    assert p.returncode == 2 and "checkpoint not found" in p.stderr


STAGE2_BENCH = """python 3.13.5 on arm64 (8 CPUs; load average 2.0); 2000 games per row, seeds from 0; decks sampled per seed from 4

benchmark                      games  seconds   games/s    steps/s steps/game rounds/game
engine only, random actions     2000     1.31    1525.1     184192      120.8        15.8
random vs random (play_game)    2000     2.97     672.8      79963      118.8        15.6
greedy vs greedy (play_game)    2000     2.29     875.2      63887       73.0        10.2
random vs random, 8 workers     2000     0.79    2534.0     301148      118.8        15.6
ppo vs greedy (play_game)       2000     3.10     644.5      53795       83.5        11.1

per-call cost (2000 sampled positions)   us/call
Game.observe(player)                        6.52
ObservationEncoder.encode(obs, mask)       50.53
Game.legal_actions() (uncached)             3.64
Game.legal_mask() (uncached)                4.33
Game.clone()                               27.62
"""


def test_parse_bench_output():
    bench = importlib.import_module("bench")
    rows, calls = bench.parse_bench(STAGE2_BENCH)
    assert rows["random vs random, K workers"]["games_per_s"] == 2534.0
    assert rows["engine only, random actions"]["steps_per_s"] == 184192
    assert calls[bench.canonical("ObservationEncoder.encode(obs, mask)")] == 50.53
    assert calls["Game.clone()"] == 27.62 and len(rows) == 5 and len(calls) == 5
    new_header = "python 3.13.5 on x86_64 (32 CPUs; load average 1.0); 100 games per row, seeds from 0"
    text = bench.format_compare(STAGE2_BENCH, [("random vs random, 2 workers", 100, 0.5, 10000, 1500),
                                               ("random vs random, 8 workers", 100, 0.5, 10000, 1500),
                                               ("engine only, random actions", 100, 0.05, 10000, 1500),
                                               ("lookahead vs lookahead (play_game)", 100, 1.0, 8000, 1000)],
                                [("Game.clone()", 55.24), ("Game.determinize(player, rng)", 80.0)], "stage2.txt",
                                new_header)
    lines = text.splitlines()
    assert lines[1] == "  old: " + STAGE2_BENCH.splitlines()[0] and lines[2] == "  new: " + new_header
    assert "different setups (machine arm64 vs x86_64, cpus 8 vs 32)" in text
    row = {ln.split("  ")[0]: ln.split() for ln in lines}
    two, eight = row["random vs random, 2 workers"], row["random vs random, 8 workers"]
    assert two[-6:] == ["2534.0", "200.0", "-", "301148", "20000", "-"]  # worker counts differ: no ratio
    assert eight[-4] == "0.08x" and row["engine only, random actions"][-4] == "1.31x"
    assert row["lookahead vs lookahead (play_game)"][-6:] == ["-", "100.0", "-", "-", "8000", "-"]  # new row
    assert "2.00x" in text and "different worker counts" in text
    assert row["Game.determinize(player, rng)"][-3:] == ["-", "80.00", "-"]  # no Stage 2 counterpart
    same = bench.format_compare(STAGE2_BENCH, [], [("Game.clone()", 27.62)], "stage2.txt",
                                "python 3.13.5 on arm64 (8 CPUs; load average 2.0); 2000 games per row")
    assert "different setups" not in same and "warning" not in same and "1.00x" in same


def test_bench_machine_line_and_load_warning():
    bench = importlib.import_module("bench")
    m = bench.parse_machine(bench.machine_line(10, 0, 4))
    assert m is not None and m["cpus"] == os.cpu_count() and m["python"] == sys.version.split()[0]
    assert bench.parse_machine(STAGE2_BENCH.splitlines()[0]) == {"python": "3.13.5", "machine": "arm64", "cpus": 8,
                                                                 "load": 2.0}
    assert bench.load_warning(4.0, 8) is None and bench.load_warning(None, 8) is None
    warn = bench.load_warning(4.1, 8)
    assert warn.startswith("warning: 1-minute load average 4.1 > 0.5 x 8 CPUs")
    busy = STAGE2_BENCH.replace("load average 2.0", "load average 7.5")
    assert "load average 7.5 > 0.5 x 8 CPUs on the old run" in bench.format_compare(busy, [], [], "old.txt")


def test_bench_cli(ckpt_dir, tmp_path):
    """--compare and --out without values: Stage 2's results/stage2/bench.txt and results/bench_stage3.txt."""
    bench = importlib.import_module("bench")
    assert bench.DEFAULT_OUT == os.path.join("results", "bench_stage3.txt")
    assert bench.DEFAULT_COMPARE == os.path.join("results", "stage2", "bench.txt")
    stage2 = tmp_path / "results" / "stage2" / "bench.txt"
    stage2.parent.mkdir(parents=True)
    stage2.write_text(STAGE2_BENCH, encoding="utf-8")
    p = subprocess.run([sys.executable, os.path.join(ROOT, "bench.py"), "--games", "6", "--workers", "2",
                        "--states", "20", "--checkpoint", str(ckpt_dir / "ckpt_00001.pt"), "--compare", "--out"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stderr
    for text in ("games/s", "steps/s", "engine only", "greedy vs greedy", "lookahead vs lookahead (play_game)",
                 "lookahead vs random (play_game)", "lookahead vs lookahead, 2 workers", "2 workers",
                 "ppo vs lookahead (play_game)", "ppo vs lookahead, 2 workers", "Game.observe",
                 "ObservationEncoder.encode(obs, mask)", "Game.clone()", "Game.determinize(player, rng)",
                 "LookaheadAgent.act_game(game, player)",
                 f"comparison with {os.path.join('results', 'stage2', 'bench.txt')}", "ratio",
                 "  old: python 3.13.5 on arm64 (8 CPUs; load average 2.0)", "  new: python "):
        assert text in p.stdout, text
    two = next(ln for ln in p.stdout.splitlines() if ln.startswith("random vs random, 2 workers") and "-" in ln)
    assert two.split()[-1] == "-"  # Stage 2 used 8 workers: no ratio
    written = (tmp_path / "results" / "bench_stage3.txt").read_text(encoding="utf-8")
    assert "comparison with" in written and "ppo vs lookahead" in written and "  old: python" in written
    assert "Game.determinize" in written
    p = subprocess.run([sys.executable, os.path.join(ROOT, "bench.py"), "--compare"], cwd=tmp_path / "results",
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 2 and "results/stage2/bench.txt".replace("/", os.sep) in p.stderr
