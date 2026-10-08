"""Evaluation tests: duplicate pairing, counts, reproducibility across worker counts, Wilson intervals,
PPO checkpoints through make_agent, and the eval.py / bench.py command lines end to end."""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from typing import List, Optional, Sequence

import pytest

from cardgame.agents import GreedyAgent, RandomAgent, make_agent
from cardgame.cards import load_ruleset
from cardgame.engine import DRAW, Game, IllegalActionError, Observation
from cardgame.evaluation import (Evaluator, GameRecord, IllegalAgentActionError, MatchResult, agent_seed,
                                 duplicate_match, outcome_for, play_game, wilson_interval)

CONFIG = load_ruleset()
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


# ---------------------------------------------------------------------- play_game
def test_play_game_record_and_agent_view():
    seed = 11
    a, b = SpyAgent(GreedyAgent(CONFIG)), SpyAgent(RandomAgent(CONFIG))
    rec = play_game(CONFIG, (a, b), seed)
    assert isinstance(rec, GameRecord)
    assert rec.seed == seed and rec.first_player == first_player_of(seed)
    assert rec.winner in (0, 1, DRAW) and 1 <= rec.rounds <= CONFIG.max_rounds
    assert rec.steps == len(a.games[-1]) + len(b.games[-1])
    assert a.seeds == [agent_seed(seed, 0)] and b.seeds == [agent_seed(seed, 1)]
    for seat, spy in enumerate((a, b)):
        assert spy.games[-1], "each seat acts at least once"
        for obs in spy.games[-1]:
            assert obs.player == seat and obs.is_my_turn and not obs.done


def test_play_game_reuses_game_and_is_deterministic():
    game = Game(CONFIG)
    agents = (RandomAgent(CONFIG), RandomAgent(CONFIG))
    first = [play_game(CONFIG, agents, s, game=game) for s in range(6)]
    again = [play_game(CONFIG, agents, s) for s in range(6)]
    assert first == again
    custom = play_game(CONFIG, agents, 3, agent_seeds=(1, 2))
    assert custom == play_game(CONFIG, agents, 3, agent_seeds=(1, 2))


@pytest.mark.parametrize("bad", [999, -1, 70, None, "0"])
def test_illegal_agent_action_raises_clear_error(bad):
    with pytest.raises(IllegalAgentActionError, match="fixed") as exc:
        play_game(CONFIG, (FixedActionAgent(bad), GreedyAgent(CONFIG)), 0)
    assert isinstance(exc.value, IllegalActionError)
    assert "seat" in str(exc.value) and "legal" in str(exc.value)


def test_agent_seed_is_deterministic_and_distinct():
    seeds = {agent_seed(s, p) for s in range(500) for p in (0, 1)}
    assert len(seeds) == 1000
    assert all(0 <= x < 2 ** 63 for x in seeds)
    assert agent_seed(7, 1) == agent_seed(7, 1) and agent_seed(-3, 0) >= 0


# ---------------------------------------------------------------------- duplicate pairing
def test_duplicate_pairs_use_identical_deals_with_swapped_seats():
    start, n = 5, 12
    res = duplicate_match("greedy", "random", n, start_seed=start)
    assert res.n_deals == n and res.games == 2 * n
    for i, (g0, g1) in enumerate(res.deals):
        s = start + i
        assert g0.seed == g1.seed == s
        assert g0.first_player == g1.first_player == first_player_of(s)
        # Game 1: A (greedy) in seat 0; game 2: B (random) in seat 0. Replays must match exactly.
        assert g0 == play_game(CONFIG, (GreedyAgent(CONFIG), RandomAgent(CONFIG)), s)
        assert g1 == play_game(CONFIG, (RandomAgent(CONFIG), GreedyAgent(CONFIG)), s)


def test_swapped_games_deal_the_same_hands():
    """Each seat is dealt the same cards in both games of a deal, whoever sits there."""
    for seed in range(8):
        g, r = SpyAgent(GreedyAgent(CONFIG)), SpyAgent(RandomAgent(CONFIG))
        play_game(CONFIG, (g, r), seed)
        play_game(CONFIG, (r, g), seed)
        for seat in (0, 1):
            game1 = (g, r)[seat].games[0][0]
            game2 = (r, g)[seat].games[1][0]
            assert game1.player == game2.player == seat
            assert game1.hand == game2.hand and game1.went_first == game2.went_first
            assert game1.my_deck_size == game2.my_deck_size


@pytest.mark.parametrize("spec", ["greedy", "random"])
def test_mirror_match_is_exactly_balanced(spec):
    """Same agent on both sides + per-seat agent seeds => both games of a deal are identical."""
    res = duplicate_match(spec, spec, 40, start_seed=100)
    for g0, g1 in res.deals:
        assert g0 == g1
    assert res.wins == res.losses
    ds = res.deal_summary()
    assert ds["a_won_both"] == ds["b_won_both"] == 0
    assert ds["split"] + ds["with_draw"] == res.n_deals
    assert res.by_seat[0].wins == res.by_seat[1].losses


# ---------------------------------------------------------------------- counting
@pytest.mark.parametrize("spec_a,spec_b", [("greedy", "random"), ("random", "greedy"), ("random", "random")])
def test_counts_add_up(spec_a, spec_b):
    res = duplicate_match(spec_a, spec_b, 25, start_seed=40)
    n = res.n_deals
    seat0, seat1, first, second, total = res.by_seat[0], res.by_seat[1], res.first, res.second, res.overall
    for attr in ("wins", "draws", "losses"):
        assert getattr(seat0, attr) + getattr(seat1, attr) == getattr(total, attr)
        assert getattr(first, attr) + getattr(second, attr) == getattr(total, attr)
    assert seat0.games == seat1.games == n  # A sits in each seat once per deal
    assert first.games == second.games == n  # same coin flip, swapped seats: A moves first exactly once
    assert total.games == res.games == 2 * n

    records = [(a_seat, rec) for pair in res.deals for a_seat, rec in enumerate(pair)]
    assert res.wins == sum(rec.winner == a_seat for a_seat, rec in records)
    assert res.draws == sum(rec.winner == DRAW for _, rec in records)
    assert first.wins == sum(rec.winner == a_seat == rec.first_player for a_seat, rec in records)
    assert res.win_rate == pytest.approx(res.wins / res.games)
    assert res.score == pytest.approx((res.wins + 0.5 * res.draws) / res.games)
    assert res.win_rate + res.draw_rate + res.loss_rate == pytest.approx(1.0)
    assert sum(res.deal_summary().values()) == n

    d = res.to_dict(include_games=True)
    assert d["games"] == 2 * n and d["wins"] == res.wins and d["agent_a"] == spec_a
    assert d["by_seat"]["0"]["games"] == n and d["by_turn_order"]["first"]["games"] == n
    assert d["win_rate_ci95"] == list(wilson_interval(res.wins, res.games))
    assert len(d["games_played"]) == 2 * n
    json.dumps(d)  # serialisable
    text = res.format()
    for word in ("overall", "seat 0", "seat 1", "first", "second", "95% CI", spec_b):
        assert word in text


def test_outcome_for():
    rec = GameRecord(0, 1, 0, 5, 40)
    assert outcome_for(rec, 1) == 1 and outcome_for(rec, 0) == -1
    assert outcome_for(rec._replace(winner=DRAW), 0) == 0


# ---------------------------------------------------------------------- reproducibility
def test_reproducible_and_independent_of_worker_count():
    args = ("random", "greedy", 30)
    r1 = duplicate_match(*args, start_seed=7, workers=1)
    r1b = duplicate_match(*args, start_seed=7, workers=1)
    r2 = duplicate_match(*args, start_seed=7, workers=2)
    assert r1.deals == r1b.deals == r2.deals
    assert counts_without_timing(r1) == counts_without_timing(r2)
    r3 = duplicate_match(*args, start_seed=8, workers=1)
    assert r3.deals != r1.deals


def test_evaluator_pool_serves_several_matches():
    with Evaluator(workers=2, config=CONFIG) as ev:
        a = ev.match("random", "greedy", 9, start_seed=3, chunk_size=2)
        b = ev.match("greedy", "random", 9, start_seed=3)
    with Evaluator(workers=1, config=CONFIG) as ev:
        a1 = ev.match("random", "greedy", 9, start_seed=3)
        b1 = ev.match("greedy", "random", 9, start_seed=3)
    assert a.deals == a1.deals and b.deals == b1.deals
    # A vs B and B vs A over the same deals are the same games with the roles relabelled.
    assert a.wins == b.losses and a.draws == b.draws


def test_bad_arguments():
    with pytest.raises(ValueError):
        duplicate_match("greedy", "random", 0)
    with pytest.raises(FileNotFoundError):
        duplicate_match("greedy", "no/such/checkpoint.pt", 2)
    with pytest.raises(ValueError):
        duplicate_match("greedy", "not-an-agent", 2)


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
    small, big = wilson_interval(70, 100), wilson_interval(1400, 2000)
    assert big[1] - big[0] < small[1] - small[0]
    lo, hi = wilson_interval(1400, 2000)  # the Stage 1 target: 70% of 2000 games
    assert lo == pytest.approx(0.6795, abs=1e-3) and hi == pytest.approx(0.7197, abs=1e-3)


# ---------------------------------------------------------------------- PPO checkpoints
@pytest.fixture(scope="module")
def ckpt_dir(tmp_path_factory):
    """Two randomly initialised PPO checkpoints (SPEC section 7 format)."""
    torch = pytest.importorskip("torch")
    from cardgame.features import ObservationEncoder
    from cardgame.rl.network import PolicyValueNet

    d = tmp_path_factory.mktemp("ckpts")
    n_actions = Game(CONFIG).action_space.n
    for update in (1, 2):
        torch.manual_seed(update)
        net = PolicyValueNet(ObservationEncoder(CONFIG).dim, n_actions, (64, 64))
        torch.save({"model": net.state_dict(), "net": net.spec(), "update": update, "env_steps": 0,
                    "args": {"hidden": [64, 64]}}, d / f"ckpt_{update:05d}.pt")
    return d


def test_ppo_checkpoint_loads_and_plays_legal_games(ckpt_dir):
    path = str(ckpt_dir / "ckpt_00001.pt")
    for spec in (path, "ppo:" + path):
        for kwargs in ({}, {"deterministic": True}, {"seed": 5}):
            agent = make_agent(spec, CONFIG, **kwargs)
            for seed in range(4):  # play_game raises IllegalAgentActionError on any illegal action
                for seats in ((agent, GreedyAgent(CONFIG)), (RandomAgent(CONFIG), agent)):
                    rec = play_game(CONFIG, seats, seed)
                    assert rec.winner in (0, 1, DRAW)


def test_ppo_duplicate_match_is_reproducible_across_workers(ckpt_dir):
    path = str(ckpt_dir / "ckpt_00002.pt")
    r1 = duplicate_match(path, "greedy", 6, start_seed=21, workers=1)
    r2 = duplicate_match(path, "greedy", 6, start_seed=21, workers=2)
    assert r1.deals == r2.deals  # stochastic policy, but seeded per game and seat
    det = duplicate_match(path, "ppo:" + str(ckpt_dir / "ckpt_00001.pt"), 4, workers=1,
                          agent_kwargs_a={"deterministic": True})
    assert det.kwargs_a == {"deterministic": True} and det.games == 8


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
    assert "random" in p.stdout and "games/s" in p.stdout
    assert "PASS" not in p.stdout and "FAIL" not in p.stdout  # no greedy row
    data = json.loads(out.read_text())
    assert data["target"] is None and len(data["results"]) == 1
    row = data["results"][0]
    assert row["opponent"] == "random" and row["games"] == 20 and row["n_deals"] == 10

    # fewer games than the brief's 2,000: no PASS/FAIL verdict, and --strict refuses to pass
    p = run_cli("eval.py", "--agent", "random", "--games", "10", "--workers", "1", "--strict")
    assert p.returncode == 1 and "INDICATIVE" in p.stdout and "PASS" not in p.stdout
    p = run_cli("eval.py", "--agent", "random", "--games", "10", "--workers", "1")
    assert p.returncode == 0 and "INDICATIVE" in p.stdout  # informational without --strict
    p = run_cli("eval.py", "--agent", "random", "--games", "2000", "--workers", "2", "--strict")
    assert p.returncode == 1 and "FAIL" in p.stdout
    p = run_cli("eval.py", "--agent", "random", "--games", "11")
    assert p.returncode == 2 and "even" in p.stderr


def test_eval_cli_checkpoints(ckpt_dir, tmp_path):
    out = tmp_path / "res.json"
    agent = str(ckpt_dir / "ckpt_00002.pt")
    p = run_cli("eval.py", "--agent", agent, "--deterministic", "--opponents", "greedy", "random",
                "--checkpoints-dir", str(ckpt_dir), "--games", "8", "--workers", "2", "--target", "0.0",
                "--json", str(out))
    assert p.returncode == 0, p.stderr
    assert "INDICATIVE" in p.stdout and "ckpt_00001.pt" in p.stdout  # 8 games < the brief's 2,000
    data = json.loads(out.read_text())
    assert data["agent_kwargs"] == {"deterministic": True}
    labels = [r["label"] for r in data["results"]]
    assert labels == ["greedy", "random", "ckpt_00001.pt"]  # the agent's own file is skipped
    assert data["target"]["opponent"] == "greedy" and data["target"]["conclusive"] is False
    assert data["target"]["passed"] is False
    for r in data["results"]:
        assert r["games"] == 8 and r["by_seat"]["0"]["games"] == 4 and r["by_turn_order"]["second"]["games"] == 4


def test_bench_cli(ckpt_dir):
    p = run_cli("bench.py", "--games", "20", "--workers", "2", "--states", "50",
                "--checkpoint", str(ckpt_dir / "ckpt_00001.pt"))
    assert p.returncode == 0, p.stderr
    for text in ("games/s", "steps/s", "engine only", "greedy vs greedy", "2 workers", "ppo vs greedy",
                 "Game.observe", "ObservationEncoder.encode"):
        assert text in p.stdout


def test_eval_cli_skips_incompatible_checkpoints(ckpt_dir, tmp_path):
    """A checkpoint trained with a different observation encoding is reported, not crashed on."""
    import shutil

    import torch
    from cardgame.features import ObservationEncoder
    from cardgame.rl.network import PolicyValueNet
    mixed = tmp_path / "mixed"
    shutil.copytree(ckpt_dir, mixed)
    net = PolicyValueNet(ObservationEncoder(CONFIG).dim + 2, 71, (8,))
    torch.save({"model": net.state_dict(), "net": net.spec(), "update": 9, "env_steps": 0, "args": {}},
               mixed / "ckpt_00009.pt")
    p = run_cli("eval.py", "--agent", "greedy", "--opponents", "random", "--checkpoints-dir", str(mixed),
                "--games", "4", "--workers", "1")
    assert p.returncode == 0, p.stderr
    assert "skipping ckpt_00009.pt" in p.stdout and "ckpt_00001.pt" in p.stdout
    p = run_cli("eval.py", "--agent", str(mixed / "ckpt_00009.pt"), "--games", "4", "--workers", "1")
    assert p.returncode == 2 and "obs_dim" in p.stderr


def test_per_seat_stats_match_the_game_records():
    res = duplicate_match("greedy", "random", 20, start_seed=40)
    for seat in (0, 1):
        recs = [pair[seat] for pair in res.deals]
        assert res.by_seat[seat].wins == sum(r.winner == seat for r in recs)
        assert res.by_seat[seat].losses == sum(r.winner == 1 - seat for r in recs)


def test_deal_summary_counts_a_single_draw():
    a_wins_seat0 = GameRecord(0, 0, 0, 10, 50)
    draw = GameRecord(0, DRAW, 0, 50, 300)
    res = MatchResult("a", "b", 0, [(a_wins_seat0, draw)])
    assert res.deal_summary() == {"a_won_both": 0, "split": 0, "b_won_both": 0, "with_draw": 1}


def test_mirror_match_uses_two_agent_instances():
    from cardgame.evaluation import _DealRunner
    runner = _DealRunner(CONFIG)
    runner.run(("random", {}, "random", {}, range(1)))
    assert len({id(a) for a in runner.agents.values()}) == 2
