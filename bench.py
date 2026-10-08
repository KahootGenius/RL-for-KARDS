#!/usr/bin/env python
"""Engine and agent throughput (games per second).

    python bench.py                                   # 1000 games per row, min(8, cpu_count) workers
    python bench.py --games 300 --workers 4
    python bench.py --checkpoint runs/ppo/best.pt     # also PPO-vs-greedy throughput

Rows: raw engine loop (legal_actions + step with random actions, no observations), full agent
loops through evaluation.play_game (observe + act every step), and the multiprocess evaluator
(timed after a warm-up match, so pool start-up and model loading are excluded).
"""
from __future__ import annotations

import argparse
import os
import platform
import random
import sys
import time
from typing import Callable, List, Optional, Sequence, Tuple

from cardgame.agents import make_agent
from cardgame.cards import GameConfig, load_ruleset
from cardgame.engine import Game
from cardgame.evaluation import Evaluator, default_workers, play_game
from cardgame.features import ObservationEncoder

Row = Tuple[str, int, float, int, int]  # name, games, seconds, total steps, total rounds


def bench_engine(config: GameConfig, n_games: int, seed: int) -> Row:
    """Bare engine: reset, legal_actions, step with uniformly random legal actions."""
    game = Game(config)
    rng = random.Random(seed)
    steps = rounds = 0
    t0 = time.perf_counter()
    for i in range(n_games):
        game.reset(seed + i)
        while not game.done:
            legal = game.legal_actions()
            game.step(legal[rng.randrange(len(legal))])
            steps += 1
        rounds += game.round
    return "engine only, random actions", n_games, time.perf_counter() - t0, steps, rounds


def bench_agents(config: GameConfig, name: str, spec_a: str, spec_b: str, n_games: int, seed: int) -> Row:
    """play_game loop (observe + act each step), alternating seats."""
    a, b = make_agent(spec_a, config), make_agent(spec_b, config)
    game = Game(config)
    steps = rounds = 0
    t0 = time.perf_counter()
    for i in range(n_games):
        rec = play_game(config, (a, b) if i % 2 == 0 else (b, a), seed + i // 2, game=game)
        steps += rec.steps
        rounds += rec.rounds
    return name, n_games, time.perf_counter() - t0, steps, rounds


def bench_pool(config: GameConfig, name: str, spec_a: str, spec_b: str, n_games: int, seed: int,
               workers: int) -> Row:
    """Evaluator throughput on `workers` processes, measured after a warm-up match."""
    n_deals = max(1, n_games // 2)
    with Evaluator(workers, config) as ev:
        ev.warmup(spec_a, spec_b)
        r = ev.match(spec_a, spec_b, n_deals, seed)
    steps = sum(g.steps for pair in r.deals for g in pair)
    rounds = sum(g.rounds for pair in r.deals for g in pair)
    return name, r.games, r.elapsed, steps, rounds


def sample_states(config: GameConfig, n_states: int, seed: int) -> List[Game]:
    """Clones of positions from random-vs-random games (every step, all phases of the game)."""
    game = Game(config)
    rng = random.Random(seed)
    states: List[Game] = []
    i = 0
    while len(states) < n_states:
        game.reset(seed + i)
        i += 1
        while not game.done and len(states) < n_states:
            states.append(game.clone())
            legal = game.legal_actions()
            game.step(legal[rng.randrange(len(legal))])
    return states


def per_call_us(fn: Callable[[object], object], items: Sequence, min_time: float = 0.25) -> float:
    """Average microseconds per fn(item) call, repeating the pass until `min_time` seconds have elapsed."""
    calls, t0 = 0, time.perf_counter()
    while True:
        for x in items:
            fn(x)
        calls += len(items)
        elapsed = time.perf_counter() - t0
        if elapsed >= min_time:
            return 1e6 * elapsed / calls


def bench_calls(config: GameConfig, n_states: int, seed: int) -> List[Tuple[str, float]]:
    states = sample_states(config, n_states, seed)
    observations = [g.observe(g.current_player()) for g in states]
    enc = ObservationEncoder(config)

    def legal_uncached(g: Game) -> list:
        g._legal = g._mask = None  # drop the cache only (invalidate() also validates)
        return g.legal_actions()

    def mask_uncached(g: Game) -> object:
        g._legal = g._mask = None  # drop the cache only (invalidate() also validates)
        return g.legal_mask()

    return [
        ("Game.observe(player)", per_call_us(lambda g: g.observe(g.current_player()), states)),
        ("ObservationEncoder.encode(obs)", per_call_us(enc.encode, observations)),
        ("Game.legal_actions() (uncached)", per_call_us(legal_uncached, states)),
        ("Game.legal_mask() (uncached)", per_call_us(mask_uncached, states)),
        ("Game.clone()", per_call_us(lambda g: g.clone(), states)),
    ]


def format_rows(rows: List[Row]) -> str:
    width = max(len("benchmark"), *(len(r[0]) for r in rows))
    out = [f"{'benchmark':<{width}s} {'games':>7s} {'seconds':>8s} {'games/s':>9s} {'steps/s':>10s} "
           f"{'steps/game':>10s} {'rounds/game':>11s}"]
    for name, games, secs, steps, rounds in rows:
        out.append(f"{name:<{width}s} {games:7d} {secs:8.2f} {games / secs:9.1f} {steps / secs:10.0f} "
                   f"{steps / games:10.1f} {rounds / games:11.1f}")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=1000, help="games per benchmark row (default 1000)")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help=f"processes for the multiprocess rows (default {default_workers()})")
    ap.add_argument("--seed", type=int, default=0, help="first deal seed (default 0)")
    ap.add_argument("--checkpoint", default=None, metavar="PATH", help="PPO checkpoint for PPO-vs-greedy rows")
    ap.add_argument("--states", type=int, default=2000, help="sampled positions for per-call costs (default 2000)")
    args = ap.parse_args(argv)
    if args.games < 2 or args.workers < 1 or args.states < 1:
        ap.error("--games must be >= 2, --workers and --states >= 1")
    if args.checkpoint is not None and not os.path.isfile(args.checkpoint):
        ap.error(f"checkpoint not found: {args.checkpoint}")

    config = load_ruleset()
    n, s, w = args.games, args.seed, args.workers
    print(f"python {platform.python_version()} on {platform.machine()} ({os.cpu_count()} CPUs); "
          f"{n} games per row, seeds from {s}")
    print()
    rows: List[Row] = [
        bench_engine(config, n, s),
        bench_agents(config, "random vs random (play_game)", "random", "random", n, s),
        bench_agents(config, "greedy vs greedy (play_game)", "greedy", "greedy", n, s),
    ]
    if w > 1:
        rows.append(bench_pool(config, f"random vs random, {w} workers", "random", "random", n, s, w))
    if args.checkpoint:
        ckpt = args.checkpoint
        rows.append(bench_agents(config, "ppo vs greedy (play_game)", ckpt, "greedy", n, s))
        if w > 1:
            rows.append(bench_pool(config, f"ppo vs greedy, {w} workers", ckpt, "greedy", n, s, w))
    print(format_rows(rows))
    print()

    calls = bench_calls(config, args.states, s)
    title = f"per-call cost ({args.states} sampled positions)"
    width = max(len(title), *(len(name) for name, _ in calls))
    print(f"{title:<{width}s} {'us/call':>9s}")
    for name, us in calls:
        print(f"{name:<{width}s} {us:9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
