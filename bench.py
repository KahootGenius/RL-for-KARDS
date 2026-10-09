#!/usr/bin/env python
"""Engine and agent throughput (games per second) and per-call costs (SPEC §11).

    python bench.py                                   # 1000 games per row, min(8, cpu_count) workers
    python bench.py --games 300 --workers 4
    python bench.py --checkpoint runs/ppo/best.pt     # also PPO-vs-greedy throughput
    python bench.py --compare --out                   # side by side with Stage 1 (results/stage1/bench.txt);
                                                      # writes results/bench_stage2.txt

Rows: raw engine loop (legal_actions + step with random actions, no observations), full agent
loops through evaluation.play_game (observe + act every step), and the multiprocess evaluator
(timed after a warm-up match, so pool start-up and model loading are excluded). Decks are sampled
per deal seed. Per-call costs are measured on positions sampled from random games; legal_actions /
legal_mask drop the engine's cache before every call. The report starts with the machine line (Python,
CPU, CPU count, 1-minute load average); a load average above half the CPU count is flagged, since other
processes then distort the timings. --compare prints both machine lines and leaves the ratio of rows
measured with different worker counts blank ("-").
"""
from __future__ import annotations

import argparse
import os
import platform
import random
import re
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from cardgame.agents import make_agent
from cardgame.cards import GameConfig, load_ruleset
from cardgame.engine import Game
from cardgame.evaluation import Evaluator, default_workers, play_game
from cardgame.features import ObservationEncoder

Row = Tuple[str, int, float, int, int]  # name, games, seconds, total steps, total rounds
DEFAULT_OUT = os.path.join("results", "bench_stage2.txt")
DEFAULT_COMPARE = os.path.join("results", "stage1", "bench.txt")  # the Stage 1 report
ROW_RE = re.compile(r"^(.*?)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$")
CALL_RE = re.compile(r"^(.*?\S)\s+([\d.]+)\s*$")
MACHINE_RE = re.compile(r"^python (\S+) on (\S+) \((\d+) CPUs(?:; load average ([\d.]+))?\)")
WORKERS_RE = re.compile(r"(\d+) workers")
BUSY_LOAD = 0.5  # flag a 1-minute load average above this fraction of the CPU count


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
    """Evaluator throughput on `workers` processes (same games as bench_agents), after a warm-up match."""
    n_deals = max(1, n_games // 2)
    with Evaluator(workers, config) as ev:
        ev.warmup(spec_a, spec_b)
        r = ev.match(spec_a, spec_b, n_deals, seed, decks="sampled")
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
    pairs = [(g.observe(g.current_player()), g.legal_mask()) for g in states]
    enc = ObservationEncoder(config)

    def legal_uncached(g: Game) -> list:
        g._legal = g._mask = None  # drop the cache only (invalidate() also validates)
        return g.legal_actions()

    def mask_uncached(g: Game) -> object:
        g._legal = g._mask = None  # drop the cache only (invalidate() also validates)
        return g.legal_mask()

    return [
        ("Game.observe(player)", per_call_us(lambda g: g.observe(g.current_player()), states)),
        ("ObservationEncoder.encode(obs, mask)", per_call_us(lambda om: enc.encode(om[0], om[1]), pairs)),
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


def format_calls(calls: List[Tuple[str, float]], n_states: int) -> str:
    title = f"per-call cost ({n_states} sampled positions)"
    width = max(len(title), *(len(name) for name, _ in calls))
    lines = [f"{title:<{width}s} {'us/call':>9s}"]
    lines += [f"{name:<{width}s} {us:9.2f}" for name, us in calls]
    return "\n".join(lines)


# ---------------------------------------------------------------------- machine line and load
def machine_line(n_games: int, seed: int, n_decks: int) -> str:
    """First report line: Python version, CPU, CPU count and 1-minute load average (if known)."""
    load = f"; load average {os.getloadavg()[0]:.1f}" if hasattr(os, "getloadavg") else ""
    return (f"python {platform.python_version()} on {platform.machine()} ({os.cpu_count()} CPUs{load}); "
            f"{n_games} games per row, seeds from {seed}; decks sampled per seed from {n_decks}")


def load_warning(load1: Optional[float], cpus: Optional[int], what: str = "this machine") -> Optional[str]:
    """A warning when the 1-minute load average exceeds BUSY_LOAD x the CPU count, else None."""
    if load1 is None or not cpus or load1 <= BUSY_LOAD * cpus:
        return None
    return (f"warning: 1-minute load average {load1:.1f} > {BUSY_LOAD:g} x {cpus} CPUs on {what}: other processes "
            f"compete for the CPU, so the timings understate the speed")


def parse_machine(header: str) -> Optional[dict]:
    """{python, machine, cpus, load} from a report's first line (load None when not recorded)."""
    m = MACHINE_RE.match(header.strip())
    if m is None:
        return None
    return {"python": m.group(1), "machine": m.group(2), "cpus": int(m.group(3)),
            "load": float(m.group(4)) if m.group(4) else None}


# ---------------------------------------------------------------------- comparison with an older output
def canonical(name: str) -> str:
    """Row key shared by Stage 1 and Stage 2 outputs (worker counts and the encode signature differ)."""
    name = re.sub(r"\d+ workers", "K workers", name.strip())
    return name.replace("encode(obs, mask)", "encode(obs)")


def parse_bench(text: str) -> Tuple[Dict[str, dict], Dict[str, float]]:
    """(benchmark rows by canonical name, per-call us by canonical name) from a bench.py output."""
    rows, calls, in_calls = {}, {}, False
    for line in text.splitlines():
        if line.startswith("per-call cost"):
            in_calls = True
            continue
        if not in_calls:
            m = ROW_RE.match(line)
            if m and not line.startswith("benchmark"):
                name = m.group(1).strip()
                rows[canonical(name)] = {"name": name, "games": int(m.group(2)), "games_per_s": float(m.group(4)),
                                         "steps_per_s": float(m.group(5)), "steps_per_game": float(m.group(6)),
                                         "rounds_per_game": float(m.group(7))}
        else:
            m = CALL_RE.match(line)
            if m:
                calls[canonical(m.group(1))] = float(m.group(2))
    return rows, calls


def format_compare(old_text: str, rows: List[Row], calls: List[Tuple[str, float]], old_label: str,
                   new_header: str = "") -> str:
    """Old and new reports side by side: both machine lines, then games/s, steps/s and per-call costs with
    new/old ratios (left blank, "-", for rows measured with different worker counts)."""
    old_rows, old_calls = parse_bench(old_text)
    old_header = old_text.splitlines()[0] if old_text.strip() else ""
    width = max([len("benchmark")] + [len(r[0]) for r in rows] + [len(c[0]) for c in calls])
    lines = [f"comparison with {old_label} (ratio = new / old)",
             f"  old: {old_header or '(no machine line)'}", f"  new: {new_header or '(no machine line)'}"]
    old_m, new_m = parse_machine(old_header), parse_machine(new_header)
    if old_m is not None and new_m is not None:
        differ = [f"{k} {old_m[k]} vs {new_m[k]}" for k in ("machine", "cpus", "python") if old_m[k] != new_m[k]]
        if differ:
            lines.append(f"warning: the reports come from different setups ({', '.join(differ)}): the ratios "
                         f"compare machines as well as code")
    if old_m is not None:
        warn = load_warning(old_m["load"], old_m["cpus"], "the old run")
        if warn:
            lines.append(warn)
    lines.append(f"{'benchmark':<{width}s} {'old games/s':>12s} {'new games/s':>12s} {'ratio':>7s} "
                 f"{'old steps/s':>12s} {'new steps/s':>12s} {'ratio':>7s}")
    notes = []
    for name, games, secs, steps, _ in rows:
        old = old_rows.get(canonical(name))
        gps, sps = games / secs, steps / secs
        if old is None:
            lines.append(f"{name:<{width}s} {'-':>12s} {gps:12.1f} {'-':>7s} {'-':>12s} {sps:12.0f} {'-':>7s}")
            continue
        old_w, new_w = WORKERS_RE.search(old["name"]), WORKERS_RE.search(name)
        same_workers = (old_w and old_w.group(1)) == (new_w and new_w.group(1))
        r_games = f"{gps / old['games_per_s']:6.2f}x" if same_workers else f"{'-':>7s}"
        r_steps = f"{sps / old['steps_per_s']:6.2f}x" if same_workers else f"{'-':>7s}"
        lines.append(f"{name:<{width}s} {old['games_per_s']:12.1f} {gps:12.1f} {r_games} "
                     f"{old['steps_per_s']:12.0f} {sps:12.0f} {r_steps}")
        if not same_workers:
            notes.append(f"  {name!r} vs old row {old['name']!r}: different worker counts, no ratio")
        elif old["name"] != name:
            notes.append(f"  {name!r} compared with old row {old['name']!r}")
    lines.append(f"{'per-call cost (us)':<{width}s} {'old':>12s} {'new':>12s} {'ratio':>7s}")
    for name, us in calls:
        old = old_calls.get(canonical(name))
        if old is None:
            lines.append(f"{name:<{width}s} {'-':>12s} {us:12.2f} {'-':>7s}")
        else:
            lines.append(f"{name:<{width}s} {old:12.2f} {us:12.2f} {us / old:6.2f}x")
    if notes:
        lines.append("note: rows matched across different settings:")
        lines += notes
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=1000, help="games per benchmark row (default 1000)")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help=f"processes for the multiprocess rows (default {default_workers()}; 1 = skip them)")
    ap.add_argument("--seed", type=int, default=0, help="first deal seed (default 0)")
    ap.add_argument("--checkpoint", default=None, metavar="PATH", help="PPO checkpoint for PPO-vs-greedy rows")
    ap.add_argument("--states", type=int, default=2000, help="sampled positions for per-call costs (default 2000)")
    ap.add_argument("--compare", nargs="?", const=DEFAULT_COMPARE, default=None, metavar="PATH",
                    help=f"an earlier bench.py output to compare against (default when given without a value: "
                         f"Stage 1's {DEFAULT_COMPARE})")
    ap.add_argument("--out", nargs="?", const=DEFAULT_OUT, default=None, metavar="PATH",
                    help=f"also write the report to PATH (default when given without a value: {DEFAULT_OUT})")
    args = ap.parse_args(argv)
    if args.games < 2 or args.workers < 1 or args.states < 1:
        ap.error("--games must be >= 2, --workers and --states >= 1")
    if args.checkpoint is not None and not os.path.isfile(args.checkpoint):
        ap.error(f"checkpoint not found: {args.checkpoint}")
    old_text = None
    if args.compare is not None:
        if not os.path.isfile(args.compare):
            ap.error(f"--compare file not found: {args.compare}")
        with open(args.compare, encoding="utf-8") as f:
            old_text = f.read()

    report: List[str] = []

    def emit(text: str = "") -> None:
        print(text, flush=True)
        report.append(text)

    config = load_ruleset()
    if args.checkpoint is not None:
        try:  # fail here with a clear message (e.g. another card pool) rather than mid-benchmark
            make_agent(args.checkpoint, config)
        except Exception as exc:  # noqa: BLE001 - reported to the user verbatim
            ap.error(f"checkpoint {args.checkpoint}: {type(exc).__name__}: {exc}")
    n, s, w = args.games, args.seed, args.workers
    header = machine_line(n, s, config.n_decks)
    emit(header)
    machine = parse_machine(header)
    warn = load_warning(machine["load"], machine["cpus"]) if machine is not None else None
    if warn:
        emit(warn)
    emit()
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
    emit(format_rows(rows))
    emit()
    calls = bench_calls(config, args.states, s)
    emit(format_calls(calls, args.states))
    if old_text is not None:
        emit()
        emit(format_compare(old_text, rows, calls, args.compare, header))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(report) + "\n")
        print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
