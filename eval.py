#!/usr/bin/env python
"""Duplicate-game evaluation: every deal seed is played twice with the seats swapped (SPEC section 8).

    python eval.py --agent runs/ppo/best.pt                                  # vs greedy, 2000 games
    python eval.py --agent runs/ppo/best.pt --deterministic --opponents greedy random \\
                   --checkpoints-dir runs/ppo --checkpoint-stride 5 --json runs/ppo/eval.json
    python eval.py --agent greedy --opponents random --games 200

Agent specs: "random", "greedy", or a PPO checkpoint ("<path>.pt" or "ppo:<path>").
All rates are from --agent's point of view; draws count as non-wins.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from cardgame.agents import make_agent
from cardgame.cards import load_ruleset
from cardgame.evaluation import Evaluator, MatchResult, checkpoint_path, default_workers

TARGET_OPPONENT = "greedy"
TARGET_GAMES = 2000  # the brief's acceptance test: >= 70% of 2,000 duplicate games vs greedy


def find_checkpoints(directory: str, stride: int = 1) -> List[str]:
    """ckpt_<update>.pt files sorted by update; every `stride`-th counted back from the newest (always kept)."""
    found = []
    for p in Path(directory).glob("ckpt_*.pt"):
        m = re.fullmatch(r"ckpt_(\d+)\.pt", p.name)
        if m:
            found.append((int(m.group(1)), str(p)))
    found.sort()
    paths = [p for _, p in found]
    return paths[::-1][::max(1, stride)][::-1]


def label(spec: str) -> str:
    """Short display name: checkpoints by file name, everything else as given."""
    path = checkpoint_path(spec)
    return os.path.basename(path) if path is not None else spec


def checkpoint_error(spec: str, config) -> Optional[str]:
    """None if `spec` is not a checkpoint or loads against this ruleset, else the reason it does not."""
    if checkpoint_path(spec) is None:
        return None
    try:
        make_agent(spec, config)
    except Exception as exc:  # noqa: BLE001 - reported to the user verbatim
        return f"{type(exc).__name__}: {exc}"
    return None


def same_file(a: str, b: str) -> bool:
    pa, pb = checkpoint_path(a), checkpoint_path(b)
    return pa is not None and pb is not None and os.path.abspath(pa) == os.path.abspath(pb)


COLUMNS = (("games", 6), ("win%", 6), ("draw%", 6), ("loss%", 6), ("win 95% CI", 14), ("seat0%", 7),
           ("seat1%", 7), ("first%", 7), ("second%", 8), ("games/s", 8))


def table_header(name_width: int) -> str:
    return f"{'opponent':<{name_width}s}" + "".join(f" {title:>{w}s}" for title, w in COLUMNS)


def table_row(name: str, r: MatchResult, name_width: int) -> str:
    lo, hi = r.win_ci()
    cells = (f"{r.games}", f"{100 * r.win_rate:.1f}", f"{100 * r.draw_rate:.1f}", f"{100 * r.loss_rate:.1f}",
             f"[{100 * lo:.1f}, {100 * hi:.1f}]", f"{100 * r.by_seat[0].win_rate:.1f}",
             f"{100 * r.by_seat[1].win_rate:.1f}", f"{100 * r.first.win_rate:.1f}",
             f"{100 * r.second.win_rate:.1f}", f"{r.games_per_s:.0f}")
    return f"{name:<{name_width}s}" + "".join(f" {c:>{w}s}" for c, (_, w) in zip(cells, COLUMNS))


def target_check(results: List[Tuple[str, MatchResult]], target: float) -> Optional[dict]:
    for spec, r in results:
        if spec == TARGET_OPPONENT:
            lo, _ = r.win_ci()
            return {"opponent": spec, "target": target, "win_rate": r.win_rate, "games": r.games,
                    "win_rate_ci95_low": lo, "conclusive": r.games >= TARGET_GAMES,
                    "passed": r.win_rate >= target and r.games >= TARGET_GAMES}
    return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True, help="agent to evaluate (random, greedy, or a .pt checkpoint)")
    ap.add_argument("--opponents", nargs="+", default=[TARGET_OPPONENT], metavar="SPEC",
                    help="opponent specs (default: greedy)")
    ap.add_argument("--checkpoints-dir", default=None, metavar="DIR",
                    help="also play against the ckpt_*.pt files in DIR (oldest first)")
    ap.add_argument("--checkpoint-stride", type=int, default=1, metavar="K",
                    help="use every K-th checkpoint, counted back from the newest (default 1 = all)")
    ap.add_argument("--games", type=int, default=2000,
                    help="games per opponent, even: N/2 deals x 2 seatings (default 2000)")
    ap.add_argument("--seed", type=int, default=0, help="first deal seed (default 0)")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help=f"worker processes (default min(8, cpu_count) = {default_workers()})")
    ap.add_argument("--deterministic", action="store_true",
                    help="argmax policy for a PPO --agent (default: sample from the policy)")
    ap.add_argument("--target", type=float, default=0.70,
                    help="win-rate target against greedy, reported as PASS/FAIL (default 0.70)")
    ap.add_argument("--json", default=None, metavar="PATH", help="write all results to this JSON file")
    ap.add_argument("--strict", action="store_true", help=f"exit with code 1 unless the greedy target PASSes (needs >= {TARGET_GAMES} games)")
    args = ap.parse_args(argv)

    if args.games < 2 or args.games % 2:
        ap.error("--games must be a positive even number (each deal is played twice)")
    if args.checkpoint_stride < 1:
        ap.error("--checkpoint-stride must be >= 1")
    if args.workers < 1:
        ap.error("--workers must be >= 1")
    if args.checkpoints_dir is not None and not os.path.isdir(args.checkpoints_dir):
        ap.error(f"--checkpoints-dir {args.checkpoints_dir!r} is not a directory")

    opponents: List[str] = []
    for spec in args.opponents:
        if spec not in opponents:
            opponents.append(spec)
    if args.checkpoints_dir is not None:
        ckpts = find_checkpoints(args.checkpoints_dir, args.checkpoint_stride)
        if not ckpts:
            print(f"note: no ckpt_*.pt files in {args.checkpoints_dir}")
        for path in ckpts:
            if not same_file(path, args.agent) and not any(same_file(path, o) for o in opponents):
                opponents.append(path)

    config = load_ruleset()
    for spec in (args.agent, *opponents):
        path = checkpoint_path(spec)
        if path is not None and not os.path.isfile(path):
            ap.error(f"checkpoint not found: {path}")
    # Load every checkpoint once up front: an incompatible one (e.g. trained with another
    # observation encoding) fails here with a clear message instead of mid-run in a worker.
    compatible = []
    for spec in opponents:
        error = checkpoint_error(spec, config)
        if error is None:
            compatible.append(spec)
        elif spec in args.opponents:
            ap.error(f"opponent {spec}: {error}")
        else:
            print(f"note: skipping {label(spec)}: {error}")
    opponents = compatible
    error = checkpoint_error(args.agent, config)
    if error is not None:
        ap.error(f"agent {args.agent}: {error}")
    if not opponents:
        ap.error("no opponents left to play")

    is_ppo = checkpoint_path(args.agent) is not None
    kwargs_a = {"deterministic": True} if (args.deterministic and is_ppo) else {}
    n_deals = args.games // 2
    workers = min(args.workers, n_deals)
    mode = (" (deterministic)" if kwargs_a else " (sampling)") if is_ppo else ""
    if args.deterministic and not is_ppo:
        print("note: --deterministic only applies to PPO checkpoints; ignored")
    print(f"agent: {args.agent}{mode}")
    print(f"{args.games} games per opponent = {n_deals} deals (seeds {args.seed}..{args.seed + n_deals - 1}) "
          f"x 2 seatings; {workers} worker{'s' if workers > 1 else ''}; games/s is wall clock after warm-up")
    print(f"seat0 plays {config.deck_names[0]}, seat1 plays {config.deck_names[1]}; "
          f"first/second = turn order from the coin flip")
    print()

    names = [label(o) for o in opponents]
    width = max(8, *(len(n) for n in names))
    print(table_header(width))
    results: List[Tuple[str, MatchResult]] = []
    with Evaluator(workers, config) as ev:
        ev.warmup(args.agent, opponents[0], agent_kwargs_a=kwargs_a)
        for spec, name in zip(opponents, names):
            r = ev.match(args.agent, spec, n_deals, args.seed, agent_kwargs_a=kwargs_a)
            results.append((spec, r))
            print(table_row(name, r, width), flush=True)

    check = target_check(results, args.target)
    if check is not None and not check["conclusive"]:
        print(f"\nINDICATIVE: win rate vs {TARGET_OPPONENT} {check['win_rate']:.1%} over {check['games']} games "
              f"(the acceptance test needs >= {TARGET_GAMES} duplicate games)")
    elif check is not None:
        verdict = "PASS" if check["passed"] else "FAIL"
        cmp = ">=" if check["passed"] else "<"
        print(f"\n{verdict}: win rate vs {TARGET_OPPONENT} {check['win_rate']:.1%} {cmp} target {args.target:.0%} "
              f"over {check['games']} duplicate games (95% CI lower bound {check['win_rate_ci95_low']:.1%})")

    if args.json:
        payload = {"agent": args.agent, "agent_kwargs": kwargs_a, "games_per_opponent": args.games,
                   "start_seed": args.seed, "workers": workers, "target": check,
                   "results": [{"opponent": spec, "label": label(spec), **r.to_dict()} for spec, r in results]}
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"results written to {args.json}")

    if args.strict and check is not None and not check["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
