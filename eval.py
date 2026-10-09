#!/usr/bin/env python
"""Duplicate-game evaluation over every deck matchup, plus the tactical scenarios (SPEC §10).

    python eval.py --agent runs/ppo/best.pt                                  # vs greedy, 2016 games
    python eval.py --agent runs/ppo/best.pt --scenarios --json runs/ppo/eval.json
    python eval.py --agent runs/ppo/best.pt --opponents greedy random \\
                   --checkpoints-dir runs/ppo --checkpoint-stride 5
    python eval.py --agent greedy --opponents random --games 256 --decks sampled

Every deal is played twice with the seats swapped. With --decks all (default) deal k uses deck pair
divmod(k % n², n), so every ordered deck pair occurs equally often and --games is rounded up to a
multiple of 2·n². Agent specs: "random", "greedy", or a PPO checkpoint ("<path>.pt" or
"ppo:<path>"). All rates are from --agent's point of view; draws count as non-wins.

Done when (SPEC §10), checked against greedy: overall win rate >= 70% over >= 2000 games (decks all,
seeds from 0, sampling policy), every unordered matchup cell >= 60%, and > 50% of the scenarios solved.
PASS/FAIL is printed only when all three were measured under these settings (--scenarios included,
--target / --cell-target / --scenario-target not lowered); otherwise the verdict is INDICATIVE with
the reasons. The worst ordered (deck-confounded) cell P[a][b] is shown next to it as a diagnostic.
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
from cardgame.evaluation import (DECK_MODES, Evaluator, MatchResult, checkpoint_path, deals_for_games,
                                 default_workers, short_deck_names)

TARGET_OPPONENT = "greedy"
TARGET_GAMES = 2000  # the brief's acceptance test: >= 70% of 2,000 duplicate games vs greedy
# The brief's thresholds (SPEC §10): lower values give an INDICATIVE verdict, never a PASS.
BRIEF_THRESHOLDS = (("target", "--target", 0.70), ("cell_target", "--cell-target", 0.60),
                    ("scenario_target", "--scenario-target", 0.50))


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
           ("seat1%", 7), ("first%", 7), ("second%", 8), ("worst cell", 22), ("games/s", 8))


def table_header(name_width: int) -> str:
    return f"{'opponent':<{name_width}s}" + "".join(f" {title:>{w}s}" for title, w in COLUMNS)


def table_row(name: str, r: MatchResult, name_width: int) -> str:
    lo, hi = r.win_ci()
    worst = r.min_cell()
    worst_text = "-" if worst is None else f"{r.cell_name(worst[0])} {100 * worst[1].win_rate:.1f}"
    cells = (f"{r.games}", f"{100 * r.win_rate:.1f}", f"{100 * r.draw_rate:.1f}", f"{100 * r.loss_rate:.1f}",
             f"[{100 * lo:.1f}, {100 * hi:.1f}]", f"{100 * r.by_seat[0].win_rate:.1f}",
             f"{100 * r.by_seat[1].win_rate:.1f}", f"{100 * r.first.win_rate:.1f}",
             f"{100 * r.second.win_rate:.1f}", worst_text, f"{r.games_per_s:.0f}")
    return f"{name:<{name_width}s}" + "".join(f" {c:>{w}s}" for c, (_, w) in zip(cells, COLUMNS))


def cells_table(results: List[Tuple[str, MatchResult]], width: int) -> str:
    """One row per opponent, one column per unordered matchup cell C{i,j} (agent's win%), plus the games
    per cell (the same deals for every opponent)."""
    first = results[0][1]
    short = short_deck_names(first.deck_names)
    keys = list(first.cells)
    titles = [f"{short[i]}-{short[j]}" for i, j in keys]
    col = max(5, *(len(t) for t in titles))
    lines = [f"{'opponent':<{width}s}" + "".join(f" {t:>{col}s}" for t in titles)]
    for name, r in results:
        cells = [f"{100 * r.cells[k].win_rate:.1f}" if r.cells[k].games else "-" for k in keys]
        lines.append(f"{name:<{width}s}" + "".join(f" {c:>{col}s}" for c in cells))
    lines.append(f"{'(games)':<{width}s}" + "".join(f" {first.cells[k].games:>{col}d}" for k in keys))
    return "\n".join(lines)


def indent(text: str, by: str = "  ") -> str:
    return "\n".join(by + line for line in text.splitlines())


def target_check(results: List[Tuple[str, MatchResult]], args, is_ppo: bool,
                 scenario_pct: Optional[float]) -> Optional[dict]:
    """The SPEC §10 "Done when" verdict against greedy (None without a greedy opponent)."""
    for spec, r in results:
        if spec != TARGET_OPPONENT:
            continue
        lo, _ = r.win_ci()
        worst, ordered = r.min_cell(), r.min_ordered_cell()
        reasons = []
        if r.games < TARGET_GAMES:
            reasons.append(f"{r.games} games < {TARGET_GAMES}")
        if args.decks != "all":
            reasons.append(f"decks={args.decks} (the test uses decks=all)")
        if args.seed != 0:
            reasons.append(f"seeds start at {args.seed} (the test uses seeds from 0)")
        if is_ppo and args.deterministic:
            reasons.append("--deterministic (the test samples from the policy)")
        for attr, flag, brief in BRIEF_THRESHOLDS:
            if getattr(args, attr) < brief:
                reasons.append(f"{flag} {getattr(args, attr):g} below the brief's {brief:g}")
        if scenario_pct is None:
            reasons.append("scenarios not run (--scenarios)")
        checks = {"overall": r.win_rate >= args.target,
                  "cells": worst is not None and worst[1].win_rate >= args.cell_target}
        if scenario_pct is not None:
            checks["scenarios"] = scenario_pct > 100 * args.scenario_target
        names = r.deck_names
        return {"opponent": spec, "target": args.target, "cell_target": args.cell_target,
                "scenario_target": args.scenario_target,
                "win_rate": r.win_rate, "games": r.games, "win_rate_ci95_low": lo,
                "worst_cell": None if worst is None else {"name": r.cell_name(worst[0]), "decks": list(worst[0]),
                                                         "win_rate": worst[1].win_rate, "games": worst[1].games,
                                                         "win_rate_ci95": list(worst[1].win_ci())},
                "worst_ordered_cell": None if ordered is None else {
                    "diagnostic": True, "name": r.ordered_cell_name(ordered[0]), "decks": list(ordered[0]),
                    "agent_deck": names[ordered[0][0]], "opponent_deck": names[ordered[0][1]],
                    "win_rate": ordered[1].win_rate, "games": ordered[1].games,
                    "win_rate_ci95": list(ordered[1].win_ci())},
                "scenarios_pct_solved": scenario_pct, "checks": checks, "conclusive": not reasons,
                "indicative_reasons": reasons, "passed": all(checks.values()) and not reasons}
    return None


def verdict_text(check: dict) -> str:
    parts = [f"win rate vs {check['opponent']} {check['win_rate']:.1%} "
             f"({'>=' if check['checks']['overall'] else '<'} {check['target']:.0%}) over {check['games']} "
             f"duplicate games (95% CI lower bound {check['win_rate_ci95_low']:.1%})"]
    wc = check["worst_cell"]
    if wc is not None:
        parts.append(f"worst matchup cell {wc['name']} {wc['win_rate']:.1%} "
                     f"({'>=' if check['checks']['cells'] else '<'} {check['cell_target']:.0%})")
    if check["scenarios_pct_solved"] is not None:
        parts.append(f"scenarios {check['scenarios_pct_solved']:.0f}% solved "
                     f"({'>' if check['checks']['scenarios'] else '<='} {100 * check['scenario_target']:.0f}%)")
    if not check["conclusive"]:
        text = "INDICATIVE (" + "; ".join(check["indicative_reasons"]) + "): " + "; ".join(parts)
    else:
        text = ("PASS: " if check["passed"] else "FAIL: ") + "; ".join(parts)
    wo = check["worst_ordered_cell"]
    if wo is not None:
        text += (f"\n  diagnostic only (the verdict uses the unordered cells, SPEC §10): worst ordered cell "
                 f"{wo['name']} = {wo['win_rate']:.1%} over {wo['games']} games (agent holds {wo['agent_deck']}, "
                 f"{check['opponent']} holds {wo['opponent_deck']}; deck-confounded)")
    return text


def run_scenario_suite(agent: str, config, is_ppo: bool, playouts: int) -> dict:
    """Scenario reports for the agent and the greedy / random baselines."""
    from cardgame.scenarios import run_scenarios
    reports = {"agent": run_scenarios(agent, config, deterministic=True, n_stochastic=playouts if is_ppo else 0)}
    for base in ("greedy", "random"):
        reports[base] = reports["agent"] if agent == base else run_scenarios(base, config)
    return reports


def format_scenarios(reports: dict, is_ppo: bool) -> str:
    agent, greedy, random_ = reports["agent"], reports["greedy"], reports["random"]
    width = max(len("scenario"), *(len(r.name) for r in agent.results))

    def det(r) -> str:
        return "-" if r.solved is None else ("yes" if r.solved else "no")

    def rate(r) -> str:
        return "-" if r.success is None else f"{100 * r.success:.0f}%"

    def total(rep) -> str:
        if rep.deterministic:
            return f"{rep.n_solved}/{rep.n}"
        return f"{rep.pct_solved:.0f}%"

    lines = [f"{'scenario':<{width}s} {'agent':>6s} {'sampled':>8s} {'greedy':>7s} {'random':>7s}  tags"]
    for a, g, r in zip(agent.results, greedy.results, random_.results):
        a_det = det(a) if a.solved is not None else rate(a)
        lines.append(f"{a.name:<{width}s} {a_det:>6s} {rate(a) if is_ppo else '-':>8s} {det(g):>7s} "
                     f"{rate(r):>7s}  {','.join(a.tags)}")
    sampled = f"{100 * agent.mean_success:.0f}%" if (is_ppo and agent.mean_success is not None) else "-"
    lines.append(f"{'solved':<{width}s} {total(agent):>6s} {sampled:>8s} {total(greedy):>7s} {total(random_):>7s}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True, help="agent to evaluate (random, greedy, or a .pt checkpoint)")
    ap.add_argument("--opponents", nargs="+", default=[TARGET_OPPONENT], metavar="SPEC",
                    help="opponent specs (default: greedy)")
    ap.add_argument("--checkpoints-dir", default=None, metavar="DIR",
                    help="also play against the ckpt_*.pt files in DIR (oldest first; incompatible ones are skipped)")
    ap.add_argument("--checkpoint-stride", type=int, default=1, metavar="K",
                    help="use every K-th checkpoint, counted back from the newest (default 1 = all)")
    ap.add_argument("--games", type=int, default=TARGET_GAMES,
                    help="games per opponent (2 per deal; decks=all rounds up to a multiple of 2*n_decks^2; "
                         "default 2000 -> 2016)")
    ap.add_argument("--decks", choices=DECK_MODES, default="all",
                    help="all: every ordered deck pair equally often (default); sampled: sample_decks(seed)")
    ap.add_argument("--seed", type=int, default=0, help="first deal seed (default 0)")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help=f"worker processes (default min(8, cpu_count) = {default_workers()})")
    ap.add_argument("--deterministic", action="store_true",
                    help="argmax policy for a PPO --agent in the matches (default: sample from the policy)")
    ap.add_argument("--target", type=float, default=0.70, help="overall win-rate target vs greedy (default 0.70)")
    ap.add_argument("--cell-target", type=float, default=0.60,
                    help="minimum win rate in every matchup cell vs greedy (default 0.60)")
    ap.add_argument("--scenarios", action="store_true",
                    help="also run the scenario suite for the agent and the greedy/random baselines")
    ap.add_argument("--scenario-target", type=float, default=0.50,
                    help="fraction of scenarios the agent must solve, exclusive (default 0.50)")
    ap.add_argument("--playouts", type=int, default=100,
                    help="sampled playouts per scenario for a PPO agent (default 100)")
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the greedy-vs-greedy reference matrix printed next to P[a][b]")
    ap.add_argument("--json", default=None, metavar="PATH", help="write all results to this JSON file")
    ap.add_argument("--strict", action="store_true",
                    help=f"exit with code 1 unless the greedy verdict is PASS (needs >= {TARGET_GAMES} games, "
                         "--scenarios and the default thresholds or stricter)")
    args = ap.parse_args(argv)

    if args.games < 1:
        ap.error("--games must be >= 1")
    if args.decks == "sampled" and args.games % 2:
        ap.error("--games must be even with --decks sampled (each deal is played twice)")
    if args.seed < 0:
        ap.error("--seed must be >= 0")
    if args.checkpoint_stride < 1:
        ap.error("--checkpoint-stride must be >= 1")
    if args.workers < 1:
        ap.error("--workers must be >= 1")
    if args.playouts < 0:
        ap.error("--playouts must be >= 0")
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
    # Load every checkpoint once up front: an incompatible one (a Stage 1 checkpoint, another card
    # pool or encoding) fails here with a clear message instead of mid-run in a worker.
    error = checkpoint_error(args.agent, config)
    if error is not None:
        ap.error(f"agent {args.agent}: {error}")
    compatible, skipped = [], []
    for spec in opponents:
        error = checkpoint_error(spec, config)
        if error is None:
            compatible.append(spec)
        elif spec in args.opponents:
            ap.error(f"opponent {spec}: {error}")
        else:
            print(f"note: skipping {label(spec)}: {error}")
            skipped.append({"path": spec, "reason": error})
    opponents = compatible
    if not opponents:
        ap.error("no opponents left to play")

    is_ppo = checkpoint_path(args.agent) is not None
    kwargs_a = {"deterministic": True} if (args.deterministic and is_ppo) else {}
    n_decks = config.n_decks
    n_deals = deals_for_games(args.games, args.decks, n_decks)
    games = 2 * n_deals
    workers = min(args.workers, n_deals)
    mode = (" (deterministic)" if kwargs_a else " (sampling)") if is_ppo else ""
    if args.deterministic and not is_ppo:
        print("note: --deterministic only applies to PPO checkpoints; ignored")
    print(f"agent: {args.agent}{mode}")
    sched = (f"decks=all: {n_decks * n_decks} deck pairs x {n_deals // (n_decks * n_decks)} deals"
             if args.decks == "all" else "decks=sampled per deal seed")
    rounded = f" (--games {args.games} rounded up)" if games != args.games else ""
    print(f"{games} games per opponent{rounded} = {n_deals} deals (seeds {args.seed}..{args.seed + n_deals - 1}) "
          f"x 2 seatings; {sched}; {workers} worker{'s' if workers > 1 else ''}")
    print("decks: " + " | ".join(f"{i} {name} ({style})" if style else f"{i} {name}"
                                 for i, (name, style) in enumerate(zip(config.deck_names, config.deck_styles))))
    print("rates are the agent's; draws count as non-wins; games/s is wall clock after warm-up")
    print()

    names = [label(o) for o in opponents]
    width = max(8, *(len(n) for n in names))
    print(table_header(width))
    results: List[Tuple[str, MatchResult]] = []
    reference: Optional[MatchResult] = None
    with Evaluator(workers, config) as ev:
        ev.warmup(args.agent, opponents[0], agent_kwargs_a=kwargs_a)
        for spec, name in zip(opponents, names):
            r = ev.match(args.agent, spec, n_deals, args.seed, agent_kwargs_a=kwargs_a, decks=args.decks)
            results.append((spec, r))
            print(table_row(name, r, width), flush=True)
        if not args.no_reference:
            for spec, r in results:
                if args.agent == "greedy" and spec == "greedy":
                    reference = r
            if reference is None:
                reference = ev.match("greedy", "greedy", n_deals, args.seed, decks=args.decks)

    print()
    print(f"== matchup cells C{{i,j}} per opponent: agent's win% over both decks of every (i,j)/(j,i) deal "
          f"({', '.join(f'{s}={n}' for s, n in zip(short_deck_names(config.deck_names), config.deck_names))})")
    print(cells_table([(label(spec), r) for spec, r in results], width))

    for spec, r in results:
        if spec not in args.opponents:
            continue  # checkpoint opponents: summary and cell rows above, full tables in --json
        print()
        print(f"== vs {label(spec)}: matchup cells C{{i,j}} = agent's win% over both decks of every (i,j)/(j,i) "
              f"deal [95% CI] (games)")
        print(indent(r.format_cells()))
        print(f"== vs {label(spec)}: deck-confounded P[a][b] = agent's win% holding deck a (rows) vs deck b")
        print(indent(r.format_matrix(reference, "greedy vs greedy (reference)",
                                     f"{label(args.agent)} vs {label(spec)}")))

    scenario_reports = None
    if args.scenarios:
        print()
        print(f"== scenarios: agent = argmax run{f' + sampled % of {args.playouts} playouts' if is_ppo else ''}; "
              f"greedy = one run; random = % of 20 seeds")
        scenario_reports = run_scenario_suite(args.agent, config, is_ppo, args.playouts)
        print(indent(format_scenarios(scenario_reports, is_ppo)))
    scenario_pct = scenario_reports["agent"].pct_solved if scenario_reports else None

    check = target_check(results, args, is_ppo, scenario_pct)
    if check is not None:
        print()
        print(verdict_text(check))

    if args.json:
        payload = {"agent": args.agent, "agent_kwargs": kwargs_a, "games_requested": args.games,
                   "games_per_opponent": games, "decks": args.decks, "start_seed": args.seed, "workers": workers,
                   "deck_names": list(config.deck_names), "deck_styles": list(config.deck_styles),
                   "target": check, "skipped": skipped,
                   "results": [{"opponent": spec, "label": label(spec), **r.to_dict()} for spec, r in results],
                   "reference": None if reference is None else reference.to_dict(),
                   "scenarios": None if scenario_reports is None else
                   {k: rep.to_dict() for k, rep in scenario_reports.items()}}
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"results written to {args.json}")

    if args.strict and (check is None or not check["passed"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
