#!/usr/bin/env python
"""Duplicate-game evaluation on random and fixed decks, head-to-head vs a baseline, and the tactical
scenarios: the Stage 3 verdict (SPEC 10).

    python eval.py --agent best.pt --baseline pooled.pt --scenarios --json out.json   # the full verdict
    python eval.py --agent runs/s3/best.pt                         # vs lookahead, random + fixed decks
    python eval.py --agent runs/s3/best.pt --opponents lookahead greedy random \\
                   --checkpoints-dir runs/s3 --checkpoint-stride 5 --decks random
    python eval.py --agent lookahead --opponents random --games 256 --decks sampled

Every deal is played twice with the seats swapped (the decks stay with the seats). Deck modes
(--decks, default "random all"; every opponent is played in every listed mode):
  random   deal k gives seat 0 / seat 1 the decks generate_deck(deck_rng(seed + k, 0 / 1)); results
           report overall, per-seat and turn-order rates (no matchup cells)
  all      the 16 ordered fixed deck pairs cycle; --games is rounded up to a multiple of 2 x n_decks^2
  sampled  sample_decks(seed) per deal
Agent specs: "random", "greedy", "lookahead", or a PPO checkpoint ("<path>.pt" or "ppo:<path>").
All rates are from --agent's point of view; draws count as non-wins.

Done when (SPEC 10), checked against lookahead with seeds from 0 and the sampling policy:
  (1) win rate >= 70% over >= 2000 duplicate games on random decks;
  (2) every unordered fixed-deck matchup cell >= 60% (decks all, >= 2000 games);
  (3) --baseline <pooled.pt>: the agent (a Transformer) beats the pooled baseline head-to-head
      (win rate > 50% over >= 2000 duplicate games on random decks);
  (4) --scenarios: > 50% of the scenarios solved (argmax).
PASS/FAIL is printed only when all four were measured under these settings (thresholds not lowered);
otherwise the verdict is INDICATIVE with the reasons. The worst ordered (deck-confounded) cell
P[a][b] is shown next to it as a diagnostic.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from cardgame.agents import make_agent
from cardgame.cards import load_ruleset
from cardgame.evaluation import (DECK_MODES, FIXED_DECK_MODES, RANDOM_DECKS, Evaluator, MatchResult, checkpoint_path,
                                 deals_for_games, default_workers, short_deck_names)

TARGET_OPPONENT = "lookahead"
TARGET_GAMES = 2000  # the brief's acceptance tests: >= 2,000 duplicate games each
DEFAULT_DECKS = (RANDOM_DECKS, "all")
BASELINE_KIND = "pooled"       # criterion (3): the Transformer vs the pooled baseline
AGENT_KIND = "transformer"
# The brief's thresholds (SPEC 10): lower values give an INDICATIVE verdict, never a PASS.
BRIEF_THRESHOLDS = (("target", "--target", 0.70), ("cell_target", "--cell-target", 0.60),
                    ("baseline_target", "--baseline-target", 0.50), ("scenario_target", "--scenario-target", 0.50))
PARTS = ("random_decks", "cells", "baseline", "scenarios")
SCENARIO_REFERENCES = ("greedy",)  # SPEC 10: "greedy is reported for reference" next to the two baselines

Result = Tuple[str, str, MatchResult]  # (opponent spec, deck mode, result)


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


def checkpoint_check(spec: str, config) -> Tuple[Optional[str], Optional[str]]:
    """(error, network kind) of an agent spec: (None, None) for scripted agents; for a checkpoint, the
    reason it does not load against this ruleset, or its network kind ("transformer", "pooled", "mlp")."""
    if checkpoint_path(spec) is None:
        return None, None
    try:
        agent = make_agent(spec, config)
    except Exception as exc:  # noqa: BLE001 - reported to the user verbatim
        return f"{type(exc).__name__}: {exc}", None
    return None, getattr(getattr(agent, "net", None), "kind", None)


def checkpoint_error(spec: str, config) -> Optional[str]:
    """None if `spec` is not a checkpoint or loads against this ruleset, else the reason it does not."""
    return checkpoint_check(spec, config)[0]


def same_file(a: str, b: str) -> bool:
    pa, pb = checkpoint_path(a), checkpoint_path(b)
    return pa is not None and pb is not None and os.path.abspath(pa) == os.path.abspath(pb)


def describe_kind(kind: Optional[str]) -> str:
    return "a scripted agent" if kind is None else f"a {kind!r} network"


# ---------------------------------------------------------------------- tables
COLUMNS = (("decks", 7), ("games", 6), ("win%", 6), ("draw%", 6), ("loss%", 6), ("win 95% CI", 14), ("seat0%", 7),
           ("seat1%", 7), ("first%", 7), ("second%", 8), ("worst cell", 22), ("games/s", 8))


def table_header(name_width: int) -> str:
    return f"{'opponent':<{name_width}s}" + "".join(f" {title:>{w}s}" for title, w in COLUMNS)


def table_row(name: str, r: MatchResult, name_width: int) -> str:
    lo, hi = r.win_ci()
    worst = r.min_cell()
    worst_text = "-" if worst is None else f"{r.cell_name(worst[0])} {100 * worst[1].win_rate:.1f}"
    decks = r.decks if isinstance(r.decks, str) else "pair"
    cells = (decks, f"{r.games}", f"{100 * r.win_rate:.1f}", f"{100 * r.draw_rate:.1f}", f"{100 * r.loss_rate:.1f}",
             f"[{100 * lo:.1f}, {100 * hi:.1f}]", f"{100 * r.by_seat[0].win_rate:.1f}",
             f"{100 * r.by_seat[1].win_rate:.1f}", f"{100 * r.first.win_rate:.1f}",
             f"{100 * r.second.win_rate:.1f}", worst_text, f"{r.games_per_s:.0f}")
    return f"{name:<{name_width}s}" + "".join(f" {c:>{w}s}" for c, (_, w) in zip(cells, COLUMNS))


def cells_table(results: List[Tuple[str, MatchResult]], width: int) -> str:
    """One row per opponent, one column per unordered matchup cell C{i,j} (agent's win%), plus the games
    per cell (the same deals for every opponent). Fixed-deck results only."""
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


# ---------------------------------------------------------------------- the verdict
def find_result(results: Sequence[Result], spec: str, mode: str) -> Optional[MatchResult]:
    for s, m, r in results:
        if s == spec and m == mode:
            return r
    return None


def _rate(r: MatchResult) -> dict:
    lo, hi = r.win_ci()
    return {"games": r.games, "win_rate": r.win_rate, "draw_rate": r.draw_rate, "score": r.score,
            "win_rate_ci95": [lo, hi]}


def stage3_verdict(results: Sequence[Result], baseline: Optional[MatchResult], args, is_ppo: bool,
                   scenario_pct: Optional[float], agent_kind: Optional[str] = None,
                   baseline_kind: Optional[str] = None) -> Optional[dict]:
    """The SPEC 10 "Done when" verdict (four parts); None when no part was measured.

    `checks` holds pass/fail for the measured parts; the verdict is conclusive only when all four were
    measured with the brief's settings (lookahead opponent, >= 2000 games each, random decks for (1)
    and (3), decks=all for (2), seeds from 0, sampling policy, thresholds not lowered, a Transformer
    agent against a pooled baseline); otherwise `indicative_reasons` lists what differs.
    """
    r1, r2 = find_result(results, TARGET_OPPONENT, RANDOM_DECKS), find_result(results, TARGET_OPPONENT, "all")
    if r1 is None and r2 is None and baseline is None and scenario_pct is None:
        return None
    reasons: List[str] = []
    if args.seed != 0:
        reasons.append(f"seeds start at {args.seed} (the test uses seeds from 0)")
    if is_ppo and args.deterministic:
        reasons.append("--deterministic (the test samples from the policy)")
    for attr, flag, brief in BRIEF_THRESHOLDS:
        if getattr(args, attr) < brief:
            reasons.append(f"{flag} {getattr(args, attr):g} below the brief's {brief:g}")
    checks, parts = {}, {k: None for k in PARTS}

    if r1 is None:
        reasons.append(f"(1) not measured: no random-deck match vs {TARGET_OPPONENT} (--decks random)")
    else:
        if r1.games < TARGET_GAMES:
            reasons.append(f"(1) {r1.games} games < {TARGET_GAMES} (random decks vs {TARGET_OPPONENT})")
        checks["random_decks"] = r1.win_rate >= args.target
        parts["random_decks"] = {"opponent": TARGET_OPPONENT, "decks": RANDOM_DECKS, "target": args.target,
                                 "passed": checks["random_decks"], **_rate(r1)}
    if r2 is None:
        reasons.append(f"(2) not measured: no fixed-deck match vs {TARGET_OPPONENT} (--decks all)")
    else:
        if r2.games < TARGET_GAMES:
            reasons.append(f"(2) {r2.games} games < {TARGET_GAMES} (fixed decks vs {TARGET_OPPONENT})")
        worst, ordered = r2.min_cell(), r2.min_ordered_cell()
        checks["cells"] = worst is not None and worst[1].win_rate >= args.cell_target
        names = r2.deck_names
        parts["cells"] = {
            "opponent": TARGET_OPPONENT, "decks": "all", "cell_target": args.cell_target, "passed": checks["cells"],
            **_rate(r2),
            "worst_cell": None if worst is None else {"name": r2.cell_name(worst[0]), "decks": list(worst[0]),
                                                     "win_rate": worst[1].win_rate, "games": worst[1].games,
                                                     "win_rate_ci95": list(worst[1].win_ci())},
            "worst_ordered_cell": None if ordered is None else {
                "diagnostic": True, "name": r2.ordered_cell_name(ordered[0]), "decks": list(ordered[0]),
                "agent_deck": names[ordered[0][0]], "opponent_deck": names[ordered[0][1]],
                "win_rate": ordered[1].win_rate, "games": ordered[1].games,
                "win_rate_ci95": list(ordered[1].win_ci())}}
    if baseline is None:
        reasons.append("(3) not measured: no baseline (--baseline <pooled.pt>)")
    else:
        if baseline.games < TARGET_GAMES:
            reasons.append(f"(3) {baseline.games} games < {TARGET_GAMES} (head-to-head vs the baseline)")
        if baseline.decks != RANDOM_DECKS:
            reasons.append(f"(3) decks={baseline.decks} (the test uses random decks)")
        if baseline_kind != BASELINE_KIND:
            reasons.append(f"(3) the baseline is {describe_kind(baseline_kind)}, not the pooled network")
        if agent_kind != AGENT_KIND:
            reasons.append(f"(3) the agent is {describe_kind(agent_kind)}, not a Transformer network")
        checks["baseline"] = baseline.win_rate > args.baseline_target
        parts["baseline"] = {"baseline": baseline.agent_b, "label": label(baseline.agent_b),
                             "baseline_kind": baseline_kind, "agent_kind": agent_kind,
                             "decks": baseline.decks if isinstance(baseline.decks, str) else list(baseline.decks),
                             "baseline_target": args.baseline_target, "passed": checks["baseline"], **_rate(baseline)}
    if scenario_pct is None:
        reasons.append("(4) not measured: scenarios not run (--scenarios)")
    else:
        checks["scenarios"] = scenario_pct > 100 * args.scenario_target
        parts["scenarios"] = {"pct_solved": scenario_pct, "scenario_target": args.scenario_target,
                              "passed": checks["scenarios"]}
    return {"opponent": TARGET_OPPONENT, "target": args.target, "cell_target": args.cell_target,
            "baseline_target": args.baseline_target, "scenario_target": args.scenario_target,
            "parts": parts, "checks": checks, "measured": [k for k in PARTS if k in checks],
            "conclusive": not reasons, "indicative_reasons": reasons,
            "passed": not reasons and all(checks.values())}


def verdict_text(check: dict) -> str:
    """The verdict line followed by one line per part (and the ordered-cell diagnostic)."""
    if not check["conclusive"]:
        head = "INDICATIVE (" + "; ".join(check["indicative_reasons"]) + ")"
    else:
        head = "PASS" if check["passed"] else "FAIL"
    lines = [f"verdict: {head}"]
    p = check["parts"]

    def ok(part: str) -> str:
        return "ok" if check["checks"][part] else "MISSED"

    if p["random_decks"] is None:
        lines.append(f"  (1) vs {check['opponent']}, random decks: not measured")
    else:
        x = p["random_decks"]
        lines.append(f"  (1) vs {check['opponent']}, random decks: win rate {x['win_rate']:.1%} "
                     f"({'>=' if x['passed'] else '<'} {x['target']:.0%}) over {x['games']} duplicate games "
                     f"(95% CI lower bound {x['win_rate_ci95'][0]:.1%}) [{ok('random_decks')}]")
    if p["cells"] is None:
        lines.append(f"  (2) vs {check['opponent']}, fixed decks: not measured")
    else:
        x, wc = p["cells"], p["cells"]["worst_cell"]
        worst = "no cell played" if wc is None else (
            f"worst matchup cell {wc['name']} {wc['win_rate']:.1%} ({'>=' if x['passed'] else '<'} "
            f"{x['cell_target']:.0%}) over {wc['games']} games")
        lines.append(f"  (2) vs {check['opponent']}, fixed decks (all): {worst}; overall {x['win_rate']:.1%} over "
                     f"{x['games']} games [{ok('cells')}]")
    if p["baseline"] is None:
        lines.append("  (3) head-to-head vs the baseline: not measured")
    else:
        x = p["baseline"]
        lines.append(f"  (3) head-to-head vs {x['label']}: win rate {x['win_rate']:.1%} "
                     f"({'>' if x['passed'] else '<='} {x['baseline_target']:.0%}; score {x['score']:.3f}) over "
                     f"{x['games']} duplicate games, random decks [{ok('baseline')}]")
    if p["scenarios"] is None:
        lines.append("  (4) scenarios: not measured")
    else:
        x = p["scenarios"]
        lines.append(f"  (4) scenarios: {x['pct_solved']:.0f}% solved, argmax "
                     f"({'>' if x['passed'] else '<='} {100 * x['scenario_target']:.0f}%) [{ok('scenarios')}]")
    wo = p["cells"]["worst_ordered_cell"] if p["cells"] is not None else None
    if wo is not None:
        lines.append(f"  diagnostic only (the verdict uses the unordered cells, SPEC 10): worst ordered cell "
                     f"{wo['name']} = {wo['win_rate']:.1%} over {wo['games']} games (agent holds {wo['agent_deck']}, "
                     f"{check['opponent']} holds {wo['opponent_deck']}; deck-confounded)")
    return "\n".join(lines)


# ---------------------------------------------------------------------- scenarios
def run_scenario_suite(agent: str, config, is_ppo: bool, playouts: int) -> dict:
    """Scenario reports for the agent (argmax run, plus sampled playouts for PPO), the lookahead /
    random baselines and the greedy reference (SPEC 10)."""
    from cardgame.scenarios import BASELINES, run_scenarios
    reports = {"agent": run_scenarios(agent, config, deterministic=True, n_stochastic=playouts if is_ppo else 0)}
    for base in BASELINES + SCENARIO_REFERENCES:
        reports[base] = reports["agent"] if agent == base else run_scenarios(base, config)
    return reports


def format_scenarios(reports: dict, is_ppo: bool) -> str:
    agent, look, random_, greedy = reports["agent"], reports["lookahead"], reports["random"], reports["greedy"]
    width = max(len("scenario"), *(len(r.name) for r in agent.results))

    def det(r) -> str:
        return "-" if r.solved is None else ("yes" if r.solved else "no")

    def rate(r) -> str:
        return "-" if r.success is None else f"{100 * r.success:.0f}%"

    def total(rep) -> str:
        if rep.deterministic:
            return f"{rep.n_solved}/{rep.n}"
        return f"{rep.pct_solved:.0f}%"

    lines = [f"{'scenario':<{width}s} {'agent':>6s} {'sampled':>8s} {'lookahead':>9s} {'random':>7s} "
             f"{'greedy':>6s}  tags"]
    for a, g, r, gr in zip(agent.results, look.results, random_.results, greedy.results):
        a_det = det(a) if a.solved is not None else rate(a)
        lines.append(f"{a.name:<{width}s} {a_det:>6s} {rate(a) if is_ppo else '-':>8s} {det(g):>9s} "
                     f"{rate(r):>7s} {det(gr):>6s}  {','.join(a.tags)}")
    sampled = f"{100 * agent.mean_success:.0f}%" if (is_ppo and agent.mean_success is not None) else "-"
    lines.append(f"{'solved':<{width}s} {total(agent):>6s} {sampled:>8s} {total(look):>9s} {total(random_):>7s} "
                 f"{total(greedy):>6s}")
    return "\n".join(lines)


# ---------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True,
                    help="agent to evaluate (random, greedy, lookahead, or a .pt checkpoint)")
    ap.add_argument("--opponents", nargs="+", default=[TARGET_OPPONENT], metavar="SPEC",
                    help="opponent specs (default: lookahead)")
    ap.add_argument("--checkpoints-dir", default=None, metavar="DIR",
                    help="also play against the ckpt_*.pt files in DIR (oldest first; incompatible ones are skipped)")
    ap.add_argument("--checkpoint-stride", type=int, default=1, metavar="K",
                    help="use every K-th checkpoint, counted back from the newest (default 1 = all)")
    ap.add_argument("--baseline", default=None, metavar="SPEC",
                    help="head-to-head opponent for criterion (3), normally the pooled checkpoint; played on random "
                         "decks with --games games")
    ap.add_argument("--games", type=int, default=TARGET_GAMES,
                    help="games per opponent and deck mode (2 per deal; decks=all rounds up to a multiple of "
                         "2*n_decks^2; default 2000 -> 2016 for decks=all)")
    ap.add_argument("--decks", nargs="+", choices=DECK_MODES, default=list(DEFAULT_DECKS), metavar="MODE",
                    help=f"deck modes to play every opponent in, from {', '.join(DECK_MODES)} (default: random all)")
    ap.add_argument("--seed", type=int, default=0, help="first deal seed (default 0)")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help=f"worker processes (default min(8, cpu_count) = {default_workers()})")
    ap.add_argument("--deterministic", action="store_true",
                    help="argmax policy for PPO checkpoints (agent and baseline) in the matches "
                         "(default: sample from the policy)")
    ap.add_argument("--target", type=float, default=0.70,
                    help="(1) win-rate target vs lookahead on random decks (default 0.70)")
    ap.add_argument("--cell-target", type=float, default=0.60,
                    help="(2) minimum win rate in every fixed-deck matchup cell vs lookahead (default 0.60)")
    ap.add_argument("--baseline-target", type=float, default=0.50,
                    help="(3) head-to-head win rate the agent must exceed vs --baseline (default 0.50)")
    ap.add_argument("--scenarios", action="store_true",
                    help="also run the scenario suite for the agent and the lookahead/random baselines")
    ap.add_argument("--scenario-target", type=float, default=0.50,
                    help="(4) fraction of scenarios the agent must solve, exclusive (default 0.50)")
    ap.add_argument("--playouts", type=int, default=100,
                    help="sampled playouts per scenario for a PPO agent (default 100)")
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the lookahead-vs-lookahead reference matrix printed next to P[a][b]")
    ap.add_argument("--json", default=None, metavar="PATH", help="write all results to this JSON file")
    ap.add_argument("--strict", action="store_true",
                    help="exit with code 1 unless the verdict is PASS (all four parts measured with the brief's "
                         "settings)")
    args = ap.parse_args(argv)

    modes: List[str] = []
    for m in args.decks:
        if m not in modes:
            modes.append(m)
    if args.games < 1:
        ap.error("--games must be >= 1")
    if args.games % 2 and any(m != "all" for m in modes):
        ap.error("--games must be even with --decks random or sampled (each deal is played twice)")
    if args.games % 2 and args.baseline is not None:
        ap.error("--games must be even with --baseline (random decks; each deal is played twice)")
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
    for spec in (args.agent, *opponents, *([args.baseline] if args.baseline else [])):
        path = checkpoint_path(spec)
        if path is not None and not os.path.isfile(path):
            ap.error(f"checkpoint not found: {path}")
    # Load every checkpoint once up front: an incompatible one (a Stage 1/2 checkpoint, another card
    # pool or encoding) fails here with a clear message instead of mid-run in a worker.
    error, agent_kind = checkpoint_check(args.agent, config)
    if error is not None:
        ap.error(f"agent {args.agent}: {error}")
    baseline_kind = None
    if args.baseline is not None:
        error, baseline_kind = checkpoint_check(args.baseline, config)
        if error is not None:
            ap.error(f"baseline {args.baseline}: {error}")
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
    kwargs_base = ({"deterministic": True} if (args.deterministic and args.baseline is not None
                                               and checkpoint_path(args.baseline) is not None) else {})
    n_decks = config.n_decks
    n_deals = {m: deals_for_games(args.games, m, n_decks) for m in modes}
    random_deals = deals_for_games(args.games, RANDOM_DECKS, n_decks)
    workers = min(args.workers, max([*n_deals.values(), random_deals if args.baseline else 1]))
    mode_text = (" (deterministic)" if kwargs_a else " (sampling)") if is_ppo else ""
    if args.deterministic and not is_ppo:
        print("note: --deterministic only applies to PPO checkpoints; ignored for the agent")
    print(f"agent: {args.agent}{mode_text}" + (f" [{agent_kind} network]" if agent_kind else ""))
    for m in modes:
        games = 2 * n_deals[m]
        rounded = f" (--games {args.games} rounded up)" if games != args.games else ""
        sched = {RANDOM_DECKS: "random decks generate_deck(deck_rng(seed, seat)) per deal",
                 "all": f"{n_decks * n_decks} fixed deck pairs x {n_deals[m] // (n_decks * n_decks)} deals",
                 "sampled": "fixed decks sample_decks(seed) per deal"}[m]
        print(f"decks={m}: {games} games per opponent{rounded} = {n_deals[m]} deals (seeds {args.seed}.."
              f"{args.seed + n_deals[m] - 1}) x 2 seatings; {sched}")
    if args.baseline is not None:
        print(f"baseline: {args.baseline}" + (f" [{baseline_kind} network]" if baseline_kind else "")
              + f": {2 * random_deals} games on random decks")
    print(f"{workers} worker{'s' if workers > 1 else ''}")
    print("fixed decks: " + " | ".join(f"{i} {name} ({style})" if style else f"{i} {name}"
                                       for i, (name, style) in enumerate(zip(config.deck_names, config.deck_styles))))
    print("rates are the agent's; draws count as non-wins; games/s is wall clock after warm-up")
    print()

    names = {o: label(o) for o in opponents}
    baseline_name = None if args.baseline is None else label(args.baseline) + " (baseline)"
    width = max(8, *(len(n) for n in names.values()), len(baseline_name or ""))
    print(table_header(width))
    results: List[Result] = []
    baseline: Optional[MatchResult] = None
    references: dict = {}
    with Evaluator(workers, config) as ev:
        ev.warmup(args.agent, opponents[0], agent_kwargs_a=kwargs_a)
        if args.baseline is not None:  # load the baseline in the workers before its match is timed
            ev.warmup(args.agent, args.baseline, agent_kwargs_a=kwargs_a, agent_kwargs_b=kwargs_base)
        for m in modes:
            for spec in opponents:
                r = ev.match(args.agent, spec, n_deals[m], args.seed, agent_kwargs_a=kwargs_a, decks=m)
                results.append((spec, m, r))
                print(table_row(names[spec], r, width), flush=True)
        if args.baseline is not None:
            baseline = ev.match(args.agent, args.baseline, random_deals, args.seed, agent_kwargs_a=kwargs_a,
                                agent_kwargs_b=kwargs_base, decks=RANDOM_DECKS)
            print(table_row(baseline_name, baseline, width), flush=True)
        if not args.no_reference:
            for m in modes:
                if m not in FIXED_DECK_MODES:
                    continue
                own = find_result(results, TARGET_OPPONENT, m) if args.agent == TARGET_OPPONENT else None
                references[m] = own if own is not None else ev.match(TARGET_OPPONENT, TARGET_OPPONENT, n_deals[m],
                                                                     args.seed, decks=m)

    short = short_deck_names(config.deck_names)
    for m in modes:
        if m not in FIXED_DECK_MODES:
            continue
        print()
        print(f"== decks={m}: matchup cells C{{i,j}} per opponent: agent's win% over both decks of every (i,j)/(j,i) "
              f"deal ({', '.join(f'{s}={n}' for s, n in zip(short, config.deck_names))})")
        print(cells_table([(label(spec), r) for spec, mm, r in results if mm == m], width))
        for spec, mm, r in results:
            if mm != m or spec not in args.opponents:
                continue  # checkpoint opponents: summary and cell rows above, full tables in --json
            print()
            print(f"== decks={m}, vs {label(spec)}: matchup cells C{{i,j}} = agent's win% over both decks of every "
                  f"(i,j)/(j,i) deal [95% CI] (games)")
            print(indent(r.format_cells()))
            print(f"== decks={m}, vs {label(spec)}: deck-confounded P[a][b] = agent's win% holding deck a (rows) "
                  f"vs deck b")
            print(indent(r.format_matrix(references.get(m), f"{TARGET_OPPONENT} vs {TARGET_OPPONENT} (reference)",
                                         f"{label(args.agent)} vs {label(spec)}")))

    scenario_reports = None
    if args.scenarios:
        print()
        print(f"== scenarios: agent = argmax run{f' + sampled % of {args.playouts} playouts' if is_ppo else ''}; "
              f"lookahead = one seeded run; random = % of 20 seeds; greedy = one run (reference)")
        scenario_reports = run_scenario_suite(args.agent, config, is_ppo, args.playouts)
        print(indent(format_scenarios(scenario_reports, is_ppo)))
    scenario_pct = scenario_reports["agent"].pct_solved if scenario_reports else None

    check = stage3_verdict(results, baseline, args, is_ppo, scenario_pct, agent_kind, baseline_kind)
    if check is not None:
        print()
        print(verdict_text(check))

    if args.json:
        payload = {"agent": args.agent, "agent_kind": agent_kind, "agent_kwargs": kwargs_a,
                   "games_requested": args.games, "decks": modes,
                   "games_per_opponent": {m: 2 * n_deals[m] for m in modes}, "start_seed": args.seed,
                   "workers": workers, "deck_names": list(config.deck_names), "deck_styles": list(config.deck_styles),
                   "verdict": check, "skipped": skipped,
                   "results": [{"opponent": spec, "label": label(spec), **r.to_dict()} for spec, m, r in results],
                   "baseline": None if baseline is None else {
                       "spec": args.baseline, "label": label(args.baseline), "kind": baseline_kind,
                       "kwargs": kwargs_base, **baseline.to_dict()},
                   "reference": {m: r.to_dict() for m, r in references.items()} or None,
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
