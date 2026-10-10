"""Deck balance report and gate (SPEC section 1.4): lookahead vs lookahead and lookahead vs random on every
deck pair, mulligan on.

    python tools/deck_balance.py [--deals 4096] [--lvr-deals N] [--workers 4] [--cards PATH] [--decks PATH]

Deal k (seed start + k) gives deck i to seat 0 and deck j to seat 1 with (i, j) = divmod(k % n^2, n)
and is played twice: A in seat 0 / B in seat 1, then the seats swapped (same shuffles and coin flip).
Moves come from `cardgame.agents.choose_action`. Every agent is reset with a seed derived from (deal, seat),
so two copies of the same agent spec make the swapped game an exact replica of the first one; it is
then recorded without being replayed.

* P[a][b]: A's win rate when A holds deck a and B deck b (both seats pooled; deck-confounded).
* C{i,j}: A's win rate over every game of the deals dealt as (i, j) or (j, i); deck strength and
  turn order cancel, so this is the agent comparison per matchup.

Gate: lookahead vs lookahead P[a][b] in [0.30, 0.70] for a != b, draws <= 5% per cell, lookahead vs random
>= 0.85 in every unordered cell C{i,j}, mean game length (lookahead vs lookahead) <= 30 rounds.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cardgame.agents import choose_action, make_agent  # noqa: E402
from cardgame.cards import GameConfig, load_ruleset  # noqa: E402
from cardgame.engine import DRAW, Game  # noqa: E402

# SPEC section 1.4 deck balance gate.
P_RANGE = (0.30, 0.70)      # lookahead vs lookahead P[a][b], a != b
MAX_DRAW_RATE = 0.05        # per cell
MIN_LVR_CELL = 0.85         # lookahead vs random, every unordered cell
MAX_MEAN_ROUNDS = 30.0      # lookahead vs lookahead
GATE_AGENT = "lookahead"
# Agents whose moves depend only on the config and the seed given to reset(): two copies of one spec in
# swapped seats replay the first game exactly (agent seeds come from the seat).
SEEDED = frozenset({"greedy", "lookahead", "random"})
_MASK64 = (1 << 64) - 1
_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


class Played(NamedTuple):
    """One game from A's point of view."""
    deal: int
    a_deck: int
    b_deck: int
    a_seat: int
    outcome: int   # +1 A won, 0 draw, -1 A lost
    rounds: int


def deal_decks(k: int, n_decks: int) -> Tuple[int, int]:
    """(seat-0 deck, seat-1 deck) of deal k: every ordered pair once per n^2 deals."""
    return divmod(k % (n_decks * n_decks), n_decks)


def agent_seed(seed: int, seat: int) -> int:
    """Per-game, per-seat agent seed (splitmix64 mix, non-negative)."""
    z = (seed * 2 + seat + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) >> 1


def play(game: Game, agents: Sequence, seed: int, decks: Tuple[int, int]) -> Tuple[int, int]:
    """One full game (mulligan as configured) with agents[p] in seat p. Returns (winner, rounds)."""
    game.reset(seed, decks=decks)
    agents[0].reset(agent_seed(seed, 0))
    agents[1].reset(agent_seed(seed, 1))
    while not game.done:
        p = game.current_player()
        action = choose_action(agents[p], game)
        if not game.is_legal(action):
            raise RuntimeError(f"{agents[p].name} chose illegal {game.describe(action)} (seed {seed})")
        game.step(action)
    return game.winner(), game.round


def replicates(spec_a: str, spec_b: str) -> bool:
    """Two copies of one seeded agent spec: the seat-swapped game is an exact replica of the first one."""
    return spec_a == spec_b and spec_a in SEEDED


def _outcome(winner: int, seat: int) -> int:
    return 0 if winner == DRAW else (1 if winner == seat else -1)


def _run_deals(config: GameConfig, spec_a: str, spec_b: str, deals: Sequence[int], start_seed: int) -> List[Played]:
    game = Game(config)
    a, b = make_agent(spec_a, config), make_agent(spec_b, config)
    replica = replicates(spec_a, spec_b)
    n = config.n_decks
    out = []
    for k in deals:
        i, j = deal_decks(k, n)
        seed = start_seed + k
        w, r = play(game, (a, b), seed, (i, j))
        out.append(Played(k, i, j, 0, _outcome(w, 0), r))
        if not replica:
            w, r = play(game, (b, a), seed, (i, j))
        out.append(Played(k, j, i, 1, _outcome(w, 1), r))
    return out


_CONFIG: Optional[GameConfig] = None


def _init_worker(config: GameConfig) -> None:
    global _CONFIG
    _CONFIG = config


def _worker_chunk(task: tuple) -> List[Played]:
    return _run_deals(_CONFIG, *task)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p, z2 = k / n, z * z
    d = 1 + z2 / n
    c = (p + z2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


@dataclass
class Cell:
    wins: int = 0
    draws: int = 0
    games: int = 0
    rounds: int = 0

    def add(self, g: Played) -> None:
        self.games += 1
        self.wins += g.outcome > 0
        self.draws += g.outcome == 0
        self.rounds += g.rounds

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else float("nan")

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games if self.games else float("nan")

    @property
    def mean_rounds(self) -> float:
        return self.rounds / self.games if self.games else float("nan")


@dataclass
class BalanceResult:
    spec_a: str
    spec_b: str
    deck_names: tuple
    games: List[Played]
    seconds: float = 0.0
    simulated: int = -1              # games actually played (mirrors of one seeded spec replicate the swap)
    P: list = field(init=False)      # P[a][b]: A holds deck a, B holds deck b
    C: dict = field(init=False)      # C[(i, j)], i <= j: deals dealt as (i, j) or (j, i)
    total: Cell = field(init=False)

    def __post_init__(self) -> None:
        if self.simulated < 0:
            self.simulated = len(self.games)
        n = len(self.deck_names)
        self.P = [[Cell() for _ in range(n)] for _ in range(n)]
        self.C = {(i, j): Cell() for i in range(n) for j in range(i, n)}
        self.total = Cell()
        for g in self.games:
            self.P[g.a_deck][g.b_deck].add(g)
            self.C[(min(g.a_deck, g.b_deck), max(g.a_deck, g.b_deck))].add(g)
            self.total.add(g)

    @property
    def n_decks(self) -> int:
        return len(self.deck_names)

    @property
    def games_per_s(self) -> float:
        """Simulated games per second."""
        return self.simulated / self.seconds if self.seconds > 0 else float("nan")

    def format(self) -> str:
        n, names = self.n_decks, self.deck_names
        w = max(8, max(len(s) for s in names) + 1)
        t = self.total
        lines = [f"{self.spec_a} vs {self.spec_b}: {t.games} games ({t.games // 2} deals, {self.simulated} simulated), "
                 f"A win {t.win_rate:.3f}, draws {t.draw_rate:.3f}, mean rounds {t.mean_rounds:.1f}"
                 + (f", {self.games_per_s:.0f} simulated games/s" if self.seconds > 0 else "")]
        for title, value in (("P[a][b] = A's win rate, A holds row deck a, B holds column deck b",
                              lambda c: f"{c.win_rate:.3f}"),
                             ("draw rate", lambda c: f"{c.draw_rate:.3f}"),
                             ("mean rounds", lambda c: f"{c.mean_rounds:.1f}")):
            lines.append(f"  {title}")
            lines.append("  " + " " * w + "".join(f"{s:>{w}}" for s in names))
            for a in range(n):
                lines.append("  " + f"{names[a]:<{w}}" + "".join(f"{value(self.P[a][b]):>{w}}" for b in range(n)))
        lines.append("  unordered cells C{i,j} (A's win rate, both decks and both seats; Wilson 95% CI)")
        for (i, j), c in self.C.items():
            lo, hi = wilson(c.wins, c.games)
            lines.append(f"    {names[i]:>{w}} / {names[j]:<{w}} {c.win_rate:.3f}  [{lo:.3f}, {hi:.3f}]  "
                         f"draws {c.draw_rate:.3f}  n={c.games}")
        return "\n".join(lines)


def run_match(config: GameConfig, spec_a: str, spec_b: str, n_deals: int, start_seed: int = 0,
              workers: int = 1) -> BalanceResult:
    """A vs B over n_deals (rounded up to a multiple of n_decks^2) duplicate deals."""
    n2 = config.n_decks ** 2
    n_deals = max(1, math.ceil(n_deals / n2)) * n2
    t0 = time.perf_counter()
    if workers <= 1:
        games = _run_deals(config, spec_a, spec_b, range(n_deals), start_seed)
    else:
        size = max(1, math.ceil(n_deals / (4 * workers)))
        tasks = [(spec_a, spec_b, range(s, min(s + size, n_deals)), start_seed) for s in range(0, n_deals, size)]
        saved = {k: os.environ.get(k) for k in _THREAD_ENV}
        for k in _THREAD_ENV:
            os.environ.setdefault(k, "1")
        try:  # a worker that dies raises BrokenProcessPool instead of hanging; map() keeps the task order
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                                     initargs=(config,)) as pool:
                games = [g for part in pool.map(_worker_chunk, tasks) for g in part]
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return BalanceResult(spec_a, spec_b, tuple(config.deck_names), games, time.perf_counter() - t0,
                         len(games) // 2 if replicates(spec_a, spec_b) else len(games))


def gate_failures(lvl: BalanceResult, lvr: BalanceResult) -> List[str]:
    """Violations of the SPEC section 1.4 deck balance gate (empty list = pass). `lvl`: lookahead vs
    lookahead; `lvr`: lookahead vs random."""
    names, n = lvl.deck_names, lvl.n_decks
    who = f"{lvl.spec_a} vs {lvl.spec_b}"
    out = []
    for a in range(n):
        for b in range(n):
            c = lvl.P[a][b]
            if a != b and not P_RANGE[0] <= c.win_rate <= P_RANGE[1]:
                out.append(f"{who} P[{names[a]}][{names[b]}] = {c.win_rate:.3f} outside {P_RANGE}")
            if c.draw_rate > MAX_DRAW_RATE:
                out.append(f"{who} draws {names[a]} vs {names[b]} = {c.draw_rate:.3f} > {MAX_DRAW_RATE}")
    for (i, j), c in lvr.C.items():
        if c.win_rate < MIN_LVR_CELL:
            out.append(f"{lvr.spec_a} vs {lvr.spec_b} C{{{names[i]},{names[j]}}} = {c.win_rate:.3f} < {MIN_LVR_CELL}")
    if lvl.total.mean_rounds > MAX_MEAN_ROUNDS:
        out.append(f"{who} mean game length {lvl.total.mean_rounds:.1f} > {MAX_MEAN_ROUNDS} rounds")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--deals", type=int, default=4096, help="deals per match (rounded up to a multiple of n^2)")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--cards", default=None, help="cards.json to evaluate (default: shipped)")
    ap.add_argument("--decks", default=None, help="decks.json to evaluate (default: shipped)")
    ap.add_argument("--lvr-deals", type=int, default=None, help="deals for lookahead vs random (default: --deals)")
    args = ap.parse_args(argv)
    config = load_ruleset(args.cards, args.decks)  # mulligan on (the shipped default)
    lvl = run_match(config, GATE_AGENT, GATE_AGENT, args.deals, args.start_seed, args.workers)
    print(lvl.format(), flush=True)
    lvr = run_match(config, GATE_AGENT, "random", args.lvr_deals or args.deals, args.start_seed, args.workers)
    print(lvr.format())
    failures = gate_failures(lvl, lvr)
    off = [abs(lvl.P[a][b].win_rate - 0.5) for a in range(lvl.n_decks) for b in range(lvl.n_decks) if a != b]
    print(f"max |P - 0.5| (a != b) = {max(off):.3f}")
    print("deck balance gate: " + ("PASS" if not failures else "FAIL\n  " + "\n  ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
