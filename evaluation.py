"""Duplicate-game evaluation (SPEC §10): play_game, duplicate matches over deck matchups, MatchResult.

Every deal (seed + deck pair: deck i in seat 0, deck j in seat 1) is played twice with identical
shuffles and coin flip: A in seat 0 vs B in seat 1, then with the seats swapped, so A plays both
decks of every deal. Agent randomness is re-seeded per game and per *seat* (`agent_seed`), so a
mirror match of identical agents is exactly balanced, and results never depend on the number of
worker processes.

Deck schedules (`decks=`): "all" deals deal k with (i, j) = divmod(k % n², n) and rounds the deal
count up to a multiple of n² (every ordered deck pair equally often); "sampled" uses
`sample_decks(seed)` (what `Game.reset(seed)` does); a pair (i, j) fixes one matchup.

Worker processes run on a `concurrent.futures.ProcessPoolExecutor` (spawn context): a worker that dies
(crash, OOM killer) makes the match raise `BrokenProcessPool` instead of waiting forever.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union

from .agents import Agent, make_agent
from .cards import GameConfig, load_ruleset
from .engine import DRAW, Game, IllegalActionError

Z95 = 1.959963984540054  # two-sided 95% normal quantile
WARMUP_SEED = 3_000_000_000  # far from eval (0..) and training (1e9..) deal seeds
DECK_MODES = ("all", "sampled")
_MASK64 = (1 << 64) - 1
_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "NUMEXPR_NUM_THREADS")

DeckSpec = Union[str, Tuple[int, int]]
Deal = Tuple[int, Optional[Tuple[int, int]]]  # (seed, deck pair or None = sample_decks(seed))


class IllegalAgentActionError(IllegalActionError):
    """An agent returned an action that is not in the legal action list it was given."""


class GameRecord(NamedTuple):
    seed: int
    winner: int        # 0 / 1, or DRAW (-1)
    first_player: int
    rounds: int
    steps: int
    decks: Tuple[int, int]  # deck index of seat 0, seat 1


def agent_seed(seed: int, seat: int) -> int:
    """Seed for the agent in `seat` of the game dealt from `seed` (splitmix64 mix, 63-bit, non-negative)."""
    z = (seed * 2 + seat + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) >> 1


def play_game(config: Optional[GameConfig], agents: Sequence[Agent], seed: int,
              agent_seeds: Optional[Sequence[int]] = None, game: Optional[Game] = None,
              decks: Optional[Sequence[int]] = None) -> GameRecord:
    """Play one full game with `agents[p]` in seat p; agents see only observe(p) + legal_actions().

    `decks` = (deck of seat 0, deck of seat 1), default `sample_decks(seed)`. `agent_seeds` defaults
    to (agent_seed(seed, 0), agent_seed(seed, 1)). Pass `game` (built from the same config) to reuse
    it. Raises IllegalAgentActionError if an agent picks an illegal action.
    """
    if len(agents) != 2:
        raise ValueError(f"expected 2 agents, got {len(agents)}")
    if game is None:
        game = Game(config)
    game.reset(seed, decks)
    seeds = agent_seeds if agent_seeds is not None else (agent_seed(seed, 0), agent_seed(seed, 1))
    agents[0].reset(seeds[0])
    agents[1].reset(seeds[1])
    steps = 0
    while not game.done:
        p = game.current_player()
        legal = game.legal_actions()
        action = agents[p].act(game.observe(p), legal)
        if action not in legal:
            desc = game.describe(action) if isinstance(action, int) else type(action).__name__
            raise IllegalAgentActionError(
                f"agent {getattr(agents[p], 'name', agents[p])!r} in seat {p} chose illegal action "
                f"{action!r} ({desc}) in game seed={seed}, decks={tuple(game.deck_ids)}, round {game.round}, "
                f"step {steps}; legal: {[game.describe(a) for a in legal]}")
        game.step(int(action))
        steps += 1
    return GameRecord(seed, game.winner(), game.first_player, game.round, steps, tuple(game.deck_ids))


def wilson_interval(successes: int, n: int, z: float = Z95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion; (0, 1) when n == 0."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    lo = 0.0 if successes <= 0 else max(0.0, center - half)
    hi = 1.0 if successes >= n else min(1.0, center + half)
    return lo, hi


def outcome_for(record: GameRecord, seat: int) -> int:
    """+1 win, 0 draw, -1 loss for the player in `seat`."""
    if record.winner == DRAW:
        return 0
    return 1 if record.winner == seat else -1


# ---------------------------------------------------------------------- deck schedules
def check_decks(decks: DeckSpec, n_decks: int) -> DeckSpec:
    """Validate a deck schedule: "all", "sampled" or a pair of deck indices."""
    if isinstance(decks, str):
        if decks not in DECK_MODES:
            raise ValueError(f"decks must be one of {DECK_MODES} or a deck pair, got {decks!r}")
        return decks
    pair = tuple(int(d) for d in decks)
    if len(pair) != 2 or not all(0 <= d < n_decks for d in pair):
        raise ValueError(f"deck pair must be two indices in [0, {n_decks}), got {decks!r}")
    return pair


def round_deals(n_deals: int, decks: DeckSpec, n_decks: int) -> int:
    """Deal count actually played: rounded up to a multiple of n² for decks="all"."""
    if decks == "all":
        block = n_decks * n_decks
        return -(-n_deals // block) * block
    return n_deals


def deals_for_games(games: int, decks: DeckSpec, n_decks: int) -> int:
    """Deals needed for at least `games` games (2 per deal), after the decks="all" rounding."""
    return round_deals(-(-games // 2), decks, n_decks)


def deal_schedule(start_seed: int, n_deals: int, decks: DeckSpec, n_decks: int) -> List[Deal]:
    """(seed, deck pair) of every deal; deal k has seed start_seed + k. None = sample_decks(seed)."""
    decks = check_decks(decks, n_decks)
    n_deals = round_deals(n_deals, decks, n_decks)
    if decks == "all":
        block = n_decks * n_decks
        return [(start_seed + k, divmod(k % block, n_decks)) for k in range(n_deals)]
    pair = None if decks == "sampled" else decks
    return [(start_seed + k, pair) for k in range(n_deals)]


# ---------------------------------------------------------------------- results
@dataclass
class WDL:
    """Win/draw/loss counts from one agent's point of view."""
    wins: int = 0
    draws: int = 0
    losses: int = 0

    def add(self, outcome: int) -> None:
        if outcome > 0:
            self.wins += 1
        elif outcome < 0:
            self.losses += 1
        else:
            self.draws += 1

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    def _rate(self, k: float) -> float:
        return k / self.games if self.games else 0.0

    @property
    def win_rate(self) -> float:
        """Draws count as non-wins."""
        return self._rate(self.wins)

    @property
    def draw_rate(self) -> float:
        return self._rate(self.draws)

    @property
    def loss_rate(self) -> float:
        return self._rate(self.losses)

    @property
    def score(self) -> float:
        """(W + D/2) / N."""
        return self._rate(self.wins + 0.5 * self.draws)

    def win_ci(self, z: float = Z95) -> Tuple[float, float]:
        return wilson_interval(self.wins, self.games, z)

    def to_dict(self) -> dict:
        lo, hi = self.win_ci()
        return {"games": self.games, "wins": self.wins, "draws": self.draws, "losses": self.losses,
                "win_rate": self.win_rate, "draw_rate": self.draw_rate, "loss_rate": self.loss_rate,
                "score": self.score, "win_rate_ci95": [lo, hi]}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


@dataclass
class MatchResult:
    """A duplicate match from A's point of view. `deals[i] = (game with A in seat 0, game with A in seat 1)`.

    * `cells[(i, j)]` (i <= j): A's results over every game of deals dealt as (i, j) or (j, i). A plays
      both decks of each such deal, so deck strength and turn order cancel (the pass/fail numbers).
    * `matrix[a][b]`: A's results when A holds deck a and B holds deck b (deck-confounded diagnostic).

    The Wilson interval treats the games as independent; paired (duplicate) games usually have
    lower variance than that, so the interval is conservative for the A-vs-B comparison.
    """
    agent_a: str
    agent_b: str
    start_seed: int
    deals: List[Tuple[GameRecord, GameRecord]] = field(repr=False)
    elapsed: float = 0.0
    kwargs_a: dict = field(default_factory=dict)
    kwargs_b: dict = field(default_factory=dict)
    decks: DeckSpec = "all"
    deck_names: tuple = ()
    overall: WDL = field(init=False)
    by_seat: Tuple[WDL, WDL] = field(init=False)
    first: WDL = field(init=False)   # games where A moved first
    second: WDL = field(init=False)
    cells: Dict[Tuple[int, int], WDL] = field(init=False, repr=False)
    matrix: List[List[WDL]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.decks, str):
            self.decks = tuple(self.decks)
        n = len(self.deck_names) or 1 + max((d for pair in self.deals for r in pair for d in r.decks), default=0)
        if not self.deck_names:
            self.deck_names = tuple(f"deck{i}" for i in range(n))
        self.overall, self.by_seat, self.first, self.second = WDL(), (WDL(), WDL()), WDL(), WDL()
        self.cells = {(i, j): WDL() for i in range(n) for j in range(i, n)}
        self.matrix = [[WDL() for _ in range(n)] for _ in range(n)]
        for pair in self.deals:
            for a_seat, rec in enumerate(pair):
                out = outcome_for(rec, a_seat)
                self.overall.add(out)
                self.by_seat[a_seat].add(out)
                (self.first if rec.first_player == a_seat else self.second).add(out)
                i, j = rec.decks
                self.cells[(min(i, j), max(i, j))].add(out)
                self.matrix[rec.decks[a_seat]][rec.decks[1 - a_seat]].add(out)

    # -- headline numbers (overall)
    @property
    def n_deals(self) -> int:
        return len(self.deals)

    @property
    def n_decks(self) -> int:
        return len(self.deck_names)

    @property
    def games(self) -> int:
        return self.overall.games

    @property
    def wins(self) -> int:
        return self.overall.wins

    @property
    def draws(self) -> int:
        return self.overall.draws

    @property
    def losses(self) -> int:
        return self.overall.losses

    @property
    def win_rate(self) -> float:
        return self.overall.win_rate

    @property
    def draw_rate(self) -> float:
        return self.overall.draw_rate

    @property
    def loss_rate(self) -> float:
        return self.overall.loss_rate

    @property
    def score(self) -> float:
        return self.overall.score

    def win_ci(self, z: float = Z95) -> Tuple[float, float]:
        return self.overall.win_ci(z)

    @property
    def games_per_s(self) -> float:
        return self.games / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def avg_rounds(self) -> float:
        return sum(r.rounds for pair in self.deals for r in pair) / max(1, self.games)

    @property
    def avg_steps(self) -> float:
        return sum(r.steps for pair in self.deals for r in pair) / max(1, self.games)

    # -- matchups
    def cell(self, i: int, j: int) -> WDL:
        """Unordered matchup cell C{i,j}."""
        return self.cells[(min(i, j), max(i, j))]

    def min_cell(self) -> Optional[Tuple[Tuple[int, int], WDL]]:
        """The played cell with the lowest win rate (ties: first in (i, j) order), or None."""
        played = [(k, w) for k, w in self.cells.items() if w.games]
        return min(played, key=lambda kw: kw[1].win_rate) if played else None

    def min_ordered_cell(self) -> Optional[Tuple[Tuple[int, int], WDL]]:
        """The played deck-confounded cell P[a][b] (A holds deck a, B deck b) with the lowest win rate
        (ties: first in row-major order), or None. Diagnostic only: pass/fail uses the unordered cells."""
        played = [((a, b), w) for a, row in enumerate(self.matrix) for b, w in enumerate(row) if w.games]
        return min(played, key=lambda kw: kw[1].win_rate) if played else None

    def cell_name(self, key: Tuple[int, int]) -> str:
        i, j = key
        return f"{self.deck_names[i]}-{self.deck_names[j]}"

    def ordered_cell_name(self, key: Tuple[int, int]) -> str:
        a, b = key
        return f"P[{self.deck_names[a]}][{self.deck_names[b]}]"

    def deal_summary(self) -> dict:
        """Per-deal outcomes: A won both seatings, split 1-1, B won both, or a draw was involved."""
        out = {"a_won_both": 0, "split": 0, "b_won_both": 0, "with_draw": 0}
        for g0, g1 in self.deals:
            o0, o1 = outcome_for(g0, 0), outcome_for(g1, 1)
            if o0 == 0 or o1 == 0:
                out["with_draw"] += 1
            elif o0 + o1 == 2:
                out["a_won_both"] += 1
            elif o0 + o1 == -2:
                out["b_won_both"] += 1
            else:
                out["split"] += 1
        return out

    def to_dict(self, include_games: bool = False) -> dict:
        names = list(self.deck_names)
        lo, lo_ordered = self.min_cell(), self.min_ordered_cell()
        d = {"agent_a": self.agent_a, "agent_b": self.agent_b, "kwargs_a": dict(self.kwargs_a),
             "kwargs_b": dict(self.kwargs_b), "start_seed": self.start_seed, "n_deals": self.n_deals,
             "decks": self.decks if isinstance(self.decks, str) else list(self.decks), "deck_names": names,
             **self.overall.to_dict(),
             "by_seat": {"0": self.by_seat[0].to_dict(), "1": self.by_seat[1].to_dict()},
             "by_turn_order": {"first": self.first.to_dict(), "second": self.second.to_dict()},
             "matchups": [{"decks": [i, j], "names": [names[i], names[j]], **w.to_dict()}
                          for (i, j), w in self.cells.items()],
             "min_cell": None if lo is None else {"decks": list(lo[0]), "name": self.cell_name(lo[0]),
                                                  **lo[1].to_dict()},
             "min_ordered_cell": None if lo_ordered is None else {
                 "decks": list(lo_ordered[0]), "name": self.ordered_cell_name(lo_ordered[0]),
                 "diagnostic": True, **lo_ordered[1].to_dict()},
             "deck_matrix": {"rows": "A's deck", "cols": "B's deck", "names": names,
                             "win_rate": [[w.win_rate if w.games else None for w in row] for row in self.matrix],
                             "draw_rate": [[w.draw_rate if w.games else None for w in row] for row in self.matrix],
                             "games": [[w.games for w in row] for row in self.matrix]},
             "deals": self.deal_summary(), "avg_rounds": self.avg_rounds, "avg_steps": self.avg_steps,
             "elapsed_s": self.elapsed, "games_per_s": self.games_per_s}
        if include_games:
            d["games_played"] = [{"a_seat": a_seat, **rec._asdict(), "decks": list(rec.decks)}
                                 for pair in self.deals for a_seat, rec in enumerate(pair)]
        return d

    # -- text
    def format_summary(self) -> str:
        if self.decks == "all":
            sched = f"decks=all ({self.n_decks ** 2} deck pairs x {self.n_deals // max(1, self.n_decks ** 2)} deals)"
        elif self.decks == "sampled":
            sched = "decks=sampled"
        else:
            sched = f"decks={self.deck_names[self.decks[0]]} vs {self.deck_names[self.decks[1]]}"
        last = self.start_seed + self.n_deals - 1
        lines = [f"A={self.agent_a} vs B={self.agent_b}: {self.games} games = {self.n_deals} deals x 2 seatings "
                 f"(seeds {self.start_seed}..{last}, {sched})",
                 f"  {'':8s} {'games':>6s} {'win':>7s} {'draw':>7s} {'loss':>7s}"]
        rows = [("overall", self.overall), ("seat 0", self.by_seat[0]), ("seat 1", self.by_seat[1]),
                ("first", self.first), ("second", self.second)]
        for label, w in rows:
            line = f"  {label:8s} {w.games:6d} {w.win_rate:7.1%} {w.draw_rate:7.1%} {w.loss_rate:7.1%}"
            if w is self.overall:
                lo, hi = w.win_ci()
                line += f"   score {w.score:.3f}   win 95% CI [{lo:.1%}, {hi:.1%}]"
            lines.append(line)
        ds = self.deal_summary()
        lines.append(f"  deals: A won both {ds['a_won_both']}, split {ds['split']}, B won both {ds['b_won_both']}, "
                     f"with a draw {ds['with_draw']}")
        lines.append(f"  avg {self.avg_rounds:.1f} rounds, {self.avg_steps:.1f} steps per game; "
                     f"{self.elapsed:.2f} s ({self.games_per_s:.0f} games/s)")
        return "\n".join(lines)

    def format_cells(self) -> str:
        """Symmetric table of the unordered matchup cells: win% [95% CI] (games), draws if any."""
        n, names = self.n_decks, self.deck_names
        entries = {}
        for (i, j), w in self.cells.items():
            if w.games:
                lo, hi = w.win_ci()
                text = f"{_pct(w.win_rate)} [{_pct(lo)},{_pct(hi)}] ({w.games})"
                if w.draws:
                    text += f" d{_pct(w.draw_rate)}"
            else:
                text = "-"
            entries[(i, j)] = entries[(j, i)] = text
        width = max(len(e) for e in entries.values())
        name_w = max(len(x) for x in names)
        lines = [f"{'':{name_w}s}" + "".join(f"  {x:>{width}s}" for x in names)]
        for i in range(n):
            lines.append(f"{names[i]:{name_w}s}" + "".join(f"  {entries[(i, j)]:>{width}s}" for j in range(n)))
        return "\n".join(lines)

    def format_matrix(self, reference: Optional["MatchResult"] = None, reference_label: str = "",
                      label: str = "") -> str:
        """P[a][b] (A holds deck a: rows; B holds deck b: columns) as win%, optionally next to a reference."""
        def block(r: "MatchResult") -> List[str]:
            names = r.deck_names
            name_w = max(len(x) for x in names)
            col_w = max(6, *(len(x) for x in names))
            corner = "A \\ B"
            rows = [f"{corner:{name_w}s}" + "".join(f" {x:>{col_w}s}" for x in names)]
            for a in range(r.n_decks):
                cells = [_pct(w.win_rate) if w.games else "-" for w in r.matrix[a]]
                rows.append(f"{names[a]:{name_w}s}" + "".join(f" {c:>{col_w}s}" for c in cells))
            return rows

        left = block(self)
        if reference is None:
            return "\n".join(left)
        right = block(reference)
        width = max(len(x) for x in left)
        title_l = label or f"A={self.agent_a} vs B={self.agent_b}"
        title_r = reference_label or f"A={reference.agent_a} vs B={reference.agent_b}"
        out = [f"{title_l:{width}s}    {title_r}"]
        out += [f"{lx:{width}s}    {rx}" for lx, rx in zip(left, right)]
        return "\n".join(out)

    def format(self) -> str:
        parts = [self.format_summary(), "  matchup cells C{i,j}: A's win% [95% CI] (games)",
                 _indent(self.format_cells()), "  deck-confounded P[a][b]: A's win% holding deck a vs deck b",
                 _indent(self.format_matrix())]
        return "\n".join(parts)


def _indent(text: str, by: str = "    ") -> str:
    return "\n".join(by + line for line in text.splitlines())


def short_deck_names(names: Sequence[str], min_len: int = 2) -> Tuple[str, ...]:
    """The shortest distinct prefixes (at least `min_len` characters) of the deck names, for compact
    tables; the full names when no shorter prefix tells them apart."""
    names = tuple(names)
    longest = max((len(x) for x in names), default=0)
    for k in range(min_len, longest):
        short = tuple(x[:k] for x in names)
        if len(set(short)) == len(short):
            return short
    return names


# ---------------------------------------------------------------------- running matches
def checkpoint_path(spec: str) -> Optional[str]:
    """The checkpoint file of a PPO spec ("<path>.pt" or "ppo:<path>"), else None."""
    path = spec[4:] if spec.startswith("ppo:") else spec
    return path if path.endswith(".pt") else None


class _DealRunner:
    """Plays chunks of duplicate deals; builds agents from spec strings and caches them."""

    MAX_CACHED = 8

    def __init__(self, config: GameConfig):
        self.config = config
        self.game = Game(config)
        self.agents: "OrderedDict[tuple, Agent]" = OrderedDict()

    def agent(self, role: str, spec: str, kwargs: dict) -> Agent:
        key = (role, spec, tuple(sorted(kwargs.items())))
        agent = self.agents.pop(key, None)
        if agent is None:
            agent = make_agent(spec, self.config, **kwargs)
        self.agents[key] = agent  # most recently used last
        while len(self.agents) > self.MAX_CACHED:
            self.agents.popitem(last=False)
        return agent

    def run(self, task: tuple) -> List[Tuple[GameRecord, GameRecord]]:
        spec_a, kw_a, spec_b, kw_b, deals = task
        # Distinct roles keep A and B separate instances even when spec_a == spec_b.
        a, b = self.agent("a", spec_a, kw_a), self.agent("b", spec_b, kw_b)
        cfg, game = self.config, self.game
        return [(play_game(cfg, (a, b), s, game=game, decks=d), play_game(cfg, (b, a), s, game=game, decks=d))
                for s, d in deals]


_RUNNER: Optional[_DealRunner] = None


def _init_worker(config: GameConfig) -> None:
    global _RUNNER
    _RUNNER = _DealRunner(config)


def _run_chunk(task: tuple) -> List[Tuple[GameRecord, GameRecord]]:
    return _RUNNER.run(task)


@contextmanager
def _single_threaded_children() -> Iterator[None]:
    """Spawned workers inherit os.environ: keep each one to one BLAS/OpenMP thread (unless set by the user)."""
    saved = {k: os.environ.get(k) for k in _THREAD_ENV}
    for k in _THREAD_ENV:
        os.environ.setdefault(k, "1")
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


class Evaluator:
    """Runs duplicate matches in-process (workers=1) or on a reusable spawn-based process pool.

    Agents are built from spec strings inside each worker and cached there, so a pool can serve
    many matches (e.g. one agent against many checkpoints) without reloading. If a worker process
    dies, the match raises `BrokenProcessPool` (the pool is shut down and cannot be reused).
    """

    def __init__(self, workers: int = 1, config: Optional[GameConfig] = None):
        self.config = config if config is not None else load_ruleset()
        self.workers = max(1, int(workers))
        self._runner: Optional[_DealRunner] = None
        self._pool: Optional[ProcessPoolExecutor] = None
        if self.workers == 1:
            self._runner = _DealRunner(self.config)
        else:
            self._pool = ProcessPoolExecutor(self.workers, mp_context=mp.get_context("spawn"),
                                             initializer=_init_worker, initargs=(self.config,))

    def match(self, spec_a: str, spec_b: str, n_deals: int, start_seed: int = 0,
              agent_kwargs_a: Optional[dict] = None, agent_kwargs_b: Optional[dict] = None,
              chunk_size: Optional[int] = None, decks: DeckSpec = "all") -> MatchResult:
        """Deals start_seed, start_seed + 1, ... x 2 seatings; decks="all" rounds n_deals up to a multiple of n²."""
        if n_deals < 1:
            raise ValueError("n_deals must be >= 1")
        if start_seed < 0:
            raise ValueError("start_seed must be >= 0")
        for spec in (spec_a, spec_b):
            path = checkpoint_path(spec)
            if path is not None and not os.path.isfile(path):
                raise FileNotFoundError(f"checkpoint not found: {path} (agent spec {spec!r})")
        decks = check_decks(decks, self.config.n_decks)
        schedule = deal_schedule(start_seed, n_deals, decks, self.config.n_decks)
        kw_a, kw_b = dict(agent_kwargs_a or {}), dict(agent_kwargs_b or {})
        n = len(schedule)
        size = chunk_size or max(1, min(64, math.ceil(n / (4 * self.workers))))
        tasks = [(spec_a, kw_a, spec_b, kw_b, schedule[k:k + size]) for k in range(0, n, size)]
        t0 = time.perf_counter()
        if self._runner is not None:
            parts = [self._runner.run(t) for t in tasks]
        else:
            parts = self._run_on_pool(tasks)
        elapsed = time.perf_counter() - t0
        deals = [pair for part in parts for pair in part]
        return MatchResult(spec_a, spec_b, start_seed, deals, elapsed, kw_a, kw_b, decks, self.config.deck_names)

    def _run_on_pool(self, tasks: List[tuple]) -> List[List[Tuple[GameRecord, GameRecord]]]:
        """Chunk results in task order (identical for any worker count); a dead worker raises."""
        if self._pool is None:
            raise RuntimeError("the Evaluator is closed")
        futures = []
        try:
            # Workers start on demand inside submit(): keep each one to one BLAS/OpenMP thread.
            with _single_threaded_children():
                for t in tasks:
                    futures.append(self._pool.submit(_run_chunk, t))
            return [f.result() for f in futures]
        except BrokenProcessPool as exc:
            self._terminate()
            raise BrokenProcessPool("an evaluation worker process died (crashed, killed or failed to start); "
                                    "the evaluator was shut down") from exc
        except BaseException:
            for f in futures:
                f.cancel()
            raise

    def warmup(self, spec_a: str, spec_b: str, agent_kwargs_a: Optional[dict] = None,
               agent_kwargs_b: Optional[dict] = None) -> None:
        """Untimed mini-match (one-deal chunks on far-away seeds) so that worker start-up and agent
        loading (e.g. importing torch) stay out of later timings."""
        self.match(spec_a, spec_b, 2 * self.workers, WARMUP_SEED, agent_kwargs_a, agent_kwargs_b, chunk_size=1,
                   decks="sampled")

    def close(self) -> None:
        """Wait for the workers to finish and stop them."""
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)

    def _terminate(self) -> None:
        """Stop the workers at once (after an error), dropping queued chunks."""
        pool, self._pool = self._pool, None
        if pool is None:
            return
        processes = list((getattr(pool, "_processes", None) or {}).values())  # cleared by shutdown()
        pool.shutdown(wait=False, cancel_futures=True)
        for proc in processes:
            with suppress(ValueError, OSError):  # already reaped by the executor
                proc.terminate()
        for proc in processes:
            with suppress(ValueError, OSError):
                proc.join(5)

    def __enter__(self) -> "Evaluator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self._terminate()
        self.close()


def duplicate_match(spec_a: str, spec_b: str, n_deals: int, start_seed: int = 0,
                    config: Optional[GameConfig] = None, workers: int = 1,
                    agent_kwargs_a: Optional[dict] = None, agent_kwargs_b: Optional[dict] = None,
                    decks: DeckSpec = "all") -> MatchResult:
    """A vs B over `n_deals` deals (rounded up for decks="all") x 2 seatings. Agents are built from
    specs (see make_agent) per worker."""
    with Evaluator(min(workers, max(1, n_deals)), config) as ev:
        return ev.match(spec_a, spec_b, n_deals, start_seed, agent_kwargs_a, agent_kwargs_b, decks=decks)


__all__ = ["BrokenProcessPool", "DECK_MODES", "DRAW", "Evaluator", "GameRecord", "IllegalAgentActionError",
           "MatchResult", "WDL", "Z95", "agent_seed", "check_decks", "checkpoint_path", "deal_schedule",
           "deals_for_games", "default_workers", "duplicate_match", "outcome_for", "play_game", "round_deals",
           "short_deck_names", "wilson_interval"]
