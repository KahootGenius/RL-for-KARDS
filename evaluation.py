"""Duplicate-game evaluation (SPEC section 8): play_game, duplicate_match, MatchResult, Wilson intervals.

Every deal seed is played twice with identical shuffles and coin flip: A in seat 0 vs B in seat 1,
then with the seats swapped. Agent randomness is re-seeded per game and per *seat* (`agent_seed`),
so a mirror match of identical agents is exactly balanced, and results never depend on the
number of worker processes.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, List, NamedTuple, Optional, Sequence, Tuple

from .agents import Agent, make_agent
from .cards import GameConfig, load_ruleset
from .engine import DRAW, Game, IllegalActionError

Z95 = 1.959963984540054  # two-sided 95% normal quantile
WARMUP_SEED = 3_000_000_000  # far from eval (0..) and training (1e9..) deal seeds
_MASK64 = (1 << 64) - 1
_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "NUMEXPR_NUM_THREADS")


class IllegalAgentActionError(IllegalActionError):
    """An agent returned an action that is not in the legal action list it was given."""


class GameRecord(NamedTuple):
    seed: int
    winner: int        # 0 / 1, or DRAW (-1)
    first_player: int
    rounds: int
    steps: int


def agent_seed(seed: int, seat: int) -> int:
    """Seed for the agent in `seat` of the game dealt from `seed` (splitmix64 mix, 63-bit, non-negative)."""
    z = (seed * 2 + seat + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) >> 1


def play_game(config: Optional[GameConfig], agents: Sequence[Agent], seed: int,
              agent_seeds: Optional[Sequence[int]] = None, game: Optional[Game] = None) -> GameRecord:
    """Play one full game with `agents[p]` in seat p; agents see only observe(p) + legal_actions().

    `agent_seeds` defaults to (agent_seed(seed, 0), agent_seed(seed, 1)). Pass `game` (built from the
    same config) to reuse it. Raises IllegalAgentActionError if an agent picks an illegal action.
    """
    if len(agents) != 2:
        raise ValueError(f"expected 2 agents, got {len(agents)}")
    if game is None:
        game = Game(config)
    game.reset(seed)
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
                f"{action!r} ({desc}) in game seed={seed}, round {game.round}, step {steps}; "
                f"legal: {[game.describe(a) for a in legal]}")
        game.step(int(action))
        steps += 1
    return GameRecord(seed, game.winner(), game.first_player, game.round, steps)


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

    def _rate(self, k: int) -> float:
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


@dataclass
class MatchResult:
    """A duplicate match from A's point of view. `deals[i] = (game with A in seat 0, game with A in seat 1)`.

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
    overall: WDL = field(init=False)
    by_seat: Tuple[WDL, WDL] = field(init=False)
    first: WDL = field(init=False)   # games where A moved first
    second: WDL = field(init=False)

    def __post_init__(self) -> None:
        self.overall, self.by_seat, self.first, self.second = WDL(), (WDL(), WDL()), WDL(), WDL()
        for pair in self.deals:
            for a_seat, rec in enumerate(pair):
                out = outcome_for(rec, a_seat)
                self.overall.add(out)
                self.by_seat[a_seat].add(out)
                (self.first if rec.first_player == a_seat else self.second).add(out)

    # -- headline numbers (overall)
    @property
    def n_deals(self) -> int:
        return len(self.deals)

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
        d = {"agent_a": self.agent_a, "agent_b": self.agent_b, "kwargs_a": dict(self.kwargs_a),
             "kwargs_b": dict(self.kwargs_b), "start_seed": self.start_seed, "n_deals": self.n_deals,
             **self.overall.to_dict(),
             "by_seat": {"0": self.by_seat[0].to_dict(), "1": self.by_seat[1].to_dict()},
             "by_turn_order": {"first": self.first.to_dict(), "second": self.second.to_dict()},
             "deals": self.deal_summary(), "avg_rounds": self.avg_rounds, "avg_steps": self.avg_steps,
             "elapsed_s": self.elapsed, "games_per_s": self.games_per_s}
        if include_games:
            d["games_played"] = [{"a_seat": a_seat, **rec._asdict()}
                                 for pair in self.deals for a_seat, rec in enumerate(pair)]
        return d

    def format(self) -> str:
        last = self.start_seed + self.n_deals - 1
        lines = [f"A={self.agent_a} vs B={self.agent_b}: {self.games} games = {self.n_deals} deals x 2 seatings "
                 f"(seeds {self.start_seed}..{last})",
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
        spec_a, kw_a, spec_b, kw_b, seeds = task
        # Distinct roles keep A and B separate instances even when spec_a == spec_b.
        a, b = self.agent("a", spec_a, kw_a), self.agent("b", spec_b, kw_b)
        cfg, game = self.config, self.game
        return [(play_game(cfg, (a, b), s, game=game), play_game(cfg, (b, a), s, game=game)) for s in seeds]


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
    many matches (e.g. one agent against many checkpoints) without reloading.
    """

    def __init__(self, workers: int = 1, config: Optional[GameConfig] = None):
        self.config = config if config is not None else load_ruleset()
        self.workers = max(1, int(workers))
        self._runner: Optional[_DealRunner] = None
        self._pool = None
        if self.workers == 1:
            self._runner = _DealRunner(self.config)
        else:
            with _single_threaded_children():
                self._pool = mp.get_context("spawn").Pool(
                    self.workers, initializer=_init_worker, initargs=(self.config,))

    def match(self, spec_a: str, spec_b: str, n_deals: int, start_seed: int = 0,
              agent_kwargs_a: Optional[dict] = None, agent_kwargs_b: Optional[dict] = None,
              chunk_size: Optional[int] = None) -> MatchResult:
        """2 * n_deals games over deal seeds [start_seed, start_seed + n_deals)."""
        if n_deals < 1:
            raise ValueError("n_deals must be >= 1")
        for spec in (spec_a, spec_b):
            path = checkpoint_path(spec)
            if path is not None and not os.path.isfile(path):
                raise FileNotFoundError(f"checkpoint not found: {path} (agent spec {spec!r})")
        kw_a, kw_b = dict(agent_kwargs_a or {}), dict(agent_kwargs_b or {})
        size = chunk_size or max(1, min(64, math.ceil(n_deals / (4 * self.workers))))
        end = start_seed + n_deals
        tasks = [(spec_a, kw_a, spec_b, kw_b, range(s, min(s + size, end))) for s in range(start_seed, end, size)]
        t0 = time.perf_counter()
        if self._pool is None:
            parts = [self._runner.run(t) for t in tasks]
        else:
            parts = list(self._pool.imap(_run_chunk, tasks))  # ordered: identical for any worker count
        elapsed = time.perf_counter() - t0
        deals = [pair for part in parts for pair in part]
        return MatchResult(spec_a, spec_b, start_seed, deals, elapsed, kw_a, kw_b)

    def warmup(self, spec_a: str, spec_b: str, agent_kwargs_a: Optional[dict] = None,
               agent_kwargs_b: Optional[dict] = None) -> None:
        """Untimed mini-match (one-deal chunks on far-away seeds) so that worker start-up and agent
        loading (e.g. importing torch) stay out of later timings."""
        self.match(spec_a, spec_b, 2 * self.workers, WARMUP_SEED, agent_kwargs_a, agent_kwargs_b, chunk_size=1)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> "Evaluator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None and self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
        self.close()


def duplicate_match(spec_a: str, spec_b: str, n_deals: int, start_seed: int = 0,
                    config: Optional[GameConfig] = None, workers: int = 1,
                    agent_kwargs_a: Optional[dict] = None, agent_kwargs_b: Optional[dict] = None) -> MatchResult:
    """A vs B over `n_deals` deals x 2 seatings. Agents are built from specs (see make_agent) per worker."""
    with Evaluator(min(workers, max(1, n_deals)), config) as ev:
        return ev.match(spec_a, spec_b, n_deals, start_seed, agent_kwargs_a, agent_kwargs_b)


__all__ = ["DRAW", "Evaluator", "GameRecord", "IllegalAgentActionError", "MatchResult", "WDL", "Z95",
           "agent_seed", "checkpoint_path", "default_workers", "duplicate_match", "outcome_for", "play_game",
           "wilson_interval"]
