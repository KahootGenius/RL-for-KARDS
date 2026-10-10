"""Rollout workers: self-play games on CPU, in-process or in spawned processes (SPEC §8, §8.1).

A `RolloutWorker` owns a set of games, a CPU copy of the latest policy, the frozen snapshots it
needs, and the scripted opponents (lookahead, random, greedy). Each `collect(quota)` plays until at
least `quota` learner transitions from games that finished during the call are available and
returns them as numpy arrays. Games still running continue at the next call (their transitions
carry the policy version that produced them). `WorkerPool` runs K workers in `spawn`ed processes
(or one in-process when K = 0) and talks to them with numpy-only messages, so CUDA tensors never
cross a process boundary.

Deals: every new game uses `cards.sample_deal(seed, config, random_deck_frac)` (each seat a random
40-card deck with probability `random_deck_frac`, else a fixed deck). Scripted opponents act
through `agents.choose_action(agent, game)` (the lookahead agent simulates on determinized copies
of the game, never on its hidden state).

Every learner transition records the float16-rounded encoding, the packed legal mask
(`encoder.n_actions` wide), the action, its behaviour log-prob, the policy version and `opp_hand`:
the opponent's true hand as counts per card index (uint8). `opp_hand` is the belief-head label and
the privileged critic's input; it never reaches the policy.

Each collect also returns a time breakdown (seconds in this worker): `engine` (legal actions and
stepping), `deal` (deck sampling + reset), `encode` (observe + legal mask + encoding), `infer` (policy
and snapshot forward passes + sampling; in server mode the wait for the learner's reply + sampling),
`scripted` (scripted opponents' decisions) and `other` (trajectory bookkeeping and the loop itself).

Pipes carry encodings as `SparseRows` (row pointers, uint16 column indices, float16 values: exact,
and ~11x smaller than the dense float16 rows, of which ~4% are nonzero): the observations of
inference requests and of the trajectories in "ok" results. The learner densifies them on receipt,
so every trajectory outside the pipe holds dense float16 `obs`.

Server mode (SPEC §8.1, `WorkerPool(server=InferenceServer(...))`, spawned workers only): workers
hold no policy network (by default, `serve_snapshots=False`, they keep CPU snapshot nets and only
the policy is served). For each batch of decisions a worker sends ("infer", net_key, obs SparseRows
(n, dim), packed mask (n, ceil(N/8))) on its pipe and blocks on the reply ("logits", float32 (n, N)
masked logits); net_key is "policy" (the latest policy) or a snapshot name. The learner process
serves while it waits for the results: it drains every request available from all workers, runs one
forward per net_key on the learner device (fp32, inference_mode) and replies. The worker samples
from those logits with its own generator and computes the behaviour log-prob itself, exactly as
with a local net, so a collect matches local mode except for float rounding in the forward
(batches mix rows of several workers). Final results still arrive as ("ok", result).

Seeding: worker w deals games with seeds `seed_base + w + K*n` (n = 0, 1, ...) and draws opponents,
seats and actions from RNGs seeded with `worker_seed(seed_base, w)` (the scripted agents get their
own derived streams), so no two workers ever play the same deal and a run is reproducible for a
fixed (seed, K, envs per worker) on CPU (in server mode up to that rounding, which can flip a
sample only at a near-tie).
"""
from __future__ import annotations

import dataclasses
import os
import random
import signal
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing import connection
from typing import NamedTuple, Optional, Union

import numpy as np
import torch

from ..agents import choose_action
from ..agents.greedy_agent import GreedyAgent
from ..agents.lookahead_agent import LookaheadAgent
from ..agents.random_agent import RandomAgent
from ..cards import GameConfig, sample_deal
from ..engine import DRAW, Game
from ..features import ObservationEncoder
from .network import build_net


OPPONENT_KINDS = ("lookahead", "random", "snapshot", "greedy")  # pool categories (`opp_weights` keys)
SCRIPTED_KINDS = ("lookahead", "random", "greedy")
TIME_KEYS = ("engine", "deal", "encode", "infer", "scripted", "other")
POLICY_KEY = "policy"  # net_key of the latest policy (snapshots are keyed by their names, "snap_NNNNN")


class RolloutWorkerError(RuntimeError):
    pass


class _CloseRequested(Exception):
    """A server-mode worker got "close" while waiting for logits: exit quietly."""


def worker_seed(seed_base: int, worker_id: int) -> int:
    """Deterministic 63-bit seed for a worker's RNG streams (splitmix64 of (seed_base, worker))."""
    z = (seed_base * 0x9E3779B97F4A7C15 + worker_id + 0x632BE59BD9B4E019) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0x7FFFFFFFFFFFFFFF


def hand_counts(hand, n_cards: int) -> np.ndarray:
    """Counts per card index of a hand (list of card indices) as uint8 (a hand holds at most 10 cards)."""
    return np.bincount(np.asarray(hand, dtype=np.intp), minlength=n_cards).astype(np.uint8)


class SparseRows(NamedTuple):
    """Exact sparse form of a float16 (n, dim) array for the pipes: row i's entries with a nonzero bit
    pattern are values[indptr[i]:indptr[i + 1]] at columns indices[indptr[i]:indptr[i + 1]] (bit-exact,
    -0.0 included). The encodings are ~4% nonzero, so this is ~10x smaller than the dense rows."""
    shape: tuple
    indptr: np.ndarray   # int32 (n + 1,)
    indices: np.ndarray  # uint16 (nnz,), int32 when dim > 65536
    values: np.ndarray   # float16 (nnz,)


def to_sparse(x: np.ndarray) -> SparseRows:
    """(n, dim) float16 -> SparseRows (~5 us per 4000-wide row: nonzero test on the uint16 bit patterns)."""
    x = np.ascontiguousarray(x, dtype=np.float16)
    n, dim = x.shape
    flat = np.flatnonzero(x.view(np.uint16))
    rows, cols = np.divmod(flat, dim)
    indptr = np.zeros(n + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(np.bincount(rows, minlength=n))
    return SparseRows((n, dim), indptr, cols.astype(np.uint16 if dim <= 65536 else np.int32), x.reshape(-1)[flat])


def to_dense(obs: Union[np.ndarray, SparseRows], out: Optional[np.ndarray] = None) -> np.ndarray:
    """SparseRows (or a dense array, returned as is unless `out` is given) -> float16 (n, dim); `out` (a
    zeroed float16 array of that shape) receives the rows in place."""
    if isinstance(obs, np.ndarray):
        if out is None:
            return obs
        out[...] = obs
        return out
    if out is None:
        out = np.zeros(obs.shape, dtype=np.float16)
    rows = np.repeat(np.arange(obs.shape[0]), np.diff(obs.indptr))
    out[rows, obs.indices] = obs.values
    return out


def _sparsify_result(result: dict) -> dict:
    """A collect result for the pipe: every trajectory's obs as SparseRows (WorkerPool densifies them)."""
    for t in result["trajs"]:
        t["obs"] = to_sparse(t["obs"])
    return result


@dataclass
class WorkerInit:
    """Everything a worker needs; picklable (no trainer, optimizer or device tensors)."""
    worker_id: int
    n_workers: int                 # K used for deal seeds (1 when in-process)
    seed_base: int
    seed_limit: int                # deal seeds must stay below this
    config: GameConfig
    net_spec: dict
    envs: int
    self_play_prob: float
    opp_weights: dict              # {"lookahead": w, "random": w, "snapshot": w, "greedy": w}
    snapshots_per_worker: int
    random_deck_frac: float = 0.7  # P(a seat gets a generated deck) per new game (cards.sample_deal)
    parent_pid: int = 0            # set by WorkerPool for spawned workers only (0 = in-process: no parent check)
    game_counter: int = 0
    inference_server: bool = False  # set by WorkerPool: ask the learner for logits (needs the pipe; SPEC §8.1)
    serve_snapshots: bool = False   # set by WorkerPool: in server mode, snapshot logits come from the learner too


@dataclass
class _Opponent:
    kind: str                      # self | lookahead | random | greedy | snapshot
    name: str
    agent: object = None


@dataclass
class _Env:
    game: Game
    opp: Optional[_Opponent] = None
    learner_seats: tuple = ()
    traj: dict = field(default_factory=dict)


class RolloutWorker:
    def __init__(self, init: WorkerInit, conn=None):
        self.init = init
        self.config = init.config
        self.encoder = ObservationEncoder(self.config)
        self.n_actions = self.encoder.n_actions
        self.n_cards = self.encoder.n_cards
        # server mode: policy logits (and snapshot logits with serve_snapshots) come from the learner over
        # `conn`; the networks it serves are not built here
        self._conn = conn if init.inference_server else None
        self._remote_snapshots = self._conn is not None and init.serve_snapshots
        self.net = None
        if self._conn is None:
            self.net = build_net(init.net_spec).eval()
            for p in self.net.parameters():
                p.requires_grad_(False)
        self.version = 0
        s = worker_seed(init.seed_base, init.worker_id)
        self.rng = random.Random(s)
        self.torch_gen = torch.Generator().manual_seed(s & 0x7FFFFFFF)
        # each scripted agent gets its own stream: seeding one with `s` would replay self.rng's draws
        self.scripted = {
            "lookahead": _Opponent("lookahead", "lookahead", LookaheadAgent(self.config, seed=worker_seed(s, 2))),
            "random": _Opponent("random", "random", RandomAgent(self.config, seed=worker_seed(s, 1))),
            "greedy": _Opponent("greedy", "greedy", GreedyAgent(self.config)),
        }
        self.snapshots: dict = {}      # name -> net (kept while a running game uses it; empty if the server serves them)
        self.pool_names: list = []     # snapshot names currently in the pool
        self.active: list = []         # this update's snapshot subset
        self.game_counter = init.game_counter
        self.envs = [_Env(Game(self.config)) for _ in range(init.envs)]
        self._x = np.zeros((init.envs, self.encoder.dim), dtype=np.float32)
        self._m = np.zeros((init.envs, self.n_actions), dtype=bool)
        self._started = False
        self._time = dict.fromkeys(TIME_KEYS, 0.0)
        self._infer_calls = self._infer_rows = 0

    # ------------------------------------------------------------------ weights and pool
    def _state_dict(self, weights: dict) -> dict:
        return {k: torch.from_numpy(np.asarray(v)) for k, v in weights.items()}

    def set_weights(self, version: int, weights: Optional[dict]) -> None:
        if weights is not None and self.net is not None:
            self.net.load_state_dict(self._state_dict(weights))
        self.version = version

    def _in_use(self) -> set:
        """Snapshot names some running game plays against."""
        return {e.opp.name for e in self.envs if e.opp is not None and e.opp.kind == "snapshot"}

    def update_pool(self, add: Optional[dict] = None, remove=(), names=None) -> None:
        for name, (spec, weights) in (add or {}).items():
            if self._remote_snapshots:  # the learner holds the snapshot networks
                continue
            net = build_net(spec).eval()
            net.load_state_dict(self._state_dict(weights))
            for p in net.parameters():
                p.requires_grad_(False)
            self.snapshots[name] = net
        if names is not None:
            self.pool_names = list(names)
        removed = set(remove)
        in_use = self._in_use()
        for name in list(self.snapshots):
            if (name in removed or name not in self.pool_names) and name not in in_use:
                del self.snapshots[name]

    # ------------------------------------------------------------------ games
    def _next_seed(self) -> int:
        seed = self.init.seed_base + self.init.worker_id + self.init.n_workers * self.game_counter
        if seed >= self.init.seed_limit:
            raise RolloutWorkerError(f"training deal seeds exhausted ({seed} >= {self.init.seed_limit}): this run's "
                                     f"seed block is used up; latest.pt stays valid, continue with --resume and "
                                     f"another --seed")
        self.game_counter += 1
        return seed

    def _sample_opponent(self) -> tuple:
        rng = self.rng
        if rng.random() < self.init.self_play_prob:
            return _Opponent("self", "self"), (0, 1)
        w = dict(self.init.opp_weights)
        if not self.active:  # no snapshot yet: its share goes to the scripted opponents
            w.pop("snapshot", None)
        kinds = [k for k in OPPONENT_KINDS if w.get(k, 0) > 0]
        total = sum(w[k] for k in kinds)
        r = rng.random() * total
        kind = kinds[-1]
        for k in kinds:
            r -= w[k]
            if r < 0:
                kind = k
                break
        if kind == "snapshot":
            name = self.active[rng.randrange(len(self.active))]
            opp = _Opponent("snapshot", name)
        else:
            opp = self.scripted[kind]
        return opp, (rng.randrange(2),)

    def _reset_env(self, env: _Env) -> None:
        seed = self._next_seed()
        env.game.reset(seed, decks=sample_deal(seed, self.config, self.init.random_deck_frac))
        env.opp, env.learner_seats = self._sample_opponent()
        env.traj = {s: [] for s in env.learner_seats}

    def _logits(self, key: str, X16: np.ndarray, M: np.ndarray) -> torch.Tensor:
        """Masked policy logits (float32, CPU) of network `key` (POLICY_KEY or a snapshot name): from the
        local CPU copy, or in server mode from the learner (one blocking round trip on the pipe)."""
        if self._conn is None or (key != POLICY_KEY and not self._remote_snapshots):
            net = self.net if key == POLICY_KEY else self.snapshots[key]
            return net.policy_logits(torch.from_numpy(X16.astype(np.float32)), torch.from_numpy(M))
        self._conn.send(("infer", key, to_sparse(X16), np.packbits(M, axis=1)))
        reply = self._conn.recv()
        if reply[0] == "close":
            raise _CloseRequested()
        if reply[0] != "logits":
            raise RolloutWorkerError(f"expected logits from the inference server, got {reply[0]!r}")
        return torch.from_numpy(reply[1])

    def _act(self, key: str, idxs: list, actions: list, record: bool) -> None:
        n = len(idxs)
        X, M = self._x[:n], self._m[:n]
        t0 = time.perf_counter()
        X[:] = 0.0
        enc = self.encoder
        for r, i in enumerate(idxs):
            g = self.envs[i].game
            g.legal_mask(out=M[r])
            enc.encode_into(g.observe(g.current), X[r], M[r])
        X16 = X.astype(np.float16)  # the learner trains on these exact (rounded) inputs
        t1 = time.perf_counter()
        with torch.inference_mode():
            logits = self._logits(key, X16, M)
            logp_all = torch.log_softmax(logits, dim=-1)
            a = torch.multinomial(logp_all.exp(), 1, generator=self.torch_gen).squeeze(-1)
            logp = logp_all.gather(-1, a.unsqueeze(-1)).squeeze(-1)
        a, logp = a.numpy(), logp.numpy()
        t2 = time.perf_counter()
        self._time["encode"] += t1 - t0
        self._time["infer"] += t2 - t1
        self._infer_calls += 1
        self._infer_rows += n
        n_cards = self.n_cards
        for r, i in enumerate(idxs):
            actions[i] = int(a[r])
            if record:
                env = self.envs[i]
                g = env.game
                p = g.current
                env.traj[p].append((X16[r].copy(), M[r].copy(), int(a[r]), float(logp[r]), self.version,
                                    hand_counts(g.hands[1 - p], n_cards)))

    def _finish(self, env: _Env, out: list, results: list) -> int:
        """Turn a finished game's learner trajectories into arrays; log one result per learner seat.

        results rows are (opponent kind, learner deck id, opponent deck id, reward); deck id -1 = a
        generated deck. Self-play logs one ("self_first", ...) row from the first player's side."""
        g = env.game
        w = g.winner()
        n = 0
        for seat in env.learner_seats:
            r = 0.0 if w == DRAW else (1.0 if w == seat else -1.0)
            traj = env.traj[seat]
            if traj:
                out.append({
                    "obs": np.stack([t[0] for t in traj]),
                    "mask": np.packbits(np.stack([t[1] for t in traj]), axis=1),
                    "act": np.array([t[2] for t in traj], dtype=np.int16),
                    "logp": np.array([t[3] for t in traj], dtype=np.float32),
                    "version": np.array([t[4] for t in traj], dtype=np.int32),
                    "opp_hand": np.stack([t[5] for t in traj]),
                    "reward": r,
                    "seed": int(g.seed), "seat": seat,  # provenance (tests, debugging); the learner ignores it
                })
                n += len(traj)
            if env.opp.kind != "self":
                results.append((env.opp.kind, g.deck_ids[seat], g.deck_ids[1 - seat], r))
        if env.opp.kind == "self":
            first = g.first_player
            results.append(("self_first", g.deck_ids[first], g.deck_ids[1 - first],
                            0.0 if w == DRAW else (1.0 if w == first else -1.0)))
        return n

    def collect(self, quota: int) -> dict:
        """Play until >= quota learner transitions from games finished during this call exist."""
        t_start = time.perf_counter()
        self._time = tm = dict.fromkeys(TIME_KEYS, 0.0)
        self._infer_calls = self._infer_rows = 0
        k = self.init.snapshots_per_worker
        self.active = self.rng.sample(self.pool_names, min(k, len(self.pool_names))) if self.pool_names else []
        if not self._started:
            t = time.perf_counter()
            for env in self.envs:
                self._reset_env(env)
            tm["deal"] += time.perf_counter() - t
            self._started = True
        out, results = [], []
        collected = env_steps = games = decisions = 0
        envs = self.envs
        actions = [0] * len(envs)
        perf = time.perf_counter
        while collected < quota:
            learner_idx, snap_groups = [], defaultdict(list)
            t0 = perf()
            scripted = 0.0
            for i, env in enumerate(envs):
                g = env.game
                legal = g.legal_actions()
                if len(legal) == 1:  # forced action (e.g. END_TURN with nothing else): no decision to learn from
                    actions[i] = legal[0]
                elif g.current in env.learner_seats:
                    learner_idx.append(i)
                elif env.opp.kind == "snapshot":
                    snap_groups[env.opp.name].append(i)
                else:
                    ts = perf()
                    actions[i] = choose_action(env.opp.agent, g)
                    scripted += perf() - ts
            tm["engine"] += perf() - t0 - scripted
            tm["scripted"] += scripted
            if learner_idx:
                self._act(POLICY_KEY, learner_idx, actions, record=True)
                decisions += len(learner_idx)
            for name, idxs in snap_groups.items():
                self._act(name, idxs, actions, record=False)
                decisions += len(idxs)
            t0 = perf()
            side = 0.0  # finishing and re-dealing, timed separately from stepping
            for i, env in enumerate(envs):
                env.game.step(actions[i])
                env_steps += 1
                if env.game.done:
                    ts = perf()
                    collected += self._finish(env, out, results)
                    games += 1
                    tf = perf()
                    self._reset_env(env)
                    td = perf()
                    tm["other"] += tf - ts
                    tm["deal"] += td - tf
                    side += td - ts
            tm["engine"] += perf() - t0 - side
            if self.init.parent_pid and env_steps % 4096 < len(envs) and os.getppid() != self.init.parent_pid:
                raise RolloutWorkerError("parent process went away")
        self.update_pool(names=self.pool_names)  # drop snapshot nets no running game needs
        seconds = perf() - t_start
        tm["other"] += max(0.0, seconds - sum(tm.values()))
        return {"worker_id": self.init.worker_id, "trajs": out, "results": results, "transitions": collected,
                "env_steps": env_steps, "games": games, "decisions": decisions, "seconds": seconds,
                "game_counter": self.game_counter, "time": dict(tm), "infer_calls": self._infer_calls,
                "infer_rows": self._infer_rows, "snapshots_in_use": sorted(self._in_use())}


# ---------------------------------------------------------------------- processes
_CHILD_ENV = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}


@contextmanager
def _child_environment():
    """Spawned children inherit os.environ: hide CUDA and pin BLAS/OpenMP to one thread."""
    saved = {k: os.environ.get(k) for k in _CHILD_ENV}
    os.environ.update(_CHILD_ENV)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def worker_main(conn, init: WorkerInit) -> None:
    """Entry point of a spawned rollout process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the learner handles Ctrl-C and stops us
    torch.set_num_threads(1)
    try:
        worker = RolloutWorker(init, conn)
        while True:
            if not conn.poll(2.0):
                if init.parent_pid and os.getppid() != init.parent_pid:
                    return
                continue
            msg = conn.recv()
            if msg[0] == "close":
                return
            _, version, weights, pool_delta, quota = msg
            del msg
            worker.set_weights(version, weights)
            worker.update_pool(**pool_delta)
            del weights, pool_delta  # the nets hold copies: free the numpy weights (~4.5 MB per net) while collecting
            conn.send(("ok", _sparsify_result(worker.collect(quota))))
    except (EOFError, BrokenPipeError, KeyboardInterrupt, _CloseRequested):
        return
    except BaseException:  # noqa: BLE001 - report everything to the learner
        try:
            conn.send(("error", init.worker_id, traceback.format_exc()))
        except Exception:  # noqa: BLE001
            pass


class InferenceServer:
    """Batched policy inference for server-mode workers (SPEC §8.1), run inside the learner process.

    `policy` is the learner's own network (switched to eval mode for each collection; its weights are
    the ones the workers would otherwise receive). Snapshot networks are built on the device on first
    use from the numpy weights registered by `begin` and cached by name; `retain` drops the names no
    worker can still ask for. With `serve_snapshots=False` only the policy is served and the workers
    run their snapshots on the CPU. `serve` answers a group of requests (one drain) with one fp32
    forward per network under inference_mode. `stats` covers the current collection: drains,
    requests, forwards, rows, forward seconds (device sync included), the snapshot share of forwards
    and forward seconds, busy seconds (unpacking the requests and the forwards; the pipe traffic,
    replies included, is WorkerPool.io_s) and rows per network key."""

    def __init__(self, policy: torch.nn.Module, device, n_actions: int, serve_snapshots: bool = False):
        self.policy = policy
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.serve_snapshots = serve_snapshots
        self.weights: dict = {}  # snapshot name -> (spec, numpy weights)
        self.nets: dict = {}     # snapshot name -> device network (built on first request)
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stats = {"forward_s": 0.0, "busy_s": 0.0, "drains": 0, "requests": 0, "forwards": 0, "rows": 0,
                      "snapshot_forwards": 0, "snapshot_forward_s": 0.0}
        self.rows_by_key = defaultdict(int)

    def begin(self, pool: list) -> None:
        """Start of a collection. `pool` = [(name, spec, numpy weights)]: the snapshots workers may sample."""
        self.policy.eval()
        if self.serve_snapshots:
            for name, spec, weights in pool:
                self.weights.setdefault(name, (spec, weights))
        self.reset_stats()

    def retain(self, keep) -> None:
        """Forget every snapshot not in `keep` (the pool plus the snapshots running games still play)."""
        keep = set(keep)
        for name in list(self.weights):
            if name not in keep:
                del self.weights[name]
                self.nets.pop(name, None)

    def net(self, key: str) -> torch.nn.Module:
        if key == POLICY_KEY:
            return self.policy
        net = self.nets.get(key)
        if net is None:
            if key not in self.weights:
                raise RolloutWorkerError(f"inference request for unknown network {key!r} "
                                         f"(known: {POLICY_KEY}, {', '.join(self.weights) or 'no snapshots'})")
            spec, weights = self.weights[key]
            net = build_net(spec)
            net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in weights.items()})
            net = net.to(self.device).eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self.nets[key] = net
        return net

    def serve(self, requests: list) -> list:
        """requests = [(key, obs SparseRows or float16 (n, dim), packed mask uint8 (n, ceil(N/8)))] -> the
        masked logits of each request (float32 numpy (n, N)), with one forward per distinct key."""
        t0 = time.perf_counter()
        groups = defaultdict(list)
        for j, (key, _, _) in enumerate(requests):
            groups[key].append(j)
        out = [None] * len(requests)
        st, dev = self.stats, self.device
        with torch.inference_mode():
            for key, js in groups.items():
                net = self.net(key)
                sizes = [requests[j][1].shape[0] for j in js]
                obs = np.zeros((sum(sizes), requests[js[0]][1].shape[1]), dtype=np.float16)
                start = 0
                for j, n in zip(js, sizes):
                    to_dense(requests[j][1], out=obs[start:start + n])
                    start += n
                mask = np.unpackbits(np.concatenate([requests[j][2] for j in js]), axis=1,
                                     count=self.n_actions).astype(bool)
                tf = time.perf_counter()
                x = torch.from_numpy(obs).to(dev).float()
                logits = net.policy_logits(x, torch.from_numpy(mask).to(dev)).float().cpu().numpy()
                dt = time.perf_counter() - tf
                st["forward_s"] += dt
                st["forwards"] += 1
                if key != POLICY_KEY:
                    st["snapshot_forwards"] += 1
                    st["snapshot_forward_s"] += dt
                st["rows"] += len(obs)
                self.rows_by_key[key] += len(obs)
                start = 0
                for j, n in zip(js, sizes):
                    out[j] = logits[start:start + n]
                    start += n
        st["drains"] += 1
        st["requests"] += len(requests)
        st["busy_s"] += time.perf_counter() - t0
        return out


class WorkerPool:
    """K rollout processes (or one in-process worker when K == 0).

    With an `InferenceServer` (spawned workers only; ignored in-process) the workers run in server mode:
    they get no policy weights (nor snapshot weights when the server serves snapshots), and `collect`
    serves their inference requests until every result is in. `io_s` is the learner's time in pipe
    traffic during the last collect: receiving requests and results, densifying the results' obs and
    sending replies (the server's busy_s covers unpacking requests and the forwards)."""

    def __init__(self, inits: list, in_process: bool, timeout: float = 900.0,
                 server: Optional[InferenceServer] = None):
        self.timeout = timeout
        self.in_process = in_process
        self.server = server = None if in_process else server
        self.local = None
        self.procs, self.conns = [], []
        self.synced: set = set()  # snapshot names every worker already holds
        self.io_s = 0.0
        if in_process:  # runs in this process: there is no parent to watch, and it uses local nets
            self.local = RolloutWorker(dataclasses.replace(inits[0], parent_pid=0, inference_server=False))
            return
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        try:
            with _child_environment():
                for init in inits:
                    parent_conn, child_conn = ctx.Pipe()
                    init = dataclasses.replace(init, parent_pid=os.getpid(), inference_server=server is not None,
                                               serve_snapshots=server is None or server.serve_snapshots)
                    proc = ctx.Process(target=worker_main, name=f"rollout-{init.worker_id}", daemon=True,
                                       args=(child_conn, init))
                    proc.start()
                    child_conn.close()
                    self.procs.append(proc)
                    self.conns.append(parent_conn)
        except BaseException:
            self.close(force=True)
            raise

    @property
    def size(self) -> int:
        return 1 if self.in_process else len(self.procs)

    def collect(self, version: int, weights: Optional[dict], pool: list, quota: int) -> list:
        """Broadcast weights + pool changes, collect from every worker. `pool` = [(name, spec, weights)].

        In server mode `weights` is not sent (None is fine), the server gets the pool and answers
        inference requests while the workers collect, and the workers only learn the pool's names
        (unless the server leaves snapshots to them: then they get new snapshots' weights as usual).

        Any failure (a worker error, death or timeout, a serving error, or Ctrl-C while waiting) stops
        every worker before the exception propagates, so nothing keeps collecting in the background."""
        names = [p[0] for p in pool]
        server = self.server
        if server is None or not server.serve_snapshots:
            add = {n: (spec, w) for n, spec, w in pool if n not in self.synced}
        else:  # the server holds the snapshot networks
            add = {}
        if server is not None:  # ... and the policy
            weights = None
        removed = [n for n in self.synced if n not in names]
        delta = {"add": add, "remove": removed, "names": names}
        self.io_s = 0.0
        if self.in_process:
            self.local.set_weights(version, weights)
            self.local.update_pool(**delta)
            outs = [self.local.collect(quota)]
        else:
            try:
                if server is not None:
                    server.begin(pool)
                msg = ("collect", version, weights, delta, quota)
                for i, conn in enumerate(self.conns):
                    try:
                        conn.send(msg)
                    except OSError:  # BrokenPipeError & co: the worker died since the last update
                        raise self._died(i) from None
                outs = self._gather()
                if server is not None:  # keep what the next collect may ask for: pool + running snapshot games
                    server.retain(set(names).union(*(o.get("snapshots_in_use", ()) for o in outs)))
            except BaseException:
                self.close(force=True)  # no-op when the failure path already closed the pool
                raise
        self.synced = set(names)
        return outs

    def _died(self, i: int) -> RolloutWorkerError:
        """Worker i is gone (broken pipe, EOF or sentinel): stop all workers, report its exit code."""
        proc = self.procs[i]
        proc.join(timeout=1.0)  # reap it so the exit code is known
        code = proc.exitcode
        if code is None:
            state = "broke its pipe (process still running)"
        else:
            state = f"died (exit code {code}{', SIGKILL: out of memory?' if code == -9 else ''})"
        self.close(force=True)
        return RolloutWorkerError(f"rollout worker {i} (pid {proc.pid}) {state}")

    def _serve(self, requests: list) -> None:
        """Answer inference requests [(worker index, ("infer", key, obs, packed mask))] in one batch."""
        if self.server is None:
            self.close(force=True)
            raise RolloutWorkerError(f"rollout worker {requests[0][0]} sent an inference request, but this pool "
                                     f"has no inference server")
        replies = self.server.serve([msg[1:] for _, msg in requests])
        t0 = time.perf_counter()
        for (i, _), logits in zip(requests, replies):
            try:
                self.conns[i].send(("logits", logits))
            except OSError:  # the worker died after asking
                raise self._died(i) from None
        self.io_s += time.perf_counter() - t0

    def _gather(self) -> list:
        """Wait for every worker's reply, serving inference requests meanwhile (server mode); a worker
        error, death or the timeout stops all workers and raises."""
        pending = set(range(len(self.conns)))
        outs = [None] * len(self.conns)
        deadline = time.time() + self.timeout
        while pending:
            conns = {self.conns[i]: i for i in pending}
            sentinels = {self.procs[i].sentinel: i for i in pending}
            ready = connection.wait(list(conns) + list(sentinels), timeout=max(0.0, deadline - time.time()))
            if not ready:
                self.close(force=True)
                raise RolloutWorkerError(f"rollout workers {sorted(pending)} timed out after "
                                         f"{self.timeout:.0f}s")
            requests = []
            for r in ready:  # replies first, so a worker that reported an error and exited keeps its traceback
                if r in conns:
                    i = conns[r]
                    t0 = time.perf_counter()
                    try:
                        msg = r.recv()
                    except (EOFError, OSError):
                        raise self._died(i) from None
                    if msg[0] == "infer":  # blocked until answered: at most one request per worker
                        requests.append((i, msg))
                        self.io_s += time.perf_counter() - t0
                        continue
                    if msg[0] == "error":
                        self.close(force=True)
                        raise RolloutWorkerError(f"rollout worker {msg[1]} failed:\n{msg[2]}")
                    for t in msg[1]["trajs"]:
                        t["obs"] = to_dense(t["obs"])
                    outs[i] = msg[1]
                    pending.discard(i)
                    self.io_s += time.perf_counter() - t0
            if requests:
                self._serve(requests)
            for r in ready:  # a dead worker with an unread message is handled by recv() on the next pass
                if r in sentinels and sentinels[r] in pending:
                    i = sentinels[r]
                    try:
                        unread = self.conns[i].poll()
                    except OSError:  # e.g. a broken pipe on Windows
                        unread = False
                    if not unread:
                        raise self._died(i)
        return outs

    def close(self, force: bool = False, grace: float = 2.0) -> None:
        """Stop the workers. Idle workers exit on "close"; whatever is still alive when ONE shared
        `grace` deadline passes (busy collecting, blocked sending a large result) is terminated.
        `force` (after Ctrl-C or a failure) skips the polite request: the pipes may hold half-sent
        messages and busy workers would not read it anyway."""
        if not force:
            for conn in self.conns:
                try:
                    conn.send(("close",))
                except Exception:  # noqa: BLE001 - a dead worker's pipe
                    pass
            deadline = time.time() + grace
            for proc in self.procs:
                proc.join(timeout=max(0.0, deadline - time.time()))
        for proc in self.procs:
            if proc.is_alive():
                proc.terminate()
        deadline = time.time() + 2.0
        for proc in self.procs:
            proc.join(timeout=max(0.0, deadline - time.time()))
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)
        for conn in self.conns:
            conn.close()
        self.procs, self.conns = [], []
