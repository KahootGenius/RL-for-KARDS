"""Rollout workers: self-play games on CPU, in-process or in spawned processes (SPEC §8).

A `RolloutWorker` owns a set of games, a CPU copy of the latest policy, the frozen snapshots it
needs, and the scripted opponents. Each `collect(quota)` plays until at least `quota` learner
transitions from games that finished during the call are available and returns them as numpy
arrays. Games still running continue at the next call (their transitions carry the policy version
that produced them). `WorkerPool` runs K workers in `spawn`ed processes (or one in-process when
K = 0) and talks to them with numpy-only messages, so CUDA tensors never cross a process boundary.

Seeding: worker w deals games with seeds `seed_base + w + K*n` (n = 0, 1, ...) and draws opponents,
seats and actions from RNGs seeded with `agent_seed(seed_base, w)`, so no two workers ever play
the same deal and a run is reproducible for a fixed (seed, K, envs per worker) on CPU.
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
from typing import Optional

import numpy as np
import torch

from ..agents.greedy_agent import GreedyAgent
from ..agents.random_agent import RandomAgent
from ..cards import GameConfig
from ..engine import DRAW, Game
from ..features import ObservationEncoder
from .network import build_net


OPPONENT_KINDS = ("greedy", "random", "snapshot")  # pool categories (`opp_weights` keys)


class RolloutWorkerError(RuntimeError):
    pass


def worker_seed(seed_base: int, worker_id: int) -> int:
    """Deterministic 63-bit seed for a worker's RNG streams (splitmix64 of (seed_base, worker))."""
    z = (seed_base * 0x9E3779B97F4A7C15 + worker_id + 0x632BE59BD9B4E019) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0x7FFFFFFFFFFFFFFF


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
    opp_weights: dict              # {"greedy": w, "random": w, "snapshot": w}
    snapshots_per_worker: int
    parent_pid: int = 0            # set by WorkerPool for spawned workers only (0 = in-process: no parent check)
    game_counter: int = 0


@dataclass
class _Opponent:
    kind: str                      # self | greedy | random | snapshot
    name: str
    agent: object = None


@dataclass
class _Env:
    game: Game
    opp: Optional[_Opponent] = None
    learner_seats: tuple = ()
    traj: dict = field(default_factory=dict)


class RolloutWorker:
    def __init__(self, init: WorkerInit):
        self.init = init
        self.config = init.config
        self.encoder = ObservationEncoder(self.config)
        self.n_actions = self.encoder.n_actions
        self.net = build_net(init.net_spec).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.version = 0
        s = worker_seed(init.seed_base, init.worker_id)
        self.rng = random.Random(s)
        self.torch_gen = torch.Generator().manual_seed(s & 0x7FFFFFFF)
        # the random opponent gets its own stream: seeding it with `s` would replay self.rng's draws
        self.scripted = {"greedy": _Opponent("greedy", "greedy", GreedyAgent(self.config)),
                         "random": _Opponent("random", "random", RandomAgent(self.config, seed=worker_seed(s, 1)))}
        self.snapshots: dict = {}      # name -> net (kept while any running game uses it)
        self.pool_names: list = []     # snapshot names currently in the pool
        self.active: list = []         # this update's snapshot subset
        self.game_counter = init.game_counter
        self.envs = [_Env(Game(self.config)) for _ in range(init.envs)]
        self._x = np.zeros((init.envs, self.encoder.dim), dtype=np.float32)
        self._m = np.zeros((init.envs, self.n_actions), dtype=bool)
        self._started = False

    # ------------------------------------------------------------------ weights and pool
    def _state_dict(self, weights: dict) -> dict:
        return {k: torch.from_numpy(np.asarray(v)) for k, v in weights.items()}

    def set_weights(self, version: int, weights: Optional[dict]) -> None:
        if weights is not None:
            self.net.load_state_dict(self._state_dict(weights))
        self.version = version

    def update_pool(self, add: Optional[dict] = None, remove=(), names=None) -> None:
        for name, (spec, weights) in (add or {}).items():
            net = build_net(spec).eval()
            net.load_state_dict(self._state_dict(weights))
            for p in net.parameters():
                p.requires_grad_(False)
            self.snapshots[name] = net
        if names is not None:
            self.pool_names = list(names)
        removed = set(remove)
        in_use = {e.opp.name for e in self.envs if e.opp is not None and e.opp.kind == "snapshot"}
        for name in list(self.snapshots):
            if (name in removed or name not in self.pool_names) and name not in in_use:
                del self.snapshots[name]

    # ------------------------------------------------------------------ games
    def _next_seed(self) -> int:
        seed = self.init.seed_base + self.init.worker_id + self.init.n_workers * self.game_counter
        if seed >= self.init.seed_limit:
            raise RolloutWorkerError(f"training deal seeds exhausted ({seed} >= {self.init.seed_limit})")
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
        env.game.reset(self._next_seed())
        env.opp, env.learner_seats = self._sample_opponent()
        env.traj = {s: [] for s in env.learner_seats}

    def _act(self, net, idxs: list, actions: list, record: bool) -> None:
        n = len(idxs)
        X, M = self._x[:n], self._m[:n]
        X[:] = 0.0
        enc = self.encoder
        for r, i in enumerate(idxs):
            g = self.envs[i].game
            g.legal_mask(out=M[r])
            enc.encode_into(g.observe(g.current), X[r], M[r])
        X16 = X.astype(np.float16)  # the learner trains on these exact (rounded) inputs
        with torch.inference_mode():
            logits = net.policy_logits(torch.from_numpy(X16.astype(np.float32)), torch.from_numpy(M))
            logp_all = torch.log_softmax(logits, dim=-1)
            a = torch.multinomial(logp_all.exp(), 1, generator=self.torch_gen).squeeze(-1)
            logp = logp_all.gather(-1, a.unsqueeze(-1)).squeeze(-1)
        a, logp = a.numpy(), logp.numpy()
        for r, i in enumerate(idxs):
            actions[i] = int(a[r])
            if record:
                env = self.envs[i]
                env.traj[env.game.current].append((X16[r].copy(), M[r].copy(), int(a[r]), float(logp[r]),
                                                   self.version))

    def _finish(self, env: _Env, out: list, results: list) -> int:
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
                    "reward": r,
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
        t0 = time.time()
        k = self.init.snapshots_per_worker
        self.active = self.rng.sample(self.pool_names, min(k, len(self.pool_names))) if self.pool_names else []
        if not self._started:
            for env in self.envs:
                self._reset_env(env)
            self._started = True
        out, results = [], []
        collected = env_steps = games = decisions = 0
        envs = self.envs
        actions = [0] * len(envs)
        while collected < quota:
            learner_idx, snap_groups = [], defaultdict(list)
            for i, env in enumerate(envs):
                g = env.game
                legal = g.legal_actions()
                if len(legal) == 1:  # forced END_TURN: no decision to learn from
                    actions[i] = legal[0]
                elif g.current in env.learner_seats:
                    learner_idx.append(i)
                elif env.opp.kind == "snapshot":
                    snap_groups[env.opp.name].append(i)
                else:
                    actions[i] = env.opp.agent.act(g.observe(g.current), legal)
            if learner_idx:
                self._act(self.net, learner_idx, actions, record=True)
                decisions += len(learner_idx)
            for name, idxs in snap_groups.items():
                self._act(self.snapshots[name], idxs, actions, record=False)
                decisions += len(idxs)
            for i, env in enumerate(envs):
                env.game.step(actions[i])
                env_steps += 1
                if env.game.done:
                    collected += self._finish(env, out, results)
                    games += 1
                    self._reset_env(env)
            if self.init.parent_pid and env_steps % 4096 < len(envs) and os.getppid() != self.init.parent_pid:
                raise RolloutWorkerError("parent process went away")
        self.update_pool(names=self.pool_names)  # drop snapshot nets no running game needs
        return {"worker_id": self.init.worker_id, "trajs": out, "results": results, "transitions": collected,
                "env_steps": env_steps, "games": games, "decisions": decisions, "seconds": time.time() - t0,
                "game_counter": self.game_counter}


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
        worker = RolloutWorker(init)
        while True:
            if not conn.poll(2.0):
                if init.parent_pid and os.getppid() != init.parent_pid:
                    return
                continue
            msg = conn.recv()
            if msg[0] == "close":
                return
            _, version, weights, pool_delta, quota = msg
            worker.set_weights(version, weights)
            worker.update_pool(**pool_delta)
            conn.send(("ok", worker.collect(quota)))
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        return
    except BaseException:  # noqa: BLE001 - report everything to the learner
        try:
            conn.send(("error", init.worker_id, traceback.format_exc()))
        except Exception:  # noqa: BLE001
            pass


class WorkerPool:
    """K rollout processes (or one in-process worker when K == 0)."""

    def __init__(self, inits: list, in_process: bool, timeout: float = 900.0):
        self.timeout = timeout
        self.in_process = in_process
        self.local = None
        self.procs, self.conns = [], []
        self.synced: set = set()  # snapshot names every worker already holds
        if in_process:  # runs in this process: there is no parent to watch
            self.local = RolloutWorker(dataclasses.replace(inits[0], parent_pid=0))
            return
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        try:
            with _child_environment():
                for init in inits:
                    parent_conn, child_conn = ctx.Pipe()
                    proc = ctx.Process(target=worker_main, name=f"rollout-{init.worker_id}", daemon=True,
                                       args=(child_conn, dataclasses.replace(init, parent_pid=os.getpid())))
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

    def collect(self, version: int, weights: dict, pool: list, quota: int) -> list:
        """Broadcast weights + pool changes, collect from every worker. `pool` = [(name, spec, weights)].

        Any failure (a worker error, death or timeout, or Ctrl-C while waiting) stops every worker
        before the exception propagates, so nothing keeps collecting in the background."""
        names = [p[0] for p in pool]
        add = {n: (spec, w) for n, spec, w in pool if n not in self.synced}
        removed = [n for n in self.synced if n not in names]
        delta = {"add": add, "remove": removed, "names": names}
        if self.in_process:
            self.local.set_weights(version, weights)
            self.local.update_pool(**delta)
            outs = [self.local.collect(quota)]
        else:
            try:
                msg = ("collect", version, weights, delta, quota)
                for i, conn in enumerate(self.conns):
                    try:
                        conn.send(msg)
                    except OSError:  # BrokenPipeError & co: the worker died since the last update
                        raise self._died(i) from None
                outs = self._gather()
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

    def _gather(self) -> list:
        """Wait for every worker's reply; a worker error, death or the timeout stops all workers and raises."""
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
            for r in ready:  # replies first, so a worker that reported an error and exited keeps its traceback
                if r in conns:
                    i = conns[r]
                    try:
                        msg = r.recv()
                    except (EOFError, OSError):
                        raise self._died(i) from None
                    if msg[0] == "error":
                        self.close(force=True)
                        raise RolloutWorkerError(f"rollout worker {msg[1]} failed:\n{msg[2]}")
                    outs[i] = msg[1]
                    pending.discard(i)
            for r in ready:  # a dead worker with an unread message is handled by recv() on the next pass
                if r in sentinels and sentinels[r] in pending and not self.conns[sentinels[r]].poll():
                    raise self._died(sentinels[r])
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
