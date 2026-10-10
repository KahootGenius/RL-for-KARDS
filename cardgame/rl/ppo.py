"""PPO self-play learner (SPEC §8): rollout workers on CPU, network updates on CUDA/MPS/CPU.

Each update: broadcast the latest policy weights (numpy) and the snapshot pool to the rollout
workers, collect at least `batch_steps` learner transitions from finished games, recompute values
with the current value network, compute GAE per seat-trajectory, and run clipped-PPO epochs on
the device. Opponents per game: the latest policy (both seats recorded) with probability
`self_play_prob`, otherwise greedy / random / a frozen snapshot by `opp_weights`. A snapshot is
frozen every `snapshot_every` updates (newest `max_snapshots` kept).

Only finished games enter a batch; games still running continue under the next policy version,
so part of a batch comes from an older policy. The stored behaviour log-probs keep the PPO ratio
a correct importance weight, values are always recomputed by the learner, and the lag is logged.
"""
from __future__ import annotations

import copy
import json
import math
import numbers
import os
import time
import warnings
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..agents.greedy_agent import GreedyAgent
from ..cards import GameConfig, load_ruleset
from ..engine import DRAW, Game
from ..features import ObservationEncoder
from .network import EntityPolicyNet, PolicyValueNet, build_net
from .rollout import OPPONENT_KINDS, RolloutWorkerError, WorkerInit, WorkerPool

TRAIN_SEED_BASE = 1_000_000_000   # training deals never overlap eval.py's default seeds (0..)
TRAIN_SEEDS_PER_RUN = 10_000_000  # deal seeds reserved per --seed value (0 <= seed < 100)
QUICK_EVAL_SEED_BASE = 2_000_000_000

__all__ = ["PPOConfig", "PPOTrainer", "RolloutWorkerError", "select_device", "default_workers"]


def default_workers() -> int:
    return max(1, min(16, (os.cpu_count() or 2) - 2))


def _cuda_error() -> Optional[str]:
    """None if a tiny op runs on cuda:0, else why it does not (e.g. a GPU newer than this torch build)."""
    try:
        if (torch.ones(4, device="cuda") * 2).sum().item() == 8.0:
            return None
        reason = "a test kernel returned a wrong result"
    except Exception as exc:  # noqa: BLE001 - e.g. "no kernel image is available for execution on the device"
        lines = str(exc).strip().splitlines()
        reason = f"{type(exc).__name__}: {lines[0] if lines else ''}"
    try:
        major, minor = torch.cuda.get_device_capability(0)
        reason += (f"; {torch.cuda.get_device_name(0)} is sm_{major}{minor}, this torch {torch.__version__} has "
                   f"kernels for {' '.join(torch.cuda.get_arch_list()) or 'no GPU architecture'}")
    except Exception:  # noqa: BLE001 - the description is best effort
        pass
    return reason


# Two separate commands (no `&&`: Windows PowerShell 5.1 cannot parse it). Uninstall first: pip keeps
# a same-version CPU build.
CUDA_TORCH_INSTALL = ("run `python -m pip uninstall -y torch`, then "
                      "`python -m pip install torch --index-url https://download.pytorch.org/whl/cu128`")


def _no_cuda_reason() -> str:
    """Why torch.cuda.is_available() is False: a CPU-only torch build, or a CUDA build that sees no GPU."""
    if torch.version.cuda is None and getattr(torch.version, "hip", None) is None:
        return f"this PyTorch ({torch.__version__}) is a CPU-only build"
    return (f"this PyTorch ({torch.__version__}) supports GPUs but sees none (`nvidia-smi` should list the GPU; "
            f"under WSL2 the Windows NVIDIA driver provides CUDA, so update that driver)")


def select_device(name: str = "auto") -> torch.device:
    """`auto` = CUDA if this torch build can run kernels on the GPU, else MPS, else CPU (with a warning
    saying why). An explicit name is returned as is."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        reason = _cuda_error()
        if reason is None:
            return torch.device("cuda")
        fallback = "mps" if torch.backends.mps.is_available() else "cpu"
        warnings.warn(f"--device auto: CUDA is available but this PyTorch build cannot run on the GPU ({reason}). "
                      f"Falling back to {fallback}. Install a torch build for this GPU (RTX 50-series / sm_120 needs "
                      f"torch >= 2.7 built for CUDA 12.8): {CUDA_TORCH_INSTALL}. Or pass --device explicitly.",
                      stacklevel=2)
        return torch.device(fallback)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    warnings.warn(f"--device auto: training on the CPU because {_no_cuda_reason()}. For an NVIDIA GPU install a "
                  f"CUDA build of torch (RTX 50-series needs CUDA 12.8): {CUDA_TORCH_INSTALL}. "
                  f"--device cpu silences this.", stacklevel=2)
    return torch.device("cpu")


@dataclass
class PPOConfig:
    run_dir: str = "runs/ppo"
    seed: int = 0
    total_updates: int = 1000
    workers: int = -1             # -1 = auto (cpu_count - 2, at most 16); 0 = in-process (tests)
    envs_per_worker: int = 32
    batch_steps: int = 65536      # learner transitions per update
    epochs: int = 4
    minibatch: int = 8192
    lr: float = 3e-4
    lr_final_frac: float = 0.1    # linear LR decay to lr * lr_final_frac
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03       # stop the epoch loop early if exceeded (0 = off)
    arch: str = "entity"          # entity | mlp
    d_model: int = 128
    id_dim: int = 16
    ctx_dim: int = 256
    pair_dim: int = 128
    pair_mlp_dim: int = 32
    attention_layers: int = 0
    shared_trunk: bool = False
    hidden: tuple = (256, 256)    # mlp only
    self_play_prob: float = 0.5
    opp_weights: dict = field(default_factory=lambda: {"greedy": 0.15, "random": 0.05, "snapshot": 0.30})
    snapshot_every: int = 10
    max_snapshots: int = 30
    snapshots_per_worker: int = 2
    eval_every: int = 10
    eval_deals: int = 256         # quick duplicate eval vs greedy over all deck pairs (2 games per deal)
    device: str = "auto"
    torch_threads: int = 4
    tensorboard: bool = True
    worker_timeout: float = 900.0


def _check_opponents(cfg: PPOConfig) -> None:
    if not 0.0 <= cfg.self_play_prob <= 1.0:
        raise ValueError(f"self_play_prob must be in [0, 1], got {cfg.self_play_prob}")
    w = cfg.opp_weights
    unknown = set(w) - set(OPPONENT_KINDS)
    if unknown:
        raise ValueError(f"unknown opponent kinds {sorted(unknown)} in opp_weights "
                         f"(allowed: {', '.join(OPPONENT_KINDS)})")
    bad = {k: v for k, v in w.items() if not (isinstance(v, numbers.Real) and math.isfinite(v) and v >= 0)}
    if bad:
        raise ValueError(f"opp_weights must be finite and >= 0, got {bad}")
    if cfg.self_play_prob < 1 and w.get("greedy", 0) + w.get("random", 0) <= 0:
        raise ValueError("opp_weights: greedy + random must be > 0 when self_play_prob < 1 (the snapshot share "
                         "falls back to them until the first snapshot exists)")


class PPOTrainer:
    def __init__(self, cfg: PPOConfig, config: Optional[GameConfig] = None):
        if not 0 <= cfg.seed < (QUICK_EVAL_SEED_BASE - TRAIN_SEED_BASE) // TRAIN_SEEDS_PER_RUN:
            raise ValueError("seed must be in [0, 100) so training deals stay disjoint from eval deals")
        for name in ("snapshot_every", "envs_per_worker", "batch_steps", "minibatch", "epochs", "total_updates"):
            if getattr(cfg, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if cfg.eval_every and cfg.eval_deals < 1:
            raise ValueError("eval_deals must be >= 1 when eval_every > 0")
        if cfg.max_snapshots < 0 or cfg.workers < -1:
            raise ValueError("max_snapshots must be >= 0 and workers >= -1")
        if cfg.arch not in ("entity", "mlp"):
            raise ValueError(f"unknown arch {cfg.arch!r}")
        _check_opponents(cfg)
        self.cfg = cfg
        self.config = config if config is not None else load_ruleset()
        torch.set_num_threads(cfg.torch_threads)
        torch.manual_seed(cfg.seed)
        self.device = select_device(cfg.device)
        self.n_workers = default_workers() if cfg.workers == -1 else cfg.workers
        self.encoder = ObservationEncoder(self.config)
        self.n_actions = self.encoder.n_actions
        self.net = self._build_net().to(self.device)
        self.actor = copy.deepcopy(self.net).to("cpu").eval()   # CPU copy for weights export and quick eval
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.update = 0
        self.env_steps = self.learner_steps = self.games = 0
        self.seed_block = TRAIN_SEED_BASE + cfg.seed * TRAIN_SEEDS_PER_RUN
        self.seed_base = self.seed_block
        self.worker_games = [0] * max(1, self.n_workers)
        self.snapshots: list = []    # [(name, path, numpy weights)]
        self.results = defaultdict(lambda: deque(maxlen=1000))  # opponent kind -> learner rewards
        self.greedy_pairs = defaultdict(lambda: deque(maxlen=200))  # (my deck, opp deck) -> rewards vs greedy
        self.best_eval = (-1, -1.0)  # (eligible: min cell >= 0.6, overall win rate)
        self.lr_anchor = (0, cfg.lr)
        self.pool: Optional[WorkerPool] = None

    def _build_net(self):
        cfg = self.cfg
        if cfg.arch == "mlp":
            return PolicyValueNet(self.encoder.dim, self.n_actions, cfg.hidden)
        return EntityPolicyNet(self.encoder.layout(), d_model=cfg.d_model, id_dim=cfg.id_dim, ctx_dim=cfg.ctx_dim,
                               pair_dim=cfg.pair_dim, pair_mlp_dim=cfg.pair_mlp_dim,
                               attention_layers=cfg.attention_layers,
                               shared_trunk=cfg.shared_trunk)

    # ------------------------------------------------------------------ workers
    def _weights(self, net=None) -> dict:
        net = self.actor if net is None else net
        return {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}

    def _ensure_pool(self) -> WorkerPool:
        if self.pool is None:
            cfg = self.cfg
            K = max(1, self.n_workers)
            inits = [WorkerInit(worker_id=w, n_workers=K, seed_base=self.seed_base,
                                seed_limit=self.seed_block + TRAIN_SEEDS_PER_RUN, config=self.config,
                                net_spec=self.net.spec(), envs=cfg.envs_per_worker,
                                self_play_prob=cfg.self_play_prob, opp_weights=dict(cfg.opp_weights),
                                snapshots_per_worker=cfg.snapshots_per_worker,
                                game_counter=self.worker_games[w] if w < len(self.worker_games) else 0)
                     for w in range(K)]
            self.pool = WorkerPool(inits, in_process=self.n_workers == 0, timeout=cfg.worker_timeout)
        return self.pool

    def close(self, force: bool = False) -> None:
        """Stop the rollout workers (`force`: terminate at once, e.g. after Ctrl-C mid-collection)."""
        if self.pool is not None:
            self.pool.close(force=force)
            self.pool = None

    def collect(self) -> tuple:
        """Collect >= batch_steps learner transitions. Returns (trajectories, collection stats)."""
        pool = self._ensure_pool()
        quota = math.ceil(self.cfg.batch_steps / pool.size)
        spec = self.net.spec()
        outs = pool.collect(self.update, self._weights(), [(n, spec, w) for n, _, w in self.snapshots], quota)
        trajs, stats = [], defaultdict(float)
        for out in outs:
            trajs.extend(out["trajs"])
            for key in ("transitions", "env_steps", "games", "decisions"):
                stats[key] += out[key]
            stats["worker_seconds"] = max(stats["worker_seconds"], out["seconds"])
            self.worker_games[out["worker_id"]] = out["game_counter"]
            for kind, my_deck, opp_deck, r in out["results"]:
                self.results[kind].append(r)
                if kind == "greedy":
                    self.greedy_pairs[(my_deck, opp_deck)].append(r)
        self.env_steps += int(stats["env_steps"])
        self.games += int(stats["games"])
        self.learner_steps += int(stats["transitions"])
        return trajs, stats

    # ------------------------------------------------------------------ learning
    def lr_at(self, update: int) -> float:
        cfg = self.cfg
        u0, lr0 = self.lr_anchor
        lr_end = cfg.lr * cfg.lr_final_frac
        frac = min(1.0, (update - u0) / max(1, cfg.total_updates - u0))
        return lr0 + (lr_end - lr0) * frac

    def _batch(self, trajs: list) -> dict:
        """Concatenate trajectories, recompute values with the current net, GAE per trajectory."""
        cfg = self.cfg
        obs = torch.from_numpy(np.concatenate([t["obs"] for t in trajs]).astype(np.float32))
        mask = np.unpackbits(np.concatenate([t["mask"] for t in trajs]), axis=1, count=self.n_actions)
        mask = torch.from_numpy(mask.astype(bool))
        act = torch.from_numpy(np.concatenate([t["act"] for t in trajs]).astype(np.int64))
        logp = torch.from_numpy(np.concatenate([t["logp"] for t in trajs]))
        version = np.concatenate([t["version"] for t in trajs])
        dev = self.device
        values = np.empty(len(act), dtype=np.float32)
        self.net.eval()
        with torch.no_grad():
            for s in range(0, len(act), 16384):
                values[s:s + 16384] = self.net.value(obs[s:s + 16384].to(dev)).float().cpu().numpy()
        adv = np.zeros_like(values)
        start = 0
        for t in trajs:
            T = len(t["act"])
            v = values[start:start + T]
            last, next_v = 0.0, 0.0
            for i in range(T - 1, -1, -1):
                rew = t["reward"] if i == T - 1 else 0.0
                delta = rew + cfg.gamma * next_v - v[i]
                last = delta + cfg.gamma * cfg.gae_lambda * last
                adv[start + i] = last
                next_v = v[i]
            start += T
        ret = adv + values
        lag = self.update - version
        return {"obs": obs, "mask": mask, "act": act, "logp": logp, "adv": torch.from_numpy(adv),
                "ret": torch.from_numpy(ret), "val": torch.from_numpy(values), "lag": lag}

    def learn(self, trajs: list) -> dict:
        cfg = self.cfg
        b = self._batch(trajs)
        dev = self.device
        N = len(b["act"])
        adv = b["adv"]
        adv_norm = ((adv - adv.mean()) / (adv.std() + 1e-8)).to(dev)
        data = {k: b[k].to(dev) for k in ("obs", "mask", "act", "logp", "ret")}
        n_mb = max(1, round(N / cfg.minibatch))
        lr = self.lr_at(self.update)
        for group in self.opt.param_groups:
            group["lr"] = lr
        self.net.train()
        stats = defaultdict(list)
        for epoch in range(cfg.epochs):
            kls = []
            for idx in torch.randperm(N, device=dev).chunk(n_mb):
                logp, ent, v = self.net.evaluate(data["obs"][idx], data["mask"][idx], data["act"][idx])
                log_ratio = logp - data["logp"][idx]
                ratio = log_ratio.exp()
                a = adv_norm[idx]
                pg_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                v_loss = 0.5 * (v - data["ret"][idx]).pow(2).mean()
                ent_mean = ent.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent_mean
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean().item()
                    clipfrac = ((ratio - 1).abs() > cfg.clip).float().mean().item()
                if epoch == 0 and not kls and "kl_first" not in stats:
                    stats["kl_first"].append(kl)  # behaviour vs current policy before any step: lag only
                if not torch.isfinite(loss):
                    stats["skipped_nonfinite"].append(1.0)
                    continue
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                kls.append(kl)
                stats["pg_loss"].append(pg_loss.item())
                stats["v_loss"].append(v_loss.item())
                stats["entropy"].append(ent_mean.item())
                stats["kl"].append(kl)
                stats["clipfrac"].append(clipfrac)
            stats["epochs"] = [epoch + 1]
            if cfg.target_kl and kls and np.mean(kls) > cfg.target_kl:
                break
        self.actor.load_state_dict({k: v.detach().cpu() for k, v in self.net.state_dict().items()})
        ret, val = b["ret"], b["val"]
        explained = 1 - (ret - val).var() / (ret.var() + 1e-8)
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        lag = b["lag"]
        out.update(lr=lr, batch=N, explained_var=float(explained), lag_mean=float(lag.mean()),
                   lag_max=int(lag.max()), stale_frac=float((lag >= 1).mean()))
        return out

    # ------------------------------------------------------------------ pool, eval, checkpoints
    def _add_snapshot(self, path: Path) -> None:
        self.snapshots.append((f"snap_{self.update:05d}", str(Path(path).resolve()), self._weights()))
        if self.cfg.max_snapshots:
            self.snapshots = self.snapshots[-self.cfg.max_snapshots:]

    def quick_eval(self, n_deals: int, deterministic: bool = False) -> dict:
        """Vectorized duplicate games of the current policy vs greedy over all deck pairs (CPU)."""
        n = self.config.n_decks
        agent = GreedyAgent(self.config)
        gen = torch.Generator().manual_seed(12345)
        jobs = []
        for d in range(n_deals):
            decks = divmod(d % (n * n), n)
            for seat in (0, 1):
                g = Game(self.config)
                g.reset(QUICK_EVAL_SEED_BASE + d, decks=decks)
                jobs.append((g, seat))
        X = np.zeros((len(jobs), self.encoder.dim), dtype=np.float32)
        M = np.zeros((len(jobs), self.n_actions), dtype=bool)
        while True:
            live = [(g, s) for g, s in jobs if not g.done]
            if not live:
                break
            mine = [(g, s) for g, s in live if g.current == s]
            for g, s in live:
                if g.current != s:
                    g.step(agent.act(g.observe(g.current), g.legal_actions()))
            if mine:
                k = len(mine)
                X[:k] = 0.0
                for r, (g, s) in enumerate(mine):
                    g.legal_mask(out=M[r])
                    self.encoder.encode_into(g.observe(s), X[r], M[r])
                x = torch.from_numpy(X[:k].astype(np.float16).astype(np.float32))
                with torch.inference_mode():
                    logits = self.actor.policy_logits(x, torch.from_numpy(M[:k]))
                    if deterministic:
                        a = logits.argmax(-1)
                    else:
                        a = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(-1)
                for r, (g, s) in enumerate(mine):
                    g.step(int(a[r]))
        wins = draws = 0
        cells = defaultdict(lambda: [0, 0])  # unordered deck pair -> [wins, games]
        for g, s in jobs:
            w = g.winner()
            key = tuple(sorted(g.deck_ids))
            cells[key][1] += 1
            if w == s:
                wins += 1
                cells[key][0] += 1
            elif w == DRAW:
                draws += 1
        rates = {f"{i}-{j}": c[0] / c[1] for (i, j), c in sorted(cells.items())}
        return {"win_rate": wins / len(jobs), "draw_rate": draws / len(jobs),
                "min_cell": min(rates.values()), "cells": rates}

    def checkpoint(self, path: Path, with_optimizer: bool = False) -> None:
        net_cpu = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        state = {"model": net_cpu, "net": self.net.spec(), "update": self.update, "env_steps": self.env_steps,
                 "learner_steps": self.learner_steps, "games": self.games, "args": asdict(self.cfg)}
        if with_optimizer:
            run_dir = self.run_dir.resolve()
            state.update(optimizer=self.opt.state_dict(), lr=self.lr_at(self.update),
                         snapshots=[(n, os.path.relpath(p, run_dir)) for n, p, _ in self.snapshots],
                         seed_base=self.seed_base, n_workers=max(1, self.n_workers),
                         worker_games=list(self.worker_games), best_eval=list(self.best_eval))
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)

    def resume(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("net") != self.net.spec():
            raise ValueError(f"checkpoint network {state.get('net')} does not match this config "
                             f"{self.net.spec()}; the architecture is fixed by the checkpoint")
        self.net.load_state_dict(state["model"])
        self.actor.load_state_dict(state["model"])
        if "optimizer" in state:
            self.opt.load_state_dict(state["optimizer"])
        self.update = state["update"]
        self.env_steps = state.get("env_steps", 0)
        self.learner_steps = state.get("learner_steps", 0)
        self.games = state.get("games", 0)
        # fresh deal seeds after everything the previous run (with its K workers) may have dealt
        K_old = state.get("n_workers", 1)
        games_old = state.get("worker_games", [self.games])
        self.seed_base = state.get("seed_base", self.seed_block) + K_old * (max(games_old) + 1)
        if not self.seed_block <= self.seed_base < self.seed_block + TRAIN_SEEDS_PER_RUN:
            raise ValueError("resumed seed counter falls outside this run's seed block; pass the run's --seed")
        self.worker_games = [0] * max(1, self.n_workers)
        self.best_eval = tuple(state.get("best_eval", (-1, -1.0)))
        saved_lr = state.get("lr", self.lr_at(self.update))
        saved_base_lr = state.get("args", {}).get("lr", self.cfg.lr)
        if saved_base_lr and self.cfg.lr != saved_base_lr:  # --lr given on resume rescales the schedule
            saved_lr *= self.cfg.lr / saved_base_lr
        self.lr_anchor = (self.update, saved_lr)
        src_dir = Path(path).resolve().parent
        for name, rel in state.get("snapshots", []):
            snap_path = (src_dir / rel).resolve()
            if not snap_path.is_file():
                warnings.warn(f"snapshot {name} not found at {snap_path}; dropped from the pool")
                continue
            ckpt = torch.load(snap_path, map_location="cpu", weights_only=False)
            net = build_net(ckpt["net"])
            net.load_state_dict(ckpt["model"])
            self.snapshots.append((name, str(snap_path), self._weights(net)))
        if self.cfg.max_snapshots:
            self.snapshots = self.snapshots[-self.cfg.max_snapshots:]

    # ------------------------------------------------------------------ main loop
    def _writer(self):
        if not self.cfg.tensorboard:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:  # noqa: BLE001 - optional dependency
            warnings.warn(f"TensorBoard logging disabled ({exc}); install tensorboard or pass --no-tensorboard")
            return None
        return SummaryWriter(str(self.run_dir / "tb"), flush_secs=10)

    def train(self, log=print) -> None:
        cfg = self.cfg
        info = asdict(cfg)
        info.update(device=str(self.device), workers_resolved=self.n_workers, obs_dim=self.encoder.dim,
                    n_actions=self.n_actions, params=sum(p.numel() for p in self.net.parameters()))
        with open(self.run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        writer = self._writer()
        metrics_file = open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8")
        t_start = time.time()
        try:
            while self.update < cfg.total_updates:
                t0 = time.time()
                trajs, cstats = self.collect()
                t1 = time.time()
                stats = self.learn(trajs)
                t2 = time.time()
                self.update += 1
                row = {"update": self.update, "env_steps": self.env_steps, "learner_steps": self.learner_steps,
                       "games": self.games, "collect_s": round(t1 - t0, 2), "learn_s": round(t2 - t1, 2),
                       "learner_steps_per_s": round(cstats["transitions"] / max(1e-9, t1 - t0), 1),
                       "decisions_per_s": round(cstats["decisions"] / max(1e-9, t1 - t0), 1),
                       "elapsed_min": round((t2 - t_start) / 60, 2), "snapshots": len(self.snapshots), **stats}
                for kind, rs in self.results.items():
                    if rs:
                        row[f"train_vs_{kind}"] = round(float(np.mean([r > 0 for r in rs])), 3)
                pairs = [np.mean([r > 0 for r in rs]) for rs in self.greedy_pairs.values() if len(rs) >= 20]
                if pairs:
                    row["train_vs_greedy_min_pair"] = round(float(min(pairs)), 3)
                if self.update % cfg.snapshot_every == 0:
                    path = self.run_dir / f"ckpt_{self.update:05d}.pt"
                    self.checkpoint(path)
                    self._add_snapshot(path)
                if cfg.eval_every and self.update % cfg.eval_every == 0:
                    ev = self.quick_eval(cfg.eval_deals)
                    row.update(eval_greedy_win_rate=round(ev["win_rate"], 3),
                               eval_greedy_draw_rate=round(ev["draw_rate"], 3),
                               eval_greedy_min_cell=round(ev["min_cell"], 3))
                    row.update({f"eval_cell_{k}": round(v, 3) for k, v in ev["cells"].items()})
                    score = (int(ev["min_cell"] >= 0.6), ev["win_rate"])
                    if score > tuple(self.best_eval):
                        self.best_eval = score
                        self.checkpoint(self.run_dir / "best.pt")
                self.checkpoint(self.run_dir / "latest.pt", with_optimizer=True)
                metrics_file.write(json.dumps(row) + "\n")
                metrics_file.flush()
                if writer is not None:
                    for k, v in row.items():
                        if isinstance(v, (int, float)) and k != "update":
                            writer.add_scalar(k, v, self.update)
                    writer.flush()
                log(format_row(row))
        except BaseException:
            self.close(force=True)  # Ctrl-C or a failure: stop busy workers now instead of waiting for them
            raise
        finally:
            metrics_file.close()
            if writer is not None:
                writer.close()
            self.close()


def format_row(row: dict) -> str:
    keys = ["update", "env_steps", "games", "learner_steps_per_s", "entropy", "kl", "clipfrac", "v_loss",
            "explained_var", "lag_mean", "train_vs_random", "train_vs_greedy", "train_vs_snapshot",
            "train_vs_self_first", "eval_greedy_win_rate", "eval_greedy_min_cell", "elapsed_min"]
    parts = []
    for k in keys:
        if k in row:
            v = row[k]
            parts.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
    return " ".join(parts)
