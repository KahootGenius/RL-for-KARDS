"""PPO self-play learner (SPEC §8): rollout workers on CPU, network updates on CUDA/MPS/CPU.

Each update: broadcast the latest policy weights (numpy) and the snapshot pool to the rollout
workers, collect at least `batch_steps` learner transitions from finished games, recompute values
with the current value network (with the opponent's true hand when the critic is privileged),
compute GAE per seat-trajectory, and run clipped-PPO epochs on the device. Opponents per game: the
latest policy (both seats recorded) with probability `self_play_prob`, otherwise lookahead / random
/ greedy / a frozen snapshot by `opp_weights`. A snapshot is frozen every `snapshot_every` updates
(newest `max_snapshots` kept). Every game is dealt with `cards.sample_deal(seed, config,
random_deck_frac)`.

Loss per PPO minibatch (default 8192 rows):
    PPO clip + vf_coef * value + belief_coef * BCE(belief_logits, opp_hand > 0) - ent_coef * entropy
The minibatch is sorted by present-token count and processed in micro-batches of `micro_batch`
rows with gradient accumulation (each micro-batch adds its share sum / minibatch to the gradient),
so the step equals the full-minibatch step while the activation memory stays bounded; the network
packs each row's present tokens, so sorted micro-batches carry little padding. On CUDA the forward
passes run under bf16 autocast (`amp`); losses are computed in fp32.

Only finished games enter a batch; games still running continue under the next policy version,
so part of a batch comes from an older policy. The stored behaviour log-probs keep the PPO ratio
a correct importance weight, values are always recomputed by the learner, and the lag is logged.

Batched inference (`inference_server`, SPEC §8.1): "on" (or "auto" with a CUDA learner device)
makes the learner process serve the workers' policy forward passes on its device while they
collect, instead of each worker running a CPU copy. Snapshot opponents stay on the workers' CPUs
by default: their batches are tiny (a few rows per worker and snapshot), so serving them would
fragment the device work into many launch-bound forwards (`serve_snapshots=True` serves them too).
In-process collection (workers = 0) always uses local nets. Server statistics (`server_*`) and the
learner's pipe time (`pool_io_s`) are logged per update.

best.pt (SPEC 8): each quick eval (vs lookahead, 16-32 games per fixed cell) may replace a
provisional best.pt and nominates candidates (cand_<update>.pt: the top `select_top` by the best.pt
rule plus the top `select_top` by P(every cell >= 0.6)). When the run reaches `total_updates`, a
selection pass re-evaluates the candidates with `select_games` games per deck mode on the selection
seeds (500,000,000+) and copies the best by the rule to best.pt (selection.json has the details).

Deal seeds: run seed s owns [1e9 + s * 1e7, 1e9 + (s + 1) * 1e7). Before each collect the learner
checks that no worker can run out of seeds during it (SeedBlockExhausted: training stops cleanly at
the update boundary; resume with another --seed to continue in that seed's block).
"""
from __future__ import annotations

import contextlib
import copy
import json
import math
import numbers
import os
import shutil
import time
import warnings
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as tF

from ..agents import choose_action
from ..agents.lookahead_agent import LookaheadAgent
from ..cards import GameConfig, deck_rng, generate_deck, load_ruleset
from ..engine import DRAW, Game
from ..features import ENCODER_VERSION, ObservationEncoder
from .agent import CheckpointError, spec_encoder_version, spec_fingerprint
from .network import NET_KINDS, STAGE2_KINDS, PolicyValueNet, PooledPolicyNet, TransformerPolicyNet, build_net
from .rollout import (OPPONENT_KINDS, SCRIPTED_KINDS, TIME_KEYS, InferenceServer, RolloutWorkerError, WorkerInit,
                      WorkerPool, worker_seed)

SELECTION_SEED_BASE = 500_000_000  # final best.pt selection pass (SPEC 8): eval 0+, selection 5e8+, training 1e9+
TRAIN_SEED_BASE = 1_000_000_000   # training deals never overlap eval.py's default seeds (0..)
TRAIN_SEEDS_PER_RUN = 10_000_000  # deal seeds reserved per --seed value (0 <= seed < 100)
QUICK_EVAL_SEED_BASE = 2_000_000_000
BEST_MIN_CELL = 0.6               # best.pt: eligible when every fixed-pair cell vs lookahead is >= this
EVAL_SAMPLING_SEED = 12345
INFERENCE_SERVER_MODES = ("auto", "on", "off")
PLANNING_TRANSITIONS_PER_GAME = 35  # seed-budget projection: ~ learner transitions per game of a trained policy
                                    # (Stage 2 desktop: 65,536 / ~1,740 games per update)

__all__ = ["PPOConfig", "PPOTrainer", "RolloutWorkerError", "CheckpointError", "SeedBlockExhausted", "select_device",
           "default_workers", "check_resumable", "best_score", "eligible_probability", "resolve_inference_server",
           "INFERENCE_SERVER_MODES", "SELECTION_SEED_BASE"]


class SeedBlockExhausted(RuntimeError):
    """This run's block of training deal seeds (TRAIN_SEEDS_PER_RUN per --seed) cannot cover the next update,
    or is used up on --resume. latest.pt stays valid; continue with --resume and another --seed."""


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


def resolve_inference_server(mode: str, device, n_workers: int) -> bool:
    """Whether the rollout workers use the learner's batched inference (SPEC §8.1): mode "on", or "auto"
    with a CUDA learner device; never with in-process collection (n_workers = 0), which keeps local nets."""
    if mode not in INFERENCE_SERVER_MODES:
        raise ValueError(f"inference_server must be one of {', '.join(INFERENCE_SERVER_MODES)}, got {mode!r}")
    if n_workers == 0:
        return False
    return mode == "on" or (mode == "auto" and torch.device(device).type == "cuda")


@dataclass
class PPOConfig:
    run_dir: str = "runs/ppo"
    seed: int = 0
    total_updates: int = 1000
    workers: int = -1             # -1 = auto (cpu_count - 2, at most 16); 0 = in-process (tests)
    envs_per_worker: int = 32
    batch_steps: int = 65536      # learner transitions per update
    epochs: int = 4
    minibatch: int = 8192         # rows per optimizer step
    micro_batch: int = 1024       # rows per forward/backward (gradient accumulation inside a minibatch)
    lr: float = 3e-4
    lr_final_frac: float = 0.1    # linear LR decay to lr * lr_final_frac
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    belief_coef: float = 0.25     # weight of the belief BCE (ignored with belief=False)
    max_grad_norm: float = 0.5
    target_kl: float = 0.03       # stop the epoch loop early if exceeded (0 = off)
    arch: str = "transformer"     # transformer | pooled | mlp
    d_model: int = 128
    layers: int = 3               # transformer only
    heads: int = 4                # transformer only
    ff: int = 256                 # transformer only
    id_dim: int = 16
    ctx_dim: int = 256            # pooled only
    pair_dim: int = 32            # width of the pointer scorer's additive pair MLP (token nets)
    hidden: tuple = (256, 256)    # mlp only
    belief: bool = True           # belief head (opponent-hand prediction) + its loss
    privileged_critic: bool = False
    shared_trunk: bool = False
    self_play_prob: float = 0.5
    opp_weights: dict = field(default_factory=lambda: {"lookahead": 0.15, "random": 0.05, "snapshot": 0.30})
    random_deck_frac: float = 0.7  # P(a seat gets a generated deck) per training deal
    snapshot_every: int = 10
    max_snapshots: int = 30
    snapshots_per_worker: int = 2
    eval_every: int = 10
    eval_deals: int = 256         # quick duplicate eval vs lookahead: half random decks, half fixed pairs
    device: str = "auto"
    torch_threads: int = 4
    amp: bool = True              # bf16 autocast for the learner's forward passes (CUDA only)
    tensorboard: bool = True
    worker_timeout: float = 900.0
    inference_server: str = "auto"  # auto (on iff the learner device is CUDA) | on | off (SPEC §8.1)
    serve_snapshots: bool = False   # with the inference server: also serve snapshot opponents (default: worker CPUs)
    select_top: int = 3           # best.pt candidates kept per ranking (quick-eval rule; P(every cell >= 0.6))
    select_games: int = 2000      # final selection pass: games per deck mode (random, all) per candidate; 0 = off


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
    if cfg.self_play_prob < 1 and sum(w.get(k, 0) for k in SCRIPTED_KINDS) <= 0:
        raise ValueError(f"opp_weights: {' + '.join(SCRIPTED_KINDS)} must be > 0 when self_play_prob < 1 (the "
                         f"snapshot share falls back to them until the first snapshot exists)")


def check_resumable(state, encoder: Optional[ObservationEncoder] = None) -> None:
    """Raise CheckpointError unless `state` is a Stage 3 train.py checkpoint (for --resume).

    Stage 1/2 checkpoints (no network kind, the Stage 2 "entity" network, or another encoder version)
    and, given `encoder`, checkpoints trained on another card pool or deck list are refused."""
    if not isinstance(state, dict) or "model" not in state or "net" not in state:
        raise CheckpointError("not a train.py checkpoint (expected latest.pt of a training run)")
    spec = state.get("net")
    kind = spec.get("kind") if isinstance(spec, dict) else None
    if kind is None:
        raise CheckpointError("this is a Stage 1 checkpoint (network spec without a kind). Stage 3 changed the "
                              "rules, the encoding and the network, so it cannot be resumed: start a new run")
    version = spec_encoder_version(spec)
    if kind in STAGE2_KINDS or version != ENCODER_VERSION:
        raise CheckpointError(f"this is a Stage 2 checkpoint (network kind {kind!r}, encoder version "
                              f"{version if version is not None else 'none'}; this code is Stage 3, encoder "
                              f"version {ENCODER_VERSION}). Stage 3 changed the rules, the encoding and the "
                              f"network, so it cannot be resumed: start a new run (python train.py --run-dir "
                              f"runs/<new name>)")
    if kind not in NET_KINDS:
        raise CheckpointError(f"unknown network kind {kind!r} (expected one of {NET_KINDS})")
    if encoder is not None and spec_fingerprint(spec) != encoder.fingerprint:
        raise CheckpointError(f"the checkpoint was trained on another card pool, deck list or encoder layout "
                              f"(fingerprint {spec_fingerprint(spec)} != {encoder.fingerprint}); a run cannot be "
                              f"resumed on different content or inputs: start a new run")


def best_score(ev: dict) -> tuple:
    """Ranking key of an evaluation for best.pt (larger is better, compared as a tuple):
    (1, random-deck win rate) when every fixed-pair cell is >= BEST_MIN_CELL, else (0, overall win rate).
    So best.pt is the highest random-deck win rate among eligible evals, or the highest overall win
    rate while no eval is eligible. `min_cell` is None when some cell was not measured: not eligible."""
    min_cell = ev.get("min_cell")
    if min_cell is not None and min_cell >= BEST_MIN_CELL:
        rnd = ev.get("random_win_rate")
        return (1, float(ev["win_rate"] if rnd is None else rnd))
    return (0, float(ev["win_rate"]))


def _beta_tail(wins: int, games: int, x: float) -> float:
    """P(p >= x) for p ~ Beta(wins + 1, games - wins + 1) (uniform prior) = P(Binomial(games + 1, x) <= wins)."""
    n = games + 1
    return min(1.0, sum(math.comb(n, k) * x ** k * (1.0 - x) ** (n - k) for k in range(wins + 1)))


def eligible_probability(ev: dict) -> float:
    """Posterior probability (uniform prior, cells independent) that every fixed-pair cell's true win rate
    is >= BEST_MIN_CELL, from an eval's per-cell [wins, games] (`cell_counts`); 0 when a cell is missing.
    A quick eval has only 16-32 games per cell, so this, not the point estimate, says how plausible
    eligibility is; it ranks the second half of the best.pt candidates (SPEC 8)."""
    counts = ev.get("cell_counts")
    if ev.get("min_cell") is None or not counts:
        return 0.0
    p = 1.0
    for wins, games in counts.values():
        p *= _beta_tail(int(wins), int(games), BEST_MIN_CELL)
    return p


class PPOTrainer:
    def __init__(self, cfg: PPOConfig, config: Optional[GameConfig] = None):
        if not 0 <= cfg.seed < (QUICK_EVAL_SEED_BASE - TRAIN_SEED_BASE) // TRAIN_SEEDS_PER_RUN:
            raise ValueError("seed must be in [0, 100) so training deals stay disjoint from eval deals")
        for name in ("snapshot_every", "envs_per_worker", "batch_steps", "minibatch", "micro_batch", "epochs",
                     "total_updates"):
            if getattr(cfg, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if cfg.eval_every and cfg.eval_deals < 1:
            raise ValueError("eval_deals must be >= 1 when eval_every > 0")
        if cfg.select_top < 1 or cfg.select_games < 0:
            raise ValueError("select_top must be >= 1 and select_games >= 0")
        if cfg.max_snapshots < 0 or cfg.workers < -1:
            raise ValueError("max_snapshots must be >= 0 and workers >= -1")
        if cfg.arch in STAGE2_KINDS:
            raise ValueError(f"arch {cfg.arch!r} is the Stage 2 network; Stage 3 uses transformer (default), "
                             f"pooled or mlp")
        if cfg.arch not in NET_KINDS:
            raise ValueError(f"unknown arch {cfg.arch!r} (expected one of {NET_KINDS})")
        if cfg.shared_trunk and cfg.privileged_critic:
            raise ValueError("shared_trunk cannot be combined with privileged_critic (the actor would see the "
                             "opponent's hand through the shared trunk)")
        if cfg.shared_trunk and cfg.arch == "mlp":
            raise ValueError("shared_trunk needs a token network (arch transformer or pooled)")
        if not 0.0 <= cfg.random_deck_frac <= 1.0:
            raise ValueError(f"random_deck_frac must be in [0, 1], got {cfg.random_deck_frac}")
        if not (math.isfinite(cfg.belief_coef) and cfg.belief_coef >= 0):
            raise ValueError(f"belief_coef must be finite and >= 0, got {cfg.belief_coef}")
        if cfg.inference_server not in INFERENCE_SERVER_MODES:
            raise ValueError(f"inference_server must be one of {', '.join(INFERENCE_SERVER_MODES)}, "
                             f"got {cfg.inference_server!r}")
        _check_opponents(cfg)
        self.cfg = cfg
        self.config = config if config is not None else load_ruleset()
        torch.set_num_threads(cfg.torch_threads)
        torch.manual_seed(cfg.seed)
        self.device = select_device(cfg.device)
        self.amp = bool(cfg.amp) and self.device.type == "cuda"
        if self.amp and not torch.cuda.is_bf16_supported():
            warnings.warn("--amp: this GPU has no bf16 support; training in fp32 (pass --no-amp to silence)")
            self.amp = False
        self.n_workers = default_workers() if cfg.workers == -1 else cfg.workers
        self.inference_server = resolve_inference_server(cfg.inference_server, self.device, self.n_workers)
        self.encoder = ObservationEncoder(self.config)
        self.n_actions = self.encoder.n_actions
        self.n_cards = self.encoder.n_cards
        lay = self.encoder.layout()
        self._present_slice = (lay["offsets"]["present"], lay["offsets"]["present"] + lay["T"])
        self.net = self._build_net(lay).to(self.device)
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
        self.lookahead_pairs = defaultdict(lambda: deque(maxlen=200))  # (my deck, opp deck) -> rewards, fixed decks
        self.best_eval = (-1, -1.0)  # best_score() of the eval that wrote best.pt
        self.candidates: list = []   # best.pt candidates for the selection pass: dicts with update, path (cand_*.pt)
        self.last_selection: Optional[dict] = None
        self._games_per_collect = 0  # most games one worker dealt in the last collect (seed-budget check)
        self.lr_anchor = (0, cfg.lr)
        self.pool: Optional[WorkerPool] = None

    def _build_net(self, layout: Optional[dict] = None):
        cfg = self.cfg
        layout = self.encoder.layout() if layout is None else layout
        if cfg.arch == "mlp":
            return PolicyValueNet.from_layout(layout, hidden=cfg.hidden, belief=cfg.belief,
                                              privileged_critic=cfg.privileged_critic)
        if cfg.arch == "pooled":
            return PooledPolicyNet(layout, d_model=cfg.d_model, ctx_dim=cfg.ctx_dim, id_dim=cfg.id_dim,
                                   belief=cfg.belief, privileged_critic=cfg.privileged_critic,
                                   shared_trunk=cfg.shared_trunk, pair_dim=cfg.pair_dim)
        return TransformerPolicyNet(layout, d_model=cfg.d_model, layers=cfg.layers, heads=cfg.heads, ff=cfg.ff,
                                    id_dim=cfg.id_dim, belief=cfg.belief, privileged_critic=cfg.privileged_critic,
                                    shared_trunk=cfg.shared_trunk, pair_dim=cfg.pair_dim)

    def _autocast(self):
        """bf16 autocast on the learner device when `amp` is active (only CUDA turns it on; the device type
        is taken from self.device so tests can exercise the same code path under MPS / CPU autocast)."""
        return torch.autocast(self.device.type, dtype=torch.bfloat16) if self.amp else contextlib.nullcontext()

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
                                snapshots_per_worker=cfg.snapshots_per_worker, random_deck_frac=cfg.random_deck_frac,
                                game_counter=self.worker_games[w] if w < len(self.worker_games) else 0)
                     for w in range(K)]
            server = (InferenceServer(self.net, self.device, self.n_actions, serve_snapshots=cfg.serve_snapshots)
                      if self.inference_server else None)
            self.pool = WorkerPool(inits, in_process=self.n_workers == 0, timeout=cfg.worker_timeout, server=server)
        return self.pool

    def close(self, force: bool = False) -> None:
        """Stop the rollout workers (`force`: terminate at once, e.g. after Ctrl-C mid-collection)."""
        if self.pool is not None:
            self.pool.close(force=force)
            self.pool = None

    def _record_result(self, kind: str, my_deck: int, opp_deck: int, r: float) -> None:
        self.results[kind].append(r)
        if kind == "lookahead":
            if my_deck >= 0 and opp_deck >= 0:
                self.lookahead_pairs[(my_deck, opp_deck)].append(r)
            elif my_deck < 0 and opp_deck < 0:
                self.results["lookahead_randdeck"].append(r)

    # ------------------------------------------------------------------ deal seeds
    def seed_budget(self) -> dict:
        """This run's block of training deal seeds: --seed, [start, end), seeds used so far (up to the next
        free seed of the busiest worker) and remaining (SPEC 8: 1e9 + seed * 1e7, TRAIN_SEEDS_PER_RUN each)."""
        K = max(1, self.n_workers)
        end = self.seed_block + TRAIN_SEEDS_PER_RUN
        nxt = self.seed_base + K * max(self.worker_games)
        return {"seed": self.cfg.seed, "start": self.seed_block, "end": end, "used": nxt - self.seed_block,
                "remaining": max(0, end - nxt)}

    def _games_left_per_worker(self) -> int:
        """Deals the most constrained worker can still make: worker w deals seed_base + w + K * n below the
        block end (rollout.RolloutWorker._next_seed)."""
        K = max(1, self.n_workers)
        limit = self.seed_block + TRAIN_SEEDS_PER_RUN
        left = []
        for w in range(K):
            first_free = self.seed_base + w + K * (self.worker_games[w] if w < len(self.worker_games) else 0)
            left.append((limit - 1 - first_free) // K + 1 if first_free < limit else 0)
        return min(left)

    def _exhausted_message(self, detail: str) -> str:
        b = self.seed_budget()
        return (f"the training deal seeds of --seed {self.cfg.seed} are used up ({b['used']:,} of "
                f"{TRAIN_SEEDS_PER_RUN:,} dealt; {detail}). latest.pt holds update {self.update}: continue with "
                f"--resume {self.run_dir / 'latest.pt'} --seed <another seed in [0, 100) that no other run uses>")

    def _check_seed_budget(self) -> None:
        """Before a collect: stop cleanly with SeedBlockExhausted when a worker could run out of deal seeds
        during it (it would otherwise fail mid-collect). The need is the worker's running games plus twice
        the most deals one worker made in the last collect (kept in latest.pt, so it survives --resume)."""
        need = self.cfg.envs_per_worker + 2 * self._games_per_collect
        left = self._games_left_per_worker()
        if left < need:
            raise SeedBlockExhausted(self._exhausted_message(
                f"the next update may need up to {need:,} deals per worker and {left:,} are left"))

    def collect(self) -> tuple:
        """Collect >= batch_steps learner transitions. Returns (trajectories, collection stats).

        stats["time"] sums the workers' time breakdown (rollout.TIME_KEYS, seconds); in server mode
        stats["server"] holds the inference server's statistics for this collect. Raises
        SeedBlockExhausted (before collecting) when the run's deal seeds could run out during it."""
        self._check_seed_budget()
        pool = self._ensure_pool()
        quota = math.ceil(self.cfg.batch_steps / pool.size)
        spec = self.net.spec()
        weights = None if pool.server is not None else self._weights()  # the server forwards self.net itself
        outs = pool.collect(self.update, weights, [(n, spec, w) for n, _, w in self.snapshots], quota)
        trajs, stats = [], defaultdict(float)
        times = dict.fromkeys(TIME_KEYS, 0.0)
        dealt = 0
        for out in outs:
            trajs.extend(out["trajs"])
            for key in ("transitions", "env_steps", "games", "decisions", "infer_calls", "infer_rows"):
                stats[key] += out[key]
            stats["worker_seconds"] = max(stats["worker_seconds"], out["seconds"])
            for k, v in out["time"].items():
                times[k] = times.get(k, 0.0) + v
            dealt = max(dealt, out["game_counter"] - self.worker_games[out["worker_id"]])
            self.worker_games[out["worker_id"]] = out["game_counter"]
            for kind, my_deck, opp_deck, r in out["results"]:
                self._record_result(kind, my_deck, opp_deck, r)
        self._games_per_collect = dealt
        stats = dict(stats)
        stats["time"] = times
        stats["n_workers"] = len(outs)
        stats["io_s"] = pool.io_s
        if pool.server is not None:
            stats["server"] = dict(pool.server.stats, rows_by_key=dict(pool.server.rows_by_key))
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

    def present_counts(self, obs: torch.Tensor) -> torch.Tensor:
        """Present-token count per row, read from the token-present block of the encoding (layout)."""
        a, b = self._present_slice
        return (obs[:, a:b] > 0.5).sum(1)

    def _values(self, obs16: torch.Tensor, opp_hand: torch.Tensor, n_present: torch.Tensor) -> np.ndarray:
        """Values of every row with the current value network (rows sorted by token count, in chunks)."""
        dev = self.device
        N = len(obs16)
        values = np.empty(N, dtype=np.float32)
        order = torch.argsort(n_present, stable=True)
        chunk = 4 * self.cfg.micro_batch  # no activations are kept under no_grad
        priv_on = getattr(self.net, "privileged_critic", False)
        self.net.eval()
        with torch.no_grad(), self._autocast():
            for s in range(0, N, chunk):
                rows = order[s:s + chunk]
                x = obs16[rows].to(dev).float()
                priv = opp_hand[rows].to(dev).float() if priv_on else None
                values[rows.numpy()] = self.net.value(x, priv).float().cpu().numpy()
        return values

    def _batch(self, trajs: list) -> dict:
        """Concatenate trajectories, recompute values with the current net, GAE per trajectory.

        Observations stay float16 (exact: the workers rounded them) until a micro-batch is used."""
        cfg = self.cfg
        obs = torch.from_numpy(np.ascontiguousarray(np.concatenate([t["obs"] for t in trajs]), dtype=np.float16))
        mask = np.unpackbits(np.concatenate([t["mask"] for t in trajs]), axis=1, count=self.n_actions)
        mask = torch.from_numpy(mask.astype(bool))
        act = torch.from_numpy(np.concatenate([t["act"] for t in trajs]).astype(np.int64))
        logp = torch.from_numpy(np.concatenate([t["logp"] for t in trajs]).astype(np.float32))
        version = np.concatenate([t["version"] for t in trajs])
        opp_hand = torch.from_numpy(np.concatenate([t["opp_hand"] for t in trajs]).astype(np.uint8))
        n_present = self.present_counts(obs)
        values = self._values(obs, opp_hand, n_present)
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
        return {"obs": obs, "mask": mask, "act": act, "logp": logp, "opp_hand": opp_hand, "n_present": n_present,
                "adv": torch.from_numpy(adv), "ret": torch.from_numpy(ret), "val": torch.from_numpy(values),
                "lag": lag}

    def _device_data(self, b: dict) -> dict:
        """The tensors the minibatch loop reads, on the learner device (advantages normalised)."""
        dev = self.device
        adv = b["adv"]
        data = {k: b[k].to(dev) for k in ("obs", "mask", "act", "logp", "ret", "opp_hand")}
        data["adv"] = ((adv - adv.mean()) / (adv.std() + 1e-8)).to(dev)
        data["n_present"] = b["n_present"]  # sorting happens on the CPU
        return data

    def _minibatch_backward(self, data: dict, idx: torch.Tensor, belief_stats: bool = False) -> dict:
        """Zero the gradients, then accumulate the gradient of the minibatch `idx` (CPU row indices) in
        micro-batches of `micro_batch` rows (rows sorted by present-token count). Every loss term is a
        minibatch mean, so each micro-batch backpropagates (sum of its rows' terms) / len(idx): the
        accumulated gradient equals the full-minibatch gradient. Returns the minibatch's statistics;
        stats["finite"] is False if the loss was not finite (the caller then skips the step).

        `belief_stats` adds belief accuracy and R-precision (fraction of the opponent's k distinct hand
        cards among the belief head's top k, a ranking/AP proxy robust to the label imbalance)."""
        cfg, dev = self.cfg, self.device
        B = len(idx)
        order = idx[torch.argsort(data["n_present"][idx], stable=True)]
        priv_on = getattr(self.net, "privileged_critic", False)
        use_belief = getattr(self.net, "belief_head", None) is not None
        self.opt.zero_grad(set_to_none=True)
        zero = torch.zeros((), device=dev)
        acc = defaultdict(lambda: zero.clone())
        for s in range(0, B, cfg.micro_batch):
            rows = order[s:s + cfg.micro_batch].to(dev)
            x = data["obs"][rows].float()
            opp = data["opp_hand"][rows]
            priv = opp.float() if priv_on else None
            with self._autocast():
                logp, ent, v, belief = self.net.evaluate(x, data["mask"][rows], data["act"][rows], priv=priv)
            logp, ent, v = logp.float(), ent.float(), v.float()
            log_ratio = logp - data["logp"][rows]
            ratio = log_ratio.exp()
            a = data["adv"][rows]
            pg = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).sum()
            vl = 0.5 * (v - data["ret"][rows]).pow(2).sum()
            es = ent.sum()
            loss = pg + cfg.vf_coef * vl - cfg.ent_coef * es
            if use_belief:
                y = opp > 0
                bl = belief.float()
                bce = tF.binary_cross_entropy_with_logits(bl, y.float(), reduction="none").mean(1).sum()
                loss = loss + cfg.belief_coef * bce
            (loss / B).backward()
            with torch.no_grad():
                acc["loss"] += loss.detach()
                acc["pg_loss"] += pg.detach()
                acc["v_loss"] += vl.detach()
                acc["entropy"] += es.detach()
                acc["kl"] += ((ratio - 1) - log_ratio).sum()
                acc["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().sum()
                if use_belief:
                    acc["belief_loss"] += bce.detach()
                    if belief_stats:
                        bl = bl.detach()
                        k = y.sum(1)
                        acc["belief_acc"] += ((bl > 0) == y).float().mean(1).sum()
                        acc["belief_pos_rate"] += y.float().mean(1).sum()
                        rank = torch.argsort(torch.argsort(bl, dim=1, descending=True), dim=1)  # 0 = top prediction
                        hits = ((rank < k.unsqueeze(1)) & y).sum(1).float()
                        has = k > 0
                        acc["belief_rprec"] += torch.where(has, hits / k.clamp(min=1).float(), zero).sum()
                        acc["belief_rows"] += has.float().sum()
        vals = dict(zip(acc.keys(), torch.stack(list(acc.values())).tolist()))  # one device sync per minibatch
        out = {"finite": math.isfinite(vals.pop("loss"))}
        rows_with_hand = vals.pop("belief_rows", 0.0)
        for k, v in vals.items():
            if k == "belief_rprec":
                out[k] = v / rows_with_hand if rows_with_hand else float("nan")
            else:
                out[k] = v / B
        return out

    def learn(self, trajs: list) -> dict:
        cfg = self.cfg
        t0 = time.perf_counter()
        b = self._batch(trajs)
        t_batch = time.perf_counter() - t0
        data = self._device_data(b)
        N = len(b["act"])
        n_mb = max(1, round(N / cfg.minibatch))
        lr = self.lr_at(self.update)
        for group in self.opt.param_groups:
            group["lr"] = lr
        self.net.train()
        stats = defaultdict(list)
        params = [p for p in self.net.parameters() if p.requires_grad]
        for epoch in range(cfg.epochs):
            kls = []
            for idx in torch.randperm(N).chunk(n_mb):
                mb = self._minibatch_backward(data, idx, belief_stats=epoch == 0)
                if "kl_first" not in stats:
                    stats["kl_first"].append(mb["kl"])  # behaviour vs current policy before any step: lag only
                if not mb.pop("finite"):
                    self.opt.zero_grad(set_to_none=True)
                    stats["skipped_nonfinite"].append(1.0)
                    continue
                grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                self.opt.step()
                kls.append(mb["kl"])
                stats["grad_norm"].append(float(grad_norm))
                for k, v in mb.items():
                    if k.startswith("belief_"):  # SPEC 8: the belief metrics are epoch-0 data
                        if k == "belief_loss":
                            stats["belief_loss_all"].append(v)  # every epoch (the fitted loss), for reference
                        if epoch == 0 and math.isfinite(v):
                            stats[k].append(v)  # fresh data, before the update trained on it
                    else:
                        stats[k].append(v)
            stats["epochs"] = [epoch + 1]
            if cfg.target_kl and kls and np.mean(kls) > cfg.target_kl:
                break
        self.opt.zero_grad(set_to_none=True)
        self.actor.load_state_dict({k: v.detach().cpu() for k, v in self.net.state_dict().items()})
        ret, val = b["ret"], b["val"]
        explained = 1 - (ret - val).var() / (ret.var() + 1e-8)
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        lag = b["lag"]
        pres = b["n_present"].float()
        out.update(lr=lr, batch=N, explained_var=float(explained), lag_mean=float(lag.mean()),
                   lag_max=int(lag.max()), stale_frac=float((lag >= 1).mean()), present_tokens=float(pres.mean()),
                   batch_prep_s=round(t_batch, 3))
        return out

    # ------------------------------------------------------------------ pool, eval, checkpoints
    def _add_snapshot(self, path: Path) -> None:
        self.snapshots.append((f"snap_{self.update:05d}", str(Path(path).resolve()), self._weights()))
        if self.cfg.max_snapshots:
            self.snapshots = self.snapshots[-self.cfg.max_snapshots:]

    def quick_eval(self, n_deals: int, deterministic: bool = False) -> dict:
        """Duplicate games of the current policy (CPU actor, vectorised over all games) vs lookahead.

        Deal d (seed QUICK_EVAL_SEED_BASE + d) is played twice with the same seed and seat decks, the
        policy in seat 0 and then in seat 1. The first n_deals // 2 deals use generated decks
        (`generate_deck(deck_rng(seed, 0 / 1))`); the rest cycle every ordered fixed deck pair. Each game
        has its own LookaheadAgent seeded from (seed, its seat), so results do not depend on the order
        in which games are stepped.

        Returns win_rate / draw_rate / games over everything, random_win_rate / random_draw_rate /
        random_games over the generated-deck deals (None without any), fixed_win_rate, `cells` = win
        rate per unordered fixed pair "i-j", `cell_counts` = [wins, games] per cell, and min_cell (the
        smallest cell; None unless every one of the n(n+1)/2 cells was played, which needs
        n_deals - n_deals // 2 >= n^2 fixed deals, so a partial eval never makes best.pt eligible)."""
        cfg = self.config
        n = cfg.n_decks
        n_rand = n_deals // 2
        jobs = []  # [game, policy seat, lookahead agent, deal kind]
        for d in range(n_deals):
            seed = QUICK_EVAL_SEED_BASE + d
            if d < n_rand:
                decks, kind = (generate_deck(deck_rng(seed, 0), cfg), generate_deck(deck_rng(seed, 1), cfg)), "random"
            else:
                decks, kind = divmod((d - n_rand) % (n * n), n), "fixed"
            for seat in (0, 1):
                g = Game(cfg)
                g.reset(seed, decks=decks)
                jobs.append((g, seat, LookaheadAgent(cfg, seed=worker_seed(seed, 1 - seat)), kind))
        gen = torch.Generator().manual_seed(EVAL_SAMPLING_SEED)
        X = np.zeros((len(jobs), self.encoder.dim), dtype=np.float32)
        M = np.zeros((len(jobs), self.n_actions), dtype=bool)
        self.actor.eval()
        while True:
            live = [j for j in jobs if not j[0].done]
            if not live:
                break
            mine = []
            for g, s, opp, _ in live:
                if g.current == s:
                    mine.append(g)
                else:
                    g.step(choose_action(opp, g))
            if mine:
                k = len(mine)
                X[:k] = 0.0
                for r, g in enumerate(mine):
                    g.legal_mask(out=M[r])
                    self.encoder.encode_into(g.observe(g.current), X[r], M[r])
                x = torch.from_numpy(X[:k].astype(np.float16).astype(np.float32))
                with torch.inference_mode():
                    a, _ = self.actor.act(x, torch.from_numpy(M[:k]), deterministic=deterministic, generator=gen)
                for r, g in enumerate(mine):
                    g.step(int(a[r]))
        tally = {"all": [0, 0, 0], "random": [0, 0, 0], "fixed": [0, 0, 0]}  # wins, draws, games
        cells = defaultdict(lambda: [0, 0])  # unordered fixed pair -> [wins, games]
        for g, s, _, kind in jobs:
            w = g.winner()
            win, draw = int(w == s), int(w == DRAW)
            for key in ("all", kind):
                t = tally[key]
                t[0] += win
                t[1] += draw
                t[2] += 1
            if kind == "fixed":
                c = cells[tuple(sorted(g.deck_ids))]
                c[0] += win
                c[1] += 1

        def rate(key: str, i: int):
            t = tally[key]
            return t[i] / t[2] if t[2] else None

        rates = {f"{i}-{j}": c[0] / c[1] for (i, j), c in sorted(cells.items())}
        complete = len(cells) == n * (n + 1) // 2
        return {"win_rate": rate("all", 0), "draw_rate": rate("all", 1), "games": tally["all"][2],
                "random_win_rate": rate("random", 0), "random_draw_rate": rate("random", 1),
                "random_games": tally["random"][2], "fixed_win_rate": rate("fixed", 0),
                "fixed_games": tally["fixed"][2], "min_cell": min(rates.values()) if complete else None,
                "cells": rates, "cell_counts": {f"{i}-{j}": list(c) for (i, j), c in sorted(cells.items())}}

    # ------------------------------------------------------------------ best.pt candidates and selection
    def _kept_candidates(self, cands: list) -> list:
        """The union of the top `select_top` candidates by the best.pt rule (best_score of the quick eval) and
        the top `select_top` by P(every cell >= BEST_MIN_CELL) (then random-deck rate); later updates win
        ties. The second ranking keeps balanced checkpoints whose small cell samples happened to dip
        below 0.6 next to the ones whose samples happened to clear it."""
        k = self.cfg.select_top
        by_rule = sorted(cands, key=lambda c: (tuple(c["score"]), c["update"]), reverse=True)[:k]
        by_prob = sorted(cands, key=lambda c: (c["p_eligible"], c["score"][1], c["update"]), reverse=True)[:k]
        keep = {c["update"] for c in by_rule + by_prob}
        return [c for c in cands if c["update"] in keep]

    def _add_candidate(self, ev: dict) -> None:
        """Offer the current policy (just quick-evaluated) as a best.pt candidate: saved as cand_<update>.pt
        while it ranks among the kept candidates, deleted once it drops out."""
        entry = {"update": self.update, "path": f"cand_{self.update:05d}.pt", "score": list(best_score(ev)),
                 "p_eligible": eligible_probability(ev), "quick": {k: ev.get(k) for k in (
                     "win_rate", "random_win_rate", "fixed_win_rate", "min_cell", "games")}}
        cands = [c for c in self.candidates if c["update"] != self.update] + [entry]
        keep = self._kept_candidates(cands)
        if any(c["update"] == self.update for c in keep):
            self.checkpoint(self.run_dir / entry["path"])
        kept = {c["update"] for c in keep}
        for c in cands:
            if c["update"] not in kept:
                with contextlib.suppress(FileNotFoundError):
                    (self.run_dir / c["path"]).unlink()
        self.candidates = keep

    def selection_eval(self, path: Path, evaluator) -> dict:
        """Evaluate one candidate checkpoint vs lookahead with `select_games` games on random decks and on
        every fixed deck pair (decks="all", rounded up to 2 n^2), deals from SELECTION_SEED_BASE (SPEC 8
        selection seeds), sampling policy. Returns a dict in quick_eval's format (best_score applies)."""
        from ..evaluation import deals_for_games
        n, games = self.config.n_decks, self.cfg.select_games
        spec = str(path)
        r = evaluator.match(spec, "lookahead", deals_for_games(games, "random", n), SELECTION_SEED_BASE,
                            decks="random")
        f = evaluator.match(spec, "lookahead", deals_for_games(games, "all", n), SELECTION_SEED_BASE, decks="all")
        cells = {f"{i}-{j}": w.win_rate for (i, j), w in sorted(f.cells.items()) if w.games}
        complete = len(cells) == n * (n + 1) // 2
        return {"win_rate": (r.wins + f.wins) / (r.games + f.games), "games": r.games + f.games,
                "random_win_rate": r.win_rate, "random_draw_rate": r.draw_rate, "random_games": r.games,
                "fixed_win_rate": f.win_rate, "fixed_games": f.games,
                "min_cell": min(cells.values()) if complete else None, "cells": cells,
                "cell_counts": {f"{i}-{j}": [w.wins, w.games] for (i, j), w in sorted(f.cells.items()) if w.games}}

    def select_best(self, log=print) -> Optional[dict]:
        """Final selection pass (SPEC 8): re-evaluate every kept candidate with `select_games` games per deck
        mode on the selection seeds, copy the best by the best.pt rule to best.pt and write selection.json.
        The quick evals that nominated the candidates have only 16-32 games per fixed cell, so this pass,
        with the verdict's sample sizes, makes the choice. Returns the selection summary (None when off or
        when there is no candidate)."""
        cfg = self.cfg
        cands = [c for c in self.candidates if (self.run_dir / c["path"]).is_file()]
        if not cfg.select_games or not cands:
            return None
        from ..evaluation import Evaluator
        workers = max(1, self.n_workers)
        log(f"selection: re-evaluating {len(cands)} best.pt candidates ({', '.join(c['path'] for c in cands)}) vs "
            f"lookahead, {cfg.select_games} games per deck mode (random, all), seeds from {SELECTION_SEED_BASE:,}, "
            f"{workers} worker{'s' if workers > 1 else ''}")
        t0 = time.time()
        rows = []
        with Evaluator(workers, self.config) as evaluator:
            for c in cands:
                sel = self.selection_eval(self.run_dir / c["path"], evaluator)
                rows.append(dict(c, select=sel, select_score=list(best_score(sel))))
                log(f"selection: {c['path']} random-deck win {sel['random_win_rate']:.3f} "
                    f"fixed {sel['fixed_win_rate']:.3f} min cell "
                    f"{'-' if sel['min_cell'] is None else format(sel['min_cell'], '.3f')} "
                    f"(quick eval: random {c['quick'].get('random_win_rate') or 0:.3f}, min cell "
                    f"{'-' if c['quick'].get('min_cell') is None else format(c['quick']['min_cell'], '.3f')})")
        best = max(rows, key=lambda r: (tuple(r["select_score"]), r["update"]))
        tmp = self.run_dir / "best.pt.tmp"
        shutil.copyfile(self.run_dir / best["path"], tmp)
        tmp.replace(self.run_dir / "best.pt")
        summary = {"chosen_update": best["update"], "chosen": best["path"], "eligible": best["select_score"][0] == 1,
                   "rule": f"highest random-deck win rate among candidates whose every fixed cell is >= "
                           f"{BEST_MIN_CELL}, else the highest overall", "games_per_mode": cfg.select_games,
                   "seed_base": SELECTION_SEED_BASE, "opponent": "lookahead", "seconds": round(time.time() - t0, 1),
                   "candidates": rows}
        with open(self.run_dir / "selection.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        self.last_selection = summary
        log(f"selection: best.pt = {best['path']} (update {best['update']}, "
            f"{'eligible' if summary['eligible'] else 'no candidate has every cell >= 0.6: highest overall'}); "
            f"details in {self.run_dir / 'selection.json'}")
        return summary

    def checkpoint(self, path: Path, with_optimizer: bool = False) -> None:
        net_cpu = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        state = {"model": net_cpu, "net": self.net.spec(), "encoder_version": ENCODER_VERSION,
                 "update": self.update, "env_steps": self.env_steps, "learner_steps": self.learner_steps,
                 "games": self.games, "args": asdict(self.cfg)}
        if with_optimizer:
            run_dir = self.run_dir.resolve()
            state.update(optimizer=self.opt.state_dict(), lr=self.lr_at(self.update),
                         snapshots=[(n, os.path.relpath(p, run_dir)) for n, p, _ in self.snapshots],
                         seed_base=self.seed_base, n_workers=max(1, self.n_workers),
                         worker_games=list(self.worker_games), best_eval=list(self.best_eval),
                         candidates=copy.deepcopy(self.candidates), games_per_collect=self._games_per_collect)
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)

    def resume(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        check_resumable(state, self.encoder)
        if state.get("net") != self.net.spec():
            raise ValueError(f"the architecture is fixed by the checkpoint, but this config differs in "
                             f"{_spec_diff(state.get('net'), self.net.spec())}: drop those flags to resume (or start "
                             f"a new run)")
        self.net.load_state_dict(state["model"])
        self.actor.load_state_dict(state["model"])
        if "optimizer" in state:
            self.opt.load_state_dict(state["optimizer"])
        self.update = state["update"]
        self.env_steps = state.get("env_steps", 0)
        self.learner_steps = state.get("learner_steps", 0)
        self.games = state.get("games", 0)
        # fresh deal seeds after everything the previous run (with its K workers) may have dealt; an explicit
        # --seed that differs from the run's continues in that seed's block (e.g. after the block ran out)
        saved_seed = (state.get("args") or {}).get("seed", self.cfg.seed)
        if saved_seed != self.cfg.seed:
            self.seed_base = self.seed_block
        else:
            K_old = state.get("n_workers", 1)
            games_old = state.get("worker_games", [self.games])
            self.seed_base = state.get("seed_base", self.seed_block) + K_old * (max(games_old) + 1)
        self.worker_games = [0] * max(1, self.n_workers)
        # deals per worker per collect, rescaled to this run's worker count (for the seed-budget check)
        self._games_per_collect = math.ceil(state.get("games_per_collect", 0) * state.get("n_workers", 1)
                                            / max(1, self.n_workers))
        if self.seed_base < self.seed_block:
            raise CheckpointError(f"the checkpoint's deal-seed counter {self.seed_base} lies before the seed block of "
                                  f"--seed {self.cfg.seed} (a damaged checkpoint?)")
        if self._games_left_per_worker() < 1:
            raise SeedBlockExhausted(self._exhausted_message("no deal seed is left"))
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
        self.candidates = []
        for c in state.get("candidates", []):
            if (src_dir / c["path"]).is_file():
                self.candidates.append(c)
            else:
                warnings.warn(f"best.pt candidate {c['path']} not found in {src_dir}; dropped from the selection")
        if self.candidates:
            self.candidates = self._kept_candidates(self.candidates)

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

    def _row(self, cstats: dict, stats: dict, t0: float, t1: float, t2: float, t_start: float) -> dict:
        row = {"update": self.update, "env_steps": self.env_steps, "learner_steps": self.learner_steps,
               "games": self.games, "collect_s": round(t1 - t0, 2), "learn_s": round(t2 - t1, 2),
               "learner_steps_per_s": round(cstats["transitions"] / max(1e-9, t1 - t0), 1),
               "decisions_per_s": round(cstats["decisions"] / max(1e-9, t1 - t0), 1),
               "elapsed_min": round((t2 - t_start) / 60, 2), "snapshots": len(self.snapshots), **stats}
        # worker time breakdown: mean seconds per worker and the share of all worker time per part
        times = cstats.get("time") or {}
        total = sum(times.values())
        k = max(1, cstats.get("n_workers", 1))
        for key, v in times.items():
            row[f"wt_{key}_s"] = round(v / k, 3)
            row[f"wt_{key}_frac"] = round(v / total, 4) if total > 0 else 0.0
        if cstats.get("infer_calls"):
            row["infer_batch_mean"] = round(cstats["infer_rows"] / cstats["infer_calls"], 2)
        if "io_s" in cstats:  # the learner's pipe traffic with the workers (requests, results, replies)
            row["pool_io_s"] = round(cstats["io_s"], 3)
        srv = cstats.get("server")
        if srv is not None:  # batched inference: device forwards for all workers (SPEC 8.1)
            row.update(server_forward_s=round(srv["forward_s"], 3), server_busy_s=round(srv["busy_s"], 3),
                       server_drains=srv["drains"], server_requests=srv["requests"], server_forwards=srv["forwards"],
                       server_rows_mean=round(srv["rows"] / srv["forwards"], 2) if srv["forwards"] else 0.0,
                       server_snapshot_forwards=srv["snapshot_forwards"],
                       server_snapshot_forward_s=round(srv["snapshot_forward_s"], 3))
        for kind, rs in self.results.items():
            if rs:
                row[f"train_vs_{kind}"] = round(float(np.mean([r > 0 for r in rs])), 3)
        pairs = [np.mean([r > 0 for r in rs]) for rs in self.lookahead_pairs.values() if len(rs) >= 20]
        if pairs:
            row["train_vs_lookahead_min_pair"] = round(float(min(pairs)), 3)
        return row

    def train(self, log=print) -> None:
        cfg = self.cfg
        info = asdict(cfg)
        info.update(device=str(self.device), amp_active=self.amp, workers_resolved=self.n_workers,
                    inference_server_active=self.inference_server,
                    obs_dim=self.encoder.dim, n_actions=self.n_actions, n_cards=self.n_cards,
                    encoder_version=ENCODER_VERSION, params=sum(p.numel() for p in self.net.parameters()))
        with open(self.run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        budget = self.seed_budget()
        projected = math.ceil(max(0, cfg.total_updates - self.update) * cfg.batch_steps / PLANNING_TRANSITIONS_PER_GAME)
        if projected > budget["remaining"]:
            warnings.warn(f"the training deal seeds of --seed {cfg.seed} may run out before update "
                          f"{cfg.total_updates}: {budget['remaining']:,} of {TRAIN_SEEDS_PER_RUN:,} are left and "
                          f"~{projected:,} are projected (at ~{PLANNING_TRANSITIONS_PER_GAME} learner transitions per "
                          f"game). If they do, training stops cleanly at an update boundary and can be continued "
                          f"with --resume <run>/latest.pt --seed <another unused seed>", stacklevel=2)
        writer = self._writer()
        metrics_file = open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8")
        t_start = time.time()
        completed = False
        try:
            while self.update < cfg.total_updates:
                t0 = time.time()
                trajs, cstats = self.collect()
                t1 = time.time()
                stats = self.learn(trajs)
                del trajs
                t2 = time.time()
                self.update += 1
                row = self._row(cstats, stats, t0, t1, t2, t_start)
                if self.update % cfg.snapshot_every == 0:
                    path = self.run_dir / f"ckpt_{self.update:05d}.pt"
                    self.checkpoint(path)
                    self._add_snapshot(path)
                if cfg.eval_every and self.update % cfg.eval_every == 0:
                    te = time.time()
                    ev = self.quick_eval(cfg.eval_deals)
                    row["eval_s"] = round(time.time() - te, 2)
                    row.update(eval_lookahead_win_rate=round(ev["win_rate"], 3),
                               eval_lookahead_draw_rate=round(ev["draw_rate"], 3),
                               eval_lookahead_fixed_win_rate=round(ev["fixed_win_rate"], 3))
                    if ev["min_cell"] is not None:  # None: eval_deals too small to play every fixed cell
                        row["eval_lookahead_min_cell"] = round(ev["min_cell"], 3)
                    if ev["random_win_rate"] is not None:
                        row["eval_lookahead_random_win_rate"] = round(ev["random_win_rate"], 3)
                        row["eval_lookahead_random_draw_rate"] = round(ev["random_draw_rate"], 3)
                    row.update({f"eval_cell_{k}": round(v, 3) for k, v in ev["cells"].items()})
                    score = best_score(ev)
                    if score > tuple(self.best_eval):  # provisional best.pt; the selection pass decides at the end
                        self.best_eval = score
                        self.checkpoint(self.run_dir / "best.pt")
                        row["best_update"] = self.update
                    self._add_candidate(ev)
                self.checkpoint(self.run_dir / "latest.pt", with_optimizer=True)
                metrics_file.write(json.dumps(row) + "\n")
                metrics_file.flush()
                if writer is not None:
                    for k, v in row.items():
                        if isinstance(v, (int, float)) and k != "update" and math.isfinite(v):
                            writer.add_scalar(k, v, self.update)
                    writer.flush()
                log(format_row(row))
            completed = True
        except BaseException:
            self.close(force=True)  # Ctrl-C or a failure: stop busy workers now instead of waiting for them
            raise
        finally:
            metrics_file.close()
            if writer is not None:
                writer.close()
            self.close()
        if completed:  # the rollout workers are stopped: the selection pass gets the CPU cores
            self.select_best(log)


_SPEC_FLAGS = {"kind": "--arch", "d_model": "--d-model", "layers": "--layers", "heads": "--heads", "ff": "--ff",
               "id_dim": "--id-dim", "ctx_dim": "--ctx-dim", "pair_dim": "--pair-dim", "hidden": "--hidden",
               "belief": "--no-belief", "privileged_critic": "--privileged-critic", "shared_trunk": "--shared-trunk"}


def _spec_diff(saved, current) -> str:
    """The network-spec keys where a checkpoint and the current config differ, with the CLI flag that sets
    each (layouts are compared by check_resumable's fingerprint check)."""
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return f"the network spec ({saved!r})"
    keys = [k for k in sorted(set(saved) | set(current)) if k != "layout" and saved.get(k) != current.get(k)]
    if not keys:
        return "the encoder layout"
    return ", ".join(f"{k} (checkpoint {saved.get(k)!r}, config {current.get(k)!r}"
                     f"{', ' + _SPEC_FLAGS[k] if k in _SPEC_FLAGS else ''})" for k in keys)


def format_row(row: dict) -> str:
    keys = ["update", "env_steps", "games", "learner_steps_per_s", "collect_s", "learn_s", "entropy", "kl",
            "clipfrac", "v_loss", "belief_loss", "belief_rprec", "explained_var", "lag_mean", "wt_infer_frac",
            "server_forward_s", "server_rows_mean",
            "train_vs_random", "train_vs_lookahead", "train_vs_snapshot", "train_vs_self_first",
            "eval_lookahead_random_win_rate", "eval_lookahead_min_cell", "eval_s", "elapsed_min"]
    parts = []
    for k in keys:
        if k in row:
            v = row[k]
            parts.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
    return " ".join(parts)
