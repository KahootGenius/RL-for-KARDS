#!/usr/bin/env python
"""Train the PPO self-play bot (Stage 3: Transformer policy with a belief head, random decks).

    python train.py --run-dir runs/ppo                     # defaults: auto workers, device auto
    python train.py --run-dir runs/smoke --updates 2       # smoke test
    python train.py --resume runs/ppo/latest.pt --updates 1500
    python train.py --run-dir runs/pooled --arch pooled    # pooled baseline (SPEC 10, criterion 3)
    python train.py --run-dir runs/nobelief --no-belief    # belief-head ablation

Rollouts run in `--workers` CPU processes (default: cpu_count - 2, at most 16; 0 = in-process);
network updates run on `--device auto` (CUDA, else MPS, else CPU), in micro-batches of
`--micro-batch` rows per forward pass (bf16 autocast on CUDA unless --no-amp). With
`--inference-server on` (the default `auto` = on when the device is CUDA) the workers' policy
forward passes run batched on the learner device instead of on the workers' CPUs (snapshot
opponents too with --serve-snapshots; by default the workers run them). TensorBoard logs go to
<run-dir>/tb (`tensorboard --logdir runs`).

best.pt: every quick eval (vs lookahead, every --eval-every updates) may replace a provisional
best.pt and nominates best.pt candidates (cand_<update>.pt, at most 2 x --select-top kept). When
the run reaches --updates, a selection pass re-evaluates the candidates vs lookahead with
--select-games games on random and on fixed decks (selection seeds 500,000,000+) and copies the
best to best.pt (details in selection.json); --select-games 0 skips it. Resuming a finished run
with the same --updates runs only the selection pass.

Boolean options that default to on are switched off with --no-<name> (--no-belief, --no-amp,
--no-tensorboard); the others are switched on with --<name> (--privileged-critic, --shared-trunk).
--opp-weights takes kind=weight pairs, separated by spaces or (quoted in PowerShell, where an
unquoted comma splits the argument) commas: --opp-weights lookahead=0.3 snapshot=0.2.

On --resume the run's saved hyperparameters are restored and output continues in the resumed
file's directory; only flags given explicitly on the command line override them. The learning
rate continues from where the run stopped (rescaled if --lr is given) and decays linearly to
lr * lr_final_frac over the remaining updates. The architecture is fixed by the checkpoint.
Stage 1 and Stage 2 checkpoints cannot be resumed (different rules, encoding and network). Each
--seed owns 10,000,000 training deal seeds; when a long run uses them up it stops cleanly and
continues with --resume <run>/latest.pt --seed <another seed>. Refused resumes and invalid
settings print one "train.py: error:" line and exit with code 2.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import asdict, fields
from pathlib import Path

import torch

from cardgame.cards import load_ruleset
from cardgame.rl.ppo import (INFERENCE_SERVER_MODES, CheckpointError, PPOConfig, PPOTrainer, SeedBlockExhausted,
                             check_resumable)

RENAMED = {"updates": "total_updates"}  # CLI flag -> PPOConfig field
CHOICES = {"inference_server": INFERENCE_SERVER_MODES}  # string options with a fixed set of values
HELP = {
    "run_dir": "output directory", "seed": "run seed in [0, 100)", "total_updates": "total PPO updates",
    "workers": "rollout processes (-1 = auto, 0 = in-process)", "envs_per_worker": "games per rollout worker",
    "batch_steps": "learner transitions per update", "minibatch": "rows per optimizer step",
    "micro_batch": "rows per forward/backward pass (gradient accumulation; bounds GPU memory)",
    "target_kl": "early-stop epochs above this KL (0 = off)",
    "belief_coef": "weight of the belief-head BCE loss",
    "arch": "transformer | pooled | mlp", "layers": "transformer layers", "heads": "attention heads",
    "ff": "transformer feed-forward width", "id_dim": "card id embedding width",
    "ctx_dim": "context width (pooled arch)", "pair_dim": "pointer scorer pair-MLP width (token nets)",
    "hidden": "MLP widths (mlp arch)", "belief": "belief head: predict the opponent's hand (auxiliary loss)",
    "privileged_critic": "the value tower also sees the opponent's true hand (the actor never does)",
    "shared_trunk": "share the policy/value towers", "self_play_prob": "P(opponent = latest self)",
    "opp_weights": ("pool weights merged over the defaults (or the resumed run's): kind=weight pairs separated by "
                    "spaces, e.g. --opp-weights lookahead=0.3 snapshot=0.2 (a comma-separated list works too but "
                    "must be quoted in PowerShell); kinds: lookahead, random, snapshot, greedy"),
    "random_deck_frac": "P(a seat gets a generated deck) per training deal",
    "snapshot_every": "N: freeze a snapshot into the pool every N updates",
    "snapshots_per_worker": "snapshots each worker plays per update", "eval_every": "quick eval every K updates",
    "eval_deals": "deals per quick eval vs lookahead (2 games each; half random decks, half fixed pairs)",
    "device": "auto | cpu | cuda | mps", "torch_threads": "learner CPU threads",
    "amp": "bf16 autocast for the learner (CUDA only)", "tensorboard": "write TensorBoard logs",
    "worker_timeout": "seconds to wait for the rollout workers per update",
    "inference_server": ("batched inference for the rollout workers on the learner device: auto (on iff the device "
                         "is CUDA) | on | off; ignored with --workers 0"),
    "serve_snapshots": "with the inference server, also serve snapshot opponents (off: the workers' CPUs run them)",
    "select_top": "best.pt candidates kept per ranking (best.pt rule; P(every fixed cell >= 0.6)) for the selection pass",
    "select_games": ("games per deck mode (random, all) per candidate in the final best.pt selection pass vs lookahead "
                     "(0 = skip; best.pt then stays the quick-eval pick)"),
}
# Errors that mean "this checkpoint cannot be resumed" (torch.load of a missing, truncated or foreign file included)
RESUME_ERRORS = (CheckpointError, SeedBlockExhausted, OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError)


def parse_weights(text: str) -> dict:
    """Parse "lookahead=0.3,snapshot=0.2" (or one pair) into a dict; PPOTrainer checks kinds and values."""
    out = {}
    for part in text.split(","):
        k, _, v = part.partition("=")
        try:
            out[k.strip()] = float(v)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected kind=weight pairs, got {part!r}") from None
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = asdict(PPOConfig())
    inverse = {v: k for k, v in RENAMED.items()}
    for f in fields(PPOConfig):
        flag = "--" + inverse.get(f.name, f.name).replace("_", "-")
        default = defaults[f.name]
        help_text = f"{HELP.get(f.name, '')} (default {default})".strip()
        if isinstance(default, bool):
            if default:
                ap.add_argument("--no-" + flag[2:], dest=f.name, action="store_false", default=None, help=help_text)
            else:
                ap.add_argument(flag, dest=f.name, action="store_true", default=None, help=help_text)
        elif isinstance(default, (tuple, list)):
            ap.add_argument(flag, dest=f.name, type=int, nargs="+", default=None, help=help_text)
        elif isinstance(default, dict):  # several tokens: PowerShell splits an unquoted a=1,b=2 at the comma
            ap.add_argument(flag, dest=f.name, type=parse_weights, nargs="+", default=None, metavar="KIND=W",
                            help=help_text)
        else:
            ap.add_argument(flag, dest=f.name, type=type(default), default=None, choices=CHOICES.get(f.name),
                            help=help_text)
    ap.add_argument("--resume", default=None, help="latest.pt of a previous (Stage 3) run")
    return ap


def make_config(args: argparse.Namespace) -> PPOConfig:
    """The defaults (or the resumed run's saved config) overridden by the flags given explicitly;
    --opp-weights entries are merged over the base weights, so naming one kind keeps the others.
    Raises CheckpointError when --resume names a Stage 1/2 (or otherwise unusable) checkpoint."""
    base = asdict(PPOConfig())
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        check_resumable(state)
        saved = state.get("args") or {}
        base.update({f.name: saved[f.name] for f in fields(PPOConfig) if f.name in saved})
        base["run_dir"] = str(Path(args.resume).parent)  # keep writing next to the resumed run
    for name, value in vars(args).items():
        if name != "resume" and value is not None:
            if isinstance(base[name], dict):  # --opp-weights: one dict per token, merged in order
                merged = {}
                for part in (value if isinstance(value, list) else [value]):
                    merged.update(part)
                value = {**base[name], **merged}
            base[name] = value
    base["hidden"] = tuple(base["hidden"])
    return PPOConfig(**base)


def fail(message: str) -> None:
    """One-line refusal on stderr and exit code 2 (SPEC 8), the same for every bad setting or checkpoint."""
    print(f"train.py: error: {message}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        cfg = make_config(args)
    except RESUME_ERRORS as exc:
        if not args.resume:
            raise
        fail(f"cannot resume from {args.resume}: {exc}")
    try:
        config = load_ruleset()
    except FileNotFoundError as exc:
        fail(f"a game data file is missing ({exc}); is the checkout complete? Get the code with git (git pull), "
             f"not GitHub's web uploader, which drops folders")
    try:
        trainer = PPOTrainer(cfg, config)
    except ValueError as exc:
        fail(f"invalid settings: {exc}")
    if args.resume:
        try:
            trainer.resume(args.resume)
        except RESUME_ERRORS as exc:
            fail(f"cannot resume from {args.resume}: {exc}")
        print(f"resumed from {args.resume} at update {trainer.update}")
    budget = trainer.seed_budget()
    print(f"device={trainer.device} amp={'bf16' if trainer.amp else 'off'} workers={trainer.n_workers} "
          f"inference_server={'on' if trainer.inference_server else 'off'} "
          f"envs={max(1, trainer.n_workers) * cfg.envs_per_worker} arch={cfg.arch} obs_dim={trainer.encoder.dim} "
          f"n_actions={trainer.n_actions} params={sum(p.numel() for p in trainer.net.parameters()):,} "
          f"run_dir={cfg.run_dir}", flush=True)
    print(f"training deal seeds (--seed {cfg.seed}): {budget['used']:,} of {budget['used'] + budget['remaining']:,} "
          f"used", flush=True)
    try:
        trainer.train(log=lambda line: print(line, flush=True))
    except KeyboardInterrupt:  # workers are already stopped; latest.pt holds the last completed update
        latest = Path(cfg.run_dir) / "latest.pt"
        hint = f"; continue with --resume {latest}" if latest.exists() else ""
        print(f"interrupted after {trainer.update} completed updates{hint}", flush=True)
        raise SystemExit(130) from None
    except SeedBlockExhausted as exc:  # stopped before a collect; latest.pt holds the last completed update
        fail(f"training stopped after {trainer.update} completed updates: {exc}")


if __name__ == "__main__":
    main()
