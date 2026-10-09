#!/usr/bin/env python
"""Train the PPO self-play bot (Stage 2).

    python train.py --run-dir runs/ppo                     # defaults: auto workers, device auto
    python train.py --run-dir runs/smoke --updates 2       # smoke test
    python train.py --resume runs/ppo/latest.pt --updates 1500

Rollouts run in `--workers` CPU processes (default: cpu_count - 2, at most 16; 0 = in-process);
network updates run on `--device auto` (CUDA, else MPS, else CPU). TensorBoard logs go to
<run-dir>/tb (`tensorboard --logdir runs`).

On --resume the run's saved hyperparameters are restored and output continues in the resumed
file's directory; only flags given explicitly on the command line override them. The learning
rate continues from where the run stopped (rescaled if --lr is given) and decays linearly to
lr * lr_final_frac over the remaining updates. The architecture is fixed by the checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from pathlib import Path

import torch

from cardgame.cards import load_ruleset
from cardgame.rl.ppo import PPOConfig, PPOTrainer

RENAMED = {"updates": "total_updates"}  # CLI flag -> PPOConfig field
HELP = {
    "run_dir": "output directory", "seed": "run seed in [0, 100)", "total_updates": "total PPO updates",
    "workers": "rollout processes (-1 = auto, 0 = in-process)", "envs_per_worker": "games per rollout worker",
    "batch_steps": "learner transitions per update", "target_kl": "early-stop epochs above this KL (0 = off)",
    "arch": "entity | mlp", "attention_layers": "transformer layers over card slots (entity arch)",
    "shared_trunk": "share the policy/value towers", "hidden": "MLP widths (mlp arch)",
    "self_play_prob": "P(opponent = latest self)",
    "opp_weights": "pool weights merged over the defaults (or the resumed run's), e.g. greedy=0.3,snapshot=0.2",
    "snapshot_every": "N: freeze a snapshot into the pool every N updates",
    "snapshots_per_worker": "snapshots each worker plays per update", "eval_every": "quick eval every K updates",
    "eval_deals": "deals per quick eval (2 games each, all deck pairs)", "device": "auto | cpu | cuda | mps",
    "torch_threads": "learner CPU threads", "tensorboard": "write TensorBoard logs",
}


def parse_weights(text: str) -> dict:
    """Parse "greedy=0.3,snapshot=0.2" into {"greedy": 0.3, "snapshot": 0.2}; PPOTrainer checks kinds and values."""
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
        elif isinstance(default, dict):
            ap.add_argument(flag, dest=f.name, type=parse_weights, default=None, help=help_text)
        else:
            ap.add_argument(flag, dest=f.name, type=type(default), default=None, help=help_text)
    ap.add_argument("--resume", default=None, help="latest.pt of a previous run")
    return ap


def make_config(args: argparse.Namespace) -> PPOConfig:
    """The defaults (or the resumed run's saved config) overridden by the flags given explicitly;
    --opp-weights entries are merged over the base weights, so naming one kind keeps the others."""
    base = asdict(PPOConfig())
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)["args"]
        base.update({f.name: saved[f.name] for f in fields(PPOConfig) if f.name in saved})
        base["run_dir"] = str(Path(args.resume).parent)  # keep writing next to the resumed run
    for name, value in vars(args).items():
        if name != "resume" and value is not None:
            base[name] = {**base[name], **value} if isinstance(value, dict) else value
    base["hidden"] = tuple(base["hidden"])
    return PPOConfig(**base)


def main() -> None:
    args = build_parser().parse_args()
    cfg = make_config(args)

    trainer = PPOTrainer(cfg, load_ruleset())
    if args.resume:
        trainer.resume(args.resume)
        print(f"resumed from {args.resume} at update {trainer.update}")
    print(f"device={trainer.device} workers={trainer.n_workers} envs={max(1, trainer.n_workers) * cfg.envs_per_worker} "
          f"obs_dim={trainer.encoder.dim} n_actions={trainer.n_actions} "
          f"params={sum(p.numel() for p in trainer.net.parameters()):,} run_dir={cfg.run_dir}", flush=True)
    try:
        trainer.train(log=lambda line: print(line, flush=True))
    except KeyboardInterrupt:  # workers are already stopped; latest.pt holds the last completed update
        latest = Path(cfg.run_dir) / "latest.pt"
        hint = f"; continue with --resume {latest}" if latest.exists() else ""
        print(f"interrupted after {trainer.update} completed updates{hint}", flush=True)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
