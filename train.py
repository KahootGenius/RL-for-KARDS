#!/usr/bin/env python
"""Train the PPO self-play bot.

    python train.py --run-dir runs/ppo                           # 800 updates, ~15-20 min on 8 cores
    python train.py --resume runs/ppo/latest.pt --updates 1200   # continue a run

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

# CLI flag -> PPOConfig field (flags are --kebab-case of the field unless listed here)
RENAMED = {"updates": "total_updates"}


def build_parser() -> argparse.ArgumentParser:
    d = PPOConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Every default is None so we can tell which flags were passed; real defaults come from PPOConfig.
    add = ap.add_argument
    add("--run-dir", help=f"output directory (default {d.run_dir})")
    add("--seed", type=int, help=f"run seed in [0, 100) (default {d.seed})")
    add("--updates", type=int, help=f"total PPO updates (default {d.total_updates})")
    add("--num-envs", type=int, help=f"parallel games (default {d.num_envs})")
    add("--batch-steps", type=int, help=f"learner transitions per update (default {d.batch_steps})")
    add("--epochs", type=int, help=f"default {d.epochs}")
    add("--minibatch", type=int, help=f"default {d.minibatch}")
    add("--lr", type=float, help=f"default {d.lr}")
    add("--lr-final-frac", type=float, help=f"final LR as a fraction of --lr (default {d.lr_final_frac})")
    add("--gamma", type=float, help=f"default {d.gamma}")
    add("--gae-lambda", type=float, help=f"default {d.gae_lambda}")
    add("--clip", type=float, help=f"default {d.clip}")
    add("--ent-coef", type=float, help=f"default {d.ent_coef}")
    add("--vf-coef", type=float, help=f"default {d.vf_coef}")
    add("--target-kl", type=float, help=f"early-stop epochs above this KL, 0 = off (default {d.target_kl})")
    add("--hidden", type=int, nargs="+", help=f"MLP widths (default {' '.join(map(str, d.hidden))})")
    add("--snapshot-every", type=int,
        help=f"N: add a frozen snapshot to the opponent pool every N updates (default {d.snapshot_every})")
    add("--max-snapshots", type=int, help=f"default {d.max_snapshots}")
    add("--self-play-prob", type=float, help=f"P(opponent = latest self) (default {d.self_play_prob})")
    add("--eval-every", type=int, help=f"quick eval vs greedy every K updates, 0 = off (default {d.eval_every})")
    add("--eval-deals", type=int, help=f"deals per quick eval, 2 games each (default {d.eval_deals})")
    add("--torch-threads", type=int, help=f"default {d.torch_threads}")
    add("--resume", default=None, help="latest.pt of a previous run")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)["args"]
        base = {f.name: saved[f.name] for f in fields(PPOConfig) if f.name in saved}
        base["run_dir"] = str(Path(args.resume).parent)  # keep writing next to the resumed run
    else:
        base = asdict(PPOConfig())
    for flag, value in vars(args).items():
        if flag != "resume" and value is not None:
            base[RENAMED.get(flag, flag)] = value
    base["hidden"] = tuple(base["hidden"])
    cfg = PPOConfig(**base)

    trainer = PPOTrainer(cfg, load_ruleset())
    if args.resume:
        trainer.resume(args.resume)
        print(f"resumed from {args.resume} at update {trainer.update}")
    print(f"obs_dim={trainer.encoder.dim} n_actions={trainer.n_actions} run_dir={cfg.run_dir}")
    trainer.train()


if __name__ == "__main__":
    main()
