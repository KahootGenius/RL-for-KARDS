# Project notes

Staged card-game AI.

- **Stage 1** (vanilla units, MLP PPO) is done.
- **Stage 2** (natures, traits, move costs, 4 decks, per-card entity encoder with attack previews,
  multiprocess rollouts, matchup cells + scenarios) is done and is the current code.
  - The desktop runs of 2026-10-09 all pass. Reference model: `models/stage2.pt` (run s2_b,
    `--attention-layers 1`).
  - Stage 3 is next. Before its first encoder or card-pool change, the user should tag the
    Stage 2 commit (`git tag stage2`): Stage 2 checkpoints only load with the Stage 2 code.

`SPEC.md` is the rules and interface contract (keep it in sync with every change); `README.md`
explains how to run everything.

## Two-machine workflow

Code is built on this laptop. The user pushes it to GitHub, pulls it on the home desktop
(Windows 11, Ryzen 9950X3D 16 cores, 32 GB RAM, RTX 5080/CUDA) and runs the real training there.

- The user works in **Windows PowerShell with native Windows Python**, not WSL. Seen 2026-10-09:
  PowerShell parse errors, and Task Manager showing Python processes.
- So every command for the desktop must be a plain `python` / `git` call that runs in Windows
  PowerShell 5.1. No `&&`, bash loops, brace expansion or `>` redirection (it writes UTF-16).
  Do file operations with `python -c` one-liners.
- The same code must run unchanged on this Mac (smoke tests, CPU/MPS, Python 3.13) and on the
  desktop (full runs, CUDA, Python 3.14.7, torch 2.11.0+cu128).
  - Keep it portable: open text files with `encoding="utf-8"`, use pathlib, use the "spawn"
    multiprocessing context, and put entry points behind `if __name__ == "__main__"`.
  - Keep printed text cp1252-safe: piped output on Windows is cp1252, so no `≥`, `→` or `γ` in
    prints.
  - Don't hard-code POSIX-only behaviour (signals, exit codes; Process.kill() exits -15 on
    Windows).

- **After every build:**
  1. Run the test suite: `.venv/bin/python -m pytest -q`.
  2. Smoke-test training with the real (default) config. Start `.venv/bin/python train.py --run-dir runs/smoke`
     only after the tests finish, wait until 2 updates are logged in `runs/smoke/metrics.jsonl`,
     then kill it and delete `runs/smoke/`. On this Mac the default 65k batch takes ~30 s per update
     on MPS; that is expected. Start the trainer with the Bash tool's background mode, not with `&`
     inside a script: a job started that way ignores SIGINT, so the Ctrl-C path goes untested.
  3. Finish by giving the user the exact list of files to push, as paths from the repo root,
     grouped into added / modified / deleted. Derive it from `git status --porcelain` when this
     folder is a git repo.
- Never commit or push; the user does that.
- **Heavy work runs on the desktop, not here.** On this laptop only run the test suite, the short
  smoke test and quick checks (a few minutes at most). Training runs, A/B experiments, checkpoint
  selection and long evaluations go into a desktop run plan for the user (exact commands + what to
  push back in `feedback/`).
- **Results from the desktop** come back in `feedback/<run-name>/`: `config.json`,
  `metrics.jsonl`, `eval.json` (from `eval.py --json`), optionally `best.pt` and a `notes.txt`
  (OS, Python/torch versions, wall-clock time, anything odd).
  - At the start of a session, after the user has pulled, read any new `feedback/` folders and
    use them to plan the next step.
  - Whole `runs/` directories never go into git; `runs/` is gitignored. A 1000-update run holds
    about 0.5 GB (a 5.3 MB checkpoint every 10 updates).
  - `elapsed_min` restarts at 0 when a run is resumed, and config.json is rewritten. A resumed
    run's wall-clock is the sum of its segments.

## Desktop measurements (reported by the user)

- 2026-10-09, `runs/s2_a` (defaults, 16 rollout workers).
  - **First start: `device=cpu`.** torch could not see CUDA.
    - About 12.2% total CPU and about 5,000 MB RAM. The 4-thread CPU learner dominated (4 of 32
      threads ≈ 12.5%).
    - Fixed by reinstalling torch from the cu128 index (`CUDA_TORCH_INSTALL` in ppo.py, README
      Setup). `--device auto` now warns when it falls back to the CPU.
  - **Resumed with `device=cuda`.** Collection and learning are sequential, so each side idles
    while the other works.
    - GPU utilisation alternates between 40–80% (learn) and about 2% (collect). GPU memory
      3.2 / 16 GB.
    - CPU alternates between about 5% (learn) and about 50% (collect: 16 workers on 32 threads).
    - About 8,000 MB RAM.
  - **Concurrency.** About 8 GB of RAM per run, out of 32 GB physical; native Windows has no WSL
    memory cap. CPU limits concurrency: one run's collection phase takes 16 workers.
  - **Split at update 105 (average of the last 10 updates):** collect 1.7 s + learn 1.3 s =
    3.0 s per update, versus about 43 s on this Mac. 1000 updates take about an hour, plus the
    in-training quick evals.
    - Quick evals run every 10 updates: 512 games vs greedy, single-threaded in the main process.
      They are not in `collect_s` / `learn_s`; compare `elapsed_min`.
    - Overlapping collection with learning would save at most about 40%. Not worth the extra
      policy lag while a run takes about an hour.
    - Compute is cheap on the desktop, so longer runs and more variants are affordable.
  - **Final wall-clock** (from metrics `elapsed_min`):
    - s2_a: 13 min on CPU + 51 min on CUDA (2.9 s/update).
    - s2_b: 93 min (5.4 s/update; attention costs 1.9×).
    - s2_c: 48 min (2.7 s/update).

## Stage 2 outcome (inputs for Stage 3)

- **Acceptance (2,016 duplicate games vs greedy, seeds 0–1007).** Full numbers in the README,
  `feedback/` and `results/stage2_eval.json`.
  - s2_a: 85.4%, worst cell 78.6%, 9/16 scenarios.
  - s2_b: 86.3%, worst cell 82.1%, 12/16.
  - s2_c: 85.4%, worst cell 78.6%, 10/16.
  - All pass. Re-running the evals on the Mac gave identical results.
- **Head-to-head.** B beats A 52.4%, C 52.3%, the laptop model 69.0% and random 99.95%. The three
  desktop runs are close in strength; B was picked after seeing the results.
- **Plateau.** The quick eval gains only about 0.4 points per 100 updates after update 500. By
  update 1000 the policy no longer beats its own recent snapshots (50.7%). More strength needs
  more pressure (opponent league, slower LR decay, bigger batch), not just more updates.
- **First-player advantage.** About 75% in self-play; vs greedy, B wins 96.9% first and 75.6%
  second. Keep every comparison seat-balanced. Consider compensating the second player when
  Stage 3 changes the rules.
- **Weak tactics.**
  - No desktop model solves fast_attack_then_move, move_cost_hold_frontline, budget_cheap_movers
    or defense_backline_order (greedy solves the last).
  - The laptop model solved the first two, so these regressed.
  - Track the solved set per scenario between versions, not only the count.
- **Architecture.** One attention layer gave +3 scenarios and +0.8 points (not significant) at
  1.9× the cost; γ 0.99 did not help. To pick Stage 3's default architecture, use several seeds,
  or compare at equal wall-clock.
- **Known minor portability gaps (not fixed).**
  - rollout.py's parent-death check (`os.getppid()`) never fires on Windows. A broken pipe still
    ends orphaned workers within one collection.
  - Snapshot paths in latest.pt use the OS separator, so a Windows latest.pt resumed on the Mac
    drops its snapshots (with a warning).

## Conventions

- Python: `.venv/bin/python` (3.13 here). Rollouts are CPU-bound (Python engine) and run in
  worker processes; only the network update uses the GPU.
- This folder is a git clone of github.com/KahootGenius/RL-for-KARDS (branch `main`), so push
  lists come from `git status --porcelain`. The user pushes with git (`git add -A`, commit, push),
  never GitHub's web uploader. That uploader stripped the folders from the Stage 1 push and the
  first Stage 2 push, which broke every `cardgame.*` import on the desktop.
- Rules live only in `engine.py` (`combat_damage` is the single combat rule; agents, features and
  scenario search call the engine). Bump `features.ENCODER_VERSION` whenever the encoding changes.
- Scripts in the scratchpad must not shadow stdlib modules (no `select.py`, `inspect.py`, ...), and
  multiprocessing scripts need an `if __name__ == "__main__"` guard.
- Seed ranges must stay disjoint: eval from 0, checkpoint selection 500,000,000+, training
  1e9 + seed·1e7, in-training quick evals 2e9+.
