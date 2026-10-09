# Project notes

Staged card-game AI. Stage 1 (vanilla units, MLP PPO) is done; Stage 2 (natures, traits, move
costs, 4 decks, per-card entity encoder with attack previews, multiprocess rollouts, matchup
cells + scenarios) is the current code. `SPEC.md` is the rules and interface contract (keep it in
sync with every change); `README.md` explains how to run everything.

## Two-machine workflow

Code is built on this laptop. The user pushes it to GitHub, pulls it on the home desktop
(Windows 11 + WSL2 Linux, Ryzen 9950X3D 16 cores, 32 GB RAM, RTX 5080/CUDA) and runs the real
training there. The same code must run unchanged on this Mac (smoke tests, CPU/MPS) and on the
desktop (full runs, CUDA). Keep it portable: open text files with `encoding="utf-8"`, use pathlib,
use the "spawn" multiprocessing context, and put entry points behind `if __name__ == "__main__"`.

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
  - Whole `runs/` directories (about 115 MB each) never go into git; `runs/` is gitignored.

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
