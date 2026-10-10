# Card-game AI

A staged project: a fast, deterministic, headless engine for a two-player lane card game, and bots
that play it: **random**, **greedy** (a scripted legacy baseline), **lookahead** (one-ply search on
determinized copies, the Stage 3 baseline) and **PPO** (self-play reinforcement learning). Cards and
decks are data (`cardgame/data/*.json`); every rule lives in `cardgame/engine.py`.
[SPEC.md](SPEC.md) is the full rules and interface contract.

* **Stage 1** (done): vanilla units, two fixed decks, MLP policy; PPO beat greedy in 75.7% of 2,000
  duplicate games. Artifacts in `models/stage1/` and `results/stage1/` (they need the Stage 1 code).
* **Stage 2** (done): unit natures (troop / fast / ranged), traits (Defense, Armor X), movement
  costs, 25 cards and 4 decks sampled per game, a feature-based per-card encoder with pointer-style
  action heads, multiprocess CPU rollouts with CUDA/MPS/CPU learning, TensorBoard, a deck-matchup
  evaluation and 16 tactical scenarios. All three desktop runs pass the done criterion; the
  reference model is `models/stage2/ppo_s2b.pt`.
* **Stage 3** (current code): card effects, operations, choices, the mulligan and new keywords
  (55 cards), random 40-card decks, a token encoder (`ENCODER_VERSION` 5) with a Transformer policy,
  a belief head and an optional privileged critic, the lookahead baseline, batched inference on the
  learner GPU, 20 scenarios and a four-part verdict. Stage 1/2 checkpoints cannot be loaded. The
  desktop run plan is under [Stage 3 desktop runs](#stage-3-desktop-runs).

## Stage 2 results (desktop runs: PASS)

Three 1000-update runs on the desktop (RTX 5080, 16 rollout workers, Windows 11, Python 3.14,
torch 2.11+cu128). Each checkpoint (`best.pt`) was picked by the in-training quick eval (seeds
2,000,000,000+). Each was then acceptance-tested on held-out deals with
`python eval.py --agent <best.pt> --scenarios`: 2,016 duplicate games against greedy on seeds 0–1007.
Run records are in `feedback/s2_{a,b,c}/`. Re-running all three evals on the laptop (macOS,
Python 3.13) gave identical results.

| run | vs greedy (95% CI) | worst matchup cell | scenarios (argmax) | wall-clock | s / update |
|---|---|---|---|---|---|
| A: defaults | 85.4% (83.8–86.9) | Blitz-Blitz 78.6% | 9/16 | 51 min* | 2.9 |
| **B: `--attention-layers 1`** | **86.3% (84.7–87.7)** | Blitz-Bulwark 82.1% | **12/16** | 93 min | 5.4 |
| C: `--gamma 0.99` | 85.4% (83.8–86.8) | Blitz-Blitz 78.6% | 10/16 | 48 min | 2.7 |

\*A's first 15 updates ran on the CPU (13 min) before PyTorch was fixed and the run was resumed on CUDA.

Done criterion (SPEC §10): ≥ 70% overall, ≥ 60% in every matchup cell, > 50% of scenarios solved.
**All three runs pass** (`conclusive` in each eval.json). Greedy itself solves 8/16 scenarios and
random averages 12% success. Run A clears the scenario bar by one scenario.

**Reference model: run B** (`models/stage2/ppo_s2b.pt`, evaluation in `results/stage2/stage2_eval.json`).
It leads on every measure. It was chosen after the results were seen, as the best of three single-seed
runs, and its lead is small. Head-to-head (duplicate games, `results/stage2/stage2_h2h.json`):

| B vs | A | C | laptop model | random |
|---|---|---|---|---|
| win rate | 52.4% (50.2–54.6) | 52.3% (50.1–54.5) | 69.0% (67.0–71.0) | 99.95% |

The laptop model is `models/stage2/ppo_laptop.pt`: 250 updates of 16k decisions; 76.3% vs greedy;
8/16 scenarios. (`--attention-layers` was a Stage 2 flag; Stage 3 replaced that network.)

B's matchup cells vs greedy (duplicate, both seats and decks of every deal):

| | Blitz | Bulwark | Volley | Legion |
|---|---|---|---|---|
| **Blitz** | 84.1 | 82.1 | 85.3 | 82.5 |
| **Bulwark** | | 90.5 | 90.9 | 87.7 |
| **Volley** | | | 88.1 | 86.9 |
| **Legion** | | | | 86.5 |

What the desktop runs showed:

* **Learning levels off by about update 500.** The quick eval rises from about 80% (updates
  100–200) to about 86% (updates 900–1000), but only about 0.4 points per 100 updates after
  update 500. By the end the policy no longer beats its own snapshots from the last 300 updates
  (50.7%). More strength needs more pressure (stronger or more varied opponents, a slower
  learning-rate decay, a larger batch), not just more updates.
* **Moving first is a big advantage.** In self-play the first player wins about 75%. Against
  greedy, B wins 96.9% going first and 75.6% going second. Duplicate (seat-swapped) evaluation
  cancels this out, but it caps win rates against strong opponents.
* **Attention costs 1.9× per update for a small gain.** At equal wall-clock (update 545) B's quick
  eval was lower than A's, 83.7% vs 85.7%. γ 0.99 did not help.
* **Known weaknesses, per scenario.**
  - No desktop model solves fast_attack_then_move, move_cost_hold_frontline or
    budget_cheap_movers (move costs and holding the frontline), or defense_backline_order (which
    greedy solves).
  - The laptop model did solve the first two, so these regressed.
  - A and C also miss ranged_base_lethal, which greedy solves.
  - Compare the solved set per scenario between versions, not just the count.
* **Scenario validity.** Every scenario's "known best move" is checked to be objectively best
  (`scenarios.dominance_violations`, independently re-verified): lethal goals win now; survival
  goals have no lethal alternative and every surviving line makes the intended kills; material
  goals are never dominated by an alternative. Six original positions failed this check (their
  intended trade competed with face damage) and were rebuilt as survival or lethal puzzles. The
  laptop model's count was 8/16 before and after.

Stage 1 → Stage 2 speed (same laptop, back-to-back but under load, so approximate;
`results/stage2/bench.txt`): engine-only random
games 0.89× (Stage 2 games are 30% shorter, so 0.64× per step), full agent loop 0.70×, clone 1.1×,
observe 2.8× and encode 4.8× (25 card slots with mask hints and attack previews).

## Setup

Python 3.10+.

```bash
python -m venv .venv
# activate it: source .venv/bin/activate (Mac/Linux) or .venv\Scripts\Activate.ps1 (PowerShell)
python -m pip install -r requirements.txt
```

**Desktop (Windows 11, NVIDIA GPU).** Native Windows (PowerShell) and WSL2 both work. Every command
in this README is a plain `python` or `git` call, so it runs in either shell. Avoid `&&`, shell
loops and `>` redirection: they differ between bash and Windows PowerShell. Under WSL2 the Windows
NVIDIA driver provides CUDA (do not install a Linux GPU driver). `nvidia-smi` must list the GPU.
RTX 50-series cards need torch ≥ 2.7 built for CUDA 12.8. `pip install -r requirements.txt` does not
always give you that: on native Windows PyPI's torch is CPU-only. Install the CUDA build explicitly,
with the same `python` that runs `train.py` (uninstall first, or pip keeps a same-version CPU build):

```bash
python -m pip uninstall -y torch
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line should print a `+cu128` version, `True` and the GPU's name. The first line of
`train.py`'s output must say `device=cuda`. With `--device auto` it falls back to the CPU and warns
why when CUDA cannot run.

**Moving code between the machines.** Use git on both sides: `git add -A`, `git commit`,
`git push` on the laptop, then `git clone` once and `git pull` after that on the desktop. Pull
before pushing on either machine. Do not use GitHub's web "Upload files" page: files dragged onto
it lose their folders, and every `cardgame.*` import then fails with "No module named 'cardgame'".
After each pull, check the checkout before training: `python -m pytest -q`, or the two-second
`python -c "import cardgame.rl.rollout"`.

## Run

Commands assume the repo root and an active venv (see Setup).

```bash
python -m pytest -q                       # full suite (~1,900 tests, a few minutes); CARDGAME_SLOW=1 adds a slow deck-balance gate
python bench.py --compare --out           # speed vs Stage 2 (results/stage2/bench.txt) -> results/bench_stage3.txt
python tools/deck_balance.py              # deck balance gate (lookahead vs lookahead / random, mulligan on)
```

### Training

Rollouts run in CPU worker processes; the network trains on CUDA if available, else MPS, else CPU.

```bash
python train.py --run-dir runs/s3                 # full run: 1000 updates, workers = cpu-2 (max 16)
python train.py --run-dir runs/smoke --updates 2  # smoke test
python train.py --resume runs/s3/latest.pt --updates 1500
tensorboard --logdir runs                         # curves (also runs/<name>/metrics.jsonl)
```

Defaults (`python train.py -h` lists every flag): the Transformer policy (`--arch transformer`, 3
layers, d_model 128, ~1.1M parameters) with the belief head (`--no-belief` turns it off), 32 games
per worker, 65,536 learner decisions per update, minibatch 8,192 in micro-batches of 1,024, 4
epochs, lr 3e-4 → 3e-5, γ 0.997, λ 0.95, clip 0.2, entropy 0.01, bf16 autocast on CUDA
(`--no-amp`). Opponents: 50% latest self, else lookahead 15% / random 5% / frozen snapshots 30%
(`--opp-weights lookahead=0.3 snapshot=0.2`; space-separated pairs work in PowerShell and bash). Each
seat gets a random 40-card deck with probability 0.7 (`--random-deck-frac`), else a fixed deck. A
snapshot every 10 updates (newest 30 kept). With a CUDA learner the workers' policy forward passes
run batched on the GPU (`--inference-server auto|on|off`; `--serve-snapshots` also serves the
snapshot opponents, which by default the workers run on their CPUs; SPEC 8.1 asks for an A/B of
that setting).

Every 10 updates a quick eval plays lookahead (256 deals x 2 seats: half on random decks, half
cycling the 16 fixed deck pairs; seeds 2,000,000,000+). It nominates **best.pt candidates**
(`cand_XXXXX.pt`: the top 3 by the best.pt rule plus the top 3 by the probability that every
fixed-pair cell is ≥ 60%) and keeps a provisional `best.pt`. A quick eval has only 16–32 games per
cell, too few to decide the 60% bar, so when the run reaches `--updates` a **selection pass**
re-evaluates the candidates against lookahead with 2,000 games on random decks and 2,016 on fixed
decks (selection seeds 500,000,000+, all rollout workers' cores) and copies the best to `best.pt`:
the highest random-deck win rate among candidates whose every fixed cell is ≥ 60%, else the
highest overall. `selection.json` records every candidate's numbers. `--select-games 0` skips the
pass; resuming a finished run with the same `--updates` runs only the pass.

A run directory holds `config.json`, `metrics.jsonl`, `tb/`, `ckpt_XXXXX.pt` (snapshots),
`cand_XXXXX.pt`, `best.pt`, `selection.json` and `latest.pt`. Each `--seed` owns 10,000,000 training
deal seeds (about 6,000–10,000 updates); if a very long run uses them up it stops cleanly at an
update boundary and continues with `--resume runs/<name>/latest.pt --seed <another seed>`. Refused
resumes and invalid flags print one `train.py: error:` line and exit with code 2. Smaller
machines: `--batch-steps 16384 --minibatch 4096`.

### Evaluation (acceptance test)

```bash
python eval.py --agent runs/s3/best.pt --baseline runs/s3_pooled/best.pt --scenarios --json runs/s3/eval.json
python eval.py --agent runs/s3/best.pt --opponents lookahead greedy --checkpoints-dir runs/s3 --checkpoint-stride 10 --decks random
```

The first line is the full SPEC 10 verdict: (1) vs lookahead on random decks (2,000 duplicate
games), (2) every unordered fixed-deck matchup cell vs lookahead ≥ 60% (2,016 games), (3) the
Transformer beats the pooled baseline head-to-head on random decks (`--baseline`), (4) > 50% of the
20 scenarios solved (argmax). Every deal is played twice with the seats swapped. The report shows
overall / per-seat / per-turn-order win rates, the matchup cells, the deck-vs-deck matrix next to
lookahead-vs-lookahead's, the scenario table (argmax and sampled success, the lookahead and random
baselines and the greedy reference) and a verdict: PASS/FAIL only when all four parts were measured
with the brief's settings, INDICATIVE otherwise (without `--baseline` part 3 is not measured).

### Stage 3 desktop runs

**Step 1: check the checkout, then a 20-minute inference A/B.** With a CUDA learner the workers'
policy inference runs batched on the GPU by default (`--inference-server auto`). That was slower
on the Mac's GPU and is unmeasured on CUDA, so time both settings once:

```bash
git pull
python -m pytest -q
python train.py --run-dir runs/srv_on --updates 30 --snapshot-every 1 --eval-every 0 --select-games 0
python train.py --run-dir runs/srv_off --updates 30 --snapshot-every 1 --eval-every 0 --select-games 0 --inference-server off
python -c "import json; [print(r, 'collect %.1fs learn %.1fs per update' % tuple(sum(x[k] for x in rows) / len(rows) for k in ('collect_s', 'learn_s'))) for r in ('srv_on', 'srv_off') for rows in [[json.loads(l) for l in open(f'runs/{r}/metrics.jsonl')][20:]]]"
```

If `srv_off` collects faster, add `--inference-server off` to the three training commands below.
Otherwise leave them as they are.

**Step 2: the three runs, one after another.** Each ends with its selection pass. Then the
evaluations.

```bash
python train.py --run-dir runs/s3                                # Transformer (default)
python train.py --run-dir runs/s3_pooled --arch pooled           # pooled baseline: SPEC 10 criterion (3)
python train.py --run-dir runs/s3_nobelief --no-belief           # ablation: belief head off
python eval.py --agent runs/s3/best.pt --baseline runs/s3_pooled/best.pt --scenarios --json runs/s3/eval.json
python eval.py --agent runs/s3_pooled/best.pt --scenarios --json runs/s3_pooled/eval.json
python eval.py --agent runs/s3_nobelief/best.pt --scenarios --json runs/s3_nobelief/eval.json
python eval.py --agent runs/s3/best.pt --opponents runs/s3_nobelief/best.pt --decks random --json runs/s3/h2h_nobelief.json
```

The first `eval.py` line is the acceptance verdict. The other three give the ablations:
* the pooled baseline against lookahead;
* the no-belief run against lookahead;
* the Transformer with vs without the belief head.

How long a run takes on the desktop is not measured yet; `elapsed_min` in `metrics.jsonl` records it.

The Stage 2 runs behind the results above were `runs/s2_a` (defaults), `runs/s2_b` (the Stage 2
flag `--attention-layers 1`, which no longer exists) and `runs/s2_c` (`--gamma 0.99`).

### Desktop → laptop feedback

After a desktop run, copy `config.json`, `metrics.jsonl`, `selection.json`, `eval.json` and `best.pt`
from `runs/<name>/` into `feedback/<name>/`, add a short `notes.txt` (OS, Python/torch versions, GPU,
wall-clock time, anything odd) and push. Whole `runs/` directories stay out of git.

These two lines work in PowerShell and bash alike. The first copies the files; the second starts
`feedback/notes.txt` with the OS, Python, torch and GPU versions. Edit the run list as needed.

```bash
python -c "import shutil, os; [(os.makedirs(f'feedback/{r}', exist_ok=True), shutil.copy2(f'runs/{r}/{f}', f'feedback/{r}/{f}')) for r in ('s3', 's3_pooled', 's3_nobelief', 'srv_on', 'srv_off') for f in ('config.json', 'metrics.jsonl', 'selection.json', 'eval.json', 'h2h_nobelief.json', 'best.pt') if os.path.exists(f'runs/{r}/{f}')]"
python -c "import sys, platform, torch; open('feedback/notes.txt', 'w', encoding='utf-8').write(f'{platform.platform()} | python {sys.version.split()[0]} | torch {torch.__version__} | GPU {torch.cuda.get_device_name(0)}\n')"
```

Then add your notes to `feedback/notes.txt`, and run `git pull`, `git add feedback`, `git commit`
and `git push` as separate commands.

## The game

Two players, 20-HP bases; destroying the enemy base wins; 50 full rounds is a draw. Each game deals
two of the four decks below (independently; mirrors allowed). A coin flip picks the first player,
who starts with 4 cards (the second with 5); everyone draws 1 at the start of each turn; round *n*
gives *n* coins and unused coins are lost. The board is P0 backline | frontline | P1 backline, at
most 5 units per zone, and the frontline is held by one side at a time. Units deploy to their
owner's backline and cannot act on the round they are deployed.

| nature | per round | reach |
|---|---|---|
| troop | move **or** attack | backline → enemy frontline; frontline → enemy backline or base |
| fast | move **and** attack (either order) | as troop |
| ranged | move or attack | any enemy unit or the base, from either zone; no return damage |

* **Moving** backline → frontline costs the card's `move_cost` (0–3), is one-way, and needs an
  empty or own frontline with space.
* **Defense**: attacks on units in a zone holding a Defense unit must target a Defense unit (ranged
  attackers too); the base is never protected.
* **Armor X**: every hit on the unit is reduced by X (minimum 0), whether it attacks or is attacked.
  Combat damage is simultaneous; the base never hits back.

**Cards and decks.** 55 cards, costs 1–8: the 25 Stage 2 units (unchanged, first in the file,
covering every nature × {none, Defense, Armor, Defense+Armor}), 20 new units with effects or Stage 3
keywords, 8 operations and 2 tokens (a Recruit unit and a Fire Mission order). Together they use every
trigger, action, target select/side/kind, the Stage 3 keywords (Blitz, Smokescreen, Fury, Ambush,
Shock, Immune) and tags with tag filters (`tests/test_data.py`). Decks (40 cards, ≤ 3 copies, 7–11 distinct effect cards each):
**Blitz** (aggro fast troops), **Bulwark** (defensive), **Volley** (ranged-heavy), **Legion**
(balanced). Cards and decks were tuned until lookahead vs lookahead sits within 46–54% for every deck
pairing (8,192 deals, mulligan on; `tools/deck_balance.py`, `tests/test_decks.py`).

**KARDS sample.** `research/kards_sample.json` encodes 110 real KARDS cards (official v53 data, all
seven kinds, ten nations) in the card schema: 68 (62%) are expressible; the other 42 name the missing
primitives. The most common gaps are countermeasures, rule modifiers (cannot be targeted, HQs cannot
gain defense, ...), cost modifiers, deck manipulation, delayed or granted effects and new triggers (a
player draws, a unit damages an HQ): candidates for Stage 4 (`tests/test_kards_sample.py`).

Choices the brief left open are marked **[clarified]** in SPEC.md. The main ones: ranged units
also cannot act on their deploy round and are bound by Defense; fast units may attack before
moving; unit flags refresh at their owner's END_TURN; the opponent's deck choice is hidden (cards
they have played are public); "every matchup ≥ 60%" is judged on duplicate (unordered) deck cells;
the Stage 1 hand limit of 10 is kept.

## Design

**Engine** (`cardgame/engine.py`). `reset(seed, decks=None)`, `current_player()`,
`legal_actions()`, `legal_mask()`, `step(a)`, `observe(player)`, `winner()`, `clone()`,
`determinize(player, rng)`. Deterministic from the seed; deck pairs come from their own stream
(`cards.sample_decks`), so `reset(s)` and `reset(s, decks=sample_decks(s))` deal the same game.
`clone()` copies units slot by slot (no deepcopy on the hot path). `engine.combat_damage` is the
single source of the combat rules. Fixed action space of 154 (SPEC 3): END_TURN, PLAY(hand slot)
×10, MOVE(backline slot) ×5, ATTACK(attacker slot, target slot) ×110, CHOOSE(target) ×17,
MULLIGAN(hand slot) ×10 and CONFIRM.

**Observation and encoding** (`cardgame/features.py`, SPEC 5–6). `observe(player)` is egocentric and
hides the opponent's hand, deck choice and deck order and the engine RNG. The encoder (version 5,
3,999 floats) turns every card slot into a token (static card features, current stats, keywords,
status flags, zone, owner, hints read from the legal mask), adds global, base, pending-choice,
revealed-card and own-deck tokens, and gives every legal attack and choice an outcome preview
computed by the engine. Feature scales are fixed constants and a fingerprint of the cards, decks,
rules and encoder version is stored in every checkpoint, so a retuned card pool cannot be loaded
into an old model by accident.

**Model** (`cardgame/rl/network.py`, SPEC 7). Card vectors from a card MLP over each card's static
features and a learned id embedding; a token MLP turns every card slot (hand, both backlines,
frontline, pending choice, revealed cards) plus global, base and deck tokens into embeddings; a
pre-norm Transformer encoder (3 layers) runs over the present tokens; one pointer scorer gives every
action (END_TURN, PLAY, MOVE, ATTACK, CHOOSE, MULLIGAN, CONFIRM) a logit from its source and target
tokens, with attack and choose previews computed by the engine. The value tower is separate; the
belief head predicts the opponent's hand (an auxiliary loss); the optional privileged critic
(`--privileged-critic`) also feeds the opponent's true hand to the value tower only. The pooled
baseline (`--arch pooled`) replaces attention with masked pooling per token group.

**Training** (`cardgame/rl/ppo.py`, `rollout.py`, SPEC 8). `--workers` spawned CPU processes each run
a batch of games and their scripted and snapshot opponents and exchange only numpy with the learner;
encodings cross the pipes in an exact sparse form (about a tenth of the dense float16 rows). With
the inference server the learner answers the workers' policy forward passes in batches on its
device while they collect. The learner recomputes values, does GAE per seat-trajectory and the PPO
epochs in micro-batches (gradient accumulation) on the device. Worker w deals seeds
`seed_base + w + K·n`, so no two workers play the same deal; resume continues past every seed used.
Worker crashes, deaths and timeouts stop the run with a clear error (the last `latest.pt` stays
valid).

**Greedy v2** (`cardgame/agents/greedy_agent.py`): take lethal; advance fast troops so they can
move and attack; play the costliest card; kill shielding Defense units; take favorable trades;
point ranged units at the most valuable target; advance troops; hit the base.

**Evaluation** (`cardgame/evaluation.py`, `eval.py`, `cardgame/scenarios.py`). Duplicate games on
random decks and over every fixed deck pair, both seats; unordered matchup cells (deck strength
cancels) decide pass/fail, the deck-vs-deck matrix is shown as a diagnostic. 20 hand-built
scenarios with a known best line (fast-troop lethal, attacking through Defense, armor math, move
budgets, operation lethal, on-death trades, effect damage through Defense, the mulligan, …), each
checked by a goal on the end-of-turn state, proven solvable by an engine search and free of cards
with random effects (a known best line never depends on the hidden RNG). In mid-game scenarios
the agent's deck holds the rest of its decklist, so the policy sees a deck as in training.

## Tests

`python -m pytest -q` (~1,900 tests): an independent reference model of the rules (written from the
spec) fuzzed against the engine on every deck pair; every nature, trait, move-cost and
frontline-control interaction as hand-built positions; hidden-information perturbation (including
the opponent's deck choice); determinism and clone independence (deepcopy forbidden on the hot
path); strict data loading and the shipped card/deck requirements; the deck-balance gate; the
encoder against an independent re-implementation; network slot wiring (permutation tests); PPO
math (GAE, advantage/clipping/entropy signs), worker seeding, protocol, failures and resume; greedy
rules and tie-breaks; evaluation bookkeeping; scenario solvability. The suite was mutation-tested:
every behaviour-changing mutant tried is caught.

## Package layout

```
cardgame/
  data/cards.json, decks.json   55 cards, 4 decks
  cards.py        CardDef / GameConfig, strict loading, sample_decks, random decks
  actions.py      fixed action space (154) with encode/decode/describe
  engine.py       Game: all rules and effects, observe(), clone(), determinize(), legal mask
  features.py     token encoding (v5) + fingerprint
  evaluation.py   duplicate matches, matchup cells, deck matrix, Wilson intervals
  scenarios.py    20 tactical scenarios, runner, solver and validity checks
  agents/         Agent protocol, RandomAgent, GreedyAgent (v2), LookaheadAgent, make_agent()
  rl/             TransformerPolicyNet / PooledPolicyNet / PolicyValueNet, rollout workers and the
                  inference server, PPO learner (selection pass), PPOAgent
train.py  eval.py  bench.py  tools/deck_balance.py
tests/            see above
models/, results/ trained models and evaluation/benchmark outputs (Stage 1 in */stage1/; Stage 2 in
                  */stage2/: models/stage2/ppo_s2b.pt, results/stage2/stage2_eval.json, ...)
feedback/         desktop run records (config, metrics, eval, best.pt) sent back for analysis
SPEC.md           rules + interface contract;   CLAUDE.md   workflow notes for the assistant
```

## Extending (Stage 3+)

New card types, traits and effects go into `cards.json` together with engine code (the loader
rejects anything it cannot run). On-death and base-damage hooks: `Game._resolve_deaths()`,
`Game._damage_base()`. New actions (operation cards, targets, mulligan) append blocks after ATTACK
(existing indices never move); the target numbering is canonical and multi-step choices will add a
`phase` to the observation. Bump `features.ENCODER_VERSION` whenever the encoding changes.
