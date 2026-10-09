# Card-game AI

A staged project: a fast, deterministic, headless engine for a two-player lane card game, and bots
that play it: **random**, **greedy** (a scripted baseline) and **PPO** (self-play reinforcement
learning). Cards and decks are data (`cardgame/data/*.json`); every rule lives in
`cardgame/engine.py`. [SPEC.md](SPEC.md) is the full rules and interface contract.

* **Stage 1** (done): vanilla units, two fixed decks, MLP policy; PPO beat greedy in 75.7% of 2,000
  duplicate games. Artifacts in `models/stage1/` and `results/stage1/` (they need the Stage 1 code).
* **Stage 2** (this version): unit natures (troop / fast / ranged), traits (Defense, Armor X),
  movement costs, 25 cards and 4 decks sampled per game, a feature-based per-card encoder with
  pointer-style action heads, multiprocess CPU rollouts with CUDA/MPS/CPU learning, TensorBoard,
  a deck-matchup evaluation and 16 tactical scenarios.

## Stage 2 results (laptop run; the full run is meant for the desktop)

`models/stage2_mac.pt` was trained on this laptop (M-series, MPS) for 250 updates of 16k decisions
(~4M decisions, ~1 h) — a sixteenth of the default desktop run. It was chosen *before* the scenario
results were looked at, by held-out win rate and worst matchup cell (selection seeds 500,000,000+).
Acceptance test (`python eval.py --agent models/stage2_mac.pt --scenarios`, 2,016 duplicate games on
seeds 0–1007; `results/stage2_mac_eval.json`):

| criterion | result | |
|---|---|---|
| beats greedy overall ≥ 70% | **76.3%** (95% CI 74.4–78.1) | pass |
| every matchup ≥ 60% | worst cell Blitz-Blitz **70.6%** (all 10 cells 70.6–81.7%) | pass |
| solves most scenarios (> 50%) | **8/16 = 50%** with argmax play (54% of sampled playouts); greedy 8/16, random 12% | **not yet** |

Matchup cells vs greedy (duplicate, both seats and decks of every deal):

| | Blitz | Bulwark | Volley | Legion |
|---|---|---|---|---|
| **Blitz** | 70.6 | 75.8 | 75.0 | 72.2 |
| **Bulwark** | | 75.4 | 79.0 | 77.8 |
| **Volley** | | | 80.2 | 76.6 |
| **Legion** | | | | 81.7 |

Against its own training run's older checkpoints it scores 68.7% (update 30), 53.1% (130), 50.3%
(230) and 48.6% (330, newer), and 99.9% against random (`results/stage2_mac_matches.json`).

What the laptop runs showed:

* **Win rate** keeps rising with training (72% at update 90 → 76–78% by 250–330); every deck
  matchup is comfortably above 60%.
* **Scenarios plateau around 8/16.** The bot reliably solves fast-troop lethal, frontline control
  with move costs, armor trades, sacrifice-to-clear and the ranged/Defense survival puzzle with a
  melee opener; it misses some lethal lines (ranged chip-lethal, move-budget lethal) and
  multi-step Defense sequencing under threat. Engine attack previews (+1–2 scenarios) and the
  pair MLP helped; a lower discount (γ 0.99) and filling the scenario decks did not change the
  count (γ 0.99 was compared at update 90). The full desktop run (16× more data) is the real
  test; `--attention-layers 1` is the next architecture lever if it still falls short.
* **Scenario validity.** Every scenario's "known best move" is checked to be objectively best
  (`scenarios.dominance_violations`, independently re-verified): lethal goals win now; survival
  goals have no lethal alternative and every surviving line makes the intended kills; material
  goals are never dominated by an alternative. Six original positions failed this check (their
  intended trade competed with face damage) and were rebuilt as survival or lethal puzzles; the
  bot's count was 8/16 before and after.

Stage 1 → Stage 2 speed (same laptop, back-to-back, `results/bench_stage2.txt`): engine-only random
games 0.89× (Stage 2 games are 30% shorter, so 0.64× per step), full agent loop 0.70×, clone 1.1×,
observe 2.8× and encode 4.8× (25 card slots with mask hints and attack previews).

## Setup

Python 3.10+.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Desktop (Windows 11 + WSL2, NVIDIA GPU).** Work inside the WSL2 Linux shell. The Windows NVIDIA
driver provides CUDA to WSL2 (do not install a Linux GPU driver). The default PyPI torch wheel for
Linux bundles CUDA; RTX 50-series cards need torch ≥ 2.7 built for CUDA 12.8+. Check:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If that prints False or warns about the GPU architecture, install a CUDA 12.8 build:
`.venv/bin/pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128`.
`train.py` falls back to CPU (with a warning) when CUDA cannot run.

**Moving code between the machines.** Use git on both sides: `git add -A`, `git commit`,
`git push` on the laptop, then `git clone` once and `git pull` after that on the desktop. Pull
before pushing on either machine. Do not use GitHub's web "Upload files" page: files dragged onto
it lose their folders, and every `cardgame.*` import then fails with "No module named 'cardgame'".
After each pull, check the checkout before training: `python -m pytest -q`, or the two-second
`python -c "import cardgame.rl.rollout"`.

## Run

Commands assume the repo root and an active venv (`source .venv/bin/activate`).

```bash
python -m pytest -q                       # full suite (~600 tests, 1–2 min); CARDGAME_SLOW=1 adds a slow deck-balance gate
python bench.py --compare results/stage1/bench.txt --out    # speed vs Stage 1 -> results/bench_stage2.txt
python tools/deck_balance.py              # deck balance matrices (greedy vs greedy / random)
```

### Training

Rollouts run in CPU worker processes; the network trains on CUDA if available, else MPS, else CPU.

```bash
python train.py --run-dir runs/s2                 # full run: 1000 updates, workers = cpu-2 (max 16)
python train.py --run-dir runs/smoke --updates 2  # smoke test
python train.py --resume runs/s2/latest.pt --updates 1500
tensorboard --logdir runs                         # curves (also runs/<name>/metrics.jsonl)
```

Defaults (`python train.py -h` lists every flag): 32 games per worker, 65,536 learner decisions
per update, minibatch 8,192, 4 epochs, lr 3e-4 → 3e-5, γ 0.997, λ 0.95, clip 0.2, entropy 0.01,
opponents 50% latest self / greedy 15% / random 5% / frozen snapshots 30% (`--opp-weights`), a
snapshot every 10 updates (newest 30 kept), quick eval vs greedy over all deck pairs every 10
updates. A run directory holds `metrics.jsonl`, `tb/`, `ckpt_XXXXX.pt`, `best.pt` (best quick
eval, preferring checkpoints whose every matchup cell is ≥ 60%) and `latest.pt`. Smaller
machines: `--batch-steps 16384 --minibatch 4096`.

### Evaluation (acceptance test)

```bash
python eval.py --agent runs/s2/best.pt --scenarios --json runs/s2/eval.json
python eval.py --agent runs/s2/best.pt --opponents greedy random --checkpoints-dir runs/s2 --checkpoint-stride 10
```

Every deal is played twice with the seats swapped, cycling through all 16 ordered deck pairs
(2,000 games → 2,016). The report shows overall / per-seat / per-turn-order win rates, the
matchup cell table, the deck-vs-deck matrix next to greedy-vs-greedy's, older checkpoints, the
scenario table (argmax and sampled success, with greedy and random baselines) and a verdict:
PASS/FAIL only when every part of the done criterion was measured, INDICATIVE otherwise.

### Desktop run plan (Stage 2)

Run A is the acceptance run; B and C are the two most promising variants (each about as long as A).

```bash
git pull && python -m pytest -q                                  # check the checkout first
python train.py --run-dir runs/s2_a                              # A: defaults
python train.py --run-dir runs/s2_b --attention-layers 1         # B: + one transformer layer over card slots
python train.py --run-dir runs/s2_c --gamma 0.99                 # C: shorter horizon (values faster wins)
python eval.py --agent runs/s2_a/best.pt --scenarios --json runs/s2_a/eval.json   # likewise for b, c
```

### Desktop → laptop feedback

After a desktop run, copy `config.json`, `metrics.jsonl`, `eval.json` and `best.pt` from
`runs/<name>/` into `feedback/<name>/`, add a short `notes.txt` (OS, Python/torch versions, GPU,
wall-clock time, anything odd) and push. Whole `runs/` directories stay out of git.

## The game (Stage 2)

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

**Cards and decks.** 25 units, costs 1–8, covering every nature × {none, Defense, Armor,
Defense+Armor}. Decks (40 cards, ≤ 3 copies): **Blitz** (aggro fast troops), **Bulwark**
(defensive), **Volley** (ranged-heavy), **Legion** (balanced). Deck counts were tuned until greedy
vs greedy sits within 46–54% for every deck pairing (`tools/deck_balance.py`, `tests/test_decks.py`).

Choices the brief left open are marked **[clarified]** in SPEC.md. The main ones: ranged units
also cannot act on their deploy round and are bound by Defense; fast units may attack before
moving; unit flags refresh at their owner's END_TURN; the opponent's deck choice is hidden (cards
they have played are public); "every matchup ≥ 60%" is judged on duplicate (unordered) deck cells;
the Stage 1 hand limit of 10 is kept.

## Design

**Engine** (`cardgame/engine.py`). `reset(seed, decks=None)`, `current_player()`,
`legal_actions()`, `legal_mask()`, `step(a)`, `observe(player)`, `winner()`, `clone()`.
Deterministic from the seed; deck pairs come from their own stream (`cards.sample_decks`), so
`reset(s)` and `reset(s, decks=sample_decks(s))` deal the same game. `clone()` copies units slot by
slot (no deepcopy on the hot path). `engine.combat_damage` is the single source of the combat rules
(the greedy bot uses it too). Fixed action space of 126: END_TURN, PLAY(hand slot) ×10, MOVE(backline
slot) ×5, ATTACK(attacker slot, target slot) ×110 with attackers = own backline + frontline and
targets = enemy backline + frontline + base.

**Observation and encoding** (`cardgame/features.py`). `observe(player)` is egocentric and hides
the opponent's hand, deck choice and deck order. The encoder turns every card (25 slots: hand, my
backline, frontline, enemy backline) into a 27-feature vector — cost, ATK, max/current HP,
move_cost, nature one-hot, Defense/Armor (+ value), status flags, zone, owner — plus a learned
card-id embedding; hints such as "playable", "can attack now", "targetable" and the combined
attack that can reach each target are read from the legal mask, never re-derived. Every legal
attack also gets an outcome preview (kills? attacker dies? damage dealt/taken) computed by the
engine's own `combat_damage`, so the policy does not have to re-learn combat arithmetic. Feature scales
are fixed constants and a fingerprint of the cards, decks, rules and encoder version is stored in
every checkpoint, so a retuned card pool cannot be loaded into an old model by accident.

**Model** (`cardgame/rl/network.py`). Separate policy and value towers; each runs a shared per-card
MLP over all slots and pools every zone (masked mean, max, sum, fill). The policy scores actions
from the slots involved: PLAY(i) from hand card i, MOVE(j) from backline unit j, ATTACK(a, t) from a
bilinear attacker-query × target-key term plus a narrow pair MLP that also reads the attack
preview, and a learned base token built from the enemy base HP. ~1.3M parameters. Optional
transformer layers over slots (`--attention-layers`).

**Training** (`cardgame/rl/ppo.py`, `rollout.py`). The Stage 1 PPO pipeline, now with `--workers`
spawned CPU processes that each run a batch of games, the latest policy and their share of
snapshot opponents, and exchange only numpy with the learner (weights out, finished trajectories
back; observations rounded to float16 on both sides). The learner recomputes values, does GAE per
seat-trajectory and the PPO epochs on the device. Worker w deals seeds `seed_base + w + K·n`, so no
two workers play the same deal; resume continues past every seed used. Worker crashes, deaths and
timeouts stop the run with a clear error (the last `latest.pt` stays valid).

**Greedy v2** (`cardgame/agents/greedy_agent.py`): take lethal; advance fast troops so they can
move and attack; play the costliest card; kill shielding Defense units; take favorable trades;
point ranged units at the most valuable target; advance troops; hit the base.

**Evaluation** (`cardgame/evaluation.py`, `eval.py`, `cardgame/scenarios.py`). Duplicate games over
every deck pair and both seats; unordered matchup cells (deck strength cancels) decide pass/fail,
the deck-vs-deck matrix is shown as a diagnostic. 16 hand-built scenarios with a known best line
(fast-troop lethal, attacking through Defense, ranged finishing a backline unit, armor math, move
budgets, clearing the frontline before lethal, …), each checked by a goal on the end-of-turn state
and proven solvable by an engine search.

## Tests

`python -m pytest -q` (~600 tests): an independent reference model of the rules (written from the
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
  data/cards.json, decks.json   25 cards, 4 decks
  cards.py        CardDef / GameConfig, strict loading, sample_decks
  actions.py      fixed action space (126) with encode/decode/describe
  engine.py       Game: all rules, observe(), clone(), legal mask, combat_damage
  features.py     per-card entity encoding + fingerprint
  evaluation.py   duplicate matches, matchup cells, deck matrix, Wilson intervals
  scenarios.py    16 tactical scenarios, runner and solver
  agents/         Agent protocol, RandomAgent, GreedyAgent (v2), make_agent()
  rl/             EntityPolicyNet / PolicyValueNet, rollout workers, PPO learner, PPOAgent
train.py  eval.py  bench.py  tools/deck_balance.py
tests/            see above
models/, results/ trained models and evaluation/benchmark outputs (Stage 1 in */stage1/)
SPEC.md           rules + interface contract;   CLAUDE.md   workflow notes for the assistant
```

## Extending (Stage 3+)

New card types, traits and effects go into `cards.json` together with engine code (the loader
rejects anything it cannot run). On-death and base-damage hooks: `Game._resolve_deaths()`,
`Game._damage_base()`. New actions (operation cards, targets, mulligan) append blocks after ATTACK
(existing indices never move); the target numbering is canonical and multi-step choices will add a
`phase` to the observation. Bump `features.ENCODER_VERSION` whenever the encoding changes.
