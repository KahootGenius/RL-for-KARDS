# Specification — Stage 2 (engine + agents + training contract)

Single source of truth for rules and interfaces. Stage 2 extends Stage 1 (vanilla units, two
fixed decks) with **natures**, **traits**, **movement costs**, **four decks sampled per game**,
feature-based card encoding, multiprocess training and matchup/scenario evaluation.
Where the brief is ambiguous, the resolution is marked **[clarified]**.

## 1. Data (`cardgame/data/`)

`cards.json` — `{"cards": [card, ...]}`; cards are indexed by file order (`CardDef.index`):

```json
{"id": "pikeman", "name": "Pikeman", "type": "unit", "nature": "troop",
 "cost": 4, "attack": 3, "health": 5, "move_cost": 1,
 "traits": {"defense": true, "armor": 1}, "effects": []}
```

* `type`: only `"unit"`. `nature` (required): `"troop"`, `"fast"` or `"ranged"`. `move_cost`:
  int ≥ 0, default 1. `cost`, `attack` ≥ 0, `health` ≥ 1 (ints, not bools).
* `traits`: object, trait → parameter. Known: `"defense": <bool>` (false = omitted) and
  `"armor": <int ≥ 1>`. `effects` must be `[]`.
* Unknown keys/traits/effects, wrong types or duplicate JSON keys → `ValueError` (the loader is
  strict so later stages extend data and engine together).

`decks.json` — `{"decks": [{"name", "style", "cards": {"<id>": count}}]}`. The loader accepts
N ≥ 1 decks, each exactly `deck_size` (40) cards and at most `max_copies` (3) of any card.

**Shipped content** (asserted by `tests/test_data.py`): 22–28 cards; every cost 1..8 occurs;
each nature × trait set {none, defense, armor, defense+armor} occurs (12 combinations); some card
has `move_cost` 0 and some ≥ 2. Decks in this order: 0 aggro (≥ 50% fast), 1 defensive (≥ 40%
Defense), 2 ranged-heavy (≥ 40% ranged), 3 balanced (each nature ≥ 20%).

**Deck balance gate** (`tests/test_decks.py`, `python tools/deck_balance.py` prints the
matrices): greedy vs greedy, deck-confounded `P[a][b] ∈ [0.35, 0.65]` for a ≠ b; draws ≤ 5% per
cell; greedy vs random ≥ 0.85 in every unordered cell; mean game length ≤ 30 rounds.

`load_ruleset(cards_path=None, decks_path=None, **overrides) -> GameConfig` (pool, `decks` =
tuple of sorted card-index tuples, `deck_names`, `deck_styles`, `n_decks`, `base_hp=20`,
`max_rounds=50`, `zone_capacity=5`, `max_hand_size=10`, `opening_hand=(4, 5)`, `deck_size=40`,
`max_copies=3`, `coin_cap=None`). `cards.sample_decks(seed, n)` → deck pair from its own
stream `random.Random(f"decks:{seed}")`.

## 2. Rules

Two players (seats 0, 1); bases have 20 HP; destroying the enemy base wins immediately.

### Setup — `reset(seed, decks=None)`
1. `seed`: non-negative int (numpy ints ok; bools/floats/None raise).
2. `decks = sample_decks(seed, n_decks)` if None, else a pair of deck indices (seat 0, seat 1).
   Independent uniform draws; mirror matchups allowed **[clarified]**. Hence
   `reset(s) ≡ reset(s, decks=sample_decks(s, n))`.
3. `rng = random.Random(seed)`; `first_player = rng.randrange(2)`; shuffle seat 0's deck then
   seat 1's (top = end of list).
4. Opening hands: first player 4, second 5. `round = 1`; start the first player's turn.

### Turns
* **Turn start (p)**: `coins[p] = round` (or `min(round, coin_cap)`); draw 1 (empty deck: none;
  hand at 10: burned — Stage 1 house rule). Hands are kept sorted by card index.
* **END_TURN (p)**: `coins[p] = 0`; every unit of p gets `summoned = moved = attacked = False`
  **[clarified: refreshed at the owner's END_TURN, so during the opponent's turn the flags show
  what each unit can do on its owner's next turn; legality is unaffected]**. If the opponent is
  the first player a new round starts; if `round == 50` the game is instead a **draw** (round
  stays 50).

### Units and action economy
A unit is created from its card (`atk`, `hp = max_hp`, `armor`, `defense`, `nature`,
`move_cost`), gets a unique `uid` (0, 1, … per game) and `summoned = True`.

| nature | can_move | can_attack |
|---|---|---|
| troop | `not summoned and not moved and not attacked` | same |
| fast | `not summoned and not moved` | `not summoned and not attacked` |
| ranged | same as troop | same as troop |

**[clarified]** Fast units may move then attack or attack then move. Ranged units also cannot act
on their deploy round, may MOVE (using their one action), and are subject to Defense.
`can_move`/`can_attack` depend only on (nature, flags), for every unit (a frontline unit may show
`can_move`); position, coins and targets are expressed only by the legal mask.

### Board
Zones: `backline[0]`, `frontline` (shared), `backline[1]`; at most 5 units each (lists; dead units
are removed and the list compacts, order preserved). `front_owner` = owner of the frontline units
or `None` when empty.

### Actions (current player p, opponent o)
* `END_TURN` — always legal while the game runs.
* `PLAY(i)` — `i < len(hand)`, `cost ≤ coins`, `len(backline[p]) < 5`. Pay, deploy to `backline[p]`.
* `MOVE(j)` — `backline[p][j]` exists, `can_move`, `front_owner in (None, p)`,
  `len(frontline) < 5`, `move_cost ≤ coins` (move_cost 0 is legal with 0 coins). Pay; the unit is
  appended to the frontline; `front_owner = p`; `moved = True`. One-way (no retreat).
* `ATTACK(a, t)` — `a` is a unit of p (`a < Z`: `backline[p][a]`; `a ≥ Z`: `front_owner == p` and
  `frontline[a−Z]`) with `can_attack`; `t` is an existing enemy unit (`t < Z`: `backline[o][t]`;
  `Z ≤ t < 2Z`: `front_owner == o` and `frontline[t−Z]`) or `t = 2Z` (enemy base); **reach** and
  **Defense** allow it. Legality never depends on damage: a 0-damage attack is legal, uses the
  action and still takes return damage. Effect: `attacked = True`, then combat.

**Reach.** Ranged: any enemy unit (either zone) or the base, from either zone. Troop/fast in the
backline: enemy frontline units only; in the frontline: enemy backline units or the base.

**Defense.** If the target is a unit and its zone holds ≥ 1 Defense unit, the target must be a
Defense unit (any of them). Applies to ranged attackers too. The base is never protected
**[clarified]**.

**Combat.** Base: `base_hp[o] -= atk` (no armor, no return); `≤ 0` → p wins at once. Unit:
simultaneously target takes `max(0, atk_att − armor_tgt)` and, unless the attacker is ranged,
the attacker takes `max(0, atk_tgt − armor_att)` (pre-combat attack values); units with `hp ≤ 0`
are removed; an emptied frontline gets `front_owner = None`. A ranged unit that is attacked
hits back normally **[clarified]**.

### 2.1 Required rule tests (hand-built positions, each also checked against the reference model)
1. Every nature: no MOVE/ATTACK on the deploy round; flags refresh at the owner's END_TURN.
2. Troop: no ATTACK after MOVE, no MOVE after ATTACK.
3. Fast: MOVE then ATTACK from the frontline (enemy backline or base); from the backline ATTACK
   that kills the last enemy frontline unit, then MOVE is legal (same attack with the target
   surviving: MOVE illegal); never a second MOVE or ATTACK.
4. Ranged: from own backline hits enemy backline, enemy frontline and base; from the frontline
   hits enemy backline and base; takes no return damage; deals return damage when attacked;
   cannot attack after MOVE.
5. Defense in the enemy backline vs a frontline troop and vs ranged; Defense in the enemy
   frontline vs a backline troop and vs ranged; two Defense units → either may be targeted;
   Defense in the other zone does not restrict; the base stays attackable; killing the last
   Defense unit lifts the restriction in the same turn.
6. Armor reduces damage to the target and the attacker's return damage; armor ≥ atk → 0 damage,
   still legal; base damage ignores armor.
7. Move cost: > coins illegal; = coins legal and coins → 0; move_cost 0 with 0 coins legal; fast
   units pay too.
8. Frontline control: enemy-held → MOVE illegal for every nature; 5 own units → illegal; an
   emptied frontline (by the defender's kill, by return damage killing the last own frontline
   attacker, by a mutual kill) gets `front_owner = None` and either side may move in.

## 3. Action space (`cardgame.actions.ActionSpace`, H = 10, Z = 5)

`0` END_TURN · `1+i` PLAY(i), i < H · `MOVE0 + j` MOVE(j), j < Z (`MOVE0 = 1+H`) ·
`ATTACK0 + a·(2Z+1) + t` ATTACK(a, t) (`ATTACK0 = 1+H+Z`; `attack(a, t)` helper).
Attacker slot `a < 2Z`: own backline 0..Z−1, frontline Z..2Z−1. Target slot `t ≤ 2Z`: enemy
backline 0..Z−1, frontline Z..2Z−1, base `BASE_TARGET = 2Z`. `N = 126`. `ActionKind`:
`END_TURN, PLAY, MOVE, ATTACK`. `decode`, `encode`, `describe`.
Forward compatibility: later stages only append blocks after ATTACK; existing indices never move;
target numbering is canonical and will be reused for targeted plays; multi-step choices
(mulligan, targets) will add an `Observation.phase` field.

## 4. Engine API (`cardgame.engine.Game`)

```python
game = Game(config=None)
game.reset(seed, decks=None); game.current_player(); game.legal_actions()  # sorted, [] when over
game.legal_mask(out=None)   # bool (N,)
game.step(a)                # IllegalActionError if not legal / not an int
game.observe(player) -> Observation; game.winner() -> None | 0 | 1 | DRAW(-1)
game.clone() -> Game        # independent; no deepcopy on hot fields
game.render(); game.invalidate()   # invalidate(): after direct edits (validates invariants)
```
State (for tests/tools): `first_player, current, round, deck_ids (seat → deck index),
deck_cards[2] (top = end), hands[2], played[2] (copies played per card index), coins[2],
base_hp[2], burned[2], backline[2], frontline, front_owner, next_uid, done, _winner, rng`.
`Unit(card, owner, *, atk, hp, max_hp=hp, armor=0, defense=False, nature=TROOP, move_cost=1,
uid=-1, summoned=False, moved=False, attacked=False)` (keyword-only after owner);
`Unit.from_card(card_def, owner, uid)`; `unit.can_move()`, `unit.can_attack()`.
Nature codes `TROOP=0, FAST=1, RANGED=2` (`cardgame.cards`).

**Changes from Stage 1** (every call site migrates): `Game.decks` (card lists) → `deck_cards`;
new `deck_ids`; `GameConfig.decks` = N decks; `Unit(card, atk, hp, owner, ready)` → keyword
form above; `ready` removed (use flags / `can_*`); `ATTACK_BASE/FRONT_ATTACK/BACK_ATTACK` and
`BASE0/FRONT0/BACK0` → `ATTACK`, `ATTACK0`, `n_attackers`, `n_targets`, `BASE_TARGET`.

`clone()`: units via `Unit.copy()` (explicit slot copy; all Unit fields are immutable scalars);
`hands`, `deck_cards`, `played` as new lists; `coins`, `base_hp`, `burned` as lists; RNG via
getstate/setstate; config and per-card tables shared; unknown attributes deep-copied.
Determinism: same seed (+ same `decks`) and actions ⇒ identical states.

## 5. Observation (egocentric NamedTuple)

```python
UnitView(card, atk, hp, max_hp, armor, defense, nature, move_cost, summoned, moved, attacked,
         can_move, can_attack)
Observation(player, is_my_turn, went_first, round, my_deck, my_coins, opp_coins, my_base_hp,
            opp_base_hp, hand, opp_hand_size, my_deck_size, opp_deck_size, my_played,
            opp_played, my_backline, opp_backline, frontline, front_owner, done, result)
```
`my_played`/`opp_played`: copies of each card index played this game (public — every played
card was on the board). `front_owner`: +1 observer, −1 opponent, 0 empty.
**Hidden**: opponent hand contents, the opponent's deck choice **[clarified]**, deck order and
content, burned cards, RNG. The no-leak test also replaces `deck_ids[o]` by another deck and
redraws `hands[o]`/`deck_cards[o]` from it (same sizes): `observe(p)`, its encoding and p's legal
actions must not change.

## 6. Feature encoding (`cardgame.features.ObservationEncoder`)

`encode(obs, mask=None)`, `encode_into(obs, row, mask=None)` (row zeroed by caller),
`encode_batch`, `split(x) -> (globals, features, ids, present)`, `dim`, `layout()`. `mask` is
the engine's bool (N,) legal mask for the observer's own turn or None; anything else raises.
Flat vector: `[globals G | slot features E·F | attack previews 2Z·(2Z+1)·4 | ids E | present E]`.
* Slots E = 25: hand [0, H) | my backline [H, H+Z) | frontline [H+Z, H+2Z) | opp backline
  [H+2Z, H+3Z). Embedding id = card index + 1 (0 = empty).
* Slot features (F = 27): present, cost, atk, max_hp, hp, move_cost, troop, fast, ranged,
  defense, has_armor, armor, summoned, moved, attacked, can_move, can_attack, playable,
  move_legal, attack_ready, targetable, incoming_atk, zone one-hot ×4, mine. `playable` …
  `incoming_atk` are read from the mask (never re-derived): `incoming_atk` = combined attack of my
  units with a legal attack on that target. **[clarified]** The status flags are the engine's
  flags, refreshed at the owner's END_TURN, so for enemy units at my decision time they describe
  readiness for the enemy's next turn (all clear) rather than "this round".
* Globals: is_my_turn, went_first, round, coins ×2, base HP ×2, hand sizes ×2, deck sizes ×2,
  front owner one-hot ×3, base_targetable, n_legal/N, base_damage_ready (combined attack with a
  legal base attack); own decklist summary (mean stats, nature and trait fractions, cost
  histogram); my_played/3, opp_played/3. `ENCODER_VERSION` is part of the fingerprint.
* Fixed scales (module constants, never derived from the pool): cost/8, atk/10, hp/10,
  move_cost/4, armor/4, coins/10, round/50, base/20, hand/10, deck/40, played/3.
* Attack previews: for every legal `ATTACK(a, t)` (mask), the outcome the engine's own combat rule
  gives (`engine.combat_damage`): kills, attacker dies, damage dealt/10, damage taken/10; for the
  base target, kills = atk ≥ enemy base HP. Zero for illegal attacks. These are action features,
  computed by the engine's rule, so the encoder still re-implements nothing.
* `pool_fingerprint(config)` hashes the cards, decks, rule constants, scales and the encoder
  version/schema; checkpoints store it in `layout` and loaders refuse a mismatch unless explicitly
  allowed.

## 7. Model (`cardgame.rl.network`)

* `EntityPolicyNet(layout, d_model=128, id_dim=16, ctx_dim=256, pair_dim=128,
  attention_layers=0, heads=4, shared_trunk=False)`: policy tower and value tower (separate
  parameters unless `shared_trunk`), each = per-card encoder `MLP([features, Embedding(id)])`
  shared across hand and zones, masked mean+max+fill-fraction pooling per group, globals MLP,
  context MLP; optional transformer layers over slots (globals token prepended).
* Pointer heads (policy tower): END = MLP(ctx); PLAY(i) = w·ReLU(U h_hand_i + V ctx);
  MOVE(j) likewise over my backline; ATTACK(a, t) = q_a·k_t/√k (bilinear, q from attacker slot +
  ctx) + w·ReLU(A h_a + T h_t + C ctx) (narrow pair MLP, `pair_mlp_dim`, for thresholds such as
  attack ≥ HP) + b(h_t), with attackers = my backline ‖ frontline and targets = opp backline ‖
  frontline ‖ base token (base token = Linear([opp base HP, 1])); the attack previews of each
  pair enter both the pair MLP and the logit directly. Pooling per group: masked mean, max,
  sum/capacity and fill fraction. Illegal actions get logit −1e9.
* Interface: `policy_logits(x, mask)`, `value(x)`, `act(x, mask, deterministic, generator)
  -> (actions, logp)`, `evaluate(x, mask, actions) -> (logp, entropy, value)`, `spec()`.
  `build_net(spec)` dispatches on `spec["kind"]` (`"entity"` / `"mlp"`); a Stage 1 checkpoint
  (no `kind`) raises a clear `ValueError`.
* All acting (workers, snapshots, quick eval, `PPOAgent`) runs on a CPU copy under
  `torch.inference_mode()`; the device copy is used only inside `learn()`.

## 8. Training (`cardgame.rl.ppo`, `cardgame.rl.rollout`, `train.py`)

Stage 1 PPO pipeline reused (GAE, clipped objective, LR decay, KL early stop, snapshots, best.pt).
* **Opponents**: per new game, latest self with probability 0.5; otherwise a pool category with
  weights greedy 0.15 / random 0.05 / snapshot 0.30 (`--opp-weights`; snapshot weight goes to
  greedy/random until a snapshot exists). A frozen snapshot is added every N = 10 updates; the
  newest 30 are kept. Each worker draws snapshot games from a per-update subset of at most 2
  snapshots (uniform each update), so snapshot inference stays batched. Mirror games record both
  seats. Reward +1/−1/0 at game end. Decks sampled per game.
* **Workers** (`--workers K`, default `max(1, min(16, cpu_count − 2))`; 0 = in-process for
  tests): spawned once (`spawn` context) with a picklable `WorkerInit` (config, net spec, PPO
  fields, worker id, seeds). Children get `CUDA_VISIBLE_DEVICES=""`, single-threaded BLAS and
  `torch.set_num_threads(1)`; never CUDA/MPS. `--envs-per-worker` (default 32).
* **Protocol** (numpy only): learner → worker `("collect", version, weights, pool_delta, quota)`
  with weights as numpy arrays and snapshots sent once (add/remove deltas); worker → learner
  finished seat-trajectories `{obs float16, mask packbits, act int16, logp float32, version
  int32, reward}` plus counters and per-opponent/per-deck-pair results. Observations are rounded
  to float16 before the worker's forward, so learner and worker see identical inputs.
* **Collection**: synchronous; each worker returns once it holds ≥ ceil(batch_steps / K)
  transitions from games finished this update; unfinished games continue under the next version.
  The learner recomputes values with the current value net and computes GAE per trajectory. Logged:
  lag mean/max, stale fraction, KL of the first minibatch.
* **Seeding**: run block `B = 1e9 + seed·1e7`; checkpoint stores `seed_base` (initially B), K and
  per-worker game counters; worker w deals `seed_base + w + K·n`. Worker RNGs (python, numpy,
  torch generator) from `agent_seed(seed_base, w)`. On resume (any K') `seed_base += K·(max n + 1)`
  and counters restart; seeds ≥ B + 1e7 raise.
* **Failures**: the learner waits with `connection.wait([conn, proc.sentinel], timeout)`; worker
  exceptions are sent back as tracebacks; any error/death/timeout terminates all workers and
  raises `RolloutWorkerError` (latest.pt stays valid). Workers ignore SIGINT and exit when the
  parent goes away.
* **Device**: `--device auto` = CUDA, else MPS, else CPU (printed, saved in config.json); tests
  always use cpu.
* **Defaults**: envs/worker 32, batch_steps 65,536, minibatch 8,192, epochs 4, lr 3e-4 → 3e-5,
  clip 0.2, ent 0.01, vf 0.5, grad-norm 0.5, target_kl 0.03, γ 0.997, λ 0.95, updates 1,000,
  snapshot_every 10, eval_every 10, eval_deals 256 (`decks="all"`).
* **Logging**: TensorBoard (`tensorboard` in requirements; writer only in the learner,
  `flush()` each update; `--no-tensorboard`); `metrics.jsonl`; checkpoints `ckpt_XXXXX.pt`,
  `best.pt` (highest quick-eval win rate among candidates whose minimum matchup cell ≥ 0.6, else
  highest overall), `latest.pt` (+ optimizer, LR, seeds, snapshot references).
* Seed ranges stay disjoint: eval 0+, selection 500,000,000+, training 1e9+, quick eval 2e9+.

## 9. Agents (`cardgame.agents`)

Protocol unchanged: `reset(seed)`, `act(obs, legal_actions) -> int`, `make_agent(spec, config)`.
Value of a unit = its card cost. Trades are judged with `engine.combat_damage(attacker, target)`
(the engine's own combat rule): a hit kills iff the damage to the target ≥ its hp; the attacker
survives iff the damage to it < its hp.

**Greedy v2** (first rule that yields an action; re-evaluated after every step):
0. **Lethal**: S = Σ atk of own units that have a legal ATTACK on the base. If S ≥ opp base HP,
   attack the base with the highest-atk such unit (ties: lower action index).
1. **Fast advance**: a legal MOVE of a fast unit that `can_attack` (key: higher atk, lower slot).
2. **Play**: the highest-cost affordable card (ties: higher atk+hp, then lower slot).
3. **Kill shielding Defense**: legal attacks that kill a Defense unit sharing its zone with ≥ 1
   non-Defense unit; max by (attacker survives, target value, −attacker value, −action index).
   **[clarified]** "kill Defense units first" = one-hit kills of a Defense unit that shields
   something; greedy does not plan focus fire (that is left for the learned policy).
4. **Favorable trade**: legal unit attacks that kill the target where the attacker survives or the
   target value > attacker value; max by (gain = target value − attacker value if it dies, target
   value, −action index).
5. **Ranged on high-value targets**: for each own ranged unit with `can_attack` (higher atk, lower
   slot first): among its legal unit targets with damage > 0, max by (kills, target value, damage,
   −action index); if none, attack the base. First ranged unit with an action wins.
6. **Advance**: MOVE any troop/fast unit that can (never ranged); key: higher atk, lower slot.
7. **Base**: a troop/fast unit in the frontline with a legal base attack (lowest frontline slot).
8. `END_TURN`.

## 10. Evaluation

* **Duplicate games**: per deal seed and deck pair (deck i seat 0, deck j seat 1), A-seat0/B-seat1
  then B-seat0/A-seat1 with identical shuffles and coin flip. `GameRecord` records `decks`.
* `decks="all"`: deal k uses `(i, j) = divmod(k % n², n)`; game counts are rounded up to a
  multiple of 2·n² (2,000 → 2,016). `decks="sampled"`: `sample_decks(seed)`.
* **Matchup cells (pass/fail)**: unordered `C{i,j}` = A's win rate over all games of deals dealt
  as (i, j) or (j, i) — each agent plays both decks of every deal, so deck strength and turn order
  cancel. 10 cells (4 mirrors + 6 cross), symmetric table with counts and Wilson 95% CIs.
  **[clarified]** "at least 60% in every matchup" is judged on these duplicate cells: an ordered
  cell P[a][b] mixes skill with deck strength (even equal players can sit far from 50% there), so
  the worst ordered cell is reported next to the verdict but does not decide it.
* **Deck-confounded matrix (diagnostic)**: `P[a][b]` = A's win rate when A holds deck a and B deck
  b, printed next to greedy-vs-greedy's.
* **Scenarios** (`cardgame.scenarios`): 12–20 `Scenario(name, tags, decks, build, goal)`.
  `build() -> Game` uses engine state (empty decks, consistent flags, `coins ≤ round`, cards from
  the named decks), agent to move. `goal(start, end) -> bool` is checked when the agent's turn
  ends (END_TURN or game over; cap 64 actions); units are identified by `uid`; the goal is False
  at the start. A PPO agent plays deterministically (argmax); also report its stochastic success
  over 100 playouts; greedy and random (mean of 20 seeds) for comparison. Tests: a DFS over the
  turn (clone, depth ≤ 12) finds a solution for every scenario; END_TURN alone never solves one;
  random solves < 50% on average.
  **Validity ("known best move")** — `dominance_violations()`, asserted for every scenario: `win`
  goals are best by definition; `survives_next_turn` goals require that the agent has no lethal
  this turn; material goals (`kills`/`no_losses`) require that every end-of-turn state missing
  the goal is weakly dominated by a goal state on (won, enemy base HP, enemy board value, own
  board value), found by an exhaustive engine search. Because ranged units can always shoot the
  base, ranged and Defense tactics are posed as survival puzzles (the enemy threatens lethal and
  only the tactic prevents it — a test asserts that every surviving line makes the intended kills)
  rather than as material goals that compete with face damage.
* **Done when**: PPO (sampling) vs greedy ≥ 70% over ≥ 2,000 duplicate games (`decks="all"`,
  seeds from 0); every unordered matchup cell ≥ 60% (point estimate; CI reported); > 50% of
  scenarios solved (argmax). `eval.py` prints a conclusive PASS/FAIL only when all three parts were
  measured under these settings (`--scenarios` included, thresholds not lowered); otherwise
  INDICATIVE with the reasons.

## 11. Benchmark

`bench.py` reports games/s, steps/s, steps and rounds per game, and per-call µs (observe,
encode, legal_actions, legal_mask, clone) and writes `results/bench_stage2.txt`. The README
compares with Stage 1 (`results/stage1/bench.txt`), measured on the same machine.
