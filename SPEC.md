# Specification — Stage 3 (engine + agents + training contract)

Single source of truth for rules and interfaces. Stage 3 extends Stage 2 (natures, traits, move
costs, four fixed decks, per-card encoder, multiprocess training) with **card effects**,
**operation cards**, **pending choices**, the **mulligan**, **randomized decks**, **determinize**,
a **Transformer** model with a belief head, and a **one-step lookahead** baseline bot. Stage 4
will import the full KARDS card pool, so effects are general data, never per-card code.
Where the brief is ambiguous, the resolution is marked **[clarified]**. Stage 2 rules that are not
mentioned here are unchanged.

## 1. Data (`cardgame/data/`)

### 1.1 Cards (`cards.json`)

`{"cards": [card, ...]}`; cards are indexed by file order (`CardDef.index`). Three kinds:

```json
{"id": "grenadier", "name": "Grenadier", "type": "unit", "nature": "troop", "cost": 3,
 "attack": 2, "health": 3, "move_cost": 1, "traits": {"defense": true},
 "effects": [{"trigger": "on_deploy", "action": "damage", "amount": 1,
              "target": {"select": "chosen", "side": "enemy", "kind": "unit"}}]}
{"id": "barrage", "name": "Barrage", "type": "operation", "cost": 3,
 "effects": [{"trigger": "on_play", "action": "damage", "amount": 1,
              "target": {"select": "all", "side": "enemy", "kind": "unit", "zone": "board"}}]}
{"id": "conscript", "name": "Conscript", "type": "unit", "nature": "troop", "cost": 1,
 "attack": 1, "health": 1, "token": true, "effects": []}
```

* `type`: `"unit"` or `"operation"`. Units need `nature`, `attack` ≥ 0, `health` ≥ 1, optional
  `move_cost` (≥ 0, default 1) and `traits`. Operations have only `id`, `name`, `type`, `cost`,
  `effects` (≥ 1, every one `on_play` with scope `self`) and optional `token`.
* `traits` (units): `"defense": bool`, `"armor": int ≥ 1`, `"blitz": bool`, `"smokescreen": bool`,
  `"fury": bool` (false = omitted). See §2.4.
* `token: true` marks a card that only effects create (`summon`, `add_card`). Tokens never appear in
  decks, and decks never reference them. `summon` and `add_card` may only name token cards
  **[clarified: so every non-token card a player holds came from their deck, which keeps
  `determinize` exact]**.
* `effects`: at most **3** per card (encoder slots), schema in §1.2.
* The loader is strict. Unknown keys, traits, triggers, actions or filters raise `ValueError`, as
  do wrong types, duplicate JSON keys and invalid combinations (§1.2 validation).

### 1.2 Effect schema

```
effect = {"trigger": T, "scope"?: S, "condition"?: C | [C, ...], "target": TG, "action": A,
          "amount"?: AM, ...action parameters}
```

**Triggers** (T) and **scopes** (S, default in bold):

| trigger | fires when | scopes |
|---|---|---|
| `on_play` | an operation is played | **`self`** (this operation); on units: `friendly` / `enemy` / `any` = an operation of that side is played |
| `on_deploy` | a unit is played from hand (not when summoned) | **`self`**, `friendly`, `enemy`, `any` (another unit of that side) |
| `on_death` | a unit is destroyed (hp ≤ 0 or `destroy`; not `return_to_hand`/`retreat`) | **`self`**, `friendly`, `enemy`, `any` |
| `on_attack` | a unit attacks (before combat damage) | **`self`**, `friendly`, `enemy`, `any` |
| `on_damaged` | a unit takes > 0 damage and survives | **`self`**, `friendly`, `enemy`, `any` |
| `on_move` | a unit moves into the frontline | **`self`**, `friendly`, `enemy`, `any` |
| `on_kill` | a unit destroys another unit with combat damage | **`self`**, `friendly`, `enemy`, `any` |
| `start_of_turn` | a turn starts (after the draw) | **`friendly`** (the controller's turn), `enemy`, `any` |
| `end_of_turn` | a turn ends (END_TURN, before cleanup) | **`friendly`**, `enemy`, `any` |

Scopes `friendly`/`enemy`/`any` watch **other** units (never the source itself), relative to the
watcher's controller. `on_play` on a unit card watches operations. Operations may only use
`on_play` + `self`. Units may not use `on_play` + `self`.

**Actions** (A) and their parameters:

| action | targets (`kind`) | parameters |
|---|---|---|
| `damage` | unit, base, unit_or_base | `amount` |
| `heal` | unit, base, unit_or_base | `amount` (int or `"full"`) |
| `buff` | unit | `atk` (int, may be < 0), `hp` (int ≥ 0), `duration` |
| `destroy` | unit | — |
| `draw` | player | `amount` |
| `gain_coins` | player | `amount` |
| `increase_max_coins` | player | `amount` (int, may be < 0) |
| `add_trait` | unit | `trait`, `amount` (armor only, default 1), `duration` |
| `remove_trait` | unit | `trait` (name or list of names), `duration` |
| `summon` | player | `card` (token id), `amount` (copies, default 1) |
| `return_to_hand` | unit | — |
| `pin` | unit | — |
| `discard` | player | `amount` (random cards) |
| `add_card` | player | `card` (token id), `amount` (copies, default 1) |
| `retreat` | unit | — |

`duration`: `"permanent"` (default) or `"turn"` (until the end of the current turn, §2.3).

**Targets** (TG): an object or a shorthand string.

```
{"select": "chosen" | "random" | "all" | "self" | "event",
 "side": "friendly" | "enemy" | "any",
 "kind": "unit" (default) | "base" | "player" | "unit_or_base",
 "zone": "board" (default) | "backline" | "frontline",
 "filter"?: F, "count"?: int >= 1 (random only, default 1)}
```

Shorthands: `"self"`, `"event"`, `"friendly_base"`, `"enemy_base"` (= `select all`,
`kind base`), `"controller"`, `"opponent"` (= `select all`, `kind player`).
* `side` is relative to the effect's controller. It is required unless `select` is `self` or `event`.
* `zone` applies to units only. The frontline's units belong to `front_owner`.
* `chosen`: the controller picks one legal option (§2.9).
* `random`: `count` distinct options drawn with the engine RNG.
* `all`: every match.
* `self`: the source unit.
* `event`: the other unit of the triggering event (§2.8).

**Filters** (F, units only; all given keys must hold):
* `nature` (name or list); `trait` (has it); `not_trait`; `damaged` (bool: hp < max_hp).
* `min_cost` / `max_cost`; `min_atk` / `max_atk`; `min_hp` / `max_hp` (current hp).
* `token` (bool); `other` (true: not the source unit).

**Amounts** (AM):
* An int ≥ 0, or `"full"` (heal only).
* `{"stat": "atk" | "hp" | "max_hp" | "cost", "of": "self" | "event"}`.
* `{"count": "units", "side", "zone"?, "filter"?}` (matching units on the board).
* `{"count": "hand" | "coins" | "deck", "side"}`.
* Expressions take optional `"times"` (int, default 1) and `"plus"` (int, default 0). The value
  is `max(0, times·value + plus)`.
* `buff.atk` / `buff.hp` accept the same forms, and a negative literal for `atk`.

**Conditions** (C, checked when the effect starts to resolve; a list = all must hold; false →
the effect is skipped):
* `{"type": "control", "side", "zone"?, "filter"?, "min"?: int, "max"?: int}` (count of matching
  units). `min` defaults to 1 when neither bound is given, and to 0 when only `max` is given.
* `{"type": "frontline", "owner": "friendly" | "enemy" | "none"}`.
* `{"type": "source_zone", "zone": "backline" | "frontline"}`.
* `{"type": "base_hp" | "hand_size", "side", "min"?: int, "max"?: int}`.
* `{"type": "turn", "whose": "own" | "opponent"}`.

**Validation** (loader, `ValueError` with the card id):
* The action/kind table above holds.
* `self`/`event` only with kind `unit`. `kind player` only with `select all`. `count` only with
  `random`.
* `chosen` only for triggers that fire during the controller's own action: `on_play`,
  `on_deploy`, `on_move`, `on_attack`, all with scope `self` **[clarified: as in KARDS and
  Hearthstone, nothing asks a player to choose during the opponent's turn, so the player to act
  is always the player whose turn it is]**.
* `event` and `{"of": "event"}` only for triggers that have an event unit (§2.8 table).
  Self-scope `on_deploy`, `on_move`, `on_death` and the turn triggers have none.
* Self-scope `on_death` effects may not target `self` (a watcher may target itself).
* `summon` names a unit token; `add_card` names any token.
* `trait` names are known traits. `amount` is used only by actions that take it.
* Amount requirements:
  * `damage`, `heal`, `draw`, `gain_coins`, `increase_max_coins` and `discard` require `amount`.
  * `summon`/`add_card` copies and `add_trait` armor default to 1 (≥ 1).
  * `buff` needs at least one of `atk`/`hp`.
  * Literal amounts are ≥ 0, except `buff.atk`, `increase_max_coins` and `gain_coins` (§1.2b).
* Operation-only restrictions: `{"of": "self"}` only with `stat: cost` (the operation's own
  cost); no `source_zone` condition.
* `self`/`event` targets take no `side`/`zone` but may take a filter (no match → fizzle).
* Filters: `trait`/`not_trait` take a name or a list ("has all" / "has none"); a `nature` list
  means "any of".
* Empty condition lists are rejected.

### 1.2b Schema extensions from the KARDS sample (phase 1b)

Added after checking the effect schema against 453 real KARDS cards (`research/`). These are the
"must" patterns plus the cheap "should" patterns, ranked by how many sampled cards use them.

**Card attributes**
* `tags`: a list of strings on any card (e.g. `["tank", "germany"]`): KARDS unit kind, nation,
  family. Filter keys `tag` (has any of the given tags; string or list) and `not_tag`.

**Traits** (units; also for `add_trait`/`remove_trait`):
* `ambush`: the first time each turn this unit is attacked by an attacker that would take return
  damage, it strikes first. If that strike kills the attacker, the ambusher takes no damage;
  otherwise the attacker's damage follows. Its per-turn use refreshes at every turn start.
* `shock`: when this unit attacks, it takes no return damage (like a ranged attacker; Ambush does
  not fire against it).
* `immune`: the unit takes no damage, from combat or effects. `destroy` still kills it.

**Triggers**
* `on_attacked`: a unit is attacked, after `on_attack` and before combat damage. The event unit
  is the attacker. Scopes as for the other unit events.
* `static`: a continuous effect, active while the source is on the board and its condition holds
  (§2.12). Allowed actions: `buff` (`atk`, `hp`, `move_cost`) and `add_trait`. Allowed selects:
  `self`, `all`, `adjacent`. No `duration`, `amount` expressions or `repeat`.

**Watcher event filters**
* `event_filter: F` on a watcher (scope ≠ `self`). For unit events F applies to the event unit.
  For `on_play` watchers it applies to the played card, with keys `max_cost` / `min_cost` / `tag`
  / `not_tag`.

**Targets**
* `select: "prev"`: the targets of the previous clause of the same card in the same trigger batch
  (§2.8b). Unit targets still on the board, bases, players.
* `select: "adjacent"`, `"of": "self" | "event" | "prev"`, `"position": "both" (default) | "left"
  | "right"`: the unit(s) next to the reference unit in its zone (slot ±1). Side filters still
  apply.
* Filter keys `pinned` (bool) and `damaged`; `tag`/`not_tag` as above.

**Amounts**
* `{"stat": "move_cost", ...}`; `"of": "prev"` (last-known values of the first prev target).
* `{"count": "base_hp", "side"}`.
* `{"event": "damage"}`: the damage the event unit just took (`on_damaged`) or dealt
  (`on_kill`: the lethal hit).

**Conditions**
* `{"type": "prev", "killed"?: bool, "filter"?: F}`: about the previous clause's targets (any
  prev unit target killed / any matching F).
* `{"type": "compare", "left": AM, "op": ">" | ">=" | "<" | "<=" | "==", "right": AM}`.
* `{"type": "history", "event": "operation_played" | "unit_deployed" | "unit_died", "side",
  "window": "turn" | "game", "min"?, "max"?}`, counted from the engine's per-turn and per-game
  counters.

**Branching and repetition**
* `target_condition: F`: evaluated per selected unit target after selection. Matching targets
  receive the action. Non-matching targets receive the `else` body if one is given, otherwise
  nothing.
* `else`: `{"action": A, ...parameters, "target"?: TG}`. It is resolved instead of the effect
  when the (pre-selection) `condition` fails, or for the targets that fail `target_condition`.
  Without its own `target`, it uses the effect's targets (the same ones under `target_condition`).
* `repeat: int ≥ 1` (default 1): the whole clause (selection + action + damage step) runs
  `repeat` times. Random targets are redrawn each time (KARDS "3 damage split at random").

**Action parameters**
* `heal` `"uncapped": true`: a base may exceed `base_hp` (cap 99).
* `gain_coins` accepts a negative amount (coins never below 0).
* `buff` `"move_cost": int` (delta; move cost never below 0), with `duration`.

**Validation additions**
* `prev` and `{"type":"prev"}` need an earlier clause on the same card with the same trigger.
* `static` restrictions as above.
* `else` bodies follow the action/kind table and may not nest `else`.
* `on_attacked` has an event unit (the attacker).


### 1.3 Decks (`decks.json`) and random decks

`decks.json` is unchanged: N ≥ 1 named fixed decks of exactly 40 cards, ≤ 3 copies each, no
tokens. The 4 shipped archetypes keep their order and styles (0 aggro fast, 1 defensive, 2
ranged-heavy, 3 balanced) and include effect cards and operations in their style.

**Random decks** (`cards.generate_deck(rng, config, required=None) -> tuple`): a legal deck
(40 cards, ≤ 3 copies, no tokens), sorted by card index, with a sensible cost curve:
* Target counts per cost bucket: costs 1:4, 2:7, 3:7, 4:6, 5:5, 6:4, 7:3, 8+:4.
* Within a bucket a card is drawn uniformly among the non-token cards with copies left.
* Operations are capped at 12 cards, so at least 28 are units.
* An empty bucket hands its quota to the nearest bucket that has cards left.
* `required` (counts per card index) is placed first. `determinize` uses it to condition on
  revealed cards.
* Deterministic for a given RNG state. `deck_rng(seed, seat) = random.Random(f"deck:{seed}:{seat}")`
  is the per-deal stream.

**Deals** (`cards.sample_deal(seed, config, random_frac) -> (deck0, deck1)`): each seat
independently gets a random deck (`generate_deck(deck_rng(seed, seat))`) with probability
`random_frac`, else a fixed deck index. The choice uses `random.Random(f"deal:{seed}")`. A deck
spec is an `int` (fixed deck index) or a tuple of 40 card indices. Training uses
`random_frac = 0.7`.

### 1.4 Shipped content (asserted by `tests/test_data.py`)

* **Pool size:** 44–56 cards, including the 25 Stage 2 units unchanged (ids and stats) and ≥ 8
  operations and ≥ 1 token.
* **Natures and traits:** every nature × trait set {none, defense, armor, defense+armor} still
  occurs (12 combinations), and some card has each of `blitz`, `smokescreen`, `fury`, `ambush`,
  `shock`, `immune`.
* **Effect coverage:** every trigger (including `static` and `on_attacked`), every action, every
  `select` (including `prev` and `adjacent`), every `side` and every `kind` occurs in some card.
  So do:
  * a filter, an amount expression, a condition, a `"turn"` duration;
  * a scope other than the default, an `event_filter`;
  * `else`/`target_condition`, `repeat`;
  * `tags`.
* **Decks:** the archetype rules of Stage 2 (≥ 50% fast / ≥ 40% Defense / ≥ 40% ranged / each
  nature ≥ 20%, counted over units) still hold. Each fixed deck has ≥ 6 effect cards (operations
  or units with effects).
* **Deck balance gate** (`tests/test_decks.py`, `python tools/deck_balance.py`): lookahead vs
  lookahead deck-confounded `P[a][b] ∈ [0.30, 0.70]` for a ≠ b; draws ≤ 5% per cell; lookahead vs
  random ≥ 0.85 in every unordered cell; mean game length ≤ 30 rounds.

### 1.5 KARDS sample (`research/kards_sample.json`)

A sample of ≥ 60 real KARDS cards (name, kind, stats, a paraphrase of the rules text, source
URL). Each card is encoded in this schema, or marked `"unsupported"` with the missing primitive.
`tests/test_kards_sample.py` parses every encoding with the card loader. The README reports
coverage (fraction expressible) and the most common unsupported patterns, which are candidates for
Stage 4.

`load_ruleset(cards_path=None, decks_path=None, **overrides) -> GameConfig` adds:
* `mulligan=True`: False skips the mulligan phase. Engine tests and scenarios use it.
* `max_effect_events=256`: the loop guard (§2.8).

`build_ruleset(cards, decks, **overrides) -> GameConfig` does the same from in-memory lists of
card dicts and deck dicts, with identical validation. Tests use it to define their own effect
cards without depending on the shipped content.

Parsed effects are frozen dataclasses (`EffectDef`, `TargetDef`, `AmountDef`, `ConditionDef`) on
`CardDef.effects`.

## 2. Rules

Two players (seats 0, 1); bases have 20 HP. Destroying the enemy base wins. If both bases reach
0 HP in the same effect, the game is a **draw** **[clarified]**.

### 2.1 Setup — `reset(seed, decks=None)`
1. `seed`: non-negative int (numpy ints ok; bools/floats/None raise).
2. `decks`: None → `sample_decks(seed, n_decks)` (two fixed decks, Stage 2 behaviour). Otherwise a
   pair of deck specs (§1.3); a tuple deck must be legal (40 cards, ≤ 3 copies, no tokens).
   `deck_ids[p]` = the fixed index, or −1 for a tuple deck. `decklists[p]` = the sorted 40-card
   tuple.
3. `rng = random.Random(seed)`; `first_player = rng.randrange(2)`; shuffle seat 0's deck then
   seat 1's (top = end of list).
4. Opening hands: first player 4, second 5.
5. With `mulligan` on: `phase = MULLIGAN`, first player to act (§2.2). Otherwise round 1 starts at
   once (`_start_turn(first)`), as in Stage 2.

### 2.2 Mulligan
* The first player decides, then the second player. The decider marks opening-hand slots for
  replacement with `MULLIGAN(i)`: legal for unmarked slots i < len(hand), and a mark cannot be
  undone. `CONFIRM` (always legal in this phase) finishes.
* On `CONFIRM` with k ≥ 1 marked cards:
  1. Draw k cards from the deck.
  2. Put the marked cards into the deck and shuffle it (engine RNG).
  3. Re-sort the hand.
  **[clarified: draw first, so a replaced card is never redrawn.]** With k = 0 nothing is drawn
  and the RNG is not used.
* After the second player confirms, round 1 starts with the first player's turn (§2.3).
* Hidden: the opponent sees nothing about the mulligan. Hand and deck sizes do not change; marks
  and replacements are private.

### 2.3 Turns
* **Turn start (p):**
  1. `turn += 1`.
  2. `coins[p] = coins_for_round(round) + coin_bonus[p]` (≥ 0).
  3. Draw 1 (empty deck: none; hand at 10: burned).
  4. `start_of_turn` triggers resolve (§2.8).
* **END_TURN (p):**
  1. `end_of_turn` triggers resolve.
  2. Every `"turn"` duration on every unit expires (§2.10).
  3. `coins[p] = 0`.
  4. p's units refresh their flags (`summoned = moved = attacked = False`, `attacks = 0`).
  5. Pins whose expiry has come are lifted (§2.4).
  6. The next player's turn starts. A new round starts when the opponent is the first player; at
     round 50 the game is instead a draw.
  If the game ends during step 1, the remaining steps are skipped.

### 2.4 Units, keywords and action economy
A unit is created from its card with `atk`, `hp = max_hp`, `armor`, the boolean keywords and
`nature`, plus a unique `uid` and `summoned = True`. Effective values live in the unit's fields.
Temporary (`"turn"`) changes are also recorded in `temp_*` fields so they can be undone.

| | can_move | can_attack |
|---|---|---|
| troop, ranged | `ready and not moved and attacks == 0` | `ready and not moved and attacks < max_attacks` |
| fast | `ready and not moved` | `ready and attacks < max_attacks` |

* `ready = not pinned and (not summoned or blitz)`.
* `max_attacks = 2 if fury else 1`.
* `attacked = attacks > 0` (kept for Stage 2 code).

**Keywords:**
* **Blitz** — the unit may act on the turn it is deployed or summoned.
* **Fury** — it may attack twice per turn. Each attack is an `ATTACK` action.
* **Smokescreen** — enemy `ATTACK`s cannot target it. It is lost when the unit moves or attacks.
  Effects (including `chosen` ones) can still target it.
* **Defense** — if the targeted zone holds an attackable (non-smokescreen) Defense unit, the
  attack must target one. Effects ignore Defense.
* **Armor X** — reduces combat damage taken, both attacking and defending. It does not reduce
  effect damage **[clarified: as KARDS Heavy Armor]**.

**Pin** (the `pin` action) — a pinned unit cannot move or attack. The pin is lifted at the end of
its controller's next turn: at END_TURN of turn index `pin_until`, where `pin_until` = the next
turn of the unit's owner (`turn + 1` if pinned during the opponent's turn, else `turn + 2`).
Pinning a pinned unit refreshes `pin_until`.

### 2.5 Board
As Stage 2: zones `backline[0]`, `frontline` (shared), `backline[1]`, at most 5 units each.
`front_owner` = owner of the frontline units or `None`.

### 2.6 Actions (current player p, opponent o)
* `END_TURN` — legal in phase MAIN.
* `PLAY(i)` — `i < len(hand)`, `cost ≤ coins`, phase MAIN.
  * **Unit:** needs `len(backline[p]) < 5`. Pay, deploy to `backline[p]`, then its `on_deploy`
    triggers resolve.
  * **Operation:** legal only if, for each of its `chosen` effects whose condition holds now,
    at least one legal option exists. Pay, put the card in p's discard pile, then its `on_play`
    effects resolve. Operations never occupy the board.
* `MOVE(j)` — as Stage 2 (can_move, frontline free or own, < 5 units, `move_cost ≤ coins`). The
  unit loses smokescreen, then `on_move` triggers resolve.
* `ATTACK(a, t)` — as Stage 2 (reach, Defense, `t = 2Z` = base), plus smokescreened targets are
  excluded. Sequence:
  1. `attacks += 1`; the attacker loses smokescreen.
  2. `on_attack` triggers resolve; the event unit is the target, or None for the base.
  3. If the game ended, or the attacker or target unit left the board, the attack ends.
  4. Base: `base_hp[o] -= atk`. Unit: simultaneous combat damage via `engine.combat_damage` (armor
     applies, ranged attackers take no return damage), then the damage step (§2.8).
* `CHOOSE(t)` — phase CHOICE, t in the pending options. The pending effect resolves on t and
  the queue continues.
* `MULLIGAN(i)`, `CONFIRM` — phase MULLIGAN (§2.2).

Phases: `MULLIGAN` (only MULLIGAN/CONFIRM legal), `MAIN`, `CHOICE` (only CHOOSE legal; the
player to act is the turn player).

### 2.7 Reach (unchanged)
* **Ranged:** any enemy unit (either zone) or the base, from either zone.
* **Troop/fast in the backline:** enemy frontline units only.
* **Troop/fast in the frontline:** enemy backline units or the base.

### 2.8 Effect resolution (event queue)
* **Instances.** A triggered effect becomes an *instance*: (effect, card, source uid or None,
  controller, event unit uid or None, last-known views of source and event unit).
* **The queue.** Instances wait in one FIFO queue and are resolved one at a time until the queue
  is empty, the game is over, or a choice is pending.
* **Listener order.** For an event about unit X (or operation card X), X's own matching effects
  (scope `self`, card order) are enqueued first. Then watchers: other units on the board whose
  effect has this trigger and a scope matching X's side relative to the watcher.
  * Watchers are taken in board order: the turn player's backline, then frontline (if theirs),
    then the opponent's backline, then frontline.
  * Turn triggers enqueue the matching effects of all units in the same board order.
* **Resolving an instance:**
  1. Check its condition (skip if false).
  2. Select targets (§2.9). No target → the instance fizzles. `chosen` → pending; the instance
     resumes at CHOOSE.
  3. Evaluate the amount.
  4. Apply the action.
  5. Run the **damage/death step**.
* **Damage step.** Damage from one effect (or one combat) is applied simultaneously.
  1. A base at ≤ 0 HP ends the game (both at once: draw); the queue and any pending choice are
     discarded.
  2. Dead units (hp ≤ 0, or destroyed) are removed in board order (zones compact; an emptied
     frontline gets `front_owner = None`).
  3. Then these are enqueued, in this order:
     * `on_damaged` for each damaged survivor (combat: target, then attacker), with event unit =
       the combat opponent or the effect's source unit;
     * `on_kill` for each unit that dealt lethal combat damage (event = the victim);
     * `on_death` for each removed unit, in removal order (watchers must still be on the board).
* **Event units:**

  | trigger | event unit |
  |---|---|
  | on_attack | the attack target |
  | on_damaged | the damage source unit |
  | on_kill | the victim |
  | watchers (deploy, death, attack, damaged, move, kill) | the unit the event is about |

  A unit action on an event or source unit that has left the board fizzles. Amounts read the
  current values while the unit is on the board, else its last-known values.
* **Loop guard.** At most `max_effect_events` (256) instances resolve per action. The rest of the
  queue is then discarded and `guard_trips += 1`. This is deterministic.
* **Randomness** uses only the game RNG `self.rng`, in a fixed order.

#### 2.8b Clause batches
All of one card's effects fired by the same event form a **batch**, in card order. The batch
records each resolved clause's targets (after `target_condition`/`else` split, including fizzled
clauses as an empty target list). `prev` reads the most recent earlier clause of the batch.


### 2.9 Targets and choices
* **Units** are listed in board order; the side and zone filters apply.
* **Bases:** `side any` = both, in the order controller's base, opponent's base.
* **Players:** `side any` = both, in the order controller, opponent.
* **`random`:**
  * if ≤ `count` candidates, all of them (no RNG draw);
  * else `rng.sample(candidates, count)` (sample order).
* **`chosen`:** the options are encoded as **CHOOSE slots** (§3).
  * No options → fizzle. Otherwise phase becomes CHOICE with `pending` set (even with a single
    option).
  * The canonical slots are `t < Z` enemy backline j, `Z ≤ t < 2Z` frontline j (either owner),
    `t = 2Z` enemy base, `2Z+1 ≤ t < 3Z+1` own backline j, `t = 3Z+1` own base. The first 2Z+1
    slots coincide with the ATTACK target slots.
* **`self` / `event`:** the unit if it is still on the board (by uid), else fizzle.

### 2.10 Action semantics
* **damage:** unit `hp -= amount` (no armor); base `base_hp -= amount`; then the damage step.
* **heal:** unit `hp = min(max_hp, hp + amount)`; base `min(config.base_hp, hp + amount)`;
  `"full"` heals to the cap.
* **buff:** `atk = max(0, atk + a)`, `max_hp += h`, `hp += h`.
  * `"turn"` records the change actually applied: `temp_atk += (new atk − old atk)` (after the
    clamp) and `temp_hp += h`.
  * At expiry `atk -= temp_atk` and `max_hp -= temp_hp`; `hp = min(hp, max_hp)` but at least 1
    (expiry never kills).
* **destroy:** the unit dies in the damage step (no damage is dealt; `on_death` fires).
* **draw:** n cards for the target player (burn rule; empty deck: nothing).
* **gain_coins:** `coins[p] += n`. They are lost at END_TURN; coins gained on the opponent's turn
  are lost at their own turn start.
* **increase_max_coins:** `coin_bonus[p] += n`. It changes income from p's next turn start;
  income is never below 0.
* **add_trait:**
  * Defense, blitz, smokescreen and fury become true; armor goes up by `amount`.
  * `"turn"` grants are undone at expiry, unless the trait was also granted permanently in the
    meantime.
* **remove_trait:**
  * Defense, blitz, smokescreen and fury become false; armor goes to 0.
  * `"turn"` removals are restored at expiry, and the unit then has the traits it would have had.
* **summon:** `amount` copies of the token unit go into the target player's backline, while there
  is space. They are `summoned` (cannot act unless blitz). `on_deploy` does not fire.
* **return_to_hand:**
  * The unit leaves the board (no `on_death`) and its card goes to its owner's hand. A full hand
    burns the card.
  * All modifications are lost.
  * The card becomes *known* to the opponent (§5).
* **pin:** §2.4.
* **discard:** n random cards from the target player's hand (`rng.sample`) go to that player's
  discard pile and become public.
* **add_card:** `amount` copies of the token card go into the target player's hand (burn rule).
  They are known to the opponent.
* **retreat:**
  * A frontline unit goes to its owner's backline, or to the owner's hand if the backline is full.
    A backline unit goes to its owner's hand.
  * No `on_death`. The frontline is updated as for deaths.

### 2.11 Required rule tests (hand-built positions)

**Stage 2:** the Stage 2 list (natures, Defense, armor, move cost, frontline control) still holds,
and the Stage 2 reference model still matches the engine on vanilla-only decks with
`mulligan=False`.

**Effects:**
* Every trigger, action, `select`, `side`, `kind`, `zone`, filter key, amount form, condition
  type and duration has a test with an exact expected state.
* Watcher scopes `friendly` / `enemy` / `any`, and "not the source".
* The §1.2b extensions:
  * `tags` filters; `ambush`, `shock` and `immune` in combat and with effects;
  * `on_attacked`, `event_filter`, `prev` / `adjacent` selects and the `prev` condition;
  * `compare` and `history` conditions, `target_condition` + `else`, `repeat`;
  * uncapped base heal, negative `gain_coins`, `buff.move_cost`.
* Static effects (§2.12): auras on/off as the source arrives and leaves; conditions toggling
  (e.g. "while damaged"); an hp aura lost without killing; stacking; static traits.
* Chained triggers:
  * an `on_death` that kills another unit whose `on_death` fires;
  * simultaneous deaths in board order;
  * a game that ends mid-chain;
  * the loop guard (two units that keep triggering each other stop after 256 instances with
    `guard_trips` 1).

**Phases and costs:**
* Choices: the pending state, the options, fizzle with no options, two choices in one chain.
  Operation playability requires options.
* Mulligan: marks, CONFIRM draws then shuffles, k = 0 leaves the RNG untouched, order first →
  second, round 1 starts after.

**Keywords:**
* blitz, fury, smokescreen and pin, including pin expiry on both players' turns.
* Effect damage ignores armor and Defense.
* `"turn"` buffs and traits expire at END_TURN, and expiry does not kill.

**Determinism and determinize:**
* RNG determinism: same seed, decks and actions ⇒ identical states, including random effects.
* `determinize` (§4).

### 2.12 Static (continuous) effects
* **Contributions.** Each unit carries its current static contributions (`static_atk`,
  `static_hp`, `static_move_cost`, `static_traits` bitmask).
* **Recompute.** After every action and after every resolved instance (and after each damage
  step), the engine recomputes all contributions from scratch:
  * every `static` effect of every unit on the board whose condition holds adds to its targets,
    in board order;
  * the deltas are applied to the effective fields: atk; max_hp; hp (an increase raises hp, a
    decrease caps hp at the new max_hp, never below 1); move_cost; traits = base ∪ temp ∪ static.
* **Fast path.** It is skipped when no static source is on the board.
* **Smokescreen exception.** Smokescreen granted by a static effect is not lost by moving or
  attacking while the source still grants it **[clarified]**.


### 2.13 Resolved details [clarified]

These details are settled by the engine and its independent tests.

**Board order and targets**
* "Board order" is the turn player's backline, then the frontline if theirs, then the opponent's
  backline, then the frontline if theirs. It applies to death removal, watchers, and
  `all`/`random` candidate lists.
* `unit_or_base` candidates: units first, then bases (controller's, then opponent's).
* `{"count": "hand" | "coins" | "deck", "side": "any"}` sums both players.

**Triggers and combat**
* A unit's own (scope `self`) effects fire even after it left the board: `on_death`, and
  `on_kill` of a killer that also died. Watchers must be on the board.
* `on_kill`: scopes are matched against the killer's side, and the event unit is the victim. The
  attacker's kill fires before the defender's.
* Combat waits until every `on_attack` chain has resolved. A loop-guard trip does not cancel the
  combat.
* Last-known values: a dead unit's last-known `hp` is its value at removal (≤ 0).
* `source_zone` is false when the source is not on the board.

**Loop guard**
* It counts every instance taken off the queue in one action, including skips and fizzles.
* It trips at most once per action.
* END_TURN and the next turn's start share one budget.

**Game end, damage and cards**
* A game that ends in a damage step leaves its dead units on the board in the final state.
* Effect damage of 0 does not fire `on_damaged`. Healing never lowers hp.
* `discard` samples hand slot positions (`rng.sample`). With n or fewer cards in hand, it takes
  all of them without an RNG draw.
* `retreat` to hand is bookkept like `return_to_hand`. A card returned, retreated or added to a
  full hand burns, with no known-card change.
* Token cards played or discarded: `known_hand` decrements if known; `revealed` never changes.
* Operation cards have nature −1 and attack/health/move cost 0.

**Mulligan and turns**
* The marked cards are appended to the deck in slot order, then the deck is shuffled.
* With a deck shorter than k, only the first marks in slot order are replaced.
* `turn` is 0 during the mulligan; the first turn is 1.

**`determinize`**
* The own deck is sorted before shuffling, so the result cannot depend on the true order.
* `seed` is None on the result.
* The opponent's marks are cleared only if the opponent is the one deciding.

**Deck specs and generation**
* A deck spec may be any non-string sequence of 40 card indices.
* `generate_deck`:
  * Cost 0 counts as cost 1.
  * Required cards count toward their bucket. An over-full bucket takes the excess from the
    nearest bucket that still has quota (ties: the cheaper bucket), and the empty-bucket hand-off
    breaks ties the same way.
  * Copies are drawn one at a time, uniformly over the eligible cards in index order.
  * Required operations count toward the cap of 12.
  * Other deck sizes scale the bucket targets.
* `sample_deal` draws `random()` then `randrange(n_decks)` for seat 0, then the same for seat 1,
  always both.

**Phase 1b details (extensions)**

*Event units:* `on_kill` watchers get the victim; `on_attacked` (self and watchers) gets the
attacker. These override the "unit the event is about" row of the §2.8 table.

*Combat keywords:*
* **Ambush** fires only if the attacker would take > 0 return damage, so not against ranged,
  shock or immune attackers, or armor above the ambusher's attack. It is spent even when its
  strike does not kill. `ambush_ready` = ambush and not yet used this turn.
* **`on_attacked`** fires once the whole `on_attack` chain is done, only if both units are still
  on the board and the game continues. It never fires for base attacks.

*Events and `prev`:*
* `event_filter` is checked when the event fires, against the event unit (current or last-known
  values). It is rejected on turn triggers.
* **What a clause records** for `prev`: the targets that received the action or the `else` body,
  without duplicates, combined over `repeat` iterations.
  * `killed` = one of them died in that clause's own damage steps.
  * A skipped or fizzled clause records nothing. A bare `{"type": "prev"}` holds if the previous
    clause hit anything.
* **A `prev` target** keeps the units still on the board plus bases/players. `kind` acts as a
  filter on them (default: everything the action can take), and `side` is not allowed.
* **`adjacent` with `"of": "prev"`** uses the first prev unit still on the board; `side` defaults
  to `any`.
* **An amount with `"of": "prev"`** reads the first prev unit (live or last-known), or 0.
* **`{"event": "damage"}`** is only valid on `on_damaged` (damage taken) and `on_kill` (the full
  lethal hit, overkill included).
* **Last-known `hp`** is the actual value at removal (a destroyed unit keeps its hp).

*`else` and `repeat`:*
* When the condition fails, the `else` body runs once (`repeat` applies only to the main body).
* Without its own target, the `else` body is selected with the effect's target.
* Operation playability checks whichever body would resolve now.

*Rejected at load:*
* `repeat` with a `chosen` target;
* a `chosen` `else` target under `target_condition`;
* `uncapped` with `"full"`;
* static `add_trait armor` (there is no static armor);
* `adjacent` of a dead `on_death` source;
* static `adjacent` other than `"of": "self"`.

*`buff`:* it needs at least one of `atk`/`hp`/`move_cost`. A `"turn"` move-cost buff is recorded
in `temp_move_cost` (a Unit field).

*Static effects:*
* **One pass:** conditions and filters read the state before the recompute; there is no
  fixed-point iteration. Static effects may use `else`/`target_condition`. Nothing is recomputed
  after the game is over.
* **When:** contributions are refreshed before every instance, before combat, after every damage
  step, after every action, and in `invalidate()`.
* **Layering:** buffs and expiry change the unit's own value (field minus `static_*`, clamped at
  0); auras sit on top. A static grant wins over a `"turn"` removal, and `remove_trait` does not
  remove a static grant.

*Other:*
* `heal` with `"uncapped"` affects bases only (cap max(99, base_hp)).
* `{"count": "base_hp", "side": "any"}` sums both bases.
* `history`: `min` defaults as for `control`. Turn counters reset at every turn start.
  `unit_deployed` counts units played from hand, not summons.
* PendingView `amount` for `add_trait` armor is the armor amount.
* Known limitation: a KARDS "target and adjacent units" splash cannot reach the neighbours once
  the main target has died (the reference unit has left the board). Deferred.

**Engine review details**
* END_TURN(p) step 4 also clears `summoned` on the opponent's units created during p's turn
  (effect pools only). Otherwise a unit summoned for the opponent would lose its owner's whole
  next turn.
* Statics are recomputed just before a deploy, move or attack event fires, so event filters see
  current auras. `on_attacked` adds a second pre-combat recompute only if it queued something.
* A `target_condition` split resolves as **one** damage step:
  1. both bodies' targets and amounts are evaluated first;
  2. both actions are applied;
  3. a unit hit by both bodies gets one `on_damaged` with the total damage.
* `determinize` sets `num_steps = 0` on the result and drops engine caches derived from hidden
  state.

### Deferred to Stage 4 (reported, not implemented)
* Countermeasures (hidden armed cards firing on the opponent's turn, cancelling effects).
* Cost modifiers for cards in hand.
* Delayed/granted effects and durations beyond "turn".
* Copies and random card generation, transform/Veteran.
* Deck manipulation (dig/shuffle/tutor), damage modifiers.
* Take control / remove, choose-one modes, set_stat.
* Global rule modifiers, Intel/Covert, Develop, Forecast.

## 3. Action space (`cardgame.actions.ActionSpace`, H = 10, Z = 5)

Stage 2 indices are unchanged; new blocks are appended:
* `0` END_TURN.
* `1+i` PLAY(i), i < H.
* `MOVE0 + j` MOVE(j), j < Z (`MOVE0 = 1+H`).
* `ATTACK0 + a·(2Z+1) + t` ATTACK(a, t) (`ATTACK0 = 1+H+Z`; `attack(a, t)` helper).
* `CHOOSE0 + t` CHOOSE(t), t < 3Z+2 (`CHOOSE0 = ATTACK0 + 2Z(2Z+1)`).
* `MULLIGAN0 + i` MULLIGAN(i), i < H.
* `CONFIRM`.

`N = 154` (CHOOSE0 = 126, MULLIGAN0 = 143, CONFIRM = 153).
* `ActionKind`: `END_TURN, PLAY, MOVE, ATTACK, CHOOSE, MULLIGAN, CONFIRM`.
* `decode`, `encode`, `describe`.
* `n_choose = 3Z+2`, `ENEMY_BASE_CHOICE = 2Z`, `OWN_BASE_CHOICE = 3Z+1`.

## 4. Engine API (`cardgame.engine.Game`)

```python
game = Game(config=None)
game.reset(seed, decks=None); game.current_player(); game.legal_actions(); game.legal_mask(out=None)
game.step(a); game.observe(player); game.winner(); game.clone(); game.render(); game.invalidate()
game.determinize(player, rng) -> Game     # NEW
game.phase  # MULLIGAN=0, MAIN=1, CHOICE=2 (module constants)
```

**State** (Stage 2 fields plus):
* `phase`, `turn`, `mulligan_marks` (set of slots), `mulligan_done[2]`, `pending`.
* `queue` (empty between actions unless a choice is pending).
* `coin_bonus[2]`, `decklists[2]`, `discard[2]` (counts per card index).
* `graveyard[2]` (dead units per card index, tokens included).
* `known_hand[2]` (counts of cards in p's hand known to the opponent).
* `revealed[2]` (§5), `guard_trips`.

**Unit fields** (keyword-only, immutable scalars so copies stay shallow):
* Stage 2 fields.
* `blitz`, `smokescreen`, `fury`, `pinned`, `pin_until`, `attacks`.
* `temp_atk`, `temp_hp`, `temp_armor`, `temp_traits` / `temp_removed` (int bitmasks).
* `base_traits` (int bitmask), `token`, `temp_move_cost`.
* §1.2b additions: `ambush`, `shock`, `immune`, `ambush_used`, and `static_atk`, `static_hp`,
  `static_move_cost`, `static_traits`.

**Clone and determinism:**
* `clone()` stays cheap and independent (queue and pending copied; per-card tables shared).
* Determinism: same seed (+ same decks) and the same actions ⇒ identical states.

**`determinize(player, rng)`** returns a clone in which everything hidden from `player` is
resampled consistently with what `player` knows:
1. **The opponent's decklist** is redrawn with `generate_deck(rng, config, required=revealed[o])`.
   `revealed[o]` holds the non-token copies `player` has seen come from o's deck.
2. **The opponent's unknown cards** are the redrawn decklist minus `revealed[o]`. They are shuffled
   and dealt to the unknown part of o's hand (its size minus the `known_hand[o]` cards, which stay)
   and to o's deck (same size). Burned cards are counts only.
3. **Own deck:** `player`'s deck keeps its contents and is reshuffled.
4. **Hidden flags:** o's mulligan marks are cleared, and `deck_ids[o] = −1`.
5. **The RNG:** the game RNG is replaced by `random.Random(rng.getrandbits(64))`, so future random
   effects are not predictable from the real RNG.

Guarantees, tested:
* `observe(player)` and (on `player`'s decision) the legal actions are identical in the original
  and the result.
* The result depends only on `player`'s information: perturbing o's hidden hand/deck/RNG gives an
  identical determinization for the same `rng` state.
* Every determinized decklist is legal and contains `revealed[o]`.
* Different `rng` states give different hidden parts.

## 5. Observation (egocentric NamedTuple)

```python
UnitView(card, atk, hp, max_hp, armor, defense, nature, move_cost, summoned, moved, attacked,
         can_move, can_attack, blitz, smokescreen, fury, pinned, attacks, temp_atk, temp_hp, token,
         ambush, shock, immune, ambush_ready, pin_turns, temp_move_cost, temp_traits, temp_removed)
PendingView(card, effect, action, amount, select, side, kind, atk, hp, previews)   # the effect waiting for CHOOSE
Observation(<Stage 2 fields>, phase, mulligan_marks, pending, turn, my_coin_bonus, opp_coin_bonus,
            my_decklist, my_deck_counts, my_known_hand, opp_known_hand, opp_revealed, my_discard,
            opp_discard, my_graveyard, opp_graveyard, my_burned, opp_burned, my_history, opp_history)
```

**New fields:**
* **Stage 2 fields** keep their positions; new fields are appended with defaults.
* **`mulligan_marks`:** the observer's own marks (a tuple of bools per hand slot during their
  mulligan, else `()`).
* **`pending`:** public, since the played card is revealed.
  * `amount`, `atk` and `hp` are resolved when the choice opens. `atk`/`hp` are the buff values,
    0 for other actions.
  * `amount` is −1 for heal `"full"` and 0 where the action has no amount.
  * `action`/`select`/`side`/`kind` are the JSON names.
* **Counts per card index (tuples of length n_cards):**
  * `my_decklist`, and `my_deck_counts` (remaining deck contents, order hidden);
  * `opp_known_hand` (cards the observer knows are in the opponent's hand);
  * `my_known_hand` (the observer's cards the opponent knows about);
  * `opp_revealed` (non-token copies seen from the opponent's deck);
  * discard piles (operations played plus discarded cards);
  * graveyards.
* **`my_burned` / `opp_burned`:** counts (the identities are lost). The count is public because
  hand and deck sizes reveal it.

* **`PendingView.previews`:** one `(kills, dealt, healed, other)` tuple per CHOOSE slot (3Z+2,
  zeros for illegal slots).
  * The engine computes it with the body that would actually apply to that option, so
    `target_condition`/`else`, immune and the resolved amounts are respected.
  * The encoder copies it; no rule is re-derived.
  * `kills` = the option would die. `dealt` = damage. `healed` = hp gained. `other` = the atk + hp
    change for buffs, else 1 for any other action that applies.
* **`my_history` / `opp_history`:** public counters, `(operations_played, units_deployed,
  units_died)` for this turn, followed by the same three for the game.
* **UnitView additions:**
  * `pin_turns`: the number of its owner's turns the pin still covers (0 = not pinned).
  * `temp_move_cost` and the bitmasks `temp_traits` / `temp_removed`: the trait changes that
    lapse at the end of this turn.

**`revealed[p]` bookkeeping** (also used by `determinize`):
* When p plays or discards a non-token card c: if `known_hand[p][c] > 0`, decrement it (the known
  copy left the hand); else `revealed[p][c] += 1` (a newly seen copy).
* `return_to_hand` of p's non-token unit: `known_hand[p][c] += 1`.
* Added token cards: `known_hand += 1`; tokens never enter `revealed`.

**Hidden:**
* The opponent's hand except `known_hand`, and both decks' order.
* The opponent's deck contents beyond `revealed`, and the opponent's deck choice.
* Mulligan marks and replacements, burned identities, and the RNG.

**No-leak tests** also replace the opponent's unknown hand cards, deck contents/order, mulligan
marks and RNG state with other legal values (same sizes, same `revealed`/`known_hand`).
`observe(p)`, its encoding and p's legal actions must not change. Covered states: positions
before and after the opponent's mulligan, after the opponent's draw effects, and before and after
random effects.

## 6. Encoding (`cardgame.features.ObservationEncoder`, `ENCODER_VERSION = 5`)

`encode(obs, mask=None)`, `encode_into`, `encode_batch`, `split(x)`, `dim`, `layout()`.
* `mask` is the observer's legal mask (own decision) or None. The encoder reads only the
  Observation and the mask.
* Flat float32 vector: `[globals G | token features T·F | token ids T | token present T |
  deck counts n_cards | attack previews 2Z·(2Z+1)·4 | choose previews (3Z+2)·4]`.

**Tokens** (T = 50, fixed order):
* global (1), hand (H), my backline (Z), frontline (Z), opponent backline (Z);
* my base, opponent base;
* pending (1, present in phase CHOICE);
* opponent-revealed cards (R = 20: unique card indices with any public presence, i.e. revealed,
  known in hand, in the graveyard or discard, ascending index, truncated to R);
* own deck summary (1).

The id is the card index + 1 (0 = none; global/base/deck tokens use 0).

**Token features** (F, one schema; unused entries 0):
* **Token type:** one-hot (global, hand, my_back, front_mine, front_opp, opp_back, my_base,
  opp_base, pending, revealed, deck).
* **Unit state:** atk, hp, max_hp, armor, defense, blitz, smokescreen, fury, pinned, summoned,
  moved, attacks/2, can_move, can_attack, temp_atk, temp_hp, move_cost, damaged, token.
* **Hand:** playable, mulligan_marked, mulligan_legal, known_to_opp.
* **Mask hints:** move_legal, attack_ready, attack_targetable, incoming_atk, choose_legal.
* **Base:** hp, attack_targetable, choose_legal, incoming_atk.
* **Revealed:** revealed, known_in_hand, graveyard, discard, on_board.
* **Pending:** effect index one-hot (3), amount.

**Globals** (G):
* is_my_turn, went_first, round, turn, phase one-hot (3);
* coins ×2, coin_bonus ×2, base HP ×2, hand sizes ×2, deck sizes ×2, burned ×2;
* front owner one-hot ×3, n_legal/N, base_damage_ready.

**Static card table** (`layout["card_table"]`, n_cards × S, looked up inside the network by id):
* type (unit/operation), token, cost, atk, hp, move_cost, nature one-hot, the eight traits
  (§2.4, §1.2b), tag one-hots over the pool's tag vocabulary (stored in the layout);
* for each of 3 effect slots: present, trigger one-hot, scope one-hot, action one-hot, select
  one-hot, side one-hot, kind one-hot, zone one-hot, literal amount (expression → 0 + is_expr
  flag), buff atk/hp, duration_turn, has_condition, has_filter, trait one-hot.

**Previews** (engine rules, not re-derived):
* **Attack previews:** for each legal ATTACK, (kills, attacker_dies, dealt, taken) from
  `combat_damage`.
* **Choose previews:** `PendingView.previews` (computed by the engine, §5), copied for the
  legal CHOOSE slots.

**Scales and fingerprint:**
* Fixed scales as Stage 2 (cost/8, atk/10, hp/10, move_cost/4, armor/4, coins/10, round/50,
  base/20, hand/10, deck/40, counts/3), plus turn/100.
* `pool_fingerprint` covers the card table, decks, rule constants, scales and `ENCODER_VERSION`.

**Resolved encoder details [clarified]** (names and order are the tuples in `features.py`; `layout()` carries
every size, offset, group range, action offset and slot→token map, plus the card table and tag vocabulary):
* **Sizes** (H = 10, Z = 5): G = 36, T = 50, F = 66, S = 228 + n_tags, dim = 3944 + n_cards (3999 for the
  shipped 55-card pool).
* **Token schema** (`TOKEN_FEATURES`): type one-hot (11); unit state (33) = the list above plus the §5 fields
  `ambush`, `shock`, `immune`, `ambush_ready`, `pin_turns` (/2), `temp_move_cost` (/4) and eight signed
  `lapse_<trait>` columns (+1 granted for this turn, −1 removed for this turn; a `"turn"` armor change sets the
  armor bit by its sign); hand (4); hints (5); `base_hp`; revealed (5); pending (7) =
  `effect0..2`, `pending_amount`, `pending_full` (heal "full"), `pending_atk`, `pending_hp`. Base tokens reuse
  `attack_targetable` / `incoming_atk` / `choose_legal`.
* **Presence:** the global, both base and the deck tokens are always present; pending iff `obs.pending`. Absent
  tokens are all zero (features, id). `card_table[c]` is token id c + 1.
* **Values:** `incoming_atk` = combined atk of my units with a legal ATTACK on the token, /10 (units) or /20
  (base). `known_to_opp` marks the first `my_known_hand[c]` hand slots holding c. Revealed counts /3; `on_board` =
  the opponent's units of that card on the board /3 (not a selection criterion). The deck-counts block holds the
  raw remaining counts (`my_deck_counts`); the network divides by 40. coin_bonus /10, burned /10, attacks /2,
  amounts and buff atk/hp /10, buff move_cost /4, `repeat` and random `count` /3.
* **Globals:** the 24 listed above, then 12 history globals (`my_history` and `opp_history`: this-turn counts /5,
  this-game counts /20).
* **Previews:** a base ATTACK previews (atk ≥ opp base hp, 0, atk/10, 0).
* **CHOOSE previews** are copied from `PendingView.previews`: kills as-is, the other three /10. The engine
  previews each option with the body that would actually apply to it (`target_condition`/`else`, immune, the
  resolved amount):
  * damage: dealt = amount, 0 on an immune unit; kills = dealt > 0 and dealt ≥ hp, bases included;
  * destroy: kills = 1;
  * heal: healed = the hp actually gained;
  * buff: other = the applied atk change + hp change;
  * any other applicable action: other = 1;
  * an option reached only by a random `else` target: zeros.
* A mask that names an empty slot raises `ValueError`.
* **Card table:** `CARD_FEATURES` (18), then 3 × `EFFECT_FEATURES` (70: the list above plus `amount_full`,
  `uncapped`, `buff_move_cost`, `has_else`, `has_event_filter`, `repeat`, `count`), then one `tag_<name>`
  column per sorted pool tag. `has_filter` = target filter or `target_condition`. The effect vocabulary is
  fixed in `features.py`; a card outside it raises `ValueError` when the encoder is built.
* **Fingerprint:** also covers every `CardDef` (so effect details the table does not show count), the schemas
  and H/Z/base_hp/max_rounds/coin_cap/opening_hand/deck_size/max_copies. It ignores `mulligan` and
  `max_effect_events`, so a checkpoint trained with the mulligan loads for mulligan-free scenarios.

## 7. Model (`cardgame.rl.network`)

* **`TransformerPolicyNet(layout, d_model=128, layers=3, heads=4, ff=256, belief=True,
  privileged_critic=False, shared_trunk=False)`** — the Stage 3 default. 2–4 layers, no positional
  encoding.
  * **Token input:** each token's input is `MLP([card_table[id], features, Embedding(id)])`.
  * **Global and deck tokens:** the global token is `Linear(globals)`. The deck token is
    `Linear(Σ_c counts_c · cardvec_c / 40)`, where `cardvec` is the same card MLP applied to every
    card in the pool.
  * **Trunk:** a pre-norm `nn.TransformerEncoder` with a key-padding mask.
  * **Towers:** separate policy and value towers (`shared_trunk` shares them; it cannot be
    combined with `privileged_critic`).
* **Policy (pointer-style, one scorer for every action type).** `score(src, tgt, type) =
  q_type(src, g) · k(tgt) / √k + w · ReLU(A src + T tgt + C g + E_type + P preview) + b(tgt)`,
  where g is the global token output.

  | action | source | target |
  |---|---|---|
  | ATTACK(a, t) | attacker token | target token (opponent backline, frontline, opponent base) |
  | CHOOSE(t) | pending token | the 3Z+2 choice tokens |
  | PLAY(i) | hand token | learned null target of that type |
  | MOVE(j) | backline token | learned null target of that type |
  | MULLIGAN(i) | hand token | learned null target of that type |
  | END_TURN, CONFIRM | global token | learned null target of that type |

  Illegal actions get logit −1e9.
* **Value** = `MLP(global output)` of the value tower. With `privileged_critic`, the value tower's
  global token also receives `Linear(Σ_c opp_hand_c · cardvec_c / 10)` (the opponent's true hand
  counts). The actor never sees them.
* **Belief head** (`belief=True`) = `Linear(global output of the policy tower) → n_cards`, a
  multi-label prediction of which cards are in the opponent's hand. It is trained with BCE against
  engine labels (`count > 0`).
* **`PooledPolicyNet(layout, d_model=128, ctx_dim=256, belief=True, ...)`** — the Stage 2 pooled
  encoder kept as the baseline.
  * Token embeddings come from the same token MLP. Masked mean/max/sum/fill pooling per group
    feeds a context MLP in place of attention.
  * The heads are the same, with the context as the global vector.
* **`PolicyValueNet`** (flat MLP) stays as a sanity baseline.
* **Interface:**
  * `policy_logits(x, mask)`, `value(x, priv=None)`, `belief_logits(x)`;
  * `act(x, mask, deterministic, generator)`;
  * `evaluate(x, mask, actions, priv=None) -> (logp, entropy, value, belief_logits)`;
  * `spec()`; `build_net(spec)` dispatches on `kind` (`"transformer"`, `"pooled"`, `"mlp"`).
  * Stage 2 checkpoints raise `CheckpointError` (encoder version / fingerprint).
* **Acting** (workers, snapshots, quick eval, `PPOAgent`) uses a CPU copy under
  `torch.inference_mode()`, except that rollout workers in server mode get their logits from the
  learner device (§8.1).

**Resolved network details [clarified]**
* **Card vectors.** A card MLP maps `[card_table[c], Embedding(c+1)]` to a card vector once per
  forward for every card (id 0 → zeros). The token MLP reads `[cardvec[id], features]`. The same
  card vectors build the deck token (`counts @ cardvec / 40`) and the privileged input
  (`priv @ cardvec / 10`). The global and deck terms are added to their token rows.
* **Tokens.**
  * The global, base and deck tokens are forced present.
  * Absent tokens are key-padded and their outputs zeroed.
  * Each row's present tokens are packed to the front and the batch is cut to its longest row,
    which is exact because there is no positional encoding.
  * There is a final LayerNorm.
* **Towers.** Separate towers share no parameters (embeddings included).
* **Scorer.**
  * `key_dim = d_model`, `pair_dim = 32`, additive type embeddings.
  * One learned null target per type (END_TURN, PLAY, MOVE, MULLIGAN, CONFIRM).
  * Separate attack/choose preview maps.
  * Policy and belief output layers use orthogonal gain 0.01; the value output uses gain 1.0.
* **Pooled baseline.** The fixed tokens (global, bases, pending, deck) are concatenated; the
  variable groups (hand, my backline, frontline, opponent backline, revealed) are pooled.
* **Specs and checkpoints.**
  * `card_table` is rebuilt from the layout in `spec()` (it is not in the state_dict).
  * The MLP's privileged input is `priv / 3` concatenated to the value input.
  * `net_from_checkpoint` refuses Stage 1 (no kind), Stage 2 (kind `entity` or encoder version ≠ 5),
    a wrong `obs_dim`, weights that do not fit, and a fingerprint mismatch. With
    `allow_pool_mismatch` it rebuilds the net on the live layout.

## 8. Training (`cardgame.rl.ppo`, `cardgame.rl.rollout`, `train.py`)

The Stage 2 pipeline is reused; changes:
* **Opponents:** latest self 0.5. Otherwise the pool weights are lookahead 0.15 / random 0.05 /
  snapshot 0.30 (`--opp-weights`; `greedy` stays an allowed kind with weight 0).
* **Decks:** each new game uses `sample_deal(seed, config, random_frac)` (default 0.7).
* **Trajectories** add `opp_hand` (uint8 counts per card index per transition) as belief labels
  and privileged critic input.
* **Loss:** PPO + `vf_coef`·value + `belief_coef`·BCE(belief, opp_hand > 0) − `ent_coef`·entropy.
  The default `belief_coef` is 0.25; `--no-belief` disables the head.
* **Quick eval** (every `eval_every` updates, vs lookahead): `eval_deals` deals, half on random
  decks (seeds `QUICK_EVAL_SEED_BASE + d`) and half cycling all fixed deck pairs. It reports the
  cells with their [wins, games]; `min_cell` is None unless every unordered cell was played (that
  needs `eval_deals - eval_deals // 2 ≥ n²` fixed deals), and a None `min_cell` is never eligible.
* **best.pt:** the rule is the highest random-deck win rate among checkpoints whose minimum
  fixed-pair cell is ≥ 0.6, else the highest overall. **[clarified]** A quick eval has only 16–32
  games per cell (256 deals), too few to decide the 0.6 bar, so the rule is applied twice:
  * During training each quick eval may replace a provisional `best.pt` (the rule on the quick
    eval) and nominates candidates `cand_<update>.pt`: the union of the top `select_top` (3)
    evals by the rule and the top `select_top` by the posterior probability (uniform prior per
    cell) that every cell is ≥ 0.6. Candidates that drop out are deleted; latest.pt lists them.
  * When the run reaches `total_updates` (after the workers stop), a **selection pass**
    re-evaluates every candidate vs lookahead with `select_games` (2000) games on random decks and
    on `decks="all"` (rounded up to 2n²), deals from `SELECTION_SEED_BASE` = 500,000,000, the
    sampling policy and `max(1, workers)` evaluation processes. The rule on these results picks
    `best.pt`; `selection.json` records every candidate. `select_games = 0` skips the pass.
    Resuming a finished run with the same `total_updates` runs only the pass.
* **New config fields:** `arch` ("transformer" | "pooled" | "mlp"), `layers`, `heads`, `ff`,
  `id_dim`, `ctx_dim`, `pair_dim` (the pointer scorer's pair-MLP width, 32), `belief`,
  `belief_coef`, `privileged_critic`, `random_deck_frac`, `micro_batch` (1024), `amp`
  (bf16 autocast, CUDA only), `inference_server` ("auto" | "on" | "off"), `serve_snapshots`
  (False; §8.1), `select_top` (3), `select_games` (2000).
  * The Stage 2-only `attention_layers` and `pair_mlp_dim` are gone.
  * Default-on booleans get `--no-<name>` flags. Dict options (`--opp-weights`) take one or more
    `kind=weight` tokens, each optionally a comma-separated list (an unquoted comma splits the
    argument in PowerShell).
  * A Stage 1/2 checkpoint on `--resume` is refused with a one-line error (exit code 2).
    **[clarified]** So is every other refused resume (architecture flags that differ from the
    checkpoint, naming each differing field and its flag; a used-up seed block; an unreadable
    file), invalid settings and a missing game data file.
  * **amp:** under CUDA autocast LayerNorm, softmax and sums run in fp32 while Linear layers give
    bf16; the Transformer tower scatters the encoder output into a buffer of the output's dtype.
* **Learner memory:** the 3-layer Transformer needs about 1.9 MB of activations per sample.
  * Each PPO minibatch is sorted by present-token count and processed in `micro_batch` chunks
    with gradient accumulation (each chunk adds `sum / minibatch size`).
  * Observations stay float16 on the device until their chunk.
  * Shuffling and sorting run on the CPU, so runs are reproducible across devices.
* **Logging:** `belief_loss`, `belief_acc`, `belief_rprec`, `belief_pos_rate` (epoch-0 data, i.e.
  before the epochs fit the batch; `belief_loss_all` is the mean over all epochs), the worker time
  breakdown (`wt_<part>_s` / `wt_<part>_frac` for engine, deal, encode, infer, scripted, other;
  `infer_batch_mean`), `pool_io_s` (the learner's pipe traffic with spawned workers: receiving
  requests and results, densifying the results, sending replies) and the quick-eval parts
  (`eval_lookahead_*`, the random-deck rate and fixed cells; `eval_lookahead_min_cell` only when
  every cell was played).
* **Inference:** in the workers on CPU, or batched on the learner device (§8.1). Measured on the
  Mac with the default model, worker-side CPU inference was about two thirds of worker time.
* Seed ranges stay disjoint: eval 0+, selection 500,000,000+ (the best.pt selection pass),
  training 1e9+, quick eval 2e9+, warm-up 3e9, scenarios 4e9.
* **Training seed blocks.** Run seed s (0 ≤ s < 100) owns the deal seeds [1e9 + s·1e7,
  1e9 + (s+1)·1e7), about 6,000–10,000 default updates. **[clarified]** Before each collect the
  learner checks that every worker has seeds left for its running games plus twice the deals it
  made in the last collect (kept in latest.pt); otherwise training stops cleanly at the update
  boundary (`SeedBlockExhausted`, a one-line error, exit 2). `train()` warns at start when the
  projection (35 learner transitions per game) exceeds the block. On `--resume` the same seed
  continues after every seed used (refused when none is left); a different explicit `--seed`
  continues in that seed's block.

### 8.1 Batched inference (`--inference-server auto|on|off`, auto = on iff CUDA)
The learner process is the inference server; it is otherwise idle while it waits for rollouts.
* **Request.** A worker sends `("infer", net_key, obs, mask packbits (n, ceil(N/8)))`, where
  net_key is `"policy"` or a snapshot name, and blocks until the reply `("logits", float32 (n, N)
  masked logits)`. A worker has at most one request in flight.
* **Pipe format [clarified].** Encodings cross every pipe as `rollout.SparseRows` (row pointers,
  uint16 column indices, float16 values of the entries with a nonzero bit pattern): bit-exact and
  about a tenth of the dense float16 rows (~4% of the entries are nonzero). This covers the
  requests' obs and the obs of the trajectories in `("ok", result)`; the learner densifies them on
  receipt, so trajectories outside the pipe hold dense float16 obs.
* **Serving.** While it waits for the results, the learner drains every request that is ready
  (no waiting for more), groups the rows by net_key and runs one fp32 forward per group on the
  device under `inference_mode`.
  * The policy is the learner net in eval mode. Its weights are the ones the workers would
    otherwise receive.
  * Snapshot nets are built on the device on first use from the snapshot weights and cached by
    name. A name is kept while it is in the pool or a running game still plays it (workers
    report `snapshots_in_use`), then dropped.
* **Sampling.** The worker samples with its own generator and computes logp itself, so sampling
  stays deterministic per worker. A collect equals local mode except for float rounding in
  batches that mix rows of several workers, which can flip a sample only at a near-tie.
* **`serve_snapshots`** (default False, `--serve-snapshots`). By default only the policy is
  served: workers keep CPU snapshot nets, which get their weights as in local mode. Reason: snapshot
  requests are tiny (about 2.4 rows; with snapshots in the pool, a worker loop sends 1 policy
  request plus about 2 snapshot requests), and each worker plays its own 2 of up to 30
  snapshots. So one drain can need about 10 launch-bound forwards of 2–3 rows each. Pick the
  setting by an A/B run on the desktop.
* **Unchanged:** final results arrive as `("ok", result)` on the same pipe, and failure handling
  and Ctrl-C work as before. A worker error, death or timeout while the learner serves stops
  every worker and raises `RolloutWorkerError`. In server mode workers build no policy net (and
  no snapshot nets with `serve_snapshots`) and get no policy weights. In-process mode
  (`--workers 0`) always uses local nets, silently.
* **Logging.** A worker's `infer` time is its wait for the reply plus sampling. Each update logs
  `server_drains`, `server_requests`, `server_forwards`, `server_rows_mean` (rows per forward),
  `server_forward_s` (device sync included), `server_snapshot_forwards`,
  `server_snapshot_forward_s` and `server_busy_s` (unpacking the requests and the forwards). The
  pipe traffic, replies included, is `pool_io_s` (§8 Logging). config.json records
  `inference_server_active`.

## 9. Agents (`cardgame.agents`)

* **Protocol:** `reset(seed)`, `act(obs, legal_actions) -> int`. An agent that needs to
  simulate sets `needs_game = True` and implements `act_game(game, player) -> int`. Callers use
  `agents.choose_action(agent, game)`.
  * Such an agent must only use `game.determinize(player, rng)`. It never reads the real game's
    hidden state.
* **LookaheadAgent** (the baseline):
  1. `sim = game.determinize(p, rng)`.
  2. For each legal action a: clone `sim` and step a. While the result has a pending choice for
     p, step the best CHOOSE by the same evaluation, greedily (≤ 3 deep).
  3. Score the result. A finished game scores ±1000 (draw 0). Otherwise
     `V = 1.0·(base_p − base_o) + 0.5·(Σ_p(atk+hp) − Σ_o(atk+hp)) + 1.0·(hand_p − hand_o)`.
     END_TURN is scored as the current position (`V(sim)` without stepping) **[clarified:
     stepping END_TURN would include the opponent's draw and start-of-turn effects, which happen
     whatever p does, and would bias the bot against ever ending its turn]**.
  4. Pick the max V. Ties go by kind priority ATTACK > PLAY > MOVE > CHOOSE > END_TURN, then the
     lower index.
  * **Mulligan:** MULLIGAN every card with cost ≥ 5, then CONFIRM.
  * Effects are handled automatically through the engine.
* **RandomAgent** is unchanged (uniform over the legal actions).
* **GreedyAgent** (Stage 2 v2) is kept as a legacy diagnostic:
  * mulligan → CONFIRM;
  * choices → the first option;
  * operations are treated as plays by cost.

## 10. Evaluation

* **Duplicate games** as Stage 2. Deck modes:
  * `random`: deal k gives seat 0 and seat 1 the decks `generate_deck(deck_rng(k, 0/1))`, and the
    seat-swapped game swaps them back.
  * `all`: the 16 ordered fixed pairs cycle; counts round up to 2·n².
  * `sampled`: `sample_decks`.
* **Matchup cells (fixed decks, pass/fail):** the unordered `C{i,j}`, as Stage 2.
* **Scenarios:** the 16 Stage 2 scenarios plus four new ones, built from named cards. All use
  `mulligan=False` except `mulligan_sanity`. Baselines: lookahead (one run) and random (mean of
  20 seeds); greedy is reported for reference (eval.py's table and eval.json `scenarios.greedy`).
  * **[clarified]** In mid-game positions the agent's deck holds the rest of its decklist (minus
    its hand and its non-token units on the board, sorted by card index), so the deck token and
    deck-size inputs look like a real game's; the opponent's deck is empty, so its start-of-turn
    draw in the survival checks reveals no hidden card. The agent draws only after its turn.
  * **[clarified]** No position holds a card that draws on the hidden RNG (random targets,
    random discards, or a token bringing them; `scenarios.rng_dependent_cards`): the solver and the
    goals see one RNG state, so such a scenario's best line would depend on luck.
    `check_position` rejects them.
  * `operation_lethal`: playing an operation triggers the last damage. **[clarified]** No
    operation in the pool can hit a base, so the damage comes from a friendly on-play watcher
    (Forward Observer's artillery trigger) fired by the operation (Fire Mission, with a CHOOSE).
  * `on_death_exploit`: a deliberate trade fires a beneficial `on_death`.
  * `effect_clears_defense`: effect damage kills a Defense unit, which opens the protected
    target.
  * `mulligan_sanity`: from the mulligan phase, replace every card with cost ≥ 5 and keep every
    card with cost ≤ 2. The goal is checked at CONFIRM.

  Validity (`dominance_violations`) and the solver handle choices and operations through the
  engine's legal actions.
* **Done when** (`eval.py` prints a conclusive PASS/FAIL only when all four were measured under
  these settings; otherwise INDICATIVE with the reasons):
  1. PPO (sampling) vs lookahead ≥ 70% over ≥ 2,000 duplicate games on **random decks**
     (seeds from 0).
  2. Every unordered fixed-deck matchup cell vs lookahead ≥ 60% (`decks="all"`, ≥ 2,000 games).
  3. The Transformer beats the pooled baseline head-to-head (> 50% over ≥ 2,000 duplicate games
     on random decks; `--baseline <pooled.pt>`, `--baseline-target` 0.50).
     * Draws count as losses. The score is reported next to the win rate.
     * Conclusive only if the agent is a Transformer and the baseline a pooled net.
  4. > 50% of scenarios solved (argmax).
* **Ablations** (desktop runs, reported in the README):
  * Transformer vs pooled (criterion 3);
  * belief head on vs off (head-to-head plus vs lookahead).

## 11. Benchmark

`bench.py` reports, and writes to `results/bench_stage3.txt`:
* games/s, steps/s, steps and rounds per game;
* per-call µs (observe, encode, legal_actions, legal_mask, clone, determinize,
  `LookaheadAgent.act_game`);
* lookahead rows (`play_game` and the worker pool) and a PPO-vs-lookahead row.

It compares with Stage 2 (`results/stage2/bench.txt`), measured on the same machine. The fixed
decks are still sampled per seed, so the rows stay comparable.
