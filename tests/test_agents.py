"""Agent tests (SPEC section 9): legality over full games, greedy v2 rules on constructed positions,
the one-step LookaheadAgent (hidden information, evaluation, choices, tie-breaks, mulligan), the
`choose_action` dispatch and the factory.

Configs: `CONFIG` is the shipped ruleset with the mulligan off (tests/conftest.py); `VANILLA_CONFIG` is
the frozen Stage 2 content, used where a helper models only the Stage 2 rules (`spec_legal`,
`flags_view`) or a threshold was measured on it. Effect cards come from the test's own pools
(`cards.build_ruleset`), never from the shipped content.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Optional, Sequence

import pytest

from cardgame import cards as cards_mod
from cardgame.actions import ActionSpace
from cardgame.agents import Agent, GameAgent, GreedyAgent, LookaheadAgent, RandomAgent, choose_action, make_agent
from cardgame.cards import FAST, RANGED, TROOP, CardDef, CardPool, GameConfig, load_ruleset
from cardgame.engine import CHOICE, DRAW, MAIN, MULLIGAN, Game, Observation, Unit, UnitView
from conftest import CONFIG, VANILLA_CONFIG, add_unit, blank_game, set_hand

N_DECKS = CONFIG.n_decks


# ---------------------------------------------------------------------- helpers
def deal_decks(k: int) -> tuple:
    """Deck pair of deal k: every ordered pair once per N_DECKS^2 deals."""
    return divmod(k % (N_DECKS * N_DECKS), N_DECKS)


def play_game(game: Game, agents: Sequence[Agent], seed: int, decks: Optional[tuple] = None) -> Optional[int]:
    """Play one full game (moves via `choose_action`), asserting every chosen action is legal. Returns the
    winner."""
    game.reset(seed, decks=decks)
    for i, agent in enumerate(agents):
        agent.reset(2 * seed + i)
    while not game.done:
        p = game.current_player()
        legal = game.legal_actions()
        action = choose_action(agents[p], game)
        assert action in legal, (
            f"{agents[p].name} (seat {p}, seed {seed}, round {game.round}) chose illegal "
            f"{game.describe(action)}; legal: {[game.describe(a) for a in legal]}")
        game.step(action)
    return game.winner()


def duplicate_match(agent_a: Agent, agent_b: Agent, deals: Sequence[int], config: GameConfig = CONFIG) -> dict:
    """Each deal (all deck pairs in turn) is played twice with seats swapped. Returns A's results."""
    game = Game(config)
    out = {"wins": 0, "draws": 0, "losses": 0, "games": 0}
    for k in deals:
        for a_seat in (0, 1):
            agents = (agent_a, agent_b) if a_seat == 0 else (agent_b, agent_a)
            w = play_game(game, agents, k, deal_decks(k))
            out["games"] += 1
            out["draws" if w == DRAW else ("wins" if w == a_seat else "losses")] += 1
    return out


def flags_view(c: CardDef, atk: int, hp: int, summoned: bool = False, moved: bool = False,
               attacked: bool = False) -> UnitView:
    """UnitView of a unit of card `c`, with can_move/can_attack from the SPEC section 2 nature table (Stage 2
    keywords only: no blitz, fury, smokescreen or pins, so `attacks` is 0 or 1)."""
    if c.nature == FAST:
        can_move, can_attack = not summoned and not moved, not summoned and not attacked
    else:
        can_move = can_attack = not summoned and not moved and not attacked
    return UnitView(c.index, atk, hp, c.health, c.armor, c.defense, c.nature, c.move_cost, summoned, moved,
                    attacked, can_move, can_attack, attacks=int(attacked))


def spec_legal(o: Observation, config: GameConfig) -> list:
    """Legal actions derived from an Observation per SPEC sections 2-3 (reach, Defense, coins, space)."""
    if o.done or not o.is_my_turn:
        return []
    sp = ActionSpace(config.max_hand_size, config.zone_capacity)
    Z = config.zone_capacity
    cost = [c.cost for c in config.cards.cards]
    legal = [sp.END_TURN]
    if len(o.my_backline) < Z:
        legal += [sp.PLAY0 + i for i, c in enumerate(o.hand) if cost[c] <= o.my_coins]
    if o.front_owner >= 0 and len(o.frontline) < Z:
        legal += [sp.MOVE0 + j for j, u in enumerate(o.my_backline) if u.can_move and u.move_cost <= o.my_coins]

    def targetable(zone):
        guards = [k for k, u in enumerate(zone) if u.defense]
        return guards if guards else list(range(len(zone)))

    attackers = [(j, u) for j, u in enumerate(o.my_backline) if u.can_attack]
    if o.front_owner == 1:
        attackers += [(Z + k, u) for k, u in enumerate(o.frontline) if u.can_attack]
    enemy_back = targetable(o.opp_backline)
    enemy_front = [Z + k for k in targetable(o.frontline)] if o.front_owner == -1 else []
    for a, u in attackers:
        if u.nature == RANGED:
            targets = enemy_back + enemy_front + [sp.BASE_TARGET]
        elif a < Z:
            targets = enemy_front
        else:
            targets = enemy_back + [sp.BASE_TARGET]
        legal += [sp.attack(a, t) for t in targets]
    return sorted(legal)


# Small hand-made pool so every rule and tie-break is exercised independently of the shipped cards.
TEST_CARDS = (  # id, nature, cost, attack, health, defense, armor, move_cost
    ("imp", TROOP, 1, 1, 1, False, 0, 1),
    ("pawn", TROOP, 1, 1, 2, False, 0, 1),
    ("guard", TROOP, 2, 2, 3, False, 0, 1),
    ("brute", TROOP, 3, 3, 3, False, 0, 1),
    ("tank", TROOP, 3, 2, 6, False, 0, 1),
    ("squire", TROOP, 3, 2, 5, False, 1, 1),
    ("plate", TROOP, 4, 3, 4, False, 2, 2),
    ("ogre", TROOP, 5, 5, 5, False, 0, 1),
    ("titan", TROOP, 7, 7, 7, False, 0, 2),
    ("mule", TROOP, 2, 4, 2, False, 0, 3),
    ("wall", TROOP, 2, 1, 4, True, 0, 1),
    ("keep", TROOP, 5, 3, 6, True, 1, 1),
    ("hare", FAST, 1, 2, 1, False, 0, 0),
    ("rider", FAST, 3, 3, 3, False, 0, 1),
    ("charger", FAST, 5, 5, 4, False, 0, 1),
    ("sentry", FAST, 3, 2, 3, True, 0, 1),
    ("sling", RANGED, 1, 1, 1, False, 0, 1),
    ("bow", RANGED, 2, 2, 1, False, 0, 1),
    ("cannon", RANGED, 6, 5, 4, False, 1, 3),
    ("tower", RANGED, 4, 1, 5, True, 2, 3),
)
TEST_POOL = CardPool(tuple(
    CardDef(i, cid, cid.title(), "unit", cost, atk, hp, nature=nat, move_cost=mc, defense=dfn, armor=arm)
    for i, (cid, nat, cost, atk, hp, dfn, arm, mc) in enumerate(TEST_CARDS)))
TEST_CONFIG = GameConfig(cards=TEST_POOL, decks=((),), mulligan=False)
CARD = {c.id: c.index for c in TEST_POOL.cards}
SP = ActionSpace(TEST_CONFIG.max_hand_size, TEST_CONFIG.zone_capacity)
Z = SP.zone_capacity
END = SP.END_TURN
BASE = SP.BASE_TARGET


def PLAY(i: int) -> int:
    return SP.PLAY0 + i


def MOVE(j: int) -> int:
    return SP.MOVE0 + j


def ATK(a: int, t: int) -> int:
    """ATTACK(a, t): a = own backline slot j or Z + frontline slot; t = enemy backline slot k, Z + frontline
    slot, or BASE."""
    return SP.attack(a, t)


def unit(card_id: str, atk: Optional[int] = None, hp: Optional[int] = None, **flags) -> UnitView:
    c = TEST_POOL[CARD[card_id]]
    return flags_view(c, c.attack if atk is None else atk, c.health if hp is None else hp, **flags)


def make_obs(hand: Sequence[str] = (), **fields) -> Observation:
    n = len(TEST_POOL)
    values = dict(player=0, is_my_turn=True, went_first=True, round=6, my_deck=0, my_coins=0, opp_coins=0,
                  my_base_hp=20, opp_base_hp=20, hand=tuple(CARD[c] for c in hand), opp_hand_size=4,
                  my_deck_size=30, opp_deck_size=30, my_played=(0,) * n, opp_played=(0,) * n,
                  my_backline=(), opp_backline=(), frontline=(), front_owner=0, done=False, result=0)
    values.update(fields)
    return Observation(**values)


def greedy_choice(o: Observation, legal: Optional[Sequence[int]] = None) -> int:
    legal = spec_legal(o, TEST_CONFIG) if legal is None else legal
    action = GreedyAgent(TEST_CONFIG).act(o, legal)
    assert action in legal, SP.describe(action)
    return action


def check(o: Observation, expected: int, also_legal: Sequence[int] = ()) -> None:
    """Greedy picks `expected`; `also_legal` lists the alternatives the position must offer."""
    legal = spec_legal(o, TEST_CONFIG)
    assert expected in legal and set(also_legal) <= set(legal), [SP.describe(a) for a in legal]
    got = greedy_choice(o, legal)
    assert got == expected, f"got {SP.describe(got)}, expected {SP.describe(expected)}"


# ---------------------------------------------------------------------- legality over full games
@pytest.mark.parametrize("spec", ["random", "greedy", "lookahead"])
def test_agent_only_returns_legal_actions(spec):
    """96 full games per agent over every deck pair: 48 mirror games, 48 against the other baseline."""
    game = Game(CONFIG)
    agent, mirror = make_agent(spec, CONFIG, seed=1), make_agent(spec, CONFIG, seed=2)
    other = make_agent("greedy" if spec == "random" else "random", CONFIG, seed=3)
    results = []
    for k in range(96):
        if k < 48:
            agents = (agent, mirror)
        else:
            agents = (agent, other) if (k // N_DECKS ** 2) % 2 == 0 else (other, agent)
        results.append(play_game(game, agents, k, deal_decks(k)))
    assert all(w in (0, 1, DRAW) for w in results)


def test_spec_legal_helper_matches_engine():
    """The constructed-position tests rely on spec_legal and flags_view (Stage 2 rules); check both at
    real states of the Stage 2 content."""
    cfg = VANILLA_CONFIG
    game = Game(cfg)
    agents = (GreedyAgent(cfg), RandomAgent(cfg, seed=0))
    checked = views = 0
    seen_natures = set()
    for k in range(48):
        game.reset(k, decks=deal_decks(k))
        while not game.done:
            p = game.current_player()
            obs, legal = game.observe(p), game.legal_actions()
            assert spec_legal(obs, cfg) == legal
            assert spec_legal(game.observe(1 - p), cfg) == []
            for v in obs.my_backline + obs.frontline + obs.opp_backline:
                c = cfg.cards[v.card]
                assert flags_view(c, v.atk, v.hp, v.summoned, v.moved, v.attacked) == v
                seen_natures.add(v.nature)
                views += 1
            checked += 1
            game.step(agents[(p + k) % 2].act(obs, legal))
        assert spec_legal(game.observe(0), cfg) == [] == game.legal_actions()
    assert checked > 2000 and views > 5000 and seen_natures == {TROOP, FAST, RANGED}


def test_test_pool_covers_natures_and_traits():
    combos = {(c.nature, c.defense, c.armor > 0) for c in TEST_POOL.cards}
    assert {(n, False, False) for n in (TROOP, FAST, RANGED)} <= combos
    assert {(n, True, False) for n in (TROOP, FAST)} <= combos and (RANGED, True, True) in combos


def test_greedy_is_deterministic():
    game = Game(CONFIG)

    def trace(seed: int) -> list:
        game.reset(seed, decks=deal_decks(seed))
        agents, actions = (GreedyAgent(CONFIG), GreedyAgent(CONFIG)), []
        while not game.done:
            p = game.current_player()
            actions.append(agents[p].act(game.observe(p), game.legal_actions()))
            game.step(actions[-1])
        return actions

    for seed in (0, 7, 123):
        assert trace(seed) == trace(seed)


def test_greedy_uses_every_mechanic_in_real_games():
    """Over full greedy-vs-greedy games: fast units move and attack in one round, ranged units shoot,
    and Defense units get killed while shielding (Stage 2 content, where the counts were measured)."""
    game, agent = Game(VANILLA_CONFIG), GreedyAgent(VANILLA_CONFIG)
    sp = game.action_space
    fast_double = ranged_shots = shield_kills = 0
    for k in range(32):
        game.reset(k, decks=deal_decks(k))
        while not game.done:
            p = game.current_player()
            action = agent.act(game.observe(p), game.legal_actions())
            if action >= sp.ATTACK0:
                a, t = divmod(action - sp.ATTACK0, sp.n_targets)
                u = game.backline[p][a] if a < Z else game.frontline[a - Z]
                fast_double += u.nature == FAST and u.moved
                ranged_shots += u.nature == RANGED
                if t != sp.BASE_TARGET:
                    zone = game.backline[1 - p] if t < Z else game.frontline
                    tgt = zone[t if t < Z else t - Z]
                    shield_kills += (tgt.defense and max(0, u.atk - tgt.armor) >= tgt.hp
                                     and any(not v.defense for v in zone))
            game.step(action)
    # measured with the Stage 2 decks: 104 / 431 / 96
    assert fast_double > 40 and ranged_shots > 150 and shield_kills > 30, (fast_double, ranged_shots, shield_kills)


# ---------------------------------------------------------------------- engine round trip
def engine_position(base_hp: int = 20, coins: int = 3) -> Game:
    """A TEST_CONFIG game on round 3 with empty decks, hands and board; seat 0 to move."""
    g = Game(TEST_CONFIG)
    g.reset(0)
    g.first_player = g.current = 0
    g.round, g.coins, g.base_hp = 3, [coins, 0], [20, base_hp]
    g.hands, g.deck_cards = [[], []], [[], []]
    g.invalidate()
    return g


def add(g: Game, owner: int, zone: str, card_id: str, **flags) -> Unit:
    u = Unit.from_card(TEST_POOL[CARD[card_id]], owner, g.next_uid)
    g.next_uid += 1
    u.summoned = False
    for k, v in flags.items():
        setattr(u, k, v)
    if zone == "back":
        g.backline[owner].append(u)
    else:
        g.frontline.append(u)
        g.front_owner = owner
    g.invalidate()
    return u


def greedy_turn(g: Game) -> list:
    """Let greedy play seat 0's turn; returns the actions taken."""
    agent, actions = GreedyAgent(TEST_CONFIG), []
    while not g.done and g.current_player() == 0:
        actions.append(agent.act(g.observe(0), g.legal_actions()))
        g.step(actions[-1])
        assert len(actions) < 64
    return actions


def test_engine_fast_unit_moves_then_finishes_the_base():
    g = engine_position(base_hp=5, coins=1)
    add(g, 0, "back", "charger")
    assert greedy_turn(g) == [MOVE(0), ATK(Z, BASE)]
    assert g.done and g.winner() == 0


def test_engine_full_turn_uses_rules_in_order():
    g = engine_position(base_hp=20, coins=4)
    g.hands[0] = [CARD["guard"]]
    add(g, 0, "back", "rider")
    add(g, 0, "back", "bow")
    add(g, 1, "back", "wall", hp=1)
    add(g, 1, "back", "titan", hp=7)
    actions = greedy_turn(g)
    # fast advance (1 coin; the bow shifts to backline slot 0), play guard (2 coins), the bow (cheaper
    # than the rider) kills the shielding wall, the rider cannot kill the titan and hits the base; the
    # new guard cannot act yet: END_TURN
    assert actions == [MOVE(0), PLAY(0), ATK(0, 0), ATK(Z, BASE), END], [SP.describe(a) for a in actions]
    assert [u.hp for u in g.backline[1]] == [7] and g.base_hp[1] == 17


# ---------------------------------------------------------------------- rule 0: lethal
def test_lethal_attacks_base_with_highest_atk_unit():
    o = make_obs(hand=("titan",), my_coins=7, front_owner=1, frontline=(unit("brute"), unit("ogre")),
                 my_backline=(unit("bow"), unit("rider")), opp_backline=(unit("imp"),), opp_base_hp=10)
    # bow 2 + brute 3 + ogre 5 = 10 >= 10; the rider (backline fast) has no base attack yet
    check(o, ATK(Z + 1, BASE), also_legal=(PLAY(0), MOVE(1), ATK(Z, 0)))


def test_no_lethal_when_sum_falls_short():
    o = make_obs(hand=("titan",), my_coins=7, front_owner=1, frontline=(unit("brute"), unit("ogre")),
                 my_backline=(unit("bow"),), opp_backline=(unit("imp"),), opp_base_hp=11)
    check(o, PLAY(0))


def test_lethal_counts_only_units_with_a_legal_base_attack():
    # backline titan (troop: no base reach) and a just-deployed ogre do not count
    o = make_obs(my_coins=2, front_owner=1, frontline=(unit("brute"), unit("ogre", summoned=True)),
                 my_backline=(unit("titan"),), opp_base_hp=4)
    check(o, MOVE(0))
    check(o._replace(opp_base_hp=3), ATK(Z, BASE))


def test_lethal_tie_takes_lower_action_index():
    # bow (backline 1) and guard (frontline 0) both have atk 2; the backline slot has the lower index
    o = make_obs(front_owner=1, frontline=(unit("guard"),), my_backline=(unit("sling"), unit("bow")),
                 opp_backline=(unit("wall"),), opp_base_hp=5)
    check(o, ATK(1, BASE), also_legal=(ATK(Z, BASE),))
    o = make_obs(front_owner=1, frontline=(unit("guard"), unit("guard")), opp_base_hp=4)
    check(o, ATK(Z, BASE))


# ---------------------------------------------------------------------- rule 1: fast advance
def test_fast_unit_advances_before_playing():
    o = make_obs(hand=("ogre",), my_coins=6, my_backline=(unit("titan"), unit("rider")))
    check(o, MOVE(1), also_legal=(PLAY(0), MOVE(0)))


def test_fast_advance_key_atk_then_slot():
    o = make_obs(my_coins=3, my_backline=(unit("hare"), unit("rider"), unit("charger", summoned=True),
                                          unit("rider")))
    check(o, MOVE(1), also_legal=(MOVE(0), MOVE(3)))


def test_fast_unit_that_attacked_is_not_a_fast_advance():
    o = make_obs(hand=("guard",), my_coins=3, my_backline=(unit("rider", attacked=True),))
    check(o, PLAY(0), also_legal=(MOVE(0),))
    check(o._replace(hand=()), MOVE(0))  # rule 6 still advances it


def test_fast_advance_needs_a_legal_move():
    blocked = make_obs(hand=("imp",), my_coins=1, my_backline=(unit("rider"),), front_owner=-1,
                       frontline=(unit("titan"),))
    check(blocked, PLAY(0))
    # rider's move cost (1) exceeds the coins; the hare moves for free
    check(make_obs(my_backline=(unit("rider"), unit("hare"))), MOVE(1))


# ---------------------------------------------------------------------- rule 2: play
def test_plays_highest_cost_affordable_card():
    check(make_obs(hand=("imp", "guard", "brute", "ogre"), my_coins=4), PLAY(2))


@pytest.mark.parametrize("hand,coins,expected", [
    (("brute", "tank"), 3, PLAY(1)),           # same cost: tank has higher atk+hp (8 > 6)
    (("tank", "brute"), 3, PLAY(0)),           # slot order does not matter, stats do
    (("brute", "squire"), 3, PLAY(1)),         # squire 7 > brute 6
    (("guard", "guard", "imp"), 2, PLAY(0)),   # identical cards: lower slot
    (("pawn", "rider", "brute"), 9, PLAY(1)),  # rider and brute tie on cost and stats: lower slot
])
def test_play_tie_breaks(hand, coins, expected):
    check(make_obs(hand=hand, my_coins=coins), expected)


def test_play_comes_before_kills_trades_ranged_moves_and_base():
    o = make_obs(hand=("imp",), my_coins=1, front_owner=1, frontline=(unit("titan"),),
                 my_backline=(unit("brute"), unit("bow")), opp_backline=(unit("wall", hp=1), unit("ogre")))
    check(o, PLAY(0), also_legal=(ATK(Z, 0), ATK(1, 0), MOVE(0), ATK(Z, BASE)))


def test_no_play_without_backline_space():
    o = make_obs(hand=("imp",), my_coins=5, my_backline=tuple(unit("pawn", summoned=True) for _ in range(5)))
    check(o, END)


# ---------------------------------------------------------------------- rule 3: kill shielding Defense
def test_kills_shielding_defense_before_a_better_trade():
    # bow can kill the shielding wall (value 2) or the lone ogre in the enemy frontline (value 5)
    o = make_obs(front_owner=-1, frontline=(unit("ogre", hp=1),), my_backline=(unit("guard"), unit("bow")),
                 opp_backline=(unit("wall", hp=2), unit("imp")))
    check(o, ATK(1, 0), also_legal=(ATK(1, Z), ATK(0, Z)))


def test_lone_defense_unit_is_a_plain_trade():
    o = make_obs(front_owner=-1, frontline=(unit("ogre", hp=1),), my_backline=(unit("guard"), unit("bow")),
                 opp_backline=(unit("wall", hp=2),))
    check(o, ATK(1, Z), also_legal=(ATK(1, 0),))


def test_kills_shielding_defense_in_the_enemy_frontline():
    o = make_obs(front_owner=-1, frontline=(unit("charger"), unit("wall", hp=2)), my_backline=(unit("guard"),))
    check(o, ATK(0, Z + 1))


def test_shield_kill_prefers_survival_over_target_value():
    # guard kills keep (value 5) but dies to its return hit; killing wall (value 2) it survives
    o = make_obs(front_owner=1, frontline=(unit("guard"),),
                 opp_backline=(unit("wall", hp=1), unit("keep", hp=1), unit("brute")))
    check(o, ATK(Z, 0), also_legal=(ATK(Z, 1),))
    # rule 4 alone would take the keep (gain 3 > 2): the shield rule really decides
    assert greedy_choice(o._replace(opp_backline=(unit("wall", hp=1), unit("keep", hp=1)))) == ATK(Z, 1)


def test_shield_kill_prefers_higher_target_value():
    o = make_obs(front_owner=1, frontline=(unit("titan"),),
                 opp_backline=(unit("wall", hp=1), unit("keep", hp=1), unit("imp")))
    check(o, ATK(Z, 1), also_legal=(ATK(Z, 0),))


def test_shield_kill_prefers_cheaper_attacker_then_lower_index():
    o = make_obs(front_owner=1, frontline=(unit("titan"), unit("pawn")),
                 opp_backline=(unit("wall", hp=1), unit("imp")))
    check(o, ATK(Z + 1, 0), also_legal=(ATK(Z, 0),))
    o = make_obs(front_owner=1, frontline=(unit("pawn"), unit("pawn")),
                 opp_backline=(unit("wall", hp=1), unit("imp")))
    check(o, ATK(Z, 0), also_legal=(ATK(Z + 1, 0),))


@pytest.mark.parametrize("keep_hp,expected", [(2, ATK(Z, 0)), (3, ATK(Z, BASE))])
def test_shield_kill_accounts_for_armor(keep_hp, expected):
    # brute deals 3 - 1 armor = 2: kills the keep at 2 hp, not at 3 (then it hits the base)
    o = make_obs(front_owner=1, frontline=(unit("brute"),), opp_backline=(unit("keep", hp=keep_hp), unit("imp")))
    check(o, expected)


# ---------------------------------------------------------------------- rule 4: favorable trade
def test_prefers_highest_gain_trade():
    o = make_obs(front_owner=1, frontline=(unit("titan"),), opp_backline=(unit("imp"), unit("ogre"), unit("guard")))
    check(o, ATK(Z, 1))


def test_trade_where_cheaper_attacker_dies_for_costlier_target():
    o = make_obs(front_owner=-1, frontline=(unit("brute", hp=2),), my_backline=(unit("guard"),))
    check(o, ATK(0, Z))


def test_trade_gain_tie_prefers_costlier_target():
    # front 0: ogre card as 3/9 kills brute and survives: gain 3, target 3
    # front 1: guard card as 5/1 kills ogre and dies: gain 5 - 2 = 3, target 5
    o = make_obs(front_owner=1, frontline=(unit("ogre", atk=3, hp=9), unit("guard", atk=5, hp=1)),
                 opp_backline=(unit("brute"), unit("ogre")))
    check(o, ATK(Z + 1, 1), also_legal=(ATK(Z, 0),))


def test_trade_full_tie_takes_lowest_action_index():
    o = make_obs(front_owner=1, frontline=(unit("titan"), unit("titan")), opp_backline=(unit("guard"), unit("guard")))
    check(o, ATK(Z, 0))
    o = make_obs(front_owner=-1, frontline=(unit("imp"), unit("imp")),
                 my_backline=(unit("pawn", summoned=True), unit("brute"), unit("brute")))
    check(o, ATK(1, Z))


@pytest.mark.parametrize("o,expected", [
    # even trade: brute vs brute, both die, equal value
    (make_obs(front_owner=-1, frontline=(unit("brute"),), my_backline=(unit("brute"),)), END),
    # no kill: pawn into tank
    (make_obs(front_owner=-1, frontline=(unit("tank"),), my_backline=(unit("pawn"),)), END),
    # trading down: a damaged brute (value 3) kills a guard (value 2) but dies
    (make_obs(front_owner=-1, frontline=(unit("guard"),), my_backline=(unit("brute", hp=2),)), END),
    # frontline attacker with only bad trades goes for the base instead
    (make_obs(front_owner=1, frontline=(unit("guard"),), opp_backline=(unit("tank"), unit("pawn", atk=3))),
     ATK(Z, BASE)),
    # chip damage that kills nothing is never a trade, even when the attacker survives
    (make_obs(front_owner=1, frontline=(unit("titan", atk=1),), opp_backline=(unit("imp", hp=2),)), ATK(Z, BASE)),
])
def test_refuses_unfavorable_or_even_trades(o, expected):
    legal = spec_legal(o, TEST_CONFIG)
    assert any(a >= SP.ATTACK0 and SP.decode(a).b != BASE for a in legal), "must offer a unit attack"
    check(o, expected)


def test_armor_decides_kills_and_survival():
    # plate (3 atk, armor 2) kills brute and survives its 3 - 2 = 1 return damage
    check(make_obs(front_owner=-1, frontline=(unit("brute"),), my_backline=(unit("plate"),)), ATK(0, Z))
    # brute deals only 1 to plate: no kill at 2 hp ...
    check(make_obs(front_owner=-1, frontline=(unit("plate", hp=2),), my_backline=(unit("brute"),)), END)
    # ... a kill at 1 hp, where brute dies (3 >= 3) but plate is worth more (4 > 3)
    check(make_obs(front_owner=-1, frontline=(unit("plate", hp=1),), my_backline=(unit("brute"),)), ATK(0, Z))


def test_attacker_armor_decides_survival():
    # plate (3/4, armor 2) kills a 5-atk brute and survives 5 - 2 = 3 < 4: a trade worth taking (gain 3);
    # without its armor it would die, and a dying plate (value 4) never trades for a brute (value 3)
    check(make_obs(front_owner=-1, frontline=(unit("brute", atk=5),), my_backline=(unit("plate"),)), ATK(0, Z))
    check(make_obs(front_owner=-1, frontline=(unit("brute", atk=6),), my_backline=(unit("plate"),)), END)


def test_trades_are_judged_with_the_engines_combat_rule(monkeypatch):
    """Greedy asks engine.combat_damage (the single source of the combat rules) instead of re-deriving them."""
    import cardgame.agents.greedy_agent as greedy_module
    o = make_obs(front_owner=-1, frontline=(unit("brute"),), my_backline=(unit("brute"),))
    check(o, END)  # even trade: both die
    calls = []

    def no_return_damage(attacker, target):
        calls.append((attacker, target))
        return attacker.atk, 0

    monkeypatch.setattr(greedy_module, "combat_damage", no_return_damage)
    assert greedy_choice(o) == ATK(0, Z) and calls  # the (patched) rule says the attacker survives


def test_ranged_attackers_always_survive_trades():
    # bow (1 hp) kills the guard: no return damage, so gain = 2 (and it beats the base attack)
    o = make_obs(front_owner=-1, frontline=(unit("guard", hp=2),), my_backline=(unit("bow"),))
    check(o, ATK(0, Z), also_legal=(ATK(0, BASE),))


def test_trade_comes_before_ranged_chip_and_advance():
    o = make_obs(my_coins=2, front_owner=1, frontline=(unit("brute"),), my_backline=(unit("titan"), unit("bow")),
                 opp_backline=(unit("guard"), unit("ogre")))
    check(o, ATK(Z, 0), also_legal=(MOVE(0), ATK(1, 1), ATK(Z, BASE)))


# ---------------------------------------------------------------------- rule 5: ranged
def test_ranged_chips_highest_value_target():
    check(make_obs(my_backline=(unit("bow"),), opp_backline=(unit("brute"), unit("titan"))), ATK(0, 1))


def test_ranged_damage_then_index_tie_breaks():
    # squire and brute both cost 3; bow deals 2 to brute but 2 - 1 = 1 to squire
    check(make_obs(my_backline=(unit("bow"),), opp_backline=(unit("squire"), unit("brute"))), ATK(0, 1))
    check(make_obs(my_backline=(unit("bow"),), opp_backline=(unit("titan"), unit("titan"))), ATK(0, 0))


def test_ranged_hits_enemy_frontline_from_backline():
    o = make_obs(front_owner=-1, frontline=(unit("titan"),), my_backline=(unit("bow"),),
                 opp_backline=(unit("imp", hp=5),))
    check(o, ATK(0, Z), also_legal=(ATK(0, 0),))


def test_ranged_target_value_comes_before_damage():
    # bow (2 atk) kills neither: tank (value 3) takes 2, cannon (value 6, armor 1) takes 1 -> the cannon
    check(make_obs(my_backline=(unit("bow"),), opp_backline=(unit("tank"), unit("cannon"))), ATK(0, 1))
    check(make_obs(my_backline=(unit("bow"),), opp_backline=(unit("cannon"), unit("tank"))), ATK(0, 0))


def test_ranged_ignores_zero_damage_targets_and_hits_base():
    check(make_obs(my_backline=(unit("sling"),), opp_backline=(unit("tower"), unit("titan"))), ATK(0, BASE))
    check(make_obs(my_backline=(unit("sling"),), opp_backline=(unit("keep"),)), ATK(0, BASE))


def test_ranged_respects_defense():
    o = make_obs(my_backline=(unit("bow"),), opp_backline=(unit("wall"), unit("titan")))
    check(o, ATK(0, 0))


def test_ranged_units_act_in_atk_then_slot_order():
    o = make_obs(my_backline=(unit("sling"), unit("bow")), opp_backline=(unit("titan"),))
    check(o, ATK(1, 0), also_legal=(ATK(0, 0),))
    o = make_obs(front_owner=1, frontline=(unit("cannon"),), my_backline=(unit("bow"),), opp_backline=(unit("titan"),))
    check(o, ATK(Z, 0), also_legal=(ATK(0, 0),))
    o = make_obs(front_owner=1, frontline=(unit("bow"),), my_backline=(unit("bow"),), opp_backline=(unit("titan"),))
    check(o, ATK(0, 0), also_legal=(ATK(Z, 0),))


def test_first_ranged_unit_with_an_action_wins():
    o = make_obs(my_backline=(unit("bow"), unit("sling")), opp_backline=(unit("titan"),))
    assert greedy_choice(o, [END, ATK(1, BASE)]) == ATK(1, BASE)
    assert greedy_choice(o, [END, ATK(0, BASE), ATK(1, 0)]) == ATK(0, BASE)


def test_ranged_shot_comes_before_advance():
    o = make_obs(my_coins=1, my_backline=(unit("guard"), unit("bow")))
    check(o, ATK(1, BASE), also_legal=(MOVE(0), MOVE(1)))


# ---------------------------------------------------------------------- rule 6: advance
def test_advances_highest_atk_troop_or_fast():
    o = make_obs(my_coins=3, my_backline=(unit("pawn"), unit("brute"), unit("titan", summoned=True), unit("guard")))
    check(o, MOVE(1))


def test_advance_tie_takes_lower_slot():
    check(make_obs(my_coins=1, my_backline=(unit("imp"), unit("guard", hp=1), unit("guard"))), MOVE(1))


def test_advance_fast_unit_that_already_attacked():
    o = make_obs(my_coins=2, my_backline=(unit("guard"), unit("rider", attacked=True)))
    check(o, MOVE(1), also_legal=(MOVE(0),))


def test_ranged_units_never_advance():
    o = make_obs(my_coins=5, my_backline=(unit("pawn"), unit("cannon")))
    assert greedy_choice(o, [END, MOVE(0), MOVE(1)]) == MOVE(0)
    assert greedy_choice(o, [END, MOVE(1)]) == END


def test_advance_respects_move_cost():
    # mule (atk 4) costs 3 coins to move; with 2 coins the guard advances
    check(make_obs(my_coins=2, my_backline=(unit("mule"), unit("guard"))), MOVE(1))
    check(make_obs(my_coins=3, my_backline=(unit("mule"), unit("guard"))), MOVE(0))


def test_advance_comes_before_base_attack():
    check(make_obs(my_coins=1, front_owner=1, frontline=(unit("guard"),), my_backline=(unit("brute"),)), MOVE(0))


def test_no_advance_when_own_frontline_full():
    o = make_obs(my_coins=5, front_owner=1, frontline=(unit("imp", attacked=True),) * 4 + (unit("pawn"),),
                 my_backline=(unit("titan"),))
    assert MOVE(0) not in spec_legal(o, TEST_CONFIG)
    check(o, ATK(Z + 4, BASE))


def test_no_advance_into_enemy_frontline():
    check(make_obs(my_coins=5, front_owner=-1, frontline=(unit("titan"),), my_backline=(unit("pawn"),)), END)


# ---------------------------------------------------------------------- rule 7: base
def test_frontline_units_attack_base_lowest_slot():
    o = make_obs(front_owner=1, frontline=(unit("pawn", attacked=True), unit("guard"), unit("brute")),
                 opp_backline=(unit("tank"),))
    check(o, ATK(Z + 1, BASE), also_legal=(ATK(Z + 2, BASE),))


def test_base_attack_ignores_defense():
    o = make_obs(front_owner=1, frontline=(unit("pawn"),), opp_backline=(unit("keep"), unit("tank")))
    check(o, ATK(Z, BASE))


def test_ranged_frontline_units_are_not_rule_7():
    # Ranged units hit the base only through rule 5 (which needs can_attack); rule 7 skips them.
    check(make_obs(front_owner=1, frontline=(unit("bow"),)), ATK(Z, BASE))
    o = make_obs(front_owner=1, frontline=(unit("bow", attacked=True), unit("guard", attacked=True)))
    assert greedy_choice(o, [END, ATK(Z, BASE), ATK(Z + 1, BASE)]) == ATK(Z + 1, BASE)
    assert greedy_choice(o, [END, ATK(Z, BASE)]) == END


# ---------------------------------------------------------------------- rule 8: end turn
@pytest.mark.parametrize("o", [
    make_obs(),
    make_obs(hand=("ogre", "titan"), my_coins=4),
    make_obs(my_backline=(unit("titan", summoned=True),), front_owner=1, frontline=(unit("guard", attacked=True),),
             opp_backline=(unit("imp"),)),
    make_obs(my_backline=(unit("rider", moved=True, attacked=True), unit("bow", attacked=True))),
])
def test_ends_turn_when_nothing_else(o):
    check(o, END)


def test_only_uses_the_given_legal_list():
    o = make_obs(hand=("titan",), my_coins=7, front_owner=1, frontline=(unit("titan"),), opp_backline=(unit("ogre"),))
    assert greedy_choice(o, [END, ATK(Z, BASE)]) == ATK(Z, BASE)
    assert greedy_choice(o, [END]) == END


# ---------------------------------------------------------------------- strength
def test_greedy_beats_random_duplicate():
    # Measured: 253/256 (0.988); per-matchup cells are gated in tests/test_decks.py (all >= 0.96).
    res = duplicate_match(GreedyAgent(CONFIG), RandomAgent(CONFIG), range(128))
    assert res["games"] == 256
    assert res["wins"] / res["games"] >= 0.9, res


# ---------------------------------------------------------------------- factory
@pytest.mark.parametrize("spec,cls", [("random", RandomAgent), ("greedy", GreedyAgent), ("lookahead", LookaheadAgent)])
def test_make_agent(spec, cls):
    agent = make_agent(spec, CONFIG)
    assert isinstance(agent, cls) and isinstance(agent, Agent)
    assert agent.name == spec
    assert getattr(agent, "needs_game", False) == (spec == "lookahead")
    assert isinstance(agent, GameAgent) == (spec == "lookahead")
    agent.reset(0)
    game = Game(CONFIG)
    game.reset(0)
    assert choose_action(agent, game) in game.legal_actions()


def test_make_agent_default_config():
    assert isinstance(make_agent("greedy"), GreedyAgent)
    assert isinstance(make_agent("random", seed=3), RandomAgent)
    la = make_agent("lookahead", seed=3)
    assert isinstance(la, LookaheadAgent) and la.config.mulligan and la.max_choice_depth == 3


@pytest.mark.parametrize("spec", ["", "Random", "greedy ", "minimax", "model.pth", "ppo:", "ppo:model.ckpt",
                                  "runs/ppo", "Lookahead", "lookahead2"])
def test_make_agent_rejects_junk(spec):
    with pytest.raises(ValueError):
        make_agent(spec, CONFIG)


# ---------------------------------------------------------------------- effect pools for the lookahead tests
def _unit(cid: str, atk: int, hp: int, cost: int = 1, nature: str = "troop", effects=(), traits=None,
          token: bool = False) -> dict:
    d = {"id": cid, "name": cid, "type": "unit", "nature": nature, "cost": cost, "attack": atk, "health": hp}
    if traits:
        d["traits"] = dict(traits)
    if token:
        d["token"] = True
    d["effects"] = [dict(e) for e in effects]
    return d


def _op(cid: str, cost: int, effects, token: bool = False) -> dict:
    d = {"id": cid, "name": cid, "type": "operation", "cost": cost, "effects": [dict(e) for e in effects]}
    if token:
        d["token"] = True
    return d


def _eff(trigger: str, action: str, target, scope: Optional[str] = None, **params) -> dict:
    e = {"trigger": trigger, "target": target, "action": action, **params}
    if scope is not None:
        e["scope"] = scope
    return e


def _chosen(side: str, kind: str = "unit") -> dict:
    return {"select": "chosen", "side": side, "kind": kind}


FX_FILLER = [_unit(f"fill{i:02d}", 1 + i % 4, 1 + (i * 3) % 5, cost=1 + i % 8,
                   nature=("troop", "fast", "ranged")[i % 3]) for i in range(14)]
FX_CARDS = FX_FILLER + [
    _unit("dummy", 1, 1),                 # vanilla bodies for hand-built positions (stats overridden there)
    _unit("post", 0, 3),                  # 0 attack: its base attack changes nothing
    _op("bolt", 1, [_eff("on_play", "damage", _chosen("enemy"), amount=2)]),
    _op("fireball", 2, [_eff("on_play", "damage", _chosen("enemy", "unit_or_base"), amount=3)]),
    _op("triple_shot", 2, [_eff("on_play", "damage", _chosen("enemy"), amount=1)] * 3),
    _op("volley", 2, [_eff("on_play", "damage", {"select": "random", "side": "enemy", "count": 2}, amount=1)]),
    _op("intel", 1, [_eff("on_play", "draw", "controller", amount=2)]),
    _op("sabotage", 1, [_eff("on_play", "discard", "opponent", amount=1)]),
    _op("rally", 2, [_eff("on_play", "buff", {"select": "all", "side": "friendly"}, atk=1, hp=1, duration="turn")]),
    _op("retreat_order", 1, [_eff("on_play", "retreat", _chosen("friendly"))]),
    _op("strip", 1, [_eff("on_play", "remove_trait", _chosen("enemy"), trait=["defense", "armor"])]),
    _op("supply", 2, [_eff("on_play", "increase_max_coins", "controller", amount=1)]),
    _unit("medic", 2, 2, cost=2, effects=[_eff("on_deploy", "heal", _chosen("friendly", "unit_or_base"), amount=3)]),
    _unit("grenadier", 2, 3, cost=3, effects=[_eff("on_deploy", "damage", _chosen("enemy"), amount=1)]),
    _unit("martyr", 1, 1, cost=1, effects=[_eff("on_death", "damage", {"select": "all", "side": "enemy"}, amount=1)]),
    _unit("bugler", 1, 2, cost=2, effects=[_eff("on_death", "summon", "controller", card="militia")]),
    _unit("scout", 1, 1, cost=1, nature="fast", effects=[_eff("on_deploy", "draw", "controller", amount=1)]),
    _unit("sniper", 1, 2, cost=2, nature="ranged",
          effects=[_eff("on_deploy", "damage", {"select": "random", "side": "enemy"}, amount=1)]),
    _unit("pinner", 2, 2, cost=2, effects=[_eff("on_deploy", "pin", _chosen("enemy"))]),
    _unit("bouncer", 2, 2, cost=3, effects=[_eff("on_deploy", "return_to_hand", _chosen("any"))]),
    _unit("watcher", 1, 3, cost=2, effects=[_eff("on_death", "buff", "self", scope="friendly", atk=1)]),
    _unit("berserker", 3, 2, cost=3, nature="fast", traits={"blitz": True, "fury": True}),
    _unit("smoker", 2, 2, cost=2, nature="ranged", traits={"smokescreen": True}),
    _unit("bulwark", 1, 4, cost=3, traits={"defense": True, "armor": 1}),
    _unit("quartermaster", 1, 2, cost=2, effects=[_eff("on_deploy", "add_card", "controller", card="ration")]),
    _unit("duelist", 2, 3, cost=3, effects=[_eff("on_attack", "damage", _chosen("enemy"), amount=1)]),
    _unit("herald", 2, 2, cost=2, effects=[_eff("on_move", "buff", _chosen("friendly"), atk=1, hp=1)]),
    _unit("drummer", 1, 4, cost=3, effects=[_eff("start_of_turn", "damage", "enemy_base", amount=1)]),
    _unit("sapper", 2, 2, cost=2, effects=[_eff("on_damaged", "damage", "event", amount=1)]),
    _unit("headhunter", 3, 2, cost=3, nature="fast", effects=[_eff("on_kill", "gain_coins", "controller", amount=1)]),
    _unit("sprout", 1, 1, cost=1, effects=[_eff("end_of_turn", "buff", "self", hp=1)]),
    _unit("titan", 7, 7, cost=7),
    _op("ration", 0, [_eff("on_play", "heal", "friendly_base", amount=2)], token=True),
    _unit("militia", 1, 1, token=True),
]
# Two units that keep damaging each other (on_damaged -> damage the source): trips a small loop guard.
LOOP_CARDS = [_unit("spiker", 1, 12, cost=4, effects=[_eff("on_damaged", "damage", "event", amount=1)])]
FX_DECKS = [
    {"name": "fx_a", "cards": {**{FX_FILLER[i]["id"]: 2 for i in range(8)}, "bolt": 3, "fireball": 2,
                               "triple_shot": 2, "volley": 2, "intel": 2, "medic": 2, "grenadier": 3, "martyr": 2,
                               "bugler": 2, "scout": 2, "sniper": 2}},
    {"name": "fx_b", "cards": {**{FX_FILLER[i]["id"]: 2 for i in range(6, 14)}, "sabotage": 2, "rally": 2,
                               "pinner": 3, "bouncer": 2, "watcher": 3, "berserker": 3, "smoker": 3,
                               "quartermaster": 3, "duelist": 3}},
]
FX_CONFIG = cards_mod.build_ruleset(FX_CARDS, FX_DECKS, mulligan=False)      # hand-built positions
FX_MULLIGAN = cards_mod.build_ruleset(FX_CARDS, FX_DECKS, mulligan=True)
FX_SMALL = cards_mod.build_ruleset(FX_CARDS + LOOP_CARDS, FX_DECKS, mulligan=True, zone_capacity=4,
                                   max_hand_size=7, max_effect_events=16)
POOLS = {"shipped": load_ruleset(mulligan=True), "fx": FX_MULLIGAN, "fx_small": FX_SMALL}
POOL_IDS = tuple(POOLS)


def has_choices(cfg: GameConfig) -> bool:
    return any(e.target.select == "chosen" for c in cfg.cards.cards for e in c.effects)
FRONT = SP.zone_capacity  # attacker/target slot of frontline position 0 (standard H=10, Z=5 layout)


def put(game: Game, owner: int, zone: str, atk: int, hp: int, card: str = "dummy", **fields) -> Unit:
    """A ready unit with exactly these stats (undamaged)."""
    return add_unit(game, owner, zone, card, atk=atk, hp=hp, max_hp=hp, **fields)


def fx_position(coins: int = 0, hand: Sequence[str] = (), enemy_base: int = 20) -> Game:
    """A FX_CONFIG game (round 1, seat 0 to act) with empty decks, so END_TURN draws nothing."""
    g = blank_game(current=0, coins=coins, config=FX_CONFIG)
    set_hand(g, 0, hand)
    g.base_hp[1] = enemy_base
    g.invalidate()
    return g


def fingerprint(g: Game) -> tuple:
    """Everything a mutation of the real game would change (hidden parts and the RNG included)."""
    return (g.render(), g.observe(0), g.observe(1), tuple(g.legal_actions()), g.rng.getstate(), g.num_steps,
            tuple(map(tuple, g.hands)), tuple(map(tuple, g.deck_cards)), len(g.queue))


def perturb_hidden(game: Game, observer: int, rng: random.Random, new_decklist: bool) -> Game:
    """A clone that differs only in what SPEC 5 hides from `observer`: the opponent's unknown hand cards
    and deck are re-dealt (same sizes; from a fresh decklist containing `revealed` if `new_decklist`),
    the opponent's mulligan marks (while it decides) and the game RNG are replaced."""
    g = game.clone()
    o = 1 - observer
    n = len(g.config.cards)
    rest = Counter(g.hands[o])
    known = []
    for c in range(n):
        k = min(g.known_hand[o][c], rest[c])
        known += [c] * k
        rest[c] -= k
    unknown_hand = list(rest.elements())
    n_deck = len(g.deck_cards[o])
    if new_decklist:
        deck = cards_mod.generate_deck(rng, g.config, required=g.revealed[o])
        pool = Counter(deck)
        pool.subtract({c: g.revealed[o][c] for c in range(n)})
        pool = list(pool.elements())
        g.decklists = tuple(deck if s == o else d for s, d in enumerate(g.decklists))
        g.deck_ids = tuple(-1 if s == o else d for s, d in enumerate(g.deck_ids))
    else:
        pool = unknown_hand + list(g.deck_cards[o])
    rng.shuffle(pool)
    assert len(pool) >= len(unknown_hand) + n_deck
    g.hands[o] = sorted(known + pool[:len(unknown_hand)])
    g.deck_cards[o] = pool[len(unknown_hand):len(unknown_hand) + n_deck]
    if g.phase == MULLIGAN and g.current == o:
        g.mulligan_marks = {i for i in range(len(g.hands[o])) if rng.random() < 0.5}
    g.rng = random.Random(rng.getrandbits(64))
    g.invalidate()
    return g


def test_effect_pools_cover_choices_triggers_and_keywords():
    pool = FX_SMALL.cards
    effects = [e for c in pool.cards for e in c.effects]
    assert {e.trigger for e in effects} == {"on_play", "on_deploy", "on_death", "on_attack", "on_damaged", "on_move",
                                            "on_kill", "start_of_turn", "end_of_turn"}
    assert {e.target.select for e in effects} == {"chosen", "random", "all", "self", "event"}
    assert {"damage", "heal", "buff", "draw", "discard", "summon", "pin", "return_to_hand", "retreat", "add_card",
            "remove_trait", "gain_coins", "increase_max_coins"} <= {e.action for e in effects}
    assert any(c.blitz and c.fury for c in pool.cards) and any(c.smokescreen for c in pool.cards)
    assert any(c.defense and c.armor for c in pool.cards) and any(c.is_operation for c in pool.cards)


# ---------------------------------------------------------------------- lookahead: only the player's information
@pytest.mark.parametrize("pool", POOL_IDS)
def test_lookahead_uses_only_the_players_information(pool):
    """At every lookahead decision (mulligan, main phase, pending choices) in seeded games, a game whose
    hidden parts (opponent's unknown hand, deck contents and order, decklist, marks, RNG) were replaced
    gives the same action, the same values and the same agent RNG state; the real game is untouched."""
    cfg = POOLS[pool]
    game, rng = Game(cfg), random.Random(11)
    phases = Counter()
    for seed in range(6):
        game.reset(seed, cards_mod.sample_deal(seed, cfg, 0.5))
        la, opp = LookaheadAgent(cfg, seed=seed), RandomAgent(cfg, seed=seed)
        while not game.done:
            p = game.current_player()
            if p != seed % 2:
                game.step(opp.act(game.observe(p), game.legal_actions()))
                continue
            phases[game.phase] += 1
            before, state = fingerprint(game), la.rng.getstate()
            action = la.act_game(game, p)
            after = la.rng.getstate()
            assert fingerprint(game) == before, "act_game mutated the real game"
            for new_decklist in (False, True):
                twin = perturb_hidden(game, p, rng, new_decklist)
                la.rng.setstate(state)
                assert la.act_game(twin, p) == action, (pool, seed, game.describe(action), new_decklist)
                assert la.rng.getstate() == after
            if game.phase != MULLIGAN and phases[game.phase] % 3 == 0:
                la.rng.setstate(state)
                values = la.action_values(game, p)
                la.rng.setstate(state)
                assert la.action_values(perturb_hidden(game, p, rng, True), p) == values
                la.rng.setstate(after)
            game.step(action)
    assert phases[MULLIGAN] >= 6 and phases[MAIN] >= 100, phases
    assert phases[CHOICE] >= 3 or not has_choices(cfg), phases


def test_lookahead_determinizes_once_for_the_player_to_act(monkeypatch):
    """The only window on the real game's hidden state is `game.determinize(player, agent rng)`: called
    once per decision with several legal actions, never with a single one or during the mulligan."""
    calls = []
    real = Game.determinize

    def spy(self, player, rng):
        calls.append((self, player, rng))
        return real(self, player, rng)

    monkeypatch.setattr(Game, "determinize", spy)
    cfg = FX_MULLIGAN
    game, la = Game(cfg), LookaheadAgent(cfg, seed=3)
    expected = 0
    for seed in range(3):
        game.reset(seed, cards_mod.sample_deal(seed, cfg, 0.5))
        while not game.done:
            p = game.current_player()
            searched = game.phase != MULLIGAN and len(game.legal_actions()) > 1
            calls.clear()
            game.step(choose_action(la, game))
            assert len(calls) == int(searched)
            if calls:
                assert calls[0][0] is game and calls[0][1] == p and calls[0][2] is la.rng
            expected += searched
    assert expected > 100


def test_lookahead_samples_random_effects_from_its_own_rng():
    """A random effect is evaluated on the determinized RNG: the agent's seed matters, the real game's RNG
    does not (SPEC 4 determinize step 5)."""
    g = fx_position(coins=2, hand=["volley"])
    for atk, hp in ((1, 1), (1, 1), (3, 3)):
        put(g, 1, "back", atk, hp)
    twin = g.clone()
    twin.rng = random.Random(987654321)
    la = LookaheadAgent(FX_CONFIG)
    seen = set()
    for seed in range(16):
        la.reset(seed)
        v = la.action_values(g, 0)
        la.reset(seed)
        assert la.action_values(twin, 0) == v
        # V = 0.5 * -(enemy atk+hp left): both 1/1s die (-3.0) or a 1/1 and the 3/3 are hit (-3.5)
        assert v[END] == -5.0 + 1.0 and v[PLAY(0)] in (-3.0, -3.5)
        seen.add(v[PLAY(0)])
    assert seen == {-3.0, -3.5}


# ---------------------------------------------------------------------- lookahead: evaluation, lethal, trades
def test_lookahead_evaluation_formula():
    g = fx_position(hand=["dummy", "dummy", "bolt"])
    set_hand(g, 1, ["dummy"])
    g.base_hp = [18, 11]
    put(g, 0, "back", 2, 3)
    put(g, 1, "back", 4, 2)
    put(g, 1, "front", 5, 5)
    la = LookaheadAgent(FX_CONFIG)
    # (18 - 11) + 0.5 * (5 - 16) + (3 - 1)
    assert la.evaluate(g, 0) == 3.5 and la.evaluate(g, 1) == -3.5
    g.frontline[0].owner = 0  # the frontline now counts for seat 0
    g.front_owner = 0
    g.invalidate()
    assert la.evaluate(g, 0) == 7 + 0.5 * (15 - 6) + 2
    g.base_hp[1] = 2
    g.frontline[0].summoned = False
    g.invalidate()
    g.step(SP.attack(FRONT, BASE))
    assert g.done and la.evaluate(g, 0) == 1000.0 and la.evaluate(g, 1) == -1000.0


def test_lookahead_finds_lethal_attack():
    g = fx_position(enemy_base=3)
    put(g, 0, "front", 3, 3)
    put(g, 1, "back", 1, 1)  # killing it is also an improvement
    la = LookaheadAgent(FX_CONFIG, seed=0)
    assert la.action_values(g, 0)[SP.attack(FRONT, BASE)] == 1000.0
    a = choose_action(la, g)
    assert a == SP.attack(FRONT, BASE)
    g.step(a)
    assert g.done and g.winner() == 0


def test_lookahead_finds_lethal_through_an_operation_choice():
    """fireball (3 damage to a chosen enemy unit or base): PLAY, then CHOOSE the enemy base."""
    g = fx_position(coins=2, hand=["fireball"], enemy_base=3)
    put(g, 1, "back", 3, 3)
    la = LookaheadAgent(FX_CONFIG, seed=0)
    values = la.action_values(g, 0)
    assert values == {END: 17 - 3 + 1, PLAY(0): 1000.0}
    actions = []
    while not g.done:
        actions.append(choose_action(la, g))
        g.step(actions[-1])
    assert actions == [PLAY(0), SP.choose(SP.ENEMY_BASE_CHOICE)] and g.winner() == 0


def test_lookahead_takes_the_favourable_trade():
    # backline troop 3/3 reaches only the enemy frontline: 3/3 (both die) or 2/2 (it survives at 1)
    g = fx_position()
    put(g, 0, "back", 3, 3)
    put(g, 1, "front", 3, 3)
    put(g, 1, "front", 2, 2)
    la = LookaheadAgent(FX_CONFIG, seed=0)
    assert la.action_values(g, 0) == {END: 0.5 * (6 - 10), SP.attack(0, FRONT): 0.5 * (0 - 4),
                                      SP.attack(0, FRONT + 1): 0.5 * (4 - 6)}
    assert choose_action(la, g) == SP.attack(0, FRONT + 1)


def test_lookahead_avoids_the_losing_trade():
    g = fx_position()
    put(g, 0, "back", 2, 2)
    put(g, 1, "front", 5, 5)
    la = LookaheadAgent(FX_CONFIG, seed=0)
    assert la.action_values(g, 0) == {END: 0.5 * (4 - 10), SP.attack(0, FRONT): 0.5 * (0 - 8)}
    assert choose_action(la, g) == END


def test_lookahead_uses_the_engines_combat_rules():
    """Armor and ranged no-return-damage come from the engine: the armored 2/4 survives the 3/3's hit
    (3 - 1 armor) and kills it only because a ranged attacker shot it first."""
    g = fx_position()
    put(g, 0, "back", 2, 1, nature=RANGED)
    put(g, 0, "back", 3, 3)
    put(g, 1, "front", 2, 3, card="bulwark", armor=1, defense=True)
    la = LookaheadAgent(FX_CONFIG, seed=0)
    v = la.action_values(g, 0)
    # ranged 2/1 into the 2/3 (armor 1): 1 damage, no return damage
    assert v[SP.attack(0, FRONT)] == v[END] + 0.5
    # troop 3/3 into it: 2 damage (2/1 left), takes 2 back (3/1)
    assert v[SP.attack(1, FRONT)] == v[END] + 0.5 * 2 - 0.5 * 2


# ---------------------------------------------------------------------- lookahead: choices
def test_lookahead_plays_a_damage_operation_on_the_best_target():
    """bolt (2 damage, chosen enemy unit): 5/5 -> 5/3, 2/2 dies, 1/1 dies. Killing the 2/2 is best, and the
    PLAY is only worth it because the choice is resolved inside the lookahead."""
    g = fx_position(coins=1, hand=["bolt"])
    for atk, hp in ((5, 5), (2, 2), (1, 1)):
        put(g, 1, "back", atk, hp)
    la = LookaheadAgent(FX_CONFIG, seed=0)
    assert la.action_values(g, 0) == {END: 1.0 + 0.5 * -16, PLAY(0): 0.5 * -12}
    assert choose_action(la, g) == PLAY(0)
    g.step(PLAY(0))
    assert g.phase == CHOICE
    assert choose_action(la, g) == SP.choose(1)
    g.step(SP.choose(1))
    assert [(u.atk, u.hp) for u in g.backline[1]] == [(5, 5), (1, 1)]
    # without resolving its own choice, the PLAY (card spent, nothing happened yet) looks worse than END_TURN
    blind = LookaheadAgent(FX_CONFIG, seed=0, max_choice_depth=0)
    g2 = fx_position(coins=1, hand=["bolt"])
    for atk, hp in ((5, 5), (2, 2), (1, 1)):
        put(g2, 1, "back", atk, hp)
    assert blind.action_values(g2, 0) == {END: -7.0, PLAY(0): -8.0} and choose_action(blind, g2) == END


@pytest.mark.parametrize("depth,value", [(0, -3.0), (1, -2.0), (2, -1.0), (3, 0.0), (4, 0.0)])
def test_lookahead_resolves_up_to_three_chained_choices(depth, value):
    """triple_shot: three chosen 1-damage clauses against three 1/1s. Each greedily resolved choice kills
    one; the default depth (3) sees all three."""
    g = fx_position(coins=2, hand=["triple_shot"])
    for _ in range(3):
        put(g, 1, "back", 1, 1)
    la = LookaheadAgent(FX_CONFIG, seed=0, max_choice_depth=depth)
    assert la.action_values(g, 0) == {END: -2.0, PLAY(0): value}
    if depth == 3:
        assert LookaheadAgent(FX_CONFIG).max_choice_depth == 3


def test_lookahead_choice_on_its_own_unit_and_base():
    """medic (heal 3, chosen friendly unit or base): the damaged 4/1 (max 6) gains 3 hp = +1.5; the base at
    15 gains 3 = +3.0 -> the base."""
    g = fx_position(coins=2, hand=["medic"])
    g.base_hp[0] = 15
    add_unit(g, 0, "back", "dummy", atk=4, hp=1, max_hp=6)
    g.invalidate()
    la = LookaheadAgent(FX_CONFIG, seed=0)
    assert choose_action(la, g) == PLAY(0)
    g.step(PLAY(0))
    assert choose_action(la, g) == SP.choose(SP.OWN_BASE_CHOICE)


# ---------------------------------------------------------------------- lookahead: tie-breaks
def test_lookahead_tie_break_kind_priority():
    """All four kinds score the same (0-attack base hit, a 1/1 played for one card, a free move, END_TURN
    with empty decks): ATTACK > PLAY > MOVE > END_TURN, although ATTACK has the highest index."""
    g = fx_position(coins=2, hand=["dummy"])
    post = put(g, 0, "front", 0, 3, card="post")
    put(g, 0, "back", 1, 1)
    la = LookaheadAgent(FX_CONFIG, seed=0)
    values = la.action_values(g, 0)
    attack = SP.attack(FRONT, BASE)
    assert set(values) == {END, PLAY(0), SP.MOVE0, attack} and len(set(values.values())) == 1
    assert attack > SP.MOVE0 > PLAY(0) > END
    assert la.act_game(g, 0) == attack
    post.attacks = 1
    g.invalidate()
    assert la.act_game(g, 0) == PLAY(0)
    set_hand(g, 0, [])
    assert la.act_game(g, 0) == SP.MOVE0


def test_lookahead_tie_break_lower_index_within_a_kind():
    g = fx_position(coins=1, hand=["dummy", "dummy"])
    put(g, 0, "front", 0, 3, card="post")
    put(g, 0, "front", 0, 3, card="post")
    la = LookaheadAgent(FX_CONFIG, seed=0)
    assert la.act_game(g, 0) == SP.attack(FRONT, BASE)
    for u in g.frontline:
        u.attacks = 1
    g.frontline.clear()
    g.front_owner = None
    put(g, 0, "back", 1, 1)
    put(g, 0, "back", 1, 1)
    g.coins[0] = 1
    g.invalidate()
    values = la.action_values(g, 0)
    assert values[PLAY(0)] == values[PLAY(1)] == values[SP.MOVE0] == values[SP.MOVE0 + 1] == values[END]
    assert la.act_game(g, 0) == PLAY(0)
    set_hand(g, 0, [])
    assert la.act_game(g, 0) == SP.MOVE0
    # equal CHOOSE options: the lower slot
    g = fx_position(coins=1, hand=["bolt"])
    put(g, 1, "back", 2, 2)
    put(g, 1, "back", 2, 2)
    g.step(PLAY(0))
    assert la.act_game(g, 0) == SP.choose(0)


# ---------------------------------------------------------------------- lookahead: mulligan, reproducibility
@pytest.mark.parametrize("pool", ["shipped", "fx"])
def test_lookahead_mulligan_replaces_every_card_costing_five_or_more(pool):
    cfg = POOLS[pool]
    game, la = Game(cfg), LookaheadAgent(cfg, seed=0)
    cost = [c.cost for c in cfg.cards.cards]
    replaced = kept = 0
    for seed in range(40):
        game.reset(seed, cards_mod.sample_deal(seed, cfg, 0.5))
        for _ in range(2):
            p = game.current_player()
            assert game.phase == MULLIGAN
            hand = list(game.hands[p])
            expected = [SP.mulligan(i) for i, c in enumerate(hand) if cost[c] >= 5] + [SP.CONFIRM]
            state = la.rng.getstate()
            got = []
            while game.phase == MULLIGAN and game.current_player() == p:
                got.append(choose_action(la, game))
                game.step(got[-1])
            assert got == expected, (seed, [cost[c] for c in hand])
            assert la.rng.getstate() == state  # the mulligan is a fixed rule: no determinization
            replaced += len(expected) - 1
            kept += len(hand) - (len(expected) - 1)
        assert game.phase == MAIN and game.turn == 1
    assert replaced >= 15 and kept >= 100


def test_lookahead_is_reproducible_for_a_seed():
    cfg = FX_MULLIGAN

    def traces(agents) -> list:
        out = []
        for seed in range(4):
            game = Game(cfg)
            game.reset(seed, cards_mod.sample_deal(seed, cfg, 1.0))
            for i, agent in enumerate(agents):
                agent.reset(100 * seed + i)
            actions = []
            while not game.done:
                actions.append(choose_action(agents[game.current_player()], game))
                game.step(actions[-1])
            out.append(actions)
        return out

    a, b = LookaheadAgent(cfg), LookaheadAgent(cfg)
    first = traces((a, b))
    assert traces((a, b)) == first  # reset(seed) reseeds: earlier games leave no trace
    assert traces((LookaheadAgent(cfg, seed=5), LookaheadAgent(cfg))) == first
    seeded, reseeded = LookaheadAgent(cfg, seed=42), LookaheadAgent(cfg, seed=1)
    reseeded.reset(42)
    assert seeded.rng.getstate() == reseeded.rng.getstate()
    reseeded.reset(None)  # None keeps the stream
    assert seeded.rng.getstate() == reseeded.rng.getstate()


# ---------------------------------------------------------------------- lookahead: legality and strength
@pytest.mark.parametrize("pool", POOL_IDS)
def test_lookahead_is_legal_and_leaves_the_game_untouched(pool):
    """Positions reached by mostly random play on effect-heavy pools (random and fixed decks, mulligan
    on): every lookahead action is legal and the real game is never mutated."""
    cfg = POOLS[pool]
    game, la, pol = Game(cfg), LookaheadAgent(cfg, seed=1), random.Random(5)
    phases = Counter()
    for seed in range(10):
        game.reset(seed, cards_mod.sample_deal(seed, cfg, 0.75))
        while not game.done:
            legal, before = game.legal_actions(), fingerprint(game)
            action = choose_action(la, game)
            assert action in legal, (pool, seed, game.describe(action))
            assert fingerprint(game) == before
            phases[game.phase] += 1
            game.step(action if pol.random() < 0.3 else pol.choice(legal))
    assert phases[MAIN] > 700 and phases[MULLIGAN] >= 20, phases
    assert phases[CHOICE] >= 10 or not has_choices(cfg), phases


@pytest.mark.parametrize("pool", POOL_IDS)
@pytest.mark.parametrize("spec", ["random", "greedy", "lookahead"])
def test_agents_play_legal_full_games_on_effect_pools(pool, spec):
    """Every agent handles the mulligan, pending choices and operations (random and fixed decks)."""
    cfg = POOLS[pool]
    game = Game(cfg)
    agent = make_agent(spec, cfg, seed=1)
    others = [make_agent(s, cfg, seed=2) for s in ("random", "greedy", "lookahead")]
    for k in range(12):
        opp = others[k % 3]
        agents = (agent, opp) if k % 2 == 0 else (opp, agent)
        assert play_game(game, agents, k, cards_mod.sample_deal(k, cfg, 0.5)) in (0, 1, DRAW)


def test_lookahead_beats_random_on_the_shipped_decks():
    # measured: 64/64
    res = duplicate_match(LookaheadAgent(CONFIG), RandomAgent(CONFIG), range(32))
    assert res["games"] == 64 and res["wins"] / res["games"] >= 0.95, res


@pytest.mark.parametrize("pool", ["fx", "fx_small"])
def test_lookahead_beats_random_on_effect_pools(pool):
    cfg = POOLS[pool]
    game, la, rnd = Game(cfg), LookaheadAgent(cfg), RandomAgent(cfg)
    wins = 0
    for k in range(16):
        for seat in (0, 1):
            agents = (la, rnd) if seat == 0 else (rnd, la)
            wins += play_game(game, agents, k, cards_mod.sample_deal(k, cfg, 0.5)) == seat
    assert wins >= 30, wins


# ---------------------------------------------------------------------- choose_action dispatch
class _ObsProbe:
    name = "obs_probe"

    def __init__(self):
        self.calls = []

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def act(self, obs: Observation, legal_actions: Sequence[int]) -> int:
        self.calls.append((obs, list(legal_actions)))
        return legal_actions[-1]


class _GameProbe(_ObsProbe):
    name = "game_probe"
    needs_game = True

    def act(self, obs: Observation, legal_actions: Sequence[int]) -> int:
        raise AssertionError("act() must not be called for a needs_game agent")

    def act_game(self, game: Game, player: int) -> int:
        self.calls.append((game, player))
        return game.legal_actions()[0]


def test_choose_action_dispatches_on_needs_game():
    game = Game(CONFIG)
    game.reset(3)
    p = game.current_player()
    obs_probe, game_probe = _ObsProbe(), _GameProbe()
    assert choose_action(obs_probe, game) == game.legal_actions()[-1]
    assert obs_probe.calls == [(game.observe(p), game.legal_actions())]
    assert choose_action(game_probe, game) == game.legal_actions()[0]
    assert len(game_probe.calls) == 1 and game_probe.calls[0][0] is game and game_probe.calls[0][1] == p
    off = _ObsProbe()
    off.needs_game = False
    choose_action(off, game)
    assert len(off.calls) == 1
    assert isinstance(obs_probe, Agent) and not isinstance(obs_probe, GameAgent) and isinstance(game_probe, GameAgent)
    # the player passed is always the player to act, also for the second seat
    game.step(END)
    choose_action(game_probe, game)
    assert game_probe.calls[-1][1] == 1 - p


def test_choose_action_rejects_a_finished_game():
    game = Game(CONFIG)
    play_game(game, (RandomAgent(CONFIG), RandomAgent(CONFIG)), 0)
    with pytest.raises(ValueError):
        choose_action(RandomAgent(CONFIG), game)
    with pytest.raises(ValueError):
        LookaheadAgent(CONFIG).act_game(game, 0)


def test_lookahead_needs_the_game():
    la = LookaheadAgent(CONFIG, seed=0)
    game = Game(CONFIG)
    game.reset(0)
    p = game.current_player()
    with pytest.raises(TypeError, match="choose_action"):
        la.act(game.observe(p), game.legal_actions())
    with pytest.raises(ValueError):
        la.act_game(game, 1 - p)
    with pytest.raises(ValueError):
        LookaheadAgent(CONFIG, max_choice_depth=-1)
    assert choose_action(la, game) in game.legal_actions()
