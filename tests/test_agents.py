"""Agent tests: legality over full games, greedy v2 rules on constructed positions, factory."""
from __future__ import annotations

from typing import Optional, Sequence

import pytest

from cardgame.actions import ActionSpace
from cardgame.agents import Agent, GreedyAgent, RandomAgent, make_agent
from cardgame.cards import FAST, RANGED, TROOP, CardDef, CardPool, GameConfig, load_ruleset
from cardgame.engine import DRAW, Game, Observation, Unit, UnitView

CONFIG = load_ruleset()
N_DECKS = CONFIG.n_decks


# ---------------------------------------------------------------------- helpers
def deal_decks(k: int) -> tuple:
    """Deck pair of deal k: every ordered pair once per N_DECKS^2 deals."""
    return divmod(k % (N_DECKS * N_DECKS), N_DECKS)


def play_game(game: Game, agents: Sequence[Agent], seed: int, decks: Optional[tuple] = None) -> Optional[int]:
    """Play one full game, asserting every chosen action is legal. Returns the winner."""
    game.reset(seed, decks=decks)
    for i, agent in enumerate(agents):
        agent.reset(2 * seed + i)
    while not game.done:
        p = game.current_player()
        legal = game.legal_actions()
        action = agents[p].act(game.observe(p), legal)
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
    """UnitView of a unit of card `c`, with can_move/can_attack from the SPEC section 2 nature table."""
    if c.nature == FAST:
        can_move, can_attack = not summoned and not moved, not summoned and not attacked
    else:
        can_move = can_attack = not summoned and not moved and not attacked
    return UnitView(c.index, atk, hp, c.health, c.armor, c.defense, c.nature, c.move_cost, summoned, moved,
                    attacked, can_move, can_attack)


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
TEST_CONFIG = GameConfig(cards=TEST_POOL, decks=((),))
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
@pytest.mark.parametrize("spec", ["random", "greedy"])
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
    """The constructed-position tests rely on spec_legal and flags_view; check both at real states."""
    game = Game(CONFIG)
    agents = (GreedyAgent(CONFIG), RandomAgent(CONFIG, seed=0))
    checked = views = 0
    seen_natures = set()
    for k in range(48):
        game.reset(k, decks=deal_decks(k))
        while not game.done:
            p = game.current_player()
            obs, legal = game.observe(p), game.legal_actions()
            assert spec_legal(obs, CONFIG) == legal
            assert spec_legal(game.observe(1 - p), CONFIG) == []
            for v in obs.my_backline + obs.frontline + obs.opp_backline:
                c = CONFIG.cards[v.card]
                assert flags_view(c, v.atk, v.hp, v.summoned, v.moved, v.attacked) == v
                seen_natures.add(v.nature)
                views += 1
            checked += 1
            game.step(agents[(p + k) % 2].act(obs, legal))
        assert spec_legal(game.observe(0), CONFIG) == [] == game.legal_actions()
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
    and Defense units get killed while shielding."""
    game, agent = Game(CONFIG), GreedyAgent(CONFIG)
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
    # measured with the shipped decks: 104 / 431 / 96
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
@pytest.mark.parametrize("spec,cls", [("random", RandomAgent), ("greedy", GreedyAgent)])
def test_make_agent(spec, cls):
    agent = make_agent(spec, CONFIG)
    assert isinstance(agent, cls) and isinstance(agent, Agent)
    assert agent.name == spec
    agent.reset(0)
    game = Game(CONFIG)
    game.reset(0)
    legal = game.legal_actions()
    assert agent.act(game.observe(game.current_player()), legal) in legal


def test_make_agent_default_config():
    assert isinstance(make_agent("greedy"), GreedyAgent)
    assert isinstance(make_agent("random", seed=3), RandomAgent)


@pytest.mark.parametrize("spec", ["", "Random", "greedy ", "minimax", "model.pth", "ppo:", "ppo:model.ckpt",
                                  "runs/ppo"])
def test_make_agent_rejects_junk(spec):
    with pytest.raises(ValueError):
        make_agent(spec, CONFIG)
