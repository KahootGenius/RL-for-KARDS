"""Agent tests: legality over full games, greedy priorities on constructed positions, factory."""
from __future__ import annotations

from typing import Optional, Sequence

import pytest

from cardgame.actions import ActionKind, ActionSpace
from cardgame.agents import Agent, GreedyAgent, RandomAgent, make_agent
from cardgame.cards import CardDef, CardPool, GameConfig, load_ruleset
from cardgame.engine import DRAW, Game, Observation, UnitView

CONFIG = load_ruleset()


# ---------------------------------------------------------------------- helpers
def play_game(game: Game, agents: Sequence[Agent], seed: int) -> Optional[int]:
    """Play one full game, asserting every chosen action is legal. Returns the winner."""
    game.reset(seed)
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


def duplicate_match(agent_a: Agent, agent_b: Agent, seeds: Sequence[int],
                    config: GameConfig = CONFIG) -> dict:
    """Each seed is played twice with seats swapped. Returns A's wins/draws/losses."""
    game = Game(config)
    out = {"wins": 0, "draws": 0, "losses": 0, "games": 0}
    for seed in seeds:
        for a_seat in (0, 1):
            agents = (agent_a, agent_b) if a_seat == 0 else (agent_b, agent_a)
            w = play_game(game, agents, seed)
            out["games"] += 1
            if w == DRAW:
                out["draws"] += 1
            elif w == a_seat:
                out["wins"] += 1
            else:
                out["losses"] += 1
    return out


def spec_legal(o: Observation, config: GameConfig) -> list:
    """Legal actions derived from an Observation per the SPEC section 2 action table."""
    if o.done or not o.is_my_turn:
        return []
    sp = ActionSpace(config.max_hand_size, config.zone_capacity)
    Z = config.zone_capacity
    cost = [c.cost for c in config.cards.cards]
    legal = [sp.END_TURN]
    if len(o.my_backline) < Z:
        legal += [sp.encode(ActionKind.PLAY, i) for i, c in enumerate(o.hand) if cost[c] <= o.my_coins]
    if o.front_owner == 0 or (o.front_owner == 1 and len(o.frontline) < Z):
        legal += [sp.encode(ActionKind.MOVE, j) for j, u in enumerate(o.my_backline) if u.ready]
    if o.front_owner == 1:
        ready = [j for j, u in enumerate(o.frontline) if u.ready]
        legal += [sp.encode(ActionKind.ATTACK_BASE, j) for j in ready]
        legal += [sp.encode(ActionKind.FRONT_ATTACK, j, k) for j in ready for k in range(len(o.opp_backline))]
    elif o.front_owner == -1:
        ready = [j for j, u in enumerate(o.my_backline) if u.ready]
        legal += [sp.encode(ActionKind.BACK_ATTACK, j, k) for j in ready for k in range(len(o.frontline))]
    return sorted(legal)


# Small hand-made pool so tie-breaks can be exercised independently of the bundled cards.
TEST_CARDS = (  # id, cost, attack, health
    ("imp", 1, 1, 1),
    ("pawn", 1, 1, 2),
    ("guard", 2, 2, 3),
    ("brute", 3, 3, 3),
    ("tank", 3, 2, 6),
    ("ogre", 5, 5, 5),
    ("titan", 7, 7, 7),
)
TEST_POOL = CardPool(tuple(CardDef(i, cid, cid.title(), "unit", cost, atk, hp)
                           for i, (cid, cost, atk, hp) in enumerate(TEST_CARDS)))
TEST_CONFIG = GameConfig(cards=TEST_POOL, decks=((), ()))
CARD = {c.id: c.index for c in TEST_POOL.cards}
SP = ActionSpace(TEST_CONFIG.max_hand_size, TEST_CONFIG.zone_capacity)
END = SP.END_TURN


def PLAY(i: int) -> int:
    return SP.encode(ActionKind.PLAY, i)


def MOVE(j: int) -> int:
    return SP.encode(ActionKind.MOVE, j)


def BASE(j: int) -> int:
    return SP.encode(ActionKind.ATTACK_BASE, j)


def FRONT(j: int, k: int) -> int:
    return SP.encode(ActionKind.FRONT_ATTACK, j, k)


def BACK(j: int, k: int) -> int:
    return SP.encode(ActionKind.BACK_ATTACK, j, k)


def unit(card_id: str, atk: Optional[int] = None, hp: Optional[int] = None, ready: bool = True) -> UnitView:
    c = TEST_POOL[CARD[card_id]]
    return UnitView(c.index, c.attack if atk is None else atk, c.health if hp is None else hp, ready)


def make_obs(hand: Sequence[str] = (), **fields) -> Observation:
    values = dict(player=0, is_my_turn=True, went_first=True, round=6, my_coins=0, opp_coins=0,
                  my_base_hp=20, opp_base_hp=20, hand=tuple(CARD[c] for c in hand), opp_hand_size=4,
                  my_deck_size=30, opp_deck_size=30, my_backline=(), opp_backline=(), frontline=(),
                  front_owner=0, done=False, result=0)
    values.update(fields)
    return Observation(**values)


def greedy_choice(o: Observation, legal: Optional[Sequence[int]] = None) -> int:
    legal = spec_legal(o, TEST_CONFIG) if legal is None else legal
    action = GreedyAgent(TEST_CONFIG).act(o, legal)
    assert action in legal
    return action


def describe(action: int) -> str:
    return SP.describe(action)


# ---------------------------------------------------------------------- legality over full games
@pytest.mark.parametrize("spec", ["random", "greedy"])
def test_agent_only_returns_legal_actions(spec):
    """200 full games per agent: 100 mirror games, 100 against the other baseline (seats alternate)."""
    game = Game(CONFIG)
    agent, mirror = make_agent(spec, CONFIG, seed=1), make_agent(spec, CONFIG, seed=2)
    other = make_agent("greedy" if spec == "random" else "random", CONFIG, seed=3)
    results = []
    for seed in range(200):
        if seed < 100:
            agents = (agent, mirror)
        else:
            agents = (agent, other) if seed % 2 == 0 else (other, agent)
        results.append(play_game(game, agents, seed))
    assert all(w in (0, 1, DRAW) for w in results)


def test_spec_legal_helper_matches_engine():
    """The constructed-position tests below rely on spec_legal; check it against the engine."""
    game = Game(CONFIG)
    agents = (GreedyAgent(CONFIG), RandomAgent(CONFIG, seed=0))
    checked = 0
    for seed in range(30):
        game.reset(seed)
        while not game.done:
            p = game.current_player()
            obs, legal = game.observe(p), game.legal_actions()
            assert spec_legal(obs, CONFIG) == legal
            checked += 1
            game.step(agents[(p + seed) % 2].act(obs, legal))
    assert checked > 1000


def test_greedy_is_deterministic():
    game = Game(CONFIG)

    def trace(seed: int) -> list:
        game.reset(seed)
        agents, actions = (GreedyAgent(CONFIG), GreedyAgent(CONFIG)), []
        while not game.done:
            p = game.current_player()
            actions.append(agents[p].act(game.observe(p), game.legal_actions()))
            game.step(actions[-1])
        return actions

    for seed in (0, 7, 123):
        assert trace(seed) == trace(seed)


# ---------------------------------------------------------------------- greedy priority 1: play
def test_plays_highest_cost_affordable_card():
    o = make_obs(hand=("imp", "guard", "brute", "ogre"), my_coins=4)
    assert greedy_choice(o) == PLAY(2)  # brute (3); ogre (5) is unaffordable


@pytest.mark.parametrize("hand,coins,expected", [
    (("brute", "tank"), 3, PLAY(1)),           # same cost: tank has higher atk+hp (8 > 6)
    (("tank", "brute"), 3, PLAY(0)),           # slot order does not matter, stats do
    (("imp", "pawn"), 1, PLAY(1)),             # pawn 3 > imp 2
    (("guard", "guard", "imp"), 2, PLAY(0)),   # identical cards: lower slot
    (("pawn", "guard", "guard"), 9, PLAY(1)),
])
def test_play_tie_breaks(hand, coins, expected):
    assert greedy_choice(make_obs(hand=hand, my_coins=coins)) == expected


def test_play_comes_before_trades_moves_and_base_attacks():
    o = make_obs(hand=("imp",), my_coins=1, front_owner=1, frontline=(unit("titan"),),
                 my_backline=(unit("brute"),), opp_backline=(unit("ogre"),))
    legal = spec_legal(o, TEST_CONFIG)
    assert {PLAY(0), FRONT(0, 0), MOVE(0), BASE(0)} <= set(legal)
    assert greedy_choice(o, legal) == PLAY(0)


def test_does_not_play_when_play_is_not_legal():
    # Full backline (5 exhausted units): PLAY is not offered even though the card is affordable.
    o = make_obs(hand=("imp",), my_coins=5, my_backline=tuple(unit("pawn", ready=False) for _ in range(5)))
    assert greedy_choice(o) == END


# ---------------------------------------------------------------------- greedy priority 2: trades
def test_prefers_highest_gain_trade():
    o = make_obs(front_owner=1, frontline=(unit("titan"),),
                 opp_backline=(unit("imp"), unit("ogre"), unit("guard")))
    assert greedy_choice(o) == FRONT(0, 1)  # gains 1, 5, 2


def test_trade_where_cheaper_attacker_dies_for_costlier_target():
    # guard (cost 2, 2/3) kills a damaged brute (cost 3, 3/2) and dies: gain 3 - 2 = 1.
    o = make_obs(front_owner=-1, frontline=(unit("brute", hp=2),), my_backline=(unit("guard"),))
    assert greedy_choice(o) == BACK(0, 0)


def test_trade_gain_tie_prefers_costlier_target():
    # Slot 0: ogre card (cost 5) as 3/9 kills brute (cost 3) and survives -> gain 3, target cost 3.
    # Slot 1: guard card (cost 2) as 5/1 kills ogre (cost 5) and dies     -> gain 3, target cost 5.
    o = make_obs(front_owner=1, frontline=(unit("ogre", atk=3, hp=9), unit("guard", atk=5, hp=1)),
                 opp_backline=(unit("brute"), unit("ogre")))
    assert greedy_choice(o) == FRONT(1, 1)
    assert FRONT(0, 0) < FRONT(1, 1)  # the pick is not just the lowest index


def test_trade_full_tie_takes_lowest_action_index():
    o = make_obs(front_owner=1, frontline=(unit("titan"), unit("titan")), opp_backline=(unit("guard"), unit("guard")))
    assert greedy_choice(o) == FRONT(0, 0)
    o = make_obs(front_owner=-1, frontline=(unit("imp"), unit("imp")),
                 my_backline=(unit("pawn", ready=False), unit("brute"), unit("brute")))
    assert greedy_choice(o) == BACK(1, 0)


def test_trade_from_backline_against_enemy_frontline():
    o = make_obs(front_owner=-1, frontline=(unit("tank"), unit("ogre", hp=2)), my_backline=(unit("guard"),))
    assert greedy_choice(o) == BACK(0, 1)  # guard can't kill tank (6 hp) but kills the damaged ogre


@pytest.mark.parametrize("o,expected", [
    # even trade: brute vs brute, both die, equal cost
    (make_obs(front_owner=-1, frontline=(unit("brute"),), my_backline=(unit("brute"),)), END),
    # no kill: pawn (1 atk) into tank (6 hp)
    (make_obs(front_owner=-1, frontline=(unit("tank"),), my_backline=(unit("pawn"),)), END),
    # trading down: a damaged brute (cost 3) kills a guard (cost 2) but dies
    (make_obs(front_owner=-1, frontline=(unit("guard"),), my_backline=(unit("brute", hp=2),)), END),
    # frontline attacker with only bad trades goes for the base instead
    # (tank: no kill; pawn as 3/2: guard kills it but dies, and pawn is cheaper)
    (make_obs(front_owner=1, frontline=(unit("guard"),), opp_backline=(unit("tank"), unit("pawn", atk=3))), BASE(0)),
    # chip damage that kills nothing is never a trade, even when the attacker survives
    (make_obs(front_owner=1, frontline=(unit("titan", atk=1),), opp_backline=(unit("imp", hp=2),)), BASE(0)),
])
def test_refuses_unfavorable_or_even_trades(o, expected):
    legal = spec_legal(o, TEST_CONFIG)
    assert any(a >= SP.FRONT0 for a in legal), "scenario must offer at least one trade"
    assert greedy_choice(o, legal) == expected, describe(greedy_choice(o, legal))


def test_trade_comes_before_push():
    o = make_obs(front_owner=1, frontline=(unit("brute"),), my_backline=(unit("titan"),),
                 opp_backline=(unit("guard"),))
    legal = spec_legal(o, TEST_CONFIG)
    assert MOVE(0) in legal and BASE(0) in legal
    assert greedy_choice(o, legal) == FRONT(0, 0)


# ---------------------------------------------------------------------- greedy priority 3: push
def test_pushes_highest_attack_ready_unit():
    o = make_obs(my_backline=(unit("pawn"), unit("brute"), unit("titan", ready=False), unit("guard")))
    assert greedy_choice(o) == MOVE(1)  # titan is exhausted (deployed this turn)


def test_push_attack_tie_takes_lower_slot():
    o = make_obs(my_backline=(unit("imp"), unit("guard", hp=1), unit("guard")))
    assert greedy_choice(o) == MOVE(1)


def test_push_comes_before_base_attack():
    o = make_obs(front_owner=1, frontline=(unit("guard"),), my_backline=(unit("brute"),))
    assert greedy_choice(o) == MOVE(0)


def test_no_push_when_own_frontline_full():
    o = make_obs(front_owner=1, frontline=(unit("imp", ready=False),) * 4 + (unit("pawn"),),
                 my_backline=(unit("titan"),))
    assert MOVE(0) not in spec_legal(o, TEST_CONFIG)
    assert greedy_choice(o) == BASE(4)


def test_no_push_into_enemy_frontline():
    o = make_obs(front_owner=-1, frontline=(unit("titan"),), my_backline=(unit("pawn"),))
    assert greedy_choice(o) == END  # blocked, and pawn into titan is a losing trade


# ---------------------------------------------------------------------- greedy priority 4: base
def test_frontline_units_attack_base():
    o = make_obs(front_owner=1, frontline=(unit("pawn", ready=False), unit("guard"), unit("brute")),
                 opp_backline=(unit("tank"),))
    assert greedy_choice(o) == BASE(1)


def test_attacks_base_when_enemy_backline_empty():
    o = make_obs(front_owner=1, frontline=(unit("imp"),), hand=("titan",), my_coins=6)
    assert greedy_choice(o) == BASE(0)  # titan is unaffordable, nothing else to do


# ---------------------------------------------------------------------- greedy priority 5: end turn
@pytest.mark.parametrize("o", [
    make_obs(),
    make_obs(hand=("ogre", "titan"), my_coins=4),
    make_obs(my_backline=(unit("titan", ready=False),), front_owner=1, frontline=(unit("guard", ready=False),),
             opp_backline=(unit("imp"),)),
])
def test_ends_turn_when_nothing_else(o):
    assert greedy_choice(o) == END


def test_only_uses_the_given_legal_list():
    # Even with an attractive state, greedy picks from what it is offered.
    o = make_obs(hand=("titan",), my_coins=7, front_owner=1, frontline=(unit("titan"),), opp_backline=(unit("ogre"),))
    assert greedy_choice(o, [END, BASE(0)]) == BASE(0)
    assert greedy_choice(o, [END]) == END


# ---------------------------------------------------------------------- strength
def test_greedy_beats_random_duplicate():
    # Measured: 198/200 wins (0.990) over deals 0..99 with seats swapped (0.9875 over 200 deals).
    res = duplicate_match(GreedyAgent(CONFIG), RandomAgent(CONFIG), range(100))
    assert res["games"] == 200
    win_rate = res["wins"] / res["games"]
    assert win_rate >= 0.9, res


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


@pytest.mark.parametrize("spec", ["", "Random", "greedy ", "minimax", "model.pth", "ppo:", "ppo:model.ckpt", "runs/ppo"])
def test_make_agent_rejects_junk(spec):
    with pytest.raises(ValueError):
        make_agent(spec, CONFIG)
