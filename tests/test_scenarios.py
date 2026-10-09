"""Scenario suite tests (SPEC §10): consistent positions, goals that are False at the start and after
END_TURN alone, a DFS solution for every scenario, random below 50% on average, the documented best
and trap lines, the lethal search behind `survives_next_turn` (engine-only: checked on rule variants),
and the runner (greedy is reported, not asserted)."""
from __future__ import annotations

from typing import Optional, Sequence

import pytest

from cardgame.actions import ActionKind
from cardgame.agents import GreedyAgent, RandomAgent
from cardgame.cards import TROOP, load_ruleset
from cardgame.engine import Game, IllegalActionError, Observation
from cardgame.scenarios import (MAX_ACTIONS, SCENARIOS, SOLVER_DEPTH, U, build_position, can_win_this_turn,
                                check_position, dominance_violations, end_of_turn_states, get_scenario,
                                play_scenario, run_scenarios, scenario_names, solve, state_key,
                                survives_next_turn, win)

CONFIG = load_ruleset()
IDS = [s.name for s in SCENARIOS]
REQUIRED_TAGS = {"fast", "ranged", "defense", "armor", "move_cost", "frontline", "lethal", "threat", "trade",
                 "sequencing"}

# The line each scenario's note describes, and the traps it warns about (action descriptions).
BEST = {
    "fast_move_attack_lethal": ["MOVE(0)", "ATTACK(front0->base)"],
    "fast_attack_then_move": ["ATTACK(back0->front0)", "MOVE(0)", "END_TURN"],
    "defense_backline_order": ["ATTACK(front0->back0)", "ATTACK(front1->back0)", "END_TURN"],
    "defense_frontline_ranged": ["ATTACK(back2->front0)", "ATTACK(back0->front0)", "ATTACK(back1->front0)",
                                 "END_TURN"],
    "ranged_finish_backline": ["ATTACK(back0->back0)", "ATTACK(back1->back0)", "END_TURN"],
    "ranged_base_lethal": ["ATTACK(back0->base)", "ATTACK(back1->base)"],
    "armor_pierce": ["ATTACK(back3->front0)", "END_TURN"],
    "armor_attacker_survives": ["ATTACK(back0->front0)", "END_TURN"],
    "move_cost_hold_frontline": ["MOVE(0)", "END_TURN"],
    "clear_frontline_then_lethal": ["ATTACK(back1->front0)", "ATTACK(back2->front0)", "MOVE(0)",
                                    "ATTACK(front0->base)"],
    "stop_lethal_kill_right_unit": ["ATTACK(back0->front0)", "END_TURN"],
    "ranged_no_return": ["ATTACK(back1->front0)", "MOVE(0)", "ATTACK(front0->base)"],
    "budget_cheap_movers": ["MOVE(1)", "MOVE(1)", "MOVE(1)", "ATTACK(front0->base)", "ATTACK(front1->base)"],
    "base_ignores_defense": ["ATTACK(front0->base)", "ATTACK(front1->base)", "ATTACK(front2->base)"],
    "sacrifice_to_clear": ["ATTACK(back1->front0)", "MOVE(0)", "ATTACK(front0->base)"],
    "ranged_through_defense": ["ATTACK(back0->back0)", "ATTACK(back2->back0)", "ATTACK(back1->back0)",
                               "END_TURN"],
}
TRAPS = {
    "fast_move_attack_lethal": [["PLAY(1)", "END_TURN"]],  # Swordsman first: no coin left to move
    "fast_attack_then_move": [["PLAY(0)", "ATTACK(back0->front0)", "END_TURN"],  # no coin to move in
                              ["ATTACK(back0->front0)", "END_TURN"]],           # frontline left empty
    "defense_backline_order": [["ATTACK(front1->back0)", "ATTACK(front0->back0)", "END_TURN"]],
    "defense_frontline_ranged": [["ATTACK(back2->front0)", "ATTACK(back0->back0)", "ATTACK(back1->front0)",
                                  "END_TURN"],  # Crossbowman wasted on the Ballista: Wolf Rider lives
                                 ["ATTACK(back0->back0)", "ATTACK(back1->back0)", "END_TURN"]],  # shoot Ballista
    "ranged_finish_backline": [["ATTACK(back0->back1)", "ATTACK(back1->back0)", "END_TURN"]],
    "ranged_base_lethal": [["ATTACK(back0->front0)", "ATTACK(back1->base)", "END_TURN"]],
    "armor_pierce": [["ATTACK(back0->front0)", "END_TURN"], ["ATTACK(back1->front0)", "END_TURN"]],
    "armor_attacker_survives": [["ATTACK(back1->front0)", "END_TURN"],   # Swordsman trades itself
                                ["ATTACK(back2->front0)", "END_TURN"]],  # Footman cannot kill and dies
    "move_cost_hold_frontline": [["PLAY(1)", "END_TURN"], ["PLAY(0)", "END_TURN"]],
    "clear_frontline_then_lethal": [["ATTACK(back1->front0)", "ATTACK(back0->front0)", "MOVE(0)", "END_TURN"],
                                    ["PLAY(1)", "ATTACK(back1->front0)", "ATTACK(back2->front0)", "END_TURN"]],
    "stop_lethal_kill_right_unit": [["ATTACK(back0->front1)", "END_TURN"], ["ATTACK(back0->back0)", "END_TURN"]],
    "ranged_no_return": [["ATTACK(back0->front0)", "ATTACK(back0->base)", "END_TURN"],  # Wolf Rider dies
                         ["ATTACK(back1->base)", "END_TURN"]],                          # 4 < 6
    "budget_cheap_movers": [["MOVE(0)", "MOVE(2)", "ATTACK(front0->base)", "ATTACK(front1->base)", "END_TURN"]],
    "base_ignores_defense": [["ATTACK(front0->back0)", "ATTACK(front1->base)", "ATTACK(front2->base)",
                              "END_TURN"]],
    "sacrifice_to_clear": [["ATTACK(back0->front0)", "MOVE(0)", "END_TURN"]],
    "ranged_through_defense": [["ATTACK(back0->back0)", "ATTACK(back1->back0)", "ATTACK(back2->back0)",
                                "END_TURN"],                              # Archer spent on the Shield Archer
                               ["ATTACK(back0->front0)", "END_TURN"]],    # killing the Footman leaves 5
}


class FixedActionAgent:
    name = "fixed"

    def __init__(self, action):
        self.action = action

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def act(self, obs: Observation, legal_actions: Sequence[int]):
        return self.action


class ScriptAgent:
    """Plays a fixed list of action descriptions, then END_TURN."""
    name = "script"

    def __init__(self, descriptions, game: Game):
        self.descriptions, self.describe = list(descriptions), game.describe

    def reset(self, seed: Optional[int] = None) -> None:
        self.todo = list(self.descriptions)

    def act(self, obs: Observation, legal_actions: Sequence[int]) -> int:
        if not self.todo:
            return 0
        want = self.todo.pop(0)
        by_name = {self.describe(a): a for a in legal_actions}
        assert want in by_name, f"{want} not legal; legal: {sorted(by_name)}"
        return by_name[want]


def play_line(name: str, descriptions) -> bool:
    sc = get_scenario(name)
    start = sc.build(CONFIG)
    ok, _ = play_scenario(sc, ScriptAgent(descriptions, start), CONFIG, start=start)
    return ok


# ---------------------------------------------------------------------- catalogue and positions
def test_catalogue():
    assert 12 <= len(SCENARIOS) <= 20
    assert len(set(IDS)) == len(IDS) == len(scenario_names())
    tags = set()
    for sc in SCENARIOS:
        assert sc.tags and sc.note and len(sc.decks) == 2
        assert all(d in CONFIG.deck_names for d in sc.decks)
        tags |= set(sc.tags)
    assert REQUIRED_TAGS <= tags, REQUIRED_TAGS - tags
    assert any(sc.build(CONFIG).current == 1 for sc in SCENARIOS)  # egocentric encoding from seat 1 too
    assert set(BEST) == set(IDS) and set(TRAPS) == set(IDS)
    with pytest.raises(KeyError):
        get_scenario("no_such_scenario")


@pytest.mark.parametrize("sc", SCENARIOS, ids=IDS)
def test_position_is_consistent_and_deterministic(sc):
    game = sc.build(CONFIG)
    assert check_position(sc, game) == []
    assert not game.done and game.legal_actions()[0] == game.action_space.END_TURN
    p = game.current
    assert game.coins[p] <= game.round  # SPEC §10
    again = sc.build(CONFIG)
    assert state_key(again) == state_key(game) and again.observe(p) == game.observe(p)
    assert again.observe(1 - p) == game.observe(1 - p)


def test_check_position_flags_inconsistencies():
    sc = get_scenario("fast_move_attack_lethal")
    game = sc.build(CONFIG)
    game.coins[game.current] += 1  # more coins than the round gives after this turn's spending
    game.backline[1 - game.current][0].attacked = True  # opponent flags are refreshed at its END_TURN
    problems = check_position(sc, game)
    assert any("coins" in x for x in problems) and any("opponent unit" in x for x in problems)
    other = sc.build(CONFIG)
    other.hands[other.current].append(CONFIG.cards.by_id("bastion").index)  # not in the Blitz deck
    assert any("bastion" in x for x in check_position(sc, other))


def test_build_position_rejects_two_frontline_owners():
    with pytest.raises(ValueError):
        build_position(CONFIG, decks=("Blitz", "Bulwark"), my_front=("footman",), opp_front=("militia",))


@pytest.mark.parametrize("sc", SCENARIOS, ids=IDS)
def test_goal_false_at_start_and_after_end_turn_alone(sc):
    game = sc.build(CONFIG)
    start = game.clone()
    assert not sc.goal(start, game)
    game.step(game.action_space.END_TURN)
    assert not sc.goal(start, game)


# ---------------------------------------------------------------------- solvability
@pytest.mark.parametrize("sc", SCENARIOS, ids=IDS)
def test_dfs_finds_a_solution(sc):
    line = solve(sc, CONFIG)
    assert line is not None and 1 <= len(line) <= SOLVER_DEPTH
    game = sc.build(CONFIG)
    start = game.clone()
    for a in line:
        assert game.is_legal(a)
        game.step(a)
    assert line[-1] == game.action_space.END_TURN or game.done
    assert sc.goal(start, game)
    assert solve(sc, CONFIG, max_depth=1) is None  # END_TURN alone is the only depth-1 line


@pytest.mark.parametrize("sc", SCENARIOS, ids=IDS)
def test_documented_best_line_solves_and_traps_fail(sc):
    assert play_line(sc.name, BEST[sc.name])
    for trap in TRAPS[sc.name]:
        assert not play_line(sc.name, trap), trap


@pytest.mark.parametrize("sc", SCENARIOS, ids=IDS)
def test_known_best_move_is_really_best(sc):
    """Material goals: every end-of-turn state that misses the goal is weakly dominated by a goal state on
    (won, enemy base HP, enemy board value, own board value); survival goals: no lethal exists this turn."""
    bad = dominance_violations(sc, CONFIG)
    start = sc.build(CONFIG)
    assert not bad, [(b.enemy_base_hp, b.enemy_board_value, b.own_board_value, [start.describe(a) for a in b.line])
                     for b in bad[:3]]


# Survival puzzles: the tactic each one teaches must be the only way to survive.
MUST_KILL = {"fast_attack_then_move": ("raider",), "defense_backline_order": ("longbowman",),
             "defense_frontline_ranged": ("wolf_rider",), "ranged_finish_backline": ("ballista", "longbowman"),
             "stop_lethal_kill_right_unit": ("swordsman",), "ranged_through_defense": ("longbowman",)}


@pytest.mark.parametrize("name", sorted(MUST_KILL))
def test_survival_requires_the_intended_kills(name):
    sc = get_scenario(name)
    start = sc.build(CONFIG)
    enemy = 1 - start.current
    wanted = {CONFIG.cards.by_id(c).index for c in MUST_KILL[name]}
    targets = {u.uid for z in (start.backline[enemy], start.frontline) for u in z if u.card in wanted}
    assert targets
    for state in end_of_turn_states(sc, CONFIG, start):
        if not state.goal:
            continue
        end = start.clone()
        for a in state.line:
            end.step(a)
        alive = {u.uid for z in (end.backline[enemy], end.frontline) for u in z}
        assert not targets & alive, [start.describe(a) for a in state.line]


def test_dominance_check_flags_an_ambiguous_material_goal():
    """A kill goal that competes with face damage (ranged units can always shoot the base) is ambiguous."""
    from cardgame.scenarios import Scenario, kills, _scenario
    ambiguous = _scenario("ambiguous", ("ranged",), kills("longbowman"), "",
                          mine="Volley", theirs="Volley", round=6, my_base=14, opp_base=13,
                          my_back=("crossbowman",), opp_back=(U("longbowman", hp=2),))
    assert isinstance(ambiguous, Scenario)
    states = end_of_turn_states(ambiguous, CONFIG)
    assert any(s.goal for s in states) and any(not s.goal for s in states)
    assert dominance_violations(ambiguous, CONFIG), "shooting the base instead is not dominated"


# ---------------------------------------------------------------------- lethal search
def test_can_win_this_turn_and_survives_next_turn():
    lethal = get_scenario("fast_move_attack_lethal").build(CONFIG)
    assert can_win_this_turn(lethal)
    lethal.base_hp[1 - lethal.current] = 6
    lethal.invalidate()
    assert not can_win_this_turn(lethal)

    sc = get_scenario("fast_attack_then_move")
    start = sc.build(CONFIG)
    ended = start.clone()
    ended.step(ended.action_space.END_TURN)
    assert can_win_this_turn(ended)  # Raider + Lancer + Wolf Rider: 14 >= 9
    assert not survives_next_turn(start, ended) and not survives_next_turn(start, start)

    # moving in with a fast unit after a kill that empties the frontline, vs Defense in the frontline
    game = build_position(CONFIG, decks=("Blitz", "Bulwark"), round=6, opp_base=5,
                          my_back=("lancer", "swordsman"), opp_front=("militia",))
    assert can_win_this_turn(game)
    game = build_position(CONFIG, decks=("Blitz", "Bulwark"), round=6, opp_base=5,
                          my_back=("lancer", "swordsman"), opp_front=("militia", "shieldbearer"))
    assert not can_win_this_turn(game)  # the Swordsman cannot kill the 4-HP Shieldbearer and the Militia
    game = build_position(CONFIG, decks=("Blitz", "Bulwark"), round=6, opp_base=5,
                          my_back=("lancer", "swordsman", "ogre"), opp_front=("militia", "shieldbearer"))
    assert can_win_this_turn(game)
    done = get_scenario("ranged_base_lethal").build(CONFIG)
    for a in ("ATTACK(back0->base)", "ATTACK(back1->base)"):
        done.step({done.describe(x): x for x in done.legal_actions()}[a])
    assert done.done and not can_win_this_turn(done)
    assert win(get_scenario("ranged_base_lethal").build(CONFIG), done)


class ChargeGame(Game):
    """Rules variant (as a later stage's card might add): units can act on the round they are deployed."""

    def step(self, action: int) -> None:
        super().step(action)
        if self.action_space.decode(action).kind == ActionKind.PLAY:
            self.backline[self.current][-1].summoned = False


class LongReachGame(Game):
    """Rules variant: troops may hit the enemy base from the backline."""

    def _compute_legal(self) -> None:
        super()._compute_legal()
        if self.done:
            return
        sp = self.action_space
        extra = {sp.attack(j, sp.BASE_TARGET) for j, u in enumerate(self.backline[self.current])
                 if u.nature == TROOP and u.can_attack()}
        self._legal = sorted(set(self._legal) | extra)
        self._mask = bytearray(self.num_actions)
        for a in self._legal:
            self._mask[a] = 1


def as_variant(game: Game, cls) -> Game:
    game.__class__ = cls
    game.invalidate()
    return game


def test_can_win_searches_plays_and_derives_who_can_act_from_the_engine():
    def position() -> Game:
        return build_position(CONFIG, decks=("Blitz", "Bulwark"), round=8, opp_base=2,
                              my_hand=("scout", "raider"), my_back=(U("lancer", summoned=True),))
    assert not can_win_this_turn(position())  # here a unit deployed this round cannot act
    charge = as_variant(position(), ChargeGame)
    assert can_win_this_turn(charge)  # play the Scout, move it in (move cost 0), hit the base for 2
    assert can_win_this_turn(charge.clone()) and isinstance(charge.clone(), ChargeGame)


def test_can_win_derives_reach_from_the_engine():
    def position() -> Game:
        return build_position(CONFIG, decks=("Legion", "Bulwark"), round=6, opp_base=4, my_back=("swordsman",))
    assert not can_win_this_turn(position())  # a backline troop cannot reach the base; after moving it cannot attack
    assert can_win_this_turn(as_variant(position(), LongReachGame))


# ---------------------------------------------------------------------- runner
def test_random_solves_less_than_half_on_average():
    report = run_scenarios("random", CONFIG)
    assert report.n == len(SCENARIOS) and not report.deterministic
    assert all(r.playouts == 20 and r.solved is None for r in report.results)
    assert report.mean_success < 0.5
    assert report.pct_solved == pytest.approx(100 * report.mean_success)
    subset = run_scenarios("random", CONFIG, scenarios=SCENARIOS[2:5])  # seeds depend on names, not order
    assert subset.results == report.results[2:5]
    print("\n" + report.format())


def test_greedy_results_are_reported():
    report = run_scenarios("greedy", CONFIG)
    assert report.deterministic and report.n == len(SCENARIOS)
    assert all(isinstance(r.solved, bool) and r.success is None and r.line for r in report.results)
    assert run_scenarios(GreedyAgent(CONFIG), CONFIG).to_dict()["results"] == report.to_dict()["results"]
    print("\n" + report.format())  # reported, not asserted (SPEC §10)


def test_run_scenarios_with_a_ppo_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from cardgame.features import ObservationEncoder
    from cardgame.rl.agent import PPOAgent
    from cardgame.rl.network import EntityPolicyNet

    torch.manual_seed(0)
    net = EntityPolicyNet(ObservationEncoder(CONFIG).layout(), d_model=32, id_dim=8, ctx_dim=64, pair_dim=32)
    path = str(tmp_path / "ckpt_00001.pt")
    torch.save({"model": net.state_dict(), "net": net.spec(), "update": 1}, path)
    subset = SCENARIOS[:3]
    report = run_scenarios(path, CONFIG, n_stochastic=2, scenarios=subset)
    assert report.agent == path and report.n == 3 and report.deterministic
    for r in report.results:
        assert isinstance(r.solved, bool) and r.playouts == 2 and 0.0 <= r.success <= 1.0 and r.line
    d = report.to_dict()
    assert d["n_solved"] == report.n_solved and len(d["results"]) == 3

    agent = PPOAgent.from_checkpoint(path, CONFIG)  # an instance: argmax run, then sampled playouts
    inst = run_scenarios(agent, CONFIG, n_stochastic=2, scenarios=subset)  # same seeds: same results
    assert agent.deterministic is False  # restored after the runs
    assert [r.solved for r in inst.results] == [r.solved for r in report.results]
    assert [r.success for r in inst.results] == [r.success for r in report.results]


def test_play_scenario_checks_legality_and_leaves_the_start_alone():
    sc = SCENARIOS[0]
    start = sc.build(CONFIG)
    key = state_key(start)
    with pytest.raises(IllegalActionError):
        play_scenario(sc, FixedActionAgent(999), CONFIG, start=start)
    ok, line = play_scenario(sc, RandomAgent(CONFIG), CONFIG, seed=3, start=start)
    assert state_key(start) == key and 1 <= len(line) <= MAX_ACTIONS
    assert line[-1] == 0 or ok
