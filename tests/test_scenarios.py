"""Scenario suite tests (SPEC 10): consistent positions (mulligan off for mid-game positions, a real
mulligan phase for mulligan_sanity), goals that are False at the start and after passing alone, a DFS
solution for every scenario, random below 50% on average, the documented best and trap lines, the
validity check (dominance, survival composites), the four Stage 3 scenarios (operations, choices,
on_death, effect damage vs Defense, the mulligan), the rule-agnostic searches (engine-only: checked on
rule variants and through operations/choices), and the runner (lookahead and greedy are reported, not
asserted)."""
from __future__ import annotations

import random
from typing import Optional, Sequence

import pytest

from collections import Counter

from cardgame.actions import ActionKind
from cardgame.agents import GreedyAgent, LookaheadAgent, RandomAgent
from cardgame.cards import TROOP, load_ruleset
from cardgame.engine import CHOICE, MAIN, MULLIGAN, Game, IllegalActionError, Observation
from cardgame.scenarios import (BASELINES, MAX_ACTIONS, SCENARIOS, SOLVER_DEPTH, U, all_of, build_position,
                                can_win_this_turn, check_position, dominance_violations, end_of_turn_states,
                                get_scenario, goal_kinds, kills, mulligan_rule, no_losses, play_scenario,
                                run_scenarios, scenario_config, scenario_names, solve, state_key, summons,
                                survives_next_turn, win)

CONFIG = load_ruleset()  # the shipped ruleset (mulligan on): scenario positions force their own setting
IDS = [s.name for s in SCENARIOS]
STAGE3 = ("operation_lethal", "on_death_exploit", "effect_clears_defense", "mulligan_sanity")
REQUIRED_TAGS = {"fast", "ranged", "defense", "armor", "move_cost", "frontline", "lethal", "threat", "trade",
                 "sequencing", "operation", "effects", "choice", "on_death", "mulligan"}

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
    # Stage 3: hand slots are sorted by card index (Fire Mission, a token, sorts before the Observer)
    "operation_lethal": ["PLAY(2)", "PLAY(2)", "CHOOSE(front0)"],
    "on_death_exploit": ["ATTACK(back0->front0)", "END_TURN"],
    "effect_clears_defense": ["PLAY(2)", "CHOOSE(enemy_back0)", "ATTACK(back0->back0)", "END_TURN"],
    "mulligan_sanity": ["MULLIGAN(3)", "MULLIGAN(4)", "CONFIRM"],
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
    "operation_lethal": [["PLAY(0)", "PLAY(1)", "END_TURN"],              # Militia first: no coin for the combo
                         ["PLAY(1)", "END_TURN"],                         # Swordsman: the Observer is unaffordable
                         ["PLAY(2)", "ATTACK(back0->front0)", "END_TURN"]],  # Fire Mission never played
    "on_death_exploit": [["ATTACK(back1->front0)", "END_TURN"],           # the Footman trade: no Recruit
                         ["PLAY(0)", "END_TURN"]],                        # the Lancer lives: lethal next turn
    "effect_clears_defense": [["PLAY(2)", "CHOOSE(enemy_back1)", "ATTACK(back0->back0)", "ATTACK(back1->back0)",
                               "END_TURN"],                               # 2 damage on the Longbowman
                              ["PLAY(1)", "ATTACK(back0->back0)", "ATTACK(back1->back0)", "END_TURN"],  # Lancer
                              ["ATTACK(back0->back0)", "ATTACK(back1->back0)", "PLAY(2)", "CHOOSE(enemy_back0)",
                               "END_TURN"]],                              # shots kill the Warden, no shot left
    "mulligan_sanity": [["CONFIRM"],                                      # keeps the Warden and the Bastion
                        ["MULLIGAN(0)", "MULLIGAN(3)", "MULLIGAN(4)", "CONFIRM"],  # replaces the Militia too
                        ["MULLIGAN(4)", "CONFIRM"]],                      # keeps the Warden
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
    """Plays a fixed list of action descriptions, then passes (END_TURN, or CONFIRM in the mulligan)."""
    name = "script"

    def __init__(self, descriptions, game: Game):
        self.descriptions, self.describe = list(descriptions), game.describe
        self.confirm = game.action_space.CONFIRM

    def reset(self, seed: Optional[int] = None) -> None:
        self.todo = list(self.descriptions)

    def act(self, obs: Observation, legal_actions: Sequence[int]) -> int:
        if not self.todo:
            return self.confirm if self.confirm in legal_actions else 0
        want = self.todo.pop(0)
        by_name = {self.describe(a): a for a in legal_actions}
        assert want in by_name, f"{want} not legal; legal: {sorted(by_name)}"
        return by_name[want]


def play_line(name: str, descriptions) -> bool:
    sc = get_scenario(name)
    start = sc.build(CONFIG)
    ok, _ = play_scenario(sc, ScriptAgent(descriptions, start), CONFIG, start=start)
    return ok


def run_line(game: Game, descriptions) -> Game:
    for d in descriptions:
        game.step({game.describe(a): a for a in game.legal_actions()}[d])
    return game


def pass_action(game: Game) -> int:
    """The action that ends the agent's turn without doing anything: END_TURN, or CONFIRM in the mulligan."""
    return game.action_space.CONFIRM if game.phase == MULLIGAN else game.action_space.END_TURN


# ---------------------------------------------------------------------- catalogue and positions
def test_catalogue():
    assert len(SCENARIOS) == 20 and IDS[16:] == list(STAGE3)  # the 16 Stage 2 scenarios + four new ones
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
    if sc.name == "mulligan_sanity":  # the only scenario that starts in the mulligan phase
        assert game.config.mulligan and game.phase == MULLIGAN and game.turn == 0
        assert game.legal_actions()[-1] == game.action_space.CONFIRM
    else:  # SPEC 10: every other scenario is built with mulligan=False
        assert not game.config.mulligan and game.phase == MAIN
        assert game.legal_actions()[0] == game.action_space.END_TURN
    assert not game.done
    p = game.current
    assert game.coins[p] <= game.round  # SPEC 10
    again = sc.build(CONFIG)
    assert state_key(again) == state_key(game) and again.observe(p) == game.observe(p)
    assert again.observe(1 - p) == game.observe(1 - p)
    assert sc.build(scenario_config(CONFIG)).observe(p) == game.observe(p)  # same position from either ruleset
    lookahead = LookaheadAgent(CONFIG, seed=0)  # determinize works on the hand-built bookkeeping
    assert lookahead.act_game(game, p) in game.legal_actions()
    sim = game.determinize(p, random.Random(1))
    assert sim.observe(p) == game.observe(p)


def test_scenario_config_forces_the_mulligan_off():
    assert CONFIG.mulligan and not scenario_config(CONFIG).mulligan and CONFIG.mulligan  # the input is untouched
    off = scenario_config(CONFIG)
    assert scenario_config(off) is off and off.cards is CONFIG.cards and off.decks == CONFIG.decks
    game = build_position(CONFIG, decks=("Blitz", "Bulwark"), round=3, my_back=("militia",))
    assert game.phase == MAIN and not game.config.mulligan and game.turn == 5  # round 3, first player: turn 5
    second = build_position(CONFIG, decks=("Blitz", "Bulwark"), seat=1, first_player=0, round=3)
    assert second.turn == 6 and second.current == 1


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


def test_build_position_bookkeeping():
    """revealed / played count the non-token units on the board; token cards in hand are known."""
    game = build_position(CONFIG, decks=("Legion", "Volley"), round=4, my_hand=("militia",),
                          my_back=("bugler", U("recruit", hp=1)), opp_back=("archer", "archer"))
    idx = CONFIG.cards.by_id
    me, opp = game.current, 1 - game.current
    assert game.revealed[me][idx("bugler").index] == 1 and game.revealed[me][idx("recruit").index] == 0
    assert game.revealed[opp][idx("archer").index] == 2 == game.played[opp][idx("archer").index]
    assert game.observe(me).opp_revealed[idx("archer").index] == 2
    sc = get_scenario("operation_lethal")
    with_token = build_position(CONFIG, decks=sc.decks, round=2, my_hand=("fire_mission",))
    assert with_token.known_hand[0][idx("fire_mission").index] == 1
    assert with_token.observe(1).opp_known_hand[idx("fire_mission").index] == 1


def test_build_position_rejects_two_frontline_owners():
    with pytest.raises(ValueError):
        build_position(CONFIG, decks=("Blitz", "Bulwark"), my_front=("footman",), opp_front=("militia",))


@pytest.mark.parametrize("sc", SCENARIOS, ids=IDS)
def test_goal_false_at_start_and_after_passing_alone(sc):
    game = sc.build(CONFIG)
    start = game.clone()
    assert not sc.goal(start, game)
    game.step(pass_action(game))  # END_TURN, or CONFIRM keeping the whole opening hand
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
    assert line[-1] == pass_action(start) or game.done
    assert sc.goal(start, game)
    assert solve(sc, CONFIG, max_depth=1) is None  # passing alone is the only depth-1 line


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
             "stop_lethal_kill_right_unit": ("swordsman",), "ranged_through_defense": ("longbowman",),
             "effect_clears_defense": ("warden", "longbowman"), "on_death_exploit": ("lancer",)}


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


def test_goal_kinds():
    assert goal_kinds(win) == {"win"} and goal_kinds(survives_next_turn) == {"survive"}
    assert goal_kinds(all_of(kills("outrider"), no_losses)) == {"material"}
    assert goal_kinds(all_of(survives_next_turn, summons("recruit"))) == {"survive", "material"}
    assert goal_kinds(mulligan_rule()) == {"rule"}
    assert goal_kinds(lambda start, end: True) == {"material"}  # an unlabelled goal is judged on material


def test_dominance_with_a_survival_part():
    """on_death_exploit: states that lose next turn count as dominated; among the surviving ones the Bugler
    trade (Footman + Recruit left) dominates the Footman trade (a lone Bugler)."""
    sc = get_scenario("on_death_exploit")
    start = sc.build(CONFIG)
    states = end_of_turn_states(sc, CONFIG, start, survival=True)
    assert all(s.survives is not None for s in states)
    assert any(not s.survives for s in states) and any(s.survives and not s.goal for s in states)
    assert all(s.survives for s in states if s.goal)
    bugler = max(s.own_board_value for s in states if s.goal)
    footman = max(s.own_board_value for s in states if s.survives and not s.goal)
    assert bugler == footman + 1  # the Recruit (cost 1) on top of the same 2-cost survivor
    assert dominance_violations(sc, CONFIG) == []
    # the same trade judged on material alone is ambiguous: not trading keeps more on the board
    from cardgame.scenarios import Scenario
    material = Scenario("material_only", sc.tags, sc.decks, sc.build, all_of(kills("lancer"), summons("recruit")))
    assert dominance_violations(material, CONFIG)
    # survival goals with lethal on the board are invalid: winning beats surviving
    lethal = Scenario("lethal_survival", ("x",), get_scenario("ranged_base_lethal").decks,
                      get_scenario("ranged_base_lethal").build, all_of(survives_next_turn, no_losses))
    bad = dominance_violations(lethal, CONFIG)
    assert len(bad) == 1 and bad[0].won


def test_operation_lethal_needs_the_operation_and_a_choice():
    """Every winning line deploys the Forward Observer, plays the Fire Mission it adds and resolves the
    pending CHOOSE: the Observer's watcher deals the final damage (SPEC 10 operation_lethal)."""
    sc = get_scenario("operation_lethal")
    start = sc.build(CONFIG)
    p = start.current
    fire = CONFIG.cards.by_id("fire_mission").index
    assert CONFIG.cards.cards[fire].token and fire not in start.hands[p]
    wins = [s for s in end_of_turn_states(sc, CONFIG, start) if s.won]
    assert wins and all(s.goal for s in wins)
    for s in wins:
        kinds = [start.action_space.decode(a).kind for a in s.line]
        assert kinds.count(ActionKind.PLAY) >= 2 and kinds[-1] == ActionKind.CHOOSE
    g = run_line(start.clone(), ["PLAY(2)", "PLAY(2)"])
    assert g.phase == CHOICE and g.current == p and not g.done
    assert {g.describe(a) for a in g.legal_actions()} == {"CHOOSE(enemy_back0)", "CHOOSE(front0)"}
    g = run_line(g, ["CHOOSE(front0)"])
    assert g.done and g.winner() == p and g.base_hp[1 - p] == 0
    assert can_win_this_turn(start)
    two = start.clone()
    two.base_hp[1 - p] = 2  # the combo deals exactly 1 to the base
    two.invalidate()
    assert not can_win_this_turn(two)


def test_effect_clears_defense_needs_the_operation_on_the_defender():
    sc = get_scenario("effect_clears_defense")
    start = sc.build(CONFIG)
    p, o = start.current, 1 - start.current
    warden = CONFIG.cards.by_id("warden").index
    assert start.backline[o][0].card == warden and start.backline[o][0].defense and start.backline[o][0].armor
    survivors = [s for s in end_of_turn_states(sc, CONFIG, start) if s.goal]
    assert survivors
    for s in survivors:
        names = [start.describe(a) for a in s.line]
        assert "CHOOSE(enemy_back0)" in names  # the operation targets the Warden
    g = run_line(start.clone(), ["PLAY(2)", "CHOOSE(enemy_back0)"])
    assert all(u.card != warden for u in g.backline[o])  # 4 effect damage through armor and Defense
    # attacks alone: the Warden's Defense binds both shots, and neither hits hard enough through armor
    g = run_line(start.clone(), ["ATTACK(back0->back0)"])
    assert g.backline[o][0].card == warden and g.backline[o][0].hp == 1


def test_mulligan_sanity_position_and_goal():
    sc = get_scenario("mulligan_sanity")
    start = sc.build(CONFIG)
    p, first = start.current, start.first_player
    cards = CONFIG.cards.cards
    assert p != first and start.mulligan_done[first] and not start.mulligan_done[p]
    assert [cards[c].cost for c in start.hands[p]] == [1, 2, 4, 5, 8]
    top = start.deck_cards[p][-len(start.hands[p]):]  # the replacements are drawn from here
    assert all(3 <= cards[c].cost <= 4 for c in top)  # never confused with kept (<= 2) or replaced (>= 5) cards
    assert play_line("mulligan_sanity", ["MULLIGAN(2)", "MULLIGAN(3)", "MULLIGAN(4)", "CONFIRM"])  # Pikeman free
    for marks in ([], [0, 3, 4], [1, 3, 4], [3], [4], [0, 1, 2, 3, 4]):
        assert not play_line("mulligan_sanity", [f"MULLIGAN({i})" for i in marks] + ["CONFIRM"]), marks
    end = run_line(start.clone(), ["MULLIGAN(3)", "MULLIGAN(4)", "CONFIRM"])
    assert end.phase == MAIN and end.current == first and end.turn == 1  # round 1 starts after the second CONFIRM
    assert Counter(end.hands[p]) == Counter(start.hands[p][:3] + top[-2:])
    assert sc.goal(start, end) and not sc.goal(start, run_line(start.clone(), ["MULLIGAN(3)", "MULLIGAN(4)"]))


def test_state_key_covers_the_rng_and_never_merges_choices():
    game = get_scenario("operation_lethal").build(CONFIG)
    other = game.clone()
    assert state_key(other) == state_key(game)
    other.rng.random()  # same position, different future random draws
    assert state_key(other) != state_key(game)
    choice = run_line(game.clone(), ["PLAY(2)", "PLAY(2)"])
    assert choice.phase == CHOICE and state_key(choice) != state_key(choice.clone())


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


@pytest.mark.parametrize("spec", ["lookahead", "greedy"])
def test_lookahead_and_greedy_results_are_reported(spec):
    report = run_scenarios(spec, CONFIG)
    assert report.deterministic and report.n == len(SCENARIOS)
    assert all(isinstance(r.solved, bool) and r.success is None and r.line for r in report.results)
    instance = LookaheadAgent(CONFIG) if spec == "lookahead" else GreedyAgent(CONFIG)
    assert run_scenarios(instance, CONFIG).to_dict()["results"] == report.to_dict()["results"]  # seeded per run
    solved = {r.name: r.solved for r in report.results}
    assert solved["mulligan_sanity"] is (spec == "lookahead")  # lookahead's fixed rule; greedy keeps its hand
    print("\n" + report.format())  # reported, not asserted (SPEC 10)


def test_scenario_baselines():
    assert BASELINES == ("lookahead", "random")


def test_run_scenarios_with_a_ppo_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from cardgame.features import ObservationEncoder
    from cardgame.rl.agent import PPOAgent
    from cardgame.rl.network import TransformerPolicyNet

    torch.manual_seed(0)
    net = TransformerPolicyNet(ObservationEncoder(CONFIG).layout(), d_model=16, layers=1, heads=2, ff=32, id_dim=4)
    path = str(tmp_path / "ckpt_00001.pt")
    torch.save({"model": net.state_dict(), "net": net.spec(), "update": 1}, path)
    subset = SCENARIOS[:2] + tuple(get_scenario(n) for n in ("operation_lethal", "mulligan_sanity"))
    report = run_scenarios(path, CONFIG, n_stochastic=2, scenarios=subset)
    assert report.agent == path and report.n == 4 and report.deterministic
    for r in report.results:
        assert isinstance(r.solved, bool) and r.playouts == 2 and 0.0 <= r.success <= 1.0 and r.line
    d = report.to_dict()
    assert d["n_solved"] == report.n_solved and len(d["results"]) == 4

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
