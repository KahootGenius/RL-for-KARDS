"""Hand-built tactical positions with a known best line (SPEC §10), a DFS solver and a runner.

A `Scenario(name, tags, decks, build, goal)` builds a mid-game position (`build(config) -> Game`, the
agent to move, empty decks, cards from the named decks, consistent flags and coins) and judges the
agent's turn with `goal(start, end) -> bool`, where `end` is the state right after the agent's
END_TURN (or the game-over state). Units are identified by `uid`. `run_scenarios` plays an agent
(spec string or instance) through all of them; `solve` searches the turn exhaustively.

The searches (`solve`, `can_win_this_turn` behind `survives_next_turn`) know no game rules: they
explore the engine's own legal actions on clones and read the results back from the engine (a turn
is over when the game ended or another player is to move), so they stay correct when later stages
change the rules.
"""
from __future__ import annotations

import functools
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from .cards import FAST, GameConfig, load_ruleset
from .engine import Game, IllegalActionError, Unit

MAX_ACTIONS = 64       # a run that has not ended its turn after this many actions fails
SOLVER_DEPTH = 12      # actions per line in `solve`, including the final END_TURN
RANDOM_SEEDS = 20      # random agent: success averaged over this many seeds
SCENARIO_SEED = 4_000_000_000  # agent seeds for playouts (outside every deal-seed range)

Goal = Callable[[Game, Game], bool]


@dataclass(frozen=True)
class Scenario:
    name: str
    tags: Tuple[str, ...]
    decks: Tuple[str, str]                             # deck names of seat 0 and seat 1
    build: Callable[..., Game] = field(repr=False)     # build(config=None) -> Game, agent to move
    goal: Goal = field(repr=False)
    note: str = ""                                     # the known best line, in words


# ---------------------------------------------------------------------- positions
@dataclass(frozen=True)
class U:
    """A unit in a scenario position: card id, current hp (default full) and this-turn flags."""
    card: str
    hp: Optional[int] = None
    summoned: bool = False
    moved: bool = False
    attacked: bool = False


UnitSpec = Union[str, U]


def build_position(config: Optional[GameConfig] = None, *, decks: Tuple[str, str], seat: int = 0,
                   round: int = 5, first_player: Optional[int] = None, coins: Optional[int] = None,
                   my_base: int = 20, opp_base: int = 20, my_hand: Sequence[str] = (),
                   opp_hand: Sequence[str] = (), my_back: Sequence[UnitSpec] = (),
                   opp_back: Sequence[UnitSpec] = (), my_front: Sequence[UnitSpec] = (),
                   opp_front: Sequence[UnitSpec] = ()) -> Game:
    """A position with `seat` to move, described from that player's side ("my" = the agent's).

    `decks` are deck names in seat order. Coins default to the round's coins minus what was spent
    this turn (summoned units' costs and moved units' move costs). Decks are empty; `played` counts
    the units on the board. uids follow the order my_back, my_front, opp_front, opp_back.
    """
    config = config if config is not None else load_ruleset()
    if my_front and opp_front:
        raise ValueError("only one side can hold the frontline")
    names = list(config.deck_names)
    deck_ids = tuple(names.index(d) for d in decks)
    pool = config.cards
    me, opp = seat, 1 - seat
    game = Game(config)
    game.reset(0, deck_ids)
    game.round = round
    game.first_player = seat if first_player is None else first_player
    game.current = me
    game.deck_cards = [[], []]
    game.hands = [[], []]
    game.hands[me] = sorted(pool.by_id(c).index for c in my_hand)
    game.hands[opp] = sorted(pool.by_id(c).index for c in opp_hand)
    game.burned = [0, 0]
    game.base_hp = [0, 0]
    game.base_hp[me], game.base_hp[opp] = my_base, opp_base
    game.played = [[0] * len(pool), [0] * len(pool)]
    uid = 0

    def units(specs: Sequence[UnitSpec], owner: int) -> List[Unit]:
        nonlocal uid
        out = []
        for spec in specs:
            spec = U(spec) if isinstance(spec, str) else spec
            card = pool.by_id(spec.card)
            u = Unit.from_card(card, owner, uid)
            uid += 1
            u.hp = card.health if spec.hp is None else spec.hp
            u.summoned, u.moved, u.attacked = spec.summoned, spec.moved, spec.attacked
            game.played[owner][card.index] += 1
            out.append(u)
        return out

    game.backline = [[], []]
    game.backline[me] = units(my_back, me)
    game.frontline = units(my_front, me) + units(opp_front, opp)
    game.backline[opp] = units(opp_back, opp)
    game.front_owner = me if my_front else (opp if opp_front else None)
    game.next_uid = uid
    spent = _spent_this_turn(game, me)
    game.coins = [0, 0]
    game.coins[me] = config.coins_for_round(round) - spent if coins is None else coins
    game.num_steps = 0
    game.done, game._winner = False, None
    game.invalidate()
    return game


def _spent_this_turn(game: Game, p: int) -> int:
    cards = game.config.cards.cards
    own = [u for u in game.backline[p]] + [u for u in game.frontline if u.owner == p]
    return sum(cards[u.card].cost for u in own if u.summoned) + sum(u.move_cost for u in own if u.moved)


def check_position(scenario: Scenario, game: Game) -> List[str]:
    """Consistency problems of a built scenario position (empty list = consistent)."""
    cfg = game.config
    cards = cfg.cards.cards
    problems = []
    p, o = game.current, 1 - game.current
    names = list(cfg.deck_names)
    try:
        want = tuple(names.index(d) for d in scenario.decks)
    except ValueError:
        return [f"unknown deck in {scenario.decks}"]
    if tuple(game.deck_ids) != want:
        problems.append(f"deck_ids {game.deck_ids} != {want}")
    if game.deck_cards != [[], []]:
        problems.append("decks are not empty")
    if game.done or game.winner() is not None:
        problems.append("game is over")
    if not 1 <= game.round <= cfg.max_rounds:
        problems.append(f"round {game.round} out of range")
    if game.coins[o] != 0:
        problems.append("the opponent has coins during the agent's turn")
    expected = cfg.coins_for_round(game.round) - _spent_this_turn(game, p)
    if game.coins[p] != expected or game.coins[p] > game.round:
        problems.append(f"coins {game.coins[p]} != round coins minus this turn's spending ({expected})")
    if not all(1 <= hp <= cfg.base_hp for hp in game.base_hp):
        problems.append(f"base hp {game.base_hp} out of range")
    try:
        game.clone().invalidate()
    except ValueError as exc:
        problems.append(f"invariant: {exc}")
    board = [(u, "back") for z in game.backline for u in z] + [(u, "front") for u in game.frontline]
    uids = [u.uid for u, _ in board]
    if len(set(uids)) != len(uids) or any(not 0 <= x < game.next_uid for x in uids):
        problems.append(f"uids {uids} not unique or >= next_uid {game.next_uid}")
    for owner in (0, 1):
        if any(u.owner != owner for u in game.backline[owner]):
            problems.append(f"backline {owner} holds a unit of the other player")
    for u, zone in board:
        c = cards[u.card]
        if (u.atk, u.max_hp, u.armor, u.defense, u.nature, u.move_cost) != (
                c.attack, c.health, c.armor, c.defense, c.nature, c.move_cost):
            problems.append(f"uid {u.uid}: stats differ from card {c.id}")
        if not 1 <= u.hp <= u.max_hp:
            problems.append(f"uid {u.uid}: hp {u.hp} not in 1..{u.max_hp}")
        if u.owner == o and (u.summoned or u.moved or u.attacked):
            problems.append(f"uid {u.uid}: opponent unit with this-turn flags (refreshed at its END_TURN)")
        if u.summoned and (zone != "back" or u.moved or u.attacked):
            problems.append(f"uid {u.uid}: summoned unit that acted or left the backline")
        if u.moved and zone != "front":
            problems.append(f"uid {u.uid}: moved but not in the frontline")
        if u.moved and u.attacked and c.nature != FAST:  # only fast units may move and attack
            problems.append(f"uid {u.uid}: non-fast unit both moved and attacked")
    for owner in (0, 1):
        deck = Counter(cfg.decks[game.deck_ids[owner]])
        hand = Counter(game.hands[owner])
        on_board = Counter(u.card for u, _ in board if u.owner == owner)
        if game.hands[owner] != sorted(game.hands[owner]) or len(game.hands[owner]) > cfg.max_hand_size:
            problems.append(f"hand {owner} unsorted or too large")
        for card_index in set(hand) | set(on_board):
            played = game.played[owner][card_index]
            if played < on_board[card_index] or played + hand[card_index] > deck[card_index]:
                problems.append(f"player {owner}: {cards[card_index].id} x{hand[card_index]} in hand, "
                                f"{played} played, deck {names[game.deck_ids[owner]]} has {deck[card_index]}")
    return problems


# ---------------------------------------------------------------------- goal helpers
def _units(game: Game) -> Dict[int, Unit]:
    out = {u.uid: u for z in game.backline for u in z}
    out.update((u.uid, u) for u in game.frontline)
    return out


def _uids(start: Game, owner: int, card_id: str) -> List[int]:
    index = start.config.cards.by_id(card_id).index
    return [u.uid for u in _units(start).values() if u.owner == owner and u.card == index]


def win(start: Game, end: Game) -> bool:
    """The agent destroyed the enemy base."""
    return end.winner() == start.current


def kills(*card_ids: str) -> Goal:
    """Every enemy unit with one of these card ids (in the start position) is dead at the end."""
    def goal(start: Game, end: Game) -> bool:
        if win(start, end):
            return True
        alive = _units(end)
        return not any(uid in alive for c in card_ids for uid in _uids(start, 1 - start.current, c))
    goal.__doc__ = f"kills {', '.join(card_ids)}"
    return goal


def no_losses(start: Game, end: Game) -> bool:
    """Every unit the agent had at the start is still alive."""
    alive = _units(end)
    return all(uid in alive for uid, u in _units(start).items() if u.owner == start.current)


def survives_next_turn(start: Game, end: Game) -> bool:
    """The agent has won, or the opponent cannot destroy the agent's base on its next turn."""
    if end.done:
        return end.winner() == start.current
    if end.current == start.current:  # turn not ended yet: judge the position after END_TURN
        end = end.clone()
        end.step(end.action_space.END_TURN)
        if end.done:
            return end.winner() == start.current
    return not can_win_this_turn(end)


def all_of(*goals: Goal) -> Goal:
    def goal(start: Game, end: Game) -> bool:
        return all(g(start, end) for g in goals)
    goal.__doc__ = " and ".join((g.__doc__ or g.__name__).strip().split("\n")[0] for g in goals)
    return goal


# ---------------------------------------------------------------------- search
_UNIT_FIELDS = ("card", "owner", "uid", "atk", "hp", "max_hp", "armor", "defense", "nature", "move_cost",
                "summoned", "moved", "attacked")


def state_key(game: Game) -> tuple:
    """The whole game state except the RNG (for transposition tables): equal keys = equal futures."""
    def zone(units):
        return tuple(tuple(getattr(u, f) for f in _UNIT_FIELDS) for u in units)
    return (game.current, game.round, game.done, game.winner(), tuple(game.coins), tuple(game.base_hp),
            game.front_owner, tuple(map(tuple, game.hands)), tuple(map(tuple, game.deck_cards)),
            tuple(map(tuple, game.played)), tuple(game.burned), game.next_uid, zone(game.backline[0]),
            zone(game.backline[1]), zone(game.frontline))


def can_win_this_turn(game: Game) -> bool:
    """True if the player to move can destroy the enemy base before its turn ends.

    Exhaustive search with the engine alone: every legal action (plays included) is stepped on a
    clone, a line ends when the game is over or another player is to move, and transpositions are
    skipped. No rule is assumed (who can act, reach, damage); children are tried in order of the
    enemy base HP they leave, so a winning line is usually found first. The cost grows with the
    number of distinct lines (hand size, coins, units), which is small in the scenario positions.
    """
    if game.done:
        return False
    p = game.current
    seen = {state_key(game)}

    def search(g: Game) -> bool:
        children = []
        for a in g.legal_actions():
            c = g.clone()
            c.step(a)
            if c.done:
                if c.winner() == p:
                    return True
                continue
            if c.current != p:
                continue  # the turn is over
            key = state_key(c)
            if key not in seen:
                seen.add(key)
                children.append(c)
        children.sort(key=lambda c: c.base_hp[1 - p])  # stable: ties keep the action order
        return any(search(c) for c in children)

    return search(game)


def solve(scenario: Scenario, config: Optional[GameConfig] = None, max_depth: int = SOLVER_DEPTH,
          start: Optional[Game] = None) -> Optional[List[int]]:
    """A line of at most `max_depth` actions (ending with END_TURN or a win) that meets the goal, else None.

    Depth-first over the agent's turn on clones, with a transposition table keyed by state (the
    goal depends only on the start and end states).
    """
    start = start if start is not None else scenario.build(config)
    p = start.current
    best_depth: Dict[tuple, int] = {}

    def search(g: Game, depth: int) -> Optional[List[int]]:
        for a in g.legal_actions():
            c = g.clone()
            c.step(a)
            if c.done or c.current != p:  # the agent's turn is over
                if scenario.goal(start, c):
                    return [a]
                continue
            if depth + 2 > max_depth:  # no room left for the closing END_TURN
                continue
            key = state_key(c)
            if best_depth.get(key, max_depth + 1) <= depth + 1:
                continue
            best_depth[key] = depth + 1
            rest = search(c, depth + 1)
            if rest is not None:
                return [a] + rest
        return None

    return search(start.clone(), 0)


class EndState(NamedTuple):
    """Where one line of the agent's turn leaves the game (scored for the dominance check)."""
    won: bool
    enemy_base_hp: int
    enemy_board_value: int  # total card cost of the opponent's units on the board
    own_board_value: int    # total card cost of the agent's units on the board
    line: tuple             # the actions that reach it
    goal: bool


def end_of_turn_states(scenario: Scenario, config: Optional[GameConfig] = None,
                       start: Optional[Game] = None) -> List[EndState]:
    """Every distinct state the agent's turn can end in (END_TURN taken or game over), exhaustively,
    with the engine alone (clone/step/legal_actions) and a transposition table."""
    start = start if start is not None else scenario.build(config)
    p, o = start.current, 1 - start.current
    cost = [c.cost for c in start.config.cards.cards]
    seen, out = set(), []

    def value(g: Game, owner: int) -> int:
        return sum(cost[u.card] for u in _units(g).values() if u.owner == owner)

    def search(g: Game, line: tuple) -> None:
        for a in g.legal_actions():
            c = g.clone()
            c.step(a)
            key = state_key(c)
            if key in seen:
                continue
            seen.add(key)
            if c.done or c.current != p:
                out.append(EndState(c.winner() == p, c.base_hp[o], value(c, o), value(c, p), line + (a,),
                                    bool(scenario.goal(start, c))))
            else:
                search(c, line + (a,))

    search(start.clone(), ())
    return out


def dominance_violations(scenario: Scenario, config: Optional[GameConfig] = None) -> List[EndState]:
    """Validity of a scenario's "known best move" (empty list = valid).

    Material goals (kills / no_losses): every end state that misses the goal must be weakly
    dominated by a goal-meeting one on (won, enemy base HP, enemy board value, own board value), so
    missing the goal never buys anything. Returns the non-dominated misses. `win` goals are valid by
    definition; `survives_next_turn` goals are valid iff the agent has no lethal this turn (winning
    would beat surviving), and a lethal start position is reported as a violation.
    """
    start = scenario.build(config)
    if scenario.goal is win:
        return []
    if scenario.goal is survives_next_turn:
        if can_win_this_turn(start):
            return [EndState(True, 0, 0, 0, (), False)]
        return []
    states = end_of_turn_states(scenario, config, start)
    goals = [s for s in states if s.goal]

    def dominated(s: EndState) -> bool:
        return any(g.won >= s.won and g.enemy_base_hp <= s.enemy_base_hp and g.enemy_board_value <= s.enemy_board_value
                   and g.own_board_value >= s.own_board_value for g in goals)

    return [s for s in states if not s.goal and not dominated(s)]


# ---------------------------------------------------------------------- running agents
def play_scenario(scenario: Scenario, agent, config: Optional[GameConfig] = None,
                  seed: Optional[int] = None, start: Optional[Game] = None) -> Tuple[bool, List[int]]:
    """One playout of the agent's turn: (goal met, actions taken). The agent sees only observe() + legal."""
    game = start.clone() if start is not None else scenario.build(config)
    start = game.clone()
    p = game.current
    agent.reset(seed)
    line: List[int] = []
    while len(line) < MAX_ACTIONS:
        legal = game.legal_actions()
        a = agent.act(game.observe(p), legal)
        if a not in legal:
            raise IllegalActionError(f"agent {getattr(agent, 'name', agent)!r} chose illegal action {a!r} in "
                                     f"scenario {scenario.name!r}")
        game.step(int(a))
        line.append(int(a))
        if game.done or game.current != p:  # the agent's turn is over
            return bool(scenario.goal(start, game)), line
    return False, line


@dataclass
class ScenarioResult:
    name: str
    tags: Tuple[str, ...]
    solved: Optional[bool]    # the deterministic run (None: the agent has no deterministic mode)
    success: Optional[float]  # fraction of stochastic playouts that met the goal (None: not run)
    playouts: int
    line: Tuple[str, ...]     # actions of the deterministic run (else of the first playout)

    def to_dict(self) -> dict:
        return {"name": self.name, "tags": list(self.tags), "solved": self.solved, "success": self.success,
                "playouts": self.playouts, "line": list(self.line)}


@dataclass
class ScenarioReport:
    agent: str
    results: List[ScenarioResult]

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def deterministic(self) -> bool:
        return all(r.solved is not None for r in self.results)

    @property
    def n_solved(self) -> Optional[int]:
        return sum(bool(r.solved) for r in self.results) if self.deterministic else None

    @property
    def mean_success(self) -> Optional[float]:
        rates = [r.success for r in self.results if r.success is not None]
        return sum(rates) / len(rates) if rates and len(rates) == self.n else None

    @property
    def pct_solved(self) -> float:
        """% of scenarios solved by the deterministic run; for agents without one, the mean playout success."""
        if not self.n:
            return 0.0
        if self.deterministic:
            return 100.0 * self.n_solved / self.n
        return 100.0 * (self.mean_success or 0.0)

    def to_dict(self) -> dict:
        return {"agent": self.agent, "n": self.n, "n_solved": self.n_solved, "pct_solved": self.pct_solved,
                "mean_success": self.mean_success, "results": [r.to_dict() for r in self.results]}

    def format(self) -> str:
        width = max([len("scenario")] + [len(r.name) for r in self.results])
        lines = [f"{'scenario':<{width}s} {'solved':>7s} {'playouts':>9s}  tags"]
        for r in self.results:
            solved = "-" if r.solved is None else ("yes" if r.solved else "no")
            rate = "-" if r.success is None else f"{100 * r.success:.0f}% /{r.playouts}"
            lines.append(f"{r.name:<{width}s} {solved:>7s} {rate:>9s}  {','.join(r.tags)}")
        ms = self.mean_success
        lines.append(f"{self.agent}: {self.pct_solved:.0f}% solved"
                     + (f" ({self.n_solved}/{self.n})" if self.deterministic else "")
                     + ("" if ms is None else f"; mean playout success {100 * ms:.0f}%"))
        return "\n".join(lines)


def _agent_modes(agent_or_spec, config: GameConfig, deterministic: bool, n_stochastic: int, n_random: int):
    """(label, deterministic-run agent or None, stochastic agent or None, number of playouts)."""
    from .agents import RandomAgent, make_agent
    if isinstance(agent_or_spec, str):
        spec = agent_or_spec
        if spec == "random":
            return spec, None, RandomAgent(config), n_random
        if spec == "greedy":
            return spec, make_agent(spec, config), None, 0
        det = make_agent(spec, config, deterministic=deterministic)
        stoch = make_agent(spec, config) if n_stochastic > 0 else None
        return spec, det, stoch, n_stochastic
    agent = agent_or_spec
    label = getattr(agent, "name", type(agent).__name__)
    if isinstance(agent, RandomAgent):
        return label, None, agent, n_random
    if hasattr(agent, "deterministic"):  # e.g. PPOAgent: toggled per run below
        return label, agent, agent if n_stochastic > 0 else None, n_stochastic
    return label, agent, None, 0


def playout_seed(seed: int, scenario: str, playout: int) -> int:
    """Agent seed of one playout, stable per (seed, scenario name, playout index) on every platform."""
    from .evaluation import agent_seed
    return agent_seed(SCENARIO_SEED + zlib.crc32(f"{scenario}:{seed}:{playout}".encode("utf-8")), 0)


def run_scenarios(agent_or_spec, config: Optional[GameConfig] = None, deterministic: bool = True,
                  n_stochastic: int = 100, scenarios: Optional[Sequence[Scenario]] = None, seed: int = 0,
                  n_random: int = RANDOM_SEEDS) -> ScenarioReport:
    """Play every scenario: one deterministic run (argmax for PPO agents, unless deterministic=False:
    then one seeded sampled run) plus `n_stochastic` sampled playouts for PPO agents; greedy once;
    random over `n_random` seeds. Agents are reseeded per playout (reproducible)."""
    config = config if config is not None else load_ruleset()
    scenarios = list(SCENARIOS if scenarios is None else scenarios)
    label, det_agent, stoch_agent, n_play = _agent_modes(agent_or_spec, config, deterministic, n_stochastic,
                                                         n_random)
    results = []
    for sc in scenarios:
        start = sc.build(config)
        desc = start.describe
        solved, line = None, None
        if det_agent is not None:
            saved = getattr(det_agent, "deterministic", None)
            if saved is not None:
                det_agent.deterministic = deterministic
            try:
                ok, actions = play_scenario(sc, det_agent, config, playout_seed(seed, sc.name, 0), start)
            finally:
                if saved is not None:
                    det_agent.deterministic = saved
            solved, line = ok, tuple(desc(a) for a in actions)
        success = None
        if stoch_agent is not None and n_play > 0:
            saved = getattr(stoch_agent, "deterministic", None)
            if saved is not None:
                stoch_agent.deterministic = False
            hits = 0
            try:
                for i in range(n_play):
                    ok, actions = play_scenario(sc, stoch_agent, config, playout_seed(seed, sc.name, 1 + i), start)
                    hits += ok
                    if line is None:
                        line = tuple(desc(a) for a in actions)
            finally:
                if saved is not None:
                    stoch_agent.deterministic = saved
            success = hits / n_play
        results.append(ScenarioResult(sc.name, tuple(sc.tags), solved, success, n_play if success is not None else 0,
                                      line or ()))
    return ScenarioReport(label, results)


# ---------------------------------------------------------------------- the scenarios
def _scenario(name: str, tags: Sequence[str], goal: Goal, note: str, *, mine: str, theirs: str, seat: int = 0,
              **position) -> Scenario:
    decks = (mine, theirs) if seat == 0 else (theirs, mine)
    build = functools.partial(build_position, decks=decks, seat=seat, **position)
    return Scenario(name, tuple(tags), decks, build, goal, note)


SCENARIOS: Tuple[Scenario, ...] = (
    _scenario(
        "fast_move_attack_lethal", ("fast", "lethal", "move_cost"), win,
        "Move the Lancer (fast) to the empty frontline and hit the base for exactly 5. Playing the "
        "Swordsman first spends the coin the move needs; the Footman was deployed this round and cannot act.",
        mine="Blitz", theirs="Bulwark", round=5, opp_base=5, my_base=12,
        my_hand=("militia", "swordsman"), my_back=("lancer", U("footman", summoned=True)),
        opp_hand=("knight", "militia", "footman"), opp_back=("pikeman", "archer")),
    _scenario(
        "fast_attack_then_move", ("fast", "frontline", "threat"), survives_next_turn,
        "Outrider (fast) kills the Raider from the backline, then moves into the emptied frontline: "
        "holding it stops the Lancer + Wolf Rider (11) from both moving in next turn. Playing the "
        "Militia first leaves no coin for the move.",
        mine="Blitz", theirs="Legion", round=5, my_base=9,
        my_hand=("militia", "paladin"), my_back=("outrider", U("knight", summoned=True)),
        opp_hand=("footman", "militia"), opp_front=("raider",), opp_back=("lancer", "wolf_rider")),
    _scenario(
        "defense_backline_order", ("defense", "sequencing", "threat"), survives_next_turn,
        "Next turn the Longbowman (4) shoots the base (4 HP) for lethal. The Shieldbearer (Defense) shields it, "
        "so the Swordsman (4) must kill the Shieldbearer first; then the Knight (3) finishes the Longbowman. "
        "Opening with the Knight on the Shieldbearer leaves both alive; no line wins this turn (7 < 14).",
        mine="Legion", theirs="Volley", round=6, my_base=4, opp_base=14,
        my_hand=("archer", "footman"), my_front=("swordsman", "knight"),
        opp_hand=("slinger", "militia"), opp_back=("shieldbearer", "longbowman")),
    _scenario(
        "defense_frontline_ranged", ("defense", "ranged", "frontline", "threat"), survives_next_turn,
        "Next turn the Wolf Rider (6) and the Ballista (5) hit the base for 11 >= 8. The Shieldbearer (Defense) "
        "guards the frontline, so a melee unit kills it first; then the Crossbowman and Archer shoot the Wolf "
        "Rider down (3 + 2 >= 4, no return damage). The Ballista (armor 1, 5 HP) cannot be killed this turn. "
        "Trading the Ogre or Swordsman into the Wolf Rider also survives, but loses a unit.",
        mine="Volley", theirs="Legion", round=7, my_base=8, opp_base=16,
        my_back=("crossbowman", "archer", "swordsman", "ogre"), my_hand=("militia",),
        opp_hand=("footman",), opp_front=("shieldbearer", "wolf_rider"), opp_back=("ballista",)),
    _scenario(
        "ranged_finish_backline", ("ranged", "armor", "threat"), survives_next_turn,
        "Next turn the Pikeman (3), Ballista (5) and Longbowman (4) hit for 12 >= 5; only killing both damaged "
        "shooters (leaving 3) survives. Only the Crossbowman (3 - 1 armor = 2) finishes the 2-HP Ballista, so the "
        "Archer must take the Longbowman. Melee cannot reach the backline past the enemy-held frontline.",
        mine="Legion", theirs="Volley", round=7, my_base=5, opp_base=15,
        my_back=("crossbowman", "archer", "footman"), my_hand=("swordsman",),
        opp_hand=("militia",), opp_front=("pikeman",),
        opp_back=(U("ballista", hp=2), U("longbowman", hp=1))),
    _scenario(
        "ranged_base_lethal", ("ranged", "lethal", "seat1"), win,
        "Crossbowman (3) + Slinger (1) shoot the base for exactly 4; the enemy holds the frontline, "
        "so melee cannot reach the base.",
        mine="Volley", theirs="Bulwark", seat=1, first_player=0, round=6, opp_base=4, my_base=11,
        my_hand=("archer", "pikeman"), my_back=("crossbowman", "slinger", "swordsman"),
        opp_hand=("militia", "footman"), opp_front=("pikeman", "warden"), opp_back=("archer",)),
    _scenario(
        "armor_pierce", ("armor", "trade"), all_of(kills("outrider"), no_losses),
        "Only the Swordsman (4 - 1 armor = 3) kills the Outrider and survives; the Raider deals 2 and "
        "dies, the Militia deals 0 and dies.",
        mine="Blitz", theirs="Bulwark", round=6, my_base=15, opp_base=15,
        my_back=("raider", "militia", "footman", "swordsman"), my_hand=("scout",),
        opp_hand=("knight",), opp_front=("outrider",), opp_back=("shieldbearer", "archer")),
    _scenario(
        "armor_attacker_survives", ("armor", "trade"), all_of(kills("swordsman"), no_losses),
        "The enemy Swordsman (4/3) holds the frontline. Our Swordsman would trade itself away and the Footman "
        "cannot kill it; only the Knight's armor (4 - 1 = 3 < 4 HP) lets it kill and survive. Nothing can reach "
        "the base this turn (the frontline is enemy-held, and troops cannot attack after moving in).",
        mine="Blitz", theirs="Legion", round=6, my_base=14, opp_base=15,
        my_back=("knight", "swordsman", "footman"), my_hand=("militia",),
        opp_hand=("archer",), opp_front=("swordsman",), opp_back=("footman",)),
    _scenario(
        "move_cost_hold_frontline", ("move_cost", "frontline", "threat"), survives_next_turn,
        "Spend the 2 coins moving the Knight (move cost 2) into the empty frontline: the Lancer and "
        "Wolf Rider (11 damage) can then not both move in. Playing a card instead loses next turn.",
        mine="Bulwark", theirs="Blitz", round=6, my_base=10, opp_base=18,
        my_hand=("archer", "footman", "militia"), my_back=("knight", U("pikeman", summoned=True)),
        opp_hand=("scout", "swordsman"), opp_back=("lancer", "wolf_rider")),
    _scenario(
        "clear_frontline_then_lethal", ("frontline", "defense", "fast", "lethal"), win,
        "Troops clear the frontline (Shieldbearer first: Defense), keeping the Lancer fresh to move in "
        "and hit the base for 5. Using the Lancer to clear, or playing the Archer, loses the lethal.",
        mine="Blitz", theirs="Bulwark", round=6, opp_base=5, my_base=12,
        my_hand=("archer", "scout"), my_back=("lancer", "swordsman", "footman", U("knight", summoned=True)),
        opp_hand=("warden",), opp_front=("shieldbearer", "militia"), opp_back=("footman",)),
    _scenario(
        "stop_lethal_kill_right_unit", ("threat", "ranged"), survives_next_turn,
        "Next turn the enemy hits for 4 + 3 + 2 = 9 >= 6. Only shooting the Swordsman (4 atk) brings it "
        "below 6; the damaged Knight is the pricier card but the wrong target.",
        mine="Volley", theirs="Legion", round=5, my_base=6, opp_base=17,
        my_hand=("militia", "longbowman"), my_back=("crossbowman", U("footman", summoned=True)),
        opp_hand=("raider", "pikeman"), opp_front=("swordsman", U("knight", hp=2)), opp_back=("archer",)),
    _scenario(
        "ranged_no_return", ("ranged", "trade", "fast", "lethal"), win,
        "The Longbowman shoots the 3-HP Ogre (ranged: no return damage), emptying the frontline; the Wolf Rider "
        "(fast) then moves in and hits the base for exactly 6. If the Wolf Rider attacks the Ogre it dies to the "
        "7 return damage, and the Longbowman alone deals only 4.",
        mine="Legion", theirs="Volley", round=7, my_base=13, opp_base=6,
        my_back=("wolf_rider", "longbowman"), my_hand=("slinger",),
        opp_hand=("knight",), opp_front=(U("ogre", hp=3),), opp_back=("archer",)),
    _scenario(
        "budget_cheap_movers", ("move_cost", "fast", "lethal"), win,
        "With 2 coins, Lancer (move 1) + Raider (move 1) hit for 8 >= 8 (the free Scout adds 2 more); moving the "
        "Cataphract (move 2) first leaves only 5 + 2 = 7.",
        mine="Blitz", theirs="Bulwark", round=6, opp_base=8, my_base=14,
        my_back=("cataphract", "lancer", "raider", "scout", U("knight", summoned=True)),
        opp_hand=("bastion",), opp_back=("pikeman",)),
    _scenario(
        "base_ignores_defense", ("defense", "lethal", "seat1"), win,
        "Defense never protects the base: Swordsman + Footman + Raider hit it for exactly 9.",
        mine="Legion", theirs="Bulwark", seat=1, first_player=0, round=7, opp_base=9, my_base=10,
        my_hand=("archer",), my_front=("swordsman", "footman", "raider"),
        opp_hand=("militia",), opp_back=("shieldbearer", "pikeman", "warden")),
    _scenario(
        "sacrifice_to_clear", ("frontline", "fast", "lethal", "trade"), win,
        "The Militia trades with the Raider (mutual kill empties the frontline); the Wolf Rider then "
        "moves in and hits for 6. Using the Wolf Rider to kill the Raider leaves no attack for the base.",
        mine="Blitz", theirs="Legion", round=5, opp_base=6, my_base=11,
        my_back=("wolf_rider", "militia", U("knight", summoned=True)), my_hand=("paladin",),
        opp_hand=("archer",), opp_front=("raider",), opp_back=("archer", "footman")),
    _scenario(
        "ranged_through_defense", ("ranged", "defense", "sequencing", "threat"), survives_next_turn,
        "Next turn the Longbowman (4), Footman (2) and Shield Archer (1) hit for 7 >= 5. Defense also binds ranged "
        "attacks: Crossbowman (3) + Slinger (1) must kill the Shield Archer (4 HP) before the Archer can finish "
        "the 2-HP Longbowman, leaving 2 damage. Shooting the Footman or the Shield Archer alone still loses.",
        mine="Volley", theirs="Volley", round=6, my_base=5, opp_base=13,
        my_back=("crossbowman", "archer", "slinger"), my_hand=("militia", "footman"),
        opp_hand=("pikeman",), opp_front=("footman",), opp_back=("shield_archer", U("longbowman", hp=2))),

)


def scenario_names() -> List[str]:
    return [s.name for s in SCENARIOS]


def get_scenario(name: str) -> Scenario:
    for s in SCENARIOS:
        if s.name == name:
            return s
    raise KeyError(f"unknown scenario {name!r}")


__all__ = ["MAX_ACTIONS", "SCENARIOS", "Scenario", "ScenarioReport", "ScenarioResult", "U", "all_of",
           "build_position", "can_win_this_turn", "check_position", "get_scenario", "kills", "no_losses",
           "play_scenario", "run_scenarios", "scenario_names", "solve", "state_key", "survives_next_turn", "win"]
