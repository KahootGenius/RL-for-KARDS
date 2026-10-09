"""Headless, deterministic game engine (Stage 2). All rules live here; see SPEC.md."""
from __future__ import annotations

import copy
import operator
import random
from bisect import insort
from typing import NamedTuple, Optional, Sequence

import numpy as np

from .actions import ActionSpace
from .cards import FAST, NATURES, RANGED, TROOP, CardDef, GameConfig, load_ruleset, sample_decks

DRAW = -1

__all__ = ["DRAW", "TROOP", "FAST", "RANGED", "IllegalActionError", "UnitView", "Observation", "Unit", "Game",
           "combat_damage"]


def combat_damage(attacker, target) -> tuple:
    """(damage to the target, damage to the attacker) of one attack between two units.

    The single source of the combat rules (SPEC §2): armor reduces each hit (never below 0) and
    ranged attackers take no return damage. Works on `Unit` and `UnitView` alike, so agents can
    reason about trades without re-implementing the rules.
    """
    to_target = max(0, attacker.atk - target.armor)
    to_attacker = 0 if attacker.nature == RANGED else max(0, target.atk - attacker.armor)
    return to_target, to_attacker


class IllegalActionError(ValueError):
    pass


class UnitView(NamedTuple):
    card: int
    atk: int
    hp: int
    max_hp: int
    armor: int
    defense: bool
    nature: int
    move_cost: int
    summoned: bool     # deployed this round
    moved: bool
    attacked: bool
    can_move: bool     # action economy only (coins, space and frontline control not included)
    can_attack: bool   # action economy only (reach, Defense and targets not included)


class Observation(NamedTuple):
    """Everything one player may see, from that player's point of view."""
    player: int
    is_my_turn: bool
    went_first: bool
    round: int
    my_deck: int       # own deck index; the opponent's deck choice is hidden
    my_coins: int
    opp_coins: int
    my_base_hp: int
    opp_base_hp: int
    hand: tuple
    opp_hand_size: int
    my_deck_size: int
    opp_deck_size: int
    my_played: tuple   # copies of each card index the observer has played this game (public)
    opp_played: tuple  # same for the opponent (every played card was on the board)
    my_backline: tuple
    opp_backline: tuple
    frontline: tuple
    front_owner: int  # +1 observer, -1 opponent, 0 empty
    done: bool
    result: int       # +1 observer won, -1 lost, 0 draw/ongoing


class Unit:
    """A unit on the board. All fields are immutable scalars, so copying a unit stays shallow; later
    stages must keep any per-unit collection immutable (tuple/frozenset, replaced on write)."""
    __slots__ = ("card", "owner", "uid", "atk", "hp", "max_hp", "armor", "defense", "nature", "move_cost",
                 "summoned", "moved", "attacked")

    def __init__(self, card: int, owner: int, *, atk: int, hp: int, max_hp: Optional[int] = None, armor: int = 0,
                 defense: bool = False, nature: int = TROOP, move_cost: int = 1, uid: int = -1,
                 summoned: bool = False, moved: bool = False, attacked: bool = False):
        self.card, self.owner, self.uid, self.atk, self.hp = card, owner, uid, atk, hp
        self.max_hp = hp if max_hp is None else max_hp
        self.armor, self.defense, self.nature, self.move_cost = armor, defense, nature, move_cost
        self.summoned, self.moved, self.attacked = summoned, moved, attacked

    @classmethod
    def from_card(cls, c: CardDef, owner: int, uid: int = -1) -> "Unit":
        """A freshly deployed unit (cannot act this round)."""
        return cls(c.index, owner, atk=c.attack, hp=c.health, max_hp=c.health, armor=c.armor, defense=c.defense,
                   nature=c.nature, move_cost=c.move_cost, uid=uid, summoned=True)

    def can_move(self) -> bool:
        return not self.summoned and not self.moved and (self.nature == FAST or not self.attacked)

    def can_attack(self) -> bool:
        return not self.summoned and not self.attacked and (self.nature == FAST or not self.moved)

    def copy(self) -> "Unit":
        cls = type(self)
        u = cls.__new__(cls)
        if cls is Unit:  # fast path: plain attribute copies, no deepcopy
            (u.card, u.owner, u.uid, u.atk, u.hp, u.max_hp, u.armor, u.defense, u.nature, u.move_cost,
             u.summoned, u.moved, u.attacked) = (
                self.card, self.owner, self.uid, self.atk, self.hp, self.max_hp, self.armor, self.defense,
                self.nature, self.move_cost, self.summoned, self.moved, self.attacked)
        else:  # subclasses (later stages) may add slots
            for klass in cls.__mro__:
                for name in getattr(klass, "__slots__", ()):
                    if hasattr(self, name):
                        setattr(u, name, copy.deepcopy(getattr(self, name)))
        return u

    def view(self) -> UnitView:
        return UnitView(self.card, self.atk, self.hp, self.max_hp, self.armor, self.defense, self.nature,
                        self.move_cost, self.summoned, self.moved, self.attacked, self.can_move(),
                        self.can_attack())

    def __repr__(self) -> str:
        flags = "".join(f for f, on in (("S", self.summoned), ("M", self.moved), ("A", self.attacked)) if on)
        return (f"Unit(card={self.card}, p{self.owner}, uid={self.uid}, {NATURES[self.nature]} {self.atk}/{self.hp}"
                f"{' def' if self.defense else ''}{f' armor{self.armor}' if self.armor else ''}"
                f"{' ' + flags if flags else ''})")


# Attributes that are immutable for the lifetime of a Game and may be shared by clones.
_SHARED_ATTRS = frozenset({"config", "action_space", "num_actions", "_cost", "_card_defs"})
_SCALAR_TYPES = (int, float, str, bool, type(None))


def _as_index(value, what: str) -> int:
    """Strict integer conversion: ints and numpy ints, never bools/floats/None."""
    if type(value) is int:
        return value
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{what} must be an integer, got {value!r}")
    return operator.index(value)


class Game:
    def __init__(self, config: Optional[GameConfig] = None):
        self.config = config if config is not None else load_ruleset()
        cfg = self.config
        self.action_space = ActionSpace(cfg.max_hand_size, cfg.zone_capacity)
        self.num_actions = self.action_space.n
        self._card_defs = cfg.cards.cards
        self._cost = tuple(c.cost for c in self._card_defs)
        # Not started: queries are safe (no legal actions, no winner) until reset().
        self.done = True
        self._winner = None
        self.current = 0
        self.round = 0
        self.num_steps = 0
        self._legal = []
        self._mask = bytearray(self.num_actions)

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int, decks: Optional[Sequence[int]] = None) -> None:
        seed = _as_index(seed, "seed")  # ints and numpy ints; rejects None/bools/floats (determinism)
        if seed < 0:
            raise ValueError("seed must be a non-negative integer")  # Random(-s) == Random(s)
        cfg = self.config
        n_decks = cfg.n_decks
        if decks is not None:
            decks = tuple(_as_index(d, "deck index") for d in decks)
            if len(decks) != 2 or not all(0 <= d < n_decks for d in decks):
                raise ValueError(f"decks must be two deck indices in [0, {n_decks}), got {decks!r}")
        if decks is None:  # own stream: reset(s) == reset(s, decks=sample_decks(s, n))
            decks = sample_decks(seed, n_decks)
        self.seed = seed
        self.rng = random.Random(seed)
        self.deck_ids = decks
        self.next_uid = 0
        self.first_player = self.rng.randrange(2)
        self.deck_cards = [list(cfg.decks[decks[0]]), list(cfg.decks[decks[1]])]
        self.rng.shuffle(self.deck_cards[0])
        self.rng.shuffle(self.deck_cards[1])
        self.hands = [[], []]
        self.played = [[0] * len(self._card_defs), [0] * len(self._card_defs)]
        self.burned = [0, 0]
        self.base_hp = [cfg.base_hp, cfg.base_hp]
        self.coins = [0, 0]
        self.backline = [[], []]
        self.frontline = []
        self.front_owner = None
        self.round = 1
        self.current = self.first_player
        self.done = False
        self._winner = None
        self.num_steps = 0
        self._legal = self._mask = None
        first = self.first_player
        self._draw(first, cfg.opening_hand[0])
        self._draw(1 - first, cfg.opening_hand[1])
        self._start_turn(first)

    def _draw(self, p: int, n: int) -> None:
        deck, hand = self.deck_cards[p], self.hands[p]
        for _ in range(n):
            if not deck:
                return  # no fatigue
            card = deck.pop()
            if len(hand) >= self.config.max_hand_size:
                self.burned[p] += 1
            else:
                insort(hand, card)  # hand kept sorted by card index (canonical order)

    def _start_turn(self, p: int) -> None:
        self.current = p
        self.coins[p] = self.config.coins_for_round(self.round)
        self._draw(p, 1)
        self._legal = self._mask = None

    # ------------------------------------------------------------------ queries
    def current_player(self) -> int:
        return self.current

    def winner(self) -> Optional[int]:
        return self._winner

    @staticmethod
    def _targetable(zone: list) -> list:
        """Slots of `zone` that may be attacked: the Defense units if there are any (SPEC §2)."""
        guards = [k for k, u in enumerate(zone) if u.defense]
        return guards if guards else list(range(len(zone)))

    def _compute_legal(self) -> None:
        sp = self.action_space
        Z = self.config.zone_capacity
        legal = []
        if not self.done:
            p = self.current
            o = 1 - p
            legal.append(sp.END_TURN)
            back = self.backline[p]
            front = self.frontline
            fo = self.front_owner
            coins = self.coins[p]
            if len(back) < Z:
                cost = self._cost
                for i, card in enumerate(self.hands[p]):
                    if cost[card] <= coins:
                        legal.append(sp.PLAY0 + i)
            if (fo is None or fo == p) and len(front) < Z:
                for j, u in enumerate(back):  # inlined Unit.can_move()
                    if (u.move_cost <= coins and not u.summoned and not u.moved
                            and (u.nature == FAST or not u.attacked)):
                        legal.append(sp.MOVE0 + j)
            # inlined Unit.can_attack()
            attackers = [(a, u) for a, u in enumerate(back)
                         if not u.summoned and not u.attacked and (u.nature == FAST or not u.moved)]
            if fo == p:
                attackers += [(Z + j, u) for j, u in enumerate(front)
                              if not u.summoned and not u.attacked and (u.nature == FAST or not u.moved)]
            if attackers:
                enemy_back = self._targetable(self.backline[o])
                enemy_front = [Z + k for k in self._targetable(front)] if fo == o else []
                base_t = sp.BASE_TARGET
                ranged_targets = enemy_back + enemy_front + [base_t]
                front_melee = enemy_back + [base_t]
                n_t = sp.n_targets
                for a, u in attackers:
                    if u.nature == RANGED:
                        targets = ranged_targets
                    elif a < Z:
                        targets = enemy_front
                    else:
                        targets = front_melee
                    first = sp.ATTACK0 + a * n_t
                    legal += [first + t for t in targets]
        mask = bytearray(self.num_actions)
        for a in legal:
            mask[a] = 1
        self._legal, self._mask = legal, mask

    def legal_actions(self) -> list:
        if self._legal is None:
            self._compute_legal()
        return list(self._legal)

    def legal_mask(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Bool mask of shape (num_actions,). Writes into `out` (bool array) when given."""
        if self._mask is None:
            self._compute_legal()
        mask = np.frombuffer(self._mask, dtype=bool)
        if out is None:
            return mask.copy()
        out[:] = mask
        return out

    def invalidate(self) -> None:
        """Drop cached legal actions after mutating state directly (tests, tools).

        Also checks the structural invariants the fixed action space relies on.
        """
        self._legal = self._mask = None
        cfg = self.config
        if not self.done:
            for p in (0, 1):
                if len(self.hands[p]) > cfg.max_hand_size:
                    raise ValueError(f"player {p} hand exceeds max_hand_size")
                if len(self.backline[p]) > cfg.zone_capacity:
                    raise ValueError(f"player {p} backline exceeds zone_capacity")
                if any(u.owner != p for u in self.backline[p]):
                    raise ValueError(f"player {p} backline holds an enemy unit")
            if len(self.frontline) > cfg.zone_capacity:
                raise ValueError("frontline exceeds zone_capacity")
            owners = {u.owner for u in self.frontline}
            if (self.front_owner is None) != (not self.frontline) or (owners and owners != {self.front_owner}):
                raise ValueError("front_owner inconsistent with frontline units")

    def is_legal(self, action: int) -> bool:
        try:
            action = _as_index(action, "action")
        except TypeError:
            return False
        if self._mask is None:
            self._compute_legal()
        return 0 <= action < self.num_actions and self._mask[action] == 1

    # ------------------------------------------------------------------ transitions
    def step(self, action: int) -> None:
        if self.done:
            raise IllegalActionError("game is over")
        try:
            action = _as_index(action, "action")
        except TypeError:
            raise IllegalActionError(f"action must be an integer, got {action!r}") from None
        if self._mask is None:
            self._compute_legal()
        if not (0 <= action < self.num_actions and self._mask[action]):
            raise IllegalActionError(f"illegal action {action} ({self.describe(action)})")
        sp = self.action_space
        Z = self.config.zone_capacity
        p = self.current
        o = 1 - p
        if action == sp.END_TURN:
            self._end_turn(p)
        elif action < sp.MOVE0:
            card = self.hands[p].pop(action - sp.PLAY0)
            self.coins[p] -= self._cost[card]
            self.played[p][card] += 1
            self.backline[p].append(Unit.from_card(self._card_defs[card], p, self.next_uid))
            self.next_uid += 1
        elif action < sp.ATTACK0:
            u = self.backline[p].pop(action - sp.MOVE0)
            self.coins[p] -= u.move_cost
            u.moved = True
            self.frontline.append(u)
            self.front_owner = p
        elif action < sp.n:
            a, t = divmod(action - sp.ATTACK0, sp.n_targets)
            attacker = self.backline[p][a] if a < Z else self.frontline[a - Z]
            attacker.attacked = True
            if t == sp.BASE_TARGET:
                self._damage_base(o, attacker.atk, attacker=p)
            else:
                target = self.backline[o][t] if t < Z else self.frontline[t - Z]
                self._combat(attacker, target)
        else:  # a new action kind was added to ActionSpace without a handler here
            raise IllegalActionError(f"no handler for action {action}")
        self.num_steps += 1
        self._legal = self._mask = None

    def _damage_base(self, p: int, amount: int, attacker: int) -> None:
        self.base_hp[p] -= amount
        if self.base_hp[p] <= 0:
            self.done = True
            self._winner = attacker

    def _combat(self, attacker: Unit, target: Unit) -> None:
        to_target, to_attacker = combat_damage(attacker, target)  # simultaneous: pre-combat values
        target.hp -= to_target
        attacker.hp -= to_attacker
        if attacker.hp <= 0 or target.hp <= 0:
            self._resolve_deaths()

    def _resolve_deaths(self) -> None:
        """Remove dead units (zones compact, order preserved). Hook point for on-death effects."""
        self.frontline = [u for u in self.frontline if u.hp > 0]
        if not self.frontline:
            self.front_owner = None
        self.backline = [[u for u in bl if u.hp > 0] for bl in self.backline]

    def _end_turn(self, p: int) -> None:
        self.coins[p] = 0  # unused coins are lost
        # Refresh the ending player's units now, so during the opponent's turn their flags already
        # describe what they can do on their owner's next turn (legality is unaffected).
        for u in self.backline[p]:
            u.summoned = u.moved = u.attacked = False
        if self.front_owner == p:
            for u in self.frontline:
                u.summoned = u.moved = u.attacked = False
        o = 1 - p
        if o == self.first_player:
            if self.round >= self.config.max_rounds:
                self.done = True
                self._winner = DRAW
                return
            self.round += 1
        self._start_turn(o)

    # ------------------------------------------------------------------ views
    def observe(self, player: int) -> Observation:
        if not hasattr(self, "rng"):
            raise RuntimeError("call reset(seed) before observe()")
        player = _as_index(player, "player")
        if player not in (0, 1):
            raise ValueError(f"player must be 0 or 1, got {player}")
        o = 1 - player
        fo = self.front_owner
        if self.done:
            w = self._winner
            result = 0 if w == DRAW else (1 if w == player else -1)
        else:
            result = 0
        return Observation(
            player, (not self.done) and self.current == player, self.first_player == player, self.round,
            self.deck_ids[player], self.coins[player], self.coins[o], self.base_hp[player], self.base_hp[o],
            tuple(self.hands[player]), len(self.hands[o]), len(self.deck_cards[player]), len(self.deck_cards[o]),
            tuple(self.played[player]), tuple(self.played[o]),
            tuple([u.view() for u in self.backline[player]]),
            tuple([u.view() for u in self.backline[o]]),
            tuple([u.view() for u in self.frontline]),
            0 if fo is None else (1 if fo == player else -1), self.done, result,
        )

    def clone(self) -> "Game":
        """Independent copy (including RNG state). Hot fields are copied by hand; unknown mutable
        attributes are deep-copied, so state added by later stages is never shared between clones."""
        cls = self.__class__
        g = cls.__new__(cls)
        d = g.__dict__
        memo = {}  # shared, so references to board units elsewhere stay aliased in the clone
        units = {}
        src = self.__dict__
        if "backline" in src:
            d["backline"] = [[units.setdefault(id(u), u.copy()) for u in z] for z in src["backline"]]
            d["frontline"] = [units.setdefault(id(u), u.copy()) for u in src["frontline"]]
        for uid, u in units.items():
            memo[uid] = u
        for k, v in src.items():
            if k in ("_legal", "_mask", "backline", "frontline"):
                continue  # caches are recomputed lazily; zones copied above
            if k in _SHARED_ATTRS or isinstance(v, _SCALAR_TYPES):
                d[k] = v
            elif k == "rng":
                r = type(v).__new__(type(v))  # skip urandom seeding; state set below
                r.setstate(v.getstate())
                d[k] = r
            elif k in ("deck_cards", "hands", "played"):
                d[k] = [list(x) for x in v]
            elif k in ("burned", "base_hp", "coins"):
                d[k] = list(v)
            elif k == "deck_ids" and type(v) is tuple and all(type(x) is int for x in v):
                d[k] = v
            else:
                d[k] = copy.deepcopy(v, memo)
        g._legal = g._mask = None
        if g.done and not hasattr(g, "rng"):  # never reset
            g._legal, g._mask = [], bytearray(g.num_actions)
        return g

    def describe(self, action: int) -> str:
        try:
            action = _as_index(action, "action")
        except TypeError:
            return f"<not an action {action!r}>"
        if 0 <= action < self.num_actions:
            return self.action_space.describe(action)
        return f"<out of range {action!r}>"

    def render(self) -> str:
        """Omniscient debug view. Never give this to an agent."""
        if not hasattr(self, "rng"):
            return "<game not started: call reset(seed)>"
        cards = self._card_defs
        tag = {TROOP: "", FAST: "^", RANGED: "~"}

        def zone(units):
            out = []
            for u in units:
                traits = ("D" if u.defense else "") + (f"A{u.armor}" if u.armor else "")
                flags = "".join(f for f, on in (("*", u.summoned), ("m", u.moved), ("a", u.attacked)) if on)
                out.append(f"{tag[u.nature]}{cards[u.card].name}({u.atk}/{u.hp}{' ' + traits if traits else ''}){flags}")
            return " ".join(out) or "-"

        owner = "-" if self.front_owner is None else f"P{self.front_owner}"
        names = self.config.deck_names
        lines = [
            f"round {self.round}  current P{self.current}  first P{self.first_player}  "
            f"done={self.done} winner={self._winner}  decks {names[self.deck_ids[0]]} vs {names[self.deck_ids[1]]}",
        ]
        for p in (0, 1):
            lines.append(f"P{p}: base {self.base_hp[p]}  coins {self.coins[p]}  deck {len(self.deck_cards[p])}  "
                         f"hand [{', '.join(cards[c].name for c in self.hands[p])}]")
        lines.append(f"P0 back : {zone(self.backline[0])}")
        lines.append(f"front({owner}): {zone(self.frontline)}")
        lines.append(f"P1 back : {zone(self.backline[1])}")
        lines.append("legend: ^fast ~ranged D=defense A=armor *=deployed this round m=moved a=attacked")
        return "\n".join(lines)
