"""Headless, deterministic Stage 1 game engine. All rules live here; see SPEC.md."""
from __future__ import annotations

import copy
import operator
import random
from bisect import insort
from typing import NamedTuple, Optional

import numpy as np

from .actions import ActionSpace
from .cards import GameConfig, load_ruleset

DRAW = -1


class IllegalActionError(ValueError):
    pass


class UnitView(NamedTuple):
    card: int
    atk: int
    hp: int
    ready: bool


class Observation(NamedTuple):
    """Everything one player may see, from that player's point of view."""
    player: int
    is_my_turn: bool
    went_first: bool
    round: int
    my_coins: int
    opp_coins: int
    my_base_hp: int
    opp_base_hp: int
    hand: tuple
    opp_hand_size: int
    my_deck_size: int
    opp_deck_size: int
    my_backline: tuple
    opp_backline: tuple
    frontline: tuple
    front_owner: int  # +1 observer, -1 opponent, 0 empty
    done: bool
    result: int       # +1 observer won, -1 lost, 0 draw/ongoing


class Unit:
    __slots__ = ("card", "atk", "hp", "owner", "ready")

    def __init__(self, card: int, atk: int, hp: int, owner: int, ready: bool = False):
        self.card, self.atk, self.hp, self.owner, self.ready = card, atk, hp, owner, ready

    def copy(self) -> "Unit":
        cls = type(self)
        u = cls.__new__(cls)
        if cls is Unit:  # fast path
            u.card, u.atk, u.hp, u.owner, u.ready = self.card, self.atk, self.hp, self.owner, self.ready
        else:  # subclasses (later stages) may add slots
            for klass in cls.__mro__:
                for name in getattr(klass, "__slots__", ()):
                    if hasattr(self, name):
                        setattr(u, name, copy.deepcopy(getattr(self, name)))
        return u

    def view(self) -> UnitView:
        return UnitView(self.card, self.atk, self.hp, self.ready)

    def __repr__(self) -> str:
        return f"Unit(card={self.card}, {self.atk}/{self.hp}, p{self.owner}{'' if self.ready else ', exhausted'})"


# Attributes that are immutable for the lifetime of a Game and may be shared by clones.
_SHARED_ATTRS = frozenset({"config", "action_space", "num_actions", "_cost", "_atk", "_hp"})
_SCALAR_TYPES = (int, float, str, bool, type(None))


def _as_index(value, what: str) -> int:
    """Strict integer conversion: ints and numpy ints, never bools/floats/None."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{what} must be an integer, got {value!r}")
    return operator.index(value)


class Game:
    def __init__(self, config: Optional[GameConfig] = None):
        self.config = config if config is not None else load_ruleset()
        cfg = self.config
        self.action_space = ActionSpace(cfg.max_hand_size, cfg.zone_capacity)
        self.num_actions = self.action_space.n
        cards = cfg.cards.cards
        self._cost = tuple(c.cost for c in cards)
        self._atk = tuple(c.attack for c in cards)
        self._hp = tuple(c.health for c in cards)
        # Not started: queries are safe (no legal actions, no winner) until reset().
        self.done = True
        self._winner = None
        self.current = 0
        self.round = 0
        self.num_steps = 0
        self._legal = []
        self._mask = bytearray(self.num_actions)

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int) -> None:
        seed = _as_index(seed, "seed")  # ints and numpy ints; rejects None/bools/floats (determinism)
        if seed < 0:
            raise ValueError("seed must be a non-negative integer")  # Random(-s) == Random(s)
        cfg = self.config
        self.seed = seed
        self.rng = random.Random(seed)
        self.first_player = self.rng.randrange(2)
        self.decks = [list(cfg.decks[0]), list(cfg.decks[1])]
        self.rng.shuffle(self.decks[0])
        self.rng.shuffle(self.decks[1])
        self.hands = [[], []]
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
        deck, hand = self.decks[p], self.hands[p]
        for _ in range(n):
            if not deck:
                return  # no fatigue in Stage 1
            card = deck.pop()
            if len(hand) >= self.config.max_hand_size:
                self.burned[p] += 1
            else:
                insort(hand, card)  # hand kept sorted by card index (canonical order)

    def _start_turn(self, p: int) -> None:
        self.current = p
        self.coins[p] = self.config.coins_for_round(self.round)
        self._draw(p, 1)
        for u in self.backline[p]:
            u.ready = True
        if self.front_owner == p:
            for u in self.frontline:
                u.ready = True
        self._legal = self._mask = None

    # ------------------------------------------------------------------ queries
    def current_player(self) -> int:
        return self.current

    def winner(self) -> Optional[int]:
        return self._winner

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
            if len(back) < Z:
                coins, cost = self.coins[p], self._cost
                for i, card in enumerate(self.hands[p]):
                    if cost[card] <= coins:
                        legal.append(sp.PLAY0 + i)
            if fo is None or (fo == p and len(front) < Z):
                for j, u in enumerate(back):
                    if u.ready:
                        legal.append(sp.MOVE0 + j)
            if fo == p:
                ready_front = [j for j, u in enumerate(front) if u.ready]
                for j in ready_front:
                    legal.append(sp.BASE0 + j)
                n_targets = len(self.backline[o])
                for j in ready_front:
                    base = sp.FRONT0 + j * Z
                    legal.extend(range(base, base + n_targets))
            elif fo == o:
                n_targets = len(front)
                for j, u in enumerate(back):
                    if u.ready:
                        base = sp.BACK0 + j * Z
                        legal.extend(range(base, base + n_targets))
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
        if not self.is_legal(action):
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
            self.backline[p].append(Unit(card, self._atk[card], self._hp[card], p, False))
        elif action < sp.BASE0:
            u = self.backline[p].pop(action - sp.MOVE0)
            u.ready = False
            self.frontline.append(u)
            self.front_owner = p
        elif action < sp.FRONT0:
            u = self.frontline[action - sp.BASE0]
            u.ready = False
            self._damage_base(o, u.atk, attacker=p)
        elif action < sp.BACK0:
            j, k = divmod(action - sp.FRONT0, Z)
            self._combat(self.frontline[j], self.backline[o][k])
        elif action < sp.n:
            j, k = divmod(action - sp.BACK0, Z)
            self._combat(self.backline[p][j], self.frontline[k])
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
        attacker.ready = False
        target.hp -= attacker.atk  # simultaneous damage
        attacker.hp -= target.atk
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
        # Positional tuple construction: observe() is on the hot path of every agent.
        new, UV = tuple.__new__, UnitView
        return new(Observation, (
            player, (not self.done) and self.current == player, self.first_player == player, self.round,
            self.coins[player], self.coins[o], self.base_hp[player], self.base_hp[o],
            tuple(self.hands[player]), len(self.hands[o]), len(self.decks[player]), len(self.decks[o]),
            tuple([new(UV, (u.card, u.atk, u.hp, u.ready)) for u in self.backline[player]]),
            tuple([new(UV, (u.card, u.atk, u.hp, u.ready)) for u in self.backline[o]]),
            tuple([new(UV, (u.card, u.atk, u.hp, u.ready)) for u in self.frontline]),
            0 if fo is None else (1 if fo == player else -1), self.done, result,
        ))

    def clone(self) -> "Game":
        """Independent deep copy (including RNG state). Unknown mutable attributes are deep-copied,
        so state added by later stages is never silently shared between clones."""
        cls = self.__class__
        g = cls.__new__(cls)
        d = g.__dict__
        memo = {}  # shared, so references to board units elsewhere stay aliased in the clone
        units = {}
        for k in ("backline", "frontline"):
            if k in self.__dict__:
                v = self.__dict__[k]
                zones = v if k == "backline" else [v]
                copies = [[units.setdefault(id(u), u.copy()) for u in z] for z in zones]
                d[k] = copies if k == "backline" else copies[0]
        for uid, u in units.items():
            memo[uid] = u
        for k, v in self.__dict__.items():
            if k in ("_legal", "_mask", "backline", "frontline"):
                continue  # caches are recomputed lazily; zones copied above
            if k in _SHARED_ATTRS or isinstance(v, _SCALAR_TYPES):
                d[k] = v
            elif k == "rng":
                r = type(v).__new__(type(v))  # skip urandom seeding; state set below
                r.setstate(v.getstate())
                d[k] = r
            elif k in ("decks", "hands"):
                d[k] = [list(x) for x in v]
            elif k in ("burned", "base_hp", "coins"):
                d[k] = list(v)
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
        names = [c.name for c in self.config.cards.cards]

        def zone(units):
            return " ".join(f"{names[u.card]}({u.atk}/{u.hp}{'' if u.ready else '*'})" for u in units) or "-"

        owner = "-" if self.front_owner is None else f"P{self.front_owner}"
        lines = [
            f"round {self.round}  current P{self.current}  first P{self.first_player}  "
            f"done={self.done} winner={self._winner}",
        ]
        for p in (0, 1):
            lines.append(f"P{p}: base {self.base_hp[p]}  coins {self.coins[p]}  deck {len(self.decks[p])}  "
                         f"hand [{', '.join(names[c] for c in self.hands[p])}]")
        lines.append(f"P0 back : {zone(self.backline[0])}")
        lines.append(f"front({owner}): {zone(self.frontline)}")
        lines.append(f"P1 back : {zone(self.backline[1])}")
        return "\n".join(lines)
