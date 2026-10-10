"""Agent interface (SPEC.md section 9).

Every agent has `name`, `reset(seed)` and `act(obs, legal_actions) -> int`: it sees only its own
Observation and the legal action list. An agent that needs to simulate (e.g. `LookaheadAgent`) also
sets `needs_game = True` and implements `act_game(game, player) -> int`. Such an agent may only look at
the real game through what `player` is entitled to see: `game.observe(player)`, the legal actions on
its own decision, and samples from `game.determinize(player, rng)`. It never reads the real game's
hidden state and never mutates the game it is given.

Callers ask for a move with `choose_action(agent, game)`, which dispatches on `needs_game`.
"""
from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

from ..engine import Game, Observation


@runtime_checkable
class Agent(Protocol):
    name: str

    def reset(self, seed: Optional[int] = None) -> None:
        """Called at the start of every game (seeds any internal randomness)."""

    def act(self, obs: Observation, legal_actions: Sequence[int]) -> int:
        """Return one of `legal_actions`."""


@runtime_checkable
class GameAgent(Agent, Protocol):
    """An agent that simulates: `needs_game = True` and `act_game(game, player)` (optional extension)."""
    needs_game: bool

    def act_game(self, game: Game, player: int) -> int:
        """Return one of `game.legal_actions()` for `player` (the player to act). Must not mutate `game`."""


def choose_action(agent: Agent, game: Game) -> int:
    """The action `agent` takes in `game` for the player to act.

    `needs_game` agents get the game itself (`act_game(game, player)`); every other agent gets
    `act(game.observe(player), game.legal_actions())`.
    """
    if game.done:
        raise ValueError("the game is over: there is no action to choose")
    p = game.current_player()
    if getattr(agent, "needs_game", False):
        return agent.act_game(game, p)
    return agent.act(game.observe(p), game.legal_actions())
