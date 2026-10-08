"""Agent interface: agents see only an Observation and the legal action list."""
from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

from ..engine import Observation


@runtime_checkable
class Agent(Protocol):
    name: str

    def reset(self, seed: Optional[int] = None) -> None:
        """Called at the start of every game (seeds any internal randomness)."""

    def act(self, obs: Observation, legal_actions: Sequence[int]) -> int:
        """Return one of `legal_actions`."""
