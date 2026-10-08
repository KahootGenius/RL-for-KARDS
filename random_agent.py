from __future__ import annotations

import random
from typing import Optional, Sequence

from ..engine import Observation


class RandomAgent:
    """Uniformly random over legal actions."""

    name = "random"

    def __init__(self, config=None, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng.seed(seed)

    def act(self, obs: Observation, legal_actions: Sequence[int]) -> int:
        return legal_actions[self.rng.randrange(len(legal_actions))]
