"""Game state module.

This module defines the `GameState` class, which tracks the current metrics
and status of the game. It encapsulates all data related to the game's progress,
such as the score, steps taken, and other relevant statistics.
"""

from dataclasses import dataclass, field

import numpy as np

from .entities import Direction


@dataclass()
class GameState:
    """Keeps track of game metrics."""

    done: bool = False
    score: int = 0
    steps_taken: int = 0
    turns_since_ate: int = 0
    fruits_eaten: int = 0
    distance_to_fruit: float = np.inf
    moves_per_fruit: float = np.inf
    danger_distances: dict[Direction, int] = field(
        default_factory=lambda: {
            Direction.UP: 0,
            Direction.RIGHT: 0,
            Direction.DOWN: 0,
            Direction.LEFT: 0,
        }
    )
