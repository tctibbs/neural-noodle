"""Direction module for the Snake Game.

This module defines the `Direction` enum, which represents the possible
movement directions in the game. Each direction is associated with an integer
value for easy handling and comparison.

Enum Members:

"""

from enum import Enum


class Direction(Enum):
    """Enum class for the directions.

    Attributes:
        UP: Represents upward movement.
        RIGHT: Represents movement to the right.
        DOWN: Represents downward movement.
        LEFT: Represents movement to the left.
    """

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
