"""Entities package for the Snake Game.

This package contains the core entities used in the game, including:
    - Cell: Represents 2D positions on the game grid.
    - Direction: Enum for movement directions.
    - Snake: Represents the player's snake.
    - Fruit: Represents the food consumed by the snake.
"""

from .direction import Direction
from .cell import Cell
from .snake import Snake
from .fruit import Fruit

__all__ = ["Direction", "Cell", "Snake", "Fruit"]
