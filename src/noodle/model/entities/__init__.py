"""Entities package for the Snake Game.

This package contains the core entities used in the game, including:
    - Point: Represents 2D positions on the game grid.
    - Direction: Enum for movement directions.
    - Snake: Represents the player's snake.
    - Fruit: Represents the food consumed by the snake.
"""

from .point import Point
from .direction import Direction
from .snake import Snake
from .fruit import Fruit

__all__ = ["Point", "Direction", "Snake", "Fruit"]
