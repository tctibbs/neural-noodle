"""View package for the Snake Game.

This package contains modules for handling the graphical representation
of the game, including rendering the grid, snake, fruit.
"""

from .colors import Colors
from .game_renderer import GameRenderer

__all__ = ["Colors", "GameRenderer"]
