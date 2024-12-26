"""Snake Game Model Package.

This package contains the core logic and components of the Snake game,
including the game logic and entities.
"""

from . import entities
from .game_state import GameState
from .game_logic import GameLogic

__all__ = ["entities", "GameState", "GameLogic"]
