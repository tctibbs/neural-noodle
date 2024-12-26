"""Noodle package.

This package contains the implementation of the Snake Game, including:
    - `model`: Handles the core game logic, state management, and entities.
    - `view`: Manages graphical rendering and the visual representation of the game.
    - `controller`: Coordinates game flow and user interactions.
"""

from . import model, view, controller

__all__ = ["model", "view", "controller"]
