"""Fruit module.

This module defines the `Fruit` entity, which represents the food that the
snake consumes in the game. The `Fruit` class manages the position and size
of the fruit on the game grid.
"""

from . import Cell


class Fruit:
    """Fruit entity.

    Attributes:
        position: The position of the fruit.
    """

    def __init__(self, position: Cell) -> None:
        assert isinstance(position, Cell), "Position must be of typee Cell."

        self._position = position

    def position(self) -> Cell:
        """Returns the position of the fruit."""
        return self._position
