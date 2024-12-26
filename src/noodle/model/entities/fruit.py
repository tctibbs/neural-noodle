"""Fruit module.

This module defines the `Fruit` entity, which represents the food that the
snake consumes in the game. The `Fruit` class manages the position and size
of the fruit on the game grid.
"""

from . import Point


class Fruit:
    """Fruit entity.

    Attributes:
        position: The position of the fruit.
        size: The size of the fruit.
    """

    def __init__(self, position: Point, size: int) -> None:
        assert isinstance(position, Point), "Position must be a Point."

        self._position = position
        self._size = size

    def position(self) -> Point:
        """Returns the position of the fruit."""
        return self._position
