"""Cell module.

This module defines the `Cell` class, a named tuple that represents
2D grid coordinates in the game.
"""

from __future__ import annotations

from typing import NamedTuple

from . import Direction


class Cell(NamedTuple):
    """Represents a single grid cell with row and column coordinates."""

    row: int
    col: int

    def __add__(self, other: Cell) -> Cell:
        """Adds two cells together to create a new cell."""
        if isinstance(other, Cell):
            return Cell(self.row + other.row, self.col + other.col)
        else:
            raise TypeError(f"Unsupported operand type for {type(other)}")

    def __sub__(self, other: Cell) -> Cell:
        """Subtracts one cell from another to create a new cell."""
        if isinstance(other, Cell):
            return Cell(self.row - other.row, self.col - other.col)
        else:
            raise TypeError(f"Unsupported operand type for {type(other)}")

    def move(self, direction: Direction) -> Cell:
        """Returns a new Cell moved in the specified direction."""
        if direction == Direction.UP:
            return Cell(self.row - 1, self.col)
        elif direction == Direction.RIGHT:
            return Cell(self.row, self.col + 1)
        elif direction == Direction.DOWN:
            return Cell(self.row + 1, self.col)
        elif direction == Direction.LEFT:
            return Cell(self.row, self.col - 1)
        return self

    def distance(self, other: Cell) -> int:
        """Returns the Manhattan distance to another cell."""
        if isinstance(other, Cell):
            return abs(self.row - other.row) + abs(self.col - other.col)
        else:
            raise TypeError(f"Unsupported operand type for {type(other)}")
