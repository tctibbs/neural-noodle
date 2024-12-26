"""Point module.

This module defines the `Point` class, a named tuple that represents
2D coordinates in the game grid. It includes support for basic arithmetic
operations such as addition and subtraction between points.
"""

from __future__ import annotations

from typing import NamedTuple


class Point(NamedTuple):
    """Named tuple class for a point."""

    x: int
    y: int

    def __add__(self, other: Point) -> Point:
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        else:
            raise TypeError(f"Unsupported operand type for {type(other)}")

    def __sub__(self, other: Point) -> Point:
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        else:
            raise TypeError(f"Unsupported operand type for {type(other)}")
