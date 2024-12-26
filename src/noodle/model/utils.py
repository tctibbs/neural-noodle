"""Utilities module.

This module provides helper functions used across the game.
"""

from .entities import Point


def _manhattan_distance(p1: Point, p2: Point) -> float:
    """Returns the Manhattan distance between two points."""
    return abs(p1.x - p2.x) + abs(p1.y - p2.y)
