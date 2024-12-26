"""Snake module.

This module defines the `Snake` entity, which represents the player's snake
in the game. The class manages the snake's movement, direction, and state,
such as its size and segments.
"""

from __future__ import annotations

from collections import deque

from . import Cell, Direction


class Snake:
    """Snake entity.

    Attributes:
        length: The length of the snake.
        starting_position: The starting position of the snake.
        starting_direction: The starting direction of the snake.
    """

    def __init__(
        self,
        length: int,
        starting_position: Cell,
        starting_direction: Direction = Direction.RIGHT,
    ) -> None:
        self._length = length
        self._segments = deque([starting_position])
        self._direction = starting_direction
        self._size = 25
        self._turns_since_eat = 0

    def direction(self) -> Direction:
        """Returns the direction of the snake."""
        return self._direction

    def set_direction(self, direction: Direction) -> None:
        """Sets the new direction of the snake, if possible."""
        if (
            (self._direction == Direction.UP and direction != Direction.DOWN)
            or (self._direction == Direction.DOWN and direction != Direction.UP)
            or (
                self._direction == Direction.LEFT
                and direction != Direction.RIGHT
            )
            or (
                self._direction == Direction.RIGHT
                and direction != Direction.LEFT
            )
        ):
            self._direction = direction

    def head(self) -> Cell:
        """Returns the head of the snake."""
        return self._segments[0]

    def tail(self) -> Cell:
        """Returns the tail of the snake."""
        return self._segments[-1]

    def segments(self) -> list[Cell]:
        """Returns the segments of the snake."""
        return list(self._segments)

    def size(self) -> int:
        """Returns the size of the snake."""
        return self._size

    def last_ate(self) -> int:
        """Returns the number of turns since the snake last ate."""
        return self._turns_since_eat

    def move(self) -> None:
        """Moves the snake based on its current direction."""
        new_head = self.head().move(self._direction)
        self._segments.appendleft(new_head)

        if len(self._segments) > self._length:
            self._segments.pop()

    def eat(self) -> None:
        """Increases the snake's length after eating."""
        self._length += 1
        self._turns_since_eat = 0
