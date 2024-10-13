"""Snake Game Entities."""
from __future__ import annotations

from collections import deque
from enum import Enum
from typing import NamedTuple


class Point(NamedTuple):
    """Named tuple class for a point."""

    x: int
    y: int

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        else:
            raise TypeError(f"Unsupported operand type for {type(other)}")

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        else:
            raise TypeError(f"Unsupported operand type for {type(other)}")


class Direction(Enum):
    """Enum class for the directions."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


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
        starting_position: Point,
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

    def head(self) -> Point:
        """Returns the head of the snake."""
        return self._segments[0]

    def tail(self) -> Point:
        """Returns the tail of the snake."""
        return self._segments[-1]

    def segments(self) -> list[Point]:
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
        x_delta = 0
        y_delta = 0
        if self._direction == Direction.UP:
            y_delta = -self._size
        elif self._direction == Direction.RIGHT:
            x_delta = self._size
        elif self._direction == Direction.DOWN:
            y_delta = self._size
        elif self._direction == Direction.LEFT:
            x_delta = -self._size

        current_head = self.head()
        new_head = Point(current_head.x + x_delta, current_head.y + y_delta)

        self._segments.appendleft(new_head)
        if len(self._segments) > self._length:
            self._segments.pop()

    def eat(self) -> None:
        """Increases the snake's length after eating."""
        self._length += 1
        self._turns_since_eat = 0


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
