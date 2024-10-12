"""Module containing the entities used in the game."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import NamedTuple

import pygame


class Colors(Enum):
    """Enum class for the colors."""
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)


class Action(Enum):
    """Enum class for the actions."""
    MOVE_STRAIGHT = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2

    @staticmethod
    def from_direction(current_direction: Direction, desired_direction: Direction) -> Action:
        """Returns the action from the given direction."""
        if desired_direction == current_direction:
            return Action.MOVE_STRAIGHT
        elif desired_direction == current_direction.move_left():
            return Action.MOVE_LEFT
        elif desired_direction == current_direction.move_right():
            return Action.MOVE_RIGHT
        else:
            # Default to moving straight if a desired action is not possible
            return Action.MOVE_STRAIGHT

class Direction(Enum):
    """Enum class for the directions."""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def move_left(self) -> Direction:
        """Returns the direction after moving left."""
        return Direction(self.value - 1 if self.value > 0 else 3)
    
    def move_right(self) -> Direction:
        """Returns the direction after moving right."""
        return Direction(self.value + 1 if self.value < 3 else 0)

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


class Entity(ABC):
    """Abstract base class for entities in the game."""
    @abstractmethod
    def render(self, surface: pygame.Surface) -> None:
        """Abstract method to render the entity on the given surface."""
        pass


class Snake(Entity):
    """Snake entity. 

    Attributes:
        length: The length of the snake.
        starting_position: The starting position of the snake.
        starting_direction: The starting direction of the snake. Defaults to right.
        color: The color of the snake. Defaults to blue.
    """
    def __init__(self, length: int, starting_position: Point, starting_direction: Direction = Direction.RIGHT, color: Colors = Colors.BLUE) -> None:
        self._length = length
        self._segments = deque([starting_position])
        self._direction = starting_direction
        self._color = color
        self._size = 25
        self._turns_since_eat = None

    def direction(self) -> Direction:
        """Returns the direction of the snake."""
        return self._direction
    
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

    def move(self, action: Action) -> None:
            """Moves the snake in the specified direction."""
            if action == Action.MOVE_STRAIGHT:
                self._direction = self._direction
            elif action == Action.MOVE_LEFT:
                self._direction = self._direction.move_left()
            elif action == Action.MOVE_RIGHT:
                self._direction = self._direction.move_right()
            
            print(f"Moving snake {self._direction}")
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
            self._segments.appendleft(Point(current_head.x + x_delta, current_head.y + y_delta))
            if len(self._segments) > self._length:
                self._segments.pop()

    def eat(self) -> None:
        """Makes the snake eat."""
        self._length += 1
        self._turns_since_eat = 0
    
    def render(self, surface: pygame.Surface) -> None:
        """Render the snake on the given surface."""
        for segment in list(self._segments)[1:]:
            pygame.draw.rect(surface, self._color.value, (*segment, self._size, self._size))
        pygame.draw.rect(surface, color=(0,255,0), rect=(*self.head(), self._size, self._size))


class Fruit(Entity):
    """Fruit entity.

    Attributes:
        position: The position of the fruit.
        size: The size of the fruit.
        color: The color of the fruit. Defaults to red.
    """
    def __init__(self, position: Point, size: int, color: Colors = Colors.RED) -> None:
        assert isinstance(position, Point), "Position must be a Point."

        self._position = position
        self._size = size
        self._color = color

    def position(self) -> Point:
        """Returns the position of the fruit."""
        return self._position

    def render(self, surface: pygame.Surface) -> None:
        """Renders the fruit on the given surface."""
        pygame.draw.rect(surface, self._color.value, (*self.position(), self._size, self._size))
