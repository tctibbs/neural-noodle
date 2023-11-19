"""Module containing the entities used in the game."""
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


class Direction(Enum):
    """Enum class for the directions."""
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Point(NamedTuple):
    """Named tuple class for a point."""
    x: int
    y: int


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
        color: The color of the snake. Defaults to blue.
    """
    def __init__(self, length: int, starting_position: Point, color: Colors = Colors.BLUE) -> None:
        self._length = length
        self._segments = deque([starting_position])
        self._color = color
        self._size = 25

    def head(self) -> Point:
        """Returns the head of the snake."""
        return self._segments[0]

    def tail(self) -> Point:  
        """Returns the tail of the snake."""
        return self._segments[-1]
    
    def segments(self) -> list[Point]:
        """Returns the segments of the snake."""
        return list(self._segments)

    def check_collision(self) -> bool:
        """Returns a boolean indicating if the snake has collided with itself."""
        return self.head() in list(self._segments)[1:]

    def move(self, direction: Direction) -> None:
        """Moves the snake in the current direction."""
        x_delta = 0
        y_delta = 0

        if direction == Direction.UP:
            y_delta -= self._size
        elif direction == Direction.DOWN:
            y_delta += self._size
        elif direction == Direction.LEFT:
            x_delta -= self._size
        elif direction == Direction.RIGHT:
            x_delta += self._size
        
        current_head = self.head()
        self._segments.appendleft(Point(current_head.x + x_delta, current_head.y + y_delta))
        if len(self._segments) > self._length:
            self._segments.pop()

    def eat(self) -> None:
        """Makes the snake eat."""
        self._length += 1
    
    def render(self, surface: pygame.Surface) -> None:
        """Render the snake on the given surface."""
        for segment in self._segments:
            pygame.draw.rect(surface, self._color.value, (*segment, self._size, self._size))


class Fruit(Entity):
    """Fruit entity.

    Attributes:
        position: The position of the fruit.
        size: The size of the fruit.
        color: The color of the fruit. Defaults to red.
    """
    def __init__(self, position: Point, size: int, color: Colors = Colors.RED) -> None:
        self._position = position
        self._size = size
        self._color = color

    def position(self) -> Point:
        """Returns the position of the fruit."""
        return self._position

    def render(self, surface: pygame.Surface) -> None:
        """Renders the fruit on the given surface."""
        pygame.draw.rect(surface, self._color.value, (*self.position(), self._size, self._size))
