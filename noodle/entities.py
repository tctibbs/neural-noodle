"""Module containing the entities used in the game."""
import pygame
from abc import ABC, abstractmethod
from collections import deque


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
        starting_direction: The starting direction of the snake. Defaults to "idle".
        color: The color of the snake. Defaults to blue.
    """
    def __init__(self, length: int, starting_position: tuple[int, int], starting_direction: str = "idle", color: tuple[int, int, int] = (0, 0, 255)) -> None:
        self.length = length
        self.segments = deque([starting_position])
        self.direction = starting_direction
        self.color = color
        self.size = 25

    def head(self) -> tuple[int, int]:
        """Returns the head of the snake."""
        return self.segments[0]

    def tail(self) -> tuple[int, int]:  
        """Returns the tail of the snake."""
        return self.segments[-1]
    
    def segments(self) -> list[tuple[int, int]]:
        """Returns the segments of the snake."""
        return list(self.segments)

    def check_collision(self) -> bool:
        """Returns a boolean indicating if the snake has collided with itself."""
        return self.head() in list(self.segments)[1:]

    def change_direction(self, new_direction: str) -> None:
        assert new_direction in ["idle", "up", "down", "left", "right"], "Invalid direction."

        if new_direction == "up" and self.direction != "down":
            self.direction = new_direction
        elif new_direction == "down" and self.direction != "up":
            self.direction = new_direction
        elif new_direction == "left" and self.direction != "right":
            self.direction = new_direction
        elif new_direction == "right" and self.direction != "left":
            self.direction = new_direction

    def move(self) -> None:
        """Moves the snake in the current direction."""
        x_delta = 0
        y_delta = 0

        if self.direction == "up":
            y_delta -= self.size
        elif self.direction == "down":
            y_delta += self.size
        elif self.direction == "left":
            x_delta -= self.size
        elif self.direction == "right":
            x_delta += self.size
        
        current_head = self.head()
        self.segments.appendleft((current_head[0] + x_delta, current_head[1] + y_delta))
        if len(self.segments) > self.length:
            self.segments.pop()

    def eat(self) -> None:
        """Makes the snake eat."""
        self.length += 1
    
    def render(self, surface: pygame.Surface) -> None:
        """Render the snake on the given surface."""
        for segment in self.segments:
            pygame.draw.rect(surface, self.color, (*segment, self.size, self.size))


class Fruit(Entity):
    """Fruit entity.

    Attributes:
        position: The position of the fruit.
        size: The size of the fruit.
        color: The color of the fruit. Defaults to red.
    """
    def __init__(self, position: tuple[int, int], size: int, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.position = position
        self.size = size
        self.color = color

    def position(self) -> tuple[int, int]:
        """Returns the position of the fruit."""
        return self.position

    def render(self, surface: pygame.Surface) -> None:
        """Renders the fruit on the given surface."""
        pygame.draw.rect(surface, self.color, (*self.position, self.size, self.size))
