import random
from dataclasses import dataclass

import numpy as np

from src.noodle.model.entities import Direction

from . import Fruit, Point, Snake


@dataclass
class GameState:
    """Keeps track of game metrics."""

    done: bool = False
    score: int = 0
    steps_taken: int = 0
    turns_since_ate: int = 0
    fruits_eaten: int = 0
    distance_to_fruit: float = np.inf
    moves_per_fruit: float = np.inf


class Model:
    """Manages the state and rules of the Snake Game."""

    def __init__(self, width: int, height: int, cell_size: int):
        self.width = width
        self.height = height
        self.cell_size = cell_size

        self.reset()

    def reset(self):
        """Resets the game state and metrics."""
        self.spawn_snake()
        self.spawn_fruit()
        self.state = GameState()

    def play_step(self, direction: Direction) -> GameState:
        """Updates the game state based on the player's action."""
        self.snake.set_direction(direction)
        self.snake.move()

        self._update_game_state()
        if self.snake.head() == self.fruit.position():
            self.snake.eat()
            self.spawn_fruit()

        return self.state

    def check_collision(self, position: Point) -> bool:
        """Checks if the snake has collided with itself or the walls."""
        if position in self.snake.segments()[1:]:
            return True
        if (
            position.x < 0
            or position.x >= self.width
            or position.y < 0
            or position.y >= self.height
        ):
            return True
        return False

    def spawn_snake(self):
        """Spawns a new snake in the center of the grid."""
        self._snake = Snake(
            length=3, starting_position=Point(self.width // 2, self.height // 2)
        )

    def spawn_fruit(self):
        """Spawns a fruit at a random location not occupied by the snake."""
        while True:
            fruit_position = Point(
                random.randint(0, self.width // self.cell_size - 1)
                * self.cell_size,
                random.randint(0, self.height // self.cell_size - 1)
                * self.cell_size,
            )
            if fruit_position not in self.snake.segments():
                break
        self._fruit = Fruit(fruit_position, self.cell_size)

    @property
    def snake(self) -> Snake:
        """Returns the snake object."""
        assert self._snake is not None, "Snake has not been initialized"
        return self._snake

    @property
    def fruit(self) -> Fruit:
        """Returns the fruit object."""
        assert self._fruit is not None, "Fruit has not been initialized"
        return self._fruit

    def _update_game_state(self) -> None:
        """Updates the game state."""
        self.state.steps_taken += 1

        if self.check_collision(self.snake.head()):
            self.state.done = True
        elif self.snake.head() == self.fruit.position():
            self.state.score += 1
            self.state.turns_since_ate = 0
            self.state.fruits_eaten += 1

            self.state.moves_per_fruit = (
                self.state.steps_taken / self.state.fruits_eaten
            )

        else:
            self.state.turns_since_ate += 1

        self.state.distance_to_fruit = (
            _manhattan_distance(
                self.snake.head(),
                self.fruit.position(),
            )
            // self.cell_size
        )


def _manhattan_distance(p1: Point, p2: Point) -> float:
    """Returns the Manhattan distance between two points."""
    return abs(p1.x - p2.x) + abs(p1.y - p2.y)
