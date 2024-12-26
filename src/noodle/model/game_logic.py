"""Game logic module.

This module defines the `GameLogic` class, which manages the core state and
rules of the game. It handles interactions between the snake, fruit, and game
grid, updates the game state, and enforces game rules such as collisions and
scoring.
"""

import random

from .entities import Cell, Direction, Fruit, Snake
from .game_state import GameState


class GameLogic:
    """Manages the state and rules of the Snake Game."""

    def __init__(self, cols: int, rows: int) -> None:
        self.cols = cols
        self.rows = rows

        self.reset()

    def reset(self) -> None:
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

    def check_collision(self, position: Cell) -> bool:
        """Checks if the snake has collided with itself or the walls."""
        if position in self.snake.segments()[1:]:
            return True
        if (
            position.row < 0
            or position.row >= self.rows
            or position.col < 0
            or position.col >= self.cols
        ):
            return True
        return False

    def spawn_snake(self) -> None:
        """Spawns a new snake in the center of the grid."""
        center_cell = Cell(self.rows // 2, self.cols // 2)
        self._snake = Snake(length=3, starting_position=center_cell)

    def spawn_fruit(self) -> None:
        """Spawns a fruit at a random location not occupied by the snake."""
        while True:
            fruit_position = Cell(
                row=random.randint(0, self.rows - 1),
                col=random.randint(0, self.cols - 1),
            )
            if fruit_position not in self.snake.segments():
                break
        self._fruit = Fruit(fruit_position)

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

        self.state.distance_to_fruit = self.snake.head().distance(
            self.fruit.position()
        )
