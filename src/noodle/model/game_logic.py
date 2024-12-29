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
        if direction not in Direction:
            raise ValueError(f"Invalid direction: {direction}")

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

    def get_cell(self, row: int, col: int) -> Cell:
        """
        Retrieves a Cell object for the specified row and column indices.

        Args:
            row: Row index of the cell.
            col: Column index of the cell.

        Returns:
            A Cell object representing the specified grid position.
        """
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return Cell(row, col)
        raise ValueError(f"Invalid cell coordinates: ({row}, {col})")

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
        self.state.danger_distances = self._get_distances_to_danger()

    def _get_distances_to_danger(self) -> dict[Direction, int]:
        """
        Calculate distances to the nearest wall or snake's body
        in all four directions: UP, RIGHT, DOWN, LEFT.

        Returns:
            Dictionary mapping of {Direction, int}.
        """
        head = self.snake.head()
        segments = self.snake.segments()

        # Distances to walls
        distance_up = head.row  # how many rows until row=0
        distance_down = (self.rows - 1) - head.row
        distance_left = head.col  # how many columns until col=0
        distance_right = (self.cols - 1) - head.col

        # Check for closer snake-body segments
        for segment in segments[1:]:
            if segment.col == head.col and segment.row < head.row:
                distance_up = min(distance_up, head.row - segment.row)
            elif segment.col == head.col and segment.row > head.row:
                distance_down = min(distance_down, segment.row - head.row)
            elif segment.row == head.row and segment.col < head.col:
                distance_left = min(distance_left, head.col - segment.col)
            elif segment.row == head.row and segment.col > head.col:
                distance_right = min(distance_right, segment.col - head.col)

        return {
            Direction.UP: int(distance_up),
            Direction.RIGHT: int(distance_right),
            Direction.DOWN: int(distance_down),
            Direction.LEFT: int(distance_left),
        }
