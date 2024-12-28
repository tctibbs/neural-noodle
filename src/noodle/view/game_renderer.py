"""Game renderer module.

This module defines the `GameRenderer` class, which is responsible for rendering
the graphical elements of the Snake game, including the grid, snake, and fruit.
It manages the game's visual representation and updates the display to reflect
the current game state.
"""

import pygame

from src.noodle import model

from . import Colors


class GameRenderer:
    """Handles rendering for the Snake game."""

    def __init__(self, width: int, height: int, rows: int, cols: int) -> None:
        # Dimensions for the inner grid
        self.grid_rows = rows
        self.grid_cols = cols

        # Calculate the cell size based on inner grid dimensions
        self.cell_size = min(width // (cols + 2), height // (rows + 2))

        # Ensure the total screen dimensions fit the walls and grid
        self.total_width = (cols + 2) * self.cell_size
        self.total_height = (rows + 2) * self.cell_size

        # Pygame setup
        self.screen = pygame.display.set_mode(
            (self.total_width, self.total_height)
        )
        self.surface = pygame.Surface(self.screen.get_size()).convert()

    def render(self, model: model.GameLogic, score: int) -> None:
        """Renders the game state onto the screen."""
        self.surface.fill(Colors.BLACK.value)
        self.render_walls()
        self.render_grid()
        self.render_snake(model.snake)
        self.render_fruit(model.fruit)
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()
        pygame.display.set_caption(f"Snake Game - Score: {score}")

    def render_grid(self) -> None:
        """Renders the inner grid on the screen."""
        for row in range(1, self.grid_rows + 1):
            for col in range(1, self.grid_cols + 1):
                rect = self._cell_to_rect(row, col)
                pygame.draw.rect(self.surface, Colors.WHITE.value, rect, 1)

    def render_walls(self) -> None:
        """Renders walls around the game grid."""
        for row in range(self.grid_rows + 2):
            for col in range(self.grid_cols + 2):
                if (
                    row == 0
                    or row == self.grid_rows + 1
                    or col == 0
                    or col == self.grid_cols + 1
                ):
                    rect = self._cell_to_rect(row, col)
                    pygame.draw.rect(self.surface, Colors.BLACK.value, rect)

    def render_snake(self, snake: model.entities.Snake) -> None:
        """Renders the snake on the screen."""
        for segment in snake.segments():
            rect = self._cell_to_rect(segment.row + 1, segment.col + 1)
            pygame.draw.rect(self.surface, Colors.BLUE.value, rect)

    def render_fruit(self, fruit: model.entities.Fruit) -> None:
        """Renders the fruit on the screen."""
        rect = self._cell_to_rect(
            fruit.position().row + 1, fruit.position().col + 1
        )
        pygame.draw.rect(self.surface, Colors.RED.value, rect)

    def _cell_to_rect(self, row: int, col: int) -> pygame.Rect:
        """Converts a grid cell to a pixel rectangle, accounting for walls."""
        x = col * self.cell_size
        y = row * self.cell_size
        return pygame.Rect(x, y, self.cell_size, self.cell_size)
