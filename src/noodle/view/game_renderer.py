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
        self.width = width
        self.height = height
        self.cell_size = self._calculate_cell_size(rows, cols)
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.surface = pygame.Surface(self.screen.get_size()).convert()

    def render(self, model: model.GameLogic, score: int) -> None:
        """Renders the game state onto the screen."""
        self.surface.fill(Colors.BLACK.value)
        self.render_grid()
        self.render_snake(model.snake)
        self.render_fruit(model.fruit)
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()
        pygame.display.set_caption(f"Snake Game - Score: {score}")

    def render_grid(self) -> None:
        """Renders the grid on the screen."""
        for row in range(self.height // self.cell_size):
            for col in range(self.width // self.cell_size):
                rect = self._cell_to_rect(model.entities.Cell(row, col))
                pygame.draw.rect(self.surface, Colors.WHITE.value, rect, 1)

    def render_snake(self, snake: model.entities.Snake) -> None:
        """Renders the snake on the screen."""
        for segment in snake.segments():
            rect = self._cell_to_rect(segment)
            pygame.draw.rect(self.surface, Colors.BLUE.value, rect)

    def render_fruit(self, fruit: model.entities.Fruit) -> None:
        """Renders the fruit on the screen."""
        rect = self._cell_to_rect(fruit.position())
        pygame.draw.rect(self.surface, Colors.RED.value, rect)

    def _calculate_cell_size(self, rows: int, cols: int) -> int:
        """Returns the cell size based on grid dimensions."""
        cell_width = self.width // cols
        cell_height = self.height // rows
        return min(cell_width, cell_height)

    def _cell_to_rect(self, cell: model.entities.Cell) -> pygame.Rect:
        """Converts a grid cell to a pixel rectangle."""
        x = cell.col * self.cell_size
        y = cell.row * self.cell_size
        return pygame.Rect(x, y, self.cell_size, self.cell_size)
