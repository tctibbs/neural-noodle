"""Game renderer module.

This module defines the `GameRenderer` class, which is responsible for rendering
the graphical elements of the Snake game, including the grid, snake, and fruit.
It manages the game's visual representation and updates the display to reflect
the current game state.
"""

import pygame

from src.noodle.model.entities import Fruit, Snake

from . import Colors


class GameRenderer:
    """Handles rendering for the Snake game."""

    def __init__(self, width: int, height: int, cell_size: int) -> None:
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.surface = pygame.Surface(self.screen.get_size()).convert()

    def render(self, snake: Snake, fruit: Fruit, score: int) -> None:
        """Renders the game state onto the screen."""
        self.surface.fill(Colors.BLACK.value)
        self.draw_grid()
        self.render_snake(snake)
        self.render_fruit(fruit)
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()
        pygame.display.set_caption(f"Snake Game - Score: {score}")

    def draw_grid(self) -> None:
        """Draws the grid on the screen."""
        for y in range(0, self.height, self.cell_size):
            for x in range(0, self.width, self.cell_size):
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.surface, Colors.WHITE.value, rect, 1)

    def render_snake(self, snake: Snake) -> None:
        """Renders the snake based on its state."""
        for segment in snake.segments():
            pygame.draw.rect(
                self.surface,
                Colors.BLUE.value,
                (*segment, snake._size, snake._size),
            )

    def render_fruit(self, fruit: Fruit) -> None:
        """Renders the fruit based on its state."""
        pygame.draw.rect(
            self.surface,
            Colors.RED.value,
            (*fruit.position(), fruit._size, fruit._size),
        )
