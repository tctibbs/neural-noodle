"""Snake Game View."""

from enum import Enum

import pygame

from src.noodle.model import Fruit, Snake


class View:
    """View for the Snake game, handles rendering."""

    def __init__(self, width: int, height: int, cell_size: int):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.surface = pygame.Surface(self.screen.get_size()).convert()

    def render(self, snake: Snake, fruit: Fruit, score: int):
        """Renders the game state onto the screen."""
        self.surface.fill(Colors.BLACK.value)
        self.draw_grid()
        self.render_snake(snake)
        self.render_fruit(fruit)
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

        # Render score
        pygame.display.set_caption(f"Snake Game - Score: {score}")

    def draw_grid(self):
        """Draws the grid on the screen."""
        for y in range(0, self.height, self.cell_size):
            for x in range(0, self.width, self.cell_size):
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.surface, Colors.WHITE.value, rect, 1)

    def render_snake(self, snake: Snake):
        """Renders the snake based on its state."""
        for segment in snake.segments():
            pygame.draw.rect(
                self.surface,
                Colors.BLUE.value,
                (*segment, snake._size, snake._size),
            )

    def render_fruit(self, fruit: Fruit):
        """Renders the fruit based on its state."""
        pygame.draw.rect(
            self.surface,
            Colors.RED.value,
            (*fruit.position(), fruit._size, fruit._size),
        )


class Colors(Enum):
    """Enum class for the colors."""

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
