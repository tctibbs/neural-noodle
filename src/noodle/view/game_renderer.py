"""Game renderer module.

This module defines the `GameRenderer` class, which is responsible for rendering
the graphical elements of the Snake game.
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

        # Metrics box height
        self.metrics_box_height = self.cell_size * 2

        # Pygame setup
        self.screen = pygame.display.set_mode(
            (self.total_width, self.total_height + self.metrics_box_height)
        )
        self.surface = pygame.Surface(self.screen.get_size()).convert()

        # Font setup
        pygame.font.init()
        self.font = pygame.font.Font(None, self.cell_size // 2)

    def render(self, model: model.GameLogic, score: int) -> None:
        """Renders the game state onto the screen."""
        self.surface.fill(Colors.GRAY.value)
        self.render_walls()
        self.render_grid()
        self.render_snake(model.snake)
        self.render_fruit(model.fruit)
        self.render_metrics(model.state)
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

    def render_metrics(self, game_state: model.GameState) -> None:
        """Renders game metrics at the bottom of the screen."""
        metrics = {
            "Fruits Eaten": game_state.fruits_eaten,
            "Turns Since Ate": game_state.turns_since_ate,
            "Distance to Fruit": f"{game_state.distance_to_fruit:.2f}",
            "Steps Taken": game_state.steps_taken,
        }

        # Draw the metrics box
        rect = pygame.Rect(
            0,
            self.total_height,
            self.total_width,
            self.metrics_box_height,
        )
        pygame.draw.rect(self.surface, Colors.BLACK.value, rect)  # Background
        pygame.draw.rect(self.surface, Colors.WHITE.value, rect, 2)  # Border

        # Add a title for the metrics section
        title = self.font.render("Game Metrics", True, Colors.YELLOW.value)
        title_x = self.total_width // 2 - title.get_width() // 2
        title_y = self.total_height + 5
        self.surface.blit(title, (title_x, title_y))

        # Calculate column layout
        num_columns = 2
        column_width = self.total_width // num_columns
        x_offsets = [column_width * i + 10 for i in range(num_columns)]
        y_start = title_y + title.get_height() + 10  # Start below the title
        line_spacing = self.cell_size // 2

        # Render metrics in columns
        metric_items = list(metrics.items())
        for i, (key, value) in enumerate(metric_items):
            column = i % num_columns
            row = i // num_columns
            x = x_offsets[column]
            y = y_start + row * line_spacing

            # Render key-value pairs
            text = self.font.render(f"{key}: {value}", True, Colors.WHITE.value)
            self.surface.blit(text, (x, y))

        # Add column separators below the title
        separator_top = title_y + title.get_height() + 5
        for x in x_offsets[1:]:
            pygame.draw.line(
                self.surface,
                Colors.WHITE.value,
                (x - 5, separator_top),
                (
                    x - 5,
                    self.total_height + self.metrics_box_height,
                ),
                1,
            )

    def _cell_to_rect(self, row: int, col: int) -> pygame.Rect:
        """Converts a grid cell to a pixel rectangle, accounting for walls."""
        x = col * self.cell_size
        y = row * self.cell_size
        return pygame.Rect(x, y, self.cell_size, self.cell_size)
