import pygame

from src.noodle import model
from src.noodle.view import Colors, GameRenderer


class TrainingGameRenderer(GameRenderer):
    """Extends GameRenderer to include training-specific visualizations."""

    def __init__(self, width: int, height: int, rows: int, cols: int) -> None:
        super().__init__(width, height, rows, cols)
        pygame.font.init()
        self.font = pygame.font.Font(None, 20)

    def render(
        self,
        model: model.GameLogic,
        score: int,
        q_values: dict | None = None,
        selected_action: int | None = None,
    ) -> None:
        """
        Renders the game state, including Q-values and other neural visualizations.

        Args:
            model: The game logic model.
            score: Current score of the game.
            q_values: A dictionary mapping grid cells to Q-values for each action.
            selected_action: The action chosen by the neural network for the current state.
        """
        super().render(model, score)

        # Render neural-specific visualizations
        if q_values is not None:
            self.render_q_values(q_values, selected_action)

    def render_q_values(
        self, q_values: dict, selected_action: int | None = None
    ) -> None:
        """
        Renders Q-net scores as percentages for each possible action on the grid.

        Args:
            q_values: A dictionary mapping grid cells to Q-values.
            selected_action: The action chosen by the neural network (optional).
        """
        for cell, value in q_values.items():
            rect = self._cell_to_rect(cell.row + 1, cell.col + 1)
            x, y = rect.center

            color = (
                Colors.YELLOW.value if selected_action else Colors.WHITE.value
            )
            text = self.font.render(f"{value:.2f}", True, color)
            self.surface.blit(text, (x - 10, y - 5))

        # Blit the updated surface to the screen
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()
