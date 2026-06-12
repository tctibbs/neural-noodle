"""Play Snake interactively with the Pygame renderer.

Run with:
    uv run python main.py
"""

from src import noodle

GRID_SIZE = 16
WIDTH, HEIGHT = 400, 400
FPS = 15


def main() -> None:
    """Launch the human-playable game."""
    game_model = noodle.model.GameLogic(cols=GRID_SIZE, rows=GRID_SIZE)
    game_view = noodle.view.GameRenderer(
        width=WIDTH, height=HEIGHT, cols=GRID_SIZE, rows=GRID_SIZE
    )
    controller = noodle.controller.GameController(game_model, game_view, FPS)
    controller.play()


if __name__ == "__main__":
    main()
