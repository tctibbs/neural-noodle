"""
Neural Noodle: Snake Game with RL Training

Run the Snake game in either:
1. Manual Mode: Play interactively (no configuration needed).
2. RL Training Mode: Train an AI agent using a YAML configuration file.

Usage:
    python main.py                          # Manual mode
    python main.py --config config.yaml     # RL training mode
"""

import argparse
import yaml
from src import noodle, neural


# Constants
GRID_SIZE = 16
WIDTH, HEIGHT = 400, 400
FPS = 15


def load_config(file_path: str) -> dict:
    """Loads the configuration from a YAML file."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snake Game with RL or Manual Mode"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file for RL training.",
    )
    args = parser.parse_args()

    if not args.config:
        print("Running in manual mode...")
        game_model = noodle.model.GameLogic(cols=GRID_SIZE, rows=GRID_SIZE)
        game_view = noodle.view.GameRenderer(
            width=WIDTH, height=HEIGHT, cols=GRID_SIZE, rows=GRID_SIZE
        )
        game_controller = noodle.controller.Controller(
            game_model, game_view, FPS
        )
        game_controller.play()
    else:
        # RL training mode
        print(f"Running in RL training mode using config: {args.config}")
        config = load_config(args.config)
        neural.train.train_snake_dqn(config)


if __name__ == "__main__":
    main()
