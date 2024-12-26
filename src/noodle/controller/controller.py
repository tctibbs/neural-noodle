"""Snake Game Controller."""

import sys
from typing import NoReturn

import pygame

from src.noodle import model, view
from src.noodle.model.entities import Direction


class Controller:
    """Controller for the Snake game, managing input and game flow."""

    def __init__(
        self, model: model.GameLogic, view: view.GameRenderer, fps: int
    ) -> None:
        self.model = model
        self.view = view
        self.fps = fps
        self.clock = pygame.time.Clock()

    def play(self) -> NoReturn:
        """Main game loop, handles user input and updates the game state."""
        while True:
            direction = self.get_user_action()
            curr_state = self.model.play_step(direction)

            self.view.render(self.model, curr_state.score)

            if curr_state.done:
                self.model.reset()

            self.clock.tick(self.fps)

    def get_user_action(self) -> Direction:
        """Handles player input (keyboard or AI)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return Direction.UP
                elif event.key == pygame.K_RIGHT:
                    return Direction.RIGHT
                elif event.key == pygame.K_DOWN:
                    return Direction.DOWN
                elif event.key == pygame.K_LEFT:
                    return Direction.LEFT

        return self.model.snake.direction()
