"""Snake Game Controller."""
import sys

import pygame

from noodle import Model, View
from noodle.model import Action, Direction


class Controller:
    """Controller for the Snake game, managing input and game flow."""

    def __init__(self, model: Model, view: View, fps: int):
        self.model = model
        self.view = view
        self.fps = fps
        self.clock = pygame.time.Clock()

    def play(self):
        """Main game loop, handles user input and updates the game state."""
        while True:
            action = self.get_user_action()
            score, done = self.model.play_step(action)

            self.view.render(self.model.snake, self.model.fruit, score)

            if done:
                self.model.reset()

            self.clock.tick(self.fps)

    def get_user_action(self) -> Action:
        """Handles player input (keyboard or AI)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            snake_direction = self.model.snake.direction()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return Action.from_direction(snake_direction, Direction.UP)
                elif event.key == pygame.K_RIGHT:
                    return Action.from_direction(
                        snake_direction, Direction.RIGHT
                    )
                elif event.key == pygame.K_DOWN:
                    return Action.from_direction(
                        snake_direction, Direction.DOWN
                    )
                elif event.key == pygame.K_LEFT:
                    return Action.from_direction(
                        snake_direction, Direction.LEFT
                    )

        return Action.MOVE_STRAIGHT
