"""Game module."""
import random
import sys
from enum import Enum

import pygame

from .entities import Colors, Direction, Fruit, Point, Snake, Action
from neural.state import get_state

# Initialize Pygame
pygame.init()

class GameState(Enum):
    """Enum class for the game states."""
    IDLE = 0
    PLAYING = 1

class SnakeGame():
    """The Snake Game.
    
    Attributes:
        width: The width of the game.mamba
        height: The height of the game.
        cell_size: The size of each cell.
        fps: The FPS of the game.
    """
    def __init__(self, width: int, height: int, cell_size: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.fps = fps

        self.screen = pygame.display.set_mode((self.width, self.height), 0, 32)
        self.surface = pygame.Surface(self.screen.get_size()).convert() 
        self.clock = pygame.time.Clock()

        self.state = GameState.IDLE
        self.snake = None
        self.fruit = None   
        self.score = 0
        self.turns = 0
        self.turns_since_eat = 0

        self.reset()

    def play(self):
        """Starts the game."""
        while True:
            user_action = self.get_user_action()
            if not user_action:
                continue
            self.play_step(user_action if user_action else Action.MOVE_STRAIGHT)
            get_state(self)

    def play_step(self, action: Direction) -> tuple[GameState, int]:
        """Plays a step in the game."""
        self.play_action(action)
        if self.check_collision(self.snake.head()):
            final_score = self.score
            self.reset()
            return self.state, final_score
        
        self.update_screen()
        self.update_score()
        self.turns += 1
        self.turns_since_eat += 1
        self.clock.tick(self.fps)
        return self.state, self.score

    def play_action(self, action: Direction) -> None:
        """Plays an action in the game."""
        self.state = GameState.PLAYING
        self.snake.move(action)
        
        if self.snake.head() == self.fruit.position():
            self.snake.eat()
            self.spawn_fruit()
            self.score += 1

    def check_collision(self, position: Point) -> bool:
        """Returns a boolean indicating if the given position is a collision."""
        if position in self.snake.segments()[1:]:
            return True
        
        if (position.x < 0 or position.x > self.width - self.cell_size or 
            position.y < 0 or position.y > self.height - self.cell_size):
            return True
        
        return False

    def spawn_snake(self) -> None:
        """Spawns a snake."""
        self.snake = Snake(length=3, starting_position=Point(self.width // 2, self.height // 2))

    def spawn_fruit(self) -> None:
        """Spawns a fruit."""
        while True:
            fruit_position = Point(random.randint(0, self.width // self.cell_size - 1) * self.cell_size, random.randint(0, self.height // self.cell_size - 1) * self.cell_size)
            if fruit_position not in self.snake.segments():
                break
            
        self.fruit = Fruit(fruit_position, self.cell_size)

    def get_user_action(self) -> Action:
        """Handles events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            snake_direction = self.snake.direction()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return Action.from_direction(snake_direction, Direction.UP)
                elif event.key == pygame.K_RIGHT:
                    return Action.from_direction(snake_direction, Direction.RIGHT)
                elif event.key == pygame.K_DOWN:
                    return Action.from_direction(snake_direction, Direction.DOWN)
                elif event.key == pygame.K_LEFT:
                    return Action.from_direction(snake_direction, Direction.LEFT)

    def reset(self) -> None:
        """Resets the game."""
        print("Resetting game...")
        self.spawn_snake()
        self.spawn_fruit()
        self.score = 0
        self.turns = 0
        self.turns_since_eat = 0
        self._current_action = Action.MOVE_STRAIGHT
        self.state = GameState.IDLE

    def update_screen(self) -> None:
        """Updates the screen."""
        self.surface.fill(Colors.BLACK.value)
        self.draw_grid()
        self.snake.render(self.surface)
        self.fruit.render(self.surface)
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

    def update_score(self) -> None:
        pygame.display.set_caption(f"Snake Game - Score: {self.score}")

    def draw_grid(self) -> None:
        """Draws the grid."""
        for y in range(0, self.height, self.cell_size):
            for x in range(0, self.width, self.cell_size):
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.surface, Colors.WHITE.value, rect, 1)
