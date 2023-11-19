"""Game module."""
import random
import sys

import pygame

from .entities import Direction, Fruit, Point, Snake, Colors


# Initialize Pygame
pygame.init()

class SnakeGame():
    """The Snake Game.
    
    Attributes:
        width: The width of the game.
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

        self.reset()

    def play(self):
        """Starts the game."""
        clock = pygame.time.Clock()
        current_action = Direction.RIGHT
        
        while True:
            user_action = self.get_action()
            current_action = user_action if user_action else current_action
            self.play_action(current_action)

            self.update_screen()
            self.update_score()
            clock.tick(self.fps)

    def play_step(self, action: Direction) -> tuple[float, bool, int]:
        """Plays a step in the game."""
        self.play_action(action)
        self.update_screen()
        self.update_score()

    def play_action(self, action: Direction) -> None:
        """Plays an action in the game."""
        self.snake.move(action)
        if self.snake.head() == self.fruit.position():
            self.snake.eat()
            self.spawn_fruit()
            self.score += 1

        if self.snake.check_collision():
            self.reset()
        
        if (self.snake.head()[0] < 0 or self.snake.head()[0] > self.width - self.cell_size or 
            self.snake.head()[1] < 0 or self.snake.head()[1] > self.height - self.cell_size):
            self.reset()

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

    def get_action(self) -> None:
        """Handles events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return Direction.UP
                if event.key == pygame.K_DOWN:
                    return Direction.DOWN
                if event.key == pygame.K_LEFT:
                    return Direction.LEFT
                if event.key == pygame.K_RIGHT:
                    return Direction.RIGHT

    def reset(self) -> None:
        """Resets the game."""
        self.spawn_snake()
        self.spawn_fruit()
        self.score = 0

    def update_screen(self) -> None:
        """Updates the screen."""
        self.surface.fill(Colors.BLACK.value)
        self.draw_grid()
        self.snake.render(self.surface)
        self.fruit.render(self.surface)
        self.screen.blit(self.surface, (0, 0))
        pygame.display.update()

    def update_score(self) -> None:
        pygame.display.set_caption(f"Snake Game - Score: {self.score}")

    def draw_grid(self) -> None:
        """Draws the grid."""
        for y in range(0, self.height, self.cell_size):
            for x in range(0, self.width, self.cell_size):
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.surface, Colors.WHITE.value, rect, 1)
