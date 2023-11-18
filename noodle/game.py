"""Game module."""
import random
import sys

import pygame

from .entities import Snake, Fruit

# Initialize Pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


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

    def run(self):
        """Runs the game."""
        clock = pygame.time.Clock()

        while True:
            self.handle_events()
            self.update_entities()
            self.update_screen()
            self.update_score()
            clock.tick(self.fps)

    def update_entities(self) -> None:
        """Updates the entities."""
        self.snake.move()
        if self.snake.head() == self.fruit.position:
            self.snake.eat()
            self.spawn_fruit()
            self.score += 1

        if self.snake.check_collision():
            self.reset()
        
        if (self.snake.head()[0] < 0 or self.snake.head()[0] > self.width - self.cell_size or 
            self.snake.head()[1] < 0 or self.snake.head()[1] > self.height - self.cell_size):
            self.reset()

    def spawn_fruit(self) -> None:
        """Spawns a fruit."""
        while True:
            fruit_position = (random.randint(0, self.width // self.cell_size - 1) * self.cell_size, random.randint(0, self.height // self.cell_size - 1) * self.cell_size)
            if fruit_position not in self.snake.segments:
                break
            
        self.fruit = Fruit(fruit_position, self.cell_size)

    def handle_events(self) -> None:
        """Handles events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.snake.change_direction("up")
                elif event.key == pygame.K_DOWN:
                    self.snake.change_direction("down")
                elif event.key == pygame.K_LEFT:
                    self.snake.change_direction("left")
                elif event.key == pygame.K_RIGHT:
                    self.snake.change_direction("right")

    def reset(self) -> None:
        """Resets the game."""
        self.snake = Snake(3, (self.width // 2, self.height // 2), "right", color=BLUE)
        self.fruit = Fruit((random.randint(0, self.width // self.cell_size - 1) * self.cell_size, random.randint(0, self.height // self.cell_size - 1) * self.cell_size), self.cell_size, color=RED)
        self.score = 0

    def update_screen(self) -> None:
        """Updates the screen."""
        self.surface.fill(BLACK)
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
                pygame.draw.rect(self.surface, WHITE, rect, 1)
