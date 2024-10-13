"Snake Game Model."
import random

from . import Action, Fruit, Point, Snake


class Model:
    """Manages the state and rules of the Snake Game."""

    def __init__(self, width: int, height: int, cell_size: int):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.score = 0
        self._snake = None
        self._fruit = None
        self.reset()

    def reset(self):
        """Resets the game state."""
        self.spawn_snake()
        self.spawn_fruit()
        self.score = 0

    def play_step(self, action: Action):
        """Updates the game state based on the player's action."""
        self.snake.move(action)
        if self.snake.head() == self.fruit.position():
            self.snake.eat()
            self.spawn_fruit()
            self.score += 1

        done = self.check_collision(self.snake.head())
        return self.score, done

    def check_collision(self, position: Point) -> bool:
        """Checks if the snake has collided with itself or the walls."""
        if position in self.snake.segments()[1:]:
            return True
        if (
            position.x < 0
            or position.x >= self.width
            or position.y < 0
            or position.y >= self.height
        ):
            return True
        return False

    def spawn_snake(self):
        """Spawns a new snake in the center of the grid."""
        self._snake = Snake(
            length=3, starting_position=Point(self.width // 2, self.height // 2)
        )

    def spawn_fruit(self):
        """Spawns a fruit at a random location not occupied by the snake."""

        while True:
            fruit_position = Point(
                random.randint(0, self.width // self.cell_size - 1)
                * self.cell_size,
                random.randint(0, self.height // self.cell_size - 1)
                * self.cell_size,
            )
            if fruit_position not in self.snake.segments():
                break
        self._fruit = Fruit(fruit_position, self.cell_size)

    @property
    def snake(self) -> Snake:
        """Returns the snake object."""
        assert self._snake is not None, "Snake has not been initialized"
        return self._snake

    @property
    def fruit(self) -> Fruit:
        """Returns the fruit object."""
        assert self._fruit is not None, "Fruit has not been initialized"
        return self._fruit
