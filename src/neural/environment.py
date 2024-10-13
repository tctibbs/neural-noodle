"""Snake Game wrapper over Gymnasium environment."""
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from src.noodle import Controller, Model, View
from src.noodle.model import Action, Point


class SnakeGameEnv(gym.Env):
    """Gymnasium environment for Snake Game."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        width: int = 400,
        height: int = 400,
        cell_size: int = 25,
        fps: int = 120,
    ) -> None:
        super(SnakeGameEnv, self).__init__()

        self.width: int = width
        self.height: int = height
        self.cell_size: int = cell_size
        self.fps: int = fps

        self.model: Model = Model(self.width, self.height, self.cell_size)
        self.view: View = View(self.width, self.height, self.cell_size)
        self.action_space: gym.Space = spaces.Discrete(3)

        # Observation space includes:
        # 1. Direction (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
        # 2. Distance to wall or body in 4 directions (up, right, down, left)
        # 3. Distance to fruit (Manhattan distance)
        self.observation_space: gym.Space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.controller: Controller = Controller(
            self.model, self.view, self.fps
        )

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)

        self.model.reset()

        obs = self._get_observation()
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply action, update the game state,
        and return the necessary Gym output.
        """
        # Convert gym action (0, 1, 2) to game Action
        if action == 0:
            action_enum = Action.MOVE_LEFT
        elif action == 1:
            action_enum = Action.MOVE_RIGHT
        else:
            action_enum = Action.MOVE_STRAIGHT

        score, done = self.model.play_step(action_enum)
        obs = self._get_observation()
        truncated = False

        # Game over if collision occurs
        if done:
            reward = -10
            terminated = True
        # Game over if snake hasn't eaten for 50 turns (time truncation)
        elif self.model.metrics.turns_since_ate >= 50:
            print("Game over: 50 turns without eating.")
            reward = -10
            terminated = True
            truncated = True
        # Reward for eating
        elif self.model.snake.head() == self.model.fruit.position():
            reward = 10
            terminated = False
        # Penalty for each step (to encourage quicker gameplay)
        else:
            reward = -1
            terminated = False

        info = {}

        # Debugging output to track actions and game state
        print(
            f"Action: {action}, Reward: {reward}"
            + f", Turns since ate: {self.model.metrics.turns_since_ate}"
            + f", Done: {done}"
        )

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        """Render the game state."""
        self.view.render(
            self.model.snake, self.model.fruit, self.model.metrics.score
        )

    def close(self) -> None:
        """Close the game (e.g., the Pygame window)."""
        pygame.quit()

    def _get_observation(self) -> np.ndarray:
        """Direction, distance to danger, and distance to fruit."""
        direction = (
            self.model.snake.direction().value
        )  # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
        distances_to_danger = self._get_distances_to_danger()
        distance_to_fruit = self._manhattan_distance(
            self.model.snake.head(), self.model.fruit.position()
        )

        # Return as a 1D array (6 values)
        return np.array(
            [direction] + distances_to_danger + [distance_to_fruit],
            dtype=np.float32,
        )

    def _get_distances_to_danger(self) -> list[float]:
        """
        Calculate distances to the nearest wall
        or the snake's body in four directions.
        """
        head = self.model.snake.head()
        segments = self.model.snake.segments()

        # Distance to the walls in each direction
        distance_up = head.y // self.cell_size
        distance_down = (self.height - head.y) // self.cell_size - 1
        distance_left = head.x // self.cell_size
        distance_right = (self.width - head.x) // self.cell_size - 1

        # Check if there's danger (snake's body) closer than the walls
        for segment in segments[1:]:  # Skip the head
            if segment.x == head.x and segment.y < head.y:
                distance_up = min(
                    distance_up, (head.y - segment.y) // self.cell_size
                )
            elif segment.x == head.x and segment.y > head.y:
                distance_down = min(
                    distance_down, (segment.y - head.y) // self.cell_size
                )
            elif segment.y == head.y and segment.x < head.x:
                distance_left = min(
                    distance_left, (head.x - segment.x) // self.cell_size
                )
            elif segment.y == head.y and segment.x > head.x:
                distance_right = min(
                    distance_right, (segment.x - head.x) // self.cell_size
                )

        return [
            float(distance_up),
            float(distance_right),
            float(distance_down),
            float(distance_left),
        ]

    def _manhattan_distance(self, p1: Point, p2: Point) -> float:
        """Calculate the Manhattan distance between two points."""
        return abs(p1.x - p2.x) + abs(p1.y - p2.y)
