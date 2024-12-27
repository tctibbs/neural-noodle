"""Snake Game wrapper over Gymnasium environment."""

import copy

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from src.noodle import model, view
from src.noodle.model.entities import Direction


class SnakeGameEnv(gym.Env):
    """Gymnasium environment for Snake Game."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        cols: int,
        rows: int,
        width: int,
        height: int,
        fps: int = 120,
    ) -> None:
        super(SnakeGameEnv, self).__init__()

        self.cols: int = cols
        self.rows: int = rows
        self.width: int = width
        self.height: int = height
        self.fps: int = fps

        # Action space: 0 - UP, 1 - RIGHT, 2 - DOWN, 3 - LEFT
        self.action_space: gym.Space = spaces.Discrete(4)

        # Observation space includes:
        # 1. Direction (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
        # 2. Distance to wall or body in 4 directions (up, right, down, left)
        # 3. Distance to fruit (Manhattan distance)
        self.observation_space: gym.Space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.model = model.GameLogic(self.cols, self.rows)
        self.view = view.GameRenderer(
            self.width, self.height, self.rows, self.cols
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
        # Update direction based on action (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
        if action == 0:
            direction = Direction.UP
        elif action == 1:
            direction = Direction.RIGHT
        elif action == 2:
            direction = Direction.DOWN
        elif action == 3:
            direction = Direction.LEFT

        prev_state = copy.deepcopy(self.model.state)
        curr_state = self.model.play_step(direction)
        obs = self._get_observation()

        truncated = False
        terminated = False
        # Game over if collision occurs
        if curr_state.done:
            reward = -10.0
            terminated = True
        # Game over if snake hasn't eaten for 50 turns (time truncation)
        elif curr_state.turns_since_ate >= 50:
            print("Game over: 50 turns without eating.")
            reward = -10.0
            terminated = True
            truncated = True
        # Reward for eating
        elif curr_state.fruits_eaten > prev_state.fruits_eaten:
            reward = 10.0
        else:
            reward = 1

        info = {}

        # Debugging output to track actions and game state
        print(
            f"Action: {action}, Reward: {reward}"
            + f", Turns since ate: {curr_state.turns_since_ate}"
            + f", Done: {curr_state.done}"
            + f", Fruits eaten: {curr_state.fruits_eaten}"
        )

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        """Render the game state."""
        self.view.render(self.model, self.model.state.score)

    def close(self) -> None:
        """Close the game (e.g., the Pygame window)."""
        pygame.quit()

    def _get_observation(self) -> np.ndarray:
        """Direction, distance to danger, and distance to fruit."""
        direction = (
            self.model.snake.direction().value
        )  # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
        distances_to_danger = self._get_distances_to_danger()
        distance_to_fruit = self.model.state.distance_to_fruit

        observation = np.array(
            [direction] + distances_to_danger + [distance_to_fruit],
            dtype=np.float32,
        )

        # Human-readable printout of the observation with consistent spacing
        direction_labels = ["UP", "RIGHT", "DOWN", "LEFT"]
        print(
            f"{'Direction:':<10} {direction_labels[direction]:<6}"
            f"{'Distance to danger:':<20}"
            f"{'Up':<3}{distances_to_danger[0]:<3.0f} "
            f"{'Right':<6}{distances_to_danger[1]:<3.0f} "
            f"{'Down':<5}{distances_to_danger[2]:<3.0f} "
            f"{'Left':<5}{distances_to_danger[3]:<3.0f}"
            f"{'Distance to fruit:':<18} {distance_to_fruit:.0f}"
        )

        return observation

    def _get_distances_to_danger(self) -> list[float]:
        """
        Calculate distances to the nearest wall
        or the snake's body in four directions.
        """
        head = self.model.snake.head()
        segments = self.model.snake.segments()

        # Distance to the walls in each direction
        distance_up = self.rows
        distance_down = self.rows - head.row - 1
        distance_left = head.col
        distance_right = self.cols - head.col - 1

        # Check if there's danger (snake's body) closer than the walls
        for segment in segments[1:]:  # Skip the head
            if segment.row == head.row and segment.col < head.col:
                distance_up = min(distance_up, (head.col - segment.col))
            elif segment.row == head.row and segment.col > head.col:
                distance_down = min(distance_down, (segment.col - head.col))
            elif segment.col == head.col and segment.row < head.row:
                distance_left = min(distance_left, (head.row - segment.row))
            elif segment.col == head.col and segment.row > head.row:
                distance_right = min(distance_right, (segment.row - head.row))

        return [
            float(distance_up),
            float(distance_right),
            float(distance_down),
            float(distance_left),
        ]
