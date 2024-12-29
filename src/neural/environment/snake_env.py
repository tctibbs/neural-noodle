"""Snake Game wrapper over Gymnasium environment."""

from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from src.neural.environment import RewardPolicy
from src.noodle import model
from src.neural.visualizations import TrainingRenderer
from src.noodle.model.entities import Direction
from src.noodle.model.game_state import GameState


class SnakeGameEnv(gym.Env):
    """Gymnasium environment for Snake Game."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        cols: int,
        rows: int,
        width: int,
        height: int,
        reward_policy: RewardPolicy,
    ) -> None:
        super().__init__()
        self.reward_policy = reward_policy

        self.cols: int = cols
        self.rows: int = rows
        self.width: int = width
        self.height: int = height

        # Action space: 0 - UP, 1 - RIGHT, 2 - DOWN, 3 - LEFT
        self.action_space = spaces.Discrete(len(Direction))

        # Observation space includes:
        #   4 one-hot directions
        #   4 normalized dangers
        #   1 normalized fruit distance
        self.observation_space: gym.Space = spaces.Box(
            low=0.0, high=1.0, shape=(9,), dtype=np.float32
        )

        self.model = model.GameLogic(self.cols, self.rows)
        self.view = TrainingRenderer(
            self.width, self.height, self.rows, self.cols
        )

    @staticmethod
    def from_config(env_config: dict) -> SnakeGameEnv:
        """Returns the Snake game environment."""
        reward_policy = RewardPolicy(env_config["reward_policy"])

        return SnakeGameEnv(
            cols=env_config["grid_size"],
            rows=env_config["grid_size"],
            width=env_config["width"],
            height=env_config["height"],
            reward_policy=reward_policy,
        )

    @property
    def input_size(self) -> int:
        """Returns the input size for the neural network."""
        assert isinstance(
            self.observation_space, gym.spaces.Box
        ), "Observation space must be of type Box."

        return int(self.observation_space.shape[0])

    @property
    def output_size(self) -> int:
        """Returns the output size for the neural network."""
        assert isinstance(
            self.action_space, gym.spaces.Discrete
        ), "Action space must be of type Discrete."
        return int(self.action_space.n)

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
        prev_state = copy.deepcopy(self.model.state)
        curr_state = self.model.play_step(Direction(action))
        obs = self._get_observation()

        terminated, truncated = self._get_termination_flags(curr_state)
        reward = self.reward_policy(prev_state, curr_state)
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human", q_values: dict | None = None) -> None:
        """Render the game state with optional Q-values."""
        self.view.render(self.model, self.model.state.score, q_values=q_values)

    def close(self) -> None:
        """Close the game (e.g., the Pygame window)."""
        pygame.quit()

    def _get_observation(self) -> np.ndarray:
        """Direction (one-hot), distance to danger, and distance to fruit (normalized)."""
        direction = self.model.snake.direction()
        distances_to_danger = list(self.model.state.danger_distances.values())
        distance_to_fruit = self.model.state.distance_to_fruit

        # One-hot encode direction
        one_hot_direction = [
            1.0 if direction == dir_enum else 0.0 for dir_enum in Direction
        ]

        # Normalize distances to danger
        max_distance = max(self.rows, self.cols)
        normalized_danger = [
            min(d / max_distance, 1.0) for d in distances_to_danger
        ]

        # Normalize distance to fruit
        max_manhattan_distance = self.rows + self.cols - 2
        normalized_fruit = min(distance_to_fruit / max_manhattan_distance, 1.0)

        # Combine one-hot direction and normalized features
        observation = np.array(
            one_hot_direction + normalized_danger + [normalized_fruit],
            dtype=np.float32,
        )

        # Debug
        print(
            f"Direction: {one_hot_direction}, "
            f"Normalized Danger: Up: {normalized_danger[0]:.2f}, Right: {normalized_danger[1]:.2f}, "
            f"Down: {normalized_danger[2]:.2f}, Left: {normalized_danger[3]:.2f}, "
            f"Normalized Fruit Distance: {normalized_fruit:.2f}"
        )

        return observation

    def _get_termination_flags(
        self, curr_state: GameState
    ) -> tuple[bool, bool]:
        """Returns if the game state indicates termination or truncation."""
        terminated = False
        truncated = False

        if curr_state.done:
            terminated = True

        return terminated, truncated
