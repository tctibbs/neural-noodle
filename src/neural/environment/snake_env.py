"""Snake Game wrapper over Gymnasium environment."""

from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from src.neural.environment import RewardPolicy
from src.noodle import model
from src.neural.visualizations import TrainingGameRenderer
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
        # 1. Direction (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
        # 2. Distance to wall or body in 4 directions (up, right, down, left)
        # 3. Distance to fruit (Manhattan distance)
        self.observation_space: gym.Space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.model = model.GameLogic(self.cols, self.rows)
        self.view = TrainingGameRenderer(
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

        print(
            f"Reward: {reward:<2}, "
            f"Turns since ate: {curr_state.turns_since_ate}, "
            f"Done: {curr_state.done}, "
            f"Fruits eaten: {curr_state.fruits_eaten}"
        )

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human", q_values: dict | None = None) -> None:
        """Render the game state with optional Q-values."""
        self.view.render(self.model, self.model.state.score, q_values=q_values)

    def close(self) -> None:
        """Close the game (e.g., the Pygame window)."""
        pygame.quit()

    def _get_observation(self) -> np.ndarray:
        """Direction, distance to danger, and distance to fruit."""
        direction = self.model.snake.direction()
        distances_to_danger = list(self.model.state.danger_distances.values())
        distance_to_fruit = self.model.state.distance_to_fruit

        observation = np.array(
            [direction.value] + distances_to_danger + [distance_to_fruit],
            dtype=np.float32,
        )

        print(
            f"{'Direction:':<10} {direction:<6}"
            f"{'Distance to danger:':<20}"
            f"{'Up':<3}{distances_to_danger[0]:<3.0f} "
            f"{'Right':<6}{distances_to_danger[1]:<3.0f} "
            f"{'Down':<5}{distances_to_danger[2]:<3.0f} "
            f"{'Left':<5}{distances_to_danger[3]:<3.0f}"
            f"{'Distance to fruit:':<18} {distance_to_fruit:.0f}"
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
