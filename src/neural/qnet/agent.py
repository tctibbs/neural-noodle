"""
QNet Agent Module

This module defines the Agent class, which manages interactions and training
of a QNet model in a reinforcement learning environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.optim import Adam  # type: ignore

from . import Model


@dataclass(frozen=True, kw_only=True)
class EpsilonSettings:
    """
    Encapsulates epsilon settings for exploration in Q-learning.
    """

    start: float
    end: float
    decay_steps: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """
        Validates the epsilon settings to ensure correctness.
        """
        if not (0 <= self.start <= 1):
            raise ValueError("Epsilon start must be between 0 and 1.")
        if not (0 <= self.end <= 1):
            raise ValueError("Epsilon end must be between 0 and 1.")
        if self.decay_steps <= 0:
            raise ValueError("Epsilon decay steps must be greater than 0.")


class Agent:
    """
    Agent for managing and training a QNet model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float,
        epsilon_settings: EpsilonSettings,
    ) -> None:
        """
        Initialize the Agent with a QNet model and training parameters.

        Args:
            model: The QNet model to be trained and updated.
            learning_rate: Learning rate for the optimizer.
            epsilon_settings: Epsilon settings for exploration.
        """
        self.qnet = model
        self.target_qnet = self.qnet.clone_with_same_settings(self.qnet)
        self.update_target_network()

        self.optimizer = Adam(self.qnet.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()

        self.epsilon_settings = epsilon_settings
        self.steps = 0

    @property
    def epsilon(self) -> float:
        """
        Calculate the current epsilon value based on the number of steps.
        """
        progress = min(self.steps / self.epsilon_settings.decay_steps, 1.0)
        return max(
            self.epsilon_settings.end,
            self.epsilon_settings.start
            - progress
            * (self.epsilon_settings.start - self.epsilon_settings.end),
        )

    @staticmethod
    def from_config(config: dict, input_dim: int, output_dim: int) -> Agent:
        """
        Create an Agent instance from a configuration dictionary.

        Args:
            config: Dictionary containing agent configuration.
            input_dim: Number of input features for the model.
            output_dim: Number of output actions for the model.

        Returns:
            An instance of Agent.
        """
        model = Model.from_config(config["model"], input_dim, output_dim)
        epsilon_settings = EpsilonSettings(**config["epsilon"])

        return Agent(
            model=model,
            learning_rate=config["learning_rate"],
            epsilon_settings=epsilon_settings,
        )

    def epsilon_greedy_action(
        self, state: np.ndarray, action_space: int
    ) -> int:
        """
        Decide whether to explore or exploit using epsilon-greedy policy.

        Args:
            state: Current state represented as a numpy array.
            action_space: The number of possible actions.

        Returns:
            int: The selected action.
        """
        if np.random.rand() < self.epsilon:  # Explore
            return np.random.randint(0, action_space)
        else:  # Exploit
            return self.get_best_action(state)

    def get_best_action(self, state: np.ndarray) -> int:
        """
        Select the best action (highest Q-value) for a given state.

        Args:
            state: Current state represented as a numpy array.

        Returns:
            int: The action with the highest Q-value.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.qnet(state_tensor)
        return q_values.argmax().item()

    def train_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Perform a single training step on the QNet model.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The resulting state after the action.
            done: Whether the episode has terminated.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.int64)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        done_tensor = torch.tensor(done, dtype=torch.float32)

        # Compute target Q-value
        with torch.no_grad():
            max_next_q = self.target_qnet(next_state_tensor).max().item()
            target_q = reward_tensor + (1 - done_tensor) * max_next_q

        # Compute current Q-value
        current_q = self.qnet(state_tensor)[action_tensor]

        # Compute loss
        loss = self.criterion(current_q, target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Increment steps
        self.steps += 1

    def update_target_network(self) -> None:
        """
        Synchronize the target Q-network with the current Q-network.
        """
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.target_qnet.eval()
