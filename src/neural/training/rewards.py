"""
Rewards Module

This module provides the reward calculation logic for the Snake RL environment.
It reads from a configuration dictionary to assign rewards or penalties based on
events such as collisions, fruit consumption, or regular steps.
"""

from src.noodle.model.game_state import GameState


class RewardPolicy:
    """
    A class that uses a reward configuration dictionary to compute rewards
    for the Snake RL environment. Can be called directly like a function.
    """

    def __init__(self, reward_config: dict) -> None:
        """
        Initialize the RewardPolicy with a reward configuration dictionary.

        Attributes:
            reward_config: A dictionary containing reward values for collisions,
                           fruit consumption, and step usage.
        """
        self.reward_config = reward_config

    def __call__(self, prev_state: GameState, curr_state: GameState) -> float:
        """
        Calls 'calculate_reward'.

        Args:
            prev_state: The game state before the current action.
            curr_state: The game state after the current action.

        Returns:
            A float representing the computed reward value.
        """
        return self.calculate_reward(prev_state, curr_state)

    def calculate_reward(
        self, prev_state: GameState, curr_state: GameState
    ) -> float:
        """
        Determine the reward based on the difference between the previous and
        current game states, using the provided configuration.

        Args:
            prev_state: The game state before the current action.
            curr_state: The game state after the current action.

        Returns:
            A float representing the computed reward value.
        """
        # Collision Reward
        if curr_state.done:
            return self.reward_config["collision"]

        # Fruit Reward
        if curr_state.fruits_eaten > prev_state.fruits_eaten:
            return self.reward_config["fruit_eaten"]

        # Default Step Reward
        return self.reward_config["step"]
