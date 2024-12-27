"""
Environment package.

This package provides the core components needed to define and customize 
the Snake game environment for use in reinforcement learning. 
"""

from .reward_policy import RewardPolicy
from .snake_env import SnakeGameEnv


__all__ = ["SnakeGameEnv", "RewardPolicy"]
