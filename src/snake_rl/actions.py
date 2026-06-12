"""Action space mappings.

The simulator always consumes absolute directions. The egocentric
space exposes turn-left, straight, turn-right relative to the snake's
current direction, which removes the wasted reverse action and pairs
naturally with egocentric observations.
"""

import numpy as np

from snake_rl.config import ActionConfig
from snake_rl.env.vec_env import VecSnake


def num_actions(config: ActionConfig) -> int:
    """Return the size of the configured action space."""
    return 3 if config.space == "egocentric" else 4


def to_absolute(
    config: ActionConfig, actions: np.ndarray, env: VecSnake
) -> np.ndarray:
    """Map policy actions to absolute directions.

    Args:
        config: Action space settings.
        actions: Policy outputs, shape (num_envs,).
        env: Environment, consulted for current directions.

    Returns:
        Absolute directions, shape (num_envs,).
    """
    if config.space == "absolute":
        return actions
    return (env.direction + actions - 1) % 4
