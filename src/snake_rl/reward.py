"""Reward computation, kept outside the simulator.

Rewards are a pure function of step events and (optionally) a
potential-based shaping term over the head-to-fruit Manhattan
distance. Keeping this separate from the environment means reward
ablations can never change the game rules.
"""

import numpy as np

from snake_rl.config import RewardConfig
from snake_rl.env.vec_env import StepResult, VecSnake


def potential(env: VecSnake) -> np.ndarray:
    """Compute the shaping potential for every environment.

    The potential is the negative Manhattan distance from head to
    fruit, normalized by board half-perimeter, so it lies in (-1, 0].

    Args:
        env: The environment to read.

    Returns:
        Array of shape (num_envs,).
    """
    head = env.head
    manhattan = np.abs(env.fruit - head).sum(axis=1)
    return -manhattan / (env.hw[:, 0] + env.hw[:, 1])


def compute_rewards(
    config: RewardConfig,
    gamma: float,
    result: StepResult,
    phi_before: np.ndarray,
    phi_after: np.ndarray,
) -> np.ndarray:
    """Compute per-environment rewards for one step.

    Args:
        config: Reward settings.
        gamma: Discount factor, used by potential-based shaping.
        result: Step events from the environment.
        phi_before: Potential before the step.
        phi_after: Potential after the step. Ignored on episode ends,
            where the terminal potential is defined as zero.

    Returns:
        Array of shape (num_envs,).
    """
    r = np.full(result.ate.shape, -config.step_cost, dtype=np.float32)
    r += config.fruit * result.ate
    r += config.death * result.died
    r += config.win * result.won
    r += config.starve_penalty * result.starved
    if config.potential_shaping:
        not_done = ~result.done
        r += config.shaping_coef * (gamma * phi_after * not_done - phi_before)
    return r
