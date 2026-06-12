"""Reward computation, kept outside the simulator.

Rewards are a pure function of step events and (optionally) a
potential-based shaping term over the head-to-fruit Manhattan
distance. Keeping this separate from the environment means reward
ablations can never change the game rules.
"""

import numpy as np
import torch

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


def compute_rewards_t(
    config: RewardConfig,
    gamma: float,
    events: dict[str, torch.Tensor],
    phi_before: torch.Tensor | None,
    phi_after: torch.Tensor | None,
) -> torch.Tensor:
    """Device-side twin of compute_rewards for the tensor backend.

    Args:
        config: Reward settings.
        gamma: Discount factor.
        events: Bool event tensors from TensorVecSnake.step.
        phi_before: Potential before the step, or None when shaping
            is off.
        phi_after: Potential after the step, or None when shaping is
            off.

    Returns:
        Float reward tensor of shape (num_envs,).
    """
    ate = events["ate"]
    r = torch.full_like(ate, 0, dtype=torch.float32) - config.step_cost
    r = r + config.fruit * ate.float()
    r = r + config.death * events["died"].float()
    r = r + config.win * events["won"].float()
    r = r + config.starve_penalty * events["starved"].float()
    if config.potential_shaping:
        if phi_before is None or phi_after is None:
            msg = "shaping enabled but potentials missing"
            raise ValueError(msg)
        not_done = (~events["done"]).float()
        r = r + config.shaping_coef * (
            gamma * phi_after * not_done - phi_before
        )
    return r
