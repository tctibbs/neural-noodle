"""Fixed evaluation protocol.

Every policy, classical or learned, is measured by the same runner
under identical dynamics: one frozen environment per episode, row0
init, fixed seeds, and aggregate metrics with mean and spread.
"""

from collections.abc import Callable
from typing import Protocol

import numpy as np
from loguru import logger

from snake_rl.config import BoardSpec, EnvConfig
from snake_rl.env.vec_env import EpisodeStats, VecSnake


class Policy(Protocol):
    """A batch policy emitting absolute directions."""

    def actions(self, env: VecSnake) -> np.ndarray:
        """Return absolute directions, shape (num_envs,)."""
        ...


def run_episodes(
    make_policy: Callable[[VecSnake], Policy],
    board: BoardSpec,
    episodes: int,
    seed: int,
    start_length: int = 3,
    starvation_factor: float = 4.0,
) -> list[EpisodeStats]:
    """Run exactly one episode per environment and collect stats.

    Args:
        make_policy: Builds the policy for the evaluation environment.
        board: Board geometry to evaluate on.
        episodes: Number of episodes, one environment each.
        seed: Base seed for the episode stream.
        start_length: Initial snake length.
        starvation_factor: Starvation cap multiplier.

    Returns:
        One stats record per episode.
    """
    config = EnvConfig(
        boards=[board],
        num_envs=episodes,
        start_length=start_length,
        starvation_factor=starvation_factor,
        init_mode="row0",
    )
    env = VecSnake(config, seed=seed, auto_reset=False)
    policy = make_policy(env)
    cells = board.cells
    hard_cap = int(cells * cells * starvation_factor) + 1_000
    for _ in range(hard_cap):
        if env.all_frozen:
            break
        env.step(policy.actions(env))
    else:
        logger.error(
            "evaluation hit the hard step cap on {}x{}; {} episodes unfinished",
            board.width,
            board.height,
            int((~env.frozen).sum()),
        )
    return env.drain_finished()


def aggregate(stats: list[EpisodeStats]) -> dict[str, float]:
    """Aggregate episode stats into reportable metrics.

    Args:
        stats: Episode records from run_episodes.

    Returns:
        Mean and spread of steps-per-apple and fill, plus win, death,
        and starvation rates. Steps-per-apple averages only episodes
        with at least one apple; their count is reported alongside.
    """
    spa = np.array(
        [s.steps / s.apples for s in stats if s.apples > 0], dtype=float
    )
    fill = np.array([s.fill for s in stats], dtype=float)
    apples = np.array([s.apples for s in stats], dtype=float)
    return {
        "episodes": float(len(stats)),
        "steps_per_apple_mean": float(spa.mean()) if spa.size else float("nan"),
        "steps_per_apple_std": float(spa.std()) if spa.size else float("nan"),
        "steps_per_apple_n": float(spa.size),
        "fill_mean": float(fill.mean()),
        "fill_std": float(fill.std()),
        "apples_mean": float(apples.mean()),
        "win_rate": float(np.mean([s.won for s in stats])),
        "death_rate": float(np.mean([s.died for s in stats])),
        "starve_rate": float(np.mean([s.starved for s in stats])),
    }
