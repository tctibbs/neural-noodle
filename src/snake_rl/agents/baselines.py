"""Non-learned baseline policies anchoring the bottom of the scale.

The random policy is the floor. The looper embodies the well-known
survival-by-looping failure mode: it circles a tight 2x2 loop forever,
never seeking fruit, eating only when a fruit happens to land in its
path, until the starvation cap ends the episode.
"""

import numpy as np

from snake_rl.env.vec_env import DOWN, LEFT, RIGHT, UP, VecSnake


class RandomPolicy:
    """Uniform random absolute directions.

    Args:
        env: The environment to act in, used for sizing and seeding.
        seed: Seed for the action stream.
    """

    def __init__(self, env: VecSnake, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._num = env.num_envs

    def actions(self, env: VecSnake) -> np.ndarray:
        """Return uniform random directions, shape (num_envs,)."""
        return self._rng.integers(0, 4, self._num)


class LooperPolicy:
    """Survival-only agent circling a tight 2x2 loop.

    From the row0 init facing right, the repeating pattern down, left,
    up, right traces a 2x2 square indefinitely. The policy tracks its
    own phase per environment and resets it when an episode ends.
    """

    _PATTERN = (DOWN, LEFT, UP, RIGHT)

    def __init__(self, env: VecSnake) -> None:
        self._phase = np.zeros(env.num_envs, dtype=np.int64)

    def actions(self, env: VecSnake) -> np.ndarray:
        """Return the next loop step per environment."""
        pattern = np.array(self._PATTERN, dtype=np.int64)
        acts = pattern[self._phase % 4]
        fresh = env.steps == 0
        acts[fresh] = pattern[0]
        self._phase = np.where(fresh, 1, self._phase + 1)
        return acts
