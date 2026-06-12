"""Observation builders for the vectorized environment.

The grid observation stacks board planes on a square canvas:
channel 0 body, channel 1 head, channel 2 fruit, channel 3 wall and
out-of-board padding. With body_decay, body cells hold time-to-vacate
in (0, 1], tail lowest, head highest, so the network can see where
space frees up. With egocentric on, the canvas rotates so the snake
always faces up.

The features9 observation reproduces the legacy 9-feature vector as an
ablation baseline: one-hot direction, four normalized distances to the
nearest obstacle, and a normalized Manhattan fruit distance.
"""

import numpy as np

from snake_rl.config import ObsConfig
from snake_rl.env.vec_env import DELTAS, VecSnake

GRID_CHANNELS = 4
FEATURES9_DIM = 9


class GridObsBuilder:
    """Builds (num_envs, 4, canvas, canvas) float32 observations.

    Args:
        config: Observation settings.
        canvas: Square canvas size. Must cover the largest board the
            environment can sample.
    """

    def __init__(self, config: ObsConfig, canvas: int) -> None:
        self.config = config
        self.canvas = canvas

    def build(self, env: VecSnake) -> np.ndarray:
        """Render the current state of every environment.

        Args:
            env: The environment to observe.

        Returns:
            Array of shape (num_envs, 4, canvas, canvas).
        """
        b = env.num_envs
        s = self.canvas
        obs = np.zeros((b, GRID_CHANNELS, s, s), dtype=np.float32)
        envs = np.arange(b)

        # Body plane, vectorized over the longest snake in the batch.
        lmax = int(env.length.max())
        k = np.arange(lmax)[None, :]
        valid = k < env.length[:, None]
        ring = (env.tail_i[:, None] + k) % env.capacity
        rows = np.take_along_axis(env.body[:, :, 0], ring, axis=1)
        cols = np.take_along_axis(env.body[:, :, 1], ring, axis=1)
        if self.config.body_decay:
            values = (k + 1.0) / env.length[:, None]
        else:
            values = np.ones_like(ring, dtype=np.float32)
        bb = np.broadcast_to(envs[:, None], ring.shape)
        obs[bb[valid], 0, rows[valid], cols[valid]] = values[valid]

        head = env.head
        obs[envs, 1, head[:, 0], head[:, 1]] = 1.0
        obs[envs, 2, env.fruit[:, 0], env.fruit[:, 1]] = 1.0

        rr = np.arange(s)[None, :, None]
        cc = np.arange(s)[None, None, :]
        wall = (rr >= env.hw[:, 0, None, None]) | (
            cc >= env.hw[:, 1, None, None]
        )
        obs[:, 3] = wall.astype(np.float32)

        if self.config.egocentric:
            for k_rot in range(1, 4):
                sel = env.direction == k_rot
                if sel.any():
                    obs[sel] = np.rot90(obs[sel], k_rot, axes=(2, 3))
        return obs


class Features9Builder:
    """Builds the legacy 9-feature vector observation.

    Args:
        config: Observation settings. Egocentric and body_decay do not
            apply to this representation.
    """

    def __init__(self, config: ObsConfig) -> None:
        self.config = config

    def build(self, env: VecSnake) -> np.ndarray:
        """Compute features for every environment.

        Args:
            env: The environment to observe.

        Returns:
            Array of shape (num_envs, 9).
        """
        b = env.num_envs
        out = np.zeros((b, FEATURES9_DIM), dtype=np.float32)
        out[np.arange(b), env.direction] = 1.0
        head = env.head
        max_dim = max(int(env.hw[:, 0].max()), int(env.hw[:, 1].max()))
        for d in range(4):
            delta = DELTAS[d]
            dist = np.zeros(b, dtype=np.float32)
            pos = head.copy()
            alive = np.ones(b, dtype=bool)
            for _ in range(max_dim):
                pos = pos + delta
                inside = (
                    (pos[:, 0] >= 0)
                    & (pos[:, 1] >= 0)
                    & (pos[:, 0] < env.hw[:, 0])
                    & (pos[:, 1] < env.hw[:, 1])
                )
                clipped = np.clip(pos, 0, env.occ.shape[1] - 1)
                blocked = (
                    ~inside
                    | env.occ[np.arange(b), clipped[:, 0], clipped[:, 1]]
                )
                dist += alive.astype(np.float32) * (~blocked).astype(np.float32)
                alive = alive & ~blocked
            out[:, 4 + d] = dist / max_dim
        manhattan = np.abs(env.fruit - head).sum(axis=1)
        out[:, 8] = manhattan / (env.hw[:, 0] + env.hw[:, 1])
        return out
