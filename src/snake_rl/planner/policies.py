"""Cycle-following planner policies.

Both policies assume the snake's body lies behind its head in cycle
order, which holds from a row0 init and is preserved by only ever
moving forward along the cycle.

The pure follower always advances one cycle position and is a
guaranteed win. The shortcut follower may jump forward along the cycle
toward the fruit when the jump provably cannot reach any body cell:
body cells occupy forward cycle distances in [dist(tail), N), so any
jump shorter than dist(tail), with a safety margin for growth, lands
on free cells and keeps the invariant.
"""

from typing import Literal

import numpy as np

from snake_rl.env.vec_env import DELTAS, VecSnake
from snake_rl.planner.hamiltonian import (
    cycle_order,
    hamiltonian_cycle,
    verify_cycle,
)


class PlannerPolicy:
    """Hamiltonian cycle policy over a vectorized environment.

    Args:
        env: The environment the policy will act in. Cycles are
            precomputed and verified for every board geometry the
            environment can sample.
        mode: Pure always follows the cycle. Shortcut jumps forward
            toward the fruit when provably safe.
        margin: Minimum cycle distance kept between a jump target and
            the tail. Covers growth on eating.
        max_fill: Shortcuts are disabled once the snake occupies this
            fraction of the board, falling back to pure following.
    """

    def __init__(
        self,
        env: VecSnake,
        mode: Literal["pure", "shortcut"] = "pure",
        margin: int = 4,
        max_fill: float = 0.5,
    ) -> None:
        self.mode = mode
        self.margin = margin
        self.max_fill = max_fill
        self._cycles: dict[tuple[int, int], np.ndarray] = {}
        self._orders: dict[tuple[int, int], np.ndarray] = {}
        for h, w in env._board_choices:
            if h % 2 == 1:
                msg = (
                    f"planner needs an even row count, got {h}x{w}; "
                    "row0 init is only cycle-consistent then"
                )
                raise ValueError(msg)
            cycle = hamiltonian_cycle(int(h), int(w))
            verify_cycle(cycle, int(h), int(w))
            self._cycles[(int(h), int(w))] = cycle
            self._orders[(int(h), int(w))] = cycle_order(cycle, int(h), int(w))

    def actions(self, env: VecSnake) -> np.ndarray:
        """Compute absolute direction actions for every environment.

        Args:
            env: The environment to act in.

        Returns:
            Absolute directions, shape (num_envs,).
        """
        acts = np.zeros(env.num_envs, dtype=np.int64)
        for i in range(env.num_envs):
            if env.frozen[i]:
                continue
            acts[i] = self._act_one(env, i)
        return acts

    def _act_one(self, env: VecSnake, i: int) -> int:
        """Pick the direction for environment i."""
        h, w = int(env.hw[i, 0]), int(env.hw[i, 1])
        n = h * w
        order = self._orders[(h, w)]
        cycle = self._cycles[(h, w)]
        head = env.body[i, env.head_i[i]]
        o_head = order[head[0], head[1]]

        target = cycle[(o_head + 1) % n]
        if self.mode == "shortcut":
            shortcut = self._best_shortcut(env, i, order, o_head, n)
            if shortcut is not None:
                target = shortcut
        delta = (int(target[0] - head[0]), int(target[1] - head[1]))
        return _direction_of(delta)

    def _best_shortcut(
        self,
        env: VecSnake,
        i: int,
        order: np.ndarray,
        o_head: int,
        n: int,
    ) -> np.ndarray | None:
        """Return the best safe jump target, or None to follow purely.

        A neighbor at forward cycle distance d is safe when
        d <= dist(fruit), so the fruit is never overshot, and
        d <= dist(tail) - margin, so the jump cannot land on or skip
        the body.
        """
        h, w = int(env.hw[i, 0]), int(env.hw[i, 1])
        if env.length[i] >= self.max_fill * n:
            return None
        tail = env.body[i, env.tail_i[i]]
        fruit = env.fruit[i]
        d_tail = (order[tail[0], tail[1]] - o_head) % n
        d_fruit = (order[fruit[0], fruit[1]] - o_head) % n

        head = env.body[i, env.head_i[i]]
        best_d = 1
        best: np.ndarray | None = None
        for delta in DELTAS:
            r, c = int(head[0] + delta[0]), int(head[1] + delta[1])
            if r < 0 or c < 0 or r >= h or c >= w:
                continue
            d = (order[r, c] - o_head) % n
            if d <= best_d or d > d_fruit or d > d_tail - self.margin:
                continue
            best_d = d
            best = np.array([r, c], dtype=np.int64)
        return best


def _direction_of(delta: tuple[int, int]) -> int:
    """Map a unit (row, col) delta to an absolute direction."""
    for k, (dr, dc) in enumerate(DELTAS.tolist()):
        if delta == (dr, dc):
            return k
    msg = f"non-unit delta {delta}"
    raise ValueError(msg)
