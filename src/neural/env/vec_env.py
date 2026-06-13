"""Single-process numpy-vectorized batched Snake environment.

One canonical implementation of the game rules is shared by the RL
agent, the baselines, and the planner oracle so every reported metric
is measured under identical dynamics.

Rules:
    The snake moves one cell per step. Eating a fruit grows the snake
    by one and respawns the fruit uniformly on a free cell. The tail
    cell vacates before head collision is checked, so moving into the
    current tail cell is legal when not eating. The episode ends on
    wall or body collision (death), on filling the board (win), or
    after ``starvation_factor * cells`` steps without eating
    (starvation).

Coordinates are (row, col). Directions: 0 up, 1 right, 2 down, 3 left.
"""

from dataclasses import dataclass

import numpy as np

from neural.config import EnvConfig

DELTAS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int64)

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3


@dataclass
class StepResult:
    """Per-step event arrays, each of shape (num_envs,).

    Attributes:
        ate: Fruit eaten this step.
        died: Wall or body collision this step.
        won: Board filled this step.
        starved: Starvation cap hit this step.
        done: Episode ended this step for any reason.
    """

    ate: np.ndarray
    died: np.ndarray
    won: np.ndarray
    starved: np.ndarray
    done: np.ndarray


@dataclass
class EpisodeStats:
    """Summary of one finished episode.

    Attributes:
        apples: Fruits eaten.
        steps: Steps taken.
        length: Snake length at episode end.
        width: Board width.
        height: Board height.
        died: Ended by collision.
        won: Ended by filling the board.
        starved: Ended by the starvation cap.
    """

    apples: int
    steps: int
    length: int
    width: int
    height: int
    died: bool
    won: bool
    starved: bool

    @property
    def fill(self) -> float:
        """Fraction of the board occupied at episode end."""
        return self.length / (self.width * self.height)


class VecSnake:
    """Batched Snake simulator backed by flat numpy arrays.

    Args:
        config: Environment settings.
        seed: Base seed. Per-env streams derive from it, so results
            are reproducible for a fixed (seed, num_envs) pair.
        auto_reset: Reset an environment in place when its episode
            ends. When false, finished environments freeze, which the
            fixed evaluation protocol uses to run exactly one episode
            per environment.
    """

    def __init__(
        self,
        config: EnvConfig,
        seed: int = 0,
        auto_reset: bool = True,
    ) -> None:
        self.config = config
        self.auto_reset = auto_reset
        self.num_envs = config.num_envs
        boards = config.boards
        self.max_h = max(b.height for b in boards)
        self.max_w = max(b.width for b in boards)
        self.capacity = self.max_h * self.max_w

        b = self.num_envs
        self._board_choices = np.array(
            [[s.height, s.width] for s in boards], dtype=np.int64
        )
        self.hw = np.zeros((b, 2), dtype=np.int64)
        self.body = np.zeros((b, self.capacity, 2), dtype=np.int64)
        self.head_i = np.zeros(b, dtype=np.int64)
        self.tail_i = np.zeros(b, dtype=np.int64)
        self.length = np.zeros(b, dtype=np.int64)
        self.occ = np.zeros((b, self.max_h, self.max_w), dtype=bool)
        self.fruit = np.zeros((b, 2), dtype=np.int64)
        self.direction = np.zeros(b, dtype=np.int64)
        self.steps = np.zeros(b, dtype=np.int64)
        self.since_fruit = np.zeros(b, dtype=np.int64)
        self.apples = np.zeros(b, dtype=np.int64)
        self.frozen = np.zeros(b, dtype=bool)

        self._rng = [
            np.random.Generator(np.random.PCG64([seed, i])) for i in range(b)
        ]
        self._finished: list[EpisodeStats] = []
        for i in range(b):
            self._reset_env(i)

    @property
    def head(self) -> np.ndarray:
        """Current head cell per env, shape (num_envs, 2)."""
        return self.body[np.arange(self.num_envs), self.head_i]

    @property
    def tail(self) -> np.ndarray:
        """Current tail cell per env, shape (num_envs, 2)."""
        return self.body[np.arange(self.num_envs), self.tail_i]

    def body_cells(self, i: int) -> np.ndarray:
        """Return env i's body cells ordered tail to head.

        Args:
            i: Environment index.

        Returns:
            Array of shape (length, 2).
        """
        idx = (self.tail_i[i] + np.arange(self.length[i])) % self.capacity
        return self.body[i, idx]

    def drain_finished(self) -> list[EpisodeStats]:
        """Return and clear stats for episodes finished so far."""
        out = self._finished
        self._finished = []
        return out

    def _reset_env(self, i: int) -> None:
        """Re-initialize environment i in place."""
        rng = self._rng[i]
        h, w = self._board_choices[rng.integers(len(self._board_choices))]
        self.hw[i] = (h, w)
        self.occ[i] = False
        n = self.config.start_length

        # Lay the snake in a straight horizontal line, head to the
        # right of the body, facing right. Row 0 keeps the body behind
        # the head in Hamiltonian cycle order for the planner.
        if self.config.init_mode == "row0":
            row = 0
        else:
            row = int(rng.integers(h))
        col0 = int(rng.integers(w - n + 1))
        cells = [(row, col0 + k) for k in range(n)]
        for k, (r, c) in enumerate(cells):
            self.body[i, k] = (r, c)
            self.occ[i, r, c] = True
        self.tail_i[i] = 0
        self.head_i[i] = n - 1
        self.length[i] = n
        self.direction[i] = RIGHT
        self.steps[i] = 0
        self.since_fruit[i] = 0
        self.apples[i] = 0
        self.frozen[i] = False
        self._spawn_fruit(i)

    def _spawn_fruit(self, i: int) -> None:
        """Place env i's fruit uniformly on a free cell."""
        h, w = self.hw[i]
        free = np.argwhere(~self.occ[i, :h, :w])
        pick = free[self._rng[i].integers(len(free))]
        self.fruit[i] = pick

    def _record(self, i: int, died: bool, won: bool, starved: bool) -> None:
        """Record env i's finished episode."""
        self._finished.append(
            EpisodeStats(
                apples=int(self.apples[i]),
                steps=int(self.steps[i]),
                length=int(self.length[i]),
                width=int(self.hw[i, 1]),
                height=int(self.hw[i, 0]),
                died=died,
                won=won,
                starved=starved,
            )
        )

    def step(self, actions: np.ndarray) -> StepResult:
        """Advance every environment by one step.

        Args:
            actions: Absolute directions, shape (num_envs,). A reverse
                move keeps the current direction. Egocentric action
                spaces map to absolute directions before this call.

        Returns:
            Event arrays for the step. Frozen environments report no
            events.
        """
        b = self.num_envs
        idx = np.arange(b)
        active = ~self.frozen

        reverse = (self.direction + 2) % 4
        eff = np.where(actions == reverse, self.direction, actions)
        self.direction = np.where(active, eff, self.direction)

        new_head = self.head + DELTAS[eff]
        oob = (
            (new_head[:, 0] < 0)
            | (new_head[:, 1] < 0)
            | (new_head[:, 0] >= self.hw[:, 0])
            | (new_head[:, 1] >= self.hw[:, 1])
        )
        nh = np.clip(new_head, 0, [self.max_h - 1, self.max_w - 1])
        ate = (
            active
            & ~oob
            & (nh[:, 0] == self.fruit[:, 0])
            & (nh[:, 1] == self.fruit[:, 1])
        )

        # Tail vacates first for non-eating envs, then collisions are
        # checked against the remaining occupancy.
        vacate = active & ~ate
        vi = idx[vacate]
        tails = self.body[vi, self.tail_i[vacate]]
        self.occ[vi, tails[:, 0], tails[:, 1]] = False
        self.tail_i[vacate] = (self.tail_i[vacate] + 1) % self.capacity

        hit_body = self.occ[idx, nh[:, 0], nh[:, 1]]
        died = active & (oob | hit_body)

        # Advance heads for surviving envs.
        move = active & ~died
        mi = idx[move]
        self.head_i[move] = (self.head_i[move] + 1) % self.capacity
        self.body[mi, self.head_i[move]] = nh[move]
        self.occ[mi, nh[move, 0], nh[move, 1]] = True
        self.length[move] += ate[move].astype(np.int64)
        self.steps[move] += 1
        self.since_fruit[move] += 1
        self.since_fruit[ate] = 0
        self.apples[ate] += 1

        cells = self.hw[:, 0] * self.hw[:, 1]
        won = move & ate & (self.length == cells)
        cap = (self.config.starvation_factor * cells).astype(np.int64)
        starved = move & (self.since_fruit >= cap)

        for i in idx[ate & ~won]:
            self._spawn_fruit(int(i))

        done = died | won | starved
        for i in idx[done]:
            self._record(
                int(i),
                died=bool(died[i]),
                won=bool(won[i]),
                starved=bool(starved[i]),
            )
            if self.auto_reset:
                self._reset_env(int(i))
            else:
                self.frozen[i] = True

        return StepResult(
            ate=ate, died=died, won=won, starved=starved, done=done
        )

    @property
    def all_frozen(self) -> bool:
        """True when every environment has finished and frozen."""
        return bool(self.frozen.all())
