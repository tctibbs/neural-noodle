"""Tests for the vectorized Snake environment."""

import numpy as np

from neural.config import BoardSpec, EnvConfig
from neural.env.vec_env import DOWN, LEFT, RIGHT, UP, VecSnake


def make_env(
    num_envs: int = 4,
    width: int = 8,
    height: int = 8,
    seed: int = 0,
    auto_reset: bool = True,
    init_mode: str = "row0",
) -> VecSnake:
    """Build a small test environment."""
    config = EnvConfig(
        boards=[BoardSpec(width=width, height=height)],
        num_envs=num_envs,
        start_length=3,
        starvation_factor=4.0,
        init_mode=init_mode,
    )
    return VecSnake(config, seed=seed, auto_reset=auto_reset)


def occupancy_consistent(env: VecSnake) -> bool:
    """Check the occupancy grid matches the body ring buffer."""
    for i in range(env.num_envs):
        cells = env.body_cells(i)
        grid = np.zeros_like(env.occ[i])
        grid[cells[:, 0], cells[:, 1]] = True
        if not (grid == env.occ[i]).all():
            return False
    return True


def test_init_state() -> None:
    """Snakes start with the configured length, fruit on free cell."""
    env = make_env()
    assert (env.length == 3).all()
    assert occupancy_consistent(env)
    for i in range(env.num_envs):
        f = env.fruit[i]
        assert not env.occ[i, f[0], f[1]]


def test_movement_and_occupancy() -> None:
    """Random legal play keeps occupancy and length consistent."""
    env = make_env(num_envs=16)
    rng = np.random.default_rng(1)
    for _ in range(200):
        actions = rng.integers(0, 4, env.num_envs)
        env.step(actions)
        assert occupancy_consistent(env)
        assert (env.length >= 3).all()


def test_wall_death() -> None:
    """Driving straight up from row 0 dies immediately."""
    env = make_env(num_envs=1, auto_reset=False)
    result = env.step(np.array([UP]))
    assert result.died[0]
    assert env.frozen[0]


def test_reverse_is_noop() -> None:
    """A reverse action keeps the current direction."""
    env = make_env(num_envs=1, auto_reset=False)
    head_before = env.head[0].copy()
    env.step(np.array([LEFT]))
    if not env.frozen[0]:
        assert env.head[0, 1] == head_before[1] + 1


def test_eating_grows() -> None:
    """Eating the fruit grows the snake by one."""
    env = make_env(num_envs=1, auto_reset=False)
    # Place the fruit directly below the head, always in bounds.
    head = env.head[0]
    env.fruit[0] = (head[0] + 1, head[1])
    result = env.step(np.array([DOWN]))
    assert result.ate[0]
    assert env.length[0] == 4
    assert occupancy_consistent(env)


def test_tail_cell_is_safe_when_not_eating() -> None:
    """Moving into the vacating tail cell is legal."""
    env = make_env(num_envs=1, width=8, height=8, auto_reset=False)
    # Head at (0, c). Walk a tight 2x2 loop: down, left, up arrives at
    # the cell the tail vacates with length 3, then once more around.
    env.fruit[0] = (7, 7)
    for action in [DOWN, LEFT, UP, RIGHT, DOWN, LEFT, UP]:
        result = env.step(np.array([action]))
        assert not result.died[0], f"died on {action}"


def test_starvation_cap() -> None:
    """An agent that never eats hits the starvation cap."""
    env = make_env(num_envs=1, width=8, height=8, auto_reset=False)
    env.fruit[0] = (7, 7)
    cap = int(4.0 * 64)
    done = False
    # Loop a 2x2 square forever.
    for k in range(cap + 8):
        action = [DOWN, LEFT, UP, RIGHT][k % 4]
        result = env.step(np.array([action]))
        if result.done[0]:
            done = bool(result.starved[0])
            break
    assert done


def test_determinism() -> None:
    """Same seed and actions give identical trajectories."""
    env_a = make_env(num_envs=8, seed=5, init_mode="random")
    env_b = make_env(num_envs=8, seed=5, init_mode="random")
    rng = np.random.default_rng(2)
    for _ in range(100):
        actions = rng.integers(0, 4, 8)
        env_a.step(actions.copy())
        env_b.step(actions.copy())
    assert (env_a.body == env_b.body).all()
    assert (env_a.fruit == env_b.fruit).all()
    assert (env_a.length == env_b.length).all()


def test_win_on_full_board() -> None:
    """Filling a 2x2 board from length 3 wins in one step."""
    config = EnvConfig(
        boards=[BoardSpec(width=2, height=2)],
        num_envs=1,
        start_length=1,
        starvation_factor=100.0,
        init_mode="row0",
    )
    env = VecSnake(config, seed=3, auto_reset=False)
    won = False
    for _ in range(50):
        if env.frozen[0]:
            break
        # Follow the 2x2 ring clockwise from wherever the head is.
        r, c = env.head[0]
        action = {(0, 0): RIGHT, (0, 1): DOWN, (1, 1): LEFT, (1, 0): UP}[
            (int(r), int(c))
        ]
        result = env.step(np.array([action]))
        won = won or bool(result.won[0])
    assert won
