"""Tests for the planner oracle policies."""

import pytest

from snake_rl.config import BoardSpec
from snake_rl.env.vec_env import VecSnake
from snake_rl.evaluate import aggregate, run_episodes
from snake_rl.planner.policies import PlannerPolicy


@pytest.mark.parametrize("mode", ["pure", "shortcut"])
@pytest.mark.parametrize(
    ("width", "height"), [(8, 8), (10, 10), (6, 8), (12, 6)]
)
def test_planner_fills_board(mode: str, width: int, height: int) -> None:
    """Both planner policies win every game on even-height boards."""
    stats = run_episodes(
        make_policy=lambda env: PlannerPolicy(env, mode=mode),
        board=BoardSpec(width=width, height=height),
        episodes=3,
        seed=11,
    )
    assert len(stats) == 3
    for s in stats:
        assert s.won, (
            f"{mode} planner lost on {width}x{height}: "
            f"fill={s.fill:.2f} died={s.died} starved={s.starved}"
        )
        assert s.fill == 1.0


def test_shortcut_not_slower_than_pure() -> None:
    """The shortcut follower is at least as efficient as pure."""
    board = BoardSpec(width=10, height=10)
    pure = aggregate(
        run_episodes(
            make_policy=lambda env: PlannerPolicy(env, mode="pure"),
            board=board,
            episodes=5,
            seed=23,
        )
    )
    cut = aggregate(
        run_episodes(
            make_policy=lambda env: PlannerPolicy(env, mode="shortcut"),
            board=board,
            episodes=5,
            seed=23,
        )
    )
    assert cut["steps_per_apple_mean"] <= pure["steps_per_apple_mean"] + 1e-9


def test_planner_rejects_odd_rows() -> None:
    """Odd-height boards are rejected for the planner."""
    from snake_rl.config import EnvConfig

    config = EnvConfig(boards=[BoardSpec(width=8, height=5)], num_envs=1)
    env = VecSnake(config, seed=0)
    with pytest.raises(ValueError, match="even row count"):
        PlannerPolicy(env, mode="pure")
