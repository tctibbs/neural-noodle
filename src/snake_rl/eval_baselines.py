"""Measure the baseline policies and write results to the ledger.

Run with:
    uv run python -m snake_rl.eval_baselines
"""

import time
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from snake_rl.agents.baselines import LooperPolicy, RandomPolicy
from snake_rl.config import BoardSpec, config_hash
from snake_rl.env.vec_env import VecSnake
from snake_rl.evaluate import Policy, aggregate, run_episodes
from snake_rl.ledger import LedgerRow, append_row
from snake_rl.logging_setup import setup_logging


class BaselineEvalConfig(BaseModel):
    """Configuration for a baseline measurement.

    Attributes:
        policy: Baseline variant.
        boards: Board geometries to measure.
        episodes: Episodes per board.
        seed: Base seed for the episode stream.
        start_length: Initial snake length.
    """

    policy: Literal["random", "looper"] = "random"
    boards: list[BoardSpec] = Field(
        default_factory=lambda: [
            BoardSpec(width=8, height=8),
            BoardSpec(width=10, height=10),
            BoardSpec(width=12, height=12),
            BoardSpec(width=16, height=16),
        ]
    )
    episodes: int = 100
    seed: int = 101
    start_length: int = 3


def _make_policy(config: BaselineEvalConfig, env: VecSnake) -> Policy:
    """Build the configured baseline policy."""
    if config.policy == "random":
        return RandomPolicy(env, seed=config.seed)
    return LooperPolicy(env)


def main() -> None:
    """Measure both baselines and append ledger rows."""
    run_id = setup_logging()
    for name in ("random", "looper"):
        config = BaselineEvalConfig(policy=name)
        chash = config_hash(config)
        for board in config.boards:
            start = time.perf_counter()
            stats = run_episodes(
                make_policy=lambda env, c=config: _make_policy(c, env),
                board=board,
                episodes=config.episodes,
                seed=config.seed,
                start_length=config.start_length,
            )
            metrics = aggregate(stats)
            elapsed = time.perf_counter() - start
            logger.info(
                "{} {}x{}: steps_per_apple={:.1f} (n={:.0f}) "
                "apples={:.2f} fill={:.1%} starve={:.0%}",
                name,
                board.width,
                board.height,
                metrics["steps_per_apple_mean"],
                metrics["steps_per_apple_n"],
                metrics["apples_mean"],
                metrics["fill_mean"],
                metrics["starve_rate"],
            )
            append_row(
                LedgerRow(
                    kind="baseline",
                    run_id=run_id,
                    run_name=f"baseline-{name}-{board.width}x{board.height}",
                    config_hash=chash,
                    seed=config.seed,
                    frames=sum(s.steps for s in stats),
                    wall_clock_s=elapsed,
                    metrics=metrics,
                    extra={
                        "board": f"{board.width}x{board.height}",
                        "config": config.model_dump(mode="json"),
                    },
                )
            )


if __name__ == "__main__":
    main()
