"""Measure the planner oracle and write results to the ledger.

Run with:
    uv run python -m neural.eval_planner
"""

import time
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from neural.config import BoardSpec, config_hash
from neural.env.vec_env import VecSnake
from neural.evaluate import aggregate, run_episodes
from neural.ledger import LedgerRow, append_row
from neural.logging_setup import setup_logging
from neural.planner.policies import PlannerPolicy


class PlannerEvalConfig(BaseModel):
    """Configuration for a planner oracle measurement.

    Attributes:
        mode: Planner policy variant.
        margin: Shortcut safety margin in cycle cells.
        max_fill: Fill fraction above which shortcuts are disabled.
        boards: Board geometries to measure.
        episodes: Episodes per board.
        seed: Base seed for the episode stream.
        start_length: Initial snake length.
    """

    mode: Literal["pure", "shortcut"] = "pure"
    margin: int = 4
    max_fill: float = 0.5
    boards: list[BoardSpec] = Field(
        default_factory=lambda: [
            BoardSpec(width=6, height=6),
            BoardSpec(width=8, height=8),
            BoardSpec(width=9, height=12),
            BoardSpec(width=10, height=10),
            BoardSpec(width=12, height=12),
            BoardSpec(width=14, height=14),
            BoardSpec(width=16, height=8),
            BoardSpec(width=16, height=16),
            BoardSpec(width=18, height=18),
            BoardSpec(width=20, height=20),
            BoardSpec(width=24, height=24),
        ]
    )
    episodes: int = 100
    seed: int = 101
    start_length: int = 3


def main() -> None:
    """Measure both planner modes and append ledger rows."""
    run_id = setup_logging()
    for mode in ("pure", "shortcut"):
        config = PlannerEvalConfig(mode=mode)
        chash = config_hash(config)
        for board in config.boards:
            start = time.perf_counter()

            def make_policy(
                env: VecSnake, cfg: PlannerEvalConfig = config
            ) -> PlannerPolicy:
                return PlannerPolicy(
                    env,
                    mode=cfg.mode,
                    margin=cfg.margin,
                    max_fill=cfg.max_fill,
                )

            stats = run_episodes(
                make_policy=make_policy,
                board=board,
                episodes=config.episodes,
                seed=config.seed,
                start_length=config.start_length,
            )
            metrics = aggregate(stats)
            elapsed = time.perf_counter() - start
            frames = sum(s.steps for s in stats)
            logger.info(
                "{} {}x{}: steps_per_apple={:.2f}+-{:.2f} fill={:.1%} "
                "win={:.0%}",
                mode,
                board.width,
                board.height,
                metrics["steps_per_apple_mean"],
                metrics["steps_per_apple_std"],
                metrics["fill_mean"],
                metrics["win_rate"],
            )
            append_row(
                LedgerRow(
                    kind="planner",
                    run_id=run_id,
                    run_name=f"planner-{mode}-{board.width}x{board.height}",
                    config_hash=chash,
                    seed=config.seed,
                    frames=frames,
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
