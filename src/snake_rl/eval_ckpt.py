"""Evaluate a saved checkpoint under the fixed protocol.

Run with:
    uv run python -m snake_rl.eval_ckpt --ckpt runs/<run>/ckpt_final.pt
"""

import argparse
import time
from pathlib import Path

import torch
from loguru import logger

from snake_rl.agents.rl_policy import RLPolicy
from snake_rl.config import TrainConfig, config_hash
from snake_rl.evaluate import aggregate, run_episodes
from snake_rl.ledger import LedgerRow, append_row
from snake_rl.logging_setup import setup_logging
from snake_rl.train import build_network


def main() -> None:
    """Load a checkpoint and run the full evaluation protocol."""
    parser = argparse.ArgumentParser(description="Checkpoint eval")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override episodes per board.",
    )
    args = parser.parse_args()
    run_id = setup_logging()
    payload = torch.load(args.ckpt, weights_only=False)
    config = TrainConfig.model_validate(payload["config"])
    net = build_network(config)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)
    episodes = args.episodes or config.eval.episodes_per_board
    start = time.perf_counter()
    for board in config.eval.boards:
        policy = RLPolicy(
            net,
            config.obs,
            config.action,
            canvas=max(board.width, board.height),
            device=device,
        )
        stats = run_episodes(
            make_policy=lambda env, p=policy: p,
            board=board,
            episodes=episodes,
            seed=config.eval.seed,
            start_length=config.env.start_length,
        )
        metrics = aggregate(stats)
        logger.info(
            "{} {}x{}: spa {:.2f}+-{:.2f} fill {:.1%} win {:.0%} apples {:.1f}",
            config.run_name,
            board.width,
            board.height,
            metrics["steps_per_apple_mean"],
            metrics["steps_per_apple_std"],
            metrics["fill_mean"],
            metrics["win_rate"],
            metrics["apples_mean"],
        )
        append_row(
            LedgerRow(
                kind="eval_final",
                run_id=run_id,
                run_name=config.run_name,
                config_hash=config_hash(config),
                seed=config.seed,
                frames=int(payload.get("frames", 0)),
                wall_clock_s=time.perf_counter() - start,
                metrics=metrics,
                extra={
                    "board": f"{board.width}x{board.height}",
                    "ckpt": str(args.ckpt),
                },
            )
        )


if __name__ == "__main__":
    main()
