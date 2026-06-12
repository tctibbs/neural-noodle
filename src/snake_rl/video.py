"""Render evaluation episodes of a checkpoint to an mp4.

Run with:
    uv run python -m snake_rl.video --ckpt runs/<run>/ckpt_final.pt \
        --board 10x10 --episodes 3 --out videos/best.mp4
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from snake_rl.agents.rl_policy import RLPolicy
from snake_rl.config import BoardSpec, EnvConfig, TrainConfig
from snake_rl.env.vec_env import VecSnake
from snake_rl.evaluate import Policy
from snake_rl.planner.policies import PlannerPolicy
from snake_rl.train import build_network

CELL = 32
MARGIN = 2

BG = np.array([24, 26, 32], dtype=np.uint8)
GRIDLINE = np.array([38, 41, 50], dtype=np.uint8)
BODY_OLD = np.array([46, 110, 64], dtype=np.uint8)
BODY_NEW = np.array([88, 200, 120], dtype=np.uint8)
HEAD = np.array([240, 240, 240], dtype=np.uint8)
FRUIT = np.array([225, 80, 80], dtype=np.uint8)


def draw_frame(env: VecSnake) -> np.ndarray:
    """Draw environment 0 as an RGB image.

    Args:
        env: Environment with at least one board.

    Returns:
        Image array of shape (h*CELL, w*CELL, 3).
    """
    h, w = int(env.hw[0, 0]), int(env.hw[0, 1])
    img = np.zeros((h * CELL, w * CELL, 3), dtype=np.uint8)
    img[:] = GRIDLINE
    for r in range(h):
        for c in range(w):
            _cell(img, r, c, BG)
    cells = env.body_cells(0)
    n = len(cells)
    for k, (r, c) in enumerate(cells.tolist()):
        t = (k + 1) / n
        color = (BODY_OLD * (1 - t) + BODY_NEW * t).astype(np.uint8)
        _cell(img, r, c, color)
    head = env.head[0]
    _cell(img, int(head[0]), int(head[1]), HEAD)
    fruit = env.fruit[0]
    _cell(img, int(fruit[0]), int(fruit[1]), FRUIT)
    return img


def _cell(img: np.ndarray, r: int, c: int, color: np.ndarray) -> None:
    """Fill one cell with a margin."""
    img[
        r * CELL + MARGIN : (r + 1) * CELL - MARGIN,
        c * CELL + MARGIN : (c + 1) * CELL - MARGIN,
    ] = color


def record(
    policy: Policy,
    env: VecSnake,
    max_steps: int,
) -> list[np.ndarray]:
    """Roll one frozen-env episode and return its frames."""
    frames = [draw_frame(env)]
    for _ in range(max_steps):
        if env.all_frozen:
            break
        env.step(policy.actions(env))
        frames.append(draw_frame(env))
    return frames


def main() -> None:
    """Render episodes from a checkpoint or the planner."""
    parser = argparse.ArgumentParser(description="Episode renderer")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--planner", choices=["pure", "shortcut"], default=None)
    parser.add_argument("--board", type=str, default="10x10")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7_777)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    width, height = (int(x) for x in args.board.split("x"))
    board = BoardSpec(width=width, height=height)

    all_frames: list[np.ndarray] = []
    for episode in range(args.episodes):
        env = VecSnake(
            EnvConfig(
                boards=[board],
                num_envs=1,
                start_length=3,
                starvation_factor=4.0,
                init_mode="row0",
            ),
            seed=args.seed + episode,
            auto_reset=False,
        )
        policy: Policy
        if args.ckpt is not None:
            payload = torch.load(args.ckpt, weights_only=False)
            config = TrainConfig.model_validate(payload["config"])
            net = build_network(config)
            net.load_state_dict(payload["state_dict"])
            net.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            net.to(device)
            policy = RLPolicy(
                net,
                config.obs,
                config.action,
                canvas=max(width, height),
                device=device,
            )
        elif args.planner is not None:
            policy = PlannerPolicy(env, mode=args.planner)
        else:
            msg = "pass --ckpt or --planner"
            raise SystemExit(msg)
        cap = board.cells * board.cells * 4
        all_frames.extend(record(policy, env, cap))
        stats = env.drain_finished()
        for s in stats:
            print(
                f"episode {episode}: apples={s.apples} steps={s.steps} "
                f"fill={s.fill:.1%} won={s.won}"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, all_frames, fps=args.fps)
    print(args.out)


if __name__ == "__main__":
    main()
