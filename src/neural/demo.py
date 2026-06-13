"""Capability-frontier demo GIF for the repo page.

Renders one trained policy side by side on boards of increasing size,
so a viewer sees the frontier directly: small boards fill and win,
large boards stay efficient per apple but stall before completion.

Each board plays its full episode to termination, then playback is
time-normalized so every panel starts and ends together regardless of
true episode length. Large boards are therefore shown sped up
relative to small ones, and each panel reports its real final
outcome, not a mid-episode snapshot. Honest by construction, it shows
both the strength and the limit.

Run with:
    uv run python -m neural.demo --ckpt runs/<run>/ckpt_final.pt \
        --boards 8x8,12x12,16x16,20x20 --out docs/demo.gif
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from neural.agents.rl_policy import RLPolicy
from neural.config import BoardSpec, EnvConfig, TrainConfig
from neural.env.vec_env import VecSnake
from neural.evaluate import Policy
from neural.train import build_network
from neural.video import BG, BODY_NEW, BODY_OLD, FRUIT, GRIDLINE, HEAD

MARGIN = 1
HEADER = 52
GAP = 18
PANEL_BG = (16, 18, 23)
WIN_BANNER = (88, 200, 120)
FAIL_BANNER = (225, 80, 80)
TEXT = (235, 237, 242)
SUBTEXT = (150, 156, 168)


def _font(size: int) -> ImageFont.FreeTypeFont:
    """Load a scalable font shipped with matplotlib."""
    path = font_manager.findfont("DejaVu Sans")
    return ImageFont.truetype(path, size)


def draw_board(env: VecSnake, cell: int) -> np.ndarray:
    """Draw environment 0 as an RGB image at the given cell size.

    Args:
        env: Single-board environment.
        cell: Pixel size of one board cell.

    Returns:
        Image of shape (h*cell, w*cell, 3).
    """
    h, w = int(env.hw[0, 0]), int(env.hw[0, 1])
    img = np.zeros((h * cell, w * cell, 3), dtype=np.uint8)
    img[:] = GRIDLINE

    def fill(r: int, c: int, color: np.ndarray) -> None:
        img[
            r * cell + MARGIN : (r + 1) * cell - MARGIN,
            c * cell + MARGIN : (c + 1) * cell - MARGIN,
        ] = color

    for r in range(h):
        for c in range(w):
            fill(r, c, BG)
    cells = env.body_cells(0)
    n = len(cells)
    for k, (r, c) in enumerate(cells.tolist()):
        t = (k + 1) / n
        fill(r, c, (BODY_OLD * (1 - t) + BODY_NEW * t).astype(np.uint8))
    head = env.head[0]
    fill(int(head[0]), int(head[1]), HEAD)
    # On a full board the fruit is gone; do not paint a stale one.
    if int(env.length[0]) < h * w:
        fruit = env.fruit[0]
        fill(int(fruit[0]), int(fruit[1]), FRUIT)
    return img


def render_panel(
    env: VecSnake, board: BoardSpec, cell: int, outcome: str
) -> np.ndarray:
    """Render a labelled panel (header plus board) at native height.

    Args:
        env: Single-board environment to draw.
        board: Board geometry.
        cell: Pixel size of one board cell.
        outcome: WIN, STALL, or empty while running.

    Returns:
        Panel image of shape (HEADER + h*cell, w*cell, 3).
    """
    board_img = draw_board(env, cell)
    w = board_img.shape[1]
    panel = np.zeros((HEADER + board_img.shape[0], w, 3), dtype=np.uint8)
    panel[:] = PANEL_BG
    panel[HEADER:] = board_img

    img = Image.fromarray(panel)
    d = ImageDraw.Draw(img)
    d.text((6, 4), f"{board.width}x{board.height}", font=_font(22), fill=TEXT)
    d.text(
        (6, 31),
        f"fill {int(env.length[0]) / board.cells:.0%}   "
        f"apples {int(env.apples[0])}",
        font=_font(15),
        fill=SUBTEXT,
    )
    if outcome:
        color = WIN_BANNER if outcome == "WIN" else FAIL_BANNER
        fnt = _font(20)
        tw = d.textlength(outcome, font=fnt)
        d.text((w - tw - 8, 6), outcome, font=fnt, fill=color)
    return np.asarray(img)


def roll_panel(
    board: BoardSpec,
    policy: Policy,
    seed: int,
    cell: int,
    n_frames: int,
    max_steps: int,
) -> tuple[list[np.ndarray], str, float]:
    """Play a full episode and return time-normalized panel frames.

    The episode is measured once, then replayed deterministically,
    rendering exactly n_frames evenly spaced over its true length so
    every panel spans the same number of frames.

    Args:
        board: Board geometry.
        policy: Policy to drive the board.
        seed: Episode seed.
        cell: Pixel size of one board cell.
        n_frames: Frames to emit for this panel.
        max_steps: Hard cap on episode length.

    Returns:
        The panel frames, the outcome label, and the final fill.
    """

    def build() -> VecSnake:
        return VecSnake(
            EnvConfig(
                boards=[board],
                num_envs=1,
                start_length=3,
                starvation_factor=4.0,
                init_mode="row0",
            ),
            seed=seed,
            auto_reset=False,
        )

    measure = build()
    length = 0
    outcome = "STALL"
    for _ in range(max_steps):
        if measure.all_frozen:
            break
        measure.step(policy.actions(measure))
        length += 1
        if measure.all_frozen:
            stats = measure.drain_finished()
            outcome = "WIN" if stats and stats[0].won else "STALL"
            break
    final_fill = float(measure.length[0]) / board.cells

    targets = [round(i / (n_frames - 1) * length) for i in range(n_frames)]
    env = build()
    frames: list[np.ndarray] = []
    step = 0
    for i, target in enumerate(targets):
        while step < target and not env.all_frozen:
            env.step(policy.actions(env))
            step += 1
        shown = outcome if i == n_frames - 1 else ""
        frames.append(render_panel(env, board, cell, shown))
    return frames, outcome, final_fill


def compose(frames: list[np.ndarray]) -> np.ndarray:
    """Tile per-panel frames horizontally on a common background."""
    height = max(f.shape[0] for f in frames)
    padded: list[np.ndarray] = []
    for f in frames:
        if f.shape[0] < height:
            pad = np.zeros((height - f.shape[0], f.shape[1], 3), dtype=np.uint8)
            pad[:] = PANEL_BG
            f = np.concatenate([f, pad], axis=0)
        padded.append(f)
    gap = np.zeros((height, GAP, 3), dtype=np.uint8)
    gap[:] = PANEL_BG
    parts: list[np.ndarray] = [gap]
    for f in padded:
        parts.append(f)
        parts.append(gap)
    strip = np.concatenate(parts, axis=1)
    band = np.zeros((GAP, strip.shape[1], 3), dtype=np.uint8)
    band[:] = PANEL_BG
    return np.concatenate([band, strip, band], axis=0)


def cell_for(board: BoardSpec, target: int) -> int:
    """Pixel cell size giving each board roughly target pixels tall."""
    return max(6, target // max(board.width, board.height))


def main() -> None:
    """Render the side-by-side capability-frontier GIF."""
    parser = argparse.ArgumentParser(description="Capability demo GIF")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--boards", type=str, default="8x8,12x12,16x16,20x20")
    parser.add_argument("--seed", type=int, default=7_777)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--max-steps", type=int, default=6_000)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--target-px", type=int, default=300)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.ckpt, weights_only=False)
    config = TrainConfig.model_validate(payload["config"])
    net = build_network(config)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)

    boards = []
    for spec in args.boards.split(","):
        w, h = (int(x) for x in spec.split("x"))
        boards.append(BoardSpec(width=w, height=h))

    panel_frames: list[list[np.ndarray]] = []
    summary: list[tuple[BoardSpec, str, float]] = []
    for i, board in enumerate(boards):
        policy = RLPolicy(
            net,
            config.obs,
            config.action,
            canvas=max(board.width, board.height),
            device=device,
        )
        frames, outcome, fill = roll_panel(
            board,
            policy,
            seed=args.seed + i,
            cell=cell_for(board, args.target_px),
            n_frames=args.frames,
            max_steps=args.max_steps,
        )
        panel_frames.append(frames)
        summary.append((board, outcome, fill))

    composed = [
        compose([panel_frames[p][t] for p in range(len(boards))])
        for t in range(args.frames)
    ]
    # Hold the final frame so the outcome banners are readable.
    composed.extend([composed[-1]] * args.fps)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, composed, fps=args.fps, loop=0)
    size_mb = args.out.stat().st_size / 1e6
    print(f"{args.out} ({len(composed)} frames, {size_mb:.1f} MB)")
    for board, outcome, fill in summary:
        print(f"  {board.width}x{board.height}: fill {fill:.0%} {outcome}")


if __name__ == "__main__":
    main()
