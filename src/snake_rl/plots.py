"""Learning-curve and oracle-gap plots from the results ledger.

Run with:
    uv run python -m snake_rl.plots --out plots
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snake_rl.ledger import DEFAULT_LEDGER, LedgerRow, read_rows


def eval_curves(
    rows: list[LedgerRow], run_name: str
) -> dict[str, dict[int, list[tuple[int, float, float]]]]:
    """Collect evaluation curves for a run.

    Args:
        rows: All ledger rows.
        run_name: Training run to select.

    Returns:
        board -> seed -> list of (frames, steps_per_apple, fill).
    """
    curves: dict[str, dict[int, list[tuple[int, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row.run_name != run_name or row.kind not in (
            "eval",
            "eval_final",
        ):
            continue
        board = str(row.extra.get("board", "?"))
        curves[board][row.seed].append(
            (
                row.frames,
                row.metrics.get("steps_per_apple_mean", float("nan")),
                row.metrics.get("fill_mean", float("nan")),
            )
        )
    return curves


def oracle_reference(
    rows: list[LedgerRow],
) -> dict[str, float]:
    """Return the latest shortcut-planner steps-per-apple per board."""
    ref: dict[str, float] = {}
    for row in rows:
        if row.kind == "planner" and "shortcut" in row.run_name:
            board = str(row.extra.get("board", "?"))
            ref[board] = row.metrics["steps_per_apple_mean"]
    return ref


def plot_run(rows: list[LedgerRow], run_name: str, out_dir: Path) -> list[Path]:
    """Write steps-per-apple and fill curves for one run.

    Args:
        rows: All ledger rows.
        run_name: Training run to plot.
        out_dir: Output directory.

    Returns:
        Paths of written figures.
    """
    curves = eval_curves(rows, run_name)
    oracle = oracle_reference(rows)
    if not curves:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for metric_idx, (metric, label) in enumerate(
        [(1, "steps per apple"), (2, "board fill")]
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors = plt.get_cmap("tab10")
        for b_idx, (board, seeds) in enumerate(sorted(curves.items())):
            series = []
            for points in seeds.values():
                pts = sorted(points)
                series.append(
                    (
                        np.array([p[0] for p in pts], dtype=float),
                        np.array([p[metric] for p in pts], dtype=float),
                    )
                )
            frames = series[0][0]
            values = np.stack([s[1] for s in series])
            mean = values.mean(axis=0)
            color = colors(b_idx)
            ax.plot(frames / 1e6, mean, label=board, color=color)
            if values.shape[0] > 1:
                spread = values.std(axis=0)
                ax.fill_between(
                    frames / 1e6,
                    mean - spread,
                    mean + spread,
                    alpha=0.2,
                    color=color,
                )
            if metric_idx == 0 and board in oracle:
                ax.axhline(
                    oracle[board],
                    color=color,
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.7,
                )
        ax.set_xlabel("frames (millions)")
        ax.set_ylabel(label)
        title = f"{run_name}: {label}"
        if metric_idx == 0:
            title += " (dashed: shortcut planner)"
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        path = out_dir / f"{run_name}-{label.replace(' ', '-')}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def main() -> None:
    """Plot every training run found in the ledger."""
    parser = argparse.ArgumentParser(description="Ledger plots")
    parser.add_argument("--out", type=Path, default=Path("plots"))
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()
    rows = read_rows(args.ledger)
    run_names = sorted(
        {r.run_name for r in rows if r.kind in ("eval", "eval_final")}
    )
    for run_name in run_names:
        for path in plot_run(rows, run_name, args.out):
            print(path)


if __name__ == "__main__":
    main()
