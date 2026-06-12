"""Markdown results summary from the ledger.

Run with:
    uv run python -m snake_rl.report
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from snake_rl.ledger import DEFAULT_LEDGER, LedgerRow, read_rows


def latest_by_key(
    rows: list[LedgerRow], kinds: tuple[str, ...]
) -> dict[tuple[str, str, int], LedgerRow]:
    """Keep the last row per (run_name, board, seed) for given kinds."""
    out: dict[tuple[str, str, int], LedgerRow] = {}
    for row in rows:
        if row.kind not in kinds:
            continue
        board = str(row.extra.get("board", "?"))
        out[(row.run_name, board, row.seed)] = row
    return out


def fmt(values: list[float]) -> str:
    """Format mean and spread across seeds."""
    arr = np.array(values, dtype=float)
    if len(arr) == 1:
        return f"{arr[0]:.2f}"
    return f"{arr.mean():.2f}+-{arr.std():.2f}"


def main() -> None:
    """Print per-run, per-board summaries grouped across seeds."""
    parser = argparse.ArgumentParser(description="Ledger report")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--kinds",
        type=str,
        default="eval_final,planner,baseline",
        help="Comma-separated row kinds to include.",
    )
    args = parser.parse_args()
    kinds = tuple(args.kinds.split(","))
    rows = read_rows(args.ledger)
    latest = latest_by_key(rows, kinds)

    grouped: dict[tuple[str, str], list[LedgerRow]] = defaultdict(list)
    for (run_name, board, _seed), row in latest.items():
        grouped[(run_name, board)].append(row)

    header = (
        f"| {'run':24s} | {'board':7s} | {'seeds':5s} | "
        f"{'steps/apple':13s} | {'fill':11s} | {'win':9s} |"
    )
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for (run_name, board), group in sorted(grouped.items()):
        spa = [r.metrics["steps_per_apple_mean"] for r in group]
        fill = [r.metrics["fill_mean"] for r in group]
        win = [r.metrics["win_rate"] for r in group]
        print(
            f"| {run_name:24s} | {board:7s} | {len(group):5d} | "
            f"{fmt(spa):13s} | {fmt(fill):11s} | {fmt(win):9s} |"
        )


if __name__ == "__main__":
    main()
