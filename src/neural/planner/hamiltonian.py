"""Hamiltonian cycle construction and verification for grid boards.

A grid graph has a Hamiltonian cycle only when at least one side is
even (the grid is bipartite, so an odd-by-odd board has an odd number
of cells and no closed tour). The construction here requires an even
number of rows and transposes internally when only the column count is
even.
"""

import numpy as np


def hamiltonian_cycle(height: int, width: int) -> np.ndarray:
    """Build a Hamiltonian cycle over a height x width grid.

    The pattern runs east along row 0, snakes through rows 1 to the
    bottom inside columns 1 and up, then returns north along column 0.

    Args:
        height: Number of rows.
        width: Number of columns.

    Returns:
        Array of shape (height * width, 2) of (row, col) cells in
        cycle order. Consecutive cells are grid-adjacent and the last
        cell is adjacent to the first.

    Raises:
        ValueError: If both sides are odd, or either side is below 2.
    """
    if height < 2 or width < 2:
        msg = f"board {height}x{width} is too small for a cycle"
        raise ValueError(msg)
    if height % 2 == 1 and width % 2 == 1:
        msg = f"board {height}x{width} has no Hamiltonian cycle"
        raise ValueError(msg)
    if height % 2 == 1:
        flipped = hamiltonian_cycle(width, height)
        return flipped[:, ::-1].copy()

    cells: list[tuple[int, int]] = [(0, c) for c in range(width)]
    for r in range(1, height):
        if r % 2 == 1:
            cols = range(width - 1, 0, -1)
        else:
            cols = range(1, width)
        cells.extend((r, c) for c in cols)
    cells.extend((r, 0) for r in range(height - 1, 0, -1))
    return np.array(cells, dtype=np.int64)


def cycle_order(cycle: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return each cell's index along the cycle.

    Args:
        cycle: Cycle cells in order, shape (height * width, 2).
        height: Number of rows.
        width: Number of columns.

    Returns:
        Array of shape (height, width) mapping cell to cycle index.
    """
    order = np.zeros((height, width), dtype=np.int64)
    order[cycle[:, 0], cycle[:, 1]] = np.arange(len(cycle))
    return order


def verify_cycle(cycle: np.ndarray, height: int, width: int) -> None:
    """Check that a cycle is a single closed Hamiltonian tour.

    Args:
        cycle: Candidate cycle, shape (height * width, 2).
        height: Number of rows.
        width: Number of columns.

    Raises:
        ValueError: If any cell is missing or repeated, any
            consecutive pair is not grid-adjacent, or the tour does
            not close.
    """
    n = height * width
    if len(cycle) != n:
        msg = f"cycle has {len(cycle)} cells, expected {n}"
        raise ValueError(msg)
    seen = set(map(tuple, cycle.tolist()))
    if len(seen) != n:
        msg = "cycle repeats at least one cell"
        raise ValueError(msg)
    diffs = np.abs(np.diff(cycle, axis=0)).sum(axis=1)
    if not (diffs == 1).all():
        msg = "cycle contains a non-adjacent consecutive pair"
        raise ValueError(msg)
    if np.abs(cycle[0] - cycle[-1]).sum() != 1:
        msg = "cycle does not close"
        raise ValueError(msg)
