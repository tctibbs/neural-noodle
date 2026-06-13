"""Tests for Hamiltonian cycle construction."""

import numpy as np
import pytest

from neural.planner.hamiltonian import (
    cycle_order,
    hamiltonian_cycle,
    verify_cycle,
)


@pytest.mark.parametrize(
    ("height", "width"),
    [
        (2, 2),
        (4, 4),
        (8, 8),
        (10, 10),
        (12, 12),
        (16, 16),
        (6, 9),
        (9, 6),
        (2, 16),
        (16, 2),
        (4, 31),
        (31, 4),
    ],
)
def test_cycle_is_valid(height: int, width: int) -> None:
    """Every constructed cycle passes single-cycle verification."""
    cycle = hamiltonian_cycle(height, width)
    verify_cycle(cycle, height, width)


@pytest.mark.parametrize(("height", "width"), [(3, 3), (5, 9), (9, 5)])
def test_odd_odd_rejected(height: int, width: int) -> None:
    """Odd-by-odd boards have no Hamiltonian cycle."""
    with pytest.raises(ValueError, match="no Hamiltonian cycle"):
        hamiltonian_cycle(height, width)


def test_too_small_rejected() -> None:
    """Boards under 2x2 are rejected."""
    with pytest.raises(ValueError, match="too small"):
        hamiltonian_cycle(1, 8)


def test_cycle_order_is_inverse() -> None:
    """cycle_order maps each cell back to its cycle index."""
    cycle = hamiltonian_cycle(8, 8)
    order = cycle_order(cycle, 8, 8)
    for k, (r, c) in enumerate(cycle.tolist()):
        assert order[r, c] == k


def test_verify_rejects_broken_cycle() -> None:
    """Verification catches a swapped pair."""
    cycle = hamiltonian_cycle(8, 8).copy()
    cycle[[3, 10]] = cycle[[10, 3]]
    with pytest.raises(ValueError, match="non-adjacent"):
        verify_cycle(cycle, 8, 8)


def test_row0_is_cycle_prefix() -> None:
    """Row 0 in increasing column order is the start of the cycle.

    The row0 init mode relies on this so the snake starts consistent
    with cycle order.
    """
    for height, width in [(8, 8), (10, 12), (6, 9)]:
        cycle = hamiltonian_cycle(height, width)
        if height % 2 == 0:
            expected = np.array([(0, c) for c in range(width)])
            assert (cycle[:width] == expected).all()
