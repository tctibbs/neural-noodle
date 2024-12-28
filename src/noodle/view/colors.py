"""Colors module.

This module defines color constants used in rendering.
"""

from enum import Enum


class Colors(Enum):
    """Enum class for colors."""

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    GRAY = (64, 64, 64)
