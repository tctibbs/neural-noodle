"""Append-only JSONL results ledger.

The ledger is the authoritative record of every measured result. Each
row carries the config hash, git SHA, seed, and timing so any number in
a plot or paper traces back to an exact configuration and commit.
"""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_LEDGER = Path("results") / "ledger.jsonl"


def git_sha() -> str:
    """Return the current git SHA, or "unknown" outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return out.stdout.strip()


class LedgerRow(BaseModel):
    """A single measured result.

    Attributes:
        kind: Row type, for example train, eval, or baseline.
        run_id: Run identifier shared with the log file.
        run_name: Human-readable run label.
        config_hash: Hash of the exact configuration.
        git_sha: Commit the code ran at.
        seed: Seed for this measurement.
        timestamp: UTC time the row was written.
        frames: Environment steps consumed at measurement time.
        wall_clock_s: Wall-clock seconds elapsed at measurement time.
        metrics: Measured values, for example steps_per_apple.
        extra: Anything else worth keeping with the row.
    """

    kind: str
    run_id: str
    run_name: str
    config_hash: str
    git_sha: str = Field(default_factory=git_sha)
    seed: int
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    frames: int = 0
    wall_clock_s: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


def append_row(row: LedgerRow, path: Path = DEFAULT_LEDGER) -> None:
    """Append a row to the ledger.

    Args:
        row: The result row.
        path: Ledger file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row.model_dump(mode="json")) + "\n")


def read_rows(path: Path = DEFAULT_LEDGER) -> list[LedgerRow]:
    """Read all rows from the ledger.

    Args:
        path: Ledger file path.

    Returns:
        All rows in write order. Empty when the file does not exist.
    """
    if not path.exists():
        return []
    rows: list[LedgerRow] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(LedgerRow.model_validate_json(line))
    return rows
