# ADR 0001: Astral toolchain on uv, Ruff, and Ty

Date: 2026-06-12

Status: accepted

## Context

The repo used Poetry, had no type checker, and pinned nothing tightly.
The owner's engineering bar mandates the Astral toolchain. The target
machine turned out to carry an RTX 5080 (Blackwell), not the 4080 the
handoff mentioned, which constrains the torch build.

## Decision

uv manages the environment and lockfile, Ruff lints and formats, Ty
type-checks. Python is pinned to 3.12. Torch installs from the cu128
wheel index because Blackwell GPUs need CUDA 12.8 or newer kernels;
torch resolves to 2.11.0+cu128 and sees the GPU. poetry.lock is
removed; uv.lock is the single source of dependency truth.

## Consequences

Anyone reproducing results needs uv, nothing else. CI runs ruff check,
ruff format --check, ty check, and pytest. The legacy src/neural
package predates these standards and will be removed rather than
retrofitted (the learning side is rebuilt in src/snake_rl).
