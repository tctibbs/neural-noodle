# ADR 0003: Planner oracle rebuilt from specification

Date: 2026-06-12

Status: accepted

## Context

The handoff describes a verified planner package with a reference
table of steps-per-apple numbers, but the package itself was not in
the repository, any branch, or the handoff materials. The planner is
the measuring stick for the whole study, so it was rebuilt from the
specification and verified against the quoted table.

## Decision

neural.planner constructs a boustrophedon Hamiltonian cycle (east
along row 0, snake through the remaining rows in columns 1 and up,
return north along column 0). Even row counts are required; an
odd-by-odd grid has no Hamiltonian cycle at all. Every cycle passes a
single-closed-tour verification at policy construction.

Two policies follow the cycle:
- Pure: always advance one cycle position. Guaranteed win.
- Shortcut: jump forward along the cycle toward the fruit when
  provably safe. Body cells occupy forward cycle distances in
  [dist(tail), N), so a jump of distance d is safe when
  d <= dist(tail) - margin (margin 4 covers growth) and d never
  overshoots the fruit (d <= dist(fruit)). Shortcuts disable above 50
  percent fill, falling back to pure following, so the win guarantee
  is preserved.

Evaluation inits the snake on row 0 facing right, which is a prefix of
the cycle, so the body-behind-head invariant holds from step one. The
fixed eval protocol uses the same init for every policy.

## Consequences

Measured with 100 episodes per board, seed 101, start length 3
(ledger run aa155ef621): pure matches the handoff table within noise
(15.8 / 25.2 / 36.2 / 64.1 steps-per-apple on 8x8 / 10x10 / 12x12 /
16x16 versus 16.1 / 26.0 / 34.9 / 66.4 quoted), with 100 percent fill
and wins everywhere. The shortcut policy is substantially tighter
than the handoff's conservative cut (10.4 / 15.1 / 20.7 / 34.9 versus
14.8 / 22.2 / 33.8 / 60.4) while keeping the 100 percent win rate,
which already delivers the requested DHCR-style tightening. The
shortcut numbers are the official optimality reference for the study.
