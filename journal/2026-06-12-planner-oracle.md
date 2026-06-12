# Lesson: rebuild the oracle before anything learns

Date: 2026-06-12

The handoff promised an included planner package; it was not anywhere
in the handoff. Rebuilding it first, before any RL code, turned out to
be the right order: the quoted reference table acted as an acceptance
test for the rebuilt oracle and, transitively, for the new vectorized
environment's rules (tail-vacates-first, fruit spawning, win
detection), since the planner only wins if the rules are right.

Numbers, 100 episodes per board, seed 101 (ledger run aa155ef621):

| Board | Pure (ours) | Pure (quoted) | Shortcut (ours) | Shortcut (quoted) |
| ----- | ----------- | ------------- | --------------- | ----------------- |
| 8x8   | 15.8        | 16.1          | 10.4            | 14.8              |
| 10x10 | 25.2        | 26.0          | 15.1            | 22.2              |
| 12x12 | 36.2        | 34.9          | 20.7            | 33.8              |
| 16x16 | 64.1        | 66.4          | 34.9            | 60.4              |

All cells 100 percent fill, 100 percent wins.

Two things mattered for the shortcut policy's gap to the quoted one:
jumping the maximum safe distance toward the fruit each step rather
than a fixed small skip, and the clean safety argument (body occupies
forward cycle distances [dist(tail), N), so any jump short of the
tail with a growth margin is provably collision-free). Efficiency did
not cost the win guarantee because shortcuts turn off at 50 percent
fill.

Watch-out recorded for later: the row0 init trick only makes the body
cycle-consistent on even-row boards. The planner refuses odd row
counts; generalization boards for the RL agent are free to be odd,
but oracle gaps can only be quoted where the oracle exists.
