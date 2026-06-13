# Abstract and introduction (draft)

## Abstract

Snake is trivially solved in the classical sense: a Hamiltonian cycle
over the grid never dies and always fills the board, so survival and
score are not research targets. We instead study efficiency, the
number of steps a policy spends per apple, and generalization across
board geometry, where a hardcoded cycle does not transfer. Against a
near-optimal shortcut-augmented cycle planner that wins every game,
we train PPO agents with an egocentric multi-channel grid observation
and turn-relative actions. On a 10x10 board the learned policy eats
at 8.9 steps per apple (three seeds) versus the planner's 15.1, while
filling 93 percent of the board and winning 78 percent of games
outright. A single policy trained on eight mixed geometries
generalizes to board sizes and aspect ratios held out of training,
winning 82 percent of games on an odd-width board no cycle planner
handles, and degrades gracefully rather than collapsing on boards
larger than any seen in training. Controlled one-variable ablations
isolate the egocentric grid representation as the dominant driver of
efficiency. All results are reproducible from a hashed config and a
results ledger; the planner oracle, both environment backends, and
the agent share one verified rule implementation.

## Introduction

The appeal of Snake as a reinforcement learning benchmark is also its
trap. Because a Hamiltonian cycle solves the game outright, any agent
report framed around survival or high score measures effort already
beaten by a few lines of classical code. A credible contribution has
to be defined relative to optimal behavior, not relative to losing.

We adopt two such framings. The first is efficiency: a cycle follower
wins but wanders, and the published gap between a naive follower and a
shortcut-augmented planner shows there is real room to move on
steps-per-apple. We measure every policy against a shortcut planner
that we verify wins 100 percent of games, and we treat its
steps-per-apple as the optimality reference. The second is
generalization: a Hamiltonian construction is specific to a board
shape (and does not even exist for odd-by-odd grids), so a single
neural policy that holds up across geometries is doing something the
classical solution cannot.

Reporting steps-per-apple alone is misleading, because a policy that
dies before the expensive endgame apples can post a low average. We
therefore always report steps-per-apple jointly with board fill and
win rate, and compare at matched fill where it matters. The contrast
class is anchored on both ends: a random policy and a
survival-by-looping agent below, the planner oracle above.
