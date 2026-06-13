# Methods (draft)

All numbers in this document trace to rows in
`results/ledger.jsonl`, keyed by config hash and git SHA. Sections
marked TODO are filled only after the corresponding ledger rows
exist.

## Problem framing

Snake on a w x h grid admits a trivial classical solution: a
Hamiltonian cycle visits every cell, so a follower never dies and
always fills the board. Survival and score are therefore solved
problems. We study efficiency, measured as steps-per-apple over an
episode, and completion, measured as the fraction of the board
occupied when the episode ends, relative to a near-optimal planner,
together with generalization of a single policy across board
geometries that a hardcoded cycle does not transfer to.

## Environment

A batched Snake simulator with standard rules: the snake advances one
cell per step, grows by one on eating, the fruit respawns uniformly
on a free cell, and the episode ends on wall or body collision, on
filling the board, or after 4 x cells steps without eating
(starvation cap). The tail cell vacates before head collision is
checked. One implementation of the rules is shared by the planner,
the baselines, and the learned agent.

## Reference policies

- Pure cycle follower: always advances one position along a
  boustrophedon Hamiltonian cycle. Guaranteed win, inefficient early.
- Shortcut cycle follower (oracle): jumps forward along the cycle
  toward the fruit when the jump provably cannot reach the body
  (body cells occupy forward cycle distances in [dist(tail), N); a
  jump of distance d is taken only if d <= dist(tail) - 4 and
  d <= dist(fruit)). Shortcuts disable above 50 percent fill. Wins
  every evaluated game on every evaluated board.
- Random policy and a survival looper (circles a 2x2 loop without
  seeking fruit) anchor the bottom of the scale.

## Agent

PPO with GAE and a clipped objective over 1024 parallel
environments. Observations are four planes on a square canvas (body
as time-to-vacate, head, fruit, wall padding), rotated so the snake
faces up. Actions are egocentric (turn left, straight, turn right).
Reward: +1 fruit, -1 death, +10 win, -0.01 per step; no distance
shaping in the main configuration. The network is a fully
convolutional residual trunk (64 channels, 4 blocks) with mean and
max pooled policy and value heads, making the weights independent of
canvas size; the same parameters run on any board geometry.

## Evaluation protocol

Fixed across all policies: one frozen environment per episode, snake
initialized on row 0 facing right (consistent with the planner's
cycle order), start length 3, starvation factor 4, 100 episodes per
board, evaluation seed 7777. Reported: mean and standard deviation
of steps-per-apple (episodes with at least one apple), mean fill,
win, death, and starvation rates. Training uses random-row inits; the
row0 eval init is a strict subset of the training distribution.

## Experiments

All measured results are in results.md and trace to the ledger.

- Main: three seeds on 10x10, 150M frames each, numpy backend.
- Ablations, one variable at a time against the main configuration,
  50M frames: legacy 9-feature observation, binary body plane,
  allocentric observation, absolute action space, no step cost,
  potential-based fruit-distance shaping.
- Generalization: one policy trained on eight mixed geometries
  (8x8 through 16x16, mixed aspect ratios), 500M frames on the
  tensor backend, evaluated on held-out 6x6, 9x12, 14x14, and 16x8
  plus extrapolation boards 18x18, 20x20, and 24x24 beyond the
  training canvas.

## Environment backends

Two rule-identical simulators share one specification: a numpy
reference (also the substrate for the planner, baselines, and all
evaluation) and a torch tensor backend whose step and observation
build run entirely on the device. Lockstep tests assert identical
events and state from identical pre-states across randomized
rollouts, and identical observations in every observation
configuration. The tensor backend trains with 16384 environments,
bfloat16 autocast, and channels-last layout.

## Hardware and reproducibility

Windows 11, RTX 5080 (16 GB), torch 2.11 cu128, Python 3.12, uv
lockfile. Numpy backend: about 36k environment steps per second at
1024 envs. Tensor backend: about 101k at canvas 10 and 43k at
canvas 16. Every run stamps its config hash, git SHA, seed, frame
count, and wall clock into the ledger.
