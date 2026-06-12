# ADR 0004: PPO with an egocentric grid and turn actions

Date: 2026-06-12

Status: accepted

## Context

The learning side needed an algorithm, an observation, an action
space, and a reward. All four must be ablatable, and the headline
metric is efficiency (steps-per-apple, board fill) plus
generalization across board geometry, not raw score.

## Decision

Algorithm: PPO with GAE, clipped objective, advantage normalization,
linear learning rate anneal. On-policy fits a wide vectorized
environment (1024 envs), is robust to reward scale, and has few
moving parts to debug. A value-based agent remains a candidate
algorithmic ablation, not the backbone.

Observation: four board planes on a square canvas (body, head, fruit,
wall padding). Body cells encode time-to-vacate instead of a binary
flag, so the network can plan through space that frees up, which
matters for late-game fill. The canvas rotates so the snake always
faces up (egocentric). The legacy 9-feature vector is kept as an
ablation baseline.

Network: fully convolutional residual trunk with mean and max pooled
heads. Pooling makes the weights canvas-size independent, which is
what lets one policy run on board sizes it never trained on. That is
the generalization mechanism, chosen over padding everything to a
fixed worst-case canvas.

Action space: egocentric turn-left, straight, turn-right. Three
actions instead of four, no wasted reverse move, and consistent with
the rotated observation. Absolute directions stay available as an
ablation.

Reward: fruit +1, death -1, win +10, and a small per-step cost
(0.01) as the efficiency pressure. The step cost makes
survival-by-looping strictly negative without making death
attractive (death also ends the stream of future fruit rewards).
Starvation ends the episode with no extra penalty. Potential-based
shaping over fruit distance exists behind a flag, default off, as an
ablation.

## Consequences

Every choice above is a typed config field, so the ablation suite is
a set of YAML diffs against the main config. The known starvation
subtlety (treated as termination rather than bootstrap-from-timeout)
is consistent across all compared configurations.
