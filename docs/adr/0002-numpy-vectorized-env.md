# ADR 0002: Single-process numpy-vectorized environment

Date: 2026-06-12

Status: accepted

## Context

The legacy training loop stepped one Python-object environment at a
time, rendered every step, and capped near 1k steps/sec. The research
plan needs orders of magnitude more throughput on a Windows machine
with an RTX 5080. Options considered: a GPU-batched tensor env, numpy
with subprocess workers, WSL2 for Linux-only tooling (EnvPool has no
Snake anyway), or a single-process numpy env with a batch dimension.

## Decision

One canonical batched simulator in numpy (snake_rl.env.vec_env), flat
arrays over a batch dimension, ring-buffer bodies, occupancy grids,
and sparse Python loops only for rare events (fruit respawn, resets).
Absolute directions at the simulator boundary; egocentric action
spaces map to absolute outside it.

Reasons:
- Debuggability and determinism beat raw speed while the science is
  unsettled. Tensorized control flow hides rule bugs; a wrong rule
  invalidates every measurement against the planner.
- Windows-native, no WSL2 or JAX friction, no subprocess pickling.
- Snake steps are cheap; the batch dimension recovers the needed
  throughput for PPO, and the GPU is reserved for the network.
- The planner, baselines, and the agent all step the same
  implementation, so efficiency comparisons are apples to apples.

## Consequences

If profiling shows training is env-bound at scale, the inner step can
be ported to torch on GPU behind the same batched interface. Episode
events (eat, death, win, starvation) come back as arrays; reward
computation stays outside the simulator so reward ablations never
touch the rules.
