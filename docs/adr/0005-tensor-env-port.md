# ADR 0005: Device-agnostic tensor environment port

Date: 2026-06-12

Status: proposed (build after the current campaign finishes, so all
compared runs in flight share one code path)

## Context

Measured on the RTX 5080 box: the grid-observation PPO configuration
trains at about 34k env steps per second, while the features9
ablation, which shares the identical environment and update loop but
skips grid observation building and the conv network, runs at 311k.
The simulator is therefore not the bottleneck; the cost sits in
CPU-side grid observation assembly, the per-step host-to-device round
trip, and the conv forward. The owner also runs the repo on a Mac, so
a CUDA-only solution would split the codebase.

## Decision

Port the environment step and observation build to pure torch tensor
operations on a single device, behind the same batched interface as
the numpy implementation:

- State lives on device: body ring buffers, head and tail indices,
  occupancy planes, fruit positions, directions, counters. All
  updates are masked tensor ops; no data-dependent Python branches.
- Fruit respawn: sample uniformly over free cells with a masked
  cumulative-sum trick or torch.multinomial over the flattened free
  mask, only for envs that ate (masked write).
- Observations are scattered directly into a device tensor; the
  egocentric rotation uses torch.rot90 on direction-grouped slices or
  precomputed index permutations. No numpy, no transfer.
- Actions stay on device end to end; the only periodic host sync is
  draining finished-episode statistics once per rollout.
- Device selection: cuda, then mps, then cpu. The same code runs on
  the 5080 box and on Apple Silicon (torch mac wheels ship MPS).
  Notable MPS caveats to handle: prefer int32 over int64 tensors and
  avoid ops MPS lacks (none required by this design).

The numpy environment stays as the reference implementation. It
remains the substrate for the planner oracle, the unit tests, video
rendering, and the fixed evaluation protocol (evaluation cost is
negligible). Training is the only consumer of the tensor env.

## Verification plan (before any trained result counts)

1. Rules equivalence: drive both envs in lockstep with identical
   action sequences and identical injected fruit positions; assert
   identical occupancy, lengths, deaths, wins, and starvation at
   every step over long random rollouts and adversarial cases
   (tail-chase, wall hugs, win on final cell).
2. Planner end-to-end: run both planner policies through a CPU
   adapter on the tensor env and require 100 percent wins, matching
   the numpy reference behavior.
3. Determinism: fixed seed, fixed actions, identical trajectories
   across two constructions on the same device.

## Expected payoff and cost

Estimated 5x to 15x training throughput on the 5080 (200k to 500k
steps per second), turning a 300M-frame run into roughly 20 to 40
minutes, and enabling native 16x16+ training at acceptable cost. On
MPS the win is smaller but the port is what makes GPU training on the
Mac possible at all. Estimated effort: roughly a day including the
equivalence suite. Risk is contained because the numpy reference and
the planner acceptance tests pin the rules.
