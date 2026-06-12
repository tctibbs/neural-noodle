# Lesson: the GPU env is launch-bound, so the batch is the lever

Date: 2026-06-12

Built the torch tensor environment (ADR 0005) and pinned it to the
numpy reference with lockstep tests: identical pre-states, identical
actions, exact event and state equality every step across hundreds of
randomized transitions, plus observation parity in all four obs
configurations.

Two lessons.

First, a debugging one: torch.from_numpy shares memory with the
source array, and .to(device) is a no-op on CPU, so the test helper
that loaded numpy state into the tensor env initially made the two
environments share state and silently corrupt each other. The
lockstep test caught it on step one. Any state-copy helper that
crosses the numpy-torch boundary needs an explicit copy.

Second, the performance one: at the batch size tuned for the numpy
env (1024), the tensor env trains no faster than numpy (about 25k
steps per second) because a Snake step is roughly fifty tiny kernels
and the loop is kernel-launch-bound, not compute-bound. Raw
step-plus-observe scaling (measured while a training run contended
for the GPU, so absolute numbers are pessimistic):

| Batch | Throughput | ms per step |
| ----- | ---------- | ----------- |
| 1024  | 34k/s      | 30.0        |
| 4096  | 180k/s     | 22.7        |
| 16384 | 718k/s     | 22.8        |

Wall time per step is flat, so frames per second scale linearly with
batch size. The tensor backend should run with 8k to 16k environments
and a shorter rollout to keep the PPO batch reasonable. The numpy
backend stays the right choice at small batch sizes, for evaluation,
and on machines without a usable GPU.
