# Lesson: one training process beats two on this box

Date: 2026-06-12

Measured throughput for the main PPO config (1024 envs, canvas 10,
64-channel trunk) on the RTX 5080 machine: about 36k env steps per
second for a single process. The legacy loop managed about 1k, so the
rebuild bought a factor of roughly 35 before any GPU-side env work.

Tried running two seeds concurrently. Combined throughput rose only
to about 45k (seed A dropped to 30k, seed B crawled at 15k), so the
second process pays far more than it earns: sequential runs finish
both seeds sooner than concurrent ones. The bottleneck is the
CPU-side numpy stepping and observation building plus a per-step GPU
sync, not GPU compute, and two processes thrash the same cores.
Decision: a sequential run queue.

Also visible at 10M frames: the 10x10-trained policy already gets 22
apples per episode on 16x16 zero-shot. The fully convolutional
pooled-head network transfers across canvas sizes out of the box,
which bodes well for the generalization study.

If throughput ever becomes the binding constraint, the next lever is
porting the env step and observation build to torch on GPU behind the
same batched interface (ADR 0002 anticipated this), not process-level
parallelism.
