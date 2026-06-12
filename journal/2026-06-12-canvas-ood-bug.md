# Lesson: pad-to-train-canvas is an out-of-distribution trap

Date: 2026-06-12

At the 20M-frame quick eval, the 10x10-trained policy scored 40
apples on 10x10, 46 on 12x12, 42 on 16x16, and exactly zero on 8x8.
The policy was not bad at small boards; the evaluation was lying.

Cause: evaluation clamped the observation canvas to the training
canvas, so the 8x8 board sat inside a 10x10 canvas with a two-cell
interior wall band. The training distribution (board exactly fills
canvas) never contains interior wall bands, so the policy faced a
state it had never seen and starved in place. Larger boards used
their own canvas size and filled it exactly, matching the training
regime, which is why they transferred cleanly.

Fix: evaluation always uses a board-exact canvas (the fully
convolutional network accepts any canvas size). Non-square boards
still produce a wall band on the square canvas by necessity; the
mixed-geometry training run includes non-square boards, so that
policy trains on wall bands.

Wider lesson for the paper: any cross-size transfer claim must hold
the canvas convention fixed between training and evaluation, or the
measured gap is an artifact of representation, not capability.
Seed 0 of main-10x10 ran its in-training quick evals under the buggy
convention (its 8x8 points before 30M frames are invalid); its final
numbers come from a post-hoc checkpoint evaluation under the fixed
convention.
