# Result: one policy, eleven boards, planner-beating efficiency

Date: 2026-06-12

gen-mixed-tensor, 500M frames in 3.2 hours on the tensor backend (43k
steps per second sustained at canvas 16, 16384 envs). Final
100-episode protocol on eleven boards, full table in
docs/paper/results.md.

What I take away.

Geometry diversity is free on the home board and buys the transfer
envelope. The mixed policy wins 92 percent on 10x10 versus 78 +- 4
for the three single-board seeds at 150M frames each. I expected a
specialist-generalist tradeoff; there is none at this scale. The
held-out boards inside the training canvas behave exactly like
trained boards of the same area, including 9x12, an odd-width
geometry where no even-row Hamiltonian construction exists for half
the orientations.

Extrapolation past the training canvas fails gracefully now, not
catastrophically. The single-board policy looped itself to
starvation beyond 1.4x its training area. The mixed policy at 18x18
fills 49 percent at steps-per-apple 18 versus the oracle's 43, and
even 24x24 (2.25x the largest trained area) eats 130 apples per
episode. The remaining gap is endgame fill on big boards, which
looks like a frames problem as much as a method problem: 16x16 fill
was still climbing monotonically at 500M.

Next lever if the owner wants it: train with canvas 24 boards in the
mix (the tensor backend makes this affordable), or curriculum from
small to large. Also worth a second seed of gen-mixed-tensor before
publishing the comparison against the single-board seeds.
