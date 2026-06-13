# Limitations and threats to validity (draft)

Grounded in the ledger; no claim here exceeds what was measured.

## Statistical

- The single-board main result is three seeds (mean and spread
  reported). The ablations are one seed each at 50M frames, so their
  rankings are indicative, not significance-tested. The
  representation effects are large enough to read through seed noise
  (legacy observation at 8 percent fill versus 64 percent); the
  reward-side effects are within plausible single-seed noise and are
  not claimed as significant.
- The generalization headline is two seeds, reported as mean and
  spread. The two seeds agree tightly (per-board win-rate spread at
  or below 5 points except the 24x24 extrapolation tail), so the
  transfer pattern is a method property rather than a single run.

## Measurement

- Steps-per-apple is undefined for episodes with zero apples and is
  averaged only over scoring episodes; the count is reported
  alongside so a low average from few survivors is visible. On the
  largest extrapolation board (24x24) this average is noisy because
  some episodes loop, and that row is flagged rather than smoothed.
- Evaluation uses a row0 init (snake on the top row), a strict subset
  of the random-row training distribution and the init the planner
  requires. This is held fixed across every compared policy, so it
  cannot bias a comparison, but absolute fill on a given board could
  differ slightly under a different eval init.

## Method scope

- The planner oracle, and therefore the quoted optimality gap, exists
  only for boards with an even row count; odd-by-odd grids have no
  Hamiltonian cycle. The RL agent is evaluated on odd-width boards
  regardless, but without an oracle column there.
- The shortcut planner is conservative by construction (shortcuts
  disabled above 50 percent fill to preserve the win guarantee). It
  is a strong, safe reference, not a proven optimum; a more
  aggressive planner would raise the bar the agent is measured
  against.
- Extrapolation past the training canvas degrades in fill (49, 39,
  23 percent at 18, 20, 24 versus 100 percent for the planner). The
  16x16 fill curve was still rising at 500M frames, so part of this
  gap is a training-budget limit rather than a method ceiling; this
  is not yet disentangled.

## Reproducibility caveats

- Two environment backends exist. They are lockstep rule-equivalent
  under test, but floating-point and batch-size differences mean the
  numpy and tensor runs are not bit-identical learning curves; they
  are compared at equal frames, not equal trajectories.
- bfloat16 autocast is on for tensor-backend training. Losses and the
  action distribution are computed in float32, but the forward pass
  is reduced precision, which can shift fine behavior versus a full
  float32 run.
