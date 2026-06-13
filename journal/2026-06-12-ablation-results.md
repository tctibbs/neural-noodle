# Result: representation is the driver, reward detail is not

Date: 2026-06-12

Six one-variable ablations against the main configuration, 50M
frames, one seed each, 100-episode final eval on 10x10. The
comparable main-10x10 reference at the same 50M budget sat at about
64 percent fill, steps-per-apple about 9.9, no wins yet.

| Variant changed            | Fill | Steps/apple | Verdict |
| -------------------------- | ---- | ----------- | ------- |
| features9 (legacy obs)     | 8%   | 173         | catastrophic, near random |
| absolute action space      | 23%  | 18.2        | severe degradation |
| allocentric grid           | 45%  | 13.5        | clear degradation |
| binary body plane          | 70%  | 11.2        | par fill, worse efficiency |
| no step cost (sparse)      | 68%  | 10.8        | par at this budget |
| potential shaping on       | 68%  | 10.5        | par at this budget |

Reading. The efficiency engine is the representation pairing: a full
egocentric grid with turn-relative actions. Remove the egocentric
frame and learning slows hard; remove the spatial observation
entirely (the legacy 9 features) and the agent cannot even forage
reliably, confirming the handoff's suspicion that the old
observation was the legacy stack's binding constraint. The body
time-to-vacate encoding pays specifically in steps-per-apple (11.2
versus 9.9 binary versus decay), consistent with the decay channel
enabling tighter routes through vacating space.

The reward-side levers moved little at this budget. Caveats: one
seed each, and 50M frames may be too early for the step-cost and
shaping effects to express, since the main run's biggest efficiency
gains landed after 60M.

Update (150M rerun): the sparse-reward ablation was rerun at the
full main budget to settle this. On 10x10 it reached steps-per-apple
9.08, fill 95 percent, win 76 percent, against the step-cost main
run's 8.94 / 93 percent / 78 percent (+- 4 percent across seeds).
The step cost is dispensable: efficiency comes from the fruit reward
and from death ending the future-reward stream, not from the
per-step penalty. This is the more interesting finding, since it
means the efficiency result is not an artifact of reward
engineering. The step cost stays in the main config as a harmless
mild regularizer, but the headline does not depend on it.
