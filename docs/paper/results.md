# Results (draft)

Every number below is a ledger row in `results/ledger.jsonl` (config
hash and git SHA per row). Evaluation: fixed protocol, 100 episodes
per board, eval seed 7777, board-exact canvas. Steps-per-apple (spa)
must be read jointly with fill and win rate: spa alone favors
policies that avoid the expensive endgame apples.

## Main result: 10x10, three seeds, 150M frames each

Agent (mean +- std over seeds) versus references on the trained
board:

| Policy             | spa          | fill        | win  |
| ------------------ | ------------ | ----------- | ---- |
| random baseline    | n/a (0 apples) | 3%        | 0%   |
| survival looper    | 401          | 3%          | 0%   |
| pure planner       | 25.2         | 100%        | 100% |
| shortcut planner   | 15.1         | 100%        | 100% |
| PPO agent (3 seeds)| 8.94 +- 0.08 | 93% +- 2%   | 78% +- 4% |

The agent eats at 0.59x the steps-per-apple of the conservative-safe
shortcut planner while completing 78 percent of games outright. The
efficiency-completion tradeoff is real but small on the trained
board. A rendered win: 97 apples, 100 percent fill, 864 steps; the
shortcut planner finishes the same board in 1388 steps.

Zero-shot transfer of the same three 10x10-trained seeds:

| Board | spa            | fill       | win        | oracle spa |
| ----- | -------------- | ---------- | ---------- | ---------- |
| 8x8   | 6.75 +- 0.01   | 98% +- 1%  | 92% +- 3%  | 10.4       |
| 12x12 | 11.66 +- 0.07  | 77% +- 2%  | 42% +- 3%  | 20.7       |
| 16x16 | 100 +- 64      | 27% +- 3%  | 0%         | 34.9       |

Near-size transfer is strong in both directions; 2.5x board area
collapses into looping behavior. The mixed-geometry campaign
addresses this.

## Ablations: 50M frames, one seed, 10x10

Main-configuration reference at the same 50M budget: about 64
percent fill, spa 9.9, no wins yet.

| Variant                  | fill | spa  |
| ------------------------ | ---- | ---- |
| legacy 9-feature obs     | 8%   | 173  |
| absolute actions         | 23%  | 18.2 |
| allocentric grid         | 45%  | 13.5 |
| binary body plane        | 70%  | 11.2 |
| no step cost             | 68%  | 10.8 |
| potential shaping        | 68%  | 10.5 |

The representation pairing (egocentric grid plus turn actions) is
the dominant driver. The body time-to-vacate channel buys
efficiency specifically (spa 9.9 versus 11.2 at par fill). Reward
details are second-order at this budget; a longer sparse-reward run
is required before claiming the step cost is dispensable, since the
main run's efficiency gains concentrated after 60M frames.

## Tensor backend validation

The torch tensor environment (ADR 0005) is rule-equivalent to the
numpy reference by lockstep test. Trained at 16384 environments with
bf16 autocast it sustains about 101k steps per second on the RTX
5080 (2.9x the numpy backend) and reaches spa 9.12, fill 87 percent,
win 62 percent on 10x10 at equal frames, slightly under the numpy
band, consistent with 4x fewer gradient updates at 4x batch, and
notably better far transfer (45 versus 27 percent fill on 16x16).
Generalization-campaign results use this backend under its own
config hash.

## Generalization campaign

TODO: gen-mixed-tensor results (trained on eight geometries up to
16x16; held-out interpolation boards 6x6, 9x12, 14x14, 16x8;
extrapolation boards 18x18, 20x20, 24x24 beyond the training
canvas). Oracle references for all eleven evaluation boards are in
the ledger (shortcut planner wins 100 percent everywhere, spa 6.6 on
6x6 up to 74.8 on 24x24).
