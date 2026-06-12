# Result: seed 0 beats the oracle on steps-per-apple at high fill

Date: 2026-06-12

main-10x10 seed 0, 150M frames, final checkpoint, fixed protocol
(100 episodes per board, board-exact canvas). Trained only on 10x10;
every other board is zero-shot transfer.

| Board | Agent spa | Oracle spa | Agent fill | Agent win | Oracle win |
| ----- | --------- | ---------- | ---------- | --------- | ---------- |
| 8x8   | 6.7       | 10.4       | 96.5%      | 88%       | 100%       |
| 10x10 | 8.9       | 15.1       | 90.8%      | 73%       | 100%       |
| 12x12 | 11.7      | 20.7       | 76.5%      | 44%       | 100%       |
| 16x16 | 70.6      | 34.9       | 25.6%      | 0%        | 100%       |

Three lessons.

First, the efficiency claim has teeth but needs careful framing. On
8x8 the agent reaches 96.5 percent fill while averaging 6.7 steps per
apple against the shortcut planner's 10.4 at 100 percent fill. The
agent goes straight at fruit early and tolerates risk the planner
provably avoids; it pays with the 12 percent of games it fails to
finish. Steps-per-apple alone favors policies that die before the
expensive endgame apples, so every efficiency number must be reported
jointly with fill and win rate, or compared at matched fill.

Second, near-size transfer is strong and far-size transfer is not.
Plus or minus 20 percent board area costs little; 2.5x the area
collapses (16x16: a quarter fill, no wins, and erratic
steps-per-apple from episodes that loop). The mixed-geometry run
exists to close exactly this gap.

Third, 8x8 transfer (88 percent wins) is better than the 10x10
training board itself (73 percent). Smaller boards are easier to
finish, and the policy's local strategy compresses well; capability
does not monotonically peak at the training size.
