![Repository Banner](social-preview.jpg)

# Neural Noodle

Efficiency-oriented reinforcement learning for Snake, measured against
a near-optimal classical planner.

## The capability frontier

![One policy across four board sizes](docs/demo.gif)

One policy (trained on mixed geometries up to 16x16) playing four
board sizes at once. It fills and wins the small boards efficiently,
and the limit is visible on the right: beyond its training scale it
stays direct toward fruit but stalls before completing the board
(16x16 at 71 percent fill, 20x20 at 22 percent). Playback is
time-normalized, so the larger boards, whose real episodes are much
longer, are shown sped up; each panel reports its true final outcome.

## Why efficiency, not score

Snake has a trivial unbeatable classical solution: follow a
Hamiltonian cycle and the board always fills. Survival and raw score
are therefore non-results. This project asks a harder question: how
close can a learned policy get to planner-level efficiency
(steps-per-apple, fraction of board filled), and does a single policy
generalize across board geometries where a hardcoded cycle does not
transfer?

Three reference points anchor every result:

- Planner oracle (upper bound): a Hamiltonian cycle follower with
  provably safe shortcuts. Wins every game on every tested board.
- Random policy (floor): dies almost immediately.
- Survival looper (failure mode): circles forever without seeking
  fruit, embodying the survival-by-looping local optimum.

## Layout

- `src/snake_rl`: the research package. Vectorized numpy Snake
  simulator, Hamiltonian planner oracle, PPO agent, fixed evaluation
  protocol, JSONL results ledger, plotting, and video rendering.
- `src/noodle`: the original human-playable Pygame game
  (`uv run python main.py`).
- `configs`: every experiment as a validated, hashed YAML config.
- `docs/adr`: architecture decision records.
- `journal`: experiment journal, one lesson per file.
- `results/ledger.jsonl`: the authoritative record of every measured
  number, keyed by config hash and git SHA.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest tests -q
uv run python -m snake_rl.eval_planner     # oracle reference numbers
uv run python -m snake_rl.train --config configs/main-10x10.yaml
uv run python -m snake_rl.plots --out plots
```

Checks: `uv run ruff check src/snake_rl tests`,
`uv run ruff format --check src/snake_rl tests`,
`uv run ty check src/snake_rl`.

## License

MIT. See [LICENSE](LICENSE).
