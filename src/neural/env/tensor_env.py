"""Torch tensor Snake environment, stepping entirely on one device.

Implements the same rules as neural.env.vec_env (the reference
implementation) with masked tensor operations: tail vacates before
head collision is checked, fruit respawns uniformly on free cells,
episodes end on collision, board fill, or starvation. Used only for
training throughput; tests pin it against the numpy reference in
lockstep (see tests/test_tensor_env.py).

Device support: cuda, mps, or cpu. The fused observe() builds grid
observations (body decay, head, fruit, wall, optional egocentric
rotation) without leaving the device.
"""

import numpy as np
import torch

from neural.config import EnvConfig, ObsConfig
from neural.env.vec_env import EpisodeStats, VecSnake

DELTAS = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=torch.long)


class TensorVecSnake:
    """Batched Snake simulator on a torch device.

    Args:
        config: Environment settings.
        obs_config: Observation settings for the fused observe().
        canvas: Square observation canvas size.
        seed: Base seed for the device generator.
        device: Torch device string.
    """

    def __init__(
        self,
        config: EnvConfig,
        obs_config: ObsConfig,
        canvas: int,
        seed: int = 0,
        device: str = "cuda",
    ) -> None:
        self.config = config
        self.obs_config = obs_config
        self.canvas = canvas
        self.device = torch.device(device)
        b = config.num_envs
        self.num_envs = b
        self.max_h = max(s.height for s in config.boards)
        self.max_w = max(s.width for s in config.boards)
        self.capacity = self.max_h * self.max_w
        if canvas < max(self.max_h, self.max_w):
            msg = f"canvas {canvas} smaller than largest board"
            raise ValueError(msg)

        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        dev = self.device
        self.deltas = DELTAS.to(dev)
        self.boards_t = torch.tensor(
            [[s.height, s.width] for s in config.boards],
            dtype=torch.long,
            device=dev,
        )
        self.hw = torch.zeros(b, 2, dtype=torch.long, device=dev)
        self.body = torch.zeros(
            b, self.capacity, 2, dtype=torch.long, device=dev
        )
        self.head_i = torch.zeros(b, dtype=torch.long, device=dev)
        self.tail_i = torch.zeros(b, dtype=torch.long, device=dev)
        self.length = torch.zeros(b, dtype=torch.long, device=dev)
        self.occ = torch.zeros(
            b, self.max_h * self.max_w, dtype=torch.bool, device=dev
        )
        self.fruit = torch.zeros(b, 2, dtype=torch.long, device=dev)
        self.direction = torch.zeros(b, dtype=torch.long, device=dev)
        self.steps = torch.zeros(b, dtype=torch.long, device=dev)
        self.since_fruit = torch.zeros(b, dtype=torch.long, device=dev)
        self.apples = torch.zeros(b, dtype=torch.long, device=dev)
        self._envs = torch.arange(b, device=dev)
        self._slots = torch.arange(self.capacity, device=dev)

        # Finished-episode stat buffers, drained once per rollout.
        cap = 8 * b
        self._stat_cap = cap
        self._stats = torch.zeros(cap, 8, dtype=torch.long, device=dev)
        self._stat_n = torch.zeros(1, dtype=torch.long, device=dev)

        self._reset_mask(torch.ones(b, dtype=torch.bool, device=dev))

    def _reset_mask(self, mask: torch.Tensor) -> None:
        """Re-initialize all environments selected by mask."""
        n = int(mask.sum())
        if n == 0:
            return
        idx = mask.nonzero(as_tuple=True)[0]
        dev = self.device
        pick = torch.randint(
            len(self.boards_t), (n,), generator=self.gen, device=dev
        )
        hw = self.boards_t[pick]
        self.hw[idx] = hw
        start = self.config.start_length

        if self.config.init_mode == "row0":
            rows = torch.zeros(n, dtype=torch.long, device=dev)
        else:
            rows = (
                torch.rand(n, generator=self.gen, device=dev)
                * hw[:, 0]
            ).long()
        col_span = hw[:, 1] - start + 1
        col0 = (
            torch.rand(n, generator=self.gen, device=dev) * col_span
        ).long()

        self.occ[idx] = False
        ks = torch.arange(start, device=dev)
        body_rows = rows[:, None].expand(n, start)
        body_cols = col0[:, None] + ks[None, :]
        self.body[idx, :start, 0] = body_rows
        self.body[idx, :start, 1] = body_cols
        flat = body_rows * self.max_w + body_cols
        self.occ[idx[:, None], flat] = True
        self.tail_i[idx] = 0
        self.head_i[idx] = start - 1
        self.length[idx] = start
        self.direction[idx] = 1
        self.steps[idx] = 0
        self.since_fruit[idx] = 0
        self.apples[idx] = 0
        self._spawn_fruit(idx)

    def _spawn_fruit(self, idx: torch.Tensor) -> None:
        """Place fruit uniformly on a free cell for selected envs."""
        if len(idx) == 0:
            return
        h = self.hw[idx, 0]
        w = self.hw[idx, 1]
        rows = self._slots[None, :] // self.max_w
        cols = self._slots[None, :] % self.max_w
        on_board = (rows < h[:, None]) & (cols < w[:, None])
        free = on_board & ~self.occ[idx]
        choice = torch.multinomial(
            free.float(), 1, generator=self.gen
        ).squeeze(1)
        self.fruit[idx, 0] = choice // self.max_w
        self.fruit[idx, 1] = choice % self.max_w

    @property
    def head(self) -> torch.Tensor:
        """Head cell per env, shape (num_envs, 2)."""
        return self.body[self._envs, self.head_i]

    def step(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Advance every environment by one step and auto-reset.

        Args:
            actions: Absolute directions, shape (num_envs,), on the
                env device. Reverse moves keep the current direction.

        Returns:
            Bool event tensors: ate, died, won, starved, done.
        """
        envs = self._envs
        reverse = (self.direction + 2) % 4
        eff = torch.where(actions == reverse, self.direction, actions)
        self.direction = eff

        new_head = self.head + self.deltas[eff]
        oob = (
            (new_head[:, 0] < 0)
            | (new_head[:, 1] < 0)
            | (new_head[:, 0] >= self.hw[:, 0])
            | (new_head[:, 1] >= self.hw[:, 1])
        )
        nh = new_head.clamp(min=0)
        nh[:, 0] = nh[:, 0].clamp(max=self.max_h - 1)
        nh[:, 1] = nh[:, 1].clamp(max=self.max_w - 1)
        nh_flat = nh[:, 0] * self.max_w + nh[:, 1]
        ate = (
            ~oob
            & (nh[:, 0] == self.fruit[:, 0])
            & (nh[:, 1] == self.fruit[:, 1])
        )

        # Tail vacates first for non-eating envs.
        vacate = ~ate
        tails = self.body[envs, self.tail_i]
        tail_flat = tails[:, 0] * self.max_w + tails[:, 1]
        keep = self.occ[envs, tail_flat] & ~vacate
        self.occ[envs, tail_flat] = keep
        self.tail_i = torch.where(
            vacate, (self.tail_i + 1) % self.capacity, self.tail_i
        )

        hit = self.occ[envs, nh_flat]
        died = oob | hit
        move = ~died

        self.head_i = torch.where(
            move, (self.head_i + 1) % self.capacity, self.head_i
        )
        write = move[:, None].expand(-1, 2)
        cur = self.body[envs, self.head_i]
        self.body[envs, self.head_i] = torch.where(write, nh, cur)
        self.occ[envs, nh_flat] = self.occ[envs, nh_flat] | move
        self.length = self.length + (ate & move).long()
        self.steps = self.steps + move.long()
        self.since_fruit = torch.where(
            ate, torch.zeros_like(self.since_fruit), self.since_fruit + move.long()
        )
        self.apples = self.apples + ate.long()

        cells = self.hw[:, 0] * self.hw[:, 1]
        won = ate & (self.length == cells)
        cap = (self.config.starvation_factor * cells).long()
        starved = move & (self.since_fruit >= cap)
        done = died | won | starved

        respawn = ate & ~won
        self._spawn_fruit(respawn.nonzero(as_tuple=True)[0])

        self._record(done, died, won, starved)
        self._reset_mask(done)
        return {
            "ate": ate,
            "died": died,
            "won": won,
            "starved": starved,
            "done": done,
        }

    def _record(
        self,
        done: torch.Tensor,
        died: torch.Tensor,
        won: torch.Tensor,
        starved: torch.Tensor,
    ) -> None:
        """Append finished-episode stats to the device buffer."""
        idx = done.nonzero(as_tuple=True)[0]
        n = len(idx)
        if n == 0:
            return
        base = self._stat_n[0]
        slots = (base + torch.arange(n, device=self.device)).clamp(
            max=self._stat_cap - 1
        )
        rows = torch.stack(
            [
                self.apples[idx],
                self.steps[idx],
                self.length[idx],
                self.hw[idx, 1],
                self.hw[idx, 0],
                died[idx].long(),
                won[idx].long(),
                starved[idx].long(),
            ],
            dim=1,
        )
        self._stats[slots] = rows
        self._stat_n[0] = (base + n).clamp(max=self._stat_cap)

    def drain_finished(self) -> list[EpisodeStats]:
        """Return and clear finished-episode stats (one host sync)."""
        n = int(self._stat_n.item())
        if n == 0:
            return []
        rows = self._stats[:n].cpu().numpy()
        self._stat_n.zero_()
        return [
            EpisodeStats(
                apples=int(r[0]),
                steps=int(r[1]),
                length=int(r[2]),
                width=int(r[3]),
                height=int(r[4]),
                died=bool(r[5]),
                won=bool(r[6]),
                starved=bool(r[7]),
            )
            for r in rows
        ]

    def potential(self) -> torch.Tensor:
        """Shaping potential, negative normalized fruit distance."""
        head = self.head
        manhattan = (self.fruit - head).abs().sum(dim=1)
        return -manhattan.float() / (self.hw[:, 0] + self.hw[:, 1]).float()

    def observe(self) -> torch.Tensor:
        """Build grid observations on device.

        Returns:
            Float tensor of shape (num_envs, 4, canvas, canvas).
        """
        b = self.num_envs
        s = self.canvas
        dev = self.device
        area = s * s
        # Channels 0..3 into a flat scratch with one dump slot per
        # channel so invalid scatter targets fall off the end.
        obs = torch.zeros(b, 4, area + 1, device=dev)

        rel = (self._slots[None, :] - self.tail_i[:, None]) % self.capacity
        valid = rel < self.length[:, None]
        if self.obs_config.body_decay:
            values = (rel + 1).float() / self.length[:, None].float()
        else:
            values = torch.ones_like(rel, dtype=torch.float32)
        body_idx = self.body[:, :, 0] * s + self.body[:, :, 1]
        body_idx = torch.where(
            valid, body_idx, torch.full_like(body_idx, area)
        )
        obs[:, 0].scatter_(1, body_idx, values * valid.float())

        head = self.head
        head_idx = (head[:, 0] * s + head[:, 1]).unsqueeze(1)
        obs[:, 1].scatter_(1, head_idx, 1.0)
        fruit_idx = (self.fruit[:, 0] * s + self.fruit[:, 1]).unsqueeze(1)
        obs[:, 2].scatter_(1, fruit_idx, 1.0)

        rows = self._canvas_rows()
        cols = self._canvas_cols()
        wall = (rows >= self.hw[:, 0, None]) | (cols >= self.hw[:, 1, None])
        obs[:, 3, :area] = wall.float()

        grid = obs[:, :, :area].reshape(b, 4, s, s)
        if self.obs_config.egocentric:
            for k in range(1, 4):
                sel = self.direction == k
                grid[sel] = torch.rot90(grid[sel], k, dims=(2, 3))
        return grid

    def _canvas_rows(self) -> torch.Tensor:
        """Row index per flat canvas cell, shape (1, canvas*canvas)."""
        s = self.canvas
        return (
            torch.arange(s * s, device=self.device)[None, :] // s
        )

    def _canvas_cols(self) -> torch.Tensor:
        """Col index per flat canvas cell, shape (1, canvas*canvas)."""
        s = self.canvas
        return torch.arange(s * s, device=self.device)[None, :] % s

    def load_state_from(self, env: VecSnake) -> None:
        """Copy full state from a numpy VecSnake (test support).

        Args:
            env: Reference environment with matching shape.
        """
        dev = self.device

        def grab(arr: np.ndarray) -> torch.Tensor:
            # Copy: from_numpy shares memory, and on CPU .to() would
            # keep sharing, letting the two envs corrupt each other.
            return torch.from_numpy(np.array(arr)).to(dev)

        self.hw = grab(env.hw)
        self.body = grab(env.body)
        self.head_i = grab(env.head_i)
        self.tail_i = grab(env.tail_i)
        self.length = grab(env.length)
        self.occ = grab(env.occ.reshape(self.num_envs, -1))
        self.fruit = grab(env.fruit)
        self.direction = grab(env.direction)
        self.steps = grab(env.steps)
        self.since_fruit = grab(env.since_fruit)
        self.apples = grab(env.apples)
