"""PPO training runner.

Run with:
    uv run python -m snake_rl.train --config configs/main-10x10.yaml
"""

import argparse
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.distributions import Categorical

from snake_rl.actions import num_actions, to_absolute
from snake_rl.agents.network import GridActorCritic, MlpActorCritic
from snake_rl.agents.rl_policy import RLPolicy
from snake_rl.config import (
    TrainConfig,
    config_hash,
    load_config,
    save_config,
)
from snake_rl.env.tensor_env import TensorVecSnake
from snake_rl.env.vec_env import VecSnake
from snake_rl.evaluate import aggregate, run_episodes
from snake_rl.ledger import LedgerRow, append_row
from snake_rl.logging_setup import setup_logging
from snake_rl.obs import Features9Builder, GridObsBuilder
from snake_rl.reward import compute_rewards, compute_rewards_t, potential


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE lines into the environment without logging them.

    Args:
        path: Dotenv file. Missing file is fine.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_network(config: TrainConfig) -> torch.nn.Module:
    """Construct the network matching the observation kind."""
    n_act = num_actions(config.action)
    if config.obs.kind == "grid":
        return GridActorCritic(config.network, n_act)
    return MlpActorCritic(config.network, n_act)


def make_canvas(config: TrainConfig) -> int:
    """Square canvas covering every training board."""
    return max(max(b.width, b.height) for b in config.env.boards)


class Trainer:
    """Owns one PPO training run end to end.

    Args:
        config: Validated run configuration.
        run_dir: Output directory for checkpoints and config.
    """

    def __init__(self, config: TrainConfig, run_dir: Path) -> None:
        self.config = config
        self.run_dir = run_dir
        self.chash = config_hash(config)
        self.run_id = setup_logging(log_dir=run_dir)
        seed_everything(config.seed)
        self.device = config.device

        self.canvas = make_canvas(config)
        self.tensor_backend = config.env.backend == "tensor"
        if self.tensor_backend and config.obs.kind != "grid":
            msg = "tensor backend supports grid observations only"
            raise ValueError(msg)
        self.env: VecSnake | TensorVecSnake
        if self.tensor_backend:
            self.env = TensorVecSnake(
                config.env,
                config.obs,
                self.canvas,
                seed=config.seed,
                device=self.device,
            )
        else:
            self.env = VecSnake(config.env, seed=config.seed)
        self.builder: GridObsBuilder | Features9Builder
        if config.obs.kind == "grid":
            self.builder = GridObsBuilder(config.obs, self.canvas)
        else:
            self.builder = Features9Builder(config.obs)
        self.amp = config.ppo.amp and self.device.startswith("cuda")
        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
        self.net = build_network(config).to(self.device)
        if self.amp and config.obs.kind == "grid":
            self.net = self.net.to(memory_format=torch.channels_last)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=config.ppo.lr, eps=1e-5
        )
        self.n_act = num_actions(config.action)
        self.frames = 0
        self.start_time = time.perf_counter()
        self.recent = deque(maxlen=2_000)
        self.wandb_run = None
        if config.wandb and os.environ.get("WANDB_API_KEY"):
            import wandb

            self.wandb_run = wandb.init(
                project="neural-noodle",
                name=f"{config.run_name}-s{config.seed}",
                config=config.model_dump(mode="json"),
            )

    def collect_rollout_tensor(self) -> tuple[torch.Tensor, ...]:
        """Collect one rollout entirely on the torch device."""
        cfg = self.config
        env = self.env
        assert isinstance(env, TensorVecSnake)
        t_len = cfg.ppo.rollout_len
        b = cfg.env.num_envs
        dev = self.device
        s = self.canvas
        obs_buf = torch.zeros(t_len, b, 4, s, s, device=dev)
        act_buf = torch.zeros(t_len, b, dtype=torch.long, device=dev)
        logp_buf = torch.zeros(t_len, b, device=dev)
        rew_buf = torch.zeros(t_len, b, device=dev)
        done_buf = torch.zeros(t_len, b, device=dev)
        val_buf = torch.zeros(t_len, b, device=dev)
        shaping = cfg.reward.potential_shaping
        ego = cfg.action.space == "egocentric"

        for t in range(t_len):
            obs = env.observe()
            net_in = (
                obs.contiguous(memory_format=torch.channels_last)
                if self.amp
                else obs
            )
            with (
                torch.no_grad(),
                torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp),
            ):
                logits, value = self.net(net_in)
            logits = logits.float()
            value = value.float()
            with torch.no_grad():
                dist = Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
            if ego:
                absolute = (env.direction + action - 1) % 4
            else:
                absolute = action
            phi_before = env.potential() if shaping else None
            events = env.step(absolute)
            phi_after = env.potential() if shaping else None
            rewards = compute_rewards_t(
                cfg.reward, cfg.ppo.gamma, events, phi_before, phi_after
            )
            obs_buf[t] = obs
            act_buf[t] = action
            logp_buf[t] = logp
            rew_buf[t] = rewards
            done_buf[t] = events["done"].float()
            val_buf[t] = value
        self.frames += t_len * b
        self.recent.extend(env.drain_finished())

        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp),
        ):
            _, last_value = self.net(env.observe())
        last_value = last_value.float()
        adv_buf = torch.zeros(t_len, b, device=dev)
        last_gae = torch.zeros(b, device=dev)
        for t in reversed(range(t_len)):
            nnt = 1.0 - done_buf[t]
            nv = last_value if t == t_len - 1 else val_buf[t + 1]
            delta = rew_buf[t] + cfg.ppo.gamma * nv * nnt - val_buf[t]
            last_gae = (
                delta + cfg.ppo.gamma * cfg.ppo.gae_lambda * nnt * last_gae
            )
            adv_buf[t] = last_gae
        ret_buf = adv_buf + val_buf
        return (
            obs_buf.reshape(t_len * b, 4, s, s),
            act_buf.reshape(-1),
            logp_buf.reshape(-1),
            adv_buf.reshape(-1),
            ret_buf.reshape(-1),
        )

    def collect_rollout(
        self,
    ) -> tuple[torch.Tensor, ...]:
        """Collect one rollout and return flattened training tensors."""
        cfg = self.config
        env = self.env
        assert isinstance(env, VecSnake)
        t_len = cfg.ppo.rollout_len
        b = cfg.env.num_envs
        obs_buf: list[torch.Tensor] = []
        act_buf = torch.zeros(t_len, b, dtype=torch.long)
        logp_buf = torch.zeros(t_len, b)
        rew_buf = torch.zeros(t_len, b)
        done_buf = torch.zeros(t_len, b)
        val_buf = torch.zeros(t_len, b)

        for t in range(t_len):
            obs_np = self.builder.build(env)
            obs = torch.from_numpy(obs_np).to(self.device)
            with torch.no_grad():
                logits, value = self.net(obs)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
            phi_before = potential(env)
            absolute = to_absolute(cfg.action, action.cpu().numpy(), env)
            result = env.step(absolute)
            phi_after = potential(env)
            rewards = compute_rewards(
                cfg.reward, cfg.ppo.gamma, result, phi_before, phi_after
            )
            self.recent.extend(env.drain_finished())

            obs_buf.append(obs)
            act_buf[t] = action.cpu()
            logp_buf[t] = logp.cpu()
            rew_buf[t] = torch.from_numpy(rewards)
            done_buf[t] = torch.from_numpy(result.done.astype(np.float32))
            val_buf[t] = value.cpu()
        self.frames += t_len * b

        with torch.no_grad():
            last_obs = torch.from_numpy(self.builder.build(env)).to(self.device)
            _, last_value = self.net(last_obs)
        adv_buf = torch.zeros(t_len, b)
        last_gae = torch.zeros(b)
        next_value = last_value.cpu()
        next_nonterminal = 1.0 - done_buf[t_len - 1]
        for t in reversed(range(t_len)):
            if t == t_len - 1:
                nnt = next_nonterminal
                nv = next_value
            else:
                nnt = 1.0 - done_buf[t]
                nv = val_buf[t + 1]
            delta = rew_buf[t] + cfg.ppo.gamma * nv * nnt - val_buf[t]
            last_gae = (
                delta + cfg.ppo.gamma * cfg.ppo.gae_lambda * nnt * last_gae
            )
            adv_buf[t] = last_gae
        ret_buf = adv_buf + val_buf

        flat_obs = torch.cat(obs_buf, dim=0)
        return (
            flat_obs,
            act_buf.reshape(-1).to(self.device),
            logp_buf.reshape(-1).to(self.device),
            adv_buf.reshape(-1).to(self.device),
            ret_buf.reshape(-1).to(self.device),
        )

    def update(
        self,
        flat_obs: torch.Tensor,
        actions: torch.Tensor,
        old_logp: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, float]:
        """Run the PPO update epochs and return loss diagnostics."""
        cfg = self.config.ppo
        n = flat_obs.shape[0]
        mb_size = n // cfg.minibatches
        idx = torch.randperm(n, device=self.device)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        count = 0
        for _ in range(cfg.epochs):
            idx = idx[torch.randperm(n, device=self.device)]
            for start in range(0, n, mb_size):
                mb = idx[start : start + mb_size]
                batch = flat_obs[mb]
                if self.amp and batch.dim() == 4:
                    batch = batch.contiguous(memory_format=torch.channels_last)
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=self.amp
                ):
                    logits, values = self.net(batch)
                logits = logits.float()
                values = values.float()
                dist = Categorical(logits=logits)
                logp = dist.log_prob(actions[mb])
                ratio = (logp - old_logp[mb]).exp()
                adv = advantages[mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg1 = -adv * ratio
                pg2 = -adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                policy_loss = torch.max(pg1, pg2).mean()
                value_loss = 0.5 * (values - returns[mb]).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()
                stats["policy_loss"] += float(policy_loss.detach())
                stats["value_loss"] += float(value_loss.detach())
                stats["entropy"] += float(entropy.detach())
                count += 1
        return {k: v / count for k, v in stats.items()}

    def evaluate(self, episodes: int) -> dict[str, dict[str, float]]:
        """Run the fixed eval protocol on every configured board."""
        self.net.eval()
        out: dict[str, dict[str, float]] = {}
        for board in self.config.eval.boards:
            # Board-exact canvas: a smaller board padded into the
            # training canvas puts an interior wall band in front of a
            # policy that never saw one, and it starves (observed at
            # 20M frames on 8x8 from a 10x10-trained policy).
            canvas = max(board.width, board.height)
            policy = RLPolicy(
                self.net,
                self.config.obs,
                self.config.action,
                canvas=canvas,
                device=self.device,
            )
            stats = run_episodes(
                make_policy=lambda env, p=policy: p,
                board=board,
                episodes=episodes,
                seed=self.config.eval.seed,
                start_length=self.config.env.start_length,
            )
            out[f"{board.width}x{board.height}"] = aggregate(stats)
        self.net.train()
        return out

    def checkpoint(self, tag: str) -> Path:
        """Save model weights and metadata."""
        path = self.run_dir / f"ckpt_{tag}.pt"
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "config": self.config.model_dump(mode="json"),
                "frames": self.frames,
                "run_id": self.run_id,
            },
            path,
        )
        return path

    def train(self) -> None:
        """Run the full training loop."""
        cfg = self.config
        save_config(cfg, self.run_dir / "config.yaml")
        logger.info(
            "run {} config {} on {} ({} envs, canvas {})",
            cfg.run_name,
            self.chash,
            self.device,
            cfg.env.num_envs,
            self.canvas,
        )
        batch = cfg.ppo.rollout_len * cfg.env.num_envs
        iterations = cfg.ppo.total_steps // batch
        next_eval = cfg.eval_every
        next_ckpt = cfg.checkpoint_every
        for it in range(1, iterations + 1):
            if cfg.ppo.anneal_lr:
                frac = 1.0 - (it - 1) / iterations
                for group in self.optimizer.param_groups:
                    group["lr"] = cfg.ppo.lr * frac
            if self.tensor_backend:
                tensors = self.collect_rollout_tensor()
            else:
                tensors = self.collect_rollout()
            losses = self.update(*tensors)
            if it % 10 == 0 or it == 1:
                sps = self.frames / (time.perf_counter() - self.start_time)
                window = list(self.recent)
                apples = (
                    float(np.mean([s.apples for s in window]))
                    if window
                    else 0.0
                )
                fill = (
                    float(np.mean([s.fill for s in window])) if window else 0.0
                )
                spa = [s.steps / s.apples for s in window if s.apples > 0]
                logger.info(
                    "it {} frames {:.1f}M sps {:.0f} apples {:.2f} "
                    "fill {:.1%} spa {:.1f} pl {:.3f} vl {:.3f} "
                    "ent {:.3f}",
                    it,
                    self.frames / 1e6,
                    sps,
                    apples,
                    fill,
                    float(np.mean(spa)) if spa else float("nan"),
                    losses["policy_loss"],
                    losses["value_loss"],
                    losses["entropy"],
                )
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {
                            "frames": self.frames,
                            "sps": sps,
                            "train/apples": apples,
                            "train/fill": fill,
                            "train/spa": (float(np.mean(spa)) if spa else None),
                            **{f"loss/{k}": v for k, v in losses.items()},
                        },
                        step=self.frames,
                    )
            if self.frames >= next_eval:
                next_eval += cfg.eval_every
                self.log_eval(self.evaluate(cfg.quick_eval_episodes))
            if self.frames >= next_ckpt:
                next_ckpt += cfg.checkpoint_every
                self.checkpoint(f"{self.frames}")
        self.checkpoint("final")
        self.log_eval(
            self.evaluate(cfg.eval.episodes_per_board), kind="eval_final"
        )
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def log_eval(
        self,
        results: dict[str, dict[str, float]],
        kind: str = "eval",
    ) -> None:
        """Log evaluation results and append ledger rows."""
        for board, metrics in results.items():
            logger.info(
                "eval {} spa {:.2f}+-{:.2f} fill {:.1%} win {:.0%} "
                "apples {:.1f}",
                board,
                metrics["steps_per_apple_mean"],
                metrics["steps_per_apple_std"],
                metrics["fill_mean"],
                metrics["win_rate"],
                metrics["apples_mean"],
            )
            append_row(
                LedgerRow(
                    kind=kind,
                    run_id=self.run_id,
                    run_name=self.config.run_name,
                    config_hash=self.chash,
                    seed=self.config.seed,
                    frames=self.frames,
                    wall_clock_s=time.perf_counter() - self.start_time,
                    metrics=metrics,
                    extra={"board": board},
                )
            )
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {f"eval/{board}/{k}": v for k, v in metrics.items()},
                    step=self.frames,
                )


def main() -> None:
    """Parse arguments and launch training."""
    parser = argparse.ArgumentParser(description="PPO Snake training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--seed", type=int, default=None, help="Override config seed."
    )
    args = parser.parse_args()
    load_dotenv()
    config = load_config(args.config)
    if args.seed is not None:
        config = config.model_copy(update={"seed": args.seed})
    run_dir = (
        Path("runs") / f"{config.run_name}-s{config.seed}-{config_hash(config)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    Trainer(config, run_dir).train()


if __name__ == "__main__":
    main()
