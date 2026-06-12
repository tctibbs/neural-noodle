"""Typed, hashable configuration schema for all experiments.

Every field that could be ablated is an explicit typed field. Configs
are validated on load, serializable to YAML, and hashed so every result
row traces to an exact configuration.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class BoardSpec(BaseModel):
    """A single board geometry.

    Attributes:
        width: Number of columns.
        height: Number of rows.
    """

    width: int = Field(ge=2, le=32)
    height: int = Field(ge=2, le=32)

    @property
    def cells(self) -> int:
        """Total number of cells on the board."""
        return self.width * self.height


class EnvConfig(BaseModel):
    """Vectorized Snake environment settings.

    Attributes:
        boards: Board geometries sampled uniformly at episode reset.
            A single entry trains on a fixed board.
        num_envs: Number of parallel environments in the batch.
        start_length: Initial snake length.
        starvation_factor: Episode ends after this multiple of the cell
            count in steps without eating. Caps survival-by-looping.
        init_mode: Random places the snake on a random row. Row0
            places it on the top row, which is consistent with the
            planner's Hamiltonian cycle order and is what the fixed
            evaluation protocol uses for every policy.
    """

    boards: list[BoardSpec] = Field(
        default_factory=lambda: [BoardSpec(width=10, height=10)],
        min_length=1,
    )
    num_envs: int = Field(default=1024, ge=1)
    start_length: int = Field(default=3, ge=1)
    starvation_factor: float = Field(default=4.0, gt=0)
    init_mode: Literal["random", "row0"] = "random"


class ObsConfig(BaseModel):
    """Observation representation settings.

    Attributes:
        kind: Grid stacks multi-channel board planes for a CNN.
            Features9 reproduces the legacy 9-feature vector as an
            ablation baseline.
        egocentric: Rotate the grid so the snake always faces up.
        body_decay: Encode body cells as time-to-vacate in [0, 1]
            instead of a binary plane.
    """

    kind: Literal["grid", "features9"] = "grid"
    egocentric: bool = True
    body_decay: bool = True


class ActionConfig(BaseModel):
    """Action space settings.

    Attributes:
        space: Egocentric uses turn-left, straight, turn-right.
            Absolute uses the four compass directions, where the
            reverse move is a no-op kept as in the legacy game.
    """

    space: Literal["egocentric", "absolute"] = "egocentric"


class RewardConfig(BaseModel):
    """Reward shaping settings.

    Attributes:
        fruit: Reward for eating a fruit.
        death: Penalty applied on death.
        win: Bonus for filling the board completely.
        step_cost: Per-step cost applying efficiency pressure.
        potential_shaping: Add potential-based shaping on the
            Manhattan distance from head to fruit.
        shaping_coef: Scale of the potential-based shaping term.
    """

    fruit: float = 1.0
    death: float = -1.0
    win: float = 10.0
    step_cost: float = 0.01
    potential_shaping: bool = False
    shaping_coef: float = 0.0


class NetworkConfig(BaseModel):
    """Policy and value network settings.

    Attributes:
        channels: Width of the convolutional trunk.
        blocks: Number of residual blocks in the trunk.
        hidden: Width of the post-pooling fully connected layer.
    """

    channels: int = Field(default=64, ge=8)
    blocks: int = Field(default=4, ge=1)
    hidden: int = Field(default=256, ge=16)


class PPOConfig(BaseModel):
    """PPO algorithm settings.

    Attributes:
        total_steps: Total environment steps for the run.
        rollout_len: Steps per environment per rollout.
        lr: Adam learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_coef: PPO clip range.
        epochs: Optimization epochs per rollout.
        minibatches: Minibatches per epoch.
        entropy_coef: Entropy bonus coefficient.
        value_coef: Value loss coefficient.
        max_grad_norm: Gradient clipping norm.
        anneal_lr: Linearly decay the learning rate to zero.
    """

    total_steps: int = Field(default=100_000_000, ge=1)
    rollout_len: int = Field(default=128, ge=8)
    lr: float = Field(default=3e-4, gt=0)
    gamma: float = Field(default=0.99, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, gt=0, le=1)
    clip_coef: float = Field(default=0.2, gt=0)
    epochs: int = Field(default=2, ge=1)
    minibatches: int = Field(default=8, ge=1)
    entropy_coef: float = Field(default=0.01, ge=0)
    value_coef: float = Field(default=0.5, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    anneal_lr: bool = True


class EvalConfig(BaseModel):
    """Fixed evaluation protocol settings.

    Attributes:
        episodes_per_board: Evaluation episodes per board geometry.
        boards: Board geometries evaluated, including held-out ones.
        seed: Base seed for the evaluation episode stream.
    """

    episodes_per_board: int = Field(default=100, ge=1)
    boards: list[BoardSpec] = Field(
        default_factory=lambda: [
            BoardSpec(width=8, height=8),
            BoardSpec(width=10, height=10),
            BoardSpec(width=12, height=12),
            BoardSpec(width=16, height=16),
        ],
        min_length=1,
    )
    seed: int = 7_777


class TrainConfig(BaseModel):
    """Top-level run configuration.

    Attributes:
        run_name: Human-readable run label.
        seed: Master seed for the run.
        env: Environment settings.
        obs: Observation settings.
        action: Action space settings.
        reward: Reward settings.
        network: Network settings.
        ppo: Algorithm settings.
        eval: Evaluation protocol settings.
        eval_every: Environment steps between evaluation passes.
        checkpoint_every: Environment steps between checkpoints.
        device: Torch device string.
        wandb: Mirror metrics to Weights and Biases. The ledger stays
            authoritative either way.
    """

    run_name: str = "dev"
    seed: int = 0
    env: EnvConfig = Field(default_factory=EnvConfig)
    obs: ObsConfig = Field(default_factory=ObsConfig)
    action: ActionConfig = Field(default_factory=ActionConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    eval_every: int = Field(default=2_000_000, ge=1)
    checkpoint_every: int = Field(default=10_000_000, ge=1)
    device: str = "cuda"
    wandb: bool = False

    @model_validator(mode="after")
    def _check_board_fits_snake(self) -> "TrainConfig":
        """Ensure the start length fits on every configured board."""
        for board in self.env.boards:
            if self.env.start_length >= board.width:
                msg = (
                    f"start_length {self.env.start_length} does not fit "
                    f"on board {board.width}x{board.height}"
                )
                raise ValueError(msg)
        return self


def config_hash(config: BaseModel) -> str:
    """Return a short stable hash identifying an exact configuration.

    Args:
        config: Any pydantic model.

    Returns:
        First 12 hex characters of the sha256 of the canonical JSON
        serialization.
    """
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def load_config(path: Path) -> TrainConfig:
    """Load and validate a training configuration from YAML.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration.
    """
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    return TrainConfig.model_validate(raw)


def save_config(config: BaseModel, path: Path) -> None:
    """Serialize a configuration to YAML.

    Args:
        config: Any pydantic model.
        path: Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(config.model_dump(mode="json"), fh, sort_keys=True)
