"""Adapter exposing a trained network as a batch policy for eval."""

import numpy as np
import torch

from neural.actions import to_absolute
from neural.config import ActionConfig, ObsConfig
from neural.env.vec_env import VecSnake
from neural.obs import Features9Builder, GridObsBuilder


class RLPolicy:
    """Greedy policy wrapper around an actor-critic network.

    Args:
        net: Trained network in eval mode.
        obs_config: Observation settings the network was trained with.
        action_config: Action space settings.
        canvas: Canvas size for grid observations. The fully
            convolutional network accepts any canvas, so evaluation
            boards may differ from training boards.
        device: Torch device for inference.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        obs_config: ObsConfig,
        action_config: ActionConfig,
        canvas: int,
        device: str = "cuda",
    ) -> None:
        self.net = net
        self.action_config = action_config
        self.device = device
        self.builder: GridObsBuilder | Features9Builder
        if obs_config.kind == "grid":
            self.builder = GridObsBuilder(obs_config, canvas)
        else:
            self.builder = Features9Builder(obs_config)

    @torch.no_grad()
    def actions(self, env: VecSnake) -> np.ndarray:
        """Return greedy absolute directions for every environment."""
        obs = torch.from_numpy(self.builder.build(env)).to(self.device)
        logits, _ = self.net(obs)
        picks = logits.argmax(dim=1).cpu().numpy()
        return to_absolute(self.action_config, picks, env)
