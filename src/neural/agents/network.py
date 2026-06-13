"""Policy and value networks.

The grid network is fully convolutional with pooled heads, so the
same weights run on any canvas size. That is what makes a single
policy testable on board geometries it never trained on.
"""

import torch
from torch import nn

from neural.config import NetworkConfig
from neural.obs import FEATURES9_DIM, GRID_CHANNELS


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection.

    Args:
        channels: Channel width.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block."""
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)


class GridActorCritic(nn.Module):
    """Convolutional trunk with pooled policy and value heads.

    Args:
        config: Network settings.
        n_actions: Size of the action space.
    """

    def __init__(self, config: NetworkConfig, n_actions: int) -> None:
        super().__init__()
        c = config.channels
        self.stem = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, c, 3, padding=1), nn.ReLU()
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(c) for _ in range(config.blocks)]
        )
        # Mean and max pooling concatenated; size-independent.
        self.policy = nn.Sequential(
            nn.Linear(2 * c, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, n_actions),
        )
        self.value = nn.Sequential(
            nn.Linear(2 * c, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, 1),
        )

    def trunk(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations into pooled features."""
        x = self.blocks(self.stem(obs))
        mean = x.mean(dim=(2, 3))
        peak = x.amax(dim=(2, 3))
        return torch.cat([mean, peak], dim=1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy logits and state values.

        Args:
            obs: Batch of grid observations.

        Returns:
            Logits of shape (batch, n_actions) and values of shape
            (batch,).
        """
        feat = self.trunk(obs)
        return self.policy(feat), self.value(feat).squeeze(-1)


class MlpActorCritic(nn.Module):
    """MLP for the legacy 9-feature observation ablation.

    Args:
        config: Network settings. Hidden width is reused.
        n_actions: Size of the action space.
    """

    def __init__(self, config: NetworkConfig, n_actions: int) -> None:
        super().__init__()
        h = config.hidden
        self.body = nn.Sequential(
            nn.Linear(FEATURES9_DIM, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )
        self.policy = nn.Linear(h, n_actions)
        self.value = nn.Linear(h, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy logits and state values."""
        feat = self.body(obs)
        return self.policy(feat), self.value(feat).squeeze(-1)
