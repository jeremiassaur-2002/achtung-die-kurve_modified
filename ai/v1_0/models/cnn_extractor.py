"""Swappable CNN backbones for the {image, vector} Dict observation.

Every backbone takes `(in_channels, out_features)` and maps a `(N, C, H, W)` uint8-
turned-float image batch to `(N, out_features)`. All of them end in an
`AdaptiveAvgPool2d`, so none are hard-coded to one `obs_resolution` - swapping to a
bigger CNN, or a bigger obs_resolution, means adding/choosing a class here, not
touching curve_env.py or train.py. Register new architectures with `register_cnn`.
"""

from __future__ import annotations

import gymnasium as gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SmallCNN(nn.Module):
    """Nature-CNN-style, 3 conv layers - fast, the default for Phase 1-3 training."""

    def __init__(self, in_channels: int, out_features: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, out_features), nn.ReLU())

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.fc(self.conv(x))


class MediumCNN(nn.Module):
    """Deeper/wider than SmallCNN - for league/Phase 4+ once more capacity is needed."""

    def __init__(self, in_channels: int, out_features: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(nn.Linear(128 * 4 * 4, out_features), nn.ReLU())

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.fc(self.conv(x))


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        y = F.relu(self.c1(x))
        y = self.c2(y)
        return F.relu(x + y)


class LargeCNN(nn.Module):
    """Small ResNet - highest capacity preset, for when a lot of training compute
    (e.g. long Colab league runs) makes the extra parameters worthwhile."""

    def __init__(self, in_channels: int, out_features: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2), nn.ReLU())
        self.stage1 = nn.Sequential(_ResidualBlock(64), nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU())
        self.stage2 = nn.Sequential(_ResidualBlock(128), nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU())
        self.stage3 = _ResidualBlock(256)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(256 * 4 * 4, out_features), nn.ReLU())

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


CNN_REGISTRY: dict[str, type[nn.Module]] = {
    "small": SmallCNN,
    "medium": MediumCNN,
    "large": LargeCNN,
}


def register_cnn(name: str, cls: type[nn.Module]) -> None:
    """Add a new backbone (must accept (in_channels, out_features), forward -> (N, out_features))."""
    CNN_REGISTRY[name] = cls


class CurveFeaturesExtractor(BaseFeaturesExtractor):
    """Combines a CNN over `image` with a small MLP over `vector`, concatenated.
    This replaces SB3's default `CombinedExtractor` for the Dict observation space -
    pass it via `policy_kwargs` (see ai/v1_0/models/policy.py)."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_arch: str = "small",
        cnn_features_dim: int = 256,
        vector_features_dim: int = 64,
    ):
        super().__init__(observation_space, features_dim=cnn_features_dim + vector_features_dim)
        if cnn_arch not in CNN_REGISTRY:
            raise ValueError(f"unknown cnn_arch {cnn_arch!r}, expected one of {list(CNN_REGISTRY)}")

        image_space = observation_space.spaces["image"]
        vector_space = observation_space.spaces["vector"]

        self.cnn = CNN_REGISTRY[cnn_arch](image_space.shape[0], cnn_features_dim)
        self.vector_mlp = nn.Sequential(
            nn.Linear(vector_space.shape[0], 64), nn.ReLU(),
            nn.Linear(64, vector_features_dim), nn.ReLU(),
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        # NOTE: SB3's preprocess_obs() has already divided the uint8 image by 255
        # (is_image_space() matches our (12,96,96) uint8 Box). Dividing again here
        # squashed inputs into [0, 1/255] and left the CNN effectively blind.
        image = observations["image"].float()
        cnn_out = self.cnn(image)
        vector_out = self.vector_mlp(observations["vector"].float())
        return th.cat([cnn_out, vector_out], dim=1)
