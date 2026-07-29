"""Wires CurveFeaturesExtractor into SB3's policy_kwargs. One helper, used
identically by plain PPO and sb3-contrib's MaskablePPO - the feature extractor
doesn't care whether masking is on, only curve_env.py's action_masks() and the
algorithm class differ between the two.
"""

from __future__ import annotations

from ai.models.cnn_extractor import CurveFeaturesExtractor

POLICY_NAME = "MultiInputPolicy"  # both PPO and MaskablePPO resolve this to their Dict-obs policy


def build_policy_kwargs(
    cnn_arch: str = "small",
    cnn_features_dim: int = 256,
    vector_features_dim: int = 64,
    net_arch: dict | list | None = None,
) -> dict:
    """Returns a `policy_kwargs` dict for `PPO(POLICY_NAME, env, policy_kwargs=...)`
    or `MaskablePPO(POLICY_NAME, env, policy_kwargs=...)`."""
    return dict(
        features_extractor_class=CurveFeaturesExtractor,
        features_extractor_kwargs=dict(
            cnn_arch=cnn_arch,
            cnn_features_dim=cnn_features_dim,
            vector_features_dim=vector_features_dim,
        ),
        net_arch=net_arch if net_arch is not None else dict(pi=[128, 128], vf=[128, 128]),
    )
