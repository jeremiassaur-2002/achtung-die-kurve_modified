"""Wires CurveFeaturesExtractor into SB3's policy_kwargs. One helper, used
identically by plain PPO and sb3-contrib's MaskablePPO - the feature extractor
doesn't care whether masking is on, only curve_env.py's action_masks() and the
algorithm class differ between the two.
"""

from __future__ import annotations

from ai.v1_0.models.cnn_extractor import CurveFeaturesExtractor

POLICY_NAME = "MultiInputPolicy"  # both PPO and MaskablePPO resolve this to their Dict-obs policy


def build_policy_kwargs(
    cnn_arch: str = "small",
    cnn_features_dim: int = 256,
    vector_features_dim: int = 64,
    net_arch: dict | list | None = None,
    algo: str = "ppo",
    lstm_kwargs: dict | None = None,
) -> dict:
    """Returns a `policy_kwargs` dict for the chosen algorithm class (see
    ai/v1_0/models/algo.py). The shared part - CurveFeaturesExtractor over the
    {image, vector} Dict space - is identical for all four; what differs:
      * qrdqn: DQN-family policies take net_arch as a plain LIST (a pi/vf dict
        is an actor-critic concept and raises a TypeError there)
      * recurrent_ppo: LSTM sizing (`lstm:` block in the YAML) is passed through
    """
    kwargs = dict(
        features_extractor_class=CurveFeaturesExtractor,
        features_extractor_kwargs=dict(
            cnn_arch=cnn_arch,
            cnn_features_dim=cnn_features_dim,
            vector_features_dim=vector_features_dim,
        ),
    )
    if algo == "qrdqn":
        if isinstance(net_arch, dict):
            raise ValueError("qrdqn erwartet net_arch als Liste (z.B. [128, 128]), nicht als pi/vf-Dict")
        kwargs["net_arch"] = net_arch if net_arch is not None else [128, 128]
        return kwargs

    kwargs["net_arch"] = net_arch if net_arch is not None else dict(pi=[128, 128], vf=[128, 128])
    if algo == "recurrent_ppo":
        lstm = dict(lstm_hidden_size=128, n_lstm_layers=1)
        lstm.update(lstm_kwargs or {})
        kwargs.update(lstm)
    return kwargs
