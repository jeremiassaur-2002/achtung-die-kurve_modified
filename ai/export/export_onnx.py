"""Exports a trained policy to ONNX for in-browser inference (ai_bot.js, via
onnxruntime-web). Action masking, if it was used during training, is a training-
time sample-efficiency trick - it is not part of the exported graph. The browser
just takes argmax/softmax-sampling over the raw policy logits, exactly like any
deployed RL policy: the agent has already learned to avoid fatal actions in the
vast majority of states, which is the whole point of having trained it.

CLI:
    python -m ai.export.export_onnx --checkpoint path/to/model.zip --out ai/exported/model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch as th
import torch.nn as nn

from ai.env.observation import VECTOR_DIM, ObsConfig


class _PolicyLogitsWrapper(nn.Module):
    """Wraps an SB3 (Maskable)ActorCriticPolicy so it can be traced by torch.onnx.export
    as a plain (image, vector) -> action_logits function."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, image: th.Tensor, vector: th.Tensor) -> th.Tensor:
        obs = {"image": image, "vector": vector}
        distribution = self.policy.get_distribution(obs)
        return distribution.distribution.logits


def _load_any_ppo(model_path: str):
    try:
        from stable_baselines3 import PPO

        return PPO.load(model_path, device="cpu")
    except Exception:
        from sb3_contrib import MaskablePPO

        return MaskablePPO.load(model_path, device="cpu")


def export_onnx(model_path: str, out_path: str, obs_cfg: ObsConfig | None = None, opset_version: int = 17) -> Path:
    obs_cfg = obs_cfg or ObsConfig()
    model = _load_any_ppo(model_path)
    policy = model.policy
    policy.eval()
    wrapper = _PolicyLogitsWrapper(policy)

    dummy_image = th.zeros((1, *obs_cfg.image_shape), dtype=th.uint8)
    dummy_vector = th.zeros((1, VECTOR_DIM), dtype=th.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    th.onnx.export(
        wrapper,
        (dummy_image, dummy_vector),
        str(out_path),
        input_names=["image", "vector"],
        output_names=["action_logits"],
        dynamic_axes={"image": {0: "batch"}, "vector": {0: "batch"}, "action_logits": {0: "batch"}},
        opset_version=opset_version,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained SB3 checkpoint to ONNX for in-browser inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="ai/exported/model.onnx")
    parser.add_argument("--obs-resolution", type=int, default=96)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    obs_cfg = ObsConfig(obs_resolution=args.obs_resolution, frame_stack=args.frame_stack)
    path = export_onnx(args.checkpoint, args.out, obs_cfg, opset_version=args.opset)
    print(f"exported ONNX model -> {path}")


if __name__ == "__main__":
    main()
