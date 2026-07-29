"""Packages a trained checkpoint for local download: the native SB3 .zip (usable
to resume training or run more evaluation matches) plus a plain PyTorch
state_dict .pt of just the policy network, for anyone who wants the weights
without going through SB3 at all.

CLI:
    python -m ai.export.export_weights --checkpoint path/to/model.zip --out-dir ai/exported
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch as th

from ai.export.export_onnx import _load_any_ppo


def export_weights(model_path: str, out_dir: str) -> dict[str, Path]:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    sb3_zip = out_dir_path / "model_sb3.zip"
    shutil.copy(model_path, sb3_zip)

    model = _load_any_ppo(model_path)
    state_dict_path = out_dir_path / "policy_state_dict.pt"
    th.save(model.policy.state_dict(), state_dict_path)

    return {"sb3_zip": sb3_zip, "state_dict": state_dict_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a trained checkpoint for local download.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="ai/exported")
    args = parser.parse_args()
    paths = export_weights(args.checkpoint, args.out_dir)
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
