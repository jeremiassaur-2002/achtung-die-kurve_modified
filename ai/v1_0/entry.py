"""Adapter zwischen `ai.run` und dem bestehenden PPO-Trainingsskript.

Bewusst duenn: `ai/v1_0/training/train.py` bleibt unveraendert lauffaehig (die
alten Aufrufe aus dem Colab-Notebook funktionieren weiter). Diese Datei sorgt
nur dafuer, dass ein Lauf ueber `ai.run` im gemeinsamen Ausgabelayout landet und
seine Zeiten in dieselbe timing.json schreibt wie v1_1.
"""

from __future__ import annotations

from pathlib import Path

from ai.core.utils.paths import RunPaths
from ai.core.utils.timing import PhaseTimer


def train(cfg: dict, paths: RunPaths, timer: PhaseTimer, resume: str | None = "auto", init_from: str | None = None) -> Path:
    from ai.v1_0.training.train import run_training

    # train.py legt <run_root>/<run_name>/ selbst an - mit run_root = ai/output/v1_0
    # und run_name = paths.run_name trifft das exakt das Verzeichnis, das ai.run
    # bereits vorbereitet hat (checkpoints/, tensorboard/, ... existieren schon).
    tmp_cfg = paths.root / "_config_effective.yaml"
    import yaml

    tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))

    with timer.phase("ppo_learn", total_timesteps=cfg.get("total_timesteps")):
        final = run_training(
            tmp_cfg,
            init_from=init_from,
            resume=None if resume in (None, "none") else resume,
            run_root=paths.root.parent,
            run_name=paths.run_name,
        )
    return Path(final)
