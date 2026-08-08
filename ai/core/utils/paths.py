"""Ein einziges, versionsunabhängiges Ausgabelayout unter ai/output/.

Warum zentral: v1_0 (PPO) und v1_1 (Dreamer) schreiben Checkpoints, TensorBoard-
Events, Videos, Reports und Timing-Logs. Wenn jede Version ihr eigenes Layout
erfindet, kann kein Sync-Skript, kein `tensorboard --logdir` und kein
Report-Tool beide gleichzeitig bedienen. Deshalb gilt fuer JEDEN Lauf:

    ai/output/<version>/<run_name>/
        checkpoints/     rotierende Zwischenstaende (Crash-Recovery)
        best/            bestes Modell des Laufs
        tensorboard/     Events (ein `--logdir ai/output` zeigt alle Laeufe)
        videos/          MP4 + JSON-Sidecars
        reports/         Meilenstein-Reports, Plots
        metrics/         metrics.csv, Monitor-Logs
        timing/          timing.json / timing.md (siehe utils/timing.py)
        config_used.yaml die tatsaechlich benutzte Config (nicht die auf Platte)

`ai/output/` ist in .gitignore - Gewichte gehoeren nicht in die Repo-History.
Der Sync (scripts/sync_output.sh) pusht bewusst nur `best/` und die kleinen
Text-Artefakte in einen separaten Branch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Repo-Root = drei Ebenen ueber dieser Datei (ai/core/utils/paths.py)
REPO_ROOT = Path(__file__).resolve().parents[3]

# Auf einem gemieteten Cloud-Rechner liegt das Repo oft auf der (kleinen)
# Container-Disk, das persistente Volume aber woanders. AI_OUTPUT_ROOT zeigt
# dann dorthin, ohne dass irgendein Aufrufer Pfade anfassen muss.
OUTPUT_ROOT = Path(os.environ.get("AI_OUTPUT_ROOT", REPO_ROOT / "ai" / "output"))

SUBDIRS = ("checkpoints", "best", "tensorboard", "videos", "reports", "metrics", "timing")


@dataclass(frozen=True)
class RunPaths:
    """Alle Pfade eines Laufs. Bewusst ein Objekt statt lose Strings, damit ein
    Tippfehler ein AttributeError ist und kein still angelegter Geisterordner."""

    version: str
    run_name: str
    root: Path

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def best(self) -> Path:
        return self.root / "best"

    @property
    def tensorboard(self) -> Path:
        return self.root / "tensorboard"

    @property
    def videos(self) -> Path:
        return self.root / "videos"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def timing(self) -> Path:
        return self.root / "timing"

    @property
    def config_used(self) -> Path:
        return self.root / "config_used.yaml"

    def mkdirs(self) -> "RunPaths":
        for name in SUBDIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        return self


def run_paths(version: str, run_name: str, output_root: Path | str | None = None) -> RunPaths:
    root = Path(output_root) if output_root is not None else OUTPUT_ROOT
    return RunPaths(version=version, run_name=run_name, root=root / version / run_name)


def latest_run(version: str, output_root: Path | str | None = None) -> RunPaths | None:
    """Juengster Lauf einer Version - das ist, was `--run-name auto` aufgreift,
    damit ein Neustart nach einem Crash keinen zweiten Lauf danebenlegt."""
    root = Path(output_root) if output_root is not None else OUTPUT_ROOT
    vdir = root / version
    if not vdir.is_dir():
        return None
    candidates = [d for d in vdir.iterdir() if d.is_dir()]
    if not candidates:
        return None
    newest = max(candidates, key=lambda d: d.stat().st_mtime)
    return RunPaths(version=version, run_name=newest.name, root=newest)
