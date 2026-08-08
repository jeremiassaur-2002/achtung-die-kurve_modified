"""Adapter zwischen `ai.run` und der v1_1-Pipeline (Dreamer + Pfadplaner).

Die Pipeline hat drei Phasen, und `ai.run` misst jede einzeln:

  1. `dataset`      - Expertentrajektorien vom Beam-Planer (data/collect.py)
  2. `world_model`  - RSSM auf diesen Trajektorien, rein offline
  3. `policy`       - Actor/Critic in der Imagination des Weltmodells

**Stand.** Phase 1 laeuft. Phase 2 und 3 sind noch nicht implementiert; dieser
Adapter bricht dort mit einer klaren Meldung ab, statt einen Platzhalter zu
trainieren, der wie Fortschritt aussieht. Wer nur Daten sammeln will, nimmt
direkt `python -m ai.run collect --version v1_1 ...`.

**Warum die Reihenfolge so und nicht Dreamer-ueblich.** DreamerV3 sammelt
normalerweise waehrend des Trainings selbst weiter (online). Hier steht die
Offline-Variante zuerst, weil sie auf gemieteter Hardware billiger ist: das
Sammeln ist CPU-gebunden und laesst sich auf vielen billigen Kernen
parallelisieren, das Modelltraining ist GPU-gebunden. Getrennt kann man die
Datensammlung auf einer CPU-Maschine laufen lassen und die GPU nur fuer die
Stunden mieten, in denen sie wirklich rechnet. Online-Sammlung als spaetere
Erweiterung bleibt moeglich - der Replay-Buffer liest dieselben Shards.
"""

from __future__ import annotations

import json
from pathlib import Path

from ai.core.utils.paths import RunPaths
from ai.core.utils.timing import PhaseTimer


def _dataset_stats(dataset_dir: Path) -> dict:
    shards = sorted((dataset_dir / "shards").glob("*.npz")) if dataset_dir.exists() else []
    indices = sorted(dataset_dir.glob("index_w*.json")) if dataset_dir.exists() else []
    total_ticks = sum(json.loads(p.read_text()).get("total_ticks", 0) for p in indices)
    return {"shards": len(shards), "total_ticks": total_ticks}


def train(cfg: dict, paths: RunPaths, timer: PhaseTimer, resume: str | None = "auto", init_from: str | None = None) -> Path:
    dataset_dir = Path(cfg.get("dataset_dir") or (paths.root.parent / "dataset"))

    with timer.phase("dataset", dir=str(dataset_dir)) as rec:
        stats = _dataset_stats(dataset_dir)
        want = int(cfg.get("dataset_min_ticks", 200_000))
        if stats["total_ticks"] < want:
            from ai.v1_1.data.collect import collect

            episodes = int(cfg.get("collect_episodes", 100))
            print(f"[v1_1] Datensatz hat {stats['total_ticks']:,} Ticks (< {want:,}) - sammle {episodes} Episoden nach")
            collect(paths.config_used, dataset_dir, episodes=episodes, seed=cfg.get("seed", 0))
            stats = _dataset_stats(dataset_dir)
        rec.meta.update(stats)
        print(f"[v1_1] Datensatz: {stats['shards']} Shards, {stats['total_ticks']:,} Ticks")

    raise SystemExit(
        "[v1_1] Weltmodell und Actor/Critic sind noch nicht implementiert (Patch 2).\n"
        f"       Der Datensatz unter {dataset_dir} ist fertig und bleibt gueltig - "
        "Patch 2 setzt genau dort auf.\n"
        "       Bis dahin nutzbar: 'ai.run collect' (Daten), der Planer als Gegner, "
        "und v1_0 fuer vollstaendiges Training."
    )
