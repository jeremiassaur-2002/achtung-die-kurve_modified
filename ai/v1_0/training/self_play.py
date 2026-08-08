"""Phase 2 self-play opponent pool: a small rotating set of recent snapshots of
the model currently being trained. Kept separate from the full League (Phase 4),
which additionally mixes in rule-based/random opponents and long-term history -
self-play is deliberately simpler (mostly-recent-self) since Phase 2's only job
is learning to win 1-on-1 against something close to its current level.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from ai.core.env.opponents import OpponentSpec


class SelfPlayPool:
    def __init__(self, root: str | Path, max_snapshots: int = 10):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_snapshots = max_snapshots
        self.snapshots: list[Path] = sorted(self.root.glob("snapshot_*.zip"), key=lambda p: p.stat().st_mtime)

    def add_snapshot(self, model_path: str | Path, step: int) -> Path:
        dest = self.root / f"snapshot_{step}.zip"
        shutil.copy(str(model_path), str(dest))
        self.snapshots.append(dest)
        while len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0).unlink(missing_ok=True)
        return dest

    def sample_specs(self, k: int, rng: random.Random) -> list[OpponentSpec]:
        if not self.snapshots:
            return [OpponentSpec(kind="random") for _ in range(k)]
        weights = [(i + 1) ** 2 for i in range(len(self.snapshots))]  # bias toward recent snapshots
        chosen = rng.choices(self.snapshots, weights=weights, k=k)
        return [OpponentSpec(kind="frozen", model_path=str(p)) for p in chosen]
