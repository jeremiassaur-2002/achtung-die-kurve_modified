"""Checkpoint registry + opponent mixing for League training (Phase 4).

Checkpoints get copied in here at fixed step milestones (see callbacks.py's
LeagueCallback) and registered with an Elo rating. Sampling opponents mixes the
current model, skill/recency-weighted historical checkpoints, every rule-based
difficulty, and the random baseline - so training never faces just one frozen
version of itself, which is what keeps cyclic ("rock-paper-scissors") strategies
from taking over, per the brief.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from ai.env.observation import ObsConfig
from ai.env.opponents import OpponentSpec
from ai.training.elo import EloRatings

RULE_BASED_DIFFICULTIES = ("easy", "medium", "hard", "hunter")


@dataclass
class LeagueEntry:
    name: str
    path: str
    step: int
    created_at: str


class League:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "registry.json"
        self.elo = EloRatings()
        self.entries: dict[str, LeagueEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.registry_path.exists():
            return
        data = json.loads(self.registry_path.read_text())
        self.entries = {k: LeagueEntry(**v) for k, v in data.get("entries", {}).items()}
        self.elo = EloRatings.from_dict(data.get("elo", {}))

    def _save(self) -> None:
        data = {"entries": {k: asdict(v) for k, v in self.entries.items()}, "elo": self.elo.to_dict()}
        self.registry_path.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------- registry

    def add_checkpoint(self, model_path: str | Path, step: int, name: str | None = None) -> LeagueEntry:
        name = name or f"checkpoint_{step}"
        dest = self.root / f"{name}.zip"
        shutil.copy(str(model_path), str(dest))
        entry = LeagueEntry(name=name, path=str(dest), step=step, created_at=datetime.now(timezone.utc).isoformat())
        self.entries[name] = entry
        self.elo.ratings.setdefault(name, self.elo.default_rating)
        self._save()
        return entry

    def best(self) -> LeagueEntry | None:
        if not self.entries:
            return None
        return max(self.entries.values(), key=lambda e: self.elo.get(e.name))

    def record_match(self, placements: list[str]) -> dict[str, float]:
        """`placements`: league entry names (or "rule_based:<difficulty>" / "random" /
        "current") ordered best to worst. Persists updated ratings."""
        result = self.elo.update_from_placements(placements)
        self._save()
        return result

    # ------------------------------------------------------------- sampling

    def sample_opponent_specs(
        self,
        k: int,
        rng: random.Random,
        current_model_path: str | Path | None = None,
        include_rule_based: bool = True,
        include_random: bool = True,
    ) -> list[OpponentSpec]:
        pool: list[OpponentSpec] = []

        entries = sorted(self.entries.values(), key=lambda e: e.step)
        if entries:
            weights = list(range(1, len(entries) + 1))  # favor more recent checkpoints
            chosen = rng.choices(entries, weights=weights, k=min(k, len(entries) * 2))
            pool.extend(OpponentSpec(kind="frozen", model_path=e.path) for e in chosen)

        if current_model_path is not None:
            pool.append(OpponentSpec(kind="frozen", model_path=str(current_model_path)))
        if include_rule_based:
            pool.extend(OpponentSpec(kind="rule_based", difficulty=d) for d in RULE_BASED_DIFFICULTIES)
        if include_random:
            pool.append(OpponentSpec(kind="random"))

        rng.shuffle(pool)
        if len(pool) < k:
            pool.extend(OpponentSpec(kind="random") for _ in range(k - len(pool)))
        return pool[:k]
