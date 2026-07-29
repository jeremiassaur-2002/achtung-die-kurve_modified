"""Elo rating for the league. The game is free-for-all, not 1v1, so a match's
finishing order is decomposed into every pairwise "who placed better" comparison
and each pair is scored as an independent 1v1 Elo update, then averaged per
participant - a standard, simple way to get an Elo number out of FFA placements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EloRatings:
    default_rating: float = 1000.0
    k_factor: float = 24.0
    ratings: dict[str, float] = field(default_factory=dict)

    def get(self, name: str) -> float:
        return self.ratings.get(name, self.default_rating)

    def update_from_placements(self, placements: list[str]) -> dict[str, float]:
        """`placements`: participant ids ordered best (index 0) to worst. Returns
        the new rating for every participant involved."""
        n = len(placements)
        if n < 2:
            return {p: self.get(p) for p in placements}

        deltas = {p: 0.0 for p in placements}
        comparisons = {p: 0 for p in placements}
        for i in range(n):
            for j in range(i + 1, n):
                a, b = placements[i], placements[j]  # a finished better than b
                ra, rb = self.get(a), self.get(b)
                expected_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
                deltas[a] += self.k_factor * (1.0 - expected_a)
                deltas[b] += self.k_factor * (0.0 - (1.0 - expected_a))
                comparisons[a] += 1
                comparisons[b] += 1

        for p in placements:
            if comparisons[p] > 0:
                self.ratings[p] = self.get(p) + deltas[p] / comparisons[p]
        return {p: self.get(p) for p in placements}

    def to_dict(self) -> dict:
        return {"default_rating": self.default_rating, "k_factor": self.k_factor, "ratings": self.ratings}

    @classmethod
    def from_dict(cls, d: dict) -> "EloRatings":
        return cls(default_rating=d.get("default_rating", 1000.0), k_factor=d.get("k_factor", 24.0), ratings=dict(d.get("ratings", {})))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "EloRatings":
        return cls.from_dict(json.loads(Path(path).read_text()))
