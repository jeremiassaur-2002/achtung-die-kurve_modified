"""Curriculum learning: difficulty ratchets up through fixed stages (opponent
count, then items) as the agent's rolling win rate clears each stage's threshold.
Transitions are soft-blended - once the threshold is hit, the probability of
drawing the *next* stage's composition ramps from 0 to 1 over `blend_window_episodes`
instead of switching all at once.

Stage progress (win-rate tracking, stage index) lives only in the main training
process - it's driven by CurriculumCallback in callbacks.py reading episode
results out of the VecEnv's `infos`. Since SubprocVecEnv workers are separate
processes, what actually reaches them is a small, plain-picklable snapshot
(`make_factory()`), refreshed via `env_method` whenever `record_episode()` reports
a state change. The snapshot bakes in a small pool of pre-sampled opponent-spec
variants (rather than one fixed list) so different parallel episodes still see
some variety between pushes.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Callable

from ai.env.opponents import OpponentSpec, SpecOpponentFactory


@dataclass
class CurriculumStage:
    name: str
    n_opponents: int
    opponent_specs_fn: Callable[[random.Random], list[OpponentSpec]]  # main-process only, never pickled
    enabled_items: set[str] | None = None
    win_rate_threshold: float = 0.6
    min_episodes: int = 200


@dataclass
class _SnapshotSpecs:
    pool: list[list[OpponentSpec]]
    next_pool: list[list[OpponentSpec]] | None
    blend_fraction: float

    def __call__(self) -> list[OpponentSpec]:
        if self.next_pool and random.random() < self.blend_fraction:
            return random.choice(self.next_pool)
        return random.choice(self.pool)


@dataclass
class _SnapshotItems:
    items: set[str] | None
    next_items: set[str] | None
    blend_fraction: float
    has_next: bool

    def __call__(self) -> set[str] | None:
        if self.has_next and random.random() < self.blend_fraction:
            return self.next_items
        return self.items


class CurriculumManager:
    def __init__(
        self,
        stages: list[CurriculumStage],
        blend_window_episodes: int = 100,
        rolling_window: int = 200,
        pool_variants: int = 8,
        rng_seed: int | None = None,
    ):
        if not stages:
            raise ValueError("need at least one stage")
        self.stages = stages
        self.blend_window_episodes = blend_window_episodes
        self.rolling_window = rolling_window
        self.pool_variants = pool_variants
        self.rng = random.Random(rng_seed)

        self.stage_idx = 0
        self._episodes_in_stage = 0
        self._recent_results: deque[bool] = deque(maxlen=rolling_window)
        self._blend_progress = 0  # 0 = not blending; counts up to blend_window_episodes

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.stage_idx]

    @property
    def is_last_stage(self) -> bool:
        return self.stage_idx >= len(self.stages) - 1

    def win_rate(self) -> float:
        if not self._recent_results:
            return 0.0
        return sum(self._recent_results) / len(self._recent_results)

    def record_episode(self, won: bool) -> bool:
        """Returns True if the effective opponent composition just changed (blend
        started, blend progressed, or a stage was committed) - i.e. push a fresh
        `make_factory()` snapshot to the workers."""
        self._recent_results.append(won)
        self._episodes_in_stage += 1
        changed = False

        if self._blend_progress > 0:
            self._blend_progress += 1
            changed = True
            if self._blend_progress >= self.blend_window_episodes:
                self.stage_idx += 1
                self._episodes_in_stage = 0
                self._recent_results.clear()
                self._blend_progress = 0
        elif (
            not self.is_last_stage
            and self._episodes_in_stage >= self.current_stage.min_episodes
            and self.win_rate() >= self.current_stage.win_rate_threshold
        ):
            self._blend_progress = 1
            changed = True

        return changed

    def make_factory(self, obs_cfg, rng_seed: int | None = None) -> SpecOpponentFactory:
        stage = self.current_stage
        pool = [stage.opponent_specs_fn(self.rng) for _ in range(self.pool_variants)]

        blend_fraction = self._blend_progress / self.blend_window_episodes if self._blend_progress else 0.0
        has_next = self._blend_progress > 0 and not self.is_last_stage
        next_pool = None
        next_items = None
        if has_next:
            next_stage = self.stages[self.stage_idx + 1]
            next_pool = [next_stage.opponent_specs_fn(self.rng) for _ in range(self.pool_variants)]
            next_items = next_stage.enabled_items

        specs_fn = _SnapshotSpecs(pool=pool, next_pool=next_pool, blend_fraction=blend_fraction)
        items_fn = _SnapshotItems(items=stage.enabled_items, next_items=next_items, blend_fraction=blend_fraction, has_next=has_next)
        return SpecOpponentFactory(specs_fn, obs_cfg, enabled_items=items_fn, rng_seed=rng_seed)
