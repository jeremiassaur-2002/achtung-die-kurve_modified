"""Controllers drive every non-hero seat inside CurveEnv. This is what makes
self-play/multiplayer/league possible on top of a single-agent SB3 algorithm:
the Gym env always trains exactly one seat (the hero); every other seat is
whatever Controller the training script plugs in for that episode.

All three controllers share one interface, so curriculum.py/league.py can swap
opponent composition per-episode without curve_env.py knowing the difference.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ai.env.engine import CurveEngine, STRAIGHT, TURN_LEFT, TURN_RIGHT
from ai.env.observation import ObsConfig, ObservationBuilder
from ai.env.rules_bot import BOT_DIFFICULTIES, RuleBasedBot

_ACTIONS = (TURN_LEFT, STRAIGHT, TURN_RIGHT)


class Controller(ABC):
    """One instance per seat; `reset()` at the start of each episode, `act()` once
    per tick using the state as of *before* that tick is simulated."""

    @abstractmethod
    def reset(self, seat_name: str) -> None: ...

    @abstractmethod
    def act(self, engine: CurveEngine, name: str, frame_hwc: np.ndarray | None) -> int: ...


class RandomController(Controller):
    """Uniform-random action every tick - the weakest baseline opponent, used in
    curriculum/league mixing and as an evaluation floor."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def reset(self, seat_name: str) -> None:
        pass

    def act(self, engine: CurveEngine, name: str, frame_hwc: np.ndarray | None) -> int:
        return self.rng.choice(_ACTIONS)


class RuleBasedController(Controller):
    """Wraps the fresh forward-simulation heuristic in rules_bot.py at a fixed
    difficulty preset."""

    def __init__(self, difficulty: str = "medium", rng: random.Random | None = None):
        if difficulty not in BOT_DIFFICULTIES:
            raise ValueError(f"unknown difficulty {difficulty!r}, expected one of {list(BOT_DIFFICULTIES)}")
        self.difficulty = difficulty
        self.rng = rng
        self._bot: RuleBasedBot | None = None

    def reset(self, seat_name: str) -> None:
        self._bot = RuleBasedBot(self.difficulty, self.rng)

    def act(self, engine: CurveEngine, name: str, frame_hwc: np.ndarray | None) -> int:
        assert self._bot is not None, "call reset() before act()"
        return self._bot.decide(engine, name)


class FrozenPolicyController(Controller):
    """A previously-trained (or currently-training, snapshotted) SB3 policy,
    run read-only for inference. Used for self-play, league opponent mixing, and
    evaluation-vs-older-checkpoints. Builds the exact same {image, vector}
    observation the hero uses (see ai/env/observation.py) so any checkpoint can
    be dropped in as an opponent regardless of which seat it occupies.
    """

    def __init__(self, model_path: str, obs_cfg: ObsConfig, deterministic: bool = False, device: str = "cpu"):
        from stable_baselines3.common.base_class import BaseAlgorithm
        from stable_baselines3 import PPO

        try:
            self._model: BaseAlgorithm = PPO.load(model_path, device=device)
        except Exception:
            from sb3_contrib import MaskablePPO

            self._model = MaskablePPO.load(model_path, device=device)
        self.obs_cfg = obs_cfg
        self.deterministic = deterministic
        self._builder: ObservationBuilder | None = None

    def reset(self, seat_name: str) -> None:
        self._builder = ObservationBuilder(self.obs_cfg)

    def act(self, engine: CurveEngine, name: str, frame_hwc: np.ndarray | None) -> int:
        assert self._builder is not None, "call reset() before act()"
        obs = self._builder.observe(engine, name, frame_hwc)
        batched = {k: v[np.newaxis, ...] for k, v in obs.items()}
        action, _ = self._model.predict(batched, deterministic=self.deterministic)
        return int(np.asarray(action).reshape(-1)[0])


@dataclass
class EpisodeConfig:
    """What `CurveEnv.reset()` needs for the next episode: who plays the other
    seats, and which items are enabled. Returned as one object (not two separate
    calls) so both are applied atomically at reset - see curriculum.py, where the
    two need to change together as stages advance."""

    opponents: list["Controller"]
    enabled_items: set[str] | None = None


@dataclass(frozen=True)
class OpponentSpec:
    """Small, picklable description of a controller - crosses SubprocVecEnv worker
    boundaries safely (unlike a live Controller instance, which for FrozenPolicy
    would mean re-pickling a whole loaded model on every curriculum update)."""

    kind: str  # "random" | "rule_based" | "frozen"
    difficulty: str | None = None  # rule_based
    model_path: str | None = None  # frozen


def build_controller(spec: OpponentSpec, obs_cfg: ObsConfig, rng: random.Random | None = None) -> Controller:
    if spec.kind == "random":
        return RandomController(rng)
    if spec.kind == "rule_based":
        return RuleBasedController(spec.difficulty or "medium", rng)
    if spec.kind == "frozen":
        assert spec.model_path is not None
        return FrozenPolicyController(spec.model_path, obs_cfg)
    raise ValueError(f"unknown opponent kind {spec.kind!r}")


class SpecOpponentFactory:
    """The `opponent_factory` CurveEnv actually receives. Holds either a fixed list
    of OpponentSpec or a callable producing one (for randomized composition), and
    caches FrozenPolicyController instances per model_path so repeated episode
    resets don't reload the same checkpoint from disk over and over."""

    def __init__(
        self,
        specs: list[OpponentSpec] | Callable[[], list[OpponentSpec]],
        obs_cfg: ObsConfig,
        enabled_items: "set[str] | None | Callable[[], set[str] | None]" = None,
        rng_seed: int | None = None,
    ):
        self._specs = specs
        self._enabled_items = enabled_items
        self.obs_cfg = obs_cfg
        self._rng = random.Random(rng_seed)
        self._frozen_cache: dict[str, FrozenPolicyController] = {}

    def __call__(self) -> EpisodeConfig:
        specs = self._specs() if callable(self._specs) else self._specs
        controllers: list[Controller] = []
        for spec in specs:
            if spec.kind == "frozen":
                assert spec.model_path is not None
                ctrl = self._frozen_cache.get(spec.model_path)
                if ctrl is None:
                    ctrl = FrozenPolicyController(spec.model_path, self.obs_cfg)
                    self._frozen_cache[spec.model_path] = ctrl
                controllers.append(ctrl)
            else:
                controllers.append(build_controller(spec, self.obs_cfg, self._rng))
        items = self._enabled_items() if callable(self._enabled_items) else self._enabled_items
        return EpisodeConfig(opponents=controllers, enabled_items=items)
