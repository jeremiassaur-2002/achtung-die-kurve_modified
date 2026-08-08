"""Shared observation construction: image (frame-stacked RGB) + vector (metadata).

Used both by curve_env.py (for the hero, the agent SB3 is actually training) and by
opponents.py's FrozenPolicyController (any self-play/league opponent that is itself
a loaded policy needs the exact same observation shape it was trained with). Kept
here, not duplicated, so hero and frozen-opponent observations can never drift apart.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from ai.core.config import game_constants as gc
from ai.core.env import renderer, sensors
from ai.core.env.engine import CurveEngine

# vector layout: [speed, size, cos(dir), sin(dir), x_norm, y_norm, wall_dist_norm,
#   reverse?, invisible?, side?, ghost?, freeze?, sine?, alive_opponents_frac,
#   field_inset_norm, sides_active?] + one-hot(which color slot is "mine")
#   + n_rays heading-relative lidar distances (sensors.ray_distances)
#   + 3 arc survival times for hold-LEFT / STRAIGHT / hold-RIGHT (sensors.
#     arc_survival_ticks / arc_horizon) - the explicit "if I keep doing X, I die
#     in t ticks" lookahead the downscaled CNN image cannot resolve near the head
VECTOR_BASE_DIM = 16


def vector_dim(cfg: "ObsConfig") -> int:
    return VECTOR_BASE_DIM + gc.MAX_PLAYERS + cfg.n_rays + 3


@dataclass
class ObsConfig:
    obs_resolution: int = 96
    frame_stack: int = 4
    # sensor features appended to the vector (see module docstring / sensors.py);
    # n_rays = 0 disables rays, arc_horizon must stay > 0 (the env's doomed check
    # and horizon shaping read the same three values)
    n_rays: int = 16
    ray_range_px: float = 64.0
    arc_horizon: int = 45

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return (3 * self.frame_stack, self.obs_resolution, self.obs_resolution)


class ObservationBuilder:
    """One instance per seat per episode - holds that seat's frame-stack history."""

    def __init__(self, cfg: ObsConfig):
        self.cfg = cfg
        self._frames: deque[np.ndarray] = deque(maxlen=cfg.frame_stack)

    def reset(self) -> None:
        self._frames.clear()

    def observe(self, engine: CurveEngine, name: str, frame_hwc: np.ndarray | None = None) -> dict[str, np.ndarray]:
        if frame_hwc is None:
            frame_hwc = renderer.render_frame(engine, self.cfg.obs_resolution)
        chw = renderer.to_chw(frame_hwc)
        if not self._frames:
            for _ in range(self.cfg.frame_stack):
                self._frames.append(chw)
        else:
            self._frames.append(chw)
        image = np.concatenate(list(self._frames), axis=0)
        vector = _build_vector(engine, name, self.cfg)
        return {"image": image, "vector": vector}


def _build_vector(engine: CurveEngine, name: str, cfg: ObsConfig) -> np.ndarray:
    p = engine.players[name]
    s = engine.c.engine_resolution
    half = s / 2.0
    alive_others = sum(1 for q in engine.players.values() if q.alive and q.name != name)
    max_others = max(1, gc.MAX_PLAYERS - 1)
    wall_dist = min(p.x, s - p.x, p.y, s - p.y) / half

    base = np.array(
        [
            p.speed,
            p.size,
            math.cos(p.dir),
            math.sin(p.dir),
            (p.x / s) * 2 - 1,
            (p.y / s) * 2 - 1,
            wall_dist,
            1.0 if p.reverse else 0.0,
            1.0 if p.invisible else 0.0,
            1.0 if p.side else 0.0,
            1.0 if p.ghost else 0.0,
            1.0 if p.freeze else 0.0,
            1.0 if p.sine_start is not None else 0.0,
            alive_others / max_others,
            engine.field_inset / half,
            1.0 if engine.sides else 0.0,
        ],
        dtype=np.float32,
    )
    # one-hot of *which of the 6 canonical colors* this player is - keyed on p.name,
    # not p.slot: slot is just this episode's participation order (the hero is always
    # slot 1, which would make this constant/uninformative), while name<->color is
    # what actually varies per episode (curve_env.py shuffles seat/color assignment)
    # and is what the CNN can learn to associate with its own trail's color.
    self_onehot = np.zeros(gc.MAX_PLAYERS, dtype=np.float32)
    self_onehot[gc.PLAYER_NAMES.index(p.name)] = 1.0

    rays = (
        sensors.ray_distances(engine, name, cfg.n_rays, cfg.ray_range_px)
        if cfg.n_rays > 0
        else np.zeros(0, dtype=np.float32)
    )
    ttc = sensors.arc_survival_ticks(engine, name, cfg.arc_horizon).astype(np.float32) / cfg.arc_horizon
    return np.concatenate([base, self_onehot, rays, ttc])
