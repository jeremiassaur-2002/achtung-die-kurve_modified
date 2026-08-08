"""Tests for ai/core/env/sensors.py and its wiring into observation/curve_env:

- rays measure the analytic border distance and detect stamped trails
- fresh own-tail pixels are exempt for rays and arcs (engine's grace rule)
- arc survival times behave monotonically approaching a wall, and reverse swaps L/R
- the doomed terminator only fires in genuinely lost states: every flagged state
  is attacked with a randomized escape search on a real engine copy, and none may
  survive meaningfully past the horizon
- vector observation shape matches vector_dim() and the env's declared space
"""

from __future__ import annotations

import copy
import math
import random

import numpy as np

from ai.core.config import game_constants as gc
from ai.core.config.game_constants import GameConstants
from ai.core.env import sensors
from ai.core.env.curve_env import CurveEnv, CurveEnvConfig, RewardConfig
from ai.core.env.engine import CurveEngine, STRAIGHT, TURN_LEFT, TURN_RIGHT
from ai.core.env.observation import ObsConfig, ObservationBuilder, vector_dim
from ai.core.env.opponents import EpisodeConfig


def _solo_engine(x=128.0, y=128.0, direction=0.0, seed=0) -> CurveEngine:
    eng = CurveEngine(GameConstants(256))
    eng.reset(["greenlee"], enabled_items=set(), seed=seed)
    p = eng.players["greenlee"]
    p.x, p.y, p.dir = x, y, direction
    return eng


def _stamp_wall(eng: CurveEngine, x0: int, x1: int, y0: int, y1: int, slot: int = 2, age: int = 10_000) -> None:
    """Paints a foreign (or old own) trail rectangle with an old stamp."""
    eng.grid[y0:y1, x0:x1] = slot
    eng.stamp[y0:y1, x0:x1] = eng.tick - age


def test_ray_hits_analytic_border():
    eng = _solo_engine(x=200.0, y=128.0, direction=0.0)  # looking +x at the right wall
    c = eng.c
    b = c.border_width + c.hitbox_size
    d = sensors.ray_distances(eng, "greenlee", n_rays=4, range_px=64.0)
    expected = ((256 - b) - 200.0) / 64.0
    assert abs(d[0] - expected) < 0.02, (d[0], expected)
    # ray 2 looks backward into open field: full range
    assert d[2] == 1.0


def test_ray_detects_trail_and_ignores_fresh_own_tail():
    eng = _solo_engine(x=100.0, y=128.0, direction=0.0)
    _stamp_wall(eng, 120, 122, 100, 156, slot=2)  # foreign trail 20px ahead
    d = sensors.ray_distances(eng, "greenlee", n_rays=8, range_px=64.0)
    assert abs(d[0] - 20.0 / 64.0) < 0.05, d[0]

    # same pixels as own slot, freshly stamped -> exempt, ray sees through
    eng2 = _solo_engine(x=100.0, y=128.0, direction=0.0)
    _stamp_wall(eng2, 120, 122, 100, 156, slot=1, age=0)
    d2 = sensors.ray_distances(eng2, "greenlee", n_rays=8, range_px=64.0)
    assert d2[0] > 20.0 / 64.0 + 0.1, d2[0]


def test_arc_ttc_monotone_toward_wall_and_full_in_open_field():
    c = GameConstants(256)
    horizon = 45
    eng_far = _solo_engine(x=128.0, y=128.0, direction=0.0)
    assert sensors.arc_survival_ticks(eng_far, "greenlee", horizon).max() == horizon

    prev = horizon + 1
    for x in (180.0, 220.0, 240.0):
        eng = _solo_engine(x=x, y=128.0, direction=0.0)
        s_ticks = int(sensors.arc_survival_ticks(eng, "greenlee", horizon)[1])  # STRAIGHT
        expected = ((256 - (c.border_width + c.hitbox_size)) - x) / c.move_speed
        assert s_ticks <= prev
        assert abs(s_ticks - expected) <= 2 or s_ticks == horizon, (x, s_ticks, expected)
        prev = s_ticks


def test_arc_ttc_reverse_swaps_left_right():
    # trail block up-left of the head: LEFT arc (screen-up when heading +x) dies sooner
    eng = _solo_engine(x=100.0, y=128.0, direction=0.0)
    _stamp_wall(eng, 100, 130, 100, 118, slot=2)
    ttc = sensors.arc_survival_ticks(eng, "greenlee", 45)
    assert ttc[0] < ttc[2], ttc

    eng.players["greenlee"].reverse = 1
    eng._sensor_cache = None  # same tick, state changed manually -> drop memo
    ttc_rev = sensors.arc_survival_ticks(eng, "greenlee", 45)
    assert ttc_rev[0] == ttc[2] and ttc_rev[2] == ttc[0], (ttc, ttc_rev)


def test_vector_dim_and_env_space_match():
    cfg = ObsConfig(obs_resolution=64, frame_stack=2, n_rays=12, ray_range_px=48.0, arc_horizon=30)
    eng = _solo_engine()
    builder = ObservationBuilder(cfg)
    obs = builder.observe(eng, "greenlee")
    assert obs["vector"].shape == (vector_dim(cfg),)
    assert obs["vector"].shape[0] == 16 + gc.MAX_PLAYERS + 12 + 3
    rays = obs["vector"][16 + gc.MAX_PLAYERS : 16 + gc.MAX_PLAYERS + 12]
    ttc = obs["vector"][-3:]
    assert np.all(rays >= 0.0) and np.all(rays <= 1.0)
    assert np.all(ttc >= 0.0) and np.all(ttc <= 1.0)

    env = CurveEnv(lambda: EpisodeConfig(opponents=[], enabled_items=set()),
                   config=CurveEnvConfig(obs=cfg), seed=3)
    o, _ = env.reset(seed=3)
    assert env.observation_space["vector"].shape == o["vector"].shape


def _random_escape_exists(eng: CurveEngine, name: str, tries: int, length: int, rng: random.Random) -> bool:
    """True if ANY random action sequence keeps the player alive for `length` engine
    ticks - run on real engine copies, i.e. with the exact game rules."""
    base = copy.deepcopy(eng)
    base._sensor_cache = None
    actions = (TURN_LEFT, STRAIGHT, TURN_RIGHT)
    strategies = [[TURN_LEFT] * length, [STRAIGHT] * length, [TURN_RIGHT] * length]
    strategies += [[actions[rng.randrange(3)] for _ in range(length)] for _ in range(tries)]
    for seq in strategies:
        sim = copy.deepcopy(base)
        p = sim.players[name]
        ok = True
        for a in seq:
            sim.step({name: a})
            if not p.alive:
                ok = False
                break
        if ok:
            return True
    return False


def test_doomed_flag_has_no_random_escape():
    """Statistical guarantee check for terminate_doomed_any: collect states flagged
    doomed during random play; a randomized escape search (120 sequences on real
    engine copies, doom_horizon + grace extra ticks long) must fail for all of them."""
    doom_horizon = 12
    horizon = 45
    rng = random.Random(0)
    flagged = []
    for ep in range(12):
        eng = CurveEngine(GameConstants(256))
        eng.reset(["greenlee"], enabled_items=set(), seed=100 + ep)
        p = eng.players["greenlee"]
        while p.alive and eng.tick < 3600 and len(flagged) < 15:
            best = int(sensors.arc_survival_ticks(eng, "greenlee", horizon).max())
            if best <= doom_horizon:
                flagged.append(copy.deepcopy(eng))
                break  # one doomed state per episode is enough
            eng.step({"greenlee": rng.choice((TURN_LEFT, STRAIGHT, TURN_RIGHT))})
    assert flagged, "random play never reached a doomed state - test setup broken?"
    for f in flagged:
        assert not _random_escape_exists(f, "greenlee", tries=120, length=doom_horizon + 10, rng=rng), \
            "a state flagged as doomed had a survivable action sequence"


def test_env_doom_termination_and_shaping_run():
    cfg = CurveEnvConfig(
        reward=RewardConfig(clearance_weight=0.05, horizon_weight=0.08,
                            terminate_doomed_any=True, doom_horizon=12,
                            terminate_doomed_border=True),
    )
    env = CurveEnv(lambda: EpisodeConfig(opponents=[], enabled_items=set()), config=cfg, seed=11)
    rng = random.Random(11)
    causes = []
    for _ in range(6):
        env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            _, reward, terminated, truncated, info = env.step(rng.choice((0, 1, 2)))
            assert math.isfinite(reward)
        if "death_cause" in info:
            causes.append(info["death_cause"])
    assert causes, "no episode ended with a death cause"
    # under random play, early doom/border_doomed termination should dominate raw impacts
    assert any(c in ("doomed", "border_doomed") for c in causes), causes
