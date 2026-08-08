"""Unit tests for the torch-free halves of the new training features: rotating
resumable checkpoints (run_persistence), the analytic border-doom termination,
the potential-based clearance shaping, and headless episode-video recording."""

import math
import random
from pathlib import Path

import pytest

from ai.core.config import game_constants as gc
from ai.core.env.curve_env import CurveEnv, CurveEnvConfig, RewardConfig
from ai.core.env.observation import ObsConfig
from ai.core.env.opponents import EpisodeConfig
from ai.v1_0.training.run_persistence import latest_checkpoint, record_episode_video, save_rotating_checkpoint


def _solo_factory() -> EpisodeConfig:
    return EpisodeConfig(opponents=[], enabled_items=set())


def _env(**reward_kw) -> CurveEnv:
    cfg = CurveEnvConfig(
        engine_resolution=256,
        obs=ObsConfig(obs_resolution=32, frame_stack=2),
        max_episode_ticks=600,
        reward=RewardConfig(**reward_kw),
    )
    env = CurveEnv(_solo_factory, config=cfg, seed=0)
    env.reset()
    return env


# ------------------------------------------------------------- checkpoints


class _FakeModel:
    def __init__(self, payload: bytes):
        self.payload = payload

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"x" * 2048 + self.payload)


def test_checkpoint_rotation_keeps_only_newest(tmp_path):
    for steps in (100, 200, 300):
        save_rotating_checkpoint(_FakeModel(str(steps).encode()), tmp_path, steps, keep=1)
        assert latest_checkpoint(tmp_path).name == f"ckpt_{steps}_steps.zip"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt_300_steps.zip"]

    # a stray .part from a crash mid-write is cleaned up by the next save
    (tmp_path / "ckpt_9_steps.zip.part").write_bytes(b"crash leftovers")
    save_rotating_checkpoint(_FakeModel(b"400"), tmp_path, 400, keep=1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt_400_steps.zip"]


def test_latest_checkpoint_empty_dir(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    assert latest_checkpoint(tmp_path / "does_not_exist") is None


# ---------------------------------------------------------------- doom check


def test_border_doom_geometry():
    env = _env(terminate_doomed_border=True)
    p = env.engine.players[env._hero_name]
    c = env.constants
    b = c.border_width + c.hitbox_size
    r_turn = c.move_speed / gc.TURN_SPEED  # size == speed == 1

    p.x, p.y, p.dir = 128.0, 128.0, 0.0  # mid-field, heading at the right wall
    assert not env._border_doomed(p)

    p.x = 256 - b - 3.0  # 3px in front of the wall, orthogonal: no turn can save this
    assert env._border_doomed(p)

    p.dir = math.pi / 2  # same spot but parallel to the wall - perfectly escapable
    assert not env._border_doomed(p)

    p.dir = 0.0
    p.x = 256 - b - (r_turn + 3.0)  # just outside the turning radius - escapable
    assert not env._border_doomed(p)


def test_doomed_step_terminates_with_penalty():
    env = _env(terminate_doomed_border=True, clearance_weight=0.0)
    p = env.engine.players[env._hero_name]
    b = env.constants.border_width + env.constants.hitbox_size
    p.x, p.y, p.dir = 256 - b - 5.0, 128.0, 0.0

    _obs, reward, terminated, _truncated, info = env.step(1)  # STRAIGHT
    assert terminated
    assert info["death_cause"] in ("border_doomed", "border")
    assert reward < -0.5  # death_penalty landed ~25 ticks before the actual wall hit


# ------------------------------------------------------------------ shaping


def test_clearance_shaping_telescopes_toward_wall():
    # gamma = 1 makes potential-based shaping telescope EXACTLY to phi_end - phi_start:
    # driving at a wall must sum to a clearly negative shaping total, and nothing about
    # the path in between (approach, turn away, approach again) can farm extra reward.
    env = _env(clearance_weight=1.0, shaping_gamma=1.0, clearance_rays=0, terminate_doomed_border=False)
    p = env.engine.players[env._hero_name]
    b = env.constants.border_width + env.constants.hitbox_size
    p.x, p.y, p.dir = 256 - b - 39.0, 128.0, 0.0
    env._prev_phi = env._clearance_phi()
    phi0 = env._prev_phi

    total, n = 0.0, 0
    for _ in range(50):
        _obs, rew, terminated, truncated, _info = env.step(1)
        total += rew
        n += 1
        assert not (terminated or truncated)

    shaping_sum = total - n * env.config.reward.alive_bonus
    assert shaping_sum == pytest.approx(env._prev_phi - phi0, abs=1e-6)
    assert shaping_sum < -0.5


def test_clearance_phi_sees_trails():
    env = _env(clearance_weight=1.0, clearance_rays=8)
    p = env.engine.players[env._hero_name]
    p.x, p.y, p.dir = 128.0, 128.0, 0.0
    far = env._clearance_phi()
    env.engine.grid[128, 140] = 5  # a foreign trail pixel 12px straight ahead
    near = env._clearance_phi()
    assert near < far


# -------------------------------------------------------------------- video


def test_record_episode_video(tmp_path):
    env = _env(clearance_weight=0.0)
    rng = random.Random(0)
    out = record_episode_video(env, lambda _obs: rng.randrange(3), tmp_path / "clip.mp4", max_ticks=90, fps=60, scale=2)
    assert out.exists()
    assert out.stat().st_size > 5_000
