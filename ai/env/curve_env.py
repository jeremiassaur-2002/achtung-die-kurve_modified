"""Gymnasium environment wrapping CurveEngine for exactly one learning agent
("the hero"). Every other seat is driven by a Controller (ai/env/opponents.py) -
random, rule-based, or a frozen policy snapshot - so curriculum.py/self_play.py/
league.py can reshape opponent composition per-episode without this class
changing. One Gym episode = one round (play until the hero dies or only one
player is left standing); the point-goal/win-margin *match* structure from the
original game is a meta-game layer evaluation/arena.py composes on top of rounds,
not something the per-tick RL loop needs to model.

Headless by construction: there is no rendering path here that opens a window -
`render_frame()` only ever produces a numpy array.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai.config import game_constants as gc
from ai.config.game_constants import GameConstants
from ai.env import renderer
from ai.env.engine import CurveEngine, STRAIGHT, TURN_LEFT, TURN_RIGHT, _mcos, _msin
from ai.env.observation import ObsConfig, ObservationBuilder, VECTOR_DIM
from ai.env.opponents import Controller, EpisodeConfig

_ACTIONS = (TURN_LEFT, STRAIGHT, TURN_RIGHT)
_DELTA = {TURN_LEFT: -1, STRAIGHT: 0, TURN_RIGHT: 1}


@dataclass
class RewardConfig:
    alive_bonus: float = 0.01
    death_penalty: float = -1.0
    kill_bonus: float = 0.5
    win_bonus: float = 2.0
    placement_scale: float = 0.3  # partial credit for outlasting some (not all) opponents

    # --- optional potential-based clearance shaping (off by default) ---
    # ONE term instead of separate "stay centered" / "approach wall and steer away" /
    # "drive parallel" bonuses: r_shape = w * (gamma * phi(s') - phi(s)), with phi =
    # normalized distance to the nearest lethal thing (border or any trail, probed by
    # short rays over the grid). Potential-based shaping provably leaves the optimal
    # policy unchanged (Ng et al. 1999) and - unlike event bonuses - cannot be farmed
    # by repeatedly approaching an obstacle just to get paid for turning away again.
    clearance_weight: float = 0.0  # 0 = off; ~0.05 is a sensible Phase-1 value
    clearance_range_px: float = 40.0  # phi saturates ("safe enough") at this distance
    clearance_rays: int = 8  # trail-probing rays; 0 = border-distance-only potential
    shaping_gamma: float = 0.99  # keep equal to ppo.gamma

    # --- doomed-state termination ---
    # Ends the episode (with death_penalty) the moment the border is PROVABLY
    # unavoidable given the turning radius r = v/omega: a max-rate turn advances
    # r*(1-cos beta) toward a wall before the heading becomes parallel (beta = angle
    # between heading and wall). Exact for a single straight wall; corners can only
    # add extra doomed states, never rescue one, and a 1px margin absorbs the
    # engine's discrete-tick/rounding quirks - so false "doomed" calls can't happen.
    # This moves the death signal ~25 ticks earlier, right onto the decision that
    # actually caused it, which is exactly where PPO's credit assignment needs it.
    terminate_doomed_border: bool = False


@dataclass
class CurveEnvConfig:
    engine_resolution: int = 256
    obs: ObsConfig = field(default_factory=ObsConfig)
    enabled_items: set[str] | None = None  # None = all items enabled
    use_action_masking: bool = False
    action_mask_horizon: int = 3  # ticks - "certainly fatal within N ticks" per the brief's own example
    max_episode_ticks: int = 3600  # 60s @ 60Hz safety cap against unbounded standoffs
    reward: RewardConfig = field(default_factory=RewardConfig)


OpponentFactory = Callable[[], EpisodeConfig]


class CurveEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, opponent_factory: OpponentFactory, config: CurveEnvConfig | None = None, seed: int | None = None):
        super().__init__()
        self.config = config or CurveEnvConfig()
        self._opponent_factory = opponent_factory
        self._rng = random.Random(seed)

        self.constants = GameConstants(self.config.engine_resolution)
        self.engine = CurveEngine(self.constants)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=self.config.obs.image_shape, dtype=np.uint8),
                # every component of _build_vector (observation.py) is normalized/flag-like
                # and stays within [-1, 1]; a finite bound avoids gymnasium's "-inf/inf Box" warning
                "vector": spaces.Box(low=-10.0, high=10.0, shape=(VECTOR_DIM,), dtype=np.float32),
            }
        )

        self._hero_name: str | None = None
        self._opponents: dict[str, Controller] = {}
        self._hero_builder = ObservationBuilder(self.config.obs)
        self._last_frame: np.ndarray | None = None
        self._tick_in_episode = 0
        self._prev_phi: float | None = None

    # ------------------------------------------------------------ reconfigure

    def set_opponent_factory(self, fn: OpponentFactory) -> None:
        """Lets curriculum.py/league.py swap opponent count/composition between episodes."""
        self._opponent_factory = fn

    def set_enabled_items(self, items: set[str] | None) -> None:
        """Manual override; normally the opponent_factory's EpisodeConfig drives this
        per-episode (see curriculum.py), which is why reset() re-derives it each time."""
        self.config.enabled_items = items

    # --------------------------------------------------------------- gym api

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)

        episode_config = self._opponent_factory()
        opponents = list(episode_config.opponents)
        self.config.enabled_items = episode_config.enabled_items
        n_players = 1 + len(opponents)
        if not (1 <= n_players <= gc.MAX_PLAYERS):
            raise ValueError(f"opponent_factory produced {len(opponents)} opponents -> {n_players} players, need 1..{gc.MAX_PLAYERS}")

        names = list(gc.PLAYER_NAMES)
        self._rng.shuffle(names)
        active_names = names[:n_players]
        self._hero_name = active_names[0]
        self._opponents = dict(zip(active_names[1:], opponents))

        episode_seed = self._rng.randrange(2**31)
        self.engine.reset(active_names, enabled_items=self.config.enabled_items, seed=episode_seed)
        self._tick_in_episode = 0
        self._prev_phi = self._clearance_phi() if self.config.reward.clearance_weight != 0.0 else None

        self._hero_builder.reset()
        for seat_name, ctrl in self._opponents.items():
            ctrl.reset(seat_name)

        self._last_frame = renderer.render_frame(self.engine, self.config.obs.obs_resolution)
        obs = self._hero_builder.observe(self.engine, self._hero_name, self._last_frame)
        return obs, {}

    def step(self, action: int):
        assert self._hero_name is not None, "call reset() before step()"
        actions = {self._hero_name: int(action)}
        for seat_name, ctrl in self._opponents.items():
            if self.engine.players[seat_name].alive:
                actions[seat_name] = ctrl.act(self.engine, seat_name, self._last_frame)

        tick_infos = self.engine.step(actions)
        self._tick_in_episode += 1

        self._last_frame = renderer.render_frame(self.engine, self.config.obs.obs_resolution)
        obs = self._hero_builder.observe(self.engine, self._hero_name, self._last_frame)

        hero_info = tick_infos[self._hero_name]
        r = self.config.reward
        reward = 0.0
        terminated = False
        info: dict = {}

        if hero_info.alive:
            reward += r.alive_bonus
        for name, ti in tick_infos.items():
            if name != self._hero_name and ti.just_died and ti.death_cause == self._hero_name:
                reward += r.kill_bonus

        if hero_info.just_died:
            reward += r.death_penalty
            terminated = True
            total_opponents = len(self._opponents)
            if total_opponents > 0:
                dead_before = sum(1 for name in self._opponents if not self.engine.players[name].alive)
                reward += r.placement_scale * (dead_before / total_opponents)
            info["death_cause"] = hero_info.death_cause
        elif self._opponents and self.engine.is_episode_over():
            # hero outlasted every opponent (solo/Phase-1 episodes have no opponents to
            # outlast, so "alive_count <= 1" must never end them - only death/truncation does)
            terminated = True
            reward += r.win_bonus
            info["won"] = True

        if not terminated and hero_info.alive and r.terminate_doomed_border:
            hero_p = self.engine.players[self._hero_name]
            wrap = self.engine.sides != 0 or hero_p.side != 0
            if not wrap and self._border_doomed(hero_p):
                terminated = True
                reward += r.death_penalty
                total_opponents = len(self._opponents)
                if total_opponents > 0:
                    dead_before = sum(1 for name in self._opponents if not self.engine.players[name].alive)
                    reward += r.placement_scale * (dead_before / total_opponents)
                info["death_cause"] = "border_doomed"  # shows up separately in the metrics breakdown

        if self._prev_phi is not None:
            # potential-based shaping; phi(terminal) := 0 is the standard PBRS convention
            new_phi = 0.0 if terminated else self._clearance_phi()
            reward += r.clearance_weight * (r.shaping_gamma * new_phi - self._prev_phi)
            self._prev_phi = new_phi

        truncated = self._tick_in_episode >= self.config.max_episode_ticks and not terminated
        if terminated or truncated:
            info["episode_ticks"] = self._tick_in_episode
            info["kills"] = self.engine.players[self._hero_name].kills
            info["items_collected"] = self.engine.players[self._hero_name].items_collected
            info["alive_opponents_remaining"] = sum(1 for name in self._opponents if self.engine.players[name].alive)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._last_frame is None:
            return None
        return self._last_frame

    # ----------------------------------------------------- shaping / doom check

    def _clearance_phi(self) -> float:
        """Normalized distance in [0, 1] from the hero to the nearest lethal thing:
        the border (analytic) and, via `clearance_rays` short grid probes, any trail.
        Freshly drawn own-tail pixels are skipped with the engine's own grace logic,
        otherwise the tail right behind the head would clamp phi permanently."""
        p = self.engine.players[self._hero_name]
        c = self.constants
        e = self.engine
        r_cfg = self.config.reward
        rng_px = r_cfg.clearance_range_px
        s = c.engine_resolution
        b = c.border_width + c.hitbox_size + e.field_inset

        wrap = e.sides != 0 or p.side != 0
        d = rng_px if wrap else min(p.x - b, (s - b) - p.x, p.y - b, (s - b) - p.y)

        step_px = 2.0
        for k in range(max(0, r_cfg.clearance_rays)):
            ang = p.dir + (k / max(1, r_cfg.clearance_rays)) * 2.0 * math.pi
            ca, sa = math.cos(ang), math.sin(ang)
            dist = step_px
            while dist <= rng_px and dist < d:
                sx, sy = p.x + ca * dist, p.y + sa * dist
                owner = e.grid_at(sx, sy)
                if owner != 0 and not (owner == p.slot and e._own_trail_is_fresh(p, sx, sy)):
                    d = dist
                    break
                dist += step_px
        return max(0.0, min(1.0, d / rng_px))

    def _border_doomed(self, p) -> bool:
        """True iff hitting the border is already unavoidable (see RewardConfig).
        Conservative by construction: item states that change the turning radius
        mid-arc (sine) or disable turning (freeze) simply opt out, and the 1px
        margin means a state is only ever flagged when it is strictly inescapable."""
        if p.freeze != 0 or p.sine_start is not None:
            return False
        step = gc.TURN_SPEED / (p.size ** gc.SIZE_TURN_EXPONENT)
        if step <= 0:
            return False
        c = self.constants
        r_turn = (c.move_speed * p.speed) / step  # minimum turning radius in px
        b = c.border_width + c.hitbox_size + self.engine.field_inset
        s = c.engine_resolution
        cx, sy = _mcos(p.dir), _msin(p.dir)
        margin = 1.0
        # (approach component toward that wall, perpendicular distance to it);
        # approach == sin(beta), beta = angle between heading and the wall plane
        walls = ((-cx, p.x - b), (cx, (s - b) - p.x), (-sy, p.y - b), (sy, (s - b) - p.y))
        for approach, dist in walls:
            if approach <= 0.0:
                continue  # moving away from / parallel to this wall
            cos_beta = math.sqrt(max(0.0, 1.0 - approach * approach))
            needed = r_turn * (1.0 - cos_beta)
            if dist < needed - margin:
                return True
        return False

    # --------------------------------------------------------- action masking

    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO integration point. Masks an action only if it is
        certainly fatal within `action_mask_horizon` ticks (straight-line continuation
        after the turn, checked against the current owner-id grid) - never masks all
        three (falls back to fully unmasked if every action looks doomed)."""
        assert self._hero_name is not None
        p = self.engine.players[self._hero_name]
        if not p.alive:
            return np.ones(3, dtype=bool)

        mask = np.zeros(3, dtype=bool)
        for action in _ACTIONS:
            mask[action] = self._survives_short_horizon(p.name, _DELTA[action])
        if not mask.any():
            return np.ones(3, dtype=bool)
        return mask

    def _survives_short_horizon(self, name: str, delta: int) -> bool:
        c = self.constants
        e = self.engine
        p = e.players[name]
        mv = c.move_speed * p.speed
        step = gc.TURN_SPEED / (p.size ** gc.SIZE_TURN_EXPONENT)
        b = c.border_width + c.hitbox_size + e.field_inset
        s = c.engine_resolution
        wrap = e.sides != 0 or p.side != 0

        sx, sy, sd = p.x, p.y, p.dir
        for _ in range(self.config.action_mask_horizon):
            sd += delta * step
            sx += _mcos(sd) * mv
            sy += _msin(sd) * mv
            if wrap:
                sx %= s
                sy %= s
            elif sx < b or sx > s - b or sy < b or sy > s - b:
                return False
            if e.grid_at(sx, sy) != 0 and not (p.ghost != 0 and e.grid_at(sx, sy) == p.slot):
                return False
        return True
