"""CurveEngine: a from-scratch, headless reimplementation of the "Achtung, die
Kurve!" simulation (script.js), driven tick-by-tick with no rendering of its own.

This is a faithful port, not an approximation: every formula, ordering quirk and
timer here has a direct counterpart in the JS source (see ai/core/config/game_constants.py
for the extracted ratios/timers, with the Phase 0 balance changes already applied:
g_robot/r_robot removed, r_freeze shortened to 500ms, r_swap swaps heading too).

One deliberate deviation: instead of the browser's RGBA-pixel-matching hitbox hack,
trails are stamped into an integer *owner-id* grid (0 = empty, 1..N = player slot).
Behaviorally this is identical (same collision rule, same "not my own trail while
ghost" exception) but it also gives exact, free attribution of "who hit whom" for
kill stats, and is what ai/core/env/renderer.py recolors into the CNN's RGB observation.

Action encoding (matches the game's real turnL/turnR keyboard interface):
    TURN_LEFT = 0, STRAIGHT = 1, TURN_RIGHT = 2
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from ai.core.config import game_constants as gc
from ai.core.config.game_constants import GameConstants

TURN_LEFT = 0
STRAIGHT = 1
TURN_RIGHT = 2

_DEG55 = math.radians(gc.LEFT_RIGHT_ANGLE_DEG)


def _mcos(angle: float) -> float:
    # script.js mathCos: cos rounded to 2 decimals before use, for both movement and
    # every collision sample point - a genuine (if minor) precision-limiting quirk of
    # the original that's reproduced here rather than silently "cleaned up".
    return round(math.cos(angle) * 100) / 100


def _msin(angle: float) -> float:
    return round(math.sin(angle) * 100) / 100


@dataclass
class ItemOnScreen:
    kind: str
    x: float
    y: float


@dataclass
class PlayerState:
    name: str
    slot: int  # 1-based index into the grid's owner-id space (0 = empty)

    x: float = 0.0
    y: float = 0.0
    dir: float = 0.0

    alive: bool = True
    score: int = 0
    turn_l: bool = False
    turn_r: bool = False

    bridge: bool = False
    bridge_frame: int = 0

    # powerup state (script.js players[p].powerup.*)
    size: float = 1.0
    speed: float = 1.0
    reverse: int = 0
    invisible: int = 0
    side: int = 0
    ghost: int = 0
    freeze: int = 0
    sine_start: int | None = None

    # stats, for evaluation/reporting
    kills: int = 0
    items_collected: int = 0
    death_cause: str | None = None  # "border" | "self" | <other player's name> | None
    death_tick: int | None = None

    @property
    def color(self) -> tuple[int, int, int]:
        return gc.PLAYER_COLORS[self.name]

    @property
    def head_color(self) -> tuple[int, int, int]:
        return gc.HEAD_COLOR_REVERSED if self.reverse != 0 else gc.HEAD_COLOR_NORMAL


@dataclass
class PlayerTickInfo:
    alive: bool
    just_died: bool
    death_cause: str | None
    items_collected: list[str] = field(default_factory=list)


class CurveEngine:
    """Owns the authoritative game state for one episode. Call `reset()` once,
    then `step(actions)` once per tick with a `{player_name: action}` dict for
    every currently-alive participant."""

    def __init__(self, constants: GameConstants):
        self.c = constants
        self.rng = random.Random()
        self.tick = 0
        self.grid: np.ndarray = np.zeros((constants.engine_resolution, constants.engine_resolution), dtype=np.uint8)
        self.stamp: np.ndarray = np.full_like(self.grid, -(10**9), dtype=np.int32)  # tick each pixel was drawn
        self.players: dict[str, PlayerState] = {}
        self.items_on_screen: list[ItemOnScreen] = []
        self.enabled_items: set[str] = set(gc.ALL_ITEMS)
        self.sides = 0  # achtung.sides: global wrap toggle, stacks via +=/-=
        self.field_inset = 0.0
        self._scheduled: list[tuple[int, Callable[[], None]]] = []
        self._pending_spawns: list[int] = []

    # ------------------------------------------------------------------ setup

    def reset(self, active_players: list[str], enabled_items: set[str] | None = None, seed: int | None = None) -> None:
        # 1 player is valid (Phase 1 solo survival - no one to award points to, but
        # border/self-trail collision still apply); scoring/win logic just never fires.
        if not (1 <= len(active_players) <= gc.MAX_PLAYERS):
            raise ValueError(f"need 1..{gc.MAX_PLAYERS} players, got {len(active_players)}")
        self.rng = random.Random(seed)
        self.tick = 0
        s = self.c.engine_resolution
        self.grid = np.zeros((s, s), dtype=np.uint8)
        self.stamp = np.full((s, s), -(10**9), dtype=np.int32)
        self.items_on_screen = []
        self.enabled_items = set(enabled_items) if enabled_items is not None else set(gc.ALL_ITEMS)
        self.sides = 0
        self.field_inset = 0.0
        self._scheduled = []
        self._pending_spawns = []

        self.players = {}
        inset = self.c.start_inset
        for i, name in enumerate(active_players):
            p = PlayerState(name=name, slot=i + 1)
            p.x = self.rng.uniform(inset, s - inset)  # script.js calcRandomStartPos(), mapped to avoid instant death
            p.y = self.rng.uniform(inset, s - inset)
            p.dir = self.rng.uniform(0, 2 * math.pi)  # script.js calcRandomStartDir()
            self.players[name] = p

    # ------------------------------------------------------------------- step

    def step(self, actions: dict[str, int]) -> dict[str, PlayerTickInfo]:
        self.tick += 1
        self._run_scheduled()
        self._run_pending_spawns()
        if self.rng.random() < gc.POWERUP_SPAWN_PROB:
            self._spawn_item()

        info: dict[str, PlayerTickInfo] = {}
        for name, p in self.players.items():
            if not p.alive:
                info[name] = PlayerTickInfo(alive=False, just_died=False, death_cause=p.death_cause)
                continue
            info[name] = self._step_player(p, actions.get(name, STRAIGHT))
        return info

    def _step_player(self, p: PlayerState, action: int) -> PlayerTickInfo:
        c = self.c
        s = c.engine_resolution
        mv = c.move_speed * p.speed

        # 1) positions computed from the OLD heading, before this tick's turn is applied -
        #    matches script.js: nextPos is computed before the turning block runs.
        old_dir = p.dir
        cos_d, sin_d = _mcos(old_dir), _msin(old_dir)
        next_x = p.x + cos_d * mv
        next_y = p.y + sin_d * mv
        prevprev_x = p.x - cos_d * mv
        prevprev_y = p.y - sin_d * mv
        prev_x, prev_y = p.x, p.y

        # 2) turning: applied to p.dir now, so it affects collision sampling this tick but
        #    only affects MOVEMENT starting next tick (also matches script.js's ordering).
        p.turn_l = action == TURN_LEFT
        p.turn_r = action == TURN_RIGHT
        turn_factor = 1.0
        if p.sine_start is not None:
            elapsed = self.tick - p.sine_start
            if elapsed < gc.SINE_TURN_DURATION_TICKS:
                turn_factor = 2.0 ** math.sin((elapsed / gc.SINE_TURN_DURATION_TICKS) * 2 * math.pi)
            else:
                p.sine_start = None
        if p.freeze == 0:
            step_amt = gc.TURN_SPEED * turn_factor / (p.size ** gc.SIZE_TURN_EXPONENT)
            if p.turn_l:
                p.dir += -step_amt if p.reverse == 0 else step_amt
            if p.turn_r:
                p.dir += step_amt if p.reverse == 0 else -step_amt

        # 3) commit movement
        prev_x, prev_y = p.x, p.y
        p.x, p.y = next_x, next_y

        # 4) wrap (sides powerup / global sides) or border-death check
        wrap = self.sides != 0 or p.side != 0
        if wrap:
            if p.x < 0:
                p.x += s; prev_x += s; prevprev_x += s
            if p.x > s:
                p.x -= s; prev_x -= s; prevprev_x -= s
            if p.y < 0:
                p.y += s; prev_y += s; prevprev_y += s
            if p.y > s:
                p.y -= s; prev_y -= s; prevprev_y -= s
        else:
            b = c.border_width + c.hitbox_size + self.field_inset  # NOT scaled by p.size - matches script.js exactly
            if p.x < b or p.x > s - b or p.y < b or p.y > s - b:
                self._kill(p, "border")
                return PlayerTickInfo(alive=False, just_died=True, death_cause=p.death_cause)

        # 5) bridge (random gaps in your own trail)
        if not p.bridge:
            if self.rng.random() < gc.BRIDGE_PROB:
                p.bridge = True
            p.bridge_frame = self.tick
        duration = (gc.BRIDGE_SIZE_TICKS / p.speed) * p.size
        if p.bridge_frame < self.tick - duration:
            p.bridge = False

        # 6) draw trail segment into the owner-id grid
        if not p.bridge and p.invisible == 0:
            width = max(1, round(c.player_size * p.size))
            self._stamp_segment(p.slot, prevprev_x, prevprev_y, p.x, p.y, width)

        # 7) item pickup - front/left/right sample points only (not front2), using the NEW dir
        front_dist = c.hitbox_size * p.size
        fx, fy = p.x + _mcos(p.dir) * front_dist, p.y + _msin(p.dir) * front_dist
        lx, ly = p.x + _mcos(p.dir - _DEG55) * front_dist, p.y + _msin(p.dir - _DEG55) * front_dist
        rx, ry = p.x + _mcos(p.dir + _DEG55) * front_dist, p.y + _msin(p.dir + _DEG55) * front_dist

        collected: list[str] = []
        if self.items_on_screen:
            hit_idx = set()
            for i, item in enumerate(self.items_on_screen):
                r = c.icon_size
                if (
                    math.hypot(fx - item.x, fy - item.y) <= r
                    or math.hypot(lx - item.x, ly - item.y) <= r
                    or math.hypot(rx - item.x, ry - item.y) <= r
                ):
                    hit_idx.add(i)
            if hit_idx:
                for i in sorted(hit_idx, reverse=True):
                    item = self.items_on_screen.pop(i)
                    collected.append(item.kind)
                    p.items_collected += 1
                    self._apply_item(item.kind, p.name)

        # 8) collision check (front, front2, left, right) - skipped while bridging/invisible
        if not p.bridge and p.invisible == 0:
            f2x = p.x + _mcos(p.dir) * gc.FRONT2_SAMPLE_DIST
            f2y = p.y + _msin(p.dir) * gc.FRONT2_SAMPLE_DIST
            for sx, sy in ((fx, fy), (f2x, f2y), (lx, ly), (rx, ry)):
                owner = self.grid_at(sx, sy)
                if owner == 0:
                    continue
                if owner == p.slot and p.ghost != 0:
                    continue  # g_ghost: immune to your own trail specifically
                if owner == p.slot and self._own_trail_is_fresh(p, sx, sy):
                    continue  # just-drawn own tail: canvas antialiasing makes these px non-lethal in the JS game
                cause = "self" if owner == p.slot else self._name_for_slot(owner)
                self._kill(p, cause)
                return PlayerTickInfo(alive=False, just_died=True, death_cause=p.death_cause, items_collected=collected)

        return PlayerTickInfo(alive=True, just_died=False, death_cause=None, items_collected=collected)

    # --------------------------------------------------------------- helpers

    def _stamp_segment(self, slot: int, xa: float, ya: float, xb: float, yb: float, width: int) -> None:
        """Rasterizes one trail segment into grid/stamp - PIXEL-IDENTICAL to the old
        full-frame approach (Image.new(S,S) + ImageDraw.line + full-grid boolean
        assignment), but ~60x cheaper: PIL line rasterization is exactly translation-
        invariant under INTEGER shifts, so drawing the segment into a tiny local
        bounding-box image (integer-offset, fractional endpoint coords preserved)
        and blitting that patch produces the same lethal pixels while touching only
        ~width^2 memory instead of the whole S x S grid per player per tick. Fuzz-
        verified bit-exact against the old path for every gameplay-scale segment
        (ai/tests/test_stamp_equivalence.py: 30k segments up to 12px length/width 32;
        a real segment spans exactly two movement steps, ~1.2px at S=256, up to ~10px
        under stacked g_fast). Only hypothetical arbitrarily-long segments could flip
        a single boundary pixel under translation (PIL float edge math) - the engine
        cannot produce those. The old way rasterized+masked 65k pixels to set ~6,
        making this the engine's dominant per-player cost in multiplayer."""
        s_h, s_w = self.grid.shape
        pad = width + 2  # covers PIL's half-width extent (+ rounding slack) around the endpoints
        x0 = max(0, math.floor(min(xa, xb)) - pad)
        y0 = max(0, math.floor(min(ya, yb)) - pad)
        x1 = min(s_w - 1, math.ceil(max(xa, xb)) + pad)
        y1 = min(s_h - 1, math.ceil(max(ya, yb)) + pad)
        if x1 < x0 or y1 < y0:
            return  # segment lies entirely outside the grid (old path: PIL clipped it to nothing)
        mask_img = Image.new("L", (x1 - x0 + 1, y1 - y0 + 1), 0)
        ImageDraw.Draw(mask_img).line([(xa - x0, ya - y0), (xb - x0, yb - y0)], fill=255, width=width)
        mask = np.asarray(mask_img) > 0
        self.grid[y0 : y1 + 1, x0 : x1 + 1][mask] = slot
        self.stamp[y0 : y1 + 1, x0 : x1 + 1][mask] = self.tick

    def self_grace_ticks(self, p) -> int:
        """How many ticks a freshly-drawn own-trail pixel stays non-lethal for its
        owner. Single source of truth - the collision check, the action mask and
        ai/core/env/sensors.py all use this same rule (sensors additionally evaluate it
        at the FUTURE tick an arc reaches the pixel)."""
        mv = max(1e-6, self.c.move_speed * p.speed)
        clearance = self.c.hitbox_size * p.size + self.c.player_size * p.size / 2 + 2.0
        return math.ceil(clearance / mv)

    def _own_trail_is_fresh(self, p, x: float, y: float, future_ticks: int = 0) -> bool:
        ix, iy = round(x), round(y)
        s = self.c.engine_resolution
        if ix < 0 or iy < 0 or ix >= s or iy >= s:
            return False
        return (self.tick + future_ticks - int(self.stamp[iy, ix])) <= self.self_grace_ticks(p)

    def grid_at(self, x: float, y: float) -> int:
        s = self.c.engine_resolution
        ix, iy = round(x), round(y)
        if ix < 0 or iy < 0 or ix >= s or iy >= s:
            return 0
        return int(self.grid[iy, ix])

    def _name_for_slot(self, slot: int) -> str | None:
        for name, p in self.players.items():
            if p.slot == slot:
                return name
        return None

    def _kill(self, p: PlayerState, cause: str) -> None:
        p.alive = False
        p.death_cause = cause
        p.death_tick = self.tick
        if cause not in (None, "border", "self"):
            killer = self.players.get(cause)
            if killer is not None:
                killer.kills += 1
        for other in self.players.values():
            if other is not p and other.alive:
                other.score += 1

    def _others_all(self, picker: str) -> list[PlayerState]:
        """Every other participant slot, alive or not - matches script.js item loops,
        which iterate the whole `players` object without an alive/ready filter."""
        return [p for name, p in self.players.items() if name != picker]

    def _schedule(self, delay_ticks: int, fn: Callable[[], None]) -> None:
        self._scheduled.append((self.tick + delay_ticks, fn))

    def _run_scheduled(self) -> None:
        due = [fn for expire, fn in self._scheduled if expire <= self.tick]
        self._scheduled = [(expire, fn) for expire, fn in self._scheduled if expire > self.tick]
        for fn in due:
            fn()

    def _run_pending_spawns(self) -> None:
        due = [t for t in self._pending_spawns if t <= self.tick]
        self._pending_spawns = [t for t in self._pending_spawns if t > self.tick]
        for _ in due:
            self._spawn_item()

    def _spawn_item(self) -> None:
        if len(self.items_on_screen) > gc.POWERUP_MAX_ON_SCREEN:
            return
        pool = [i for i in gc.ALL_ITEMS if i in self.enabled_items]
        if not pool:
            return
        s = self.c.engine_resolution
        kind = self.rng.choice(pool)
        x = self.rng.randrange(0, s)
        y = self.rng.randrange(0, s)
        self.items_on_screen.append(ItemOnScreen(kind, x, y))

    # ------------------------------------------------------------ item effects

    def _apply_item(self, name: str, picker: str) -> None:
        if name == gc.RANDOM_ITEM:
            pool = [i for i in self.enabled_items if i != gc.RANDOM_ITEM]
            if not pool:
                return
            name = self.rng.choice(pool)

        p = self.players[picker]

        if name == "g_slow":
            p.speed *= 0.5
            self._schedule(gc.SELF_BUFF_TICKS, lambda p=p: setattr(p, "speed", p.speed * 2))
        elif name == "g_fast":
            p.speed *= 2
            self._schedule(gc.SELF_BUFF_TICKS, lambda p=p: setattr(p, "speed", p.speed * 0.5))
        elif name == "g_thin":
            p.size *= 0.5
            self._schedule(gc.SELF_BUFF_TICKS, lambda p=p: setattr(p, "size", p.size * 2))
        elif name == "g_side":
            p.side += 1
            self._schedule(gc.SELF_BUFF_TICKS, lambda p=p: setattr(p, "side", p.side - 1))
        elif name == "g_invisible":
            p.invisible += 1
            self._schedule(gc.SELF_BUFF_TICKS, lambda p=p: setattr(p, "invisible", p.invisible - 1))
        elif name == "g_sine":
            p.sine_start = self.tick
        elif name == "g_ghost":
            p.ghost += 1
            self._schedule(gc.SELF_BUFF_TICKS, lambda p=p: setattr(p, "ghost", p.ghost - 1))

        elif name == "r_slow":
            for other in self._others_all(picker):
                other.speed *= 0.5
                self._schedule(gc.ENEMY_DEBUFF_TICKS, lambda o=other: setattr(o, "speed", o.speed * 2))
        elif name == "r_fast":
            for other in self._others_all(picker):
                other.speed *= 2
                self._schedule(gc.ENEMY_DEBUFF_TICKS, lambda o=other: setattr(o, "speed", o.speed * 0.5))
        elif name == "r_thick":
            for other in self._others_all(picker):
                other.size *= 2
                self._schedule(gc.ENEMY_DEBUFF_TICKS, lambda o=other: setattr(o, "size", o.size * 0.5))
        elif name == "r_reverse":
            for other in self._others_all(picker):
                other.reverse += 1
                self._schedule(gc.ENEMY_DEBUFF_TICKS, lambda o=other: setattr(o, "reverse", o.reverse - 1))
        elif name == "r_sine":
            for other in self._others_all(picker):
                other.sine_start = self.tick
        elif name == "r_swap":
            # swaps position AND heading with the nearest other ALIVE player (Phase 0 change:
            # each participant continues as if they'd physically become the other, own color kept)
            picker_p = self.players[picker]
            target = None
            best_d = math.inf
            for name_, q in self.players.items():
                if name_ == picker or not q.alive:
                    continue
                d = math.hypot(q.x - picker_p.x, q.y - picker_p.y)
                if d < best_d:
                    best_d = d
                    target = q
            if target is not None:
                tx, ty, td = target.x, target.y, target.dir
                target.x, target.y, target.dir = picker_p.x, picker_p.y, picker_p.dir
                picker_p.x, picker_p.y, picker_p.dir = tx, ty, td
        elif name == "r_freeze":
            for other in self._others_all(picker):
                other.freeze += 1
                self._schedule(gc.FREEZE_TICKS, lambda o=other: setattr(o, "freeze", o.freeze - 1))

        elif name == "b_clear":
            self.grid.fill(0)
        elif name == "b_more":
            for d in gc.B_MORE_DELAYS_TICKS:
                self._pending_spawns.append(self.tick + d)
        elif name == "b_sides":
            self.sides += 1
            self._schedule(gc.SELF_BUFF_TICKS, lambda: setattr(self, "sides", self.sides - 1))
        elif name == "b_shrink":
            self.field_inset = min(self.field_inset + self.c.shrink_step, self.c.shrink_cap)
            self._schedule(gc.SELF_BUFF_TICKS, lambda: setattr(self, "field_inset", self.field_inset - self.c.shrink_step))
        elif name == "b_grow":
            self.field_inset = max(self.field_inset - self.c.grow_step, self.c.grow_cap)
            self._schedule(gc.SELF_BUFF_TICKS, lambda: setattr(self, "field_inset", self.field_inset + self.c.grow_step))

    # ------------------------------------------------------------- game-level

    def alive_count(self) -> int:
        return sum(1 for p in self.players.values() if p.alive)

    def is_episode_over(self) -> bool:
        return self.alive_count() <= 1

    def point_goal(self) -> int:
        return (len(self.players) - 1) * gc.POINT_GOAL_PER_OPPONENT

    def match_winner(self) -> str | None:
        """None if no one has clinched it yet (script.js checkGameState's 2-point-margin rule)."""
        ranked = sorted(self.players.values(), key=lambda p: p.score, reverse=True)
        if len(ranked) < 2:
            return ranked[0].name if ranked else None
        if ranked[0].score >= self.point_goal() and ranked[0].score - ranked[1].score > gc.WIN_MARGIN - 1:
            return ranked[0].name
        return None
