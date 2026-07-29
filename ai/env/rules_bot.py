"""A rule-based heuristic opponent, written fresh for this training system (the
project's old bot.js has been deleted). It exists for three things the brief asks
for: a scripted opponent for curriculum/self-play/league mixing, a fixed baseline
to evaluate the RL agent against ("Performance gegen regelbasierte Bots"), and proof
that a short-horizon forward simulation is enough to make Discrete(3) reasonable to
mask (see curve_env.py's action masking, which reuses the same idea at a much
shorter horizon).

Approach: for each of the 3 actions, roll forward a fixed number of ticks against
the engine's current owner-id grid (read-only - no mutation of real game state),
score how long it survives and, if it survives the whole window, how well it's
aimed at a nearby item / a lead point in front of the nearest opponent. Difficulty
presets trade off lookahead depth, reaction latency, and how aggressively items/
opponents are pursued.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ai.config import game_constants as gc
from ai.env.engine import CurveEngine, PlayerState, STRAIGHT, TURN_LEFT, TURN_RIGHT, _mcos, _msin


@dataclass(frozen=True)
class BotDifficulty:
    lookahead: int      # ticks of forward simulation
    react: int          # ticks a chosen action is held before re-deciding
    turn_commit: int     # ticks the simulated turn is held before assuming straight-line
    straight_bias: float
    noise: float          # chance of a random, non-optimal action (mistakes)
    powerup_weight: float
    aggression: float     # how hard to cut off the nearest opponent
    wall_avoid: float     # how strongly to steer away from hugging the border


BOT_DIFFICULTIES: dict[str, BotDifficulty] = {
    "easy": BotDifficulty(lookahead=35, react=4, turn_commit=30, straight_bias=8, noise=0.10, powerup_weight=0.3, aggression=0.3, wall_avoid=0.3),
    "medium": BotDifficulty(lookahead=60, react=2, turn_commit=30, straight_bias=5, noise=0.03, powerup_weight=0.7, aggression=0.8, wall_avoid=0.5),
    "hard": BotDifficulty(lookahead=90, react=1, turn_commit=30, straight_bias=3, noise=0.00, powerup_weight=1.0, aggression=1.2, wall_avoid=0.6),
    "hunter": BotDifficulty(lookahead=110, react=1, turn_commit=30, straight_bias=2, noise=0.00, powerup_weight=1.3, aggression=1.8, wall_avoid=0.5),
}

_ACTIONS = (TURN_LEFT, STRAIGHT, TURN_RIGHT)  # -> steering deltas -1, 0, +1
_DELTA = {TURN_LEFT: -1, STRAIGHT: 0, TURN_RIGHT: 1}


class RuleBasedBot:
    """Stateful per-seat controller: call `decide(engine, name)` once per tick."""

    def __init__(self, difficulty: str = "medium", rng: random.Random | None = None):
        self.cfg = BOT_DIFFICULTIES[difficulty]
        self.rng = rng or random.Random()
        self._hold = 0
        self._last_action = STRAIGHT
        self._streak = 0

    def decide(self, engine: CurveEngine, name: str) -> int:
        if self._hold > 0:
            self._hold -= 1
        else:
            action = self._pick_action(engine, name)
            self._hold = self.cfg.react
            if action == self._last_action and action != STRAIGHT:
                self._streak += 1
            else:
                self._streak = 0
            self._last_action = action
        return self._last_action

    # ------------------------------------------------------------------ core

    def _pick_action(self, engine: CurveEngine, name: str) -> int:
        cfg = self.cfg
        p = engine.players[name]
        c = engine.c
        mv = c.move_speed * p.speed
        half = c.hitbox_size * p.size
        step = gc.TURN_SPEED / (p.size ** gc.SIZE_TURN_EXPONENT)
        skip = max(2, math.ceil(half / mv) + 1)
        margin = c.engine_resolution * 0.14

        best_item = self._nearest_item(engine, p)
        opponent = self._nearest_opponent(engine, name, p)

        best_action, best_score = STRAIGHT, -math.inf
        for action in _ACTIONS:
            a = _DELTA[action]
            survived, ex, ey, end_dir = self._simulate(engine, p, a, mv, step, skip)
            if survived <= cfg.lookahead:
                score = survived
            else:
                score = 1e6
                if action == STRAIGHT:
                    score += cfg.straight_bias
                b = c.border_width + half
                s = c.engine_resolution
                wall_dist = min(ex - b, s - b - ex, ey - b, s - b - ey)
                if wall_dist < margin:
                    score -= cfg.wall_avoid * (1 - wall_dist / margin) * 8
                if a != 0 and a == _DELTA.get(self._last_action, 0) and self._streak > 18:
                    score -= (self._streak - 18) * 0.6
                if best_item is not None and best_item[2] < c.engine_resolution * 0.6:
                    score += cfg.powerup_weight * best_item[3] * self._aim(p, end_dir, step, best_item[0], best_item[1]) * 8
                if opponent is not None and opponent[2] < c.engine_resolution * 0.65:
                    lead = min(opponent[2] * 0.85, mv * cfg.lookahead * 0.8)
                    lx = opponent[0] + _mcos(opponent[3]) * lead
                    ly = opponent[1] + _msin(opponent[3]) * lead
                    score += cfg.aggression * self._aim(p, end_dir, step, lx, ly) * 12
            score += self.rng.random() * 1e-3
            if score > best_score:
                best_score, best_action = score, action

        if self.rng.random() < cfg.noise:
            best_action = self.rng.choice(_ACTIONS)
        return best_action

    def _simulate(self, engine: CurveEngine, p: PlayerState, a: int, mv: float, step: float, skip: int):
        cfg = self.cfg
        s = engine.c.engine_resolution
        wrap = engine.sides != 0 or p.side != 0
        b = engine.c.border_width + engine.c.hitbox_size + engine.field_inset
        sx, sy, sd = p.x, p.y, p.dir
        ex, ey = sx, sy
        survived = cfg.lookahead + 1
        for f in range(1, cfg.lookahead + 1):
            if f <= cfg.turn_commit:
                sd += a * step
            sx += _mcos(sd) * mv
            sy += _msin(sd) * mv
            if wrap:
                if sx < 0:
                    sx += s
                elif sx > s:
                    sx -= s
                if sy < 0:
                    sy += s
                elif sy > s:
                    sy -= s
            elif sx < b or sx > s - b or sy < b or sy > s - b:
                survived = f
                break
            if f > skip and engine.grid_at(sx, sy) != 0:
                survived = f
                break
            ex, ey = sx, sy
        return survived, ex, ey, sd

    @staticmethod
    def _aim(p: PlayerState, current_dir: float, step: float, tx: float, ty: float) -> float:
        desired = math.atan2(ty - p.y, tx - p.x)
        diff = abs(math.atan2(math.sin(desired - current_dir), math.cos(desired - current_dir)))
        return 1 - diff / math.pi

    @staticmethod
    def _nearest_item(engine: CurveEngine, p: PlayerState):
        best = None
        best_d = math.inf
        for item in engine.items_on_screen:
            d = math.hypot(p.x - item.x, p.y - item.y)
            if d < best_d:
                value = 1.5 if item.kind in gc.ENEMY_ITEMS else 1.0
                best_d, best = d, (item.x, item.y, d, value)
        return best

    @staticmethod
    def _nearest_opponent(engine: CurveEngine, name: str, p: PlayerState):
        best = None
        best_d = math.inf
        for other_name, q in engine.players.items():
            if other_name == name or not q.alive:
                continue
            d = math.hypot(p.x - q.x, p.y - q.y)
            if d < best_d:
                best_d, best = d, (q.x, q.y, d, q.dir)
        return best
