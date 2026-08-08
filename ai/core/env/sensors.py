"""Fast grid sensors, shared by observation.py (policy inputs) and curve_env.py
(lookahead shaping + doomed-state termination).

Why this module exists: the CNN's view of the field is a 96x96 downscale of the
256x256 engine grid. At that scale the 2px trail survives only as a ~1px line at
30-75% brightness and the head dot is sub-pixel - the *global* layout is visible,
but "how far is the nearest lethal pixel from MY head, in MY heading frame" is
almost impossible to read out, especially through an average-pooled CNN. These
sensors hand the policy exactly that missing near-field geometry, computed
directly on the authoritative owner-id grid:

- `ray_distances()`: K lidar-style distances (heading-relative, so the feature is
  rotation-equivariant) to the nearest lethal pixel - border analytic, trails via
  one vectorized grid gather - normalized to [0, 1].
- `arc_survival_ticks()`: for the three *pure* strategies (hold LEFT, hold
  STRAIGHT, hold RIGHT), how many ticks the player survives before hitting
  border/trail, capped at `horizon`. This is the literal "wenn ich bis zu dem
  Zeitpunkt nicht lenke, sterbe ich" signal: ttc[STRAIGHT] < horizon says *when*
  going straight becomes fatal, and max(ttc) collapsing toward 0 says the state
  is running out of escapes altogether (curve_env uses that for potential
  shaping and for early "doomed" termination).

All probing skips freshly-drawn own-tail pixels with the engine's own grace rule
(engine.self_grace_ticks), evaluated at the FUTURE tick an arc reaches the pixel:
a pixel that is fresh now may be lethal by the time we get there.

Arc simulation mirrors the engine's tick ordering (movement uses the heading from
BEFORE this tick's turn) but uses exact cos/sin instead of the engine's 2-decimal
rounded _mcos/_msin: over the <=64-tick horizons used here the drift is < 0.5px,
and sensing is approximate by nature - only real engine steps decide death.
Sampling is center-point only (no +-55 deg side probes), i.e. deliberately
optimistic: an arc is only counted as dying when even its optimistic center path
dies, which is exactly the conservative direction the doomed-terminator needs.

Everything is numpy-vectorized (one fancy-index gather per query) and memoized
per (engine, tick, player), so curve_env's shaping/doom check and the observation
builder share one computation per tick even though both ask for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ai.core.config import game_constants as gc
from ai.core.env.engine import CurveEngine, PlayerState

# local arc shapes depend only on (per-tick turn angle, horizon) - cache them.
# positions are for a unit move speed, heading +x at the origin; scale by mv and
# rotate by p.dir at query time.
_ARC_CACHE: dict[tuple[float, int], np.ndarray] = {}
_RAY_UNIT_CACHE: dict[int, np.ndarray] = {}


def _local_arcs(step_amt: float, horizon: int) -> np.ndarray:
    """(3, horizon, 2) float32: unit-speed center paths for actions (LEFT, STRAIGHT,
    RIGHT) in the heading-local frame. Engine ordering: tick t moves along the
    heading as of BEFORE tick t's turn, so position t uses heading (t-1)*delta."""
    key = (round(step_amt, 6), horizon)
    arcs = _ARC_CACHE.get(key)
    if arcs is None:
        t = np.arange(horizon, dtype=np.float64)  # heading index used for move t+1
        arcs_list = []
        for delta in (-1.0, 0.0, 1.0):
            ang = delta * step_amt * t
            steps = np.stack([np.cos(ang), np.sin(ang)], axis=1)  # (H, 2) unit moves
            arcs_list.append(np.cumsum(steps, axis=0))
        arcs = np.stack(arcs_list, axis=0).astype(np.float32)  # (3, H, 2)
        if len(_ARC_CACHE) > 64:
            _ARC_CACHE.clear()
        _ARC_CACHE[key] = arcs
    return arcs


def _ray_units(n_rays: int) -> np.ndarray:
    """(K, 2) float32 unit vectors at heading-relative angles 2*pi*k/K (k=0 = straight
    ahead, angles increase clockwise in screen coords, matching the engine's y-down
    frame and TURN_RIGHT = +dir)."""
    units = _RAY_UNIT_CACHE.get(n_rays)
    if units is None:
        ang = 2.0 * math.pi * np.arange(n_rays, dtype=np.float64) / n_rays
        units = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)
        _RAY_UNIT_CACHE[n_rays] = units
    return units


def _rotate(points: np.ndarray, direction: float) -> np.ndarray:
    """Rotate (..., 2) heading-local points into world frame for heading `direction`."""
    c, s = math.cos(direction), math.sin(direction)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points @ rot.T


def _lethal_mask(
    engine: CurveEngine,
    p: PlayerState,
    pts: np.ndarray,
    future_ticks: np.ndarray | int,
    wrap: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """For flat (N, 2) world points, returns (lethal, oob) boolean masks.
    `lethal`: point contains a trail pixel that would kill `p` when reached
    `future_ticks` ticks from now (own fresh tail + ghost exemptions applied).
    `oob`: point lies outside the [0, s) grid (only possible when not wrapping;
    callers treat that as border territory)."""
    s = engine.c.engine_resolution
    xy = pts
    if wrap:
        xy = np.mod(xy, s)
    ix = np.rint(xy[:, 0]).astype(np.int64)
    iy = np.rint(xy[:, 1]).astype(np.int64)
    oob = (ix < 0) | (iy < 0) | (ix >= s) | (iy >= s)
    ix_c = np.clip(ix, 0, s - 1)
    iy_c = np.clip(iy, 0, s - 1)
    owner = engine.grid[iy_c, ix_c]
    lethal = (owner != 0) & ~oob
    own = owner == p.slot
    if p.ghost != 0:
        lethal &= ~own
    else:
        stamp = engine.stamp[iy_c, ix_c]
        age_when_reached = (engine.tick + future_ticks) - stamp
        fresh = own & (age_when_reached <= engine.self_grace_ticks(p))
        lethal &= ~fresh
    return lethal, oob


def ray_distances(engine: CurveEngine, name: str, n_rays: int, range_px: float, step_px: float = 1.0) -> np.ndarray:
    """(n_rays,) float32 in [0, 1]: for each heading-relative direction, the distance
    to the nearest lethal thing (border or any non-exempt trail pixel), divided by
    `range_px` and capped at 1.0 ("clear for at least range_px")."""
    p = engine.players[name]
    c = engine.c
    s = c.engine_resolution
    wrap = engine.sides != 0 or p.side != 0

    units = _rotate(_ray_units(n_rays), p.dir)  # (K, 2)
    n_steps = max(1, int(range_px / step_px))
    dists = (np.arange(1, n_steps + 1, dtype=np.float32) * step_px)  # (D,)
    pts = np.array([p.x, p.y], dtype=np.float32) + units[:, None, :] * dists[None, :, None]  # (K, D, 2)

    lethal, oob = _lethal_mask(engine, p, pts.reshape(-1, 2), 0, wrap)
    lethal = lethal.reshape(n_rays, n_steps)
    oob = oob.reshape(n_rays, n_steps)

    out = np.full(n_rays, range_px, dtype=np.float32)
    if not wrap:
        # analytic border: lethal boundary sits at inset b (same b as the engine's
        # death check), which oob alone (grid edge) would place too far out
        b = c.border_width + c.hitbox_size + engine.field_inset
        for k in range(n_rays):
            ux, uy = float(units[k, 0]), float(units[k, 1])
            d_wall = range_px
            if ux < -1e-9:
                d_wall = min(d_wall, (b - p.x) / ux)
            elif ux > 1e-9:
                d_wall = min(d_wall, ((s - b) - p.x) / ux)
            if uy < -1e-9:
                d_wall = min(d_wall, (b - p.y) / uy)
            elif uy > 1e-9:
                d_wall = min(d_wall, ((s - b) - p.y) / uy)
            out[k] = max(0.0, d_wall)
        lethal |= oob  # anything past the grid edge is definitely wall
    hit_any = lethal.any(axis=1)
    first_hit = np.argmax(lethal, axis=1)
    out[hit_any] = np.minimum(out[hit_any], dists[first_hit[hit_any]])
    return (out / range_px).astype(np.float32)


def simulate_deltas(engine: CurveEngine, name: str, deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Beliebige Aktionsfolgen vorwaerts simulieren - die Verallgemeinerung von
    `arc_survival_ticks`, das nur die drei reinen Strategien kennt.

    `deltas`: (N, H) int/float in {-1, 0, +1}, N Kandidatensequenzen à H Ticks.
    Rueckgabe `(positions, survived)`:
      - positions (N, H, 2) float32: Weltkoordinaten der Kopfposition je Tick
      - survived  (N,) int32: Ticks bis zum ersten toedlichen Punkt, gedeckelt
        bei H ("ueberlebt die ganze Sequenz")

    Gleiche Modellierung wie `arc_survival_ticks` und damit dieselben Grenzen:
    exakte cos/sin statt der 2-Nachkommastellen-Rundung der Engine, nur
    Mittelpunkt-Sampling (keine +-55-Grad-Seitenproben) und ein eingefrorenes
    Gitter - Gegner zeichnen waehrend des Horizonts weiter, was hier nicht
    auftaucht. Das ist fuer einen Planner die richtige Richtung: er unterschaetzt
    nie die eigene Beweglichkeit, aber er darf sich auf ein Ergebnis "ueberlebt"
    nicht blind verlassen, sondern muss jeden Tick neu planen (Receding Horizon).

    Bewegungsreihenfolge wie in der Engine: Zug t bewegt sich entlang der
    Richtung VOR der Drehung von Zug t, also nutzt Position t die Summe der
    Deltas 0..t-1.
    """
    p = engine.players[name]
    c = engine.c
    s = c.engine_resolution
    deltas = np.asarray(deltas, dtype=np.float32)
    if deltas.ndim != 2:
        raise ValueError(f"deltas muss (N, H) sein, war {deltas.shape}")
    n_seq, horizon = deltas.shape
    if horizon == 0:
        return np.zeros((n_seq, 0, 2), dtype=np.float32), np.zeros(n_seq, dtype=np.int32)

    mv = c.move_speed * p.speed
    step_amt = gc.TURN_SPEED / (p.size ** gc.SIZE_TURN_EXPONENT)
    if p.reverse != 0:
        # r_reverse dreht um, welche AKTION welche Richtung erzeugt. Der Planner
        # arbeitet in Aktionsraum, die Geometrie hier in Weltraum - also einmal
        # hier spiegeln statt an jeder Aufrufstelle.
        deltas = -deltas

    # kumulierte Richtung VOR dem jeweiligen Zug: exklusives Praefix der Deltas
    turned = np.cumsum(deltas, axis=1) - deltas  # (N, H)
    ang = p.dir + turned * step_amt
    steps = np.stack([np.cos(ang), np.sin(ang)], axis=-1) * mv  # (N, H, 2)
    world = np.cumsum(steps, axis=1) + np.array([p.x, p.y], dtype=np.float32)

    wrap = engine.sides != 0 or p.side != 0
    future = np.tile(np.arange(1, horizon + 1, dtype=np.int64), n_seq)
    lethal, oob = _lethal_mask(engine, p, world.reshape(-1, 2), future, wrap)
    lethal = lethal.reshape(n_seq, horizon)
    oob = oob.reshape(n_seq, horizon)

    if not wrap:
        b = c.border_width + c.hitbox_size + engine.field_inset
        xs, ys = world[:, :, 0], world[:, :, 1]
        lethal |= (xs < b) | (xs > s - b) | (ys < b) | (ys > s - b) | oob

    survived = np.full(n_seq, horizon, dtype=np.int32)
    dead_any = lethal.any(axis=1)
    survived[dead_any] = np.argmax(lethal, axis=1)[dead_any].astype(np.int32)
    return world.astype(np.float32), survived


def clearance_at(engine: CurveEngine, name: str, points: np.ndarray, n_rays: int, range_px: float, step_px: float = 2.0) -> np.ndarray:
    """(N,) float32 in [0, 1]: Freiraum um beliebige Weltpunkte - die kleinste
    Distanz zu etwas Toedlichem, gemessen auf `n_rays` gleichverteilten
    Richtungen und auf `range_px` normiert.

    Der Planner braucht das als Tiebreaker: mehrere Sequenzen ueberleben den
    ganzen Horizont, aber die eine endet mitten im freien Feld und die andere in
    einer Sackgasse, deren Ende erst zwei Ticks hinter dem Horizont liegt.
    Bewusst richtungsunabhaengig (volle 360 Grad, nicht kopfrelativ): bewertet
    wird die Guete einer POSITION, nicht die einer Blickrichtung.
    """
    p = engine.players[name]
    c = engine.c
    s = c.engine_resolution
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    n_pts = pts.shape[0]
    if n_pts == 0:
        return np.zeros(0, dtype=np.float32)

    units = _ray_units(n_rays)  # (K, 2), Weltframe genuegt hier
    n_steps = max(1, int(range_px / step_px))
    dists = np.arange(1, n_steps + 1, dtype=np.float32) * step_px
    probe = pts[:, None, None, :] + units[None, :, None, :] * dists[None, None, :, None]  # (N, K, D, 2)

    wrap = engine.sides != 0 or p.side != 0
    lethal, oob = _lethal_mask(engine, p, probe.reshape(-1, 2), 0, wrap)
    lethal = lethal.reshape(n_pts, n_rays, n_steps)
    oob = oob.reshape(n_pts, n_rays, n_steps)
    if not wrap:
        lethal |= oob

    hit = lethal.any(axis=2)
    first = np.argmax(lethal, axis=2)
    d = np.where(hit, dists[first], range_px)  # (N, K)

    if not wrap:
        b = c.border_width + c.hitbox_size + engine.field_inset
        wall = np.minimum.reduce([pts[:, 0] - b, (s - b) - pts[:, 0], pts[:, 1] - b, (s - b) - pts[:, 1]])
        d = np.minimum(d.min(axis=1), np.clip(wall, 0.0, range_px))
    else:
        d = d.min(axis=1)
    return np.clip(d / range_px, 0.0, 1.0).astype(np.float32)


def arc_survival_ticks(engine: CurveEngine, name: str, horizon: int) -> np.ndarray:
    """(3,) int32: ticks survived when holding (LEFT, STRAIGHT, RIGHT) for up to
    `horizon` ticks against the CURRENT grid and border, center-point sampled.
    horizon means "survives at least the whole horizon". Memoized per
    (engine, tick, player, horizon) - the observation builder, the potential
    shaping and the doomed check all share one computation.
    Freeze/sine states fall back to `horizon` (their turn dynamics diverge from
    the plain arc model; claiming lookahead there would be wrong more often
    than useful)."""
    cache: dict = getattr(engine, "_sensor_cache", None)
    if cache is None or cache.get("tick") != engine.tick:
        cache = {"tick": engine.tick}
        engine._sensor_cache = cache
    key = ("arc", name, horizon)
    if key in cache:
        return cache[key]

    p = engine.players[name]
    c = engine.c
    s = c.engine_resolution
    out = np.full(3, horizon, dtype=np.int32)
    if p.freeze != 0 or p.sine_start is not None:
        cache[key] = out
        return out

    mv = c.move_speed * p.speed
    step_amt = gc.TURN_SPEED / (p.size ** gc.SIZE_TURN_EXPONENT)
    arcs = _local_arcs(step_amt, horizon)  # (3, H, 2) unit-speed local
    world = _rotate(arcs * mv, p.dir) + np.array([p.x, p.y], dtype=np.float32)  # (3, H, 2)

    wrap = engine.sides != 0 or p.side != 0
    future = np.tile(np.arange(1, horizon + 1, dtype=np.int64), 3)  # per flattened point
    lethal, oob = _lethal_mask(engine, p, world.reshape(-1, 2), future, wrap)
    lethal = lethal.reshape(3, horizon)
    oob = oob.reshape(3, horizon)

    if not wrap:
        b = c.border_width + c.hitbox_size + engine.field_inset
        xs = world[:, :, 0]
        ys = world[:, :, 1]
        border_dead = (xs < b) | (xs > s - b) | (ys < b) | (ys > s - b)
        lethal |= border_dead | oob

    dead_any = lethal.any(axis=1)
    first_dead = np.argmax(lethal, axis=1)  # index of first lethal tick (0-based -> tick t+1... see below)
    # point t (0-based) is reached at future tick t+1; dying "at tick t+1" means we
    # survived t ticks:
    out[dead_any] = first_dead[dead_any].astype(np.int32)
    if p.reverse != 0:
        # r_reverse flips which ACTION turns which way; the reachable paths are the
        # same mirrored set, so the LEFT/RIGHT survival times simply swap.
        out = out[::-1].copy()
    cache[key] = out
    return out
